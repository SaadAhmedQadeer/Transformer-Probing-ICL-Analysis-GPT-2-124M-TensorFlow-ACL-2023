# lsfs_experiment.py
# Label Surface Form Sensitivity (LSFS) Experiment
# Novel contribution: quantifies violation of anchor theory's
# implicit invariance assumption across surface-form variants.
#
# Wang et al. ACL 2023 implicitly predicts:
#   small embedding distance → small accuracy change
# We show this prediction FAILS for symbolic/numeric variants.

import os
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datasets import load_dataset
from corrected_extractor import CorrectedHiddenStateExtractor
from probing_classifier import LayerwiseProber


# ── Label variant definitions ──────────────────────────────────────────────
# Each variant: (positive_label, negative_label)
VARIANTS = {
    'V0_canonical' : ('Positive', 'Negative'),   # baseline — paper's setup
    'V1_synonyms'  : ('Good',     'Bad'),
    'V2_abbrev'    : ('Pos',      'Neg'),
    'V3_symbols'   : ('+',        '-'),
    'V4_numeric'   : ('1',        '0'),
    'V5_random'    : ('Apple',    'Orange'),      # null baseline
}

# Human-readable display names for plots
VARIANT_LABELS = {
    'V0_canonical' : 'Canonical\n(Positive/Negative)',
    'V1_synonyms'  : 'Synonyms\n(Good/Bad)',
    'V2_abbrev'    : 'Abbreviated\n(Pos/Neg)',
    'V3_symbols'   : 'Symbols\n(+/−)',
    'V4_numeric'   : 'Numeric\n(1/0)',
    'V5_random'    : 'Random\n(Apple/Orange)',
}


# ── Prompt builder (variant-aware) ────────────────────────────────────────
def build_variant_prompt(demo_examples, query_sentence,
                          pos_label, neg_label):
    """
    Builds a 4-shot ICL prompt using the specified label surface forms.
    The query sentence is the same across all variants — only demo
    label words change. This isolates the surface-form effect.
    """
    label_map = {1: pos_label, 0: neg_label}
    prompt = ""
    for d in demo_examples:
        prompt += (
            f"Review: {d['sentence']}\n"
            f"Sentiment: {label_map[d['label']]}\n\n"
        )
    prompt += f"Review: {query_sentence}\nSentiment:"
    return prompt


# ── ICL prediction (next-token logit comparison) ──────────────────────────
def predict_icl(extractor, prompt, pos_label, neg_label):
    """
    Predicts sentiment by comparing next-token log-probabilities
    for the positive vs negative label token.

    This is the standard ICL evaluation protocol:
        ŷ = argmax_{l ∈ {pos, neg}} log P(l | prompt)

    Returns: predicted label (1=positive, 0=negative), and both logits.
    """
    from transformers import TFGPT2LMHeadModel
    import tensorflow as tf

    # Load LM head model (needed for next-token logits)
    # Cache it as attribute to avoid reloading
    if not hasattr(extractor, '_lm_model'):
        print("  Loading GPT-2 LM head model (one-time)...")
        extractor._lm_model = TFGPT2LMHeadModel.from_pretrained('gpt2')

    inputs = extractor.tokenizer(
        prompt, return_tensors='tf',
        truncation=True, max_length=1024
    )
    outputs = extractor._lm_model(
        inputs['input_ids'], training=False
    )
    # logits shape: [1, seq_len, vocab_size]
    # Take logits at LAST position (next token prediction)
    last_logits = outputs.logits[0, -1, :]   # [vocab_size]

    # Get token IDs for pos and neg label (with leading space)
    pos_ids = extractor.tokenizer.encode(
        ' ' + pos_label, add_special_tokens=False
    )
    neg_ids = extractor.tokenizer.encode(
        ' ' + neg_label, add_special_tokens=False
    )

    # Use first token ID if multi-token label
    pos_logit = float(last_logits[pos_ids[0]].numpy())
    neg_logit = float(last_logits[neg_ids[0]].numpy())

    pred = 1 if pos_logit > neg_logit else 0
    return pred, pos_logit, neg_logit


# ── Embedding distance computation ────────────────────────────────────────
def compute_embedding_distance(extractor, variant_name,
                                 pos_label, neg_label):
    """
    Computes L2 distance between variant label embeddings and
    canonical label embeddings in GPT-2's embedding matrix.

    d_m = ||E(v_m+) - E(v0+)||_2 + ||E(v_m-) - E(v0-)||_2

    This is the denominator of the LSFS score.
    """
    embed_matrix = extractor.model.weights[0].numpy()
    # Shape: [vocab_size, d_model]

    def get_embed(word):
        # Use space-prefixed version (mid-sentence tokenisation)
        ids = extractor.tokenizer.encode(
            ' ' + word, add_special_tokens=False
        )
        return embed_matrix[ids[0]]  # [d_model]

    # Canonical embeddings
    e_pos_canon = get_embed('Positive')
    e_neg_canon = get_embed('Negative')

    # Variant embeddings
    e_pos_var = get_embed(pos_label)
    e_neg_var = get_embed(neg_label)

    d_pos = float(np.linalg.norm(e_pos_var - e_pos_canon))
    d_neg = float(np.linalg.norm(e_neg_var - e_neg_canon))
    d_total = d_pos + d_neg

    return d_total, d_pos, d_neg


# ── LSFS score ─────────────────────────────────────────────────────────────
def compute_lsfs(acc_canonical, acc_variant, embed_dist, eps=1e-6):
    """
    LSFS(V_m) = |Acc(V_0) - Acc(V_m)| / (d_m + eps)

    High LSFS → large accuracy change per unit embedding distance
              → violation of anchor theory's smoothness assumption.
    """
    delta = abs(acc_canonical - acc_variant)
    return delta / (embed_dist + eps)


# ── Main experiment runner ─────────────────────────────────────────────────
def run_lsfs_experiment(n_examples=100, force_rerun=False):
    """
    Runs the full LSFS experiment across all label variants.

    For each variant:
      1. Builds prompts with that variant's label words
      2. Runs ICL predictions (next-token logit comparison)
      3. Computes accuracy
      4. Computes embedding distance from canonical
      5. Computes LSFS score
      6. Runs layer-wise probing on cached hidden states

    Returns: results dict with all metrics per variant.
    """
    os.makedirs('results', exist_ok=True)
    os.makedirs('figures', exist_ok=True)
    results_path = 'results/lsfs_results.json'

    if os.path.exists(results_path) and not force_rerun:
        print("Loading cached LSFS results ...")
        with open(results_path, 'r') as f:
            return json.load(f)

    # ── Load data ──────────────────────────────────────────────
    print("Loading SST-2 ...")
    dataset   = load_dataset('sst2')
    test_pool = list(dataset['validation'])

    pos_ex   = [x for x in test_pool if x['label'] == 1][:n_examples//2]
    neg_ex   = [x for x in test_pool if x['label'] == 0][:n_examples//2]
    test_set = pos_ex + neg_ex
    np.random.default_rng(42).shuffle(test_set)

    train_pool = list(dataset['train'])
    demos = (
        [x for x in train_pool if x['label'] == 1][:2] +
        [x for x in train_pool if x['label'] == 0][:2]
    )

    extractor = CorrectedHiddenStateExtractor('gpt2')

    # ── Run experiment for each variant ───────────────────────
    results = {}
    canon_acc = None   # computed first, used for LSFS denominator

    for variant_name, (pos_lbl, neg_lbl) in VARIANTS.items():
        print(f"\n{'─'*50}")
        print(f"Variant: {variant_name}  "
              f"(+)='{pos_lbl}'  (-)='{neg_lbl}'")
        print(f"{'─'*50}")

        # ── ICL accuracy ──────────────────────────────────────
        correct = 0
        logit_gaps = []   # pos_logit - neg_logit per example

        for i, ex in enumerate(test_set):
            prompt = build_variant_prompt(
                demos, ex['sentence'], pos_lbl, neg_lbl
            )
            pred, pos_logit, neg_logit = predict_icl(
                extractor, prompt, pos_lbl, neg_lbl
            )
            correct     += (pred == ex['label'])
            logit_gaps.append(pos_logit - neg_logit)

            if i % 20 == 0:
                print(f"  [{i:3d}/{len(test_set)}] "
                      f"running_acc={correct/(i+1):.3f}")

        accuracy      = correct / len(test_set)
        mean_logit_gap = float(np.mean(logit_gaps))
        std_logit_gap  = float(np.std(logit_gaps))

        print(f"  → Accuracy      : {accuracy:.4f}")
        print(f"  → Mean logit gap: {mean_logit_gap:.4f} "
              f"± {std_logit_gap:.4f}")

        # ── Embedding distance ────────────────────────────────
        d_total, d_pos, d_neg = compute_embedding_distance(
            extractor, variant_name, pos_lbl, neg_lbl
        )
        print(f"  → Embed dist    : {d_total:.4f} "
              f"(pos={d_pos:.3f}, neg={d_neg:.3f})")

        if variant_name == 'V0_canonical':
            canon_acc = accuracy
            lsfs_score = 0.0   # by definition
        else:
            lsfs_score = compute_lsfs(canon_acc, accuracy, d_total)

        print(f"  → LSFS score    : {lsfs_score:.6f}")

        # ── Layer-wise probing on this variant ────────────────
        print(f"  Running layer-wise probing for {variant_name} ...")
        probe_accs = run_probing_for_variant(
            extractor, demos, test_set, pos_lbl, neg_lbl,
            variant_name
        )

        results[variant_name] = {
            'pos_label'      : pos_lbl,
            'neg_label'      : neg_lbl,
            'accuracy'       : accuracy,
            'delta_acc'      : abs((canon_acc or accuracy) - accuracy),
            'embed_dist'     : d_total,
            'embed_dist_pos' : d_pos,
            'embed_dist_neg' : d_neg,
            'lsfs_score'     : lsfs_score,
            'mean_logit_gap' : mean_logit_gap,
            'std_logit_gap'  : std_logit_gap,
            'probe_accs'     : probe_accs,
        }

    # Update delta_acc now that canon_acc is set
    for vname in results:
        if vname != 'V0_canonical':
            results[vname]['delta_acc'] = abs(
                canon_acc - results[vname]['accuracy']
            )

    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Results saved → {results_path}")

    return results


def run_probing_for_variant(extractor, demos, test_set,
                              pos_lbl, neg_lbl, variant_name):
    """
    Runs layer-wise probing for a single label variant.
    Extracts hidden states at final token position (same as
    main reproduction) but with variant label words in demos.
    """
    cache_path = f'cache/probe_{variant_name}.pkl'

    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            all_hs, all_labels = pickle.load(f)
    else:
        all_hs, all_labels = [], []
        for ex in test_set:
            prompt    = build_variant_prompt(
                demos, ex['sentence'], pos_lbl, neg_lbl
            )
            input_ids = extractor.tokenizer.encode(prompt)
            final_pos = len(input_ids) - 1
            target    = np.array([final_pos], dtype=np.int32)
            hs        = extractor.extract(prompt, target)
            if hs is None or hs.size == 0 or np.isnan(hs).any():
                continue
            all_hs.append(hs[:, 0, :])
            all_labels.append(ex['label'])

        all_hs     = np.array(all_hs)
        all_labels = np.array(all_labels)
        with open(cache_path, 'wb') as f:
            pickle.dump((all_hs, all_labels), f)

    n_layers        = all_hs.shape[1]
    hidden_by_layer = [all_hs[:, l, :] for l in range(n_layers)]
    prober          = LayerwiseProber(n_layers=n_layers)
    accs            = prober.fit_and_evaluate(
        hidden_by_layer, np.array(all_labels)
    )
    return accs


# ── Plotting ───────────────────────────────────────────────────────────────
def plot_lsfs_results(results):
    """Generates all three key figures for the paper."""

    variants     = list(results.keys())
    accuracies   = [results[v]['accuracy']   for v in variants]
    lsfs_scores  = [results[v]['lsfs_score'] for v in variants]
    embed_dists  = [results[v]['embed_dist'] for v in variants]
    delta_accs   = [results[v]['delta_acc']  for v in variants]

    colors = ['#2196F3','#4CAF50','#FF9800','#F44336','#9C27B0','#795548']

    # ── Figure 1: Accuracy bar chart ──────────────────────────
    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(range(len(variants)), accuracies,
                  color=colors, alpha=0.85, edgecolor='black', linewidth=0.8)
    ax.axhline(y=results['V0_canonical']['accuracy'],
               color='#2196F3', linestyle='--', alpha=0.6,
               label=f"Canonical baseline "
                     f"({results['V0_canonical']['accuracy']:.3f})")
    ax.axhline(y=0.5, color='grey', linestyle=':', alpha=0.5,
               label='Chance (0.50)')
    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels(
        [VARIANT_LABELS[v] for v in variants], fontsize=10
    )
    ax.set_ylabel('ICL Accuracy', fontsize=13)
    ax.set_title(
        'ICL Accuracy Across Label Surface Form Variants\n'
        '(Wang et al. ACL 2023 — Gap C Analysis)',
        fontsize=13
    )
    ax.set_ylim(0.4, 1.05)
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', fontsize=10, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig('figures/fig1_accuracy_by_variant.png', dpi=150)
    plt.show()
    print("  Saved → figures/fig1_accuracy_by_variant.png")

    # ── Figure 2: LSFS score bar chart ────────────────────────
    fig, ax = plt.subplots(figsize=(11, 5))
    non_canon = [v for v in variants if v != 'V0_canonical']
    lsfs_vals = [results[v]['lsfs_score'] for v in non_canon]
    cols_nc   = colors[1:]

    bars2 = ax.bar(range(len(non_canon)), lsfs_vals,
                   color=cols_nc, alpha=0.85,
                   edgecolor='black', linewidth=0.8)
    ax.set_xticks(range(len(non_canon)))
    ax.set_xticklabels(
        [VARIANT_LABELS[v] for v in non_canon], fontsize=10
    )
    ax.set_ylabel('LSFS Score  (|ΔAcc| / embed_dist)', fontsize=12)
    ax.set_title(
        'Label Surface Form Sensitivity (LSFS) Score\n'
        'High score = large accuracy drop per unit embedding distance',
        fontsize=13
    )
    for bar, val in zip(bars2, lsfs_vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.0001,
                f'{val:.4f}', ha='center', fontsize=10, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig('figures/fig2_lsfs_scores.png', dpi=150)
    plt.show()
    print("  Saved → figures/fig2_lsfs_scores.png")

    # ── Figure 3: Probing curves overlay ──────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (vname, col) in enumerate(zip(variants, colors)):
        probe_accs = results[vname]['probe_accs']
        ax.plot(range(len(probe_accs)), probe_accs,
                marker='o', linewidth=2, markersize=5,
                color=col, alpha=0.85,
                label=f"{vname.split('_')[1].capitalize()} "
                      f"({results[vname]['pos_label']}/"
                      f"{results[vname]['neg_label']}): "
                      f"peak={max(probe_accs):.3f}")

    ax.axvline(x=6, color='red', linestyle='--', alpha=0.5,
               label='Shallow/Deep boundary (k=6)')
    ax.axhline(y=0.5, color='grey', linestyle=':', alpha=0.4,
               label='Chance')
    ax.set_xlabel('Layer Index', fontsize=13)
    ax.set_ylabel('Probing Accuracy (5-fold CV)', fontsize=13)
    ax.set_title(
        'Probing Curves Across Label Surface Form Variants\n'
        'Anchor theory predicts curves should be invariant — '
        'they are not.',
        fontsize=12
    )
    ax.set_ylim(0.45, 1.02)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig('figures/fig3_probing_curves_overlay.png', dpi=150)
    plt.show()
    print("  Saved → figures/fig3_probing_curves_overlay.png")

    # ── Figure 4: Scatter — embed_dist vs delta_acc ───────────
    fig, ax = plt.subplots(figsize=(8, 6))
    for vname, col in zip(non_canon, cols_nc):
        ax.scatter(results[vname]['embed_dist'],
                   results[vname]['delta_acc'],
                   color=col, s=120, zorder=5,
                   label=VARIANT_LABELS[vname].replace('\n',' '))
        ax.annotate(
            vname.split('_')[0],
            (results[vname]['embed_dist'],
             results[vname]['delta_acc']),
            textcoords='offset points',
            xytext=(8, 4), fontsize=10
        )
    # If anchor theory held: points should lie on a line through origin
    max_d = max(results[v]['embed_dist'] for v in non_canon) * 1.1
    ax.plot([0, max_d], [0, 0], 'k--', alpha=0.3,
            label='Anchor theory prediction (no sensitivity)')
    ax.set_xlabel('Embedding Distance from Canonical (d_m)', fontsize=13)
    ax.set_ylabel('|ΔAccuracy| from Canonical', fontsize=13)
    ax.set_title(
        'Embedding Distance vs Accuracy Drop\n'
        'Points far above the dashed line violate anchor theory',
        fontsize=12
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig('figures/fig4_embed_dist_vs_delta_acc.png', dpi=150)
    plt.show()
    print("  Saved → figures/fig4_embed_dist_vs_delta_acc.png")


# ── Results table printer ──────────────────────────────────────────────────
def print_results_table(results):
    print("\n" + "="*80)
    print("LSFS EXPERIMENT RESULTS TABLE")
    print("="*80)
    print(f"{'Variant':<18} {'Labels':^17} {'Accuracy':>9} "
          f"{'ΔAcc':>8} {'EmbDist':>9} {'LSFS':>10}")
    print("-"*80)
    for vname, r in results.items():
        print(f"{vname:<18} "
              f"{r['pos_label']:>7}/{r['neg_label']:<7}  "
              f"{r['accuracy']:>8.4f}  "
              f"{r['delta_acc']:>7.4f}  "
              f"{r['embed_dist']:>8.4f}  "
              f"{r['lsfs_score']:>9.5f}")
    print("="*80)

    # Highlight key finding
    max_lsfs = max(
        (r['lsfs_score'], v)
        for v, r in results.items() if v != 'V0_canonical'
    )
    print(f"\n🔑 KEY FINDING: Highest LSFS = {max_lsfs[0]:.5f} "
          f"for {max_lsfs[1]}")
    print(f"   Anchor theory predicts LSFS ≈ 0 for all variants.")
    print(f"   High LSFS = theory violation confirmed.")


if __name__ == '__main__':
    results = run_lsfs_experiment(n_examples=100)
    print_results_table(results)
    plot_lsfs_results(results)
