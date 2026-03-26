# projection_fix.py
# Canonical Anchor Projection — Novel Mathematical Fix
#
# MOTIVATION:
#   The LSFS experiment shows that label surface form variants with
#   large embedding distances from canonical labels cause disproportionate
#   accuracy drops. The anchor theory (Wang et al.) has no mechanism
#   to explain or fix this.
#
# THE FIX:
#   Before the forward pass, project variant label embeddings toward
#   the canonical label embeddings in GPT-2's embedding matrix.
#
#   Projection operator:
#     Ẽ(v_m) = E(v_0) + α · (E(v_m) - E(v_0)) / ||E(v_m) - E(v_0)||_2
#
#   α=0 → fully replace variant with canonical embedding (maximum correction)
#   α=1 → keep variant embedding unchanged (no correction)
#
#   We search for α* that minimises accuracy degradation across variants.

import os
import json
import copy
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from datasets import load_dataset
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer
from corrected_extractor import CorrectedHiddenStateExtractor
from lsfs_experiment import (
    VARIANTS, build_variant_prompt, predict_icl
)


# ── Projection operator ────────────────────────────────────────────────────
class ProjectionFix:
    """
    Implements the Canonical Anchor Projection.

    Modifies GPT-2's embedding matrix in-place (on a copy) by
    interpolating variant label token embeddings toward their
    canonical counterparts.

    This is a TRAINING-FREE intervention — no gradient updates,
    no fine-tuning. Pure embedding-space geometry.
    """

    def __init__(self, extractor):
        self.extractor = extractor
        self.tokenizer = extractor.tokenizer

        # Get embedding matrix: [vocab_size, d_model]
        self.original_embed = extractor.model.weights[0].numpy().copy()
        self.vocab_size, self.d_model = self.original_embed.shape

    def get_token_id(self, word):
        """Returns token ID for space-prefixed word (mid-sentence form)."""
        ids = self.tokenizer.encode(' ' + word, add_special_tokens=False)
        return ids[0]

    def compute_projection(self, alpha,
                            variants_to_fix=None):
        """
        Computes the projected embedding matrix for a given alpha.

        For each variant label word v_m with canonical v_0:
            Ẽ(v_m) = E(v_0) + α · (E(v_m) - E(v_0))
                              ──────────────────────
                              NOTE: alpha scales the RESIDUAL
                              from canonical. So:
                              α=0 → Ẽ(v_m) = E(v_0) [full projection]
                              α=1 → Ẽ(v_m) = E(v_m) [no change]

        This is equivalent to:
            Ẽ(v_m) = (1-α)·E(v_0) + α·E(v_m)
        i.e., a linear interpolation between canonical and variant.
        """
        if variants_to_fix is None:
            variants_to_fix = list(VARIANTS.keys())

        new_embed = self.original_embed.copy()

        # Canonical embeddings
        pos_canon_id = self.get_token_id('Positive')
        neg_canon_id = self.get_token_id('Negative')
        e_pos_canon  = self.original_embed[pos_canon_id]
        e_neg_canon  = self.original_embed[neg_canon_id]

        for vname in variants_to_fix:
            if vname == 'V0_canonical':
                continue
            pos_lbl, neg_lbl = VARIANTS[vname]

            pos_var_id = self.get_token_id(pos_lbl)
            neg_var_id = self.get_token_id(neg_lbl)

            e_pos_var = self.original_embed[pos_var_id]
            e_neg_var = self.original_embed[neg_var_id]

            # Linear interpolation toward canonical
            new_embed[pos_var_id] = (
                (1 - alpha) * e_pos_canon + alpha * e_pos_var
            )
            new_embed[neg_var_id] = (
                (1 - alpha) * e_neg_canon + alpha * e_neg_var
            )

        return new_embed

    def apply_projection_to_model(self, alpha, variants_to_fix=None):
        """
        Applies projection by patching the embedding matrix in the
        model's TF variables. Returns a context manager that restores
        the original embeddings afterward.
        """
        new_embed = self.compute_projection(alpha, variants_to_fix)
        # Assign to the embedding weight
        self.extractor.model.weights[0].assign(new_embed)

        # Also patch the LM head model if it exists
        if hasattr(self.extractor, '_lm_model'):
            self.extractor._lm_model.weights[0].assign(new_embed)

    def restore_original_embeddings(self):
        """Restores original embedding matrix."""
        self.extractor.model.weights[0].assign(self.original_embed)
        if hasattr(self.extractor, '_lm_model'):
            self.extractor._lm_model.weights[0].assign(self.original_embed)


# ── Alpha sweep experiment ─────────────────────────────────────────────────
def run_projection_experiment(n_examples=100, force_rerun=False):
    """
    Sweeps alpha ∈ {0.0, 0.1, 0.2, ..., 1.0} and measures ICL
    accuracy for each variant at each alpha value.

    Key questions:
      1. What is α* (optimal alpha) that maximises mean accuracy
         across non-canonical variants?
      2. Does the fix recover the canonical accuracy?
      3. Is the fix uniform across variants, or does it help
         symbolic/numeric more than synonyms?
    """
    os.makedirs('results', exist_ok=True)
    results_path = 'results/projection_results.json'

    if os.path.exists(results_path) and not force_rerun:
        print("Loading cached projection results ...")
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
    projector = ProjectionFix(extractor)

    # Alpha values to sweep
    alphas = [round(a * 0.1, 1) for a in range(11)]
    # [0.0, 0.1, 0.2, ..., 1.0]

    results = {
        'alphas'   : alphas,
        'variants' : {}
    }

    # Run baseline (alpha=1.0, no projection) first
    for vname, (pos_lbl, neg_lbl) in VARIANTS.items():
        results['variants'][vname] = {
            'pos_label'    : pos_lbl,
            'neg_label'    : neg_lbl,
            'accs_by_alpha': {}
        }

    # ── Alpha sweep ────────────────────────────────────────────
    for alpha in alphas:
        print(f"\n{'─'*50}")
        print(f"Alpha = {alpha:.1f}  "
              f"({'no correction' if alpha==1.0 else 'full correction' if alpha==0.0 else 'partial correction'})")
        print(f"{'─'*50}")

        # Apply projection
        projector.apply_projection_to_model(alpha)

        for vname, (pos_lbl, neg_lbl) in VARIANTS.items():
            correct = 0
            for ex in test_set:
                prompt = build_variant_prompt(
                    demos, ex['sentence'], pos_lbl, neg_lbl
                )
                pred, _, _ = predict_icl(
                    extractor, prompt, pos_lbl, neg_lbl
                )
                correct += (pred == ex['label'])

            acc = correct / len(test_set)
            results['variants'][vname]['accs_by_alpha'][str(alpha)] = acc
            print(f"  {vname:<18}  acc={acc:.4f}")

        # Restore for next iteration
        projector.restore_original_embeddings()

    # ── Find optimal alpha per variant ────────────────────────
    print("\n=== OPTIMAL ALPHA ANALYSIS ===")
    for vname in VARIANTS:
        if vname == 'V0_canonical':
            continue
        accs = results['variants'][vname]['accs_by_alpha']
        best_alpha = max(accs.items(), key=lambda x: float(x[1]))
        results['variants'][vname]['optimal_alpha'] = float(best_alpha[0])
        results['variants'][vname]['optimal_acc']   = best_alpha[1]
        print(f"  {vname:<18}: α*={best_alpha[0]}  "
              f"acc={best_alpha[1]:.4f}")

    # ── Recovery rate ──────────────────────────────────────────
    canon_acc_no_proj = results['variants']['V0_canonical'][
        'accs_by_alpha']['1.0']
    print(f"\n  Canonical accuracy (no projection): {canon_acc_no_proj:.4f}")

    for vname in VARIANTS:
        if vname == 'V0_canonical': continue
        base_acc = results['variants'][vname]['accs_by_alpha']['1.0']
        best_acc = results['variants'][vname].get('optimal_acc', base_acc)
        gap      = canon_acc_no_proj - base_acc
        recovery = (best_acc - base_acc) / (gap + 1e-9)
        results['variants'][vname]['gap_recovered_pct'] = (
            float(recovery) * 100
        )
        print(f"  {vname:<18}: gap={gap:.4f}  "
              f"recovered={recovery*100:.1f}%")

    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Saved → {results_path}")

    return results


# ── Plotting ───────────────────────────────────────────────────────────────
def plot_projection_results(results):
    alphas   = results['alphas']
    variants = results['variants']

    colors = {
        'V0_canonical' : '#2196F3',
        'V1_synonyms'  : '#4CAF50',
        'V2_abbrev'    : '#FF9800',
        'V3_symbols'   : '#F44336',
        'V4_numeric'   : '#9C27B0',
        'V5_random'    : '#795548',
    }

    # ── Figure 5: Accuracy vs Alpha for all variants ──────────
    fig, ax = plt.subplots(figsize=(12, 6))
    for vname, vdata in variants.items():
        accs = [vdata['accs_by_alpha'][str(a)] for a in alphas]
        ax.plot(alphas, accs,
                marker='o', linewidth=2, markersize=6,
                color=colors.get(vname, 'grey'),
                label=f"{vname.split('_')[1].capitalize()} "
                      f"({vdata['pos_label']}/{vdata['neg_label']})")

    ax.axvline(x=1.0, color='red', linestyle='--', alpha=0.5,
               label='α=1.0 (no projection, baseline)')
    ax.set_xlabel('Projection Strength α\n'
                  '(0 = full projection to canonical, '
                  '1 = no change)',
                  fontsize=12)
    ax.set_ylabel('ICL Accuracy', fontsize=13)
    ax.set_title(
        'Effect of Canonical Anchor Projection on ICL Accuracy\n'
        'α* minimises accuracy degradation across variants',
        fontsize=13
    )
    ax.set_xticks(alphas)
    ax.legend(fontsize=9, loc='lower left')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig('figures/fig5_projection_alpha_sweep.png', dpi=150)
    plt.show()
    print("  Saved → figures/fig5_projection_alpha_sweep.png")

    # ── Figure 6: Recovery rate bar chart ─────────────────────
    non_canon = [v for v in variants if v != 'V0_canonical']
    recoveries = [
        variants[v].get('gap_recovered_pct', 0) for v in non_canon
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    bar_colors = [colors.get(v, 'grey') for v in non_canon]
    bars = ax.bar(range(len(non_canon)), recoveries,
                  color=bar_colors, alpha=0.85,
                  edgecolor='black', linewidth=0.8)
    ax.axhline(y=50, color='red', linestyle='--', alpha=0.6,
               label='50% recovery threshold')
    ax.axhline(y=100, color='green', linestyle='--', alpha=0.4,
               label='Full recovery')
    ax.set_xticks(range(len(non_canon)))
    ax.set_xticklabels(
        [f"{v.split('_')[1].capitalize()}\n"
         f"({variants[v]['pos_label']}/{variants[v]['neg_label']})"
         for v in non_canon],
        fontsize=10
    )
    ax.set_ylabel('Gap Recovery (%)', fontsize=13)
    ax.set_title(
        'Accuracy Gap Recovery via Canonical Anchor Projection\n'
        'Recovery = (Acc(α*) - Acc(α=1)) / (Acc(canon) - Acc(α=1))',
        fontsize=12
    )
    for bar, rec in zip(bars, recoveries):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 1,
                f'{rec:.1f}%', ha='center',
                fontsize=11, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, max(recoveries) * 1.25 if recoveries else 120)
    fig.tight_layout()
    fig.savefig('figures/fig6_gap_recovery.png', dpi=150)
    plt.show()
    print("  Saved → figures/fig6_gap_recovery.png")


def print_projection_table(results):
    variants = results['variants']
    alphas   = results['alphas']

    print("\n" + "="*90)
    print("PROJECTION FIX RESULTS TABLE")
    print("="*90)
    print(f"{'Variant':<18} {'Labels':^15} "
          f"{'Acc(α=1.0)':>11} "
          f"{'Acc(α*)':>9} "
          f"{'α*':>5} "
          f"{'Recovery':>10}")
    print("-"*90)

    canon_acc = variants['V0_canonical']['accs_by_alpha']['1.0']

    for vname, vdata in variants.items():
        base_acc = vdata['accs_by_alpha']['1.0']
        best_acc = vdata.get('optimal_acc', base_acc)
        opt_a    = vdata.get('optimal_alpha', 1.0)
        recovery = vdata.get('gap_recovered_pct', 0.0)

        print(f"{vname:<18} "
              f"{vdata['pos_label']:>6}/{vdata['neg_label']:<6}  "
              f"{base_acc:>10.4f}  "
              f"{best_acc:>8.4f}  "
              f"{opt_a:>4.1f}  "
              f"{recovery:>9.1f}%")

    print("="*90)
    print(f"\n  Canonical (no projection): {canon_acc:.4f}")


if __name__ == '__main__':
    results = run_projection_experiment(n_examples=100)
    print_projection_table(results)
    plot_projection_results(results)
