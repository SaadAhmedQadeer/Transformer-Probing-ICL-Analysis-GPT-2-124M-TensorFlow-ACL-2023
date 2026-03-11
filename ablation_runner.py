# ablation_runner.py
# Full Ablation Study Orchestrator
#
# Runs all experiments in sequence and produces the final
# results tables needed for the technical report and CV.
#
# Ablation design — each experiment isolates ONE variable:
#
#   ABL-1: Label surface form only (variant accuracy)
#          → Confirms LSFS effect exists
#
#   ABL-2: Demo order control
#          → Rules out that accuracy drop is due to
#            demo ordering, not label surface form
#
#   ABL-3: Projection fix with alpha sweep
#          → Tests whether embedding-space correction works
#
#   ABL-4: Projection fix on canonical labels
#          → Verifies fix does NOT hurt canonical accuracy
#            (sanity check / negative control)
#
#   ABL-5: K-shot count (k=1,2,4)
#          → Tests whether LSFS effect scales with k

import os, json
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from corrected_extractor import CorrectedHiddenStateExtractor
from lsfs_experiment import (
    VARIANTS, VARIANT_LABELS,
    build_variant_prompt, predict_icl,
    run_lsfs_experiment, plot_lsfs_results, print_results_table
)
from projection_fix import (
    ProjectionFix,
    run_projection_experiment,
    plot_projection_results,
    print_projection_table
)


def load_data(n_examples=100):
    """Shared data loader used across all ablations."""
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
    return dataset, test_set, demos


# ── ABL-2: Demo order control ──────────────────────────────────────────────
def ablation_demo_order(extractor, test_set, demos, n_orders=3):
    """
    Tests 3 different demo orderings with canonical labels.
    If accuracy is stable across orderings, the LSFS effect
    is genuinely due to surface form, not ordering.
    """
    print("\n=== ABL-2: Demo Order Control ===")
    results = {}
    label_map = {0: "Negative", 1: "Positive"}

    orderings = [
        ('pos_first',  demos),
        ('neg_first',  list(reversed(demos))),
        ('interleaved',[demos[0], demos[2], demos[1], demos[3]])
    ]

    for order_name, ordered_demos in orderings:
        correct = 0
        for ex in test_set:
            prompt = build_variant_prompt(
                ordered_demos, ex['sentence'],
                'Positive', 'Negative'
            )
            pred, _, _ = predict_icl(
                extractor, prompt, 'Positive', 'Negative'
            )
            correct += (pred == ex['label'])

        acc = correct / len(test_set)
        results[order_name] = acc
        print(f"  Order '{order_name}': acc={acc:.4f}")

    std = np.std(list(results.values()))
    print(f"  Std across orderings: {std:.4f}")
    print(f"  {'✅ Order is not the confound' if std < 0.03 else '⚠️ Order affects accuracy'}")

    return results


# ── ABL-5: K-shot scaling ──────────────────────────────────────────────────
def ablation_kshot_scaling(extractor, test_set, train_pool,
                             k_values=(1, 2, 4)):
    """
    Tests LSFS effect at different k-shot counts.
    Hypothesis: LSFS effect is larger at k=1 (less context)
    and smaller at k=4 (more demonstrations provide more signal).
    """
    print("\n=== ABL-5: K-shot Scaling ===")
    results = {}

    for k in k_values:
        results[k] = {}
        # Build k demos (balanced as much as possible)
        k_pos  = max(1, k // 2)
        k_neg  = k - k_pos
        k_demos = (
            [x for x in train_pool if x['label'] == 1][:k_pos] +
            [x for x in train_pool if x['label'] == 0][:k_neg]
        )

        for vname, (pos_lbl, neg_lbl) in VARIANTS.items():
            correct = 0
            for ex in test_set[:50]:  # Use 50 for speed in ablation
                prompt = build_variant_prompt(
                    k_demos, ex['sentence'], pos_lbl, neg_lbl
                )
                pred, _, _ = predict_icl(
                    extractor, prompt, pos_lbl, neg_lbl
                )
                correct += (pred == ex['label'])

            acc = correct / 50
            results[k][vname] = acc

        print(f"  k={k}:")
        for vname, acc in results[k].items():
            print(f"    {vname:<20}: {acc:.4f}")

    return results


# ── Final summary table ────────────────────────────────────────────────────
def print_final_summary(lsfs_results, proj_results, order_results,
                          kshot_results):
    """
    Prints the complete ablation summary table as it would appear
    in Table 2 of the technical report.
    """
    print("\n" + "="*80)
    print("COMPLETE ABLATION SUMMARY — Table 2 of Technical Report")
    print("="*80)

    print("\n── Main Result (ABL-1): LSFS by Variant ──")
    print_results_table(lsfs_results)

    print("\n── ABL-2: Demo Order Robustness ──")
    for order, acc in order_results.items():
        print(f"  {order:<15}: {acc:.4f}")

    print("\n── ABL-3: Projection Fix ──")
    print_projection_table(proj_results)

    print("\n── ABL-5: K-shot Scaling ──")
    for k, variant_accs in kshot_results.items():
        canon  = variant_accs.get('V0_canonical', 0)
        symbol = variant_accs.get('V3_symbols',   0)
        gap    = canon - symbol
        print(f"  k={k}: canon={canon:.4f}  "
              f"symbols={symbol:.4f}  gap={gap:.4f}")

    print("\n" + "="*80)
    print("KEY SCIENTIFIC CLAIMS (for technical report)")
    print("="*80)

    if lsfs_results:
        canon_acc = lsfs_results['V0_canonical']['accuracy']
        max_drop  = max(
            lsfs_results[v]['delta_acc']
            for v in lsfs_results if v != 'V0_canonical'
        )
        max_lsfs  = max(
            lsfs_results[v]['lsfs_score']
            for v in lsfs_results if v != 'V0_canonical'
        )
        print(f"\n  1. Canonical ICL accuracy: {canon_acc:.4f}")
        print(f"  2. Maximum accuracy drop:  {max_drop:.4f} "
              f"({max_drop*100:.1f}% absolute)")
        print(f"  3. Maximum LSFS score:     {max_lsfs:.5f}")
        print(f"     → Anchor theory predicts LSFS ≈ 0; "
              f"we observe {max_lsfs:.3f}")

    if proj_results:
        recoveries = [
            proj_results['variants'][v].get('gap_recovered_pct', 0)
            for v in proj_results['variants']
            if v != 'V0_canonical'
        ]
        if recoveries:
            mean_rec = np.mean(recoveries)
            print(f"\n  4. Mean gap recovery via projection: "
                  f"{mean_rec:.1f}%")
            print(f"     → Embedding-space correction recovers "
                  f"{mean_rec:.0f}% of accuracy on average")


# ── Master runner ──────────────────────────────────────────────────────────
def run_all_ablations(n_examples=100):
    """
    Runs all 5 ablations in sequence. Results are cached so each
    ablation only runs once. Safe to re-run.
    """
    print("="*60)
    print("FULL ABLATION STUDY")
    print(f"n_examples={n_examples} per variant")
    print("="*60)

    dataset, test_set, demos = load_data(n_examples)
    extractor = CorrectedHiddenStateExtractor('gpt2')

    # ── ABL-1: Main LSFS experiment ───────────────────────────
    print("\n>>> ABL-1: LSFS Experiment")
    lsfs_results = run_lsfs_experiment(n_examples=n_examples)
    plot_lsfs_results(lsfs_results)

    # ── ABL-2: Demo order control ─────────────────────────────
    print("\n>>> ABL-2: Demo Order Control")
    order_cache = 'results/abl2_order.json'
    if os.path.exists(order_cache):
        with open(order_cache) as f:
            order_results = json.load(f)
    else:
        order_results = ablation_demo_order(extractor, test_set, demos)
        with open(order_cache, 'w') as f:
            json.dump(order_results, f, indent=2)

    # ── ABL-3: Projection fix ─────────────────────────────────
    print("\n>>> ABL-3: Projection Fix")
    proj_results = run_projection_experiment(n_examples=n_examples)
    plot_projection_results(proj_results)

    # ── ABL-4: Projection on canonical (sanity check) ─────────
    print("\n>>> ABL-4: Canonical Under Projection (Sanity Check)")
    print("  Verifying projection does NOT hurt canonical accuracy ...")
    projector = ProjectionFix(extractor)
    for alpha in [0.0, 0.3, 0.5]:
        projector.apply_projection_to_model(alpha)
        correct = 0
        for ex in test_set[:50]:
            prompt = build_variant_prompt(
                demos, ex['sentence'], 'Positive', 'Negative'
            )
            pred, _, _ = predict_icl(
                extractor, prompt, 'Positive', 'Negative'
            )
            correct += (pred == ex['label'])
        acc = correct / 50
        print(f"  Canonical at α={alpha}: {acc:.4f} "
              f"{'✅' if alpha==0.0 or acc >= 0.75 else '⚠️'}")
        projector.restore_original_embeddings()

    # ── ABL-5: K-shot scaling ─────────────────────────────────
    print("\n>>> ABL-5: K-shot Scaling")
    kshot_cache = 'results/abl5_kshot.json'
    if os.path.exists(kshot_cache):
        with open(kshot_cache) as f:
            kshot_results = {int(k): v
                             for k, v in json.load(f).items()}
    else:
        train_pool = list(dataset['train'])
        kshot_results = ablation_kshot_scaling(
            extractor, test_set, train_pool
        )
        with open(kshot_cache, 'w') as f:
            json.dump(kshot_results, f, indent=2)

    # ── Final summary ─────────────────────────────────────────
    print_final_summary(
        lsfs_results, proj_results,
        order_results, kshot_results
    )

    print("\n✅ ALL ABLATIONS COMPLETE")
    print("   Results saved in results/")
    print("   Figures saved in figures/")
    print("   Ready for Phase 5: Academic Deliverables")

    return lsfs_results, proj_results, order_results, kshot_results


if __name__ == '__main__':
    run_all_ablations(n_examples=100)
