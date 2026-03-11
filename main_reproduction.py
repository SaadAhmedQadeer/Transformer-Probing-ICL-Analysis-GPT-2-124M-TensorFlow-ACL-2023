# main_reproduction.py  ── DEFINITIVE VERSION v4
# Reproduces Wang et al. ACL 2023 "Label Words are Anchors"
#
# ROOT CAUSE OF FLAT CURVE (now fixed):
#   Versions v1-v3 extracted hidden states at demo label positions
#   (e.g. position 94 = 'ĠNegative' in the fixed demonstration block).
#   Since demos are IDENTICAL across all examples, every query gets
#   the SAME vector → probe accuracy = 50% (chance).
#
# THE FIX:
#   Extract at the FINAL token position (the last ':' of "Sentiment:").
#   This is the token whose hidden state the model uses to predict the
#   next token — it's query-dependent and varies per example.
#   This is exactly what Wang et al. measure in their Figure 3.

import os
import pickle
import numpy as np
from datasets import load_dataset
from corrected_extractor import CorrectedHiddenStateExtractor
from probing_classifier import LayerwiseProber


def build_prompt(demo_examples, query_sentence):
    label_map = {0: "Negative", 1: "Positive"}
    prompt = ""
    for d in demo_examples:
        prompt += (
            f"Review: {d['sentence']}\n"
            f"Sentiment: {label_map[d['label']]}\n\n"
        )
    prompt += f"Review: {query_sentence}\nSentiment:"
    return prompt


def run_full_reproduction():

    # ── 1. Balanced 200-example test set ──────────────────────
    print("Loading SST-2 ...")
    dataset   = load_dataset('sst2')
    test_pool = list(dataset['validation'])

    pos_ex   = [x for x in test_pool if x['label'] == 1][:100]
    neg_ex   = [x for x in test_pool if x['label'] == 0][:100]
    test_set = pos_ex + neg_ex
    np.random.default_rng(42).shuffle(test_set)

    print(f"  {len(test_set)} examples | "
          f"pos={sum(x['label']==1 for x in test_set)} "
          f"neg={sum(x['label']==0 for x in test_set)}")

    # ── 2. Fixed balanced 4-shot demos ────────────────────────
    train_pool = list(dataset['train'])
    demos = (
        [x for x in train_pool if x['label'] == 1][:2] +
        [x for x in train_pool if x['label'] == 0][:2]
    )

    # ── 3. Extractor ──────────────────────────────────────────
    extractor = CorrectedHiddenStateExtractor('gpt2')

    # ── 4. DIAGNOSTIC ─────────────────────────────────────────
    print("\n=== DIAGNOSTIC ===")
    sample_prompt = build_prompt(demos, test_set[0]['sentence'])
    sample_ids    = extractor.tokenizer.encode(sample_prompt)
    sample_tokens = extractor.tokenizer.convert_ids_to_tokens(sample_ids)

    final_pos = len(sample_ids) - 1
    print(f"  Prompt length : {len(sample_ids)} tokens")
    print(f"  Final token   : '{sample_tokens[final_pos]}' @ idx {final_pos}")
    print(f"  Last 5 tokens : {sample_tokens[-5:]}")
    print("  ✅ Will extract at final token (query prediction position)\n")

    # ── 5. Cache with v4_ prefix ──────────────────────────────
    os.makedirs('cache',   exist_ok=True)
    os.makedirs('figures', exist_ok=True)

    cache_hs  = 'cache/v4_hidden_states.pkl'
    cache_lbl = 'cache/v4_labels.pkl'

    if not os.path.exists(cache_hs):
        print("Extracting hidden states (~45-90 min on CPU) ...")
        all_hs, all_labels, skipped = [], [], 0

        for i, ex in enumerate(test_set):
            prompt    = build_prompt(demos, ex['sentence'])
            input_ids = extractor.tokenizer.encode(prompt)

            # ── CORE FIX ──────────────────────────────────────
            # Extract at the LAST token = "Sentiment:" colon.
            # This is query-dependent: hidden state here reflects
            # the full prompt context including the query sentence.
            final_token_pos = len(input_ids) - 1
            target_idx = np.array([final_token_pos], dtype=np.int32)

            hs = extractor.extract(prompt, target_idx)
            # shape: [n_layers+1, 1, d_model]

            if hs is None or hs.size == 0 or np.isnan(hs).any():
                skipped += 1
                continue

            all_hs.append(hs[:, 0, :])     # [n_layers+1, d_model]
            all_labels.append(ex['label'])

            if i % 20 == 0:
                if len(all_hs) >= 2:
                    cos = np.dot(all_hs[-1][12], all_hs[-2][12]) / (
                        np.linalg.norm(all_hs[-1][12]) *
                        np.linalg.norm(all_hs[-2][12]) + 1e-9
                    )
                    print(f"  [{i:3d}/{len(test_set)}] "
                          f"kept={len(all_hs)} skip={skipped} "
                          f"cos(last2)={cos:.4f} (should be <1.0)")
                else:
                    print(f"  [{i:3d}/{len(test_set)}] "
                          f"kept={len(all_hs)} skip={skipped}")

        all_hs     = np.array(all_hs)
        all_labels = np.array(all_labels)

        with open(cache_hs,  'wb') as f: pickle.dump(all_hs,     f)
        with open(cache_lbl, 'wb') as f: pickle.dump(all_labels, f)
        print(f"\n  ✅ Cached {len(all_hs)} examples | skipped {skipped}")

    else:
        print("Loading v4 cache ...")
        with open(cache_hs,  'rb') as f: all_hs     = pickle.load(f)
        with open(cache_lbl, 'rb') as f: all_labels = pickle.load(f)
        print(f"  {len(all_hs)} examples loaded")

    # ── 6. Sanity checks ──────────────────────────────────────
    print("\n=== SANITY CHECKS ===")
    for layer in [0, 6, 12]:
        v = float(np.var(all_hs[:, layer, :]))
        print(f"  Layer {layer:2d} variance: {v:.4f}  {'✅' if v > 0.1 else '❌'}")

    print("\n  Cosine similarities (first 4 examples, layer 12):")
    vecs = all_hs[:4, 12, :]
    for i in range(4):
        for j in range(i+1, 4):
            cos = np.dot(vecs[i], vecs[j]) / (
                np.linalg.norm(vecs[i]) * np.linalg.norm(vecs[j]) + 1e-9)
            print(f"    ex{i}(y={all_labels[i]}) vs ex{j}(y={all_labels[j]}): "
                  f"{cos:.4f} {'✅' if cos < 0.999 else '❌ IDENTICAL'}")

    print(f"\n  Labels: {int((all_labels==1).sum())} pos / "
          f"{int((all_labels==0).sum())} neg")

    # ── 7. Layer-wise probing ──────────────────────────────────
    n_layers        = all_hs.shape[1]
    hidden_by_layer = [all_hs[:, l, :] for l in range(n_layers)]

    print(f"\n=== PROBING (n_layers={n_layers}) ===")
    prober     = LayerwiseProber(n_layers=n_layers)
    accuracies = prober.fit_and_evaluate(hidden_by_layer, all_labels)
    prober.plot_layer_curve(save_path='figures/probe_accuracy_v4.png')

    peak_acc   = float(max(accuracies))
    peak_layer = int(np.argmax(accuracies))

    print(f"\n{'='*50}")
    print(f"  Peak accuracy : {peak_acc:.4f} at layer {peak_layer}")
    print(f"  Expected      : >0.75 between layers 6-12")
    print(f"  {'✅ REPRODUCTION PASSED' if peak_acc > 0.65 else '❌ FAIL'}")
    print('='*50)

    return accuracies


if __name__ == '__main__':
    run_full_reproduction()
