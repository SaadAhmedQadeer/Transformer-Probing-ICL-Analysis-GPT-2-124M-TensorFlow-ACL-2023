# forensic_debug.py
# Run this INSTEAD of main_reproduction.py
# Takes ~2 minutes. Paste ALL output back to your advisor.
# This will definitively identify the exact failure point.

import numpy as np
import pickle, os
from datasets import load_dataset
from corrected_extractor import CorrectedHiddenStateExtractor

print("="*60)
print("FORENSIC DIAGNOSTIC v1")
print("="*60)

# ── STEP 1: Tokenisation ground truth ─────────────────────────
print("\n[STEP 1] GPT-2 tokenisation ground truth")
extractor = CorrectedHiddenStateExtractor('gpt2')
tok = extractor.tokenizer

tests = ["Positive", " Positive", "Negative", " Negative"]
for t in tests:
    ids = tok.encode(t, add_special_tokens=False)
    print(f"  encode('{t}') = {ids}")

# ── STEP 2: Build ONE prompt and decode every token ───────────
print("\n[STEP 2] Full token-by-token prompt decode")
dataset    = load_dataset('sst2')
train_pool = list(dataset['train'])
demos = (
    [x for x in train_pool if x['label'] == 1][:2] +
    [x for x in train_pool if x['label'] == 0][:2]
)
label_map = {0: "Negative", 1: "Positive"}

prompt = ""
for d in demos:
    prompt += f"Review: {d['sentence']}\nSentiment: {label_map[d['label']]}\n\n"
test_ex = list(dataset['validation'])[0]
prompt += f"Review: {test_ex['sentence']}\nSentiment:"

input_ids = tok.encode(prompt)
tokens    = tok.convert_ids_to_tokens(input_ids)

print(f"  Total tokens: {len(input_ids)}")
print(f"  Full token sequence:")
for idx, (tid, tkn) in enumerate(zip(input_ids, tokens)):
    marker = "  <<<< LABEL CANDIDATE" if tkn in [
        'Positive','Negative','ĠPositive','ĠNegative'
    ] else ""
    print(f"    [{idx:3d}] id={tid:6d}  '{tkn}'{marker}")

# ── STEP 3: Label position detection ──────────────────────────
print("\n[STEP 3] Label position detection")
found = extractor.get_label_positions_corrected(
    input_ids, ["Positive", "Negative"]
)
print(f"  found = {found}")
print(f"  Count = {len(found)}  (expected >= 4)")

if len(found) == 0:
    print("\n  ❌ CRITICAL: Zero label positions found.")
    print("  This is the root cause of the flat curve.")
    print("  Check Step 2 output — what tokens appear for labels?")
    print("  Look for 'ĠPositive' or 'ĠNegative' in the list above.")
else:
    print(f"  ✅ Found {len(found)} label positions")
    for pos, lbl in found:
        ctx = tokens[max(0,pos-2):pos+3]
        print(f"    '{lbl}' @ idx {pos} | context={ctx}")

# ── STEP 4: Single forward pass hidden state check ────────────
print("\n[STEP 4] Single forward pass — hidden state extraction")
if len(found) >= 1:
    target_idx = np.array([found[-1][0]], dtype=np.int32)
    print(f"  Extracting at position {target_idx[0]} ...")
    hs = extractor.extract(prompt, target_idx)
    print(f"  Returned type  : {type(hs)}")
    print(f"  Returned shape : {hs.shape if hasattr(hs,'shape') else 'N/A'}")
    print(f"  Expected shape : (13, 1, 768)")
    if hasattr(hs, 'shape') and hs.shape == (13, 1, 768):
        print(f"  Shape ✅")
        print(f"  Variance layer  0: {np.var(hs[0,0,:]):.6f}")
        print(f"  Variance layer  6: {np.var(hs[6,0,:]):.6f}")
        print(f"  Variance layer 12: {np.var(hs[12,0,:]):.6f}")
        print(f"  Any NaN: {np.isnan(hs).any()}")
        print(f"  Min/Max: {hs.min():.4f} / {hs.max():.4f}")
    else:
        print(f"  ❌ Wrong shape or type — extraction is broken")
else:
    print("  Skipping — no label positions found in Step 3")

# ── STEP 5: Run 5 examples and compare hidden states ──────────
print("\n[STEP 5] Cross-example variance check (5 examples)")
print("  If hidden states are identical across examples →")
print("  probe sees same input for every sample → 50% accuracy")

test_pool = list(dataset['validation'])
pos_ex = [x for x in test_pool if x['label'] == 1][:3]
neg_ex = [x for x in test_pool if x['label'] == 0][:3]
five_ex = pos_ex[:2] + neg_ex[:2]   # 2 pos, 2 neg

layer12_vecs = []
layer12_labels = []

for ex in five_ex:
    p = ""
    for d in demos:
        p += f"Review: {d['sentence']}\nSentiment: {label_map[d['label']]}\n\n"
    p += f"Review: {ex['sentence']}\nSentiment:"

    ids  = tok.encode(p)
    fp   = extractor.get_label_positions_corrected(ids, ["Positive","Negative"])

    if not fp:
        print(f"  ⚠️  No label found for: '{ex['sentence'][:40]}...'")
        continue

    tidx = np.array([fp[-1][0]], dtype=np.int32)
    hs   = extractor.extract(p, tidx)   # [13,1,768]

    if hs is None or hs.size == 0:
        continue

    layer12_vecs.append(hs[12, 0, :])
    layer12_labels.append(ex['label'])
    print(f"  label={ex['label']} | anchor='{fp[-1][1]}'@{fp[-1][0]} "
          f"| hs[12] norm={np.linalg.norm(hs[12,0,:]):.4f} "
          f"| first5={np.round(hs[12,0,:5],3)}")

if len(layer12_vecs) >= 2:
    vecs = np.array(layer12_vecs)
    print(f"\n  Cross-example variance at layer 12: {np.var(vecs):.6f}")
    if np.var(vecs) < 1e-4:
        print("  ❌ VECTORS ARE IDENTICAL — wrong positions or caching bug")
    else:
        print("  ✅ Vectors differ across examples — extraction is correct")

    # Pairwise cosine similarity
    from numpy.linalg import norm
    print(f"\n  Pairwise cosine similarities (should NOT all be ~1.0):")
    for i in range(len(vecs)):
        for j in range(i+1, len(vecs)):
            cos = np.dot(vecs[i], vecs[j]) / (norm(vecs[i]) * norm(vecs[j]))
            print(f"    ex{i}(label={layer12_labels[i]}) vs "
                  f"ex{j}(label={layer12_labels[j]}): cos={cos:.4f}")

print("\n" + "="*60)
print("DIAGNOSTIC COMPLETE — paste ALL output above to advisor")
print("="*60)
