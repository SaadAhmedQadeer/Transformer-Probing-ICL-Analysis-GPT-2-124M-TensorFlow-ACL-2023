# test_extractor.py
# Run this FIRST (takes ~30 seconds) to verify everything works
# before launching the full 90-minute extraction.
#
# Expected output:
#   ✅ Model loaded
#   ✅ Label positions found (non-empty list)
#   ✅ Context windows show ĠPositive / ĠNegative
#   ✅ Hidden state shape: (13, 768)
#   ✅ Variance > 0

from corrected_extractor import CorrectedHiddenStateExtractor
import numpy as np

extractor = CorrectedHiddenStateExtractor('gpt2')

sample_prompt = (
    "Review: This film was absolutely wonderful.\n"
    "Sentiment: Positive\n\n"
    "Review: I hated every minute of it.\n"
    "Sentiment: Negative\n\n"
    "Review: The acting was superb.\n"
    "Sentiment: Positive\n\n"
    "Review: Boring and predictable plot.\n"
    "Sentiment: Negative\n\n"
    "Review: A masterpiece of modern cinema.\n"
    "Sentiment:"
)

# ── Test 1: tokenisation ──────────────────────────────────────────────────
input_ids = extractor.tokenizer.encode(sample_prompt)
tokens    = extractor.tokenizer.convert_ids_to_tokens(input_ids)
found     = extractor.get_label_positions_corrected(
                input_ids, ["Positive", "Negative"]
            )

print(f"\nTotal tokens : {len(input_ids)}")
print(f"Label hits   : {found}\n")

for pos, label in found:
    ctx = tokens[max(0, pos - 2): pos + 3]
    print(f"  '{label}' @ idx {pos} | context: {ctx}")

assert len(found) >= 4, (
    f"❌ Expected ≥4 label positions, got {len(found)}. "
    "Tokenisation is broken."
)
print("\n✅ TEST 1 PASSED — label positions found correctly")

# ── Test 2: hidden state extraction ──────────────────────────────────────
import numpy as np

target_idx = np.array([found[-1][0]], dtype=np.int32)
hs = extractor.extract(sample_prompt, target_idx)

print(f"\nHidden state shape : {hs.shape}")
print(f"Expected           : (13, 1, 768)")

assert hs.shape == (13, 1, 768), (
    f"❌ Wrong shape: {hs.shape}. Expected (13, 1, 768)."
)
print("✅ TEST 2 PASSED — hidden state shape correct")

# ── Test 3: variance ─────────────────────────────────────────────────────
variance = float(np.var(hs))
print(f"\nHidden state variance : {variance:.6f}")
assert variance > 1e-4, "❌ Variance near zero — something is wrong."
print("✅ TEST 3 PASSED — variance is non-zero")

print("\n" + "="*50)
print("ALL TESTS PASSED — safe to run main_reproduction.py")
print("="*50)
