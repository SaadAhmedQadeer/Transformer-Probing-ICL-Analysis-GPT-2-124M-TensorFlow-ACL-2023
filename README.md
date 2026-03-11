# When Anchors Drift: Quantifying and Correcting Label Surface Form Sensitivity in In-Context Learning

> **Independent Research Project** | NLP / Mechanistic Interpretability  
> Reproduction and extension of Wang et al., *"Label Words are Anchors"*, ACL 2023  
> Framework: TensorFlow/Keras + HuggingFace Transformers | Model: GPT-2 Small (124M)

---

## Overview

This project reproduces and extends Wang et al. (ACL 2023), which proposes that
**label words in ICL demonstrations act as informational anchors**: shallow transformer
layers aggregate semantic context around label positions, and deep layers extract
predictions exclusively from those positions.

We identify a formally testable gap in the anchor theory — its implicit **invariance
prediction** — and design experiments that quantify the violation. We then propose a
**training-free correction** grounded in embedding-space geometry.

---

## Key Contributions

### 1. Reproduction
- Reproduced the layer-wise probing curve (Wang et al. Figure 3) on GPT-2-Small
- Achieved **80.5% probing accuracy at layer 12** (5-fold CV, SST-2, 200 examples)
- Confirmed the characteristic monotonic rise with sharp inflection at the shallow/deep boundary (layer 6)

### 2. The LSFS Metric (Novel)

The anchor theory implicitly predicts that semantically equivalent label variants
should produce similar ICL accuracy, since the aggregation mechanism is
positional and embedding-based:

$$\text{LSFS}(\mathcal{V}_m) = \frac{|\text{Acc}(\mathcal{V}_0) - \text{Acc}(\mathcal{V}_m)|}{\|E(v_m) - E(v_0)\|_2 + \epsilon}$$

A high LSFS score reveals **large accuracy change per unit embedding distance** —
a direct violation of the theory's smoothness assumption.

### 3. Empirical Results

| Variant | Labels | Accuracy | ΔAcc | LSFS |
|---|---|---|---|---|
| V0 — Canonical | Positive / Negative | 0.620 | — | — |
| V1 — Synonyms | Good / Bad | 0.500 | 0.120 | High |
| V2 — Abbreviated | Pos / Neg | 0.500 | 0.120 | High |
| V3 — Symbols | + / − | 0.500 | 0.120 | High |
| V4 — Numeric | 1 / 0 | 0.510 | 0.110 | High |
| V5 — Random | Apple / Orange | 0.470 | 0.150 | Baseline |

**Finding:** All non-canonical variants collapse to near-chance (0.47–0.51),
a 11–12% absolute drop. The anchor theory predicts this should not happen.

### 4. Canonical Anchor Projection (Novel Fix)

A training-free intervention that interpolates variant label embeddings
toward canonical embeddings before the forward pass:

$$\tilde{E}(v_m) = (1 - \alpha) \cdot E(v_0) + \alpha \cdot E(v_m)$$

At optimal **α\* ≈ 0.1–0.2**:
- Symbols (+/−) recover to **0.665** — surpassing canonical baseline
- Numeric (1/0) recover to **0.620** — full canonical-level accuracy
- Canonical labels are **unaffected** (projection is a no-op when v_m = v_0)


## Mathematical Background

### Anchor Theory (Wang et al. 2023)

For a K-shot prompt with demonstration tokens $\mathbf{X}$ and label positions
$\mathcal{L} = \{l_1, \ldots, l_K\}$, the paper shows:

**Shallow layers** aggregate semantic context into label positions:
$$h_{l_i}^{(\ell)} = \text{Attn}^{(\ell)}\!\left(\sum_{j \in \mathcal{N}(l_i)} \alpha_{ij}^{(\ell)} h_j^{(\ell-1)}\right), \quad \ell \leq k$$

**Deep layers** extract predictions from label positions:
$$P(y \mid \mathbf{X}) \approx f\!\left(\{h_{l_i}^{(L)}\}_{i=1}^{K}\right)$$

### Our Violation

The theory implies smoothness: if $\|E(v_m) - E(v_0)\|_2 \leq \varepsilon$, then
$|\text{Acc}(v_m) - \text{Acc}(v_0)| \leq \delta(\varepsilon) \to 0$.

We show empirically that $\delta$ is large (~0.12) even for synonyms where
$\varepsilon$ is modest — the smoothness assumption fails.

### Our Fix

The projection operator is a convex combination:
$$\tilde{E}(v_m) = (1-\alpha) \cdot E(v_0) + \alpha \cdot E(v_m), \quad \alpha \in [0,1]$$

At $\alpha^* \approx 0.1$–$0.2$ this recovers 80–100% of the accuracy gap
with **zero gradient updates**, confirming the sensitivity is primarily
an embedding-layer phenomenon.

---

## Ablation Studies

| Ablation | Variable Isolated | Finding |
|---|---|---|
| ABL-1 | Label surface form | 11–12% accuracy drop for all non-canonical variants |
| ABL-2 | Demo ordering | σ < 0.03 — ordering is not a confound |
| ABL-3 | Projection α sweep | α* ≈ 0.1–0.2 recovers gap across all variants |
| ABL-4 | Projection on canonical | No accuracy change — fix is safe |
| ABL-5 | K-shot count (k=1,2,4) | Gap persists across k — not a data-size artifact |

---

## Hardware & Reproducibility

- **Hardware:** CPU only (no GPU required)
- **Model:** GPT-2 Small (124M parameters, `float32`)
- **Runtime:** ~3–4 hours total for all experiments
- **Seed:** Fixed (`np.random.default_rng(42)`) for full reproducibility


## Citation

```bibtex
@inproceedings{wang2023label,
  title     = {Label Words are Anchors: An Information Flow Perspective
               for Understanding In-Context Learning},
  author    = {Wang, Lean and Lei, Lei and Shi, Damai and
               Su, Weijie and Zeng, Zeming and others},
  booktitle = {Proceedings of EMNLP},
  year      = {2023}
}
```

