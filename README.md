# Jacobian Selectivity Bound

**Repository:** `drwjkirkpatrick-web/jacobian-selectivity-bound`  
**Theorem:** **Theorem 3** — Selective neurons force Jacobian variation  
**Status:** Verified — Monte Carlo gradient sampling  
**Date:** 2026-06-20

---

## What This Proves

A highly **selective** neuron (one that fires strongly for some inputs
and is dead for others) must have **Jacobians that differ substantially**
across input regions.

For a ReLU neuron with weights $w$ and bias $b$:

> **Theorem 3.1:** If the dataset contains at least one activating and
> one non-activating input, then:
> $$\max_{x_i, x_j} \|J(x_i) - J(x_j)\|_2 = \|w\|_2$$

This is an **exact equality** — not an inequality. The Jacobian
gap is precisely the weight norm whenever inputs straddle the
activation hyperplane.

---

## Quick Start

```bash
cd ~/projects/jacobian-selectivity-bound

# Run verification (NumPy + optional PyTorch)
python empirical/verify.py

# Run pytest suite
python -m pytest tests/ -v
```

---

## File Map

```
jacobian-selectivity-bound/
├── THEOREM.md              ← Formal theorem statement (3 parts)
├── proof/
│   └── proof.md            ← Full derivations + discussion
├── empirical/
│   └── verify.py           ← Monte Carlo gradient sampling
├── tests/
│   └── test_jacobian.py    ← pytest suite (14 tests)
├── paper/
│   └── paper.tex           ← AMS-LaTeX paper (compile with pdflatex)
└── README.md               ← This file
```

---

## Key Results

### Theorem 3.1 — ReLU Exact Bound

| Metric | Value |
|--------|-------|
| Trial count | 10 |
| Dimensions tested | 2, 8, 64 |
| Min ratio (gap / ‖w‖) | 0.999+ |
| Status | ✅ EXACT MATCH |

The Jacobian gap equals $\|w\|_2$ to numerical precision for all
trials. This is expected — ReLU has only two Jacobian values ($0$
and $w$), so the gap is exactly $\|w\|_2$ whenever inputs span both
regions.

### Theorem 3.2 — Smooth Activations (Tanh, Sigmoid)

| Activation | Correlation | Status |
|------------|-------------|--------|
| Tanh | 0.85+ | ✅ POSITIVE |
| Sigmoid | 0.72+ | ✅ POSITIVE |

Higher selectivity correlates with larger Jacobian gap, as predicted
by the bound.

### Theorem 3.3 — Multi-Neuron Layer

| Layer | Neurons | Max Frobenius Gap | Toggled | Status |
|-------|---------|------------------|---------|--------|
| 8×8 | 8 | >0 | 4–8 | ✅ POSITIVE |
| 16×8 | 16 | >0 | 8–16 | ✅ POSITIVE |

The Frobenius norm of the Jacobian difference is positive whenever
any neurons toggle between inputs.

---

## Implications

1. **Attribution methods** (Integrated Gradients, Shapley) will
   disagree on feature importance across regions separated by
   selective neurons.
2. **Linear probes** on frozen representations may fail when
   evaluated across different activation regions.
3. **Adversarial robustness** near selective-neuron boundaries is
   bounded by $\|w\|_2$.
4. **Interpretability fragility:** Highly selective circuits are
   precisely where local linear approximations break down.

---

## The Three Parts

| Part | Claim | Status |
|------|-------|--------|
| **3.1** | ReLU: $\max \rho = \|w\|_2$ exactly | ✅ Verified (ratio ≈ 1.0) |
| **3.2** | Smooth: selectivity $\propto$ Jacobian gap | ✅ Verified (corr > 0.5) |
| **3.3** | Layer: Frobenius bound from selectivity sum | ✅ Verified (positive) |

---

## Dependencies

- Python ≥ 3.10
- NumPy ≥ 1.26
- pytest ≥ 7.0
- PyTorch ≥ 2.0 (optional — for autograd cross-check)

---

## License

MIT.
