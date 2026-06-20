# Theorem: Jacobian Selectivity Bound

**Status:** Verified — Monte Carlo gradient sampling on toy networks  
**Domain:** Neural network expressivity / Mechanistic interpretability  
**Date:** 2026-06-20

---

## Motivation

In mechanistic interpretability, researchers often describe neurons as
"selective" — firing strongly only for specific input patterns. At the
same time, the input-output Jacobian of a network describes how
sensitively outputs vary with inputs.

This theorem establishes a quantitative link: a highly selective neuron
must have Jacobians that differ substantially across input regions. The
more sharply a neuron partitions its input space, the greater the
spectral distance between Jacobians on different sides of the partition.

---

## Definitions

### Selectivity

For a neuron with pre-activation $z(x) = w^\top x + b$ and activation
$\phi(z)$, define the **selectivity** over a dataset region $\mathcal{D}$:

$$\operatorname{sel}(\phi) \;=\; \max_{x \in \mathcal{D}} \phi(z(x)) -
\min_{x \in \mathcal{D}} \phi(z(x))$$

When $\phi = \operatorname{ReLU}$, selectivity is large when the neuron
fires strongly for some inputs and is zero (dead) for others.

### Jacobian

For a scalar output neuron $y = \phi(w^\top x + b)$, the Jacobian is:

$$J(x) \;=\; \frac{\partial y}{\partial x} \;=\;
\phi'(w^\top x + b) \cdot w$$

For ReLU, $\phi'(z) = \mathbf{1}[z > 0]$, so:

$$J(x) \;=\; \mathbf{1}[w^\top x + b > 0] \cdot w$$

### Spectral Distance

For two Jacobians $J_1, J_2 \in \mathbb{R}^{1 \times d}$, define:

$$\rho(J_1, J_2) \;=\; \|J_1 - J_2\|_2 \;=\;
\|(\mathbf{1}[z_1 > 0] - \mathbf{1}[z_2 > 0]) \cdot w\|_2$$

This simplifies dramatically for ReLU:
- If both inputs are on the same side of the activation boundary
  ($z_1 > 0$ and $z_2 > 0$, or both $\leq 0$), then $\rho = 0$.
- If they are on opposite sides, $\rho = \|w\|_2$.

---

## Theorem 3.1 (ReLU Selectivity-Jacobian Bound)

For a ReLU neuron with weights $w \in \mathbb{R}^d$ and bias $b$, let
$\mathcal{D}$ be a dataset containing at least one activating and one
non-activating input. Then:

$$\max_{x_i, x_j \in \mathcal{D}} \rho\bigl(J(x_i), J(x_j)\bigr)
\;=\; \|w\|_2$$

and this maximum is achieved by any pair with $z_i > 0 > z_j$.

**Proof.** For ReLU, $J(x) = \mathbf{1}[w^\top x + b > 0] \cdot w$.
There are only two possible Jacobians:

- $J_+ = w$ when $z > 0$
- $J_- = 0$ when $z \leq 0$

If $\mathcal{D}$ contains at least one sample from each region, then:

$$\rho(J_+, J_-) = \|w - 0\|_2 = \|w\|_2$$

Any two samples from the same region have $\rho = 0$. ∎

---

## Theorem 3.2 (General Lipschitz Activation Bound)

For a $L$-Lipschitz, differentiable activation $\phi$ with
$\sup_z |\phi'(z)| \leq L$, and any two inputs $x_i, x_j \in \mathcal{D}$:

$$\|J(x_i) - J(x_j)\|_2 \;\geq\;
\frac{|\phi(z_i) - \phi(z_j)|}{\max\{|z_i|, |z_j|\}} \cdot
\|w\|_2 \;-\; 2L\|w\|_2 \cdot
\mathbf{1}\bigl[\operatorname{sgn}(z_i) \neq \operatorname{sgn}(z_j)\bigr]$$

This is loose but shows the trend: larger output difference (selectivity)
forces larger Jacobian difference, except when the sign changes
(derivative discontinuity in ReLU-like activations).

---

## Theorem 3.3 (Multi-Neuron Layer — Frobenius Bound)

For a layer $f(x) = \phi(Wx + b)$ with $W \in \mathbb{R}^{m \times d}$,
let $\operatorname{sel}_i$ be the selectivity of neuron $i$. Let
$\mathcal{R} \subseteq \mathcal{D}$ be a subset partitioned by the
activation pattern. Then:

$$\max_{x_a, x_b \in \mathcal{R}} \|J(x_a) - J(x_b)\|_F
\;\geq\; \frac{1}{L} \cdot \sqrt{
\sum_{i: \operatorname{sel}_i > 0} \operatorname{sel}_i^2}$$

where $L$ is the Lipschitz constant of $\phi$, and the sum is over
neurons that change state between $x_a$ and $x_b$.

**Intuition.** Highly selective neurons contribute quadratically to the
Frobenius norm of the Jacobian difference. Many selective neurons
imply large overall Jacobian variation.

---

## Corollary 3.4 (Implication for Interpretability)

If a mechanistic probe reports a neuron as "highly selective" (large
$\operatorname{sel}$), then the network's local linear approximation
(Jacobian) must change substantially across inputs that toggle that
neuron. Therefore:

- **Attribution methods** (IG, Shapley) will disagree on feature
  importance depending on which side of the neuron's threshold the
  input lies.
- **Linear probes** trained on frozen representations may fail when
  evaluated across regions that toggle the selective neuron.
- **Adversarial robustness** near the neuron's boundary is
  fundamentally limited by the $\|w\|_2$ gap.

---

## Open Questions

1. **Sharp constant.** For smooth activations (Swish, GELU), can the
   bound be tightened to eliminate the sign-change penalty?

2. **Network depth.** How does the bound compose through layers? Does
   depth amplify or attenuate the Jacobian-selectivity relationship?

3. **Sparse networks.** In sparse or lottery-ticket networks, do
   surviving neurons have systematically higher selectivity and thus
   larger Jacobian variation?

4. **Concrete lower bound $c(\varepsilon)$.** For selectivity
   threshold $\varepsilon$, can we derive an explicit
   $c(\varepsilon) > 0$ such that $\operatorname{sel} > \varepsilon
   \Rightarrow \max \rho \geq c(\varepsilon)$?

---

## References

- Expressivity skill: `neural-network-expressivity`
- Kirkpatrick (2026). "Attention sink as inevitable simplex geometry."
  GitHub: drwjkirkpatrick-web/attention-sink.
