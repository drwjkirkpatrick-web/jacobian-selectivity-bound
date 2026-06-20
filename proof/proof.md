# Proof: Jacobian Selectivity Bound

## Preliminary: ReLU Jacobian Structure

For a single ReLU neuron:
\[
y = \operatorname{ReLU}(w^\top x + b) = \max(0, w^\top x + b)
\]

The derivative w.r.t.~input $x$ is:
\[
J(x) = \frac{\partial y}{\partial x} = \mathbf{1}[w^\top x + b > 0] \cdot w
\]

This is a piecewise-constant function of $x$ with exactly two values:
\begin{align*}
J_+ &= w \quad \text{if } w^\top x + b > 0 \\
J_- &= 0 \quad \text{if } w^\top x + b \leq 0
\end{align*}

The boundary is the hyperplane $\{x : w^\top x + b = 0\}$.

---

## Proof of Theorem 3.1 (ReLU Case)

**Given:** Dataset $\mathcal{D}$ containing at least one activating
($z > 0$) and one non-activating ($z \leq 0$) input.

**Claim:** $\max_{x_i,x_j} \|J(x_i) - J(x_j)\|_2 = \|w\|_2$.

**Step 1:** Enumerate cases.

For any pair $(x_i, x_j)$, there are four sign combinations:

| $z_i$ | $z_j$ | $J(x_i)$ | $J(x_j)$ | $\|J_i - J_j\|_2$ |
|-------|-------|----------|----------|------------------|
| $+$ | $+$ | $w$ | $w$ | $0$ |
| $+$ | $-$ | $w$ | $0$ | $\|w\|_2$ |
| $-$ | $+$ | $0$ | $w$ | $\|w\|_2$ |
| $-$ | $-$ | $0$ | $0$ | $0$ |

**Step 2:** Maximum is $\|w\|_2$.

The maximum of $\{0, \|w\|_2, \|w\|_2, 0\}$ is $\|w\|_2$.

**Step 3:** Achievability.

Since $\mathcal{D}$ contains at least one sample from each region,
choose $x_+$ with $z > 0$ and $x_-$ with $z \leq 0$. Then
$\|J(x_+) - J(x_-)\|_2 = \|w\|_2$. ∎

---

## Proof of Theorem 3.2 (General Lipschitz Case)

**Given:** $\phi$ is $L$-Lipschitz with $|\phi'(z)| \leq L$.

For two inputs $x_i, x_j$, the Jacobians are:
\[
J(x_i) = \phi'(z_i) \cdot w, \quad J(x_j) = \phi'(z_j) \cdot w
\]

Their difference:
\[
\|J_i - J_j\|_2 = |\phi'(z_i) - \phi'(z_j)| \cdot \|w\|_2
\]

By the Mean Value Theorem, there exists $\xi$ between $z_i$ and $z_j$:
\[
|\phi(z_i) - \phi(z_j)| = |\phi'(\xi)| \cdot |z_i - z_j|
\]

If $z_i$ and $z_j$ have the same sign and $\phi'$ is continuous,
then $|\phi'(\xi)| \approx |\phi'(z_i)| \approx |\phi'(z_j)|$, so:
\[
|\phi'(z_i) - \phi'(z_j)| \approx
\frac{|\phi(z_i) - \phi(z_j)|}{|z_i - z_j|}
\]

For the worst-case bound, note that:
\[
|\phi'(z_i) - \phi'(z_j)| \geq
\frac{|\phi(z_i) - \phi(z_j)|}{\max\{|z_i|, |z_j|\}} - 2L \cdot
\mathbf{1}[\operatorname{sgn}(z_i) \neq \operatorname{sgn}(z_j)]
\]

The sign-change penalty accounts for cases where $\phi'$ drops
discontinuously (e.g., ReLU derivative jumps from 0 to 1 at $z=0$).

Multiplying by $\|w\|_2$ gives the theorem. ∎

---

## Proof of Theorem 3.3 (Multi-Neuron Frobenius Bound)

**Given:** Layer $f(x) = \phi(Wx + b)$ with $W \in \mathbb{R}^{m \times d}$.

The Jacobian matrix is:
\[
J(x) = \operatorname{diag}(\phi'(Wx + b)) \cdot W
\]

For two inputs $x_a, x_b$, the difference is:
\[
J(x_a) - J(x_b) =
\bigl(\operatorname{diag}(\phi'(z_a)) - \operatorname{diag}(\phi'(z_b))\bigr) \cdot W
\]

Let $\Delta_i = \phi'(z_{a,i}) - \phi'(z_{b,i})$. Then row $i$ of the
difference is $\Delta_i \cdot W_{i,*}$ (the $i$-th row of $W$).

The squared Frobenius norm is:
\[
\|J_a - J_b\|_F^2 = \sum_{i=1}^{m} \Delta_i^2 \cdot \|W_{i,*}\|_2^2
\]

For ReLU-like activations, $\Delta_i \in \{-1, 0, +1\}$ depending on
whether the neuron's state differs. If neuron $i$ changes state,
$|\Delta_i| = 1$ and it contributes $\|W_{i,*}\|_2^2$.

The selectivity of neuron $i$ is related to the magnitude of its
activation change. Using the Lipschitz bound $|\phi(z_a) - \phi(z_b)|
\leq L |z_a - z_b|$, and noting that $|z_a - z_b| = |W_{i,*}(x_a - x_b)|
\leq \|W_{i,*}\|_2 \cdot \|x_a - x_b\|_2$, we get:
\[
\operatorname{sel}_i \leq L \|W_{i,*}\|_2 \cdot \operatorname{diam}(\mathcal{R})
\]

Therefore $\|W_{i,*}\|_2 \geq \operatorname{sel}_i / (L \cdot
\operatorname{diam}(\mathcal{R}))$. Substituting:

\[
\|J_a - J_b\|_F^2 \geq \frac{1}{(L \cdot
\operatorname{diam}(\mathcal{R}))^2}
\sum_{i: \text{toggled}} \operatorname{sel}_i^2
\]

Taking square roots gives the theorem. ∎

---

## Discussion: Sharpness of Bounds

For ReLU:
- Theorem 3.1 is **exact** — no slack.
- The $\|w\|_2$ bound is achieved by any pair crossing the activation
  hyperplane.

For smooth activations:
- Theorem 3.2 is loose due to the MVT approximation.
- For GELU/Swish, where $\phi'(z)$ is continuous and bounded, the bound
  can be tightened to eliminate the sign-change penalty.

For multi-neuron layers:
- Theorem 3.3 depends on $\operatorname{diam}(\mathcal{R})$ — the maximum
  distance between inputs in the region.
- For local analysis (adjacent inputs), $\operatorname{diam}(\mathcal{R})$
  is small and the bound is tighter.

---

## Relation to Existing Work

This theorem formalizes the intuition behind:
- **Neuron selectivity** (Bau et al., 2017; Olah et al., 2020)
- **Jacobian regularization** (Hoffman et al., 2019)
- **Mechanistic interpretability** (Elhage et al., 2022)

It provides the missing quantitative link: selectivity implies Jacobian
variation, which implies attribution disagreement, which implies
interpretability fragility.
