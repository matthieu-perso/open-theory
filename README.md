# OpenTheory Core

Lean&nbsp;4 library and **Massot blueprint** for machine-checked deep learning theory.

[![Lean 4](https://img.shields.io/badge/Lean_4-v4.34.0-blue.svg)](https://leanprover.github.io/)
[![Mathlib 4](https://img.shields.io/badge/Mathlib_4-pinned-brightgreen.svg)](https://github.com/leanprover-community/mathlib4)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-yellow.svg)](LICENSE)

This repository is the source of mathematical truth. The Next.js dashboard is a separate repo that only **reads** `blueprint/blueprint.json`. There is no API on this library. Agents that prove lemmas talk to Lean through the **local** MCP server in `mcp/`.

The Lean library follows the dependency order of the theory:

```text
OpenTheory/Data/             hypotheses on the sample and the spectrum
OpenTheory/Architectures/    networks and expressivity
OpenTheory/Parameterization/ µP and NTK: how width scales initialization and learning rates
OpenTheory/Representation/   features, Gram matrix, alignment
OpenTheory/Optimization/     empirical risk and descent
OpenTheory/Generalization/   Lipschitz bounds, operator norm, Rademacher
OpenTheory/Dynamics/         contraction, spectral bias, feature learning
OpenTheory/EndToEnd.lean     population risk ≤ ε_approx + ε_opt + ε_gen
```

---

## Dual track

| Track | Where | What “done” means |
| :--- | :--- | :--- |
| Informal | `blueprint/src/content.tex` | Human statement, citations, proof sketch |
| Formal | `OpenTheory/*.lean` | Kernel-checked declaration |
| Graph | plasTeX + `leanblueprint` | Green fill = `\leanok` **on the proof** |

`\leanok` on a lemma environment means the *statement* exists in Lean (it may still contain `sorry`). `\leanok` inside `\begin{proof}` means the proof has no `sorry`. Definitions use `\leanok` on the environment itself.

---

## Build the math

```bash
curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh
source ~/.profile
lake exe cache get
lake build
lake env lean --run scripts/audit_axioms.lean
python3 scripts/check_lean_decls.py
python3 scripts/export_blueprint.py blueprint/blueprint.json
```

Codespaces: this repo includes `.devcontainer/` (Lean + `leanblueprint`).

---

## Build the blueprint (HTML / PDF)

```bash
pip install leanblueprint
# graphviz + headers: see https://pygraphviz.github.io/documentation/stable/install.html
leanblueprint checkdecls
leanblueprint pdf
leanblueprint web
leanblueprint serve
```

GitHub Actions compiles the blueprint on `main` and deploys **GitHub Pages** (enable Pages → Source: GitHub Actions):

- <https://matthieu-perso.github.io/open-theory/>
- <https://matthieu-perso.github.io/open-theory/blueprint/>
- <https://matthieu-perso.github.io/open-theory/docs/>

Details: [`blueprint/README.md`](blueprint/README.md).

For a topic-by-topic map of relevant Mathlib modules, external Lean libraries,
version compatibility, and exact import patterns, see
[`docs/lean-libraries.md`](docs/lean-libraries.md).

For adoption priorities, implementation milestones, and their acceptance criteria,
see [`docs/reuse-plan.md`](docs/reuse-plan.md).

For the next learning guarantees, neural-architecture and interpretability tracks,
with a dependency DAG and reuse gates, see
[`docs/deep-learning-roadmap.md`](docs/deep-learning-roadmap.md).

For the first closed nonlinear MLP expected-error bound (one fixed input,
bounded finite label noise, first-layer-only training), see
[`docs/mlp-risk-guarantee.md`](docs/mlp-risk-guarantee.md). This is not yet a
multi-input or unseen-input generalization guarantee.

The implemented reuse slice uses pinned Mathlib and TorchLean: generic datasets
and population laws, finite-dimensional ridge regression with exact-GD bounds,
biased ReLU MLP/residual Lipschitz bounds, exact masked attention bounds,
masked/causal transformer inference, finite-second-moment population-risk envelopes,
and upstream SGD/momentum/Adam/AdamW state machines. Risk envelopes are not
generalization-gap or transformer-training guarantees. A narrow FoML subset now
supplies a conditional empirical-risk oracle bridge; its high-probability
generalization modules did not port directly. A narrow StatLean subset supplies
the first checked high-probability finite-class bound. Continuous neural-class
generalization remains the next statistical target.
Finite-model selection now has a proved empirical minimizer and an excess-risk
tail bound, including an approximate-selection variant; see
[`FiniteLearning.lean`](OpenTheory/Generalization/FiniteLearning.lean).
[`Covering.lean`](OpenTheory/Generalization/Covering.lean) extends that statistical
route to continuous classes with finite uniform loss covers, including an
approximate-learner excess-risk bound. Compactness supplies cover existence,
not a numerical cardinality estimate. Bounded-label ridge training now has a
sample-independent parameter-radius bound for every zero-initialized GD iterate.
[`TrainedRidge.lean`](OpenTheory/Generalization/TrainedRidge.lean) combines that
radius, a step legal for every input bound `B`, a grid cover of the training
ball, and the finite-class tail. The loss is squared error normalized by
`(2RB+T)²`. The excess budget pays the optimization gap and the comparator's
regularization. The grid bound is not sharp, and the result is linear ridge
regression, not a neural-network guarantee.
[`RidgeRisk.lean`](OpenTheory/Generalization/RidgeRisk.lean) is the informative
companion: with the design held fixed and white label noise of scale `σ`, the
ridge solution that exact gradient descent approaches has in-sample mean squared
error at most `(λ/4)‖θ*‖² + σ² df / m`. The degrees of freedom are at most the
rank of the design, hence at most `min(d, m)`, and a rank-`r` design replaces
`df` by `r`. A zero design has degrees of freedom zero even when the parameter
dimension is positive. The bias term is the regularization price and does not
shrink with the number of rows. This is the classical fixed-design
bias–variance shape, not a random-design generalization theorem and not a
neural-network theorem. It is the linear prototype of a kernel or NTK guarantee.
An MLP parameter-ball grid cover is exponential in the number of weights and
does not see this spectrum.
[`KernelRidge.lean`](OpenTheory/Generalization/KernelRidge.lean) puts the same
bound on the kernel Gram `K = c A A*`. Gradient descent still trains the primal
parameter, and its predictions equal the smoother `K(K+λI)⁻¹`. Equal kernels
give equal predictions, and the rank of `K` equals the rank of the feature map.
This is the classical primal–dual identity. A frozen feature map inherits it.
It does not prove that a network stays near its linearization.
[`FeatureStep.lean`](OpenTheory/Dynamics/FeatureStep.lean) is the one-step
comparison that separates those regimes for a two-layer network. On the loss
`(1/2)(f(x)-y)²`, one gradient step changes hidden preactivation `i` by
`-η₁ (f(x)-y) σ'(preᵢ) W₂ᵢ ‖x‖²`. With the readout at most its initialization
scale, the checked NTK rate moves that preactivation by `O(1/√m)` and the
checked μP rate moves it by `O(1/m)`. Both vanish as the width grows. The
first-layer rate `η₁ = m`, with a readout coordinate of size `1/m`, moves it by
a width-independent amount. This is the one-step calculation behind the
feature-learning exponent. It is not an infinite-width limit and not a
generalization bound.
[`TangentStep.lean`](OpenTheory/Dynamics/TangentStep.lean) compares that step
with its tangent update. If the activation derivative is `M`-Lipschitz, the
output gap is at most `M |f(x)-y|² L² ‖x‖⁴ / √m` for the checked NTK rate and
the same quantity over `m²` for the checked μP rate. The feature-learning rate
leaves a gap of order `|f(x)-y|²`.
[`PredictionGap.lean`](OpenTheory/Generalization/PredictionGap.lean) says a
predictor within `δ` of the ridge predictor has in-sample mean squared error
at most twice the ridge bound plus `2δ²/m`. That `δ` is not yet the tangent
gap: the tangent theorem compares the network with one tangent step, not with
the ridge solution.
See [dependency status](docs/dependency-status.md) for tested
imports and libraries still awaiting integration. The 11 legacy research targets
remain open; the axiom audit rejects `sorryAx` in other OpenTheory declarations.

---

## MCP (local agents only)

The server lives in **`mcp/`** next to Lake. Cursor / Claude spawn it over stdio. It is not hosted on Pages or Vercel.

```bash
cd mcp && npm install && npm run build
node dist/index.js
```

---

## Contribute

1. Pick an open node (proof not `\leanok`, often `sorry` in Lean).
2. Prove it. Do not add axioms.
3. Tag `\leanok` in the **proof** in `content.tex`.
4. Refresh `blueprint.json` and open a PR.

CI runs `lake build`, the axiom auditor (`sorryAx` allowed on open targets; anything else illegal), `\lean` name checks, and a stale-graph check on `blueprint.json`.
