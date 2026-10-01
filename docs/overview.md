# OpenTheory core (math)

OpenTheory formalizes pieces of deep learning theory in **Lean 4**. This directory is the math repository: Lean, the Massot blueprint, CI, and a local MCP server.

The web dashboard is a **different** GitHub repo. It fetches `blueprint/blueprint.json` from this repository (GitHub raw or Pages). Core does not expose an HTTP API.

---

## Architecture

```
OpenTheory/Data/             sample, spectral gap, spiked model
OpenTheory/Architectures/    activation, two-layer net, residual step, expressivity
OpenTheory/Parameterization/ µP and NTK scale factors
OpenTheory/Representation/   features, Gram matrix, alignment
OpenTheory/Optimization/     empirical risk, descent path, convergence rate
OpenTheory/Generalization/   Lipschitz bounds, operator norm, Rademacher
OpenTheory/Dynamics/         contraction, spectral bias, feature-learning conjecture
OpenTheory/EndToEnd.lean     population risk ≤ ε_approx + ε_opt + ε_gen
blueprint/src/               leanblueprint (web.tex, print.tex, content.tex)
scripts/                     axiom audit, JSON export, \lean name check
mcp/                         stdio MCP for local agents
home_page/                   Jekyll landing page for GitHub Pages
```

---

## Blueprint convention

Patrick Massot / `leanblueprint`:

- `\lean{Name}` — declaration in this library or Mathlib
- `\leanok` on a definition or statement — the statement is formalized
- `\leanok` inside `proof` — the proof is formalized (no `sorry`)
- `\uses{labels}` — dependency graph

CI refuses `\lean` names that are not declarations.

---

## Genesis tree

### Formalized definitions and elementary lemmas (proof `\leanok`)

- Data: `def:sample`, `def:spectral_gap`, `def:spiked_model`
- Architectures: `def:activation`, `def:two_layer_net`, `def:residual`, `def:approx_error`, `lem:approx_error_pointwise`
- Parameterization: `def:mup_scaling`, `def:ntk_scaling`
- Representation: `def:features`, `def:gram`, `def:alignment`, `lem:alignment_nonneg`
- Optimization: `def:empirical_risk`, `def:descent`
- Generalization: `lem:lipschitz_gelu`, `lem:bounded_mean`
- End-to-end: `def:end_to_end`, `lem:population_risk`

### Statements in Lean, proofs open (`sorry`)

- `lem:expressivity_subspace` — finite-sample approximation on the spike
- `lem:alignment_le_one`, `lem:alignment_certificate`
- `lem:risk_exponential` — Grönwall bound under a descent rate
- `lem:layer_lipschitz`, `lem:forward_lipschitz`
- `lem:rademacher_bound` — symmetrization scaffold
- `lem:gaussian_operator_norm` — Gaussian operator-norm tails
- `lem:spectral_bias_mup` — spectral bias under µP (bounty)
- `lem:representation_contraction` — residual contraction (bounty)
- `conj:grand_feature_learning` — global alignment (apex bounty)

---

## Local commands

```bash
lake exe cache get
lake build
lake env lean --run scripts/audit_axioms.lean
python3 scripts/check_lean_decls.py
python3 scripts/export_blueprint.py
```
