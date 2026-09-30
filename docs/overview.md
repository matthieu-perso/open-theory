# OpenTheory core (math)

OpenTheory formalizes pieces of deep learning theory in **Lean 4**. This directory is the math repository: Lean, the Massot blueprint, CI, and a local MCP server.

The web dashboard is a **different** GitHub repo. It fetches `blueprint/blueprint.json` from this repository (GitHub raw or Pages). Core does not expose an HTTP API.

---

## Architecture

```
OpenTheory/*.lean     kernel-checked library
blueprint/src/        leanblueprint (web.tex, print.tex, content.tex)
scripts/              axiom audit, JSON export, \lean name check
mcp/                  stdio MCP for local agents
home_page/            Jekyll landing page for GitHub Pages
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

- `def:activation`, `def:two_layer_net`, `def:mup_scaling`
- `lem:lipschitz_gelu` — Lipschitz field of an activation
- `lem:bounded_mean` — empirical means of 1-bounded sequences

### Statements in Lean, proofs open (`sorry`)

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
