# OpenTheory Core

Lean&nbsp;4 library and **Massot blueprint** for machine-checked deep learning theory.

[![Lean 4](https://img.shields.io/badge/Lean_4-v4.11.0-blue.svg)](https://leanprover.github.io/)
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
