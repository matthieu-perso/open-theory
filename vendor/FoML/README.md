# FoML deterministic ERM subset

Upstream: https://github.com/auto-res/lean-rademacher
Revision: `db9f18714b0b9934553c37db1b41250462dc7faa` (Lean 4.32.0).
License: MIT; original license preserved in `LICENSE`.

Only the four-module import closure of `FoML.Learning.ERM` is included.
These files are unmodified upstream sources, built by the root Lake library
against pinned Lean/Mathlib 4.34.0. No ignored discovery clone is required.
The OpenTheory axiom audit follows the imported oracle theorem transitively.

The larger `FoML.Generalization.Learning` compatibility probe failed in
`ForMathlib/MeasureTheory/Measure/Real`, `Rademacher/Symmetrization`, and
`Probability/McDiarmid` (set-membership simplification, changed Fin equivalence,
removed algebra lemma, and integral/measure API elaboration). It is not included.
No confidence theorem or statistical-rate integration is claimed by this subset.
