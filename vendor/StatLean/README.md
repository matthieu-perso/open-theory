# StatLean finite-class PAC subset

Upstream: https://github.com/StatLean/Stat-Lean
Revision: `d41b0bb4c68a36721b8bf8ad191af47a44670b9c` (Lean 4.29.1).
License: Apache-2.0; original license preserved in `LICENSE`.

This directory contains the eight-module import closure of
`StatLean.StatisticalLearning.FiniteClass.AgnosticPAC`. The root Lake library
builds it against pinned Lean/Mathlib 4.34.0.

Compatibility change:

- `FiniteClass/UniformConvergence.lean`: unfold the local `μ` abbreviation when
  transporting coordinate independence; Lean 4.34 no longer closes that
  definitional equality via the upstream `simpa [Function.comp]` invocation.

The mathematical statement and proof structure are unchanged. The OpenTheory
axiom audit follows every imported declaration used by local bridges.
