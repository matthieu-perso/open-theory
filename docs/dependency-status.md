# Dependency integration status

This records inspected source revisions, not a claim that every repository is
compatible or that every theorem in it satisfies our axiom policy.

## Common environment

Lean `v4.34.0`, Mathlib `5ed2965256430c3649e86755f9576b54eca72435`.
The root configuration and Lake manifest pin the actual dependency graph.
The existing OpenTheory library builds on this pair after replacing the removed
`BigOperators.Group.Finset` import and explicitly importing `Fintype.Fin`.

| Repository | Inspected commit | Declared Lean | Integration |
|---|---|---|---|
| TorchLean | `2d148d85839d97b02b06e8e02be61e3d57be3ad7` | 4.34.0 | Pinned; affine, Lipschitz, GD and optimizer modules used in checked adapters |
| gradient-descent-lean | `9c43286f4171bf39dd13abdcc2198ce4aaeaa0f7` | 4.28.0 | Probe failed in `Smooth.lean`: gradient/Fréchet simp API changes; not a root dependency |
| Optlib | `5da27c5f95aa6a8a45b8c14b968ade4c13ff18c3` | 4.13.0 | Source inspected; not built on the common toolchain |
| FoML / lean-rademacher | `db9f18714b0b9934553c37db1b41250462dc7faa` | 4.32.0 | Four unmodified deterministic-ERM modules vendored; confidence-theorem compatibility probe failed |
| StatLean | `d41b0bb4c68a36721b8bf8ad191af47a44670b9c` | 4.29.1 | Eight-module finite-class PAC subset vendored; one definitional compatibility patch |
| Arlib | `6262f60ad7556be62e94e87f6b5811cf299e7d94` | 4.33.0 | Source inspected; compatibility pending |
| ProbabilityApproximation | `a2fc5bda874d9b69186c0ba165bf79f3b9897662` | 4.32.0 | Source inspected; compatibility pending |
| ArtificialTheorems | `0cc7308db111442198cdb6b5ee09be28fb6b56a3` | 4.27.0 | Source inspected; SafeVerify/lean4checker dependencies need evaluation |
| Neural Network Proofs | `f90942517be8b66dd34574212ada69b2130a48e5` | 4.32.0-rc1 | Source inspected; architecture/hypothesis transfer pending |
| SciLean | `95f8119a2884e9c41f82136523bd5568ea7075c5` | 4.28.0-rc1 | Source inspected; LeanBLAS/native integration separate |
| TensorLib | `90be53e35b6e0d6c6c6147761c035af36d71f09c` | 4.33.0 | Source inspected; alternative tensor representation |
| Tyr | `ee7f080dda906475ba2c888814b2771f2793bc92` | 4.29.0 | Source inspected; LibTorch/runtime integration separate |
| lean4-mlir | `3336f4a78663ac9a4ce5914309a2b739af1dc6e9` | 4.34.0 | Source inspected; compiler integration pending |

Local discovery clones are under ignored `.lake/upstream/`; they are not
dependencies and are not required to reproduce the root build. The survey's
older moving-branch version observations should not override this pinned record.

## Trust and acceptance

The FoML subset is recorded in `vendor/FoML/README.md`, with its MIT license.
`Generalization/StatisticalBridge.lean` connects empirical/population risks,
proves bounded-loss integrability, and reuses the deterministic approximate-ERM
oracle inequality. Its optimization and uniform-deviation premises are explicit.
The larger FoML confidence-theorem probe fails in set simplification,
symmetrization and McDiarmid APIs on 4.34; no FoML high-probability theorem is
integrated.

The alternative narrow StatLean path is recorded in `vendor/StatLean/README.md`
with its Apache-2.0 license. It reuses the project's sub-Gaussian Hoeffding,
finite-union, uniform-convergence and agnostic-PAC proofs. One compatibility
change unfolds a local measure abbreviation and function composition; the
mathematical statement is unchanged. `Population.finiteClass_deviation_le`
connects that result to OpenTheory's supervised empirical/population risks.

TorchLean's `docs/TRUST_BOUNDARIES.md` distinguishes exact-real specifications,
logical floating-point models, executable replacements, and native runtimes.
Its `NN.MLTheory.Optimization.StronglyConvexGD` and
`NN.Proofs.Analysis.Lipschitz.Network` also build on the common environment.
The local `Torch.legacy_correspondence`, `Torch.MLP.lipschitz`,
`Torch.MLP.residual_lipschitz`, `Ridge.training`, and `Optimizer.sgd_eq_step`
provide substantive adapter proofs. The root axiom audit checks their transitive
dependencies, including imported proofs.

Attention additionally reuses `NN.Proofs.Autograd.FDeriv.HardMaskedSoftmax`
(coordinate formula and positive denominator),
`NN.Proofs.Models.Attention.CausalMask` (blocked/future-key zero weights), and
`NN.Spec.Models.Transformer` (multi-head attention, feedforward, encoder semantics).
The local wrapper adds configurable positive normalization epsilon and masks;
default-epsilon unmasked inference agrees with the upstream encoder. Exact-real
hard masking has an explicit zero result for all-blocked rows, not a finite
negative sentinel. Convexity and integration bounds reuse Mathlib. No runtime,
floating-point refinement, full-transformer Lipschitz, or SGD convergence theorem
is asserted by these imports.

Optimizer conventions at this pin: momentum uses `b ← momentum*b + g`,
then `p ← p - learningRate*b`; Adam advances the step counter before bias
correction and adds epsilon after the square root; AdamW applies decoupled
weight decay to parameters and passes the raw gradient to Adam. Square root
uses a nonnegative clamp in the upstream tensor primitive. These total
definitions do not enforce `0 ≤ beta < 1`, positive epsilon, nonnegative second
moments, or unbiased gradient estimates. Such hypotheses and their preservation
must be supplied for future numerical/stochastic guarantees. The imported
zero-weight-decay AdamW/Adam equality is an algebraic result, not convergence.

`gradient-descent-lean` exposes `IsLSmooth`, `IsStronglyConvex`, `IsMinimiser`,
`gdSeq`, and `gd_strongly_convex_convergence`. The last theorem additionally
requires stationarity of the minimizer, positive smoothness/convexity constants,
and a positive step at most `1/L`. A ridge adapter must prove those premises.
Its file headers say Apache 2.0 while its repository LICENSE says MIT; preserve
provenance and resolve this inconsistency before distributing a modified port.

Adding every repository to one Lake environment does not isolate their different
Lean and Mathlib versions. Pending entries are deliberately not represented as
working integrations. The next adoption steps are narrow imports, substantive
theorem instantiations, and transitive axiom checks—not umbrella imports alone.
