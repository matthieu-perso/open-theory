# OpenTheory: initial implementation and reuse plan

Prepared 2026-10-12. This is the implementation roadmap; proposed modules and
milestones below are not claims of completed formalization. The companion
[library survey](lean-libraries.md) provides a broader inventory.

## Implementation checkpoint (2026-10-12)

The next-work recommendation, dependency DAG, acceptance criteria and longer-term
deep-learning tracks are in [deep-learning-roadmap.md](deep-learning-roadmap.md).
That roadmap extends this initial plan; it does not change the completed status below.

The original plan and baseline below are retained as historical context.
Current dependency pins and probe failures are in [dependency status](dependency-status.md).

- **M1 core migration:** Lean/Mathlib 4.34, pinned TorchLean, full-library build,
  strict axiom policy for new declarations, Lean-environment blueprint name
  resolution, and non-mutating freshness checks are implemented. HTML/PDF
  blueprint rendering has not been tested in this checkpoint.
- **M2 deterministic foundation:** `Data/Foundation.lean` separates observations
  from boundedness; defines Euclidean parameters, empirical/population risk,
  explicit integrability, and IID via Mathlib independence and common laws.
  The old sample has a proved bounded adapter. Second-moment applications now
  use Mathlib integrability directly; sub-Gaussian applications remain pending.
- **M3 training example:** `Optimization/FiniteRidge.lean` proves the requested
  finite-data objective identity and instantiates a general finite-dimensional
  ridge theorem. Gradient, stationary-point existence, minimization, gradient
  Lipschitzness, and strong monotonicity are proved rather than assumed.
  `Ridge.admissible_step` supplies a positive admissible step.
- **M4 neural subset:** `Architectures/Torch.lean` reuses biased TorchLean affine
  layers and ReLU, proves MLP/residual global Lipschitz bounds, and proves exact
  correspondence with the old bias-free network using its original activation.
  The constants use a Frobenius upper bound, not the exact spectral norm.
  `Architectures/Attention.lean` now proves exact hard-mask normalization,
  blocked-key zero weights, convex-hull and norm bounds, fixed-weight value
  stability, and a coordinate bound for the actual upstream attention output.
  All-blocked rows return zero; normalization requires an allowed key, supplied
  by the diagonal for causal masks. `Architectures/Transformer.lean` provides
  masked/causal post-norm inference using upstream multi-head attention, feedforward
  and layer normalization, with positive dimensions/epsilon and head divisibility.
  Its unmasked default-epsilon form agrees with the upstream encoder.
  This completes the planned basic attention extension, not full transformer
  stability, differentiation, or training convergence; CNN adapters remain pending.
- **M5 catalog subset:** `Optimization/Optimizers.lean` imports upstream
  SGD/momentum/Adam/AdamW state machines and checks the SGD/GD correspondence
  and Adam's first-step index. No stochastic or Adam convergence claim is made.
- **Risk-envelope extension:** `Generalization/PopulationBounds.lean` proves
  integrability and an expected squared-error bound for any Lipschitz predictor
  on arbitrary measurable observations with finite input/label second moments.
  `Generalization/AttentionRisk.lean` bounds row risk by `(B + T)^2` for bounded
  values/labels, even with data-dependent scores and masks. Both require measurable
  loss explicitly. These are risk envelopes, not generalization-gap bounds or
  completion of M6. They make no boundedness claim for arbitrary unbounded data.
- **Statistical bridge, preliminary:** four unchanged FoML modules now supply
  a deterministic approximate-ERM oracle inequality through our risk adapters.
  Bounded-loss integrability is proved separately. The high-probability path
  failed the 4.34 probe in symmetrization/McDiarmid and a measure helper; its
  integration and concrete trained-model application remain unfinished.
- **Finite-class confidence bound:** a narrowly vendored StatLean subset proves
  that `[0,1]`-valued losses over a fixed finite class have uniform-deviation
  tail at most `2 |H| exp (-2 n ε²)`. A one-line definitional compatibility
  patch is documented. This is the first checked sample-complexity guarantee,
  but it does not yet cover the continuous ridge or neural parameter classes.
- **Ridge objective convergence:** the existing distance convergence now yields
  an explicit geometric bound on the regularized objective gap of the actual
  gradient-descent iterates. This discharges the deterministic optimization
  component needed by the next trained-predictor theorem.

The moment bound is
`E[‖f(X) - Y‖²] ≤ 3 * (K² * E[‖X‖²] + ‖f(0)‖² + E[‖Y‖²])`.
Raw observations may be images, text, graphs, or time series: the theorem takes
an explicit encoding into a normed input space and a label map. Their required
moments and loss measurability must be established for each application. No IID
assumption is needed for this expectation bound, but that does not supply a
concentration result for dependent samples.

The GD provider changed after testing: `gradient-descent-lean` fails its
`Smooth.lean` compatibility probe on 4.34; TorchLean's corresponding theorem
builds. Consequently our bound uses
`q = 1 - 2*η*λ + η²*L²`, with `η > 0` and `η*L² < 2*λ`, where
`L = ‖c A* A + λ I‖ + |λ|` and `c = 1/n`. This is a conservative valid
step-size regime, not the sharper `η ≤ 1/L` originally proposed.

All 11 legacy open targets remain open. The stricter audit permits `sorryAx`
only for their exact names and checks transitive dependencies for every other
safe OpenTheory declaration. Test-only fake axioms and `sorry` fixtures are not
imported by the library.

Next: instantiate these envelopes for concrete encoded datasets and MLPs;
prove bounded-domain stability when attention scores also vary; then integrate
a bounded-loss generalization theorem and stochastic optimizer hypotheses. Libraries listed
as pending in the dependency record have **not** been incorporated into the
root proof environment. This checkpoint does not complete all of M0–M5.

2026-10-13: finite model selection now constructs an empirical minimizer and
proves its excess-risk tail, with an approximate-selection variant. This completes
the finite-model learning example; continuous ridge/MLP guarantees remain pending.

Validation: the latest full root build passes (4,410 jobs). The strict audit
resolves all 86 blueprint references and finds `sorryAx` only in the 11 named
legacy targets. The updated finite-class/selector regression and four Python tests pass;
the preceding checkpoint's data/ridge/architecture/attention/transformer/risk/
optimizer and audit tests remain wired into CI. The regenerated feed contains
41 nodes and passes its freshness check.

## Objective and first deliverable

Build a reusable Lean library connecting data assumptions, model semantics,
losses, and optimizer updates to mathematical guarantees. Prefer an existing
definition or theorem plus a small adapter over a second implementation.

The first complete example will be finite-data ridge regression with exact
gradient descent and a convergence theorem. It exercises the whole interface
with assumptions we can actually discharge. The next example will be a biased
two-layer MLP with a proved forward-map correspondence and an input Lipschitz
bound. This establishes a route to residual networks and attention without
making nonconvex training convergence a prerequisite for the initial release.

The initial release should let a user import a model, supply data and explicit
hypotheses, instantiate an optimizer, and obtain a theorem about the resulting
mathematical model. Executable training and numerical refinement are a later,
separately tested integration.

## Current baseline and findings

Inspected working tree based on commit
`b653947752d2344ffc7f8f7f47fdbc6e18827a62`, with existing documentation edits
and `ImportSmoke.lean` present. Those pre-existing changes are preserved.

- Lean and Mathlib are pinned to `v4.11.0`; Mathlib is the only direct dependency.
- `lake build` passes with 11 declarations emitting `sorry` warnings.
- `python3 scripts/check_lean_decls.py` passes for 48 blueprint references.
- Blueprint export produces 31 nodes and matches the existing JSON after
  excluding its generation timestamp.
- The axiom script initially failed to compile: `Lean.importModules` requires
  `Array Import`, but the call supplied a list. The initial maintenance change
  replaces `[...]` with `#[...]`. The audit then exposed false positives from
  unsafe compiler intermediates; it now audits safe declarations using Lean's
  safety metadata, preserving the existing allowed-axiom list. Unsafe runtime
  code is outside this kernel-proof audit.
- After repair, the audit passes: 11 safe declarations depend on `sorryAx`,
  with no unauthorized axioms reported. `lake build audit_axioms` also passes.
  A temporary regression check confirms safety metadata distinguishes the
  roots and the collector still detects a forbidden axiom transitively; the
  forbidden test axiom is not part of the repository or its imports.

The following issues determine the order of work:

| Existing area | Finding | Planned treatment |
|---|---|---|
| `Data/Sample.lean` | Observations and unit-boundedness assumptions are bundled | Add generic observations and separate predicates; provide an adapter from `Sample` |
| `Data/Model.lean`, `Data/Spectrum.lean` | Spike, noise, and gap are not connected to a distribution or covariance operator | Add a population law and actual covariance equation before claiming a spiked-model theorem |
| `Architectures/Basic.lean` | Globally differentiable activations exclude ReLU; the network has no biases | Keep regularity as hypotheses; reuse an upstream affine-layer definition with biases |
| `Architectures/Expressivity.lean` | A universal-approximation theorem cannot be imported unchanged into this bias-free network | Audit the statement: when the activation vanishes at zero, every current network outputs zero there, so it cannot approximate every target on a sample containing zero |
| `Generalization/Bounds.lean` | Lipschitz constants are quantified after the input pair | State a single constant valid for every pair, preferably through `LipschitzWith` |
| `Optimization/GradientFlow.lean` | Loss descent is assumed rather than derived from a parameter ODE | Define actual parameter updates first; add continuous-time dynamics later |
| Rademacher, Gaussian-norm and residual-contraction targets | Conclusions currently describe constants without the named random process or network | Replace the mathematical specifications before attempting proofs |
| `EndToEnd.lean` | The current theorem combines scalar inequalities | Later instantiate those inequalities with actual risk definitions and proved bounds |

Use `EuclideanSpace ℝ (Fin d)` or `PiLp 2` when the mathematics requires
Euclidean geometry. A plain `Fin d → ℝ` has the default supremum norm. Keeping
coordinate functions for indexing is fine, but replacing the existing `l2Norm`
with bare `norm` would change the mathematics. Matrix operator and Frobenius
norms must also be distinguished explicitly.

## Reuse decisions

These are evaluation priorities, not a list of packages already integrated.
Upstream documentation was reviewed; external repositories have not been
locally built or subjected to OpenTheory's axiom policy in this planning pass.

| Area | First candidate | Adoption test / alternative |
|---|---|---|
| Analysis, probability, linear algebra | Mathlib | Use its interfaces directly; search the selected revision before adding a local definition |
| Network and tensor semantics | [TorchLean](https://github.com/lean-dojo/TorchLean), especially `NN.Spec` and focused proof modules | Compile a real-valued affine/MLP example and prove a coordinate correspondence; check the package's native build requirements |
| Architecture/backward-pass proofs | [lean4-mlir](https://github.com/brettkoonce/lean4-mlir) | Evaluate a proof-only affine/MLP import if TorchLean does not fit; defer compiler integration |
| Deterministic optimizer theory | [Optlib](https://github.com/optsuite/optlib) | Instantiate a GD theorem on ridge regression; compare the smaller [gradient-descent-lean](https://github.com/velvetmonkey/gradient-descent-lean) if porting is cheaper |
| Stochastic optimization | [ArtificialTheorems](https://github.com/GasStationManager/ArtificialTheorems) | Inspect precise SGD hypotheses, licensing, and transitive axioms before adopting individual results |
| Generalization | [FoML / lean-rademacher](https://github.com/auto-res/lean-rademacher) | Instantiate a bounded finite hypothesis class first, then the network loss class |
| Expressivity | [Neural Network Proofs](https://github.com/davorrunje/neural-network-proofs) | Match biases, activation assumptions, domain, output type, and norm before transferring a UAT |

TorchLean advertises mathematical specifications, runtime optimizers, and
selected correctness theorems. Its
[trust-boundary document](https://github.com/lean-dojo/TorchLean/blob/main/docs/TRUST_BOUNDARIES.md)
must be read for the actual execution path we select. A runtime operation being
available does not establish its real-valued derivative or convergence theorem.

Keep SciLean, TensorLib, Tyr and additional statistics libraries as alternatives
from the survey. Add one only when a concrete missing capability justifies it.
Choose one canonical model representation and one optimizer-theory provider
for the first release; maintaining several competing tensor systems would
increase adapter work.

For each selected dependency, record its exact commit, Lean version, Mathlib
commit, license, narrow import, exact declaration, theorem assumptions, build
result, and axiom closure. A README claim or matching theorem name is discovery
evidence, not an acceptance test. In particular, ArtificialTheorems explicitly
documents an extra unproved premise in one approximation result; such a premise
must be discharged or remain visible in the resulting theorem.

## Compatibility and adoption procedure

The first answer's version numbers were observations of moving upstream
branches, not a tested compatible dependency set. TorchLean's published
[toolchain](https://github.com/lean-dojo/TorchLean/blob/main/lean-toolchain)
currently specifies `v4.34.0`; FoML's published
[toolchain](https://github.com/auto-res/lean-rademacher/blob/main/lean-toolchain)
specifies `v4.27.0-rc1`. Neither is a drop-in dependency for the current project.

1. Create isolated compatibility projects and resolve immutable upstream
   commits. Build each candidate with its declared toolchain first.
2. Start a common-version trial using TorchLean's selected toolchain if its
   mathematical API passes the model probe. Inspect its Mathlib manifest at
   that same commit; do not infer compatibility from separately cached pages.
3. Prove one adapter and instantiate one substantive upstream theorem in the
   same trial project. An umbrella import compiling is insufficient.
4. If a theory library needs migration, prefer a small, attributable port of
   the required modules or an isolated compatibility fork. Preserve licenses
   and provenance. Record the maintenance cost before adding it to the core.
5. Select the common Lean/Mathlib pair from those results. Upgrade OpenTheory
   in an isolated worktree containing the intended current source changes.
   Repair imports, proof API changes and auxiliary scripts, and rerun all checks.
6. Commit tested dependency revisions in the Lake configuration and manifest.
   Add dependencies one at a time. Do not depend on moving `main`/`master` heads.

Lake does not provide independent incompatible Mathlib universes inside one
proof environment. A downstream integration package helps isolate native
runtime dependencies, but its Lean/Mathlib versions must still agree with the
OpenTheory revision it imports.

If the runtime package is too expensive to adopt, retain the Mathlib core and
port only the necessary semantics/proofs with attribution. This fallback should
be chosen from probe results, rather than rebuilding an entire tensor library.

## Milestones and acceptance criteria

### M0 — Establish the baseline and select reusable results

Deliver a dependency decision record, a mathematical statement inventory, and
three small probes: a model correspondence, a GD theorem instantiation, and a
generalization theorem instantiation. Record successes and failed compatibility
attempts. Select the architecture and optimization providers before broad API
refactoring. The generalization probe may remain a documented migration task
without delaying deterministic training.

Done when the baseline audit runs, each initial dependency has a tested pin,
  and every proposed reused result has an exact declaration and clear hypothesis
mapping. The baseline build and statement inventory have started in this pass;
external integration probes remain implementation work.

### M1 — Migrate the toolchain and strengthen verification

Upgrade using the versions selected in M0. Keep the old build reproducible and
preserve the user's uncommitted work. Update the blueprint exporter, which
currently hardcodes `v4.11.0`, to read the actual toolchain.

Retain the global axiom audit, and add a strict list of completed declarations
whose transitive dependencies must not contain `sorryAx`. The current auditor
allows `sorryAx` everywhere it scans, so a passing audit alone is not evidence
that a completed milestone is proved. Extend declaration validation to resolve
imported upstream names in Lean; the current Python check only scans local text.

Check blueprint freshness against the committed file before overwriting it;
the current workflow exports into that file before comparing, masking staleness.

Done when the full library, selected dependencies, audit, declaration checks,
and blueprint build pass together on the chosen toolchain.

### M2 — Data, assumptions, parameter geometry, and losses

Define finite observations independently of boundedness, with adapters for the
current `Sample`. Define a population distribution on input/label pairs and a
joint law for random samples; state IID as independence plus the common law.
Boundedness, finite moments, and sub-Gaussianity should be optional predicates
using Mathlib's existing definitions where available.

Use the selected upstream model's parameter space and forward map. Introduce
an OpenTheory wrapper only where a common theorem interface requires it.
Provide empirical mean loss and population expected loss, with explicit
measurability/integrability hypotheses. In Lean, merely writing an integral
does not prove integrability. Start with squared loss; add cross-entropy when
the output semantics and positivity conditions are specified.

Done when the old sample and bias-free forward map have explicit adapters,
Euclidean norm correspondence is proved, and ridge-regression loss is a
well-defined differentiable function of parameters.

### M3 — First complete training theorem

Use the objective

`L(θ) = (1 / (2n)) Σᵢ (⟪θ, xᵢ⟫ − yᵢ)² + (λ / 2) ‖θ‖²`, with `n > 0`, `λ > 0`.

Reuse matrix calculus and the chosen GD library to identify the gradient,
establish smoothness and strong convexity, and connect the recurrence
`θₖ₊₁ = θₖ − η ∇L(θₖ)` to the imported convergence theorem. Derive an admissible
step-size bound from the dataset/covariance; discharge minimizer-existence and
stationarity requirements rather than hiding them in a bundled assumption.

Done when one example composes data, loss, the actual parameter update, and a
proved convergence bound with no `sorryAx` in its dependency closure. A small
rational/numerical example may accompany it as a sanity check, but the exact-real
theorem is the deliverable. Global convex convergence does not automatically
transfer to the later nonlinear networks.

### M4 — Reusable neural architectures

Adopt affine layers with biases, activations, composition, and residual addition
from the selected provider. Keep parameter fields separate from fixed
configuration. Prove correspondence to the existing bias-free `TwoLayerNet`
when biases are zero, and prove a global input Lipschitz bound using Mathlib
composition lemmas and explicit operator norms.

Then add scaled dot-product attention and a small transformer block: projections,
multi-head splitting/concatenation, causal masking, normalization, and residual
paths. Specify positive dimensions and normalization epsilon. For masked
softmax, require at least one allowed key per row or define the empty-row
behavior explicitly. Distinguish training/evaluation state and randomness for
dropout or batch normalization if introduced.

Done when a two-layer MLP and residual MLP have checked semantics and substantive
bounds, and the attention example has shape correctness, normalized nonnegative
weights, and a convex-combination property. Differentiation claims for ReLU must
handle its nondifferentiability at zero explicitly. Full CNNs/transformers can
then reuse the same interfaces; their inclusion is beyond the minimum release.

### M5 — Optimizer catalog and stochastic training

Reuse optimizer definitions when their equations are explicit and available
over the required scalars. Otherwise add small state/step definitions with a
proved correspondence to the provider. Keep the gradient estimator separate
from the update rule. For coordinatewise optimizers, use an explicit finite
coordinate structure rather than assuming an arbitrary parameter space has
coordinatewise squares and division.

Initial catalog: GD/SGD, momentum, Adam and AdamW. Record momentum conventions,
bias-correction indexing, epsilon placement, schedules, and decoupled weight
decay. Prove elementary state invariants and update correspondence. These are
different deliverables from a convergence theorem.

For an imported SGD theorem, supply a filtration, adapted iterates, conditional
unbiasedness, moment/variance bounds, and the required learning-rate conditions.
Do not model sampling without replacement as IID by default. AdamW convergence
for unrestricted deep networks is not a promised milestone; theorem scope must
follow the actual assumptions of a selected result.

### M6 — Generalization, expressivity, and executable integration

Instantiate FoML on a bounded measurable loss class and a random IID sample.
Discharge finiteness/separability and supremum-measurability conditions as
required by the selected theorem. Squared loss needs a justified bound, such as
bounded labels and predictions. Show the result applies uniformly before
substituting parameters trained on that same sample.

Transfer a selected approximation theorem only after matching its network
definition and hypotheses. Connect actual empirical/population risks and the
optimization result to a documented excess-risk decomposition, including its
constants and comparison class. The existing scalar inequality can remain a
final arithmetic helper.

Finally, create an optional downstream runtime integration with one CPU example.
Record which parts are proved over reals, which numerical error bounds are
proved, and which FFI/compiler operations remain trusted. GPU and MLIR work
follows only when a concrete execution requirement justifies it.

Done when one bounded learning problem has a real probability bound, and one
executable example has a precise account of its relationship to the theory.
These are follow-on milestones after the initial release.

## Work packaging and expected outcome

Use small, independently buildable changes in this order:

1. Baseline repair, statement inventory, and dependency probes.
2. Tested toolchain/dependency migration and verification improvements.
3. Data and model adapters with loss definitions.
4. Ridge regression plus imported GD convergence.
5. MLP/residual semantics and Lipschitz results.
6. Optimizer specifications, followed by attention, SGD and generalization.

M0–M3 are the first complete release. M4's MLP/residual portion is the first
neural-network extension. M4's transformer portion and M5–M6 are separate
increments. Estimate their effort after the compatibility probes: proof ports
and missing hypotheses can dominate implementation time, so a fixed delivery
date is not yet justified.

For every mathematical change, write the human statement and assumption map,
identify the imported result, implement the adapter, and check the theorem's
axioms. Update `content.tex` and regenerate `blueprint.json` only after the
corresponding declaration exists. A proof receives `\leanok` only after its
entire dependency closure is free of `sorryAx`.

Required checks remain:

```bash
lake build
lake env lean --run scripts/audit_axioms.lean
python3 scripts/check_lean_decls.py
python3 scripts/export_blueprint.py blueprint/blueprint.json
```

The initial result is a small, reproducible library with an end-to-end training
theorem and reusable data/model/optimizer interfaces. It establishes how to
reuse external Lean developments without duplicating their foundations.
Research claims about general transformer training, feature learning, µP or
spectral bias remain explicit later targets whose mathematical statements must
be validated before proof work begins.
