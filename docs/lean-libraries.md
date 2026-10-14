# Lean libraries for OpenTheory

This is a practical survey of reusable Lean libraries for the mathematics in
OpenTheory. It was checked on 2026-10-12 against this repository, its pinned
Mathlib source, and the upstream project documentation linked below.

The short recommendation is:

1. Use the pinned Mathlib modules first. They are already installed and cover
   most of the foundations OpenTheory needs.
2. Prefer narrow imports in library files and `import OpenTheory` only in
   examples or downstream projects that need the whole local API.
3. Evaluate external libraries in a separate upgrade branch. OpenTheory is
   pinned to Lean and Mathlib `v4.11.0`; the most relevant external projects
   currently use newer, mutually different Lean versions.

This survey is broad rather than literally exhaustive: Lean packages and
Mathlib change continuously, and many research repositories are not stable
libraries. The entries below are the ones with a credible, direct connection
to OpenTheory's present blueprint.

## What is installed now

[`lakefile.lean`](../lakefile.lean) declares one direct dependency:

```lean
require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git" @ "v4.11.0"
```

The toolchain in [`lean-toolchain`](../lean-toolchain) is
`leanprover/lean4:v4.11.0`. Mathlib brings its normal support packages such as
Aesop and Batteries transitively, but those are proof/programming support,
not separate mathematical foundations for this project.

No extra Lake dependency is required to use another Mathlib module. Add an
`import Mathlib....` line to the Lean file and build.

## Importing OpenTheory itself

Inside this repository, import only the preceding OpenTheory modules that a
file actually uses:

```lean
import OpenTheory.Data.Sample
import OpenTheory.Representation.Gram

namespace OpenTheory

-- New declarations here can use Sample and dataGram.

end OpenTheory
```

For a downstream project, or a scratch file that needs everything currently
exported by the project:

```lean
import OpenTheory

open OpenTheory

#check Sample
#check dataGram
#check TwoLayerNet.forward
```

The import path follows the file path with `/` replaced by `.` and `.lean`
removed. For example,
`OpenTheory/Representation/Gram.lean` becomes
`import OpenTheory.Representation.Gram`.

## Mathlib modules to use now

Every module in this section was checked against the source tree of the
repository's pinned Mathlib `v4.11.0`. The existing OpenTheory imports and the
full OpenTheory library build were also checked after writing this guide.

### Finite vectors, sums, matrices, and linear maps

Useful for `Data`, `Architectures`, `Representation`, and all finite-width or
finite-sample statements:

```lean
import Mathlib.Algebra.BigOperators.Group.Finset
import Mathlib.Data.Fintype.Card
import Mathlib.Data.Matrix.Basic
import Mathlib.LinearAlgebra.Matrix.DotProduct
import Mathlib.LinearAlgebra.Matrix.ToLin
import Mathlib.Analysis.InnerProductSpace.PiL2
```

Use these for finite sums, `Fin d -> ℝ` vectors, matrix multiplication,
matrix-vector products, conversion between matrices and linear maps, dot
products, and Euclidean inner-product/norm structure.

OpenTheory currently models vectors as `Fin d -> ℝ`. That is compatible with
Mathlib's `Pi`-space instances, so a wholesale replacement with a custom
vector type is unnecessary. A useful next refactor is to reuse Mathlib's
`inner` and `norm` instead of maintaining parallel `dotProduct` and `l2Norm`
definitions.

### Gram matrices, symmetric matrices, eigenvalues, and spectra

Useful for `Representation/Gram.lean`, alignment, spectral bias, spiked
models, and operator-norm work:

```lean
import Mathlib.LinearAlgebra.Matrix.Hermitian
import Mathlib.LinearAlgebra.Matrix.PosDef
import Mathlib.LinearAlgebra.Matrix.Spectrum
import Mathlib.Analysis.InnerProductSpace.Rayleigh
import Mathlib.Analysis.InnerProductSpace.Spectrum
import Mathlib.Analysis.Matrix
```

These modules provide Hermitian/symmetric and positive-definite matrix theory,
finite-dimensional spectral theorems, Rayleigh-quotient results, and norms on
matrices. `Mathlib.Analysis.Matrix` connects the matrix norm used there to the
operator norm of the associated linear map.

For OpenTheory, this is preferable to defining a spectral gap only as two
unrelated real numbers: eventually connect `SpectralGap` to eigenvalues of a
symmetric covariance or Gram operator.

### Norms, continuous linear maps, and Lipschitz bounds

Useful for network layer bounds, residual contractions, feature alignment,
and generalization:

```lean
import Mathlib.Topology.MetricSpace.Lipschitz
import Mathlib.Analysis.NormedSpace.OperatorNorm.Basic
import Mathlib.Analysis.InnerProductSpace.Basic
```

The central reusable interfaces are `LipschitzWith`, `AntilipschitzWith`,
`Isometry`, `ContinuousLinearMap`, `norm`, and `opNorm`. Using `LipschitzWith`
in `ActivationFunction` would make Mathlib's composition and continuity lemmas
available directly.

### Derivatives, gradients, and gradient flow

Useful for `Optimization`, parameter dynamics, NTK calculations, and
feature-learning trajectories:

```lean
import Mathlib.Analysis.Calculus.Deriv.Basic
import Mathlib.Analysis.Calculus.FDeriv.Basic
import Mathlib.Analysis.Calculus.FDeriv.Comp
import Mathlib.Analysis.Calculus.FDeriv.Pi
import Mathlib.Analysis.Calculus.Gradient.Basic
import Mathlib.Analysis.InnerProductSpace.Calculus
```

Use `HasDerivAt` for scalar-to-scalar paths, `HasFDerivAt` for Fréchet
derivatives, and `HasGradientAt`/`gradient` for real or complex inner-product
spaces. A mathematically meaningful gradient-flow model should define the
parameter space as a normed inner-product space and connect the path derivative
to the negative loss gradient; the present `DescentPath` only records the
resulting scalar risk derivative.

### Convexity and optimization foundations

Useful for convergence rates and loss geometry:

```lean
import Mathlib.Analysis.Convex.Function
import Mathlib.Analysis.Convex.Deriv
import Mathlib.Analysis.Convex.Strong
import Mathlib.Analysis.Convex.Jensen
```

Mathlib `v4.11.0` has substantial convex-analysis infrastructure, including
convex functions, derivative characterizations, Jensen inequalities, and
strong/uniform convexity. It does not package a full modern ML optimizer API,
so iteration-specific gradient-descent theorems may still come from a small
local layer or, after a version migration, Optlib.

### ODEs and differential inequalities

Useful for continuous-time training and exponential convergence:

```lean
import Mathlib.Analysis.ODE.Gronwall
import Mathlib.Analysis.ODE.PicardLindelof
```

`Gronwall` is the natural basis for replacing the scaffold in
`Optimization/Convergence.lean` with a real exponential risk bound.
`PicardLindelof` supplies local existence/uniqueness machinery when a training
trajectory is modeled as an actual ODE.

### Measure theory, probability, expectation, and Gaussian variables

Useful for replacing deterministic probability placeholders in
`Generalization` and for random initialization:

```lean
import Mathlib.Probability.Notation
import Mathlib.Probability.Integration
import Mathlib.Probability.Independence.Basic
import Mathlib.Probability.Moments
import Mathlib.Probability.Variance
import Mathlib.Probability.Distributions.Gaussian
import Mathlib.MeasureTheory.Constructions.Pi
```

These provide probability spaces, expectation notation, independence,
moments, variance, Gaussian measures, product measurable spaces, and random
vectors represented as measurable functions. This is the appropriate base for
stating `gaussian_operator_norm_bound` as a probability theorem rather than an
existential inequality about a constant.

The pinned `v4.11.0` tree does **not** contain the newer Mathlib sub-Gaussian
and concentration-inequality API. Do not assume a theorem found in the latest
online Mathlib docs is available here; select version `v4.11.0` or inspect the
local source.

### Integrals, Lp spaces, and elementary tail bounds

Useful for population risk, square-integrable predictors, and Markov/Chebyshev
arguments:

```lean
import Mathlib.MeasureTheory.Integral.Bochner
import Mathlib.MeasureTheory.Integral.MeanInequalities
import Mathlib.MeasureTheory.Function.LpSpace
import Mathlib.MeasureTheory.Function.LpSeminorm.ChebyshevMarkov
```

Use Bochner integration for vector-valued expectations, `Lp`/`MemLp` for
integrability assumptions, and the Chebyshev/Markov module for basic tail
bounds.

### Asymptotic width/sample-size statements

Useful for μP/NTK width limits and rates in `n`, `m`, or depth:

```lean
import Mathlib.Analysis.Asymptotics.Asymptotics
import Mathlib.Analysis.Asymptotics.AsymptoticEquivalent
import Mathlib.Analysis.Asymptotics.Theta
```

Prefer filter-based `IsBigO`, `IsLittleO`, `IsTheta`, and asymptotic
equivalence when the theorem genuinely concerns a width limit. The current
scaling structures describe finite formulas but not convergence as width tends
to infinity.

### Approximation and universal approximation ingredients

Useful for `Architectures/Expressivity.lean`:

```lean
import Mathlib.Topology.ContinuousFunction.StoneWeierstrass
import Mathlib.Algebra.Polynomial.Basic
import Mathlib.Analysis.SpecialFunctions.Pow.Real
```

Stone-Weierstrass and polynomial approximation are useful ingredients, but the
pinned Mathlib release does not provide an off-the-shelf neural-network
universal approximation theorem matching `TwoLayerNet`. Expect to build an
architecture-specific bridge or evaluate TorchLean's newer formalization.

### Proof automation

Mathlib also supplies the tactics already used throughout Lean developments:

```lean
import Mathlib.Tactic
```

That umbrella import includes tactics such as `linarith`, `nlinarith`, `ring`,
`norm_num`, `positivity`, `aesop`, and `fun_prop`. In stable library files,
importing a narrower tactic module keeps dependencies clearer; in exploration,
`Mathlib.Tactic` is convenient.

## Highest-value mapping for the current open targets

| OpenTheory target | Start with | Later external candidate |
|---|---|---|
| `exists_subspace_approximation` | Stone-Weierstrass, polynomials, finite-dimensional topology | TorchLean for comparison |
| `alignment_le_one` and `alignment_feature_distance` | matrix multiplication, trace/Frobenius identities, Cauchy-Schwarz | none needed |
| `risk_exponential_approach` | `Mathlib.Analysis.ODE.Gronwall` | Optlib for discrete GD |
| `layer_lipschitz`, `forward_lipschitz` | `LipschitzWith`, Pi norms, operator norms | none needed |
| `rademacher_symmetrization` | Mathlib probability, product measures, expectations | FoML / lean-rademacher |
| `gaussian_operator_norm_bound` | Gaussian distributions, moments, matrix/linear-map operator norms | newer Mathlib concentration, StatLean, ProbabilityApproximation |
| `spectral_bias_mup_alignment` | matrix spectrum, Rayleigh quotients, calculus, asymptotics | no direct mature drop-in found |
| `resnet_representation_contraction` | Lipschitz maps and operator norms | none needed |
| `feature_learning_resnet_alignment` | all of the preceding infrastructure | no direct drop-in theorem found |

## External libraries worth tracking

External dependencies must agree on a Lean/Mathlib revision. Lake does not
isolate two incompatible Mathlib versions inside one project. The versions
below are the upstream versions observed on the survey date, not versions that
have been installed or tested in OpenTheory.

### FoML / lean-rademacher: strongest direct match for generalization

[FoML](https://github.com/auto-res/lean-rademacher) formalizes empirical and
expected Rademacher complexity, symmetrization, bounded differences,
McDiarmid-style bounds, separable hypothesis classes, covering numbers,
Dudley's entropy integral, linear predictors, RKHS predictors, and ERM.

- Direct relevance: extremely high for `Generalization/Rademacher.lean`.
- Public root import: `import FoML`.
- Observed toolchain: Lean `v4.27.0-rc1`, with Mathlib tracking `master`.
- Current verdict: study and plan a port; do not add to the `v4.11.0` branch.

After aligning toolchains, a pinned dependency would look like:

```lean
require FoML from git
  "https://github.com/auto-res/lean-rademacher.git" @ "<tested-commit>"
```

Then:

```lean
import FoML
```

### StatLean: broad statistics and statistical learning

[StatLean](https://github.com/StatLean/Stat-Lean) includes concentration
inequalities, high-dimensional statistics, optimization, empirical risk,
uniform convergence, VC theory, Rademacher complexity, stability, and PAC-Bayes
bounds.

- Direct relevance: high for probability, generalization, and Gaussian/high-
  dimensional results.
- Public root import: `import StatLean`.
- Observed toolchain: Lean/Mathlib `v4.29.1`.
- Current verdict: a valuable source after a coordinated upgrade; its breadth
  also means a large dependency surface.

```lean
require StatLean from git
  "https://github.com/StatLean/Stat-Lean.git" @ "<tested-commit>"
```

### Optlib: formal optimization algorithms

[Optlib](https://github.com/optsuite/optlib) develops gradients,
subdifferentials, smoothness, convexity, optimality conditions, gradient and
proximal methods, Nesterov acceleration, LASSO, and related convergence rates.

- Direct relevance: high for `Optimization`.
- Example narrow imports:
  `import Optlib.Algorithm.GD.GradientDescent` and
  `import Optlib.Algorithm.GD.GradientDescentStronglyConvex`.
- Observed toolchain: Lean `v4.13.0`.
- Current verdict: closest external project to the current pin, but still
  requires an upgrade/compatibility test. Some of its foundations have already
  migrated into Mathlib, so import only the algorithm modules actually used.

```lean
require optlib from git
  "https://github.com/optsuite/optlib.git" @ "<tested-commit>"
```

### gradient-descent-lean: small proof spine for classical GD

[gradient-descent-lean](https://github.com/velvetmonkey/gradient-descent-lean)
is a compact library proving descent, the convex `O(1/k)` rate, and geometric
strongly-convex convergence over real Hilbert spaces.

- Direct relevance: high but narrower than Optlib.
- Imports include `GradientDescent.Smooth`, `GradientDescent.Convex`, and
  `GradientDescent.Convergence`.
- Observed toolchain: Lean/Mathlib `v4.28.0`.
- Current verdict: useful reference or small future dependency; version is
  incompatible with the present branch.

### SciLean: differentiable scientific computing and AD

[SciLean](https://github.com/lecopivo/SciLean) targets symbolic/numerical
scientific computing, automatic differentiation, optimization, differential
equations, and machine learning.

- Direct relevance: medium for executable differentiation and experiments;
  lower for OpenTheory's present theorem-only core.
- Root import: `import SciLean`.
- Current documented SciLean dependency tag: `v4.20.1`.
- Operational cost: OpenBLAS and platform-specific linker configuration.
- Current verdict: keep optional. Do not make the proof library depend on a
  numerical runtime unless executable training becomes an explicit goal.

SciLean's own README contains the required `moreLinkArgs`; use that current
configuration rather than copying an old linker path.

### TorchLean: neural-network execution and verification

[TorchLean](https://github.com/lean-dojo/TorchLean) provides shape-checked
tensors, model definitions, autograd, optimizers, CPU/GPU execution,
PyTorch interchange, numerical-error reasoning, robustness verification, and
a formalized universal-approximation result.

- Direct relevance: medium for architecture semantics and expressivity;
  it solves a broader execution/verification problem than OpenTheory.
- Main API import: `import NN.API`.
- Observed toolchain: Lean `v4.34.0`.
- Operational cost: substantially heavier, with optional LibTorch/CUDA pieces.
- Current verdict: compare definitions and the approximation theorem in an
  experimental branch; avoid coupling the Lean theory core to the runtime now.

### Arlib: finite probability and concentration

[Arlib](https://github.com/meelgroup/arlib) contains reusable finite
probability, concentration, information theory, randomized algorithms, and
coresets.

- Direct relevance: medium. Its elementary `FinProb` model can simplify finite
  Rademacher-sign arguments, but OpenTheory will also need Mathlib's
  measure-theoretic probability for Gaussian results.
- Area import: `import Arlib.Probability`; full import: `import Arlib`.
- Observed toolchain: Lean `v4.33.0`.
- Current verdict: consider only after deciding whether to use its finite-
  probability abstraction.

### ProbabilityApproximation: advanced Gaussian approximation

[ProbabilityApproximation](https://github.com/Polarnova/ProbabilityApproximation)
formalizes Berry-Esseen bounds, multivariate Gaussian approximation, Gaussian
pushforwards and covariance infrastructure, whitening, and supporting convex
geometry.

- Direct relevance: medium for future high-dimensional Gaussian work, but it
  does not directly supply the standard rectangular Gaussian operator-norm
  bound.
- Root import: `import ProbabilityApproximation`.
- Observed release: `v0.9.6`, Lean/Mathlib `v4.32.0`.
- Current verdict: track or reuse ideas after an upgrade; too far from the
  present toolchain for an immediate dependency.

## Safely adding an external package

Do this on a branch because it can update `lean-toolchain`, `lake-manifest.json`,
Mathlib, and many proof scripts.

1. Read the dependency's `lean-toolchain` and its Mathlib revision.
2. Choose one common Lean/Mathlib version for the whole project.
3. Pin a release tag or commit, not `main` or `master`.
4. Add one `require` entry to `lakefile.lean`.
5. Update and test:

   ```bash
   lake update
   lake exe cache get
   lake build
   lake env lean --run scripts/audit_axioms.lean
   python3 scripts/check_lean_decls.py
   ```

6. Add the narrowest useful `import Package.Module` to the consuming file.
7. Re-run the full OpenTheory checks before merging.

A generic dependency entry is:

```lean
require packageName from git
  "https://github.com/owner/repository.git" @ "<release-tag-or-commit>"
```

Adding `require` makes package modules visible; it does not automatically put
their declarations in every file. Each Lean file must still contain an
`import`.

## How to discover the right import

For the pinned Mathlib version, search the checked-out source rather than the
latest web docs:

```bash
# Search declaration names or concepts.
rg -n "Gronwall|HasGradientAt|LipschitzWith" .lake/packages/mathlib/Mathlib

# Convert the matching path to an import.
# .lake/packages/mathlib/Mathlib/Analysis/ODE/Gronwall.lean
# -> import Mathlib.Analysis.ODE.Gronwall
```

In a scratch Lean file:

```lean
import Mathlib.Analysis.ODE.Gronwall
import Mathlib.Analysis.Calculus.Gradient.Basic
import Mathlib.Topology.MetricSpace.Lipschitz

#check HasDerivAt
#check HasGradientAt
#check LipschitzWith
```

Useful checks are:

- `#check Name` to see whether a declaration is in scope and inspect its type.
- `#print Name` to inspect its definition or theorem statement.
- Ctrl/Cmd-click in the Lean editor to open its source.
- `lake env lean path/to/File.lean` for a focused compile.
- `lake build OpenTheory.Representation.Gram` for a focused Lake target.

Avoid `import Mathlib` in core files merely to make an unknown name appear. It
is fine for exploration, but once the needed declaration is known, use the
module that defines or re-exports it.

## Recommended adoption order

1. Refactor current finite vectors, norms, and Lipschitz hypotheses onto
   Mathlib interfaces without changing the toolchain.
2. Use Mathlib probability to state genuine random-sample and Gaussian claims.
3. Use Grönwall and the Mathlib gradient API for the optimization layer.
4. Plan a coordinated Mathlib upgrade; do not upgrade separately for each
   external package.
5. After upgrading, evaluate FoML first for generalization and Optlib (or the
   smaller gradient-descent library) for optimization.
6. Keep SciLean and TorchLean optional unless OpenTheory expands from formal
   mathematical guarantees into executable/trained network verification.

## Upstream references

- [Mathlib repository](https://github.com/leanprover-community/mathlib4)
- [Mathlib documentation](https://leanprover-community.github.io/mathlib4_docs/)
- [FoML / lean-rademacher](https://github.com/auto-res/lean-rademacher)
- [StatLean](https://github.com/StatLean/Stat-Lean)
- [Optlib](https://github.com/optsuite/optlib)
- [gradient-descent-lean](https://github.com/velvetmonkey/gradient-descent-lean)
- [SciLean](https://github.com/lecopivo/SciLean)
- [TorchLean](https://github.com/lean-dojo/TorchLean)
- [Arlib](https://github.com/meelgroup/arlib)
- [ProbabilityApproximation](https://github.com/Polarnova/ProbabilityApproximation)
