# Deep-learning theory: next implementation plan

Planning snapshot: 2026-10-12. This is a proposed dependency roadmap, not a list
of proved results. Existing completion status remains in [reuse-plan.md](reuse-plan.md);
tested dependency revisions remain in [dependency-status.md](dependency-status.md).

## Recommendation

Build a reusable chain from **data assumptions + model + training algorithm** to
**a quantitative guarantee**, then extend that chain across architecture families.
Do not grow an architecture catalog disconnected from learning theorems. Equally,
do not restrict the project to classical regression: use regression to validate
the statistical bridge, then require a genuine neural-network application.

Maintain three connected tracks:

1. Learning: optimization, sample complexity, population/excess risk.
2. Architecture: expressivity, stability, symmetry, representation and scaling.
3. Interpretation: precisely specified interventions and explanation fidelity.

Execution/rounding correctness is a separate fourth track, not a consequence of
real-valued mathematics. “All aspects” is a modular research program, not one
universal theorem or a promise of assumption-free guarantees.

## Reuse before new proofs

| Need | Preferred source | Work belonging in OpenTheory |
| --- | --- | --- |
| Probability, analysis, matrices | Mathlib, already pinned | Assumption discharge and small adapters |
| Tensor/model/optimizer semantics | TorchLean, already pinned | Coordinate correspondence and architecture-specific bounds |
| Uniform deviation and approximate ERM | [FoML](https://github.com/auto-res/lean-rademacher) | Match our risk, sample law, loss class and learner |
| Stability, PAC/VC/PAC-Bayes | [StatLean](https://github.com/StatLean/Stat-Lean) | Select one theorem matching the intended assumptions |
| Empirical processes, least squares, spectral probability | [SLT](https://github.com/YuanheZ/lean-stat-learning-theory) | Evaluate a pinned narrow import before choosing it |
| Optimization beyond existing GD | TorchLean first; Optlib / other surveyed providers if needed | Gradient-estimator and algorithm correspondences |
| Approximation/expressivity | Neural Network Proofs, then Mathlib | Match activation, biases, topology and quantitative scope |
| Runtime/compiler refinement | SciLean, Tyr, lean4-mlir as evaluated candidates | Prove a selected execution path agrees within an error bound |

FoML's deterministic ERM subset and StatLean's finite-class confidence subset
are vendored and audited. FoML's larger confidence stack failed its compatibility
probe. SLT remains an evaluated candidate, not an integrated dependency.
Do not integrate three competing statistics foundations for overlapping results.
Choose the smallest successfully audited theorem path; retain other providers
for capabilities it lacks. Never rewrite concentration or symmetrization just
because an adapter is needed. A port must preserve provenance and license.

## Dependency DAG

Arrows mean prerequisite to deliverable, not Lean imports. E is existing checked
infrastructure; numbered nodes are proposed work. Shared node IDs denote joins.
This planning graph is intentionally separate from the verified blueprint graph.

```text
E: data/risk + ridge GD + MLP/attention/transformer semantics
|
+--> 1: audited statistical bridge ------> 2: trained linear/ridge guarantee
|           |                                  |
|           +--------------------+-------------+
|                                v
+--> 3: architecture bounds ---> 4: neural-class generalization
|           |                          |
|           +--> 5: ablation fidelity   +----+
|                                           v
+--> 6: stochastic optimization ------> 7: combined neural learning guarantee
|
+--> 8: quantitative expressivity ----------> 7
|
+--> 9: representation / NTK / width-scaling research
|
+--> 10: execution refinement (separate trust boundary)
```

Additional edges: 3 -> 6 (verify the chosen model's smoothness assumptions);
3 -> 8 and 3 -> 9 (fixed architecture semantics); 1 -> 9 (random initialization
and concentration). Node 5 needs no generalization theorem. Node 4 needs no
proof that SGD converges. Node 7 combines only compatible results; a stationarity
bound from node 6 is not an excess-risk optimization bound.

## First three implementation increments

Progress: the four-module deterministic FoML ERM subset is vendored with
provenance and OpenTheory risk adapters. FoML's confidence stack failed its
4.34 probe, so an eight-module StatLean subset was selected instead; one
definitional compatibility patch suffices. The checked finite-class theorem
gives the exact tail `2 |H| exp (-2 n ε²)` for `[0,1]` losses. Increment 1's
finite-class acceptance test is complete. Continuous neural classes still need
Rademacher/covering machinery rather than discretization by assertion.

Increment 2's trained-ridge join is checked. The exact-GD objective rate is
implemented. `Ridge.training_radius` and `Dataset.ridge_training_radius` bound
the minimizer by `R` and every zero-initialized iterate by `2R`, under bounded
labels and `T² ≤ λR²`. For inputs bounded by `B`, the step
`λ / (B² + 2λ)²` is legal for every such dataset, with contraction at most
`1 - (λ / (B² + 2λ))²`.

`Generalization/Covering.lean` extends StatLean concentration to arbitrary
classes with a finite uniform loss cover: cover error `ρ` gives deviation
threshold `ε + 2ρ` and tail `2N exp(-2nε²)`. `BallCover.euclideanBall_net`
bounds the Euclidean ball by a radially projected floor grid. Squared loss on
the training ball is divided by `(2RB + T)²`, so the normalized loss lies in
`[0,1]`. `TrainedRidge.trained_ridge_tail` combines the actual iterates, the
regularized objective gap, the comparator's regularization, and that cover.
The grid cardinality is convenient, not sharp. Comparators have norm at most
`R`. The population is supported on the same input and label bounds. This is
a linear ridge guarantee. Its cardinality is exponential in the parameter
dimension and does not see the design spectrum.

The informative statistical target for this learner is checked in
`RidgeRisk.fixed_design_mse`. With the design held fixed and labels `A θ* + ε`
for white noise of scale `σ`, the in-sample mean squared error of the ridge
solution that exact gradient descent approaches is at most
`(λ/4)‖θ*‖² + σ² df / m`. The degrees of freedom are at most `rank(A)`, hence
at most `min(d, m)`. A design of rank at most `r` replaces the variance term by
`σ² r / m`. A zero design has degrees of freedom zero while the parameter
dimension stays positive. The bias term is the regularization price and does
not vanish as the number of rows grows. Ambient dimension enters only through
the rank. This is the classical fixed-design bias–variance shape, not a
random-design generalization theorem and not a neural-network theorem. It is
the linear prototype of a kernel or NTK guarantee.

An MLP parameter-ball grid cover is exponential in the number of weights and
does not see that spectrum, so it is not the route to an informative neural
guarantee. `KernelRidge.prediction_eq_kernelRidge` transfers the shape to the
kernel operator `K = c A A*` of a fixed feature map: primal gradient descent
predicts with the smoother `K(K+λI)⁻¹`, equal kernels give equal predictions,
and `rank(K) = rank(A)`. This is the classical primal–dual identity. A frozen
feature map inherits the ridge bound. Kernel ridge applies to a network only
while its hidden features stay put. `FeatureStep.preactivation_step` computes
one gradient step on `(1/2)(f(x)-y)²` for `TwoLayerNet`: the preactivation
changes by `-η₁ (f(x)-y) σ'(preᵢ) W₂ᵢ ‖x‖²`. With the readout at most its
initialization scale, `NTKScaling.standard` moves it by `O(1/√m)` and
`MuPScaling.standard` moves it by `O(1/m)`. Both are kernel-regime steps. The
first-layer rate `featureLearningRate m = m`, with a readout coordinate of size
`1/m`, has a width-free upper bound `|res| L ‖x‖²` and a lower bound `c³` when
the residual, derivative, readout floor, and input energy are all at least `c`.
This is the one-step calculation behind the feature-learning exponent.
`TangentStep.ntk_tangent_gap` turns it into an output error: if `σ'` is
`M`-Lipschitz, one NTK step stays within `M |f-y|² L² ‖x‖⁴ / √m` of its tangent
update, and one checked μP step stays within the same quantity over `m²`. The
feature-learning rate leaves an output gap of order `|f-y|²`.
`PredictionGap.expected_mse_of_near_ridge` gives the fixed-design bound to any
predictor within `δ` of the ridge predictor, at the price of a factor two and
an additive `2δ²/m`. The tangent gap does not discharge that `δ`. The
multi-step distance from the network to the ridge solution, the infinite-width
dichotomy, and a feature-learning excess-risk bound are not proved.
MLP-class generalization and the intervention milestone remain separate future
work.

2026-10-14 intermediate step: `MLPTrajectory.iterate_remainder_le` now bounds
the actual initialization-centered, regularized first-layer trajectory against
its initial Taylor model evaluated at the same displacement, for one example
and a finite horizon with an explicit residual envelope. The follow-on theorem
`MLPComparison.explicit_nonlinear_linearized_gap` now compares independently
trained nonlinear and linearized predictions for that one-example setting,
using a proved Jacobian-drift bound and accumulated local errors.
`MLPEndpoint.nonlinear_endpoint_gap` now adds the exact geometric optimization
error to a proved centered ridge endpoint on the actual initial-Jacobian
features. This is a one-example deterministic guarantee under residual bounds;
`MLPFixedDesign` now constructs the multi-row initial-Jacobian design, proves
its affine Taylor correspondence, and reuses ridge GD contraction and the
fixed-design expected MSE theorem for these frozen features. The multi-row
nonlinear trajectory comparison and its noise integration remain open. Thus
the statistical ridge δ above is still undischarged.
The narrower `MLPRisk.expected_mse` now closes the nonlinear MLP chain for one
fixed input and bounded, centered finite label noise: residuals and the ridge
gap are both derived. This is a finite-horizon expected prediction-error bound,
not a multi-input or unseen-input generalization theorem. See the
[exact statement and scope](mlp-risk-guarantee.md).
`MLPBatchBounds` verifies dataset-wide Taylor error, Jacobian operator drift,
and their combined local residual-term discrepancy for arbitrary fixed inputs.
`MLPBatchTrajectory` and `MLPBatchRisk` complete the multi-input trajectory
accumulation, derive the finite-horizon parameter/shift/residual envelopes from
`‖F₀ - y‖ ≤ S`, discharge the deterministic distance `batchGap` to the centered
multi-input ridge prediction, and prove the multi-observation statistical risk
bounds `batch_trained_expected_mse` and `batch_trained_expected_mse_of_rank`.
See [the MLP trajectory proof map](mlp-trajectory-status.md)
for the exact bound and scope.

### 1. Audit and integrate one statistical bridge

First candidate: FoML's
`approxERM_excessRisk_tail_bound_separable_of_sample_empirical_le_delta` and
its bounded-linear-predictor applications. Inspect the exact imported statement,
not only its name. Its sample-dependent ERM example uses outer probability;
either retain that formulation explicitly or prove measurability before calling
the event an ordinary measurable probability event.

Prove agreement with our empirical/population risks and IID sample law. Check
loss bounds, measurability, parameter continuity/separability, sample size,
confidence parameter, and whether the class is fixed independently of the sample.

Done when one nontrivial bounded loss-class confidence theorem builds on our
toolchain and its full axiom closure passes. A finite-class example is an adapter
test, not the final deep-learning deliverable.

### 2. Obtain a guarantee for an actual trained predictor

2026-10-13 increment: `Generalization/FiniteLearning.lean` constructs a finite
empirical-risk selector, proves its minimization property, and bounds the outer
probability that its population risk exceeds any candidate's by more than
`2 ε`: at most `2 |H| exp (-2 n ε²)`. Approximate selection adds its empirical
optimization error `η`. This applies to any fixed finite family of models,
including neural architectures, with measurable losses in `[0,1]` and IID data.
Candidates must be fixed independently of the selection sample; models trained
on separate data need a conditioning argument or an explicitly fixed collection.
The selector uses classical choice over exact real losses, so executable search
is a separate refinement. This completes a finite-model learning example.
The trained-ridge join is checked in `TrainedRidge.trained_ridge_tail`, with
the normalized loss, grid cover, and uniform step recorded above. The
data-and-optimizer bound for that same ridge solution is
`RidgeRisk.fixed_design_mse`: fixed-design in-sample error tracks ridge degrees
of freedom, which are at most the rank of the design. The MLP milestone below
remains pending, and a parameter-grid cover is not the next statistical step.

Use existing ridge GD with bounded inputs/labels and a fixed comparator ball.
Prove a sample-independent radius containing the selected iterates and minimizer;
choose a step size valid uniformly under these data bounds. Convert the current
distance-to-minimizer rate into an objective-gap bound. Track our half-squared-loss
and regularizer conventions exactly.

Combine uniform deviation, the objective gap, and the comparator's regularization
penalty. The intended result is a high-probability comparison of the trained
predictor's population risk with each predictor in the fixed ball, with explicit
sample-size, confidence, iteration and regularization terms. Do not silently
replace a regularized empirical minimizer by an unregularized ERM.

Important alternative check: the inspected StatLean regularized-ERM stability
route assumes globally Lipschitz losses. Squared loss on unrestricted parameters
does not qualify. Use a matching bounded-domain/smooth-loss theorem if available,
or choose another provider; clipping predictions changes the optimization problem.

Done when the theorem names the actual training iterates, has discharged rather
than assumed optimizer premises, and has a concrete nonvacuous parameter example.
If the radius/step proof blocks this increment, report that gap; a constrained
or clipped substitute must be documented as a different learner.

### 3. Transfer to neural networks; prove an intervention guarantee

For bounded-parameter MLPs, prove parameter-to-output control and a quantitative
loss-class complexity bound using imported contraction/covering results. A grid
cover of the parameter ball is exponential in the number of weights and ignores
the data spectrum; it is not an informative neural guarantee. The shape worth
transferring is the fixed-design effective dimension already checked for ridge.
`KernelRidge.lean` records it for the kernel of a fixed feature map.
`FeatureStep.lean` shows when one gradient step keeps those features in that
regime: the checked NTK and μP rates freeze preactivations as width grows, and
first-layer rate equal to the width does not. The infinite-width dichotomy and
the excess risk of a feature-learning run are still open.
The existing input-Lipschitz theorem alone is insufficient for class complexity.
Instantiate the uniform statistical theorem for any learner returning parameters
in that class. State an approximate-ERM consequence conditionally until an actual
optimizer supplies its empirical optimization guarantee.

In the architecture track, prove bounded-domain attention stability when Q/K/V
vary, then compose projections, residuals, feedforward layers and normalization.
Record dimension, sequence length, parameter norms, epsilon and mask dependence.
Handle changes of a discrete mask separately from fixed-mask continuity.

For interpretation, start with zeroing one head's contribution without changing
other heads. Bound the projected contribution, then propagate through downstream
layers. This is a precise ablation guarantee, not token deletion, head removal
with rescaling, or a claim that attention weights explain a decision. Formalize
token interventions separately because they may change every score and value.

Done when one nontrivial MLP-class generalization bound and one architecture-linked
ablation theorem pass the audit. Report constants and a numerical regime where
the bounds are informative, not merely finite.

## Expansion after those increments

| Track | Next relevant deliverable | Boundary to preserve |
| --- | --- | --- |
| CNNs | Reuse convolution semantics; prove operator bounds and translation equivariance with boundary conventions | Weight sharing and padding must match the executable specification |
| Transformers | Full-block bounded-domain stability, then model-class complexity | Row attention bounds are not full-model guarantees; record length dependence |
| GNNs / sets | Aggregation bounds and permutation equivariance | Degree bounds, normalization and graph sampling matter |
| Recurrent / state-space models | Finite-horizon and contractive-state stability | Time dependence and long-horizon constants cannot be hidden |
| SGD | Conditional unbiasedness, variance, filtration and an imported convergence theorem | Nonconvex stationarity is not global minimization or generalization |
| Adam / AdamW | Valid state invariants, then only a theorem for a matching algorithm variant | Do not infer general convergence from update definitions |
| Expressivity | Transfer an approximation theorem; obtain a rate where available | Qualitative universality alone supplies no finite-width error rate |
| Representation / scaling | The one-step preactivation split is checked; the infinite-width dichotomy and feature-learning excess risk are not | Frozen-feature learning is distinct from learning the representation |
| Generative models | Specify likelihood/score objective, data/noise law and sampling error separately | Supervised squared-error theory is not a generative-quality theorem |
| Execution | One CPU refinement with rounding/error hypotheses | Exact-real proofs do not certify FFI, floating point or GPU kernels |

PAC-Bayes, algorithmic stability and compression are alternative routes to neural
generalization when worst-case class bounds are too loose. Select one only after
checking that its quantitative conclusion improves the intended application.

## Data coverage and acceptance discipline

Images, text, graphs and time series can share abstract observations and encodings,
but not necessarily sampling assumptions. Start with bounded IID data; extend
separately to sub-Gaussian laws, finite-moment/heavy-tailed laws, dependent samples,
and distribution shift when a matching theorem is selected. Finite second moments
alone do not imply sub-Gaussian concentration. Encoding semantics, measurability
and moment assumptions belong in each application theorem.

For every increment:

1. Write the human theorem, comparator and assumption map before Lean code.
2. Record upstream commit, license, exact declaration, compatibility and axiom closure.
3. Add a failing adapter/edge-case check, then the proof; preserve existing work.
4. Run full `lake build`, strict axiom audit, Lean/Python regressions and blueprint checks.
5. Update verified blueprint nodes only after proof validation; keep proposed nodes here.
6. State whether the result is semantic, deterministic, in expectation, high-probability,
   asymptotic, conditional on optimization, or about executable computation.

The next release should demonstrate both a trained-predictor statistical theorem
and a neural-network generalization application. Broader architecture coverage
must reuse these interfaces rather than reproduce the statistical foundations.
