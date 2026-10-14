# MLP trajectory bridge — 2026-10-14

Target: transfer the fixed-design ridge risk bound to a trained two-layer MLP
through its initial Jacobian features. This is not an MLP parameter-grid cover.
The existing one-step claims, three scaling comparisons, and blueprint wording
are unchanged.

## Implemented increment

`OpenTheory/Dynamics/MLPTrajectory.lean` defines first-layer displacement `U`
from initialization, keeps the readout fixed, and adds decay `−ηλU` to the
existing squared-loss gradient step. Thus the regularizer is centered at
initialization. The data term is still one fixed example, not a minibatch.

The initial Taylor model is
`f₀(x) + Σᵢ aᵢ σ′(w₀ᵢ·x) (Uᵢ·x)`.
Reusing the existing activation remainder proof gives, for arbitrary `U`,

    |f(W₀+U,x) − linearizedForward(U,x)|
      ≤ M Σᵢ |aᵢ| |Uᵢ·x|².

For the actual T-step regularized trajectory, assume:

- `η ≥ 0` and `|1−ηλ| ≤ 1`;
- fixed readout coordinates bounded by `A ≥ 0`;
- `|σ′| ≤ L`, with `σ′` M-Lipschitz;
- residual magnitude at every preceding training step bounded by `R ≥ 0`.

The proved preactivation displacement is at most
`T η R A L ‖x‖²`. Consequently the output remainder is at most

    M m A (T η R A L ‖x‖²)².

These are finite-horizon bounds for the concrete network update, not an
abstract assumed Taylor-error constant. They remain conditional on the
residual envelope. The constant is M, not the optional M/2 sharpening.

## Independently trained linearized trajectory

`OpenTheory/Dynamics/MLPComparison.lean` now defines separate GD on the affine
initial Taylor model and proves its predictions obey the frozen-kernel scalar
recurrence. Its weights are not equated with the nonlinear weights.

Let `X = ‖x‖²` and let `K₀ = X Σᵢ aᵢ² σ′(w₀ᵢ·x)²`. In addition to the
assumptions above, require `|1−η(K₀+λ)| ≤ 1`. Define

    D = T η R A L X
    B = M m A D²
    G = m A² L M D X.

`explicit_nonlinear_linearized_gap` proves

    |f_T(x) − independentlyTrainedLinearPrediction_T(x)|
      ≤ B + T η (B |K₀| + R G).

Here B is the nonlinear Taylor remainder and G bounds the initial/current
Jacobian correlation drift. The proof derives both from the concrete MLP
trajectory and accumulates the resulting local defects through the
nonexpansive frozen-kernel recurrence. It does not assume the final prediction
gap. Tests include zero horizon, accumulated-error constants, and equality
when the activation derivative is constant.

This closes the conditional one-example trajectory comparison. The residual
envelope R remains a premise.

## Centered ridge endpoint

`OpenTheory/Dynamics/MLPEndpoint.lean` constructs the actual initial-Jacobian
feature vector, indexed by `(neuron, input coordinate)`, and proves its squared
norm is K₀. Its one-row design is the inner product with that vector.
Reusing `Ridge.stationary_minimizes` proves that

    f_ridge = f₀ + K₀ (y−f₀)/(K₀+λ),  λ > 0,

is the prediction of a ridge minimizer on these features, with centered labels
and displacement regularization. This includes the degenerate K₀=0 case,
where the prediction remains f₀, not zero.

`linearPrediction_error` proves the exact error formula

    |f_linear,T − f_ridge| = |1−η(K₀+λ)|^T |f₀−f_ridge|.

Mathlib's geometric limit gives `linearPrediction_tendsto` when the factor is
strictly below one. `nonlinear_endpoint_gap` combines this with the checked
MLP comparison, giving the explicit deterministic network-to-ridge distance

    B + T η (B |K₀| + R G) + |1−η(K₀+λ)|^T |f₀−f_ridge|.

Only the optimization term necessarily decreases with T. The nonlinear
comparison term can grow, so no infinite-time nonlinear convergence is claimed.
The one-example ridge endpoint is now connected, but the statistical δ in
`PredictionGap.expected_mse_of_near_ridge` is not yet discharged for general
fixed designs/noise laws. A restricted one-input expected-risk guarantee is
now complete below; the general multi-input guarantee is not.

## Multi-observation initial features

`OpenTheory/Generalization/MLPFixedDesign.lean` constructs the initial-Jacobian
design for any fixed collection of inputs. `predict_flatten` proves that its
affine predictions equal the MLP Taylor model, including the initial output.
With labels centered by that output and loss normalized by the sample count,
`gd_to_ridge` reuses the existing ridge GD contraction theorem.
`ridge_expected_mse` reuses the fixed-design bias–variance bound:

    expected in-sample MSE ≤ (λ/4) ‖u★‖² + σ² degreesOfFreedom / n.

Here the signal is the initial affine model at displacement `u★`, and noise
satisfies `RidgeRisk.LabelNoise`. This is a risk bound for ridge-trained initial
features, not yet for nonlinear training or arbitrary target functions.

## Multi-input local comparison estimates

`MLPBatchBounds` now supplies the deterministic estimates for arbitrary fixed
inputs (not necessarily orthogonal). If every hidden preactivation displacement
is at most D, each input has squared norm at most E, readout coordinates are
bounded by A, and the activation derivative is M-Lipschitz, then

    ‖F(U) − (F₀ + J₀ U)‖ ≤ √n M m A D² = B_batch
    ‖J(U) − J₀‖ ≤ √n √m A M D √E = G_batch.

The second norm is the operator norm on Euclidean spaces. The proof reuses
`forward_remainder_le`, the verified flattening correspondence, and
`Dataset.observation_opNorm_le`.

`adjoint_residual_gap_le` combines them into

    ‖J(U)ᵀ(F(U)−y) − J₀ᵀ(F₀+J₀U−y)‖
      ≤ ‖J₀‖ B_batch + G_batch R,

provided `‖F(U)−y‖ ≤ R`. These are local estimates at a given displacement.
They do not yet compare independently trained multi-input trajectories, derive
their D/R envelopes, or prove a new statistical risk guarantee.

## Multi-input nonlinear trajectory and fixed-design statistical risk

`OpenTheory/Dynamics/MLPBatchTrajectory.lean` and `OpenTheory/Generalization/MLPBatchRisk.lean`
complete the full multi-observation chain:

1. **Derived multi-input envelopes (`batch_norm_le`, `batch_shift_le`, `batch_residual_le`)**:
   From `‖F₀ − y‖ ≤ S` and input energy `‖X_r‖² ≤ E`, with growth factor
   `g_n = 1 + (η/n) (√n √m A L E) (1 + √n m A L √E)`, every step `t ≤ T` of the
   normalized nonlinear batch GD trajectory satisfies
   `‖u_t‖ ≤ S g_n^T`, `|preShift(U_t)(X_r)_i| ≤ S g_n^T √E = D_batch`, and
   `‖F(u_t) − y‖ ≤ (1 + √n m A L √E) S g_n^T = R_batch`. No trajectory-level
   hypothesis is assumed.
2. **Accumulated nonlinear-to-linear trajectory comparison (`nonlinear_linear_param_gap`, `nonlinear_linear_output_gap`)**:
   Under the ridge step condition `η L_ridge² < 2λ`, the linear initial-feature
   batch step is nonexpansive, and accumulating the local adjoint-residual defect
   `(η/n)(‖J₀‖ B_batch + G_batch R_batch)` over `T` steps bounds the parameter and
   output distance between the nonlinear network and independently trained
   initial-feature ridge GD.
3. **Closed distance to the multi-input ridge prediction (`batch_trained_ridge_gap`)**:
   Combining the trajectory comparison with geometric ridge GD convergence
   (`‖v_T − u★_ridge‖ ≤ √(q^T) S / √(nλ)`) discharges the deterministic distance
   `δ = batchGap T m n η lam A L M E ‖J₀‖ q S` between the `T`-step nonlinear
   batch output and `MLPFixedDesign.ridgePrediction`.
4. **Multi-observation expected MSE (`batch_trained_expected_mse`, `batch_trained_expected_mse_of_rank`)**:
   For labels `y = F₀ + J₀ u★ + z` with white label noise `RidgeRisk.LabelNoise μ σ`
   satisfying `‖z‖ ≤ B_z` almost surely (so `S = ‖J₀ u★‖ + B_z`), the expected
   in-sample MSE of the `T`-step nonlinear network over the `n` observations is at most
   `2 ((λ/4) ‖u★‖² + σ² df(J₀, λ) / n) + 2 δ² / n ≤ 2 ((λ/4) ‖u★‖² + σ² rank(J₀) / n) + 2 δ² / n`.
