import OpenTheory.Data.Foundation
import OpenTheory.Optimization.LocalSharpnessDescent
import OpenTheory.Optimization.PreconditionedDescent
import OpenTheory.Optimization.StochasticSharpnessSGD
import OpenTheory.Optimization.MomentumAdamTrajectory
import OpenTheory.Dynamics.MLPLocalSharpness

/-!
# Variational Optimizer Optimality, Residual-Preconditioned Momentum (RPM), and Exact Regime Classification

Why do different optimizers—scalar SGD, Heavy-Ball / EMA Momentum, Diagonal Preconditioning /
RMSProp, Adam / AdamW, and Residual-Adapted Preconditioning—dominate in different training regimes?
And what is the **mathematically optimal coordinate-wise learning-rate rule** when both
anisotropic coordinate curvature `S_i`, residual-dependent curvature `S_i(R_t)`, and EMA-damped
mini-batch gradient noise `((1 - β)/(1 + β)) * σ_i^2 / B` are present simultaneously?

This module answers both questions from first principles with zero `sorry`:

1. **Coordinate-Wise Residual-Coupled Local Sharpness (`coordResidualSharpness`,
   `coordResidualSharpness_pos`, `coordResidualSharpness_mono_in_residual`,
   `coordResidualSharpness_strict_mono_in_residual`, `mlp_residual_drop_strictly_flattens_landscape`)**:
   Generalizing the scalar derived sharpness `mlpLocalSharpness` from `MLPLocalSharpness` to
   individual coordinates `i : Fin p`, the local sharpness of coordinate `i` at prediction residual
   norm `R = ‖F(u) - y‖` decomposes into a Gauss-Newton / Fisher component `lam + J_i^2` plus a
   residual-coupled Hessian remainder `(2 * R / √n) * C_i`:
   `coordResidualSharpness lam J_coord C_coord n R i = lam + (J_coord i)^2 + 2 * R * C_coord i / √n`.
   As training reduces the prediction residual `R_t < R_0`, the local loss landscape **strictly
   flattens** (`coordResidualSharpness ... R_t i < coordResidualSharpness ... R_0 i`).

2. **Exact Variational Optimality of Residual-Preconditioned Momentum (RPM)
   (`dampedCoordVar`, `totalCoordMoment`, `rpmOptimalStep`, `rpmResidualStep`,
   `optimizerNetProgress_eq_coord_sum`, `optimizerNetProgress_rpmOptimalStep_eq`,
   `rpm_variational_sub_eq_sum_sq`, `rpm_variational_global_optimality`)**:
   For the general stochastic momentum-preconditioned objective progress functional
   `optimizerNetProgress η S g β σ_coord B` from `MomentumAdamTrajectory`, completing the square
   coordinate-by-coordinate proves that the **Residual-Preconditioned Momentum (RPM) step**
   `rpmOptimalStep S g β σ_coord B i = (g i)^2 / (S i * ((g i)^2 + ((1 - β)/(1 + β)) * σ_i^2 / B))`
   is the **unique global maximizer** over all coordinate step vectors `η : Fin p → ℝ`, with exact
   variational suboptimality identity
   `optimizerNetProgress η* - optimizerNetProgress η = ∑ i, (S_i * M_i / 2) * (η_i - η*_i)^2 ≥ 0`
   and maximum achievable progress `∑ i, ((g i)^2)^2 / (2 * S i * totalCoordMoment g β σ_coord B i)`.

3. **Systematic Regime Classification: Why Each Optimizer Is Optimal in Its Exact Regime
   (`rpmOptimalStep_zero_noise_eq_optimalDiagStep`,
   `rpmOptimalStep_isotropic_zero_noise_eq_scalar`,
   `rpm_strict_speedup_from_residual_drop`,
   `rpm_momentum_strictly_increases_optimal_progress`)**:
   - **Regime I (Noise-Free Anisotropic Regime, `σ_coord = 0`)**: `rpmOptimalStep` reduces
     identically to `optimalDiagStep S i = 1 / S i` (diagonal Hessian / Gauss-Newton preconditioning).
   - **Regime II (Noise-Free Isotropic Regime, `S_i = S_0` and `σ_coord = 0`)**: `rpmOptimalStep`
     reduces to constant scalar GD `1 / S_0` and achieves `‖g‖^2 / (2 * S_0)`.
   - **Regime III (Residual-Decreasing Trajectory Regime, `R_t < R_0` so `S_t < S_0`)**:
     Static-sharpness preconditioners remain stuck at conservative step sizes tuned to initial
     curvature `S_0`, whereas adapting to the flattened curvature `S_t` yields a strictly larger
     variational progress ceiling AND strictly beats `rpmOptimalStep S_0` evaluated on the true
     landscape `S_t`.
   - **Regime IV (Stochastic Mini-Batch Regime with EMA Momentum `0 < β < 1`)**:
     EMA momentum strictly shrinks `dampedCoordVar` relative to memoryless SGD (`β = 0`), strictly
     increasing the variational maximum progress across all active noisy coordinates.

4. **Dynamic Residual-Adapted PL Contraction Acceleration
   (`rpm_dynamic_pl_contraction_strict_improvement`, `rpm_residual_drop_pl_contraction_strict_improvement`)**:
   When training reduces the prediction residual so that local sharpness drops from `S_init` to
   `S_curr < S_init`, stepping with the residual-adapted learning rate `1 / S_curr` yields a
   strictly smaller (faster) per-step Polyak-Łojasiewicz contraction factor
   `1 - μ / S_curr < 1 - μ / S_init`.
-/

noncomputable section

open scoped BigOperators RealInnerProductSpace

namespace OpenTheory.ResidualPreconditionedMomentum

open OpenTheory.LocalSharpnessDescent
open OpenTheory.PreconditionedDescent
open OpenTheory.StochasticSharpnessSGD
open OpenTheory.MomentumAdamTrajectory
open OpenTheory.MLPLocalSharpness

variable {p : ℕ}

/-! ## 1. Coordinate-Wise Residual-Coupled Local Sharpness -/

/-- **Coordinate-Wise Residual-Coupled Local Sharpness (`coordResidualSharpness`)**:
At regularizer `lam`, coordinate-wise Jacobian / Lipschitz sensitivities `J_coord : Fin p → ℝ`,
coordinate-wise second-order activation curvature sensitivities `C_coord : Fin p → ℝ`, sample size
`n : ℕ`, and current batch prediction residual norm `R = ‖F(u) - y‖`, the local sharpness of
coordinate `i : Fin p` is
`lam + (J_coord i) ^ 2 + 2 * R * C_coord i / Real.sqrt (n : ℝ)`. -/
def coordResidualSharpness (lam : ℝ) (J_coord C_coord : Fin p → ℝ) (n : ℕ) (R : ℝ)
    (i : Fin p) : ℝ :=
  lam + (J_coord i) ^ 2 + 2 * R * C_coord i / Real.sqrt (n : ℝ)

/-- Lower bound `lam ≤ coordResidualSharpness` when `0 ≤ R` and `0 ≤ C_coord i`. -/
theorem lam_le_coordResidualSharpness {lam R : ℝ} (J_coord C_coord : Fin p → ℝ) (n : ℕ)
    (i : Fin p) (hR : 0 ≤ R) (hC : 0 ≤ C_coord i) :
    lam ≤ coordResidualSharpness lam J_coord C_coord n R i := by
  unfold coordResidualSharpness
  have hsq : 0 ≤ (J_coord i) ^ 2 := sq_nonneg (J_coord i)
  have hrem : 0 ≤ 2 * R * C_coord i / Real.sqrt (n : ℝ) := by positivity
  linarith

/-- **Strict Positivity of Coordinate Residual Sharpness (`coordResidualSharpness_pos`)**:
Whenever `0 < lam`, `0 ≤ R`, and `0 ≤ C_coord i`, the coordinate local sharpness is strictly
positive. -/
theorem coordResidualSharpness_pos {lam R : ℝ} (J_coord C_coord : Fin p → ℝ) (n : ℕ)
    (i : Fin p) (hlam : 0 < lam) (hR : 0 ≤ R) (hC : 0 ≤ C_coord i) :
    0 < coordResidualSharpness lam J_coord C_coord n R i :=
  hlam.trans_le (lam_le_coordResidualSharpness J_coord C_coord n i hR hC)

/-- **Exact Sharpness Reduction Identity from Residual Drop (`coordResidualSharpness_sub_eq`)**:
Reducing the prediction residual from `R₂` to `R₁` reduces coordinate sharpness by the exact amount
`2 * (R₂ - R₁) * C_coord i / √n`. -/
theorem coordResidualSharpness_sub_eq (lam : ℝ) (J_coord C_coord : Fin p → ℝ) (n : ℕ)
    (R₁ R₂ : ℝ) (i : Fin p) :
    coordResidualSharpness lam J_coord C_coord n R₂ i -
        coordResidualSharpness lam J_coord C_coord n R₁ i =
      2 * (R₂ - R₁) * C_coord i / Real.sqrt (n : ℝ) := by
  unfold coordResidualSharpness
  ring

/-- **Monotonicity of Coordinate Sharpness in the Prediction Residual
(`coordResidualSharpness_mono_in_residual`)**:
When `R₁ ≤ R₂` and `0 ≤ C_coord i`, coordinate sharpness at `R₁` is at most that at `R₂`. -/
theorem coordResidualSharpness_mono_in_residual (lam : ℝ) (J_coord C_coord : Fin p → ℝ) (n : ℕ)
    {R₁ R₂ : ℝ} (i : Fin p) (hR : R₁ ≤ R₂) (hC : 0 ≤ C_coord i) :
    coordResidualSharpness lam J_coord C_coord n R₁ i ≤
      coordResidualSharpness lam J_coord C_coord n R₂ i := by
  have hdiff := coordResidualSharpness_sub_eq lam J_coord C_coord n R₁ R₂ i
  have hsub : 0 ≤ R₂ - R₁ := sub_nonneg.mpr hR
  have hnonneg : 0 ≤ 2 * (R₂ - R₁) * C_coord i / Real.sqrt (n : ℝ) := by positivity
  linarith

/-- **Strict Monotonicity of Coordinate Sharpness in the Prediction Residual
(`coordResidualSharpness_strict_mono_in_residual`)**:
When `R₁ < R₂`, `0 < C_coord i`, and `0 < n`, reducing the prediction residual from `R₂` to `R₁`
**strictly flattens** the local curvature of coordinate `i`:
`coordResidualSharpness lam J_coord C_coord n R₁ i < coordResidualSharpness lam J_coord C_coord n R₂ i`. -/
theorem coordResidualSharpness_strict_mono_in_residual (lam : ℝ) (J_coord C_coord : Fin p → ℝ)
    {n : ℕ} {R₁ R₂ : ℝ} (i : Fin p)
    (hR : R₁ < R₂) (hC : 0 < C_coord i) (hn : 0 < n) :
    coordResidualSharpness lam J_coord C_coord n R₁ i <
      coordResidualSharpness lam J_coord C_coord n R₂ i := by
  have hdiff := coordResidualSharpness_sub_eq lam J_coord C_coord n R₁ R₂ i
  have hn0 : (0 : ℝ) < n := Nat.cast_pos.mpr hn
  have hsqrt : 0 < Real.sqrt (n : ℝ) := Real.sqrt_pos.mpr hn0
  have hsub : 0 < R₂ - R₁ := sub_pos.mpr hR
  have hpos : 0 < 2 * (R₂ - R₁) * C_coord i / Real.sqrt (n : ℝ) := by positivity
  linarith

/-- **Connection to Derived MLP Local Sharpness (`coordResidualSharpness_eq_mlpLocalSharpness`)**:
When `J_coord i = m * A * L * √E` and `C_coord i = M * m * A * E` (`0 < n`, `0 ≤ E`, `0 ≤ m * A * L`),
`coordResidualSharpness` recovers the exact derived two-layer MLP local sharpness
`mlpLocalSharpness m n lam A L M E R` from `OpenTheory.MLPLocalSharpness`. -/
theorem coordResidualSharpness_eq_mlpLocalSharpness {m n : ℕ} (hn : 0 < n)
    (lam A L M E R : ℝ) (hE : 0 ≤ E) (i : Fin p) :
    coordResidualSharpness lam
        (fun _ => (m : ℝ) * A * L * Real.sqrt E)
        (fun _ => M * (m : ℝ) * A * E) n R i =
      mlpLocalSharpness m n lam A L M E R := by
  rw [mlpLocalSharpness_eq_of_pos hn lam A L M E R hE]
  unfold coordResidualSharpness
  have hEsq : (Real.sqrt E) ^ 2 = E := Real.sq_sqrt hE
  have hJ : ((m : ℝ) * A * L * Real.sqrt E) ^ 2 =
      (m : ℝ) ^ 2 * A ^ 2 * L ^ 2 * E := by
    calc
      ((m : ℝ) * A * L * Real.sqrt E) ^ 2
        = (m : ℝ) ^ 2 * A ^ 2 * L ^ 2 * (Real.sqrt E) ^ 2 := by ring
      _ = (m : ℝ) ^ 2 * A ^ 2 * L ^ 2 * E := by rw [hEsq]
  rw [hJ]
  ring

/-- **Physical Landscape Flattening Along Training (`mlp_residual_drop_strictly_flattens_landscape`)**:
As training reduces the empirical prediction residual from initial level `R_0` to `R_t < R_0`
(`0 < m`, `0 < n`, `0 < A`, `0 < M`, `0 < E`), the derived two-layer MLP local sharpness
`mlpLocalSharpness m n lam A L M E R` **strictly decreases** (`S(R_t) < S(R_0)`). -/
theorem mlp_residual_drop_strictly_flattens_landscape {m n : ℕ}
    (lam A L M E R_t R_0 : ℝ)
    (hm : 0 < m) (hn : 0 < n) (hA : 0 < A) (hM : 0 < M) (hE : 0 < E)
    (hR : R_t < R_0) :
    mlpLocalSharpness m n lam A L M E R_t <
      mlpLocalSharpness m n lam A L M E R_0 := by
  have hm0 : (0 : ℝ) < m := Nat.cast_pos.mpr hm
  have hn0 : (0 : ℝ) < n := Nat.cast_pos.mpr hn
  have hC : 0 < mlpOutputCurv m n A M E := by
    unfold mlpOutputCurv
    have hsqrt : 0 < Real.sqrt (n : ℝ) := Real.sqrt_pos.mpr hn0
    positivity
  unfold mlpLocalSharpness
  have hsub : 0 < R_0 - R_t := sub_pos.mpr hR
  have hdiff :
      (lam + (n : ℝ)⁻¹ * (mlpOutputLip m n A L E ^ 2 + 2 * R_0 * mlpOutputCurv m n A M E)) -
        (lam + (n : ℝ)⁻¹ * (mlpOutputLip m n A L E ^ 2 + 2 * R_t * mlpOutputCurv m n A M E)) =
      (n : ℝ)⁻¹ * (2 * (R_0 - R_t) * mlpOutputCurv m n A M E) := by ring
  have hpos : 0 < (n : ℝ)⁻¹ * (2 * (R_0 - R_t) * mlpOutputCurv m n A M E) := by positivity
  linarith

/-! ## 2. Exact Variational Optimality of Residual-Preconditioned Momentum (RPM) -/

/-- **EMA-Damped Coordinate Gradient Variance (`dampedCoordVar`)**:
The stationary variance of the EMA first moment in coordinate `i` under mini-batch size `B` and
momentum parameter `β`:
`((1 - β) / (1 + β)) * ((σ_coord i) ^ 2 / B)`. -/
def dampedCoordVar (β : ℝ) (σ_coord : Fin p → ℝ) (B : ℝ) (i : Fin p) : ℝ :=
  ((1 - β) / (1 + β)) * ((σ_coord i) ^ 2 / B)

/-- Non-negativity of `dampedCoordVar` when `0 ≤ β ≤ 1` and `0 < B`. -/
theorem dampedCoordVar_nonneg {β B : ℝ} (σ_coord : Fin p → ℝ) (i : Fin p)
    (hβ0 : 0 ≤ β) (hβ1 : β ≤ 1) (hB : 0 < B) :
    0 ≤ dampedCoordVar β σ_coord B i := by
  unfold dampedCoordVar
  have h1 : 0 ≤ 1 - β := sub_nonneg.mpr hβ1
  have h2 : 0 < 1 + β := by linarith
  positivity

/-- At zero noise `σ_coord = 0`, `dampedCoordVar` vanishes. -/
@[simp]
theorem dampedCoordVar_zero_noise (β B : ℝ) (i : Fin p) :
    dampedCoordVar β (fun _ => 0) B i = 0 := by
  simp [dampedCoordVar]

/-- **Total Coordinate Second Moment (`totalCoordMoment`)**:
The effective second moment of the EMA momentum direction in coordinate `i`, combining the
deterministic squared gradient signal `(g i) ^ 2` with the EMA-damped stochastic variance
`dampedCoordVar β σ_coord B i`:
`(g i) ^ 2 + ((1 - β) / (1 + β)) * ((σ_coord i) ^ 2 / B)`. -/
def totalCoordMoment (g : Parameters p) (β : ℝ) (σ_coord : Fin p → ℝ) (B : ℝ) (i : Fin p) : ℝ :=
  (g i) ^ 2 + dampedCoordVar β σ_coord B i

/-- Non-negativity of `totalCoordMoment` when `0 ≤ β ≤ 1` and `0 < B`. -/
theorem totalCoordMoment_nonneg (g : Parameters p) {β B : ℝ} (σ_coord : Fin p → ℝ) (i : Fin p)
    (hβ0 : 0 ≤ β) (hβ1 : β ≤ 1) (hB : 0 < B) :
    0 ≤ totalCoordMoment g β σ_coord B i := by
  unfold totalCoordMoment
  have h1 := dampedCoordVar_nonneg σ_coord i hβ0 hβ1 hB
  positivity

/-- Strict positivity of `totalCoordMoment` whenever `g i ≠ 0` (with `0 ≤ β ≤ 1` and `0 < B`). -/
theorem totalCoordMoment_pos_of_grad_ne_zero (g : Parameters p) {β B : ℝ}
    (σ_coord : Fin p → ℝ) (i : Fin p)
    (hβ0 : 0 ≤ β) (hβ1 : β ≤ 1) (hB : 0 < B) (hgi : g i ≠ 0) :
    0 < totalCoordMoment g β σ_coord B i := by
  unfold totalCoordMoment
  have hg2 : 0 < (g i) ^ 2 := sq_pos_of_ne_zero hgi
  have hd : 0 ≤ dampedCoordVar β σ_coord B i := dampedCoordVar_nonneg σ_coord i hβ0 hβ1 hB
  linarith

/-- **Globally Optimal Variational Coordinate Step Size (`rpmOptimalStep`)**:
The Residual-Preconditioned Momentum (RPM) coordinate learning rate that uniquely maximizes
`optimizerNetProgress η S g β σ_coord B`:
`η*_i = (g i)^2 / (S i * totalCoordMoment g β σ_coord B i)`. -/
def rpmOptimalStep (S : Fin p → ℝ) (g : Parameters p) (β : ℝ)
    (σ_coord : Fin p → ℝ) (B : ℝ) (i : Fin p) : ℝ :=
  (g i) ^ 2 / (S i * totalCoordMoment g β σ_coord B i)

/-- **Residual-Coupled RPM Coordinate Learning Rate (`rpmResidualStep`)**:
Specialization of `rpmOptimalStep` to the residual-dependent coordinate sharpness profile
`coordResidualSharpness lam J_coord C_coord n R`. -/
def rpmResidualStep (lam : ℝ) (J_coord C_coord : Fin p → ℝ) (n : ℕ) (R : ℝ)
    (g : Parameters p) (β : ℝ) (σ_coord : Fin p → ℝ) (B : ℝ) (i : Fin p) : ℝ :=
  rpmOptimalStep (coordResidualSharpness lam J_coord C_coord n R) g β σ_coord B i

/-- **Coordinate-Sum Decomposition of `optimizerNetProgress` (`optimizerNetProgress_eq_coord_sum`)**:
`optimizerNetProgress η S g β σ_coord B` decomposes into an exact coordinate sum of quadratic
functions of `η i`:
`∑ i : Fin p, (η i * (g i) ^ 2 - (S i / 2) * (η i) ^ 2 * totalCoordMoment g β σ_coord B i)`. -/
theorem optimizerNetProgress_eq_coord_sum (η S : Fin p → ℝ) (g : Parameters p) (β : ℝ)
    (σ_coord : Fin p → ℝ) (B : ℝ) :
    optimizerNetProgress η S g β σ_coord B =
      ∑ i : Fin p, (η i * (g i) ^ 2 -
        (S i / 2) * (η i) ^ 2 * totalCoordMoment g β σ_coord B i) := by
  unfold optimizerNetProgress coordProgress totalCoordMoment dampedCoordVar
  rw [Finset.mul_sum, ← Finset.sum_sub_distrib]
  refine Finset.sum_congr rfl fun i _ => ?_
  ring

/-- **Closed-Form Maximum Net Progress Identity (`optimizerNetProgress_rpmOptimalStep_eq`)**:
When `∀ i, 0 < S i` and `∀ i, 0 < totalCoordMoment g β σ_coord B i`, the variational optimal step
`rpmOptimalStep S g β σ_coord B` achieves expected net progress
`∑ i : Fin p, ((g i) ^ 2) ^ 2 / (2 * S i * totalCoordMoment g β σ_coord B i)`. -/
theorem optimizerNetProgress_rpmOptimalStep_eq (S : Fin p → ℝ) (g : Parameters p) (β : ℝ)
    (σ_coord : Fin p → ℝ) (B : ℝ)
    (hS : ∀ i, 0 < S i) (hM : ∀ i, 0 < totalCoordMoment g β σ_coord B i) :
    optimizerNetProgress (rpmOptimalStep S g β σ_coord B) S g β σ_coord B =
      ∑ i : Fin p, ((g i) ^ 2) ^ 2 / (2 * S i * totalCoordMoment g β σ_coord B i) := by
  rw [optimizerNetProgress_eq_coord_sum]
  unfold rpmOptimalStep
  refine Finset.sum_congr rfl fun i _ => ?_
  have hSi : S i ≠ 0 := ne_of_gt (hS i)
  have hMi : totalCoordMoment g β σ_coord B i ≠ 0 := ne_of_gt (hM i)
  field_simp [hSi, hMi]
  ring

/-- **Exact Variational Completion-of-Squares Theorem (`rpm_variational_sub_eq_sum_sq`)**:
For **any** candidate coordinate learning-rate vector `η : Fin p → ℝ`, the suboptimality gap
between `rpmOptimalStep` and `η` on `optimizerNetProgress` is an exact weighted sum of squares:
`optimizerNetProgress (rpmOptimalStep S g β σ_coord B) S g β σ_coord B - optimizerNetProgress η S g β σ_coord B = ∑ i, (S i * totalCoordMoment g β σ_coord B i / 2) * (η i - rpmOptimalStep S g β σ_coord B i) ^ 2`. -/
theorem rpm_variational_sub_eq_sum_sq (η S : Fin p → ℝ) (g : Parameters p) (β : ℝ)
    (σ_coord : Fin p → ℝ) (B : ℝ)
    (hS : ∀ i, 0 < S i) (hM : ∀ i, 0 < totalCoordMoment g β σ_coord B i) :
    optimizerNetProgress (rpmOptimalStep S g β σ_coord B) S g β σ_coord B -
        optimizerNetProgress η S g β σ_coord B =
      ∑ i : Fin p,
        (S i * totalCoordMoment g β σ_coord B i / 2) *
          (η i - rpmOptimalStep S g β σ_coord B i) ^ 2 := by
  rw [optimizerNetProgress_rpmOptimalStep_eq S g β σ_coord B hS hM,
    optimizerNetProgress_eq_coord_sum, ← Finset.sum_sub_distrib]
  unfold rpmOptimalStep
  refine Finset.sum_congr rfl fun i _ => ?_
  have hSi : S i ≠ 0 := ne_of_gt (hS i)
  have hMi : totalCoordMoment g β σ_coord B i ≠ 0 := ne_of_gt (hM i)
  field_simp [hSi, hMi]
  ring

/-- **Global Variational Optimality of Residual-Preconditioned Momentum
(`rpm_variational_global_optimality`)**:
Under positive coordinate sharpness `∀ i, 0 < S i` and positive second moment
`∀ i, 0 < totalCoordMoment g β σ_coord B i`, `rpmOptimalStep S g β σ_coord B` maximizes
`optimizerNetProgress η S g β σ_coord B` over **all** coordinate learning-rate vectors
`η : Fin p → ℝ`. -/
theorem rpm_variational_global_optimality (η S : Fin p → ℝ) (g : Parameters p) (β : ℝ)
    (σ_coord : Fin p → ℝ) (B : ℝ)
    (hS : ∀ i, 0 < S i) (hM : ∀ i, 0 < totalCoordMoment g β σ_coord B i) :
    optimizerNetProgress η S g β σ_coord B ≤
      optimizerNetProgress (rpmOptimalStep S g β σ_coord B) S g β σ_coord B := by
  have hdiff := rpm_variational_sub_eq_sum_sq η S g β σ_coord B hS hM
  have hnonneg : 0 ≤ ∑ i : Fin p,
      (S i * totalCoordMoment g β σ_coord B i / 2) *
        (η i - rpmOptimalStep S g β σ_coord B i) ^ 2 := by
    refine Finset.sum_nonneg fun i _ => ?_
    have hSi : 0 < S i := hS i
    have hMi : 0 < totalCoordMoment g β σ_coord B i := hM i
    positivity
  linarith

/-- **Strict Variational Dominance Over Any Distinct Step Vector
(`rpm_variational_strict_optimality`)**:
If `η : Fin p → ℝ` differs from `rpmOptimalStep S g β σ_coord B` in at least one coordinate
`i₀ : Fin p` (`η i₀ ≠ rpmOptimalStep S g β σ_coord B i₀`), then `rpmOptimalStep` achieves
**strictly greater** expected net progress than `η`. -/
theorem rpm_variational_strict_optimality (η S : Fin p → ℝ) (g : Parameters p) (β : ℝ)
    (σ_coord : Fin p → ℝ) (B : ℝ)
    (hS : ∀ i, 0 < S i) (hM : ∀ i, 0 < totalCoordMoment g β σ_coord B i)
    (i₀ : Fin p) (hne : η i₀ ≠ rpmOptimalStep S g β σ_coord B i₀) :
    optimizerNetProgress η S g β σ_coord B <
      optimizerNetProgress (rpmOptimalStep S g β σ_coord B) S g β σ_coord B := by
  have hdiff := rpm_variational_sub_eq_sum_sq η S g β σ_coord B hS hM
  have hterm_nonneg : ∀ i ∈ Finset.univ,
      0 ≤ (S i * totalCoordMoment g β σ_coord B i / 2) *
        (η i - rpmOptimalStep S g β σ_coord B i) ^ 2 := by
    intro i _
    have hSi : 0 < S i := hS i
    have hMi : 0 < totalCoordMoment g β σ_coord B i := hM i
    positivity
  have hterm_pos : 0 < (S i₀ * totalCoordMoment g β σ_coord B i₀ / 2) *
      (η i₀ - rpmOptimalStep S g β σ_coord B i₀) ^ 2 := by
    have hSi₀ : 0 < S i₀ := hS i₀
    have hMi₀ : 0 < totalCoordMoment g β σ_coord B i₀ := hM i₀
    have hsq : 0 < (η i₀ - rpmOptimalStep S g β σ_coord B i₀) ^ 2 :=
      sq_pos_of_ne_zero (sub_ne_zero.mpr hne)
    positivity
  have hsum_pos := Finset.sum_pos' hterm_nonneg ⟨i₀, Finset.mem_univ i₀, hterm_pos⟩
  linarith

/-! ## 3. Systematic Regime Classification: Why Each Optimizer Is Optimal in Its Regime -/

/-- **Regime I: Noise-Free Anisotropic Regime (`rpmOptimalStep_zero_noise_eq_optimalDiagStep`)**:
When mini-batch gradient noise vanishes (`σ_coord = 0`) and `g i ≠ 0`, the general variational
step `rpmOptimalStep` reduces **identically** to the optimal diagonal sharpness preconditioner
`optimalDiagStep S i = 1 / S i`. -/
theorem rpmOptimalStep_zero_noise_eq_optimalDiagStep (S : Fin p → ℝ) (g : Parameters p)
    (β B : ℝ) (i : Fin p) (hgi : g i ≠ 0) :
    rpmOptimalStep S g β (fun _ => 0) B i = optimalDiagStep S i ∧
      optimalDiagStep S i = 1 / S i := by
  refine ⟨?_, rfl⟩
  unfold rpmOptimalStep optimalDiagStep totalCoordMoment dampedCoordVar
  have hg2 : (g i) ^ 2 ≠ 0 := pow_ne_zero 2 hgi
  simp only [ne_eq, OfNat.ofNat_ne_zero, not_false_eq_true, zero_pow, zero_div, mul_zero, add_zero]
  simpa only [one_mul] using mul_div_mul_right (1 : ℝ) (S i) hg2

/-- **Regime II: Noise-Free Isotropic Regime (`rpmOptimalStep_isotropic_zero_noise_eq_scalar`)**:
When curvature is isotropic (`S = fun _ => S_0`, `0 < S_0`) and noise vanishes (`σ_coord = 0`),
`rpmOptimalStep` reduces to constant scalar GD `1 / S_0` on every active coordinate (`g i ≠ 0`),
and `optimizerNetProgress` with step `1 / S_0` achieves `‖g‖ ^ 2 / (2 * S_0)`. -/
theorem rpmOptimalStep_isotropic_zero_noise_eq_scalar (S_0 : ℝ) (g : Parameters p)
    (β B : ℝ) (hS0 : 0 < S_0) :
    (∀ i : Fin p, g i ≠ 0 →
      rpmOptimalStep (fun _ => S_0) g β (fun _ => 0) B i = 1 / S_0) ∧
    optimizerNetProgress (fun _ => 1 / S_0) (fun _ => S_0) g β (fun _ => 0) B =
      ‖g‖ ^ 2 / (2 * S_0) := by
  refine ⟨fun i hgi => (rpmOptimalStep_zero_noise_eq_optimalDiagStep (fun _ => S_0) g β B i hgi).1,
    ?_⟩
  have hpen : optimizerNoisePenalty (p := p) (fun _ : Fin p => 1 / S_0)
      (fun _ : Fin p => S_0) (fun _ : Fin p => 0) B = 0 := by
    simp [optimizerNoisePenalty]
  rw [optimizerNetProgress_eq_sub_penalty, hpen, mul_zero, sub_zero]
  exact coordProgress_scalar_max_eq S_0 g (ne_of_gt hS0)

/-- **Regime III (Part A): Exact Variational Progress Gain from Residual / Sharpness Drop
(`rpm_progress_gap_from_sharpness_drop_eq`)**:
When coordinate sharpness drops from `S_0` to `S_t`, the gain in maximum achievable net progress
is the exact coordinate sum
`∑ i : Fin p, ((S_0 i - S_t i) * ((g i) ^ 2) ^ 2) / (2 * S_t i * S_0 i * totalCoordMoment g β σ_coord B i)`. -/
theorem rpm_progress_gap_from_sharpness_drop_eq (S_t S_0 : Fin p → ℝ) (g : Parameters p)
    (β : ℝ) (σ_coord : Fin p → ℝ) (B : ℝ)
    (hSt : ∀ i, 0 < S_t i) (hS0 : ∀ i, 0 < S_0 i)
    (hM : ∀ i, 0 < totalCoordMoment g β σ_coord B i) :
    optimizerNetProgress (rpmOptimalStep S_t g β σ_coord B) S_t g β σ_coord B -
        optimizerNetProgress (rpmOptimalStep S_0 g β σ_coord B) S_0 g β σ_coord B =
      ∑ i : Fin p,
        ((S_0 i - S_t i) * ((g i) ^ 2) ^ 2) /
          (2 * S_t i * S_0 i * totalCoordMoment g β σ_coord B i) := by
  rw [optimizerNetProgress_rpmOptimalStep_eq S_t g β σ_coord B hSt hM,
    optimizerNetProgress_rpmOptimalStep_eq S_0 g β σ_coord B hS0 hM,
    ← Finset.sum_sub_distrib]
  refine Finset.sum_congr rfl fun i _ => ?_
  have hSti : S_t i ≠ 0 := ne_of_gt (hSt i)
  have hS0i : S_0 i ≠ 0 := ne_of_gt (hS0 i)
  have hMi : totalCoordMoment g β σ_coord B i ≠ 0 := ne_of_gt (hM i)
  field_simp [hSti, hS0i, hMi]

/-- **Regime III (Part B): Residual-Adapted RPM Strictly Beats Static-Sharpness Preconditioning
(`rpm_strict_speedup_from_residual_drop`)**:
Suppose training reduces the prediction residual so that coordinate sharpness drops from initial
profile `S_0` to `0 < S_t i ≤ S_0 i` for all `i`, with strict flattening `S_t i₀ < S_0 i₀` on at
least one active coordinate `i₀` (`g i₀ ≠ 0`). Then:
1. The maximum achievable progress at the flattened curvature `S_t` is **strictly greater** than
   at the initial curvature `S_0`:
   `optimizerNetProgress (rpmOptimalStep S_0 g β σ_coord B) S_0 g β σ_coord B < optimizerNetProgress (rpmOptimalStep S_t g β σ_coord B) S_t g β σ_coord B`.
2. Furthermore, on the true current landscape `S_t`, stepping with the residual-adapted rate
   `rpmOptimalStep S_t` **strictly beats** stepping with the static-sharpness rate
   `rpmOptimalStep S_0`:
   `optimizerNetProgress (rpmOptimalStep S_0 g β σ_coord B) S_t g β σ_coord B < optimizerNetProgress (rpmOptimalStep S_t g β σ_coord B) S_t g β σ_coord B`. -/
theorem rpm_strict_speedup_from_residual_drop (S_t S_0 : Fin p → ℝ) (g : Parameters p)
    (β : ℝ) (σ_coord : Fin p → ℝ) (B : ℝ)
    (hSt : ∀ i, 0 < S_t i) (hS_le : ∀ i, S_t i ≤ S_0 i)
    (hM : ∀ i, 0 < totalCoordMoment g β σ_coord B i)
    (i₀ : Fin p) (hi₀_lt : S_t i₀ < S_0 i₀) (hgi₀ : g i₀ ≠ 0) :
    optimizerNetProgress (rpmOptimalStep S_0 g β σ_coord B) S_0 g β σ_coord B <
        optimizerNetProgress (rpmOptimalStep S_t g β σ_coord B) S_t g β σ_coord B ∧
      optimizerNetProgress (rpmOptimalStep S_0 g β σ_coord B) S_t g β σ_coord B <
        optimizerNetProgress (rpmOptimalStep S_t g β σ_coord B) S_t g β σ_coord B := by
  have hS0 : ∀ i, 0 < S_0 i := fun i => (hSt i).trans_le (hS_le i)
  refine ⟨?_, ?_⟩
  · have hgap := rpm_progress_gap_from_sharpness_drop_eq S_t S_0 g β σ_coord B hSt hS0 hM
    have hterm_nonneg : ∀ i ∈ Finset.univ,
        0 ≤ ((S_0 i - S_t i) * ((g i) ^ 2) ^ 2) /
          (2 * S_t i * S_0 i * totalCoordMoment g β σ_coord B i) := by
      intro i _
      have hSti : 0 < S_t i := hSt i
      have hS0i : 0 < S_0 i := hS0 i
      have hMi : 0 < totalCoordMoment g β σ_coord B i := hM i
      have hsub : 0 ≤ S_0 i - S_t i := sub_nonneg.mpr (hS_le i)
      positivity
    have hterm_pos : 0 < ((S_0 i₀ - S_t i₀) * ((g i₀) ^ 2) ^ 2) /
        (2 * S_t i₀ * S_0 i₀ * totalCoordMoment g β σ_coord B i₀) := by
      have hSti₀ : 0 < S_t i₀ := hSt i₀
      have hS0i₀ : 0 < S_0 i₀ := hS0 i₀
      have hMi₀ : 0 < totalCoordMoment g β σ_coord B i₀ := hM i₀
      have hsub : 0 < S_0 i₀ - S_t i₀ := sub_pos.mpr hi₀_lt
      have hg4 : 0 < ((g i₀) ^ 2) ^ 2 := by
        have hg2 : 0 < (g i₀) ^ 2 := sq_pos_of_ne_zero hgi₀
        positivity
      positivity
    have hsum_pos := Finset.sum_pos' hterm_nonneg ⟨i₀, Finset.mem_univ i₀, hterm_pos⟩
    linarith
  · apply rpm_variational_strict_optimality (rpmOptimalStep S_0 g β σ_coord B) S_t g β σ_coord B
      hSt hM i₀
    unfold rpmOptimalStep
    intro heq
    have hSti₀ : S_t i₀ ≠ 0 := ne_of_gt (hSt i₀)
    have hS0i₀ : S_0 i₀ ≠ 0 := ne_of_gt (hS0 i₀)
    have hMi₀ : totalCoordMoment g β σ_coord B i₀ ≠ 0 := ne_of_gt (hM i₀)
    have hg2 : (g i₀) ^ 2 ≠ 0 := pow_ne_zero 2 hgi₀
    have h_eq_S : S_t i₀ = S_0 i₀ := by
      have h1 : (g i₀) ^ 2 * (S_t i₀ * totalCoordMoment g β σ_coord B i₀) =
          (g i₀) ^ 2 * (S_0 i₀ * totalCoordMoment g β σ_coord B i₀) := by
        exact (div_eq_div_iff (mul_ne_zero hS0i₀ hMi₀) (mul_ne_zero hSti₀ hMi₀)).mp heq
      have h2 : S_t i₀ * totalCoordMoment g β σ_coord B i₀ =
          S_0 i₀ * totalCoordMoment g β σ_coord B i₀ :=
        mul_left_cancel₀ hg2 h1
      exact mul_right_cancel₀ hMi₀ h2
    exact (ne_of_lt hi₀_lt) h_eq_S

/-- **Explicit Residual-Coupled Specialization of `rpm_strict_speedup_from_residual_drop`
(`rpmResidualStep_strict_speedup`)**:
When the prediction residual drops (`0 ≤ R_t < R_0`, `0 < lam`, `0 < n`), `∀ i, 0 ≤ C_coord i`,
and at least one active coordinate `i₀` (`g i₀ ≠ 0`) has positive curvature sensitivity
`0 < C_coord i₀`, the residual-coupled step `rpmResidualStep ... R_t` strictly beats the static
initial-residual step `rpmResidualStep ... R_0` on the current landscape `S(R_t)`. -/
theorem rpmResidualStep_strict_speedup {lam R_t R_0 : ℝ}
    (J_coord C_coord : Fin p → ℝ) {n : ℕ} (g : Parameters p)
    (β : ℝ) (σ_coord : Fin p → ℝ) (B : ℝ)
    (hlam : 0 < lam) (hn : 0 < n) (hRt : 0 ≤ R_t) (hR_lt : R_t < R_0)
    (hC : ∀ i, 0 ≤ C_coord i)
    (hM : ∀ i, 0 < totalCoordMoment g β σ_coord B i)
    (i₀ : Fin p) (hCi₀ : 0 < C_coord i₀) (hgi₀ : g i₀ ≠ 0) :
    let S_t := coordResidualSharpness lam J_coord C_coord n R_t
    let S_0 := coordResidualSharpness lam J_coord C_coord n R_0
    optimizerNetProgress (rpmResidualStep lam J_coord C_coord n R_0 g β σ_coord B) S_0 g β σ_coord B <
        optimizerNetProgress (rpmResidualStep lam J_coord C_coord n R_t g β σ_coord B) S_t g β σ_coord B ∧
      optimizerNetProgress (rpmResidualStep lam J_coord C_coord n R_0 g β σ_coord B) S_t g β σ_coord B <
        optimizerNetProgress (rpmResidualStep lam J_coord C_coord n R_t g β σ_coord B) S_t g β σ_coord B := by
  intro S_t S_0
  have hSt : ∀ i, 0 < S_t i := fun i =>
    coordResidualSharpness_pos J_coord C_coord n i hlam hRt (hC i)
  have hS_le : ∀ i, S_t i ≤ S_0 i := fun i =>
    coordResidualSharpness_mono_in_residual lam J_coord C_coord n i hR_lt.le (hC i)
  have hi₀_lt : S_t i₀ < S_0 i₀ :=
    coordResidualSharpness_strict_mono_in_residual lam J_coord C_coord i₀ hR_lt hCi₀ hn
  exact rpm_strict_speedup_from_residual_drop S_t S_0 g β σ_coord B
    hSt hS_le hM i₀ hi₀_lt hgi₀

/-- **Regime IV: EMA Momentum Strictly Increases the Variational Optimum Under Mini-Batch Noise
(`rpm_momentum_strictly_increases_optimal_progress`)**:
When `0 < β < 1`, `0 < B`, `∀ i, 0 < S i`, `∀ i, 0 < totalCoordMoment g β σ_coord B i`,
`∀ i, 0 < totalCoordMoment g 0 σ_coord B i`, and there exists at least one coordinate `i₀ : Fin p`
with non-zero gradient `g i₀ ≠ 0` and non-zero mini-batch noise `σ_coord i₀ ≠ 0`, the EMA-damped
variational optimum `optimizerNetProgress (rpmOptimalStep S g β σ_coord B) S g β σ_coord B` is
**strictly larger** than the memoryless (`β = 0`) variational optimum
`optimizerNetProgress (rpmOptimalStep S g 0 σ_coord B) S g 0 σ_coord B`. -/
theorem rpm_momentum_strictly_increases_optimal_progress (S : Fin p → ℝ) (g : Parameters p)
    {β B : ℝ} (σ_coord : Fin p → ℝ)
    (hβ0 : 0 < β) (hβ1 : β < 1) (hB : 0 < B)
    (hS : ∀ i, 0 < S i)
    (hMβ : ∀ i, 0 < totalCoordMoment g β σ_coord B i)
    (i₀ : Fin p) (hgi₀ : g i₀ ≠ 0) (hσi₀ : σ_coord i₀ ≠ 0) :
    optimizerNetProgress (rpmOptimalStep S g 0 σ_coord B) S g 0 σ_coord B <
      optimizerNetProgress (rpmOptimalStep S g β σ_coord B) S g β σ_coord B := by
  let η₀ := rpmOptimalStep S g 0 σ_coord B
  have hS_nonneg : ∀ i, 0 ≤ S i := fun i => (hS i).le
  have hM0_i₀ : 0 < totalCoordMoment g 0 σ_coord B i₀ :=
    totalCoordMoment_pos_of_grad_ne_zero g σ_coord i₀ le_rfl zero_lt_one.le hB hgi₀
  have hη₀_i₀ : η₀ i₀ ≠ 0 := by
    unfold η₀ rpmOptimalStep
    have hg2 : 0 < (g i₀) ^ 2 := sq_pos_of_ne_zero hgi₀
    have hSi₀ : 0 < S i₀ := hS i₀
    exact ne_of_gt (by positivity)
  have h1 : optimizerNetProgress η₀ S g 0 σ_coord B <
      optimizerNetProgress η₀ S g β σ_coord B :=
    adam_momentum_strictly_beats_preconditioned_no_momentum η₀ S g σ_coord
      hβ0 hβ1 hS_nonneg hB i₀ (hS i₀) hη₀_i₀ hσi₀
  have h2 : optimizerNetProgress η₀ S g β σ_coord B ≤
      optimizerNetProgress (rpmOptimalStep S g β σ_coord B) S g β σ_coord B :=
    rpm_variational_global_optimality η₀ S g β σ_coord B hS hMβ
  exact h1.trans_le h2

/-! ## 4. Dynamic Residual-Adapted PL Contraction Acceleration -/

/-- **Dynamic Residual-Adapted PL Contraction Strict Improvement
(`rpm_dynamic_pl_contraction_strict_improvement`)**:
When the local sharpness drops from `S_init` to `S_curr < S_init` (`0 < μ ≤ S_curr < S_init`),
adapting the learning rate to `1 / S_curr` yields a **strictly smaller (faster)** per-step
Polyak-Łojasiewicz contraction factor `0 ≤ 1 - μ / S_curr < 1 - μ / S_init < 1`, with exact
contraction improvement `μ * (S_init - S_curr) / (S_curr * S_init) > 0`. -/
theorem rpm_dynamic_pl_contraction_strict_improvement {μ S_curr S_init : ℝ}
    (hμ : 0 < μ) (hμ_le : μ ≤ S_curr) (hS_lt : S_curr < S_init) :
    0 ≤ 1 - μ / S_curr ∧
      1 - μ / S_curr < 1 - μ / S_init ∧
      (1 - μ / S_init) - (1 - μ / S_curr) =
        μ * (S_init - S_curr) / (S_curr * S_init) := by
  have hS_curr : 0 < S_curr := hμ.trans_le hμ_le
  have hS_init : 0 < S_init := hS_curr.trans hS_lt
  have hdiv_le : μ / S_curr ≤ 1 := (div_le_one hS_curr).mpr hμ_le
  have hdiv_lt : μ / S_init < μ / S_curr :=
    div_lt_div_of_pos_left hμ hS_curr hS_lt
  refine ⟨by linarith, by linarith, ?_⟩
  have hSc_ne : S_curr ≠ 0 := ne_of_gt hS_curr
  have hSi_ne : S_init ≠ 0 := ne_of_gt hS_init
  field_simp [hSc_ne, hSi_ne]
  ring

/-- **Residual-Coupled PL Contraction Acceleration (`rpm_residual_drop_pl_contraction_strict_improvement`)**:
When the prediction residual drops (`0 ≤ R_t < R_0`, `0 < μ ≤ lam`, `0 < C_coord i`, `0 < n`),
the coordinate PL contraction factor `1 - μ / coordResidualSharpness ... R_t i` is strictly
smaller (faster) than at the initial residual `R_0`. -/
theorem rpm_residual_drop_pl_contraction_strict_improvement {μ lam R_t R_0 : ℝ}
    (J_coord C_coord : Fin p → ℝ) {n : ℕ} (i : Fin p)
    (hμ : 0 < μ) (hμ_lam : μ ≤ lam) (hn : 0 < n)
    (hRt : 0 ≤ R_t) (hR_lt : R_t < R_0) (hC : 0 < C_coord i) :
    let S_t := coordResidualSharpness lam J_coord C_coord n R_t i
    let S_0 := coordResidualSharpness lam J_coord C_coord n R_0 i
    0 ≤ 1 - μ / S_t ∧ 1 - μ / S_t < 1 - μ / S_0 := by
  intro S_t S_0
  have hlam_le : lam ≤ S_t := lam_le_coordResidualSharpness J_coord C_coord n i hRt hC.le
  have hμ_le : μ ≤ S_t := hμ_lam.trans hlam_le
  have hS_lt : S_t < S_0 :=
    coordResidualSharpness_strict_mono_in_residual lam J_coord C_coord i hR_lt hC hn
  have h := rpm_dynamic_pl_contraction_strict_improvement hμ hμ_le hS_lt
  exact ⟨h.1, h.2.1⟩

end OpenTheory.ResidualPreconditionedMomentum
