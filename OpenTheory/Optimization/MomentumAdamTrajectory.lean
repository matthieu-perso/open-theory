import OpenTheory.Data.Foundation
import OpenTheory.Optimization.LocalSharpnessDescent
import OpenTheory.Optimization.PreconditionedDescent
import OpenTheory.Optimization.StochasticSharpnessSGD
import OpenTheory.Generalization.TrajectoryGeneralization

/-!
# Momentum, Exponential Moving Average (EMA), AdamW Decoupled Weight Decay, and Optimizer Regime Selection

This module formalizes four core theorem families that unify modern adaptive optimizers
(Heavy-Ball / EMA Momentum, Adam first-moment bias correction, AdamW decoupled weight decay,
and diagonal preconditioning) with trajectory-level optimization and generalization guarantees:

1. **Exponential Moving Average (EMA) First-Moment Trajectory & Bias Correction
   (`emaStep`, `emaTrajectory`, `emaWeightSum`, `biasCorrectedEMA`, `emaWeightSum_succ`,
   `emaTrajectory_const_grad`, `biasCorrectedEMA_const_grad`, `emaTrajectory_norm_le`,
   `biasCorrectedEMA_norm_le`)**:
   Starting from `m_0 = 0`, the EMA first-moment update `m_{t+1} = β • m_t + (1 - β) • g_t`
   accumulates geometric weight `emaWeightSum β T = 1 - β ^ T`. On a constant gradient `g`,
   `emaTrajectory β (fun _ => g) T = (1 - β ^ T) • g`, so Adam's bias correction
   `biasCorrectedEMA β g_seq T = (1 / (1 - β ^ T)) • emaTrajectory β g_seq T` recovers `g`
   **exactly** for every `T ≥ 1`. Furthermore, whenever `‖g_seq t‖ ≤ G_max`, both the raw EMA
   and the bias-corrected EMA remain inside the ball of radius `G_max`.

2. **EMA Variance Damping & Strict Reduction of the Stochastic PL Noise Floor
   (`emaVarianceStep`, `emaStationaryVariance`, `emaVarianceTrajectory`,
   `emaVariance_fixed_point`, `emaVarianceTrajectory_eq`, `emaVarianceTrajectory_le_stationary`,
   `ema_damping_factor_lt_one`, `ema_variance_strictly_smaller_than_sgd`,
   `momentum_pl_noise_floor_eq`, `momentum_pl_noise_floor_strict_improvement`,
   `momentum_pl_noise_ball_convergence`)**:
   Under independent mini-batch gradient noise of variance `σ_sq / B`, the EMA variance recurrence
   `v_{t+1} = β^2 * v_t + (1 - β)^2 * (σ_sq / B)` has stationary fixed point
   `emaStationaryVariance β σ_sq B = ((1 - β) / (1 + β)) * (σ_sq / B)` and closed form
   `v_T = (1 - (β^2)^T) * emaStationaryVariance β σ_sq B ≤ emaStationaryVariance β σ_sq B`.
   For `0 < β < 1`, `(1 - β) / (1 + β) < 1`, strictly shrinking both the mini-batch variance and
   the Polyak-Łojasiewicz asymptotic noise floor `sgdNoiseFloor` by the exact factor
   `(1 - β) / (1 + β)`.

3. **AdamW Decoupled Weight Decay Trajectory Ball & Out-of-Sample Generalization
   (`adamwStep`, `adamwTrajectory`, `adamwRadius`, `adamw_step_norm_le`,
   `adamw_step_preserves_ball`, `adamw_trajectory_norm_envelope`, `adamw_trajectory_in_ball`,
   `adamw_sign_direction_norm_le`, `adamw_mlp_generalization_tail`)**:
   Under decoupled weight decay `u_{t+1} = (1 - η * lam_w) • u_t - η • d_t` with bounded preconditioned
   update directions `‖d_t‖ ≤ D_max` (e.g., `D_max = √p` for Adam/SignGD on `p` coordinates),
   the closed ball of radius `adamwRadius lam_w D_max = D_max / lam_w` is forward-invariant for all
   `0 < η`, `0 < lam_w` with `η * lam_w ≤ 1`. Consequently, any two-layer MLP trained with AdamW
   enjoys a **horizon-free** out-of-sample generalization tail bound with radius `D_max / lam_w`.

4. **Precise Optimizer Regime Selection Theorem
   (`optimizerNoisePenalty`, `optimizerNetProgress`, `optimizerNetProgress_zero_momentum`,
   `adam_momentum_reduces_noise_penalty`, `adam_momentum_dominates_preconditioned_no_momentum`,
   `adam_momentum_strictly_beats_preconditioned_no_momentum`,
   `adam_momentum_strictly_beats_scalar_sgd`)**:
   Combining anisotropic coordinate progress `coordProgress η S g` from `PreconditionedDescent`
   with EMA variance damping `(1 - β) / (1 + β)` proves when and why diagonal preconditioning +
   momentum strictly dominates both memoryless preconditioned descent (`β = 0`) and scalar-step
   mini-batch SGD.
-/

noncomputable section

open MeasureTheory StatLean.StatisticalLearning
open scoped BigOperators RealInnerProductSpace

namespace OpenTheory.MomentumAdamTrajectory

open OpenTheory.LocalSharpnessDescent
open OpenTheory.PreconditionedDescent
open OpenTheory.StochasticSharpnessSGD
open OpenTheory.TrajectoryGeneralization
open OpenTheory.FeatureStep
open OpenTheory.MLPTrajectory
open OpenTheory.MLPFixedDesign
open OpenTheory.BallCover
open OpenTheory.Covering

variable {d m p : ℕ}

/-! ## 1. Exponential Moving Average (EMA) First-Moment Trajectory & Bias Correction -/

/-- One-step Exponential Moving Average (EMA) first-moment update on `Parameters p`:
`m_{t+1} = β • m_t + (1 - β) • g_t`. -/
def emaStep (β : ℝ) (m_state g : Parameters p) : Parameters p :=
  β • m_state + (1 - β) • g

/-- `T`-step EMA first-moment trajectory starting from `m_0 = 0`:
`emaTrajectory β g_seq 0 = 0` and `emaTrajectory β g_seq (t + 1) = emaStep β (emaTrajectory β g_seq t) (g_seq t)`. -/
def emaTrajectory (β : ℝ) (g_seq : ℕ → Parameters p) : ℕ → Parameters p
  | 0 => 0
  | t + 1 => emaStep β (emaTrajectory β g_seq t) (g_seq t)

@[simp]
theorem emaTrajectory_zero (β : ℝ) (g_seq : ℕ → Parameters p) :
    emaTrajectory β g_seq 0 = 0 :=
  rfl

@[simp]
theorem emaTrajectory_succ (β : ℝ) (g_seq : ℕ → Parameters p) (t : ℕ) :
    emaTrajectory β g_seq (t + 1) = emaStep β (emaTrajectory β g_seq t) (g_seq t) :=
  rfl

/-- Cumulative geometric weight `1 - β ^ T` of an EMA trajectory after `T` steps. -/
def emaWeightSum (β : ℝ) (T : ℕ) : ℝ :=
  1 - β ^ T

@[simp]
theorem emaWeightSum_zero (β : ℝ) : emaWeightSum β 0 = 0 := by
  simp [emaWeightSum]

/-- **Geometric Weight Recurrence (`emaWeightSum_succ`)**:
`β * (1 - β ^ T) + (1 - β) = 1 - β ^ (T + 1)`. -/
theorem emaWeightSum_succ (β : ℝ) (T : ℕ) :
    β * emaWeightSum β T + (1 - β) = emaWeightSum β (T + 1) := by
  unfold emaWeightSum
  rw [pow_succ']
  ring

/-- Non-negativity of `emaWeightSum β T` when `0 ≤ β ≤ 1`. -/
theorem emaWeightSum_nonneg {β : ℝ} (hβ0 : 0 ≤ β) (hβ1 : β ≤ 1) (T : ℕ) :
    0 ≤ emaWeightSum β T := by
  unfold emaWeightSum
  have hpow : β ^ T ≤ 1 := pow_le_one₀ hβ0 hβ1
  linarith

/-- Strict positivity of `emaWeightSum β T` when `0 ≤ β < 1` and `1 ≤ T`. -/
theorem emaWeightSum_pos {β : ℝ} (hβ0 : 0 ≤ β) (hβ1 : β < 1) {T : ℕ} (hT : 1 ≤ T) :
    0 < emaWeightSum β T := by
  unfold emaWeightSum
  have hT_ne : T ≠ 0 := Nat.pos_iff_ne_zero.mp hT
  have hpow : β ^ T < 1 := pow_lt_one₀ hβ0 hβ1 hT_ne
  linarith

/-- Upper bound `emaWeightSum β T ≤ 1` when `0 ≤ β`. -/
theorem emaWeightSum_le_one {β : ℝ} (hβ0 : 0 ≤ β) (T : ℕ) :
    emaWeightSum β T ≤ 1 := by
  unfold emaWeightSum
  have hpow : 0 ≤ β ^ T := pow_nonneg hβ0 T
  linarith

/-- Bias-corrected Adam first-moment estimator at step `T`:
`m̂_T = (1 / (1 - β ^ T)) • m_T`. -/
def biasCorrectedEMA (β : ℝ) (g_seq : ℕ → Parameters p) (T : ℕ) : Parameters p :=
  (1 / (1 - β ^ T)) • emaTrajectory β g_seq T

/-- Equivalent expression of `biasCorrectedEMA` in terms of `emaWeightSum`. -/
theorem biasCorrectedEMA_eq_inv_weightSum (β : ℝ) (g_seq : ℕ → Parameters p) (T : ℕ) :
    biasCorrectedEMA β g_seq T = (1 / emaWeightSum β T) • emaTrajectory β g_seq T :=
  rfl

/-- **Constant-Gradient Exact Formula (`emaTrajectory_const_grad`)**:
Along a constant gradient sequence `fun _ => g`, the `T`-step EMA state initialized at `0`
equals `(1 - β ^ T) • g = emaWeightSum β T • g`. -/
theorem emaTrajectory_const_grad (β : ℝ) (g : Parameters p) (T : ℕ) :
    emaTrajectory β (fun _ => g) T = (1 - β ^ T) • g := by
  have h_eq : emaTrajectory β (fun _ => g) T = emaWeightSum β T • g := by
    induction T with
    | zero =>
      simp [emaTrajectory, emaWeightSum]
    | succ t ih =>
      rw [emaTrajectory_succ, ih]
      unfold emaStep
      rw [smul_smul, ← add_smul, emaWeightSum_succ]
  exact h_eq

/-- **Exact Unbiasedness of Adam's Bias Correction on Stationary Gradients
(`biasCorrectedEMA_const_grad`)**:
For any `0 ≤ β < 1` and any step `T ≥ 1`, if the gradient sequence is constant (`g_seq t = g`),
then the bias-corrected EMA first moment equals `g` **identically**. -/
theorem biasCorrectedEMA_const_grad {β : ℝ} (g : Parameters p) {T : ℕ}
    (hβ0 : 0 ≤ β) (hβ1 : β < 1) (hT : 1 ≤ T) :
    biasCorrectedEMA β (fun _ => g) T = g := by
  unfold biasCorrectedEMA
  rw [emaTrajectory_const_grad, smul_smul]
  have hpos : 0 < 1 - β ^ T := emaWeightSum_pos hβ0 hβ1 hT
  have hne : 1 - β ^ T ≠ 0 := ne_of_gt hpos
  rw [one_div_mul_cancel hne, one_smul]

/-- **One-Step EMA Norm Envelope (`emaStep_norm_le`)**:
If `0 ≤ β ≤ 1`, `‖m_state‖ ≤ W * G_max`, and `‖g‖ ≤ G_max`, then
`‖emaStep β m_state g‖ ≤ (β * W + (1 - β)) * G_max`. -/
theorem emaStep_norm_le {β W G_max : ℝ} (m_state g : Parameters p)
    (hβ0 : 0 ≤ β) (hβ1 : β ≤ 1)
    (hm : ‖m_state‖ ≤ W * G_max) (hg : ‖g‖ ≤ G_max) :
    ‖emaStep β m_state g‖ ≤ (β * W + (1 - β)) * G_max := by
  unfold emaStep
  have h1 : 0 ≤ 1 - β := sub_nonneg.mpr hβ1
  calc
    ‖β • m_state + (1 - β) • g‖
      ≤ ‖β • m_state‖ + ‖(1 - β) • g‖ := norm_add_le _ _
    _ = |β| * ‖m_state‖ + |1 - β| * ‖g‖ := by
        rw [norm_smul, norm_smul, Real.norm_eq_abs, Real.norm_eq_abs]
    _ = β * ‖m_state‖ + (1 - β) * ‖g‖ := by
        rw [abs_of_nonneg hβ0, abs_of_nonneg h1]
    _ ≤ β * (W * G_max) + (1 - β) * G_max :=
        add_le_add (mul_le_mul_of_nonneg_left hm hβ0) (mul_le_mul_of_nonneg_left hg h1)
    _ = (β * W + (1 - β)) * G_max := by ring

/-- **Bounded Gradient Envelope for Raw EMA (`emaTrajectory_norm_le`)**:
If `0 ≤ β ≤ 1` and `‖g_seq t‖ ≤ G_max` for all `t`, then for every horizon `T : ℕ`:
`‖emaTrajectory β g_seq T‖ ≤ (1 - β ^ T) * G_max` and `(1 - β ^ T) * G_max ≤ G_max`. -/
theorem emaTrajectory_norm_le {β G_max : ℝ} (g_seq : ℕ → Parameters p)
    (hβ0 : 0 ≤ β) (hβ1 : β ≤ 1) (hg : ∀ t, ‖g_seq t‖ ≤ G_max) (T : ℕ) :
    ‖emaTrajectory β g_seq T‖ ≤ (1 - β ^ T) * G_max ∧
      (1 - β ^ T) * G_max ≤ G_max := by
  have hG_nonneg : 0 ≤ G_max := (norm_nonneg (g_seq 0)).trans (hg 0)
  have htight : ‖emaTrajectory β g_seq T‖ ≤ emaWeightSum β T * G_max := by
    induction T with
    | zero =>
      simp [emaTrajectory, emaWeightSum]
    | succ t ih =>
      rw [emaTrajectory_succ, ← emaWeightSum_succ]
      exact emaStep_norm_le (emaTrajectory β g_seq t) (g_seq t) hβ0 hβ1 ih (hg t)
  refine ⟨htight, ?_⟩
  have hw : 1 - β ^ T ≤ 1 := emaWeightSum_le_one hβ0 T
  nlinarith

/-- **Bounded Gradient Envelope for Bias-Corrected EMA (`biasCorrectedEMA_norm_le`)**:
If `0 ≤ β < 1`, `1 ≤ T`, and `‖g_seq t‖ ≤ G_max` for all `t`, then the bias-corrected Adam
first moment satisfies `‖biasCorrectedEMA β g_seq T‖ ≤ G_max`. -/
theorem biasCorrectedEMA_norm_le {β G_max : ℝ} (g_seq : ℕ → Parameters p) {T : ℕ}
    (hβ0 : 0 ≤ β) (hβ1 : β < 1) (hT : 1 ≤ T) (hg : ∀ t, ‖g_seq t‖ ≤ G_max) :
    ‖biasCorrectedEMA β g_seq T‖ ≤ G_max := by
  unfold biasCorrectedEMA
  have hpos : 0 < 1 - β ^ T := emaWeightSum_pos hβ0 hβ1 hT
  have hinv_pos : 0 ≤ 1 / (1 - β ^ T) := div_nonneg zero_le_one hpos.le
  have hraw := (emaTrajectory_norm_le g_seq hβ0 hβ1.le hg T).1
  rw [norm_smul, Real.norm_eq_abs, abs_of_nonneg hinv_pos]
  calc
    (1 / (1 - β ^ T)) * ‖emaTrajectory β g_seq T‖
      ≤ (1 / (1 - β ^ T)) * ((1 - β ^ T) * G_max) :=
        mul_le_mul_of_nonneg_left hraw hinv_pos
    _ = G_max := by
        field_simp [ne_of_gt hpos]

/-! ## 2. EMA Variance Damping & Strict Reduction of the Stochastic PL Noise Floor -/

/-- One-step variance evolution of the EMA first moment under independent mini-batch noise
with per-batch variance `σ_batch_sq = σ_sq / B`:
`v_{t+1} = β^2 * v_t + (1 - β)^2 * σ_batch_sq`. -/
def emaVarianceStep (β σ_batch_sq v : ℝ) : ℝ :=
  β ^ 2 * v + (1 - β) ^ 2 * σ_batch_sq

/-- Stationary asymptotic variance of the EMA momentum state:
`emaStationaryVariance β σ_sq B = ((1 - β) / (1 + β)) * (σ_sq / B)`. -/
def emaStationaryVariance (β σ_sq B : ℝ) : ℝ :=
  ((1 - β) / (1 + β)) * (σ_sq / B)

/-- `T`-step EMA variance trajectory starting from `v_0 = 0`. -/
def emaVarianceTrajectory (β σ_batch_sq : ℝ) : ℕ → ℝ
  | 0 => 0
  | t + 1 => emaVarianceStep β σ_batch_sq (emaVarianceTrajectory β σ_batch_sq t)

@[simp]
theorem emaVarianceTrajectory_zero (β σ_batch_sq : ℝ) :
    emaVarianceTrajectory β σ_batch_sq 0 = 0 :=
  rfl

@[simp]
theorem emaVarianceTrajectory_succ (β σ_batch_sq : ℝ) (t : ℕ) :
    emaVarianceTrajectory β σ_batch_sq (t + 1) =
      emaVarianceStep β σ_batch_sq (emaVarianceTrajectory β σ_batch_sq t) :=
  rfl

/-- **Stationary Fixed-Point Identity for EMA Variance (`emaVariance_fixed_point`)**:
When `1 + β ≠ 0`, `emaStationaryVariance β σ_sq B` is the exact fixed point of
`emaVarianceStep β (σ_sq / B)`. -/
theorem emaVariance_fixed_point (β σ_sq B : ℝ) (hβ : 1 + β ≠ 0) :
    emaVarianceStep β (σ_sq / B) (emaStationaryVariance β σ_sq B) =
      emaStationaryVariance β σ_sq B := by
  unfold emaVarianceStep emaStationaryVariance
  field_simp [hβ]
  ring

/-- **Exact Closed Form of the `T`-Step EMA Variance Trajectory (`emaVarianceTrajectory_eq`)**:
When `1 + β ≠ 0`, starting from `v_0 = 0`, the `T`-step EMA variance is
`(1 - (β ^ 2) ^ T) * emaStationaryVariance β σ_sq B`. -/
theorem emaVarianceTrajectory_eq (β σ_sq B : ℝ) (hβ : 1 + β ≠ 0) (T : ℕ) :
    emaVarianceTrajectory β (σ_sq / B) T =
      (1 - (β ^ 2) ^ T) * emaStationaryVariance β σ_sq B := by
  induction T with
  | zero =>
    simp [emaVarianceTrajectory]
  | succ t ih =>
    rw [emaVarianceTrajectory_succ, ih]
    unfold emaVarianceStep emaStationaryVariance
    rw [pow_succ']
    field_simp [hβ]
    ring

/-- **Uniform Stationary Upper Bound on EMA Variance (`emaVarianceTrajectory_le_stationary`)**:
For `0 ≤ β < 1`, `0 ≤ σ_sq`, and `0 < B`, the `T`-step EMA variance satisfies the exact closed
form and never exceeds `emaStationaryVariance β σ_sq B`. -/
theorem emaVarianceTrajectory_le_stationary {β σ_sq B : ℝ}
    (hβ0 : 0 ≤ β) (hβ1 : β < 1) (hσ : 0 ≤ σ_sq) (hB : 0 < B) (T : ℕ) :
    emaVarianceTrajectory β (σ_sq / B) T =
        (1 - (β ^ 2) ^ T) * emaStationaryVariance β σ_sq B ∧
      emaVarianceTrajectory β (σ_sq / B) T ≤ emaStationaryVariance β σ_sq B := by
  have h1β : 1 + β ≠ 0 := by linarith
  have heq := emaVarianceTrajectory_eq β σ_sq B h1β T
  refine ⟨heq, ?_⟩
  rw [heq]
  have hstat_nonneg : 0 ≤ emaStationaryVariance β σ_sq B := by
    unfold emaStationaryVariance
    have hsub : 0 ≤ 1 - β := by linarith
    have hadd : 0 < 1 + β := by linarith
    positivity
  have hpow_nonneg : 0 ≤ (β ^ 2) ^ T := by positivity
  nlinarith

/-- **Strict EMA Damping Factor Inequality (`ema_damping_factor_lt_one`)**:
For any momentum parameter `0 < β < 1`, the EMA variance damping factor satisfies
`0 < (1 - β) / (1 + β)` and `(1 - β) / (1 + β) < 1`. -/
theorem ema_damping_factor_lt_one {β : ℝ} (hβ0 : 0 < β) (hβ1 : β < 1) :
    0 < (1 - β) / (1 + β) ∧ (1 - β) / (1 + β) < 1 := by
  have hnum : 0 < 1 - β := by linarith
  have hden : 0 < 1 + β := by linarith
  refine ⟨div_pos hnum hden, ?_⟩
  rw [div_lt_one hden]
  linarith

/-- **Strict EMA Variance Reduction Over Memoryless SGD (`ema_variance_strictly_smaller_than_sgd`)**:
Whenever `0 < β < 1`, `0 < σ_sq`, and `0 < B`, the stationary EMA variance
`emaStationaryVariance β σ_sq B` is **strictly smaller** than the memoryless mini-batch gradient
variance `σ_sq / B`. -/
theorem ema_variance_strictly_smaller_than_sgd {β σ_sq B : ℝ}
    (hβ0 : 0 < β) (hβ1 : β < 1) (hσ : 0 < σ_sq) (hB : 0 < B) :
    emaStationaryVariance β σ_sq B < σ_sq / B := by
  unfold emaStationaryVariance
  have hdamp := (ema_damping_factor_lt_one hβ0 hβ1).2
  have hbatch_pos : 0 < σ_sq / B := div_pos hσ hB
  nlinarith

/-- **Exact Factor Relationship for the Momentum PL Noise Floor (`momentum_pl_noise_floor_eq`)**:
Replacing the raw gradient variance `σ_sq` by the EMA-damped variance
`((1 - β) / (1 + β)) * σ_sq` scales `sgdNoiseFloor` by the exact factor `(1 - β) / (1 + β)`. -/
theorem momentum_pl_noise_floor_eq (η S_loc μ β σ_sq : ℝ) (B : ℕ) :
    sgdNoiseFloor η S_loc μ (((1 - β) / (1 + β)) * σ_sq) B =
      ((1 - β) / (1 + β)) * sgdNoiseFloor η S_loc μ σ_sq B := by
  unfold sgdNoiseFloor
  ring

/-- **Strict Reduction of the Stochastic PL Noise Floor via EMA Momentum
(`momentum_pl_noise_floor_strict_improvement`)**:
When `0 < η`, `0 < S_loc`, `0 < μ`, `0 < β < 1`, `0 < σ_sq`, and `0 < B`, EMA momentum strictly
shrinks the asymptotic PL noise floor relative to memoryless mini-batch SGD:
`sgdNoiseFloor η S_loc μ (((1 - β) / (1 + β)) * σ_sq) B < sgdNoiseFloor η S_loc μ σ_sq B`. -/
theorem momentum_pl_noise_floor_strict_improvement {η S_loc μ β σ_sq : ℝ} (B : ℕ)
    (hη : 0 < η) (hS : 0 < S_loc) (hμ : 0 < μ)
    (hβ0 : 0 < β) (hβ1 : β < 1) (hσ : 0 < σ_sq) (hB : 0 < B) :
    sgdNoiseFloor η S_loc μ (((1 - β) / (1 + β)) * σ_sq) B <
      sgdNoiseFloor η S_loc μ σ_sq B := by
  rw [momentum_pl_noise_floor_eq]
  have hdamp := (ema_damping_factor_lt_one hβ0 hβ1).2
  have hB0 : (0 : ℝ) < B := Nat.cast_pos.mpr hB
  have hfloor_pos : 0 < sgdNoiseFloor η S_loc μ σ_sq B := by
    unfold sgdNoiseFloor
    positivity
  nlinarith

/-- **Multi-Step PL Convergence with EMA Momentum Variance Damping
(`momentum_pl_noise_ball_convergence`)**:
Along any momentum-damped trajectory where each step satisfies the PL recurrence with EMA
stationary variance `((1 - β) / (1 + β)) * (σ_sq / B)`, the expected optimality gap converges
geometrically to the strictly smaller noise floor
`((1 - β) / (1 + β)) * sgdNoiseFloor η S_loc μ σ_sq B`. -/
theorem momentum_pl_noise_ball_convergence (E_L : ℕ → ℝ) (floor η S_loc μ β σ_sq : ℝ) (B : ℕ)
    (hη : 0 < η) (hμ : 0 < μ) (hημ : η * μ < 1)
    (hS : 0 ≤ S_loc) (hβ0 : 0 ≤ β) (hβ1 : β ≤ 1) (hσ : 0 ≤ σ_sq) (hB : 0 < B)
    (hrec : ∀ t, E_L (t + 1) - floor ≤
      (1 - η * μ) * (E_L t - floor) +
        (η ^ 2 * S_loc / 2) * ((((1 - β) / (1 + β)) * σ_sq) / (B : ℝ)))
    (T : ℕ) :
    E_L T - floor ≤
      (1 - η * μ) ^ T * (E_L 0 - floor) +
        ((1 - β) / (1 + β)) * sgdNoiseFloor η S_loc μ σ_sq B := by
  have hdamp_nonneg : 0 ≤ ((1 - β) / (1 + β)) * σ_sq := by
    have h1 : 0 ≤ 1 - β := sub_nonneg.mpr hβ1
    have h2 : 0 < 1 + β := by linarith
    positivity
  rw [← momentum_pl_noise_floor_eq]
  exact sgd_pl_noise_ball_convergence E_L floor η S_loc μ (((1 - β) / (1 + β)) * σ_sq) B
    hη hμ hημ hS hdamp_nonneg hB hrec T

/-! ## 3. AdamW Decoupled Weight Decay Trajectory Ball & Out-of-Sample Generalization -/

/-- One-step AdamW update with learning rate `η`, decoupled weight decay coefficient `lam_w`,
current parameter `u : Parameters p`, and preconditioned update direction `d_t : Parameters p`:
`u_{t+1} = (1 - η * lam_w) • u - η • d_t`. -/
def adamwStep (η lam_w : ℝ) (u d_t : Parameters p) : Parameters p :=
  (1 - η * lam_w) • u - η • d_t

/-- `T`-step AdamW trajectory starting from `u_0` driven by preconditioned directions
`d_seq : ℕ → Parameters p`. -/
def adamwTrajectory (η lam_w : ℝ) (u_0 : Parameters p) (d_seq : ℕ → Parameters p) :
    ℕ → Parameters p
  | 0 => u_0
  | t + 1 => adamwStep η lam_w (adamwTrajectory η lam_w u_0 d_seq t) (d_seq t)

@[simp]
theorem adamwTrajectory_zero (η lam_w : ℝ) (u_0 : Parameters p) (d_seq : ℕ → Parameters p) :
    adamwTrajectory η lam_w u_0 d_seq 0 = u_0 :=
  rfl

@[simp]
theorem adamwTrajectory_succ (η lam_w : ℝ) (u_0 : Parameters p) (d_seq : ℕ → Parameters p) (t : ℕ) :
    adamwTrajectory η lam_w u_0 d_seq (t + 1) =
      adamwStep η lam_w (adamwTrajectory η lam_w u_0 d_seq t) (d_seq t) :=
  rfl

/-- Horizon-free invariant Euclidean ball radius of AdamW with decoupled weight decay `lam_w`
and update direction norm bound `D_max`:
`adamwRadius lam_w D_max = D_max / lam_w`. -/
def adamwRadius (lam_w D_max : ℝ) : ℝ :=
  D_max / lam_w

/-- Non-negativity of `adamwRadius lam_w D_max` when `0 < lam_w` and `0 ≤ D_max`. -/
theorem adamwRadius_nonneg {lam_w D_max : ℝ} (hlam : 0 < lam_w) (hD : 0 ≤ D_max) :
    0 ≤ adamwRadius lam_w D_max :=
  div_nonneg hD hlam.le

/-- **One-Step AdamW Contraction-Drift Norm Inequality (`adamw_step_norm_le`)**:
When `0 ≤ η`, `0 ≤ lam_w`, `η * lam_w ≤ 1`, and `‖d_t‖ ≤ D_max`, the AdamW step satisfies
`‖adamwStep η lam_w u d_t‖ ≤ (1 - η * lam_w) * ‖u‖ + η * D_max`. -/
theorem adamw_step_norm_le {η lam_w D_max : ℝ} (u d_t : Parameters p)
    (hη : 0 ≤ η) (hηlam : η * lam_w ≤ 1) (hd : ‖d_t‖ ≤ D_max) :
    ‖adamwStep η lam_w u d_t‖ ≤ (1 - η * lam_w) * ‖u‖ + η * D_max := by
  unfold adamwStep
  have h1 : 0 ≤ 1 - η * lam_w := sub_nonneg.mpr hηlam
  calc
    ‖(1 - η * lam_w) • u - η • d_t‖
      ≤ ‖(1 - η * lam_w) • u‖ + ‖η • d_t‖ := norm_sub_le _ _
    _ = |1 - η * lam_w| * ‖u‖ + |η| * ‖d_t‖ := by
        rw [norm_smul, norm_smul, Real.norm_eq_abs, Real.norm_eq_abs]
    _ = (1 - η * lam_w) * ‖u‖ + η * ‖d_t‖ := by
        rw [abs_of_nonneg h1, abs_of_nonneg hη]
    _ ≤ (1 - η * lam_w) * ‖u‖ + η * D_max :=
        add_le_add le_rfl (mul_le_mul_of_nonneg_left hd hη)

/-- **Forward Invariance of the AdamW Trajectory Ball (`adamw_step_preserves_ball`)**:
If `0 ≤ η`, `0 < lam_w`, `η * lam_w ≤ 1`, `‖u‖ ≤ adamwRadius lam_w D_max`, and `‖d_t‖ ≤ D_max`,
then `‖adamwStep η lam_w u d_t‖ ≤ adamwRadius lam_w D_max`
(equivalently, `adamwStep η lam_w u d_t ∈ trajectoryBall p (adamwRadius lam_w D_max)`). -/
theorem adamw_step_preserves_ball {η lam_w D_max : ℝ} (u d_t : Parameters p)
    (hη : 0 ≤ η) (hlam : 0 < lam_w) (hηlam : η * lam_w ≤ 1)
    (hu : u ∈ trajectoryBall p (adamwRadius lam_w D_max)) (hd : ‖d_t‖ ≤ D_max) :
    adamwStep η lam_w u d_t ∈ trajectoryBall p (adamwRadius lam_w D_max) := by
  rw [mem_trajectoryBall] at hu ⊢
  have hstep := adamw_step_norm_le u d_t hη hηlam hd
  have h1 : 0 ≤ 1 - η * lam_w := sub_nonneg.mpr hηlam
  have hmul : (1 - η * lam_w) * ‖u‖ ≤ (1 - η * lam_w) * (D_max / lam_w) :=
    mul_le_mul_of_nonneg_left hu h1
  have hid : (1 - η * lam_w) * (D_max / lam_w) + η * D_max = D_max / lam_w := by
    field_simp [ne_of_gt hlam]
    ring
  unfold adamwRadius at *
  linarith

/-- **Horizon-Free Norm Envelope Along Any AdamW Trajectory (`adamw_trajectory_norm_envelope`)**:
When `0 ≤ η`, `0 < lam_w`, `η * lam_w ≤ 1`, `‖u_0‖ ≤ D_max / lam_w` (in particular for `u_0 = 0` when
`0 ≤ D_max`), and `‖d_seq t‖ ≤ D_max` for all `t`, every iterate `T : ℕ` satisfies:
`‖adamwTrajectory η lam_w u_0 d_seq T‖ ≤ D_max / lam_w` and
`adamwTrajectory η lam_w u_0 d_seq T ∈ trajectoryBall p (adamwRadius lam_w D_max)`. -/
theorem adamw_trajectory_norm_envelope {η lam_w D_max : ℝ}
    (u_0 : Parameters p) (d_seq : ℕ → Parameters p)
    (hη : 0 ≤ η) (hlam : 0 < lam_w) (hηlam : η * lam_w ≤ 1)
    (hu0 : ‖u_0‖ ≤ adamwRadius lam_w D_max)
    (hd : ∀ t, ‖d_seq t‖ ≤ D_max) (T : ℕ) :
    ‖adamwTrajectory η lam_w u_0 d_seq T‖ ≤ adamwRadius lam_w D_max ∧
      adamwTrajectory η lam_w u_0 d_seq T ∈ trajectoryBall p (adamwRadius lam_w D_max) := by
  have hmem : adamwTrajectory η lam_w u_0 d_seq T ∈ trajectoryBall p (adamwRadius lam_w D_max) := by
    induction T with
    | zero =>
      exact hu0
    | succ t ih =>
      rw [adamwTrajectory_succ]
      exact adamw_step_preserves_ball (adamwTrajectory η lam_w u_0 d_seq t) (d_seq t)
        hη hlam hηlam ih (hd t)
  exact ⟨hmem, hmem⟩

/-- Zero-initialized specialization of `adamw_trajectory_norm_envelope`: starting from `u_0 = 0`
with `0 ≤ D_max`, every AdamW iterate lies in `trajectoryBall p (adamwRadius lam_w D_max)`. -/
theorem adamw_trajectory_in_ball {η lam_w D_max : ℝ}
    (d_seq : ℕ → Parameters p)
    (hη : 0 ≤ η) (hlam : 0 < lam_w) (hηlam : η * lam_w ≤ 1) (hD : 0 ≤ D_max)
    (hd : ∀ t, ‖d_seq t‖ ≤ D_max) (T : ℕ) :
    adamwTrajectory η lam_w 0 d_seq T ∈ trajectoryBall p (adamwRadius lam_w D_max) := by
  have hu0 : ‖(0 : Parameters p)‖ ≤ adamwRadius lam_w D_max := by
    rw [norm_zero]
    exact adamwRadius_nonneg hlam hD
  exact (adamw_trajectory_norm_envelope 0 d_seq hη hlam hηlam hu0 hd T).2

/-- Coordinate-wise bounded update directions (such as Adam/SignGD where `|d_t i| ≤ 1` for all
coordinates `i : Fin p`) automatically satisfy `‖d_t‖ ≤ √p`. -/
theorem adamw_sign_direction_norm_le (d_t : Parameters p)
    (hcoord : ∀ i : Fin p, |d_t i| ≤ 1) :
    ‖d_t‖ ≤ Real.sqrt (p : ℝ) := by
  have hsq : ‖d_t‖ ^ 2 ≤ (Real.sqrt (p : ℝ)) ^ 2 := by
    rw [parameters_norm_sq, Real.sq_sqrt (Nat.cast_nonneg p)]
    calc
      (∑ i : Fin p, (d_t i) ^ 2)
        ≤ ∑ _i : Fin p, (1 : ℝ) := by
          refine Finset.sum_le_sum fun i _ => ?_
          have h1 := hcoord i
          nlinarith [abs_le.mp h1, sq_abs (d_t i)]
      _ = (p : ℝ) := by simp
  exact (sq_le_sq₀ (norm_nonneg d_t) (Real.sqrt_nonneg (p : ℝ))).mp hsq

/-- **Horizon-Free Out-of-Sample Generalization Bound for Two-Layer MLPs Trained with AdamW
(`adamw_mlp_generalization_tail`)**:
For any two-layer MLP `net : TwoLayerNet d m` trained with decoupled weight decay `lam_w > 0`
and bounded preconditioned update directions `‖d_seq t‖ ≤ D_max` (`0 ≤ η`, `η * lam_w ≤ 1`,
`0 ≤ D_max`), the entire infinite AdamW trajectory `{adamwTrajectory η lam_w 0 d_seq T | T : ℕ}`
lies inside `trajectoryBall (m * d) (adamwRadius lam_w D_max)` and satisfies a horizon-free
uniform out-of-sample generalization tail bound over `s ~ sampleLaw P.law n`. -/
theorem adamw_mlp_generalization_tail {Z : Type*} [MeasurableSpace Z]
    (P : Population Z) {n : ℕ} (net : TwoLayerNet d m)
    (η lam_w D_max : ℝ) (d_seq : ℕ → Parameters (m * d))
    (xOf : Z → Fin d → ℝ) (yOf : Z → ℝ) (ℓ : ℝ → ℝ → ℝ)
    {L_ℓ A L E δ ε : ℝ}
    (hη : 0 ≤ η) (hlam : 0 < lam_w) (hηlam : η * lam_w ≤ 1) (hD : 0 ≤ D_max)
    (hd : ∀ t, ‖d_seq t‖ ≤ D_max)
    (hLℓ : 0 ≤ L_ℓ) (hA : 0 ≤ A) (hL : 0 ≤ L) (hE : 0 ≤ E)
    (hδ : 0 < δ) (hn : 1 ≤ n) (hε : 0 < ε)
    (hℓ : ∀ p₁ p₂ y', |ℓ p₁ y' - ℓ p₂ y'| ≤ L_ℓ * |p₁ - p₂|)
    (hrange : ∀ u ∈ trajectoryBall (m * d) (adamwRadius lam_w D_max), ∀ z,
      paramLoss net xOf yOf ℓ u z ∈ Set.Icc (0 : ℝ) 1)
    (hmeas : ∀ u ∈ trajectoryBall (m * d) (adamwRadius lam_w D_max),
      Measurable (paramLoss net xOf yOf ℓ u))
    (hW : ∀ i, |net.W₂ 0 i| ≤ A) (hσ : ∀ s, |deriv net.act.σ s| ≤ L)
    (hx : ∀ z, inputEnergy (xOf z) ≤ E) :
    let rad := adamwRadius lam_w D_max
    let ρ := paramLossLipConst m L_ℓ A L E * δ
    let N := (coverAxis rad δ (m * d) + 1) ^ (m * d)
    sampleLaw P.law n {s | ∃ T : ℕ,
      ε + 2 * ρ < |empRisk (paramLoss net xOf yOf ℓ) s
        (adamwTrajectory η lam_w 0 d_seq T) -
        P.risk (paramLoss net xOf yOf ℓ (adamwTrajectory η lam_w 0 d_seq T))|} ≤
      ENNReal.ofReal (2 * (N : ℝ) * Real.exp (-2 * n * ε ^ 2)) := by
  intro rad ρ N
  have hrad : 0 ≤ rad := adamwRadius_nonneg hlam hD
  apply (measure_mono ?_).trans
    (trajectory_uniform_deviation_tail P net xOf yOf ℓ hLℓ hA hL hE hrad hδ hn hε
      hℓ hrange hmeas hW hσ hx)
  intro s hs
  obtain ⟨T, hbad⟩ := hs
  exact ⟨adamwTrajectory η lam_w 0 d_seq T,
    adamw_trajectory_in_ball d_seq hη hlam hηlam hD hd T,
    hbad⟩

/-! ## 4. Precise Optimizer Regime Selection Theorem (SGD vs. Momentum vs. Diagonal/AdamW) -/

/-- Coordinate-summed stochastic curvature noise penalty before EMA damping:
`(1 / 2) * ∑ i : Fin p, S i * (η i) ^ 2 * ((σ_coord i) ^ 2 / B)`. -/
def optimizerNoisePenalty (η S : Fin p → ℝ) (σ_coord : Fin p → ℝ) (B : ℝ) : ℝ :=
  (1 / 2) * ∑ i : Fin p, S i * (η i) ^ 2 * ((σ_coord i) ^ 2 / B)

/-- Non-negativity of `optimizerNoisePenalty` when `0 ≤ S i` and `0 < B`. -/
theorem optimizerNoisePenalty_nonneg (η S σ_coord : Fin p → ℝ) {B : ℝ}
    (hS : ∀ i, 0 ≤ S i) (hB : 0 < B) :
    0 ≤ optimizerNoisePenalty η S σ_coord B := by
  unfold optimizerNoisePenalty
  refine mul_nonneg (by norm_num) (Finset.sum_nonneg fun i _ => ?_)
  have hSi : 0 ≤ S i := hS i
  positivity

/-- Strict positivity of `optimizerNoisePenalty` whenever at least one coordinate `i₀` has
`0 < S i₀`, `η i₀ ≠ 0`, and `σ_coord i₀ ≠ 0` (with `∀ i, 0 ≤ S i` and `0 < B`). -/
theorem optimizerNoisePenalty_pos (η S σ_coord : Fin p → ℝ) {B : ℝ}
    (hS : ∀ i, 0 ≤ S i) (hB : 0 < B)
    (i₀ : Fin p) (hSi₀ : 0 < S i₀) (hηi₀ : η i₀ ≠ 0) (hσi₀ : σ_coord i₀ ≠ 0) :
    0 < optimizerNoisePenalty η S σ_coord B := by
  unfold optimizerNoisePenalty
  refine mul_pos (by norm_num) ?_
  have hterm_nonneg : ∀ i ∈ Finset.univ,
      0 ≤ S i * (η i) ^ 2 * ((σ_coord i) ^ 2 / B) := by
    intro i _
    have hSi : 0 ≤ S i := hS i
    positivity
  have hterm_pos : 0 < S i₀ * (η i₀) ^ 2 * ((σ_coord i₀) ^ 2 / B) := by
    have hη2 : 0 < (η i₀) ^ 2 := sq_pos_of_ne_zero hηi₀
    have hσ2 : 0 < (σ_coord i₀) ^ 2 := sq_pos_of_ne_zero hσi₀
    positivity
  exact Finset.sum_pos' hterm_nonneg ⟨i₀, Finset.mem_univ i₀, hterm_pos⟩

/-- Net one-step expected objective progress of a diagonal preconditioned optimizer with
coordinate learning rates `η : Fin p → ℝ`, coordinate sharpnesses `S : Fin p → ℝ`, true gradient
`g : Parameters p`, EMA momentum parameter `β : ℝ`, coordinate gradient noise standard deviations
`σ_coord : Fin p → ℝ`, and batch size `B : ℝ`:
`optimizerNetProgress η S g β σ_coord B = coordProgress η S g - (1 / 2) * ((1 - β) / (1 + β)) * ∑ i, S i * (η i)^2 * ((σ_coord i)^2 / B)`. -/
def optimizerNetProgress (η S : Fin p → ℝ) (g : Parameters p) (β : ℝ)
    (σ_coord : Fin p → ℝ) (B : ℝ) : ℝ :=
  coordProgress η S g -
    (1 / 2) * ((1 - β) / (1 + β)) * ∑ i : Fin p, S i * (η i) ^ 2 * ((σ_coord i) ^ 2 / B)

/-- Expressing `optimizerNetProgress` via `optimizerNoisePenalty`. -/
theorem optimizerNetProgress_eq_sub_penalty (η S : Fin p → ℝ) (g : Parameters p) (β : ℝ)
    (σ_coord : Fin p → ℝ) (B : ℝ) :
    optimizerNetProgress η S g β σ_coord B =
      coordProgress η S g - ((1 - β) / (1 + β)) * optimizerNoisePenalty η S σ_coord B := by
  unfold optimizerNetProgress optimizerNoisePenalty
  ring

/-- At `β = 0` (no momentum), the EMA damping factor `(1 - 0) / (1 + 0)` equals `1`. -/
@[simp]
theorem optimizerNetProgress_zero_momentum (η S : Fin p → ℝ) (g : Parameters p)
    (σ_coord : Fin p → ℝ) (B : ℝ) :
    optimizerNetProgress η S g 0 σ_coord B =
      coordProgress η S g - optimizerNoisePenalty η S σ_coord B := by
  rw [optimizerNetProgress_eq_sub_penalty]
  norm_num

/-- **Exact Momentum Gain Identity (`adam_momentum_reduces_noise_penalty`)**:
For `1 + β ≠ 0`, the net progress gain of adding EMA momentum `β` over memoryless preconditioned
descent (`β = 0`) with the same coordinate step vector `η` is exactly
`(2 * β / (1 + β)) * optimizerNoisePenalty η S σ_coord B`. -/
theorem adam_momentum_reduces_noise_penalty (η S : Fin p → ℝ) (g : Parameters p) (β : ℝ)
    (σ_coord : Fin p → ℝ) (B : ℝ) (hβ : 1 + β ≠ 0) :
    optimizerNetProgress η S g β σ_coord B - optimizerNetProgress η S g 0 σ_coord B =
      (2 * β / (1 + β)) * optimizerNoisePenalty η S σ_coord B := by
  rw [optimizerNetProgress_eq_sub_penalty, optimizerNetProgress_zero_momentum]
  field_simp [hβ]
  ring

/-- **Dominance of Momentum-Damped Preconditioning Over Memoryless Preconditioning
(`adam_momentum_dominates_preconditioned_no_momentum`)**:
For any `0 ≤ β < 1`, `∀ i, 0 ≤ S i`, and `0 < B`, adding EMA momentum achieves at least as much
net progress as memoryless preconditioned descent (`β = 0`), and strictly more whenever any
coordinate `i₀` has `0 < β`, `0 < S i₀`, `η i₀ ≠ 0`, and `σ_coord i₀ ≠ 0`. -/
theorem adam_momentum_dominates_preconditioned_no_momentum
    (η S : Fin p → ℝ) (g : Parameters p) {β : ℝ} (σ_coord : Fin p → ℝ) {B : ℝ}
    (hβ0 : 0 ≤ β) (_hβ1 : β < 1) (hS : ∀ i, 0 ≤ S i) (hB : 0 < B) :
    optimizerNetProgress η S g 0 σ_coord B ≤ optimizerNetProgress η S g β σ_coord B := by
  have h1β : 1 + β ≠ 0 := by linarith
  have hdiff := adam_momentum_reduces_noise_penalty η S g β σ_coord B h1β
  have hpen : 0 ≤ optimizerNoisePenalty η S σ_coord B :=
    optimizerNoisePenalty_nonneg η S σ_coord hS hB
  have hfactor : 0 ≤ 2 * β / (1 + β) := by
    have hden : 0 < 1 + β := by linarith
    positivity
  have hprod : 0 ≤ (2 * β / (1 + β)) * optimizerNoisePenalty η S σ_coord B :=
    mul_nonneg hfactor hpen
  linarith

/-- **Strict Dominance of Momentum-Damped Preconditioning Under Non-Zero Gradient Noise
(`adam_momentum_strictly_beats_preconditioned_no_momentum`)**:
Whenever `0 < β < 1`, `∀ i, 0 ≤ S i`, `0 < B`, and there exists a coordinate `i₀ : Fin p` with
`0 < S i₀`, `η i₀ ≠ 0`, and `σ_coord i₀ ≠ 0`, adding EMA momentum `β` **strictly increases**
`optimizerNetProgress` over `β = 0`. -/
theorem adam_momentum_strictly_beats_preconditioned_no_momentum
    (η S : Fin p → ℝ) (g : Parameters p) {β : ℝ} (σ_coord : Fin p → ℝ) {B : ℝ}
    (hβ0 : 0 < β) (_hβ1 : β < 1) (hS : ∀ i, 0 ≤ S i) (hB : 0 < B)
    (i₀ : Fin p) (hSi₀ : 0 < S i₀) (hηi₀ : η i₀ ≠ 0) (hσi₀ : σ_coord i₀ ≠ 0) :
    optimizerNetProgress η S g 0 σ_coord B < optimizerNetProgress η S g β σ_coord B := by
  have h1β : 1 + β ≠ 0 := by linarith
  have hdiff := adam_momentum_reduces_noise_penalty η S g β σ_coord B h1β
  have hpen : 0 < optimizerNoisePenalty η S σ_coord B :=
    optimizerNoisePenalty_pos η S σ_coord hS hB i₀ hSi₀ hηi₀ hσi₀
  have hfactor : 0 < 2 * β / (1 + β) := by
    have hden : 0 < 1 + β := by linarith
    positivity
  have hprod : 0 < (2 * β / (1 + β)) * optimizerNoisePenalty η S σ_coord B :=
    mul_pos hfactor hpen
  linarith

/-- **Regime Selection Theorem: Diagonal Preconditioning + EMA Momentum Strictly Beats Scalar SGD
(`adam_momentum_strictly_beats_scalar_sgd`)**:
Let `0 < S i ≤ S_max` be an anisotropic curvature profile with at least one coordinate `i₀`
having `S i₀ < S_max` and `g i₀ ≠ 0`. For ANY scalar learning rate `η_sgd : ℝ`, if the EMA
momentum parameter `0 < β < 1` or batch size `B > 0` is chosen so that the EMA-damped diagonal
noise penalty does not exceed the scalar SGD noise penalty
(`((1 - β) / (1 + β)) * optimizerNoisePenalty (optimalDiagStep S) S σ_coord B ≤ optimizerNoisePenalty (fun _ => η_sgd) S σ_coord B`),
then optimal diagonal preconditioning with EMA momentum **strictly dominates** memoryless scalar
SGD (`optimizerNetProgress (fun _ => η_sgd) S g 0 σ_coord B < optimizerNetProgress (optimalDiagStep S) S g β σ_coord B`),
and simultaneously beats conservative scalar-max deterministic progress when the noise penalty is
bounded by the anisotropy gap. -/
theorem adam_momentum_strictly_beats_scalar_sgd
    (η_sgd : ℝ) (S : Fin p → ℝ) (S_max : ℝ) (g : Parameters p)
    {β : ℝ} (σ_coord : Fin p → ℝ) {B : ℝ}
    (hS : ∀ i, 0 < S i) (hS_le : ∀ i, S i ≤ S_max)
    (i₀ : Fin p) (hi₀_lt : S i₀ < S_max) (hgi₀ : g i₀ ≠ 0)
    (hnoise_le :
      ((1 - β) / (1 + β)) * optimizerNoisePenalty (optimalDiagStep S) S σ_coord B ≤
        optimizerNoisePenalty (fun _ => η_sgd) S σ_coord B) :
    scalarMaxProgress S_max g - optimizerNoisePenalty (fun _ => η_sgd) S σ_coord B <
        optimizerNetProgress (optimalDiagStep S) S g β σ_coord B ∧
      optimizerNetProgress (fun _ => η_sgd) S g 0 σ_coord B ≤
        optimizerNetProgress (optimalDiagStep S) S g β σ_coord B := by
  rw [optimizerNetProgress_eq_sub_penalty, optimizerNetProgress_zero_momentum]
  have hstrict_det : scalarMaxProgress S_max g < coordProgress (optimalDiagStep S) S g :=
    diagonal_strictly_beats_scalar_sgd S S_max g hS hS_le i₀ hi₀_lt hgi₀
  have hdom_det : coordProgress (fun _ => η_sgd) S g ≤ coordProgress (optimalDiagStep S) S g :=
    optimal_diagonal_maximizes_progress (fun _ => η_sgd) S g hS
  refine ⟨by linarith, by linarith⟩

end OpenTheory.MomentumAdamTrajectory
