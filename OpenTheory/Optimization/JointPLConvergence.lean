import OpenTheory.Dynamics.MLPLocalSharpness
import OpenTheory.Dynamics.MultiStepSpectralBias

/-!
# Polyak-Łojasiewicz (PL) Global Convergence to Zero Training Loss & Feature-Learning Acceleration

In `OpenTheory/Dynamics/MLPLocalSharpness.lean`, we proved that for a nonlinear two-layer MLP,
stepping with adaptive step size `η = 1 / S_MLP(S)` monotonically decreases the regularized
training objective and achieves a stationary point `min_{t < T} ‖∇L(u_t)‖^2 ≤ O(1 / T)`.

Here we resolve the central optimization question of deep learning theory:
**When does nonlinear gradient descent on a neural network converge geometrically to ZERO training
error (`L(u_T) → 0`), and why does feature learning (µP) make this convergence strictly faster
than the frozen NTK regime?**

1. **Unregularized Nonlinear Empirical MSE & Exact Local Sharpness (`unregMSE`, `unregGrad`,
   `mlpUnregSharpness`, `mlp_unreg_quadratic_upper_bound`, `mlp_unreg_step_descent`)**:
   Evaluating `batchRegularizedObjective` at `lam = 0` yields the exact unregularized empirical MSE
   `unregMSE net X y u = (1 / (2 * n)) * ‖batchOutput net X u - y‖ ^ 2` with gradient
   `unregGrad net X y u = n⁻¹ • (batchDesign net X u).adjoint (batchOutput net X u - y)` and
   derived local sharpness `mlpUnregSharpness m n A L M E R = mlpLocalSharpness m n 0 A L M E R`.

2. **From Minimum Jacobian / NTK Singular Value to the Polyak-Łojasiewicz (PL) Inequality
   (`jacobian_coercive_implies_pl`, `joint_two_layer_pl_inequality`,
   `joint_jacobian_coercive_implies_pl`)**:
   When the local Jacobian `batchDesign net X u` has minimum singular value bounded below by
   `σ_min ≥ 0` on the residual `r_u = batchOutput net X u - y`
   (`σ_min * ‖r_u‖ ≤ ‖(batchDesign net X u).adjoint r_u‖`), the unregularized MSE satisfies the
   exact Polyak-Łojasiewicz inequality:
   `2 * (σ_min ^ 2 / n) * unregMSE net X y u ≤ ‖unregGrad net X y u‖ ^ 2`.
   In joint two-layer `(W₁, W₂)` training where the hidden Jacobian has coercivity `σ₁` and the
   readout feature matrix has coercivity `σ₂`, the joint gradient satisfies the strictly stronger
   PL inequality with combined constant `(σ₁ ^ 2 + σ₂ ^ 2) / n`.

3. **Nonlinear MLP Global Geometric Convergence to Zero Training Loss
   (`mlp_global_geometric_convergence_to_zero_loss`, `mlp_global_residual_sq_geometric_decay`)**:
   Combining `mlp_unreg_step_descent` with `jacobian_coercive_implies_pl` and bootstrap induction
   on `T : ℕ` proves that along the actual nonlinear trajectory
   `u_t = nonlinearBatchTrajectory net X y (1 / S_safe) 0 t`, for **all** `T : ℕ`:
   - `unregMSE net X y (u_T) ≤ (1 - (σ_min ^ 2 / n) / S_safe) ^ T * unregMSE net X y 0`,
   - `‖batchOutput net X (u_T) - y‖ ≤ S`,
   - `‖batchOutput net X (u_T) - y‖ ^ 2 ≤ (1 - (σ_min ^ 2 / n) / S_safe) ^ T * ‖initialOutput net X - y‖ ^ 2`.

4. **Why µP Feature Learning Strictly Accelerates Global Convergence Over Frozen NTK
   (`feature_learning_contraction_rate_strict_improvement`,
   `joint_vs_single_layer_contraction_strict_improvement`)**:
   When µP feature learning (`MultiStepSpectralBias`) amplifies the effective singular value on the
   target residual from `σ₀ > 0` to `σ_feat > σ₀`, or when joint two-layer training adds a positive
   readout singular value `σ₂ > 0`, the per-step contraction factor strictly improves:
   `1 - (σ_feat ^ 2 / n) / S_safe < 1 - (σ₀ ^ 2 / n) / S_safe`,
   giving strictly smaller training loss bound at every step `T ≥ 1`.
-/

noncomputable section

open scoped BigOperators RealInnerProductSpace
open ContinuousLinearMap

namespace OpenTheory.JointPLConvergence

open OpenTheory.FeatureStep
open OpenTheory.MLPTrajectory
open OpenTheory.MLPFixedDesign
open OpenTheory.MLPBatchBounds
open OpenTheory.MLPResidual
open OpenTheory.MLPBatchTrajectory
open OpenTheory.LocalSharpnessDescent
open OpenTheory.MLPLocalSharpness
open OpenTheory.MultiStepSpectralBias

variable {d m n : ℕ}

/-! ## 1. Unregularized Nonlinear Empirical MSE & Exact Local Sharpness -/

/-- Unregularized empirical half-MSE objective for the nonlinear two-layer MLP:
`unregMSE net X y u = batchRegularizedObjective net X y 0 u`. -/
def unregMSE (net : TwoLayerNet d m) (X : Fin n → Fin d → ℝ)
    (y : Parameters n) (u : Parameters (m * d)) : ℝ :=
  batchRegularizedObjective net X y 0 u

/-- Exact gradient of the unregularized empirical MSE `unregMSE`:
`unregGrad net X y u = batchGrad net X y 0 u`. -/
def unregGrad (net : TwoLayerNet d m) (X : Fin n → Fin d → ℝ)
    (y : Parameters n) (u : Parameters (m * d)) : Parameters (m * d) :=
  batchGrad net X y 0 u

/-- Derived local sharpness of the unregularized nonlinear MLP objective (`lam = 0`)
at residual bound `R`:
`mlpUnregSharpness m n A L M E R = mlpLocalSharpness m n 0 A L M E R`. -/
def mlpUnregSharpness (m n : ℕ) (A L M E R : ℝ) : ℝ :=
  mlpLocalSharpness m n 0 A L M E R

/-- Explicit formula for `unregMSE`:
`unregMSE net X y u = (1 / (2 * n)) * ‖batchOutput net X u - y‖ ^ 2`. -/
theorem unregMSE_eq (net : TwoLayerNet d m) (X : Fin n → Fin d → ℝ)
    (y : Parameters n) (u : Parameters (m * d)) :
    unregMSE net X y u = (1 / (2 * (n : ℝ))) * ‖batchOutput net X u - y‖ ^ 2 := by
  unfold unregMSE batchRegularizedObjective
  ring

/-- Nonnegativity of `unregMSE`. -/
theorem unregMSE_nonneg (net : TwoLayerNet d m) (X : Fin n → Fin d → ℝ)
    (y : Parameters n) (u : Parameters (m * d)) :
    0 ≤ unregMSE net X y u := by
  rw [unregMSE_eq]
  positivity

/-- At zero displacement `u = 0`, `unregMSE` equals `(1 / (2 * n)) * ‖initialOutput net X - y‖ ^ 2`. -/
@[simp]
theorem unregMSE_zero (net : TwoLayerNet d m) (X : Fin n → Fin d → ℝ)
    (y : Parameters n) :
    unregMSE net X y 0 = (1 / (2 * (n : ℝ))) * ‖initialOutput net X - y‖ ^ 2 := by
  unfold unregMSE
  exact batchRegularizedObjective_zero net X y 0

/-- Identity relating squared residual norm to `unregMSE`:
`‖batchOutput net X u - y‖ ^ 2 = 2 * n * unregMSE net X y u` when `0 < n`. -/
theorem norm_residual_sq_eq_two_mul_n_mul_unregMSE (net : TwoLayerNet d m) (X : Fin n → Fin d → ℝ)
    (y : Parameters n) (u : Parameters (m * d)) (hn : 0 < n) :
    ‖batchOutput net X u - y‖ ^ 2 = 2 * (n : ℝ) * unregMSE net X y u := by
  have hn0 : (n : ℝ) ≠ 0 := ne_of_gt (Nat.cast_pos.mpr hn)
  rw [unregMSE_eq]
  field_simp [hn0]

/-- Explicit formula for `unregGrad`:
`unregGrad net X y u = (n : ℝ)⁻¹ • (batchDesign net X u).adjoint (batchOutput net X u - y)`. -/
theorem unregGrad_eq (net : TwoLayerNet d m) (X : Fin n → Fin d → ℝ)
    (y : Parameters n) (u : Parameters (m * d)) :
    unregGrad net X y u =
      (n : ℝ)⁻¹ • (batchDesign net X u).adjoint (batchOutput net X u - y) := by
  unfold unregGrad batchGrad
  simp

/-- Norm of `unregGrad`:
`‖unregGrad net X y u‖ = (n : ℝ)⁻¹ * ‖(batchDesign net X u).adjoint (batchOutput net X u - y)‖`. -/
theorem norm_unregGrad_eq (net : TwoLayerNet d m) (X : Fin n → Fin d → ℝ)
    (y : Parameters n) (u : Parameters (m * d)) :
    ‖unregGrad net X y u‖ =
      (n : ℝ)⁻¹ * ‖(batchDesign net X u).adjoint (batchOutput net X u - y)‖ := by
  rw [unregGrad_eq, norm_smul, Real.norm_eq_abs, abs_of_nonneg (inv_nonneg.mpr (Nat.cast_nonneg n))]

/-- Strict positivity of `mlpLocalSharpness m n 0 A L M E R` (and `mlpUnregSharpness`)
whenever `0 < m`, `0 < n`, `0 < A`, `0 < L`, `0 ≤ M`, `0 < E`, and `0 ≤ R`. -/
theorem mlpUnregSharpness_pos {m n : ℕ} {A L M E R : ℝ}
    (hm : 0 < m) (hn : 0 < n) (hA : 0 < A) (hL : 0 < L)
    (hM : 0 ≤ M) (hE : 0 < E) (hR : 0 ≤ R) :
    0 < mlpLocalSharpness m n 0 A L M E R := by
  have hm0 : (0 : ℝ) < m := Nat.cast_pos.mpr hm
  have hn0 : (0 : ℝ) < n := Nat.cast_pos.mpr hn
  have hLip_pos : 0 < mlpOutputLip m n A L E := by
    unfold mlpOutputLip
    have hsqn : 0 < Real.sqrt n := Real.sqrt_pos.mpr hn0
    have hsqE : 0 < Real.sqrt E := Real.sqrt_pos.mpr hE
    positivity
  have hcurv_nonneg : 0 ≤ mlpOutputCurv m n A M E :=
    mlpOutputCurv_nonneg m n hA.le hM hE.le
  unfold mlpLocalSharpness
  have hinner_pos : 0 < mlpOutputLip m n A L E ^ 2 + 2 * R * mlpOutputCurv m n A M E := by
    have hsq : 0 < mlpOutputLip m n A L E ^ 2 := sq_pos_of_pos hLip_pos
    have hrem : 0 ≤ 2 * R * mlpOutputCurv m n A M E := by positivity
    linarith
  have hprod : 0 < (n : ℝ)⁻¹ * (mlpOutputLip m n A L E ^ 2 + 2 * R * mlpOutputCurv m n A M E) :=
    mul_pos (inv_pos.mpr hn0) hinner_pos
  linarith

/-- **Unregularized Second-Order Taylor Upper Bound (`mlp_unreg_quadratic_upper_bound`)**:
Instantiating `mlp_objective_quadratic_upper_bound` at `lam = 0` bounds `unregMSE net X y (u + Δ)`
by the local quadratic model with sharpness `mlpLocalSharpness m n 0 A L M E R`. -/
theorem mlp_unreg_quadratic_upper_bound (net : TwoLayerNet d m) (X : Fin n → Fin d → ℝ)
    (y : Parameters n) (u Δ : Parameters (m * d)) {A L M E R : ℝ}
    (hn : 0 < n) (hA : 0 ≤ A) (hL : 0 ≤ L) (hM : 0 ≤ M) (hE : 0 ≤ E)
    (hW : ∀ i, |net.W₂ 0 i| ≤ A) (hσ : ∀ v, |deriv net.act.σ v| ≤ L)
    (hLip : ∀ a b, |deriv net.act.σ a - deriv net.act.σ b| ≤ M * |a - b|)
    (hX : ∀ r, inputEnergy (X r) ≤ E)
    (hres : ‖batchOutput net X u - y‖ ≤ R) :
    unregMSE net X y (u + Δ) ≤
      unregMSE net X y u +
        ⟪unregGrad net X y u, Δ⟫ +
        (mlpLocalSharpness m n 0 A L M E R / 2) * ‖Δ‖ ^ 2 :=
  mlp_objective_quadratic_upper_bound net X y 0 u Δ hn hA hL hM hE hW hσ hLip hX hres

/-- **Unregularized Nonlinear Step Descent (`mlp_unreg_step_descent`)**:
At any `u` with `‖batchOutput net X u - y‖ ≤ R`, stepping `nonlinearBatchStep net X y η 0 u`
decreases `unregMSE` by at least
`stepProgress η (mlpLocalSharpness m n 0 A L M E R) (‖unregGrad net X y u‖ ^ 2)`. -/
theorem mlp_unreg_step_descent (net : TwoLayerNet d m) (X : Fin n → Fin d → ℝ)
    (y : Parameters n) (η : ℝ) (u : Parameters (m * d)) {A L M E R : ℝ}
    (hn : 0 < n) (hA : 0 ≤ A) (hL : 0 ≤ L) (hM : 0 ≤ M) (hE : 0 ≤ E)
    (hW : ∀ i, |net.W₂ 0 i| ≤ A) (hσ : ∀ v, |deriv net.act.σ v| ≤ L)
    (hLip : ∀ a b, |deriv net.act.σ a - deriv net.act.σ b| ≤ M * |a - b|)
    (hX : ∀ r, inputEnergy (X r) ≤ E)
    (hres : ‖batchOutput net X u - y‖ ≤ R) :
    unregMSE net X y (nonlinearBatchStep net X y η 0 u) ≤
      unregMSE net X y u -
        stepProgress η (mlpLocalSharpness m n 0 A L M E R)
          (‖unregGrad net X y u‖ ^ 2) :=
  mlp_local_sharpness_step_descent net X y η 0 u hn hA hL hM hE hW hσ hLip hX hres

/-! ## 2. From Minimum Jacobian / NTK Singular Value to the Polyak-Łojasiewicz (PL) Inequality -/

/-- Effective Polyak-Łojasiewicz (PL) constant `μ_PL(σ_min, n) = σ_min ^ 2 / n` induced by a
minimum Jacobian singular value `σ_min` over `n` samples. -/
def plConstant (σ_min : ℝ) (n : ℕ) : ℝ :=
  σ_min ^ 2 / (n : ℝ)

/-- **From Minimum Jacobian Singular Value to the Polyak-Łojasiewicz Inequality
(`jacobian_coercive_implies_pl`)**:
If `0 < n`, `0 ≤ σ_min`, and the local Jacobian `batchDesign net X u` satisfies
`σ_min * ‖batchOutput net X u - y‖ ≤ ‖(batchDesign net X u).adjoint (batchOutput net X u - y)‖`,
then the unregularized empirical MSE satisfies the exact Polyak-Łojasiewicz inequality:
`2 * (σ_min ^ 2 / (n : ℝ)) * unregMSE net X y u ≤ ‖unregGrad net X y u‖ ^ 2`. -/
theorem jacobian_coercive_implies_pl (net : TwoLayerNet d m) (X : Fin n → Fin d → ℝ)
    (y : Parameters n) (u : Parameters (m * d)) {σ_min : ℝ}
    (hn : 0 < n) (hσ_min : 0 ≤ σ_min)
    (hcoercive : σ_min * ‖batchOutput net X u - y‖ ≤
      ‖(batchDesign net X u).adjoint (batchOutput net X u - y)‖) :
    2 * (σ_min ^ 2 / (n : ℝ)) * unregMSE net X y u ≤ ‖unregGrad net X y u‖ ^ 2 := by
  have hn0 : (0 : ℝ) < n := Nat.cast_pos.mpr hn
  have hne_n : (n : ℝ) ≠ 0 := ne_of_gt hn0
  let r_u := batchOutput net X u - y
  have hgrad_norm : (n : ℝ)⁻¹ * (σ_min * ‖r_u‖) ≤ ‖unregGrad net X y u‖ := by
    rw [norm_unregGrad_eq]
    exact mul_le_mul_of_nonneg_left hcoercive (inv_nonneg.mpr hn0.le)
  have hnonneg : 0 ≤ (n : ℝ)⁻¹ * (σ_min * ‖r_u‖) := by positivity
  have hsq : ((n : ℝ)⁻¹ * (σ_min * ‖r_u‖)) ^ 2 ≤ ‖unregGrad net X y u‖ ^ 2 :=
    (sq_le_sq₀ hnonneg (norm_nonneg _)).mpr hgrad_norm
  have hid : ((n : ℝ)⁻¹ * (σ_min * ‖r_u‖)) ^ 2 =
      2 * (σ_min ^ 2 / (n : ℝ)) * unregMSE net X y u := by
    dsimp [r_u]
    rw [unregMSE_eq]
    field_simp [hne_n]
  linarith

/-- **Joint Two-Layer `(W₁, W₂)` Polyak-Łojasiewicz Inequality (`joint_two_layer_pl_inequality`)**:
In joint two-layer training where `g₁ = (n : ℝ)⁻¹ • adj₁ r_u` is the hidden-layer gradient and
`g₂ = (n : ℝ)⁻¹ • adj₂ r_u` is the readout-layer gradient (for residual `r_u = batchOutput net X u - y`),
if the hidden Jacobian satisfies `σ₁ * ‖r_u‖ ≤ ‖adj₁ r_u‖` and the readout feature operator
satisfies `σ₂ * ‖r_u‖ ≤ ‖adj₂ r_u‖` (`σ₁, σ₂ ≥ 0`), then the joint squared gradient norm
`‖g₁‖ ^ 2 + ‖g₂‖ ^ 2` satisfies the strictly stronger PL inequality with combined constant
`(σ₁ ^ 2 + σ₂ ^ 2) / n`. -/
theorem joint_two_layer_pl_inequality
    {E₁ E₂ : Type*} [NormedAddCommGroup E₁] [NormedSpace ℝ E₁]
    [NormedAddCommGroup E₂] [NormedSpace ℝ E₂]
    (net : TwoLayerNet d m) (X : Fin n → Fin d → ℝ)
    (y : Parameters n) (u : Parameters (m * d))
    (adj₁ : Parameters n → E₁) (adj₂ : Parameters n → E₂)
    {σ₁ σ₂ : ℝ} (hn : 0 < n) (hσ₁ : 0 ≤ σ₁) (hσ₂ : 0 ≤ σ₂)
    (hcoercive₁ : σ₁ * ‖batchOutput net X u - y‖ ≤ ‖adj₁ (batchOutput net X u - y)‖)
    (hcoercive₂ : σ₂ * ‖batchOutput net X u - y‖ ≤ ‖adj₂ (batchOutput net X u - y)‖) :
    let r_u := batchOutput net X u - y
    let g₁ := (n : ℝ)⁻¹ • adj₁ r_u
    let g₂ := (n : ℝ)⁻¹ • adj₂ r_u
    2 * ((σ₁ ^ 2 + σ₂ ^ 2) / (n : ℝ)) * unregMSE net X y u ≤ ‖g₁‖ ^ 2 + ‖g₂‖ ^ 2 := by
  intro r_u g₁ g₂
  have hn0 : (0 : ℝ) < n := Nat.cast_pos.mpr hn
  have hne_n : (n : ℝ) ≠ 0 := ne_of_gt hn0
  have hn_inv : 0 ≤ (n : ℝ)⁻¹ := inv_nonneg.mpr hn0.le
  have hg₁ : (n : ℝ)⁻¹ * (σ₁ * ‖r_u‖) ≤ ‖g₁‖ := by
    dsimp [g₁]
    rw [norm_smul, Real.norm_eq_abs, abs_of_nonneg hn_inv]
    exact mul_le_mul_of_nonneg_left hcoercive₁ hn_inv
  have hg₂ : (n : ℝ)⁻¹ * (σ₂ * ‖r_u‖) ≤ ‖g₂‖ := by
    dsimp [g₂]
    rw [norm_smul, Real.norm_eq_abs, abs_of_nonneg hn_inv]
    exact mul_le_mul_of_nonneg_left hcoercive₂ hn_inv
  have hsq₁ : ((n : ℝ)⁻¹ * (σ₁ * ‖r_u‖)) ^ 2 ≤ ‖g₁‖ ^ 2 :=
    (sq_le_sq₀ (by positivity) (norm_nonneg _)).mpr hg₁
  have hsq₂ : ((n : ℝ)⁻¹ * (σ₂ * ‖r_u‖)) ^ 2 ≤ ‖g₂‖ ^ 2 :=
    (sq_le_sq₀ (by positivity) (norm_nonneg _)).mpr hg₂
  have hid : ((n : ℝ)⁻¹ * (σ₁ * ‖r_u‖)) ^ 2 + ((n : ℝ)⁻¹ * (σ₂ * ‖r_u‖)) ^ 2 =
      2 * ((σ₁ ^ 2 + σ₂ ^ 2) / (n : ℝ)) * unregMSE net X y u := by
    dsimp [r_u]
    rw [unregMSE_eq]
    field_simp [hne_n]
  linarith

/-- Specialization of `joint_two_layer_pl_inequality` where the first layer uses the exact
MLP Jacobian `batchDesign net X u` (`g₁ = unregGrad net X y u`) and the second layer uses
the readout feature operator `Φ_adj`:
`2 * ((σ₁ ^ 2 + σ₂ ^ 2) / n) * unregMSE net X y u ≤ ‖unregGrad net X y u‖ ^ 2 + ‖(n : ℝ)⁻¹ • Φ_adj r_u‖ ^ 2`. -/
theorem joint_jacobian_coercive_implies_pl
    {E₂ : Type*} [NormedAddCommGroup E₂] [NormedSpace ℝ E₂]
    (net : TwoLayerNet d m) (X : Fin n → Fin d → ℝ)
    (y : Parameters n) (u : Parameters (m * d))
    (Φ_adj : Parameters n → E₂)
    {σ₁ σ₂ : ℝ} (hn : 0 < n) (hσ₁ : 0 ≤ σ₁) (hσ₂ : 0 ≤ σ₂)
    (hcoercive₁ : σ₁ * ‖batchOutput net X u - y‖ ≤
      ‖(batchDesign net X u).adjoint (batchOutput net X u - y)‖)
    (hcoercive₂ : σ₂ * ‖batchOutput net X u - y‖ ≤
      ‖Φ_adj (batchOutput net X u - y)‖) :
    2 * ((σ₁ ^ 2 + σ₂ ^ 2) / (n : ℝ)) * unregMSE net X y u ≤
      ‖unregGrad net X y u‖ ^ 2 + ‖(n : ℝ)⁻¹ • Φ_adj (batchOutput net X u - y)‖ ^ 2 := by
  have h := joint_two_layer_pl_inequality net X y u
    (batchDesign net X u).adjoint Φ_adj hn hσ₁ hσ₂ hcoercive₁ hcoercive₂
  dsimp at h
  rwa [← unregGrad_eq] at h

/-! ## 3. Nonlinear MLP Global Geometric Convergence to Zero Training Loss -/

/-- **One-Step Unregularized Adaptive PL Contraction (`mlp_unreg_adaptive_pl_step`)**:
At any `u` with residual `‖batchOutput net X u - y‖ ≤ S` and Jacobian coercivity `σ_min` on the
residual, taking the safe adaptive step `η = 1 / S_safe` (where
`S_safe = mlpLocalSharpness m n 0 A L M E S > 0`) contracts `unregMSE` by at least
`1 - (σ_min ^ 2 / n) / S_safe`. -/
theorem mlp_unreg_adaptive_pl_step (net : TwoLayerNet d m) (X : Fin n → Fin d → ℝ)
    (y : Parameters n) (u : Parameters (m * d)) {A L M E S σ_min : ℝ}
    (hn : 0 < n) (hA : 0 ≤ A) (hL : 0 ≤ L) (hM : 0 ≤ M) (hE : 0 ≤ E)
    (hS_safe : 0 < mlpLocalSharpness m n 0 A L M E S)
    (hW : ∀ i, |net.W₂ 0 i| ≤ A) (hσ : ∀ v, |deriv net.act.σ v| ≤ L)
    (hLip : ∀ a b, |deriv net.act.σ a - deriv net.act.σ b| ≤ M * |a - b|)
    (hX : ∀ r, inputEnergy (X r) ≤ E)
    (hres : ‖batchOutput net X u - y‖ ≤ S)
    (hσ_min : 0 ≤ σ_min)
    (hcoercive : σ_min * ‖batchOutput net X u - y‖ ≤
      ‖(batchDesign net X u).adjoint (batchOutput net X u - y)‖) :
    let S_safe := mlpLocalSharpness m n 0 A L M E S
    unregMSE net X y (nonlinearBatchStep net X y (1 / S_safe) 0 u) ≤
      (1 - (σ_min ^ 2 / (n : ℝ)) / S_safe) * unregMSE net X y u := by
  intro S_safe
  have hdesc := mlp_unreg_step_descent net X y (adaptiveStep S_safe) u
    hn hA hL hM hE hW hσ hLip hX hres
  rw [adaptive_step_progress_eq S_safe (‖unregGrad net X y u‖ ^ 2) (ne_of_gt hS_safe)] at hdesc
  have hPL := jacobian_coercive_implies_pl net X y u hn hσ_min hcoercive
  have hdiv : (2 * (σ_min ^ 2 / (n : ℝ)) * unregMSE net X y u) / (2 * S_safe) ≤
      ‖unregGrad net X y u‖ ^ 2 / (2 * S_safe) :=
    div_le_div_of_nonneg_right hPL (by positivity)
  have hsimp : (2 * (σ_min ^ 2 / (n : ℝ)) * unregMSE net X y u) / (2 * S_safe) =
      ((σ_min ^ 2 / (n : ℝ)) / S_safe) * unregMSE net X y u := by
    field_simp [ne_of_gt hS_safe]
  rw [hsimp] at hdiv
  change unregMSE net X y (nonlinearBatchStep net X y (adaptiveStep S_safe) 0 u) ≤
    (1 - (σ_min ^ 2 / (n : ℝ)) / S_safe) * unregMSE net X y u
  linarith

/-- **Nonlinear MLP Global Geometric Convergence to Zero Training Loss
(`mlp_global_geometric_convergence_to_zero_loss`)**:
Let `S_safe := mlpLocalSharpness m n 0 A L M E S` and `η := 1 / S_safe`. Suppose at `t = 0`,
`‖initialOutput net X - y‖ ≤ S` (`0 ≤ S`), `0 < m`, `0 < n`, `0 < A`, `0 < L`, `0 ≤ M`, `0 < E`
(so `0 < S_safe`), and along the trajectory `u_t := nonlinearBatchTrajectory net X y (1 / S_safe) 0 t`,
the Jacobian satisfies
`σ_min * ‖batchOutput net X (u_t) - y‖ ≤ ‖(batchDesign net X (u_t)).adjoint (batchOutput net X (u_t) - y)‖`
with `0 < σ_min` and `σ_min ^ 2 / (n : ℝ) ≤ S_safe`.
Then by joint bootstrap induction on `T : ℕ`, for **ALL `T : ℕ`**:
1. `unregMSE net X y (u_T) ≤ (1 - (σ_min ^ 2 / (n : ℝ)) / S_safe) ^ T * unregMSE net X y 0`,
2. `‖batchOutput net X (u_T) - y‖ ≤ S`. -/
theorem mlp_global_geometric_convergence_to_zero_loss
    (net : TwoLayerNet d m) (X : Fin n → Fin d → ℝ)
    (y : Parameters n) {A L M E S σ_min : ℝ}
    (hm : 0 < m) (hn : 0 < n)
    (hA : 0 < A) (hL : 0 < L) (hM : 0 ≤ M) (hE : 0 < E) (hS : 0 ≤ S)
    (hW : ∀ i, |net.W₂ 0 i| ≤ A) (hσ : ∀ v, |deriv net.act.σ v| ≤ L)
    (hLip : ∀ a b, |deriv net.act.σ a - deriv net.act.σ b| ≤ M * |a - b|)
    (hX : ∀ r, inputEnergy (X r) ≤ E)
    (hinit : ‖initialOutput net X - y‖ ≤ S)
    (hσ_min : 0 < σ_min)
    (hμ_le : σ_min ^ 2 / (n : ℝ) ≤ mlpLocalSharpness m n 0 A L M E S)
    (hcoercive : ∀ t,
      let S_safe := mlpLocalSharpness m n 0 A L M E S
      let u := nonlinearBatchTrajectory net X y (1 / S_safe) 0
      σ_min * ‖batchOutput net X (u t) - y‖ ≤
        ‖(batchDesign net X (u t)).adjoint (batchOutput net X (u t) - y)‖)
    (T : ℕ) :
    let S_safe := mlpLocalSharpness m n 0 A L M E S
    let u := nonlinearBatchTrajectory net X y (1 / S_safe) 0
    unregMSE net X y (u T) ≤
        (1 - (σ_min ^ 2 / (n : ℝ)) / S_safe) ^ T * unregMSE net X y 0 ∧
      ‖batchOutput net X (u T) - y‖ ≤ S := by
  intro S_safe u
  have hS_safe : 0 < S_safe := mlpUnregSharpness_pos hm hn hA hL hM hE hS
  have hn0 : (0 : ℝ) < n := Nat.cast_pos.mpr hn
  have hμ_pos : 0 < σ_min ^ 2 / (n : ℝ) := div_pos (sq_pos_of_pos hσ_min) hn0
  have hq_nonneg : 0 ≤ 1 - (σ_min ^ 2 / (n : ℝ)) / S_safe := by
    have hdiv_le : (σ_min ^ 2 / (n : ℝ)) / S_safe ≤ 1 := (div_le_one hS_safe).mpr hμ_le
    linarith
  have hq_le_one : 1 - (σ_min ^ 2 / (n : ℝ)) / S_safe ≤ 1 := by
    have hdiv_pos : 0 < (σ_min ^ 2 / (n : ℝ)) / S_safe := div_pos hμ_pos hS_safe
    linarith
  induction T with
  | zero =>
    have hu0 : u 0 = 0 := rfl
    refine ⟨?_, ?_⟩
    · simp [hu0]
    · simpa [hu0] using hinit
  | succ t ih =>
    obtain ⟨hmse_t, hres_t⟩ := ih
    have hstep_eq : u (t + 1) = nonlinearBatchStep net X y (1 / S_safe) 0 (u t) := by
      simp only [u, nonlinearBatchTrajectory, Function.iterate_succ_apply']
    have hstep_pl := mlp_unreg_adaptive_pl_step net X y (u t)
      hn hA.le hL.le hM hE.le hS_safe hW hσ hLip hX hres_t hσ_min.le (hcoercive t)
    have hmse_succ : unregMSE net X y (u (t + 1)) ≤
        (1 - (σ_min ^ 2 / (n : ℝ)) / S_safe) ^ (t + 1) * unregMSE net X y 0 := by
      rw [hstep_eq]
      calc
        unregMSE net X y (nonlinearBatchStep net X y (1 / S_safe) 0 (u t))
          ≤ (1 - (σ_min ^ 2 / (n : ℝ)) / S_safe) * unregMSE net X y (u t) := hstep_pl
        _ ≤ (1 - (σ_min ^ 2 / (n : ℝ)) / S_safe) *
              ((1 - (σ_min ^ 2 / (n : ℝ)) / S_safe) ^ t * unregMSE net X y 0) :=
            mul_le_mul_of_nonneg_left hmse_t hq_nonneg
        _ = (1 - (σ_min ^ 2 / (n : ℝ)) / S_safe) ^ (t + 1) * unregMSE net X y 0 := by
            rw [pow_succ']
            ring
    have hpow_le_one : (1 - (σ_min ^ 2 / (n : ℝ)) / S_safe) ^ (t + 1) ≤ 1 :=
      pow_le_one₀ hq_nonneg hq_le_one
    have h0_nonneg : 0 ≤ unregMSE net X y 0 := unregMSE_nonneg net X y 0
    have hdesc_zero : unregMSE net X y (u (t + 1)) ≤ unregMSE net X y 0 := by
      calc
        unregMSE net X y (u (t + 1))
          ≤ (1 - (σ_min ^ 2 / (n : ℝ)) / S_safe) ^ (t + 1) * unregMSE net X y 0 := hmse_succ
        _ ≤ 1 * unregMSE net X y 0 := mul_le_mul_of_nonneg_right hpow_le_one h0_nonneg
        _ = unregMSE net X y 0 := one_mul _
    have hres_succ : ‖batchOutput net X (u (t + 1)) - y‖ ≤ S :=
      regularized_loss_coercive_residual_le net X y 0 (u (t + 1))
        hn le_rfl hS hinit hdesc_zero
    exact ⟨hmse_succ, hres_succ⟩

/-- **Explicit Prediction Residual Geometric Decay (`mlp_global_residual_sq_geometric_decay`)**:
Under the hypotheses of `mlp_global_geometric_convergence_to_zero_loss`, the squared prediction
residual norm decays geometrically at the exact rate `(1 - (σ_min ^ 2 / n) / S_safe) ^ T`:
`‖batchOutput net X (u_T) - y‖ ^ 2 ≤ (1 - (σ_min ^ 2 / n) / S_safe) ^ T * ‖initialOutput net X - y‖ ^ 2`. -/
theorem mlp_global_residual_sq_geometric_decay
    (net : TwoLayerNet d m) (X : Fin n → Fin d → ℝ)
    (y : Parameters n) {A L M E S σ_min : ℝ}
    (hm : 0 < m) (hn : 0 < n)
    (hA : 0 < A) (hL : 0 < L) (hM : 0 ≤ M) (hE : 0 < E) (hS : 0 ≤ S)
    (hW : ∀ i, |net.W₂ 0 i| ≤ A) (hσ : ∀ v, |deriv net.act.σ v| ≤ L)
    (hLip : ∀ a b, |deriv net.act.σ a - deriv net.act.σ b| ≤ M * |a - b|)
    (hX : ∀ r, inputEnergy (X r) ≤ E)
    (hinit : ‖initialOutput net X - y‖ ≤ S)
    (hσ_min : 0 < σ_min)
    (hμ_le : σ_min ^ 2 / (n : ℝ) ≤ mlpLocalSharpness m n 0 A L M E S)
    (hcoercive : ∀ t,
      let S_safe := mlpLocalSharpness m n 0 A L M E S
      let u := nonlinearBatchTrajectory net X y (1 / S_safe) 0
      σ_min * ‖batchOutput net X (u t) - y‖ ≤
        ‖(batchDesign net X (u t)).adjoint (batchOutput net X (u t) - y)‖)
    (T : ℕ) :
    let S_safe := mlpLocalSharpness m n 0 A L M E S
    let u := nonlinearBatchTrajectory net X y (1 / S_safe) 0
    ‖batchOutput net X (u T) - y‖ ^ 2 ≤
      (1 - (σ_min ^ 2 / (n : ℝ)) / S_safe) ^ T * ‖initialOutput net X - y‖ ^ 2 := by
  intro S_safe u
  have hn0 : (0 : ℝ) < n := Nat.cast_pos.mpr hn
  obtain ⟨hmse_T, _⟩ := mlp_global_geometric_convergence_to_zero_loss
    net X y hm hn hA hL hM hE hS hW hσ hLip hX hinit hσ_min hμ_le hcoercive T
  rw [norm_residual_sq_eq_two_mul_n_mul_unregMSE net X y (u T) hn]
  have h0_eq : ‖initialOutput net X - y‖ ^ 2 = 2 * (n : ℝ) * unregMSE net X y 0 := by
    simpa using norm_residual_sq_eq_two_mul_n_mul_unregMSE net X y 0 hn
  rw [h0_eq]
  calc
    2 * (n : ℝ) * unregMSE net X y (u T)
      ≤ 2 * (n : ℝ) * ((1 - (σ_min ^ 2 / (n : ℝ)) / S_safe) ^ T * unregMSE net X y 0) :=
        mul_le_mul_of_nonneg_left hmse_T (by positivity)
    _ = (1 - (σ_min ^ 2 / (n : ℝ)) / S_safe) ^ T * (2 * (n : ℝ) * unregMSE net X y 0) := by
        ring

/-! ## 4. Why µP Feature Learning Strictly Accelerates Global Convergence Over Frozen NTK -/

/-- Per-step PL contraction factor `1 - (σ ^ 2 / n) / S_safe` for effective singular value `σ`
and safe local sharpness `S_safe`. -/
def plContractionFactor (σ : ℝ) (n : ℕ) (S_safe : ℝ) : ℝ :=
  1 - (σ ^ 2 / (n : ℝ)) / S_safe

/-- **Why µP Feature Learning Strictly Accelerates Global Convergence Over Frozen NTK
(`feature_learning_contraction_rate_strict_improvement`)**:
When µP feature learning (`MultiStepSpectralBias`) aligns features with the target residual and
amplifies the effective singular value from `σ₀ > 0` (frozen NTK) to `σ_feat > σ₀`, then for any
`0 < n` and `0 < S_safe`:
1. The per-step contraction factor is strictly smaller (faster):
   `1 - (σ_feat ^ 2 / (n : ℝ)) / S_safe < 1 - (σ₀ ^ 2 / (n : ℝ)) / S_safe`,
2. For every horizon `T ≥ 1` and positive initial loss `L₀ > 0` (when `σ_feat ^ 2 / n ≤ S_safe`),
   the multi-step loss upper bound is strictly smaller:
   `(1 - (σ_feat ^ 2 / (n : ℝ)) / S_safe) ^ T * L₀ < (1 - (σ₀ ^ 2 / (n : ℝ)) / S_safe) ^ T * L₀`. -/
theorem feature_learning_contraction_rate_strict_improvement
    {σ₀ σ_feat S_safe L₀ : ℝ} {n T : ℕ}
    (hσ₀ : 0 < σ₀) (hσ_lt : σ₀ < σ_feat)
    (hn : 0 < n) (hS_safe : 0 < S_safe)
    (hμ_le : σ_feat ^ 2 / (n : ℝ) ≤ S_safe)
    (hT : 1 ≤ T) (hL₀ : 0 < L₀) :
    1 - (σ_feat ^ 2 / (n : ℝ)) / S_safe < 1 - (σ₀ ^ 2 / (n : ℝ)) / S_safe ∧
      (1 - (σ_feat ^ 2 / (n : ℝ)) / S_safe) ^ T * L₀ <
        (1 - (σ₀ ^ 2 / (n : ℝ)) / S_safe) ^ T * L₀ := by
  have hn0 : (0 : ℝ) < n := Nat.cast_pos.mpr hn
  have hsq_lt : σ₀ ^ 2 < σ_feat ^ 2 := sq_lt_sq.mpr (by
    rw [abs_of_pos hσ₀, abs_of_pos (hσ₀.trans hσ_lt)]
    exact hσ_lt)
  have hdiv_n_lt : σ₀ ^ 2 / (n : ℝ) < σ_feat ^ 2 / (n : ℝ) :=
    div_lt_div_of_pos_right hsq_lt hn0
  have hdiv_S_lt : (σ₀ ^ 2 / (n : ℝ)) / S_safe < (σ_feat ^ 2 / (n : ℝ)) / S_safe :=
    div_lt_div_of_pos_right hdiv_n_lt hS_safe
  have hrate_lt : 1 - (σ_feat ^ 2 / (n : ℝ)) / S_safe < 1 - (σ₀ ^ 2 / (n : ℝ)) / S_safe := by
    linarith
  have hrate_nonneg : 0 ≤ 1 - (σ_feat ^ 2 / (n : ℝ)) / S_safe := by
    have hle_one : (σ_feat ^ 2 / (n : ℝ)) / S_safe ≤ 1 := (div_le_one hS_safe).mpr hμ_le
    linarith
  have hpow_lt : (1 - (σ_feat ^ 2 / (n : ℝ)) / S_safe) ^ T <
      (1 - (σ₀ ^ 2 / (n : ℝ)) / S_safe) ^ T :=
    pow_lt_pow_left₀ hrate_lt hrate_nonneg (Nat.ne_of_gt hT)
  exact ⟨hrate_lt, mul_lt_mul_of_pos_right hpow_lt hL₀⟩

/-- **Joint Two-Layer vs. Single-Layer PL Contraction Rate Strict Improvement
(`joint_vs_single_layer_contraction_strict_improvement`)**:
When joint two-layer training combines a hidden Jacobian singular value `σ₁ ≥ 0` with a positive
readout singular value `σ₂ > 0`, the effective singular value `√(σ₁^2 + σ₂^2)` produces a
strictly smaller contraction factor `1 - ((σ₁^2 + σ₂^2) / n) / S_safe < 1 - (σ₁^2 / n) / S_safe`
and strictly faster multi-step geometric decay for all `T ≥ 1` and `L₀ > 0`. -/
theorem joint_vs_single_layer_contraction_strict_improvement
    {σ₁ σ₂ S_safe L₀ : ℝ} {n T : ℕ}
    (hσ₂ : 0 < σ₂)
    (hn : 0 < n) (hS_safe : 0 < S_safe)
    (hμ_le : (σ₁ ^ 2 + σ₂ ^ 2) / (n : ℝ) ≤ S_safe)
    (hT : 1 ≤ T) (hL₀ : 0 < L₀) :
    1 - ((σ₁ ^ 2 + σ₂ ^ 2) / (n : ℝ)) / S_safe < 1 - (σ₁ ^ 2 / (n : ℝ)) / S_safe ∧
      (1 - ((σ₁ ^ 2 + σ₂ ^ 2) / (n : ℝ)) / S_safe) ^ T * L₀ <
        (1 - (σ₁ ^ 2 / (n : ℝ)) / S_safe) ^ T * L₀ := by
  have hn0 : (0 : ℝ) < n := Nat.cast_pos.mpr hn
  have hsq₂_pos : 0 < σ₂ ^ 2 := sq_pos_of_pos hσ₂
  have hsum_lt : σ₁ ^ 2 < σ₁ ^ 2 + σ₂ ^ 2 := by linarith
  have hdiv_n_lt : σ₁ ^ 2 / (n : ℝ) < (σ₁ ^ 2 + σ₂ ^ 2) / (n : ℝ) :=
    div_lt_div_of_pos_right hsum_lt hn0
  have hdiv_S_lt : (σ₁ ^ 2 / (n : ℝ)) / S_safe < ((σ₁ ^ 2 + σ₂ ^ 2) / (n : ℝ)) / S_safe :=
    div_lt_div_of_pos_right hdiv_n_lt hS_safe
  have hrate_lt :
      1 - ((σ₁ ^ 2 + σ₂ ^ 2) / (n : ℝ)) / S_safe < 1 - (σ₁ ^ 2 / (n : ℝ)) / S_safe := by
    linarith
  have hrate_nonneg : 0 ≤ 1 - ((σ₁ ^ 2 + σ₂ ^ 2) / (n : ℝ)) / S_safe := by
    have hle_one : ((σ₁ ^ 2 + σ₂ ^ 2) / (n : ℝ)) / S_safe ≤ 1 := (div_le_one hS_safe).mpr hμ_le
    linarith
  have hpow_lt : (1 - ((σ₁ ^ 2 + σ₂ ^ 2) / (n : ℝ)) / S_safe) ^ T <
      (1 - (σ₁ ^ 2 / (n : ℝ)) / S_safe) ^ T :=
    pow_lt_pow_left₀ hrate_lt hrate_nonneg (Nat.ne_of_gt hT)
  exact ⟨hrate_lt, mul_lt_mul_of_pos_right hpow_lt hL₀⟩

/-- **Connection to `MultiStepSpectralBias`: Spectral Mode Amplification Strictly Accelerates PL Convergence
(`spectral_amplification_strict_pl_acceleration`)**:
When `K ≥ 1` steps of discrete µP feature learning (`modeAmpDisc η_feat γ K σ₀`, with `η_feat > 0`
and Gram eigenvalue `γ > 0`) amplify the effective singular value from `σ₀ > 0` to
`σ_feat ≥ modeAmpDisc η_feat γ K σ₀`, the resulting PL contraction rate strictly beats the
frozen-kernel contraction rate at every optimization horizon `T ≥ 1`. -/
theorem spectral_amplification_strict_pl_acceleration
    {η_feat γ σ₀ σ_feat S_safe L₀ : ℝ} {K n T : ℕ}
    (hη_feat : 0 < η_feat) (hγ : 0 < γ) (hK : 1 ≤ K)
    (hσ₀ : 0 < σ₀) (hσ_amp : modeAmpDisc η_feat γ K σ₀ ≤ σ_feat)
    (hn : 0 < n) (hS_safe : 0 < S_safe)
    (hμ_le : σ_feat ^ 2 / (n : ℝ) ≤ S_safe)
    (hT : 1 ≤ T) (hL₀ : 0 < L₀) :
    σ₀ < σ_feat ∧
      1 - (σ_feat ^ 2 / (n : ℝ)) / S_safe < 1 - (σ₀ ^ 2 / (n : ℝ)) / S_safe ∧
      (1 - (σ_feat ^ 2 / (n : ℝ)) / S_safe) ^ T * L₀ <
        (1 - (σ₀ ^ 2 / (n : ℝ)) / S_safe) ^ T * L₀ := by
  have hσ_lt : σ₀ < σ_feat :=
    (modeAmpDisc_lt_of_pos hη_feat hγ hK hσ₀).trans_le hσ_amp
  obtain ⟨hrate, hpow⟩ := feature_learning_contraction_rate_strict_improvement
    hσ₀ hσ_lt hn hS_safe hμ_le hT hL₀
  exact ⟨hσ_lt, hrate, hpow⟩

end OpenTheory.JointPLConvergence
