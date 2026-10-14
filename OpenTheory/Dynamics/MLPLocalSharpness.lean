import OpenTheory.Optimization.LocalSharpnessDescent

/-!
# Derived Local Sharpness and Horizon-Free Safe-Step Training for Two-Layer MLPs

In `OpenTheory/Optimization/LocalSharpnessDescent.lean`, the local quadratic upper bound (`hquad`)
along a gradient step was taken as an abstract hypothesis. Here we **derive the exact local
sharpness `mlpLocalSharpness` of the nonlinear two-layer MLP objective from first principles**
using the dataset-wide Taylor remainder (`MLPBatchBounds.output_remainder_le`) and Lipschitz output
control (`MLPBatchTrajectory.batchOutput_change_le`).

 Specifically, for the regularized batch training objective
`batchRegularizedObjective net X y lam u = (1 / (2n)) ‖batchOutput net X u - y‖^2 + (lam / 2) ‖u‖^2`:

1. **Exact Second-Order Taylor Upper Bound (`mlp_objective_quadratic_upper_bound`)**:
   For **any** parameter displacement `u` with `‖batchOutput net X u - y‖ ≤ R` and **any** step
   vector `Δ : Parameters (m * d)`,
   `L(u + Δ) ≤ L(u) + ⟪batchGrad net X y lam u, Δ⟫ + (mlpLocalSharpness m n lam A L M E R / 2) ‖Δ‖^2`,
   where `mlpLocalSharpness m n lam A L M E R = lam + n⁻¹ * (K_lip^2 + 2 R K_curv)`.

2. **One-Step Nonlinear Gradient Descent Progress (`mlp_local_sharpness_step_descent`,
   `mlp_adaptive_step_optimal_drop`)**:
   Evaluating at `Δ = -η • batchGrad net X y lam u` proves that `nonlinearBatchStep` decreases
   `batchRegularizedObjective` by at least `stepProgress η S_loc (‖batchGrad(u)‖^2)`, and at the
   optimal adaptive step `η = 1 / S_loc` decreases the objective by at least `‖batchGrad(u)‖^2 / (2 S_loc)`.

3. **Unconditional Horizon-Free Safe-Step Coercive Trajectory (`mlp_safe_step_coercive_trajectory`,
   `mlp_adaptive_step_cumulative_grad_sq_le`)**:
   Starting from `u_0 = 0` with `‖initialOutput net X - y‖ ≤ S` and fixing the safe sharpness
   `S_safe = mlpLocalSharpness m n lam A L M E S`, for **any** step size `0 ≤ η ≤ 2 / S_safe`
   (including `η = 1 / S_safe`) and **any** `lam > 0` (requiring **NO** weight-decay dominance
   `q_nl < 1`), the nonlinear batch trajectory `nonlinearBatchTrajectory` satisfies for **all** `t : ℕ`:
   - `batchRegularizedObjective(u_{t+1}) ≤ batchRegularizedObjective(u_t) ≤ batchRegularizedObjective(0)`,
   - `‖batchOutput net X (u_t) - y‖ ≤ S`,
   - `‖u_t‖ ≤ S / √(n * lam)`,
   - `|preShift (unflatten (u_t)) (X r) i| ≤ (S / √(n * lam)) * √E`,
   and at `η = 1 / S_safe`, the cumulative squared gradient norm over any `T` steps is bounded by
   `∑_{t < T} ‖batchGrad(u_t)‖^2 ≤ 2 * S_safe * (S^2 / (2n))`.
-/

noncomputable section

open scoped BigOperators RealInnerProductSpace
open ContinuousLinearMap

namespace OpenTheory.MLPLocalSharpness

open OpenTheory.FeatureStep
open OpenTheory.MLPTrajectory
open OpenTheory.MLPFixedDesign
open OpenTheory.MLPBatchBounds
open OpenTheory.MLPResidual
open OpenTheory.MLPBatchTrajectory
open OpenTheory.LocalSharpnessDescent

variable {d m n : ℕ}

/-! ## 1. Shifted-Network Composition and Exact Batch Gradient Identity -/

/-- `unflatten` is additive on flattened parameter vectors. -/
theorem unflatten_add (u v : Parameters (m * d)) :
    unflatten (u + v) = unflatten u + unflatten v := by
  ext i j
  rfl

/-- Composing two first-layer displacements adds the displacement matrices. -/
theorem atDisplacement_add (net : TwoLayerNet d m) (U V : Matrix (Fin m) (Fin d) ℝ) :
    atDisplacement (atDisplacement net U) V = atDisplacement net (U + V) := by
  cases net
  simp [atDisplacement, add_assoc]

/-- Evaluating `batchOutput` at `u + Δ` is identical to evaluating `batchOutput` at `Δ`
around the shifted network `atDisplacement net (unflatten u)`. -/
theorem batchOutput_add (net : TwoLayerNet d m) (X : Fin n → Fin d → ℝ)
    (u Δ : Parameters (m * d)) :
    batchOutput net X (u + Δ) =
      batchOutput (atDisplacement net (unflatten u)) X Δ := by
  simp only [batchOutput, output, unflatten_add, atDisplacement_add]

/-- The initial output of the shifted network `atDisplacement net (unflatten u)` is `batchOutput net X u`. -/
@[simp]
theorem initialOutput_atDisplacement (net : TwoLayerNet d m) (X : Fin n → Fin d → ℝ)
    (u : Parameters (m * d)) :
    initialOutput (atDisplacement net (unflatten u)) X = batchOutput net X u :=
  rfl

/-- Exact gradient of `batchRegularizedObjective net X y lam` at parameter displacement `u`:
`batchGrad(u) = n⁻¹ • J(u)ᵀ (F(u) - y) + lam • u`. -/
def batchGrad (net : TwoLayerNet d m) (X : Fin n → Fin d → ℝ)
    (y : Parameters n) (lam : ℝ) (u : Parameters (m * d)) : Parameters (m * d) :=
  (n : ℝ)⁻¹ • (batchDesign net X u).adjoint (batchOutput net X u - y) + lam • u

/-- `nonlinearBatchStep` is exactly the standard gradient descent step `u - η • batchGrad(u)`. -/
theorem nonlinearBatchStep_eq_sub_batchGrad (net : TwoLayerNet d m) (X : Fin n → Fin d → ℝ)
    (y : Parameters n) (η lam : ℝ) (u : Parameters (m * d)) :
    nonlinearBatchStep net X y η lam u = u - η • batchGrad net X y lam u := by
  unfold nonlinearBatchStep batchGrad
  rw [smul_add, smul_smul, smul_smul, sub_smul, one_smul]
  abel

/-- Inner product of `batchGrad` with any perturbation `Δ`. -/
theorem inner_batchGrad_eq (net : TwoLayerNet d m) (X : Fin n → Fin d → ℝ)
    (y : Parameters n) (lam : ℝ) (u Δ : Parameters (m * d)) :
    ⟪batchGrad net X y lam u, Δ⟫ =
      (n : ℝ)⁻¹ * ⟪batchOutput net X u - y, batchDesign net X u Δ⟫ + lam * ⟪u, Δ⟫ := by
  unfold batchGrad
  rw [inner_add_left, inner_smul_left, inner_smul_left, adjoint_inner_left]
  simp

/-! ## 2. Derived Curvature, Lipschitz, and Local Sharpness Constants -/

/-- Dataset-wide output Lipschitz constant `√n * m * A * L * √E` with respect to `‖Δ‖`. -/
def mlpOutputLip (m n : ℕ) (A L E : ℝ) : ℝ :=
  Real.sqrt n * ((m : ℝ) * A * L * Real.sqrt E)

/-- Dataset-wide second-order Taylor remainder curvature constant `√n * m * A * M * E`
with respect to `‖Δ‖^2`. -/
def mlpOutputCurv (m n : ℕ) (A M E : ℝ) : ℝ :=
  Real.sqrt n * (M * (m : ℝ) * A * E)

/-- **Derived Local Sharpness (`mlpLocalSharpness`)** of the regularized nonlinear MLP batch
objective at residual level `R`:
`S_loc(R) = lam + n⁻¹ * (mlpOutputLip^2 + 2 * R * mlpOutputCurv)`. -/
def mlpLocalSharpness (m n : ℕ) (lam A L M E R : ℝ) : ℝ :=
  lam + (n : ℝ)⁻¹ * (mlpOutputLip m n A L E ^ 2 + 2 * R * mlpOutputCurv m n A M E)

theorem mlpOutputLip_nonneg (m n : ℕ) {A L E : ℝ}
    (hA : 0 ≤ A) (hL : 0 ≤ L) (_hE : 0 ≤ E) :
    0 ≤ mlpOutputLip m n A L E := by
  unfold mlpOutputLip
  positivity

theorem mlpOutputCurv_nonneg (m n : ℕ) {A M E : ℝ}
    (hA : 0 ≤ A) (hM : 0 ≤ M) (hE : 0 ≤ E) :
    0 ≤ mlpOutputCurv m n A M E := by
  unfold mlpOutputCurv
  positivity

theorem lam_le_mlpLocalSharpness (m n : ℕ) (lam : ℝ) {A L M E R : ℝ}
    (hA : 0 ≤ A) (hM : 0 ≤ M) (hE : 0 ≤ E) (hR : 0 ≤ R) :
    lam ≤ mlpLocalSharpness m n lam A L M E R := by
  unfold mlpLocalSharpness
  have hcurv := mlpOutputCurv_nonneg m n hA hM hE
  have hterm : 0 ≤ (n : ℝ)⁻¹ * (mlpOutputLip m n A L E ^ 2 + 2 * R * mlpOutputCurv m n A M E) := by
    positivity
  linarith

theorem mlpLocalSharpness_pos (m n : ℕ) {lam A L M E R : ℝ}
    (hlam : 0 < lam) (hA : 0 ≤ A) (hM : 0 ≤ M) (hE : 0 ≤ E) (hR : 0 ≤ R) :
    0 < mlpLocalSharpness m n lam A L M E R :=
  hlam.trans_le (lam_le_mlpLocalSharpness m n lam hA hM hE hR)

theorem mlpLocalSharpness_mono_residual (m n : ℕ) (lam A L M E : ℝ) {R₁ R₂ : ℝ}
    (hA : 0 ≤ A) (hM : 0 ≤ M) (hE : 0 ≤ E) (hR : R₁ ≤ R₂) :
    mlpLocalSharpness m n lam A L M E R₁ ≤ mlpLocalSharpness m n lam A L M E R₂ := by
  unfold mlpLocalSharpness
  have hcurv := mlpOutputCurv_nonneg m n hA hM hE
  have hn : 0 ≤ (n : ℝ)⁻¹ := inv_nonneg.mpr (Nat.cast_nonneg n)
  have hmul : (n : ℝ)⁻¹ * (mlpOutputLip m n A L E ^ 2 + 2 * R₁ * mlpOutputCurv m n A M E) ≤
      (n : ℝ)⁻¹ * (mlpOutputLip m n A L E ^ 2 + 2 * R₂ * mlpOutputCurv m n A M E) := by
    apply mul_le_mul_of_nonneg_left _ hn
    linarith [mul_le_mul_of_nonneg_right hR hcurv]
  linarith

/-- For `n > 0`, the `1 / n` factor cancels `√n ^ 2` in `mlpOutputLip ^ 2`. -/
theorem mlpLocalSharpness_eq_of_pos {m n : ℕ} (hn : 0 < n) (lam A L M E R : ℝ) (hE : 0 ≤ E) :
    mlpLocalSharpness m n lam A L M E R =
      lam + (m : ℝ) ^ 2 * A ^ 2 * L ^ 2 * E +
        (2 * R / Real.sqrt n) * (M * (m : ℝ) * A * E) := by
  have hn0 : (0 : ℝ) < n := Nat.cast_pos.mpr hn
  have hnsq : Real.sqrt n ^ 2 = (n : ℝ) := Real.sq_sqrt hn0.le
  have hEsq : Real.sqrt E ^ 2 = E := Real.sq_sqrt hE
  have hne : Real.sqrt n ≠ 0 := ne_of_gt (Real.sqrt_pos.mpr hn0)
  have hne_n : (n : ℝ) ≠ 0 := ne_of_gt hn0
  unfold mlpLocalSharpness mlpOutputLip mlpOutputCurv
  have h1 : (Real.sqrt n * ((m : ℝ) * A * L * Real.sqrt E)) ^ 2 =
      (n : ℝ) * ((m : ℝ) ^ 2 * A ^ 2 * L ^ 2 * E) := by
    calc
      (Real.sqrt n * ((m : ℝ) * A * L * Real.sqrt E)) ^ 2
        = Real.sqrt n ^ 2 * ((m : ℝ) ^ 2 * A ^ 2 * L ^ 2 * Real.sqrt E ^ 2) := by ring
      _ = (n : ℝ) * ((m : ℝ) ^ 2 * A ^ 2 * L ^ 2 * E) := by rw [hnsq, hEsq]
  have h2 : (n : ℝ)⁻¹ * (2 * R * (Real.sqrt n * (M * (m : ℝ) * A * E))) =
      (2 * R / Real.sqrt n) * (M * (m : ℝ) * A * E) := by
    calc
      (n : ℝ)⁻¹ * (2 * R * (Real.sqrt n * (M * (m : ℝ) * A * E)))
        = (Real.sqrt n ^ 2)⁻¹ * (2 * R * (Real.sqrt n * (M * (m : ℝ) * A * E))) := by rw [hnsq]
      _ = (2 * R / Real.sqrt n) * (M * (m : ℝ) * A * E) := by
        field_simp [hne]
  rw [h1, mul_add, h2, inv_mul_cancel_left₀ hne_n]
  ring

/-! ## 3. Local Remainder and Output-Step Estimates Around Any Displacement `u` -/

/-- Nonlinear Taylor remainder vector `F(u + Δ) - F(u) - J(u) Δ`. -/
def stepRemainder (net : TwoLayerNet d m) (X : Fin n → Fin d → ℝ)
    (u Δ : Parameters (m * d)) : Parameters n :=
  batchOutput net X (u + Δ) - batchOutput net X u - batchDesign net X u Δ

/-- Decomposition of the displaced output: `F(u + Δ) = F(u) + J(u) Δ + stepRemainder`. -/
theorem batchOutput_add_eq (net : TwoLayerNet d m) (X : Fin n → Fin d → ℝ)
    (u Δ : Parameters (m * d)) :
    batchOutput net X (u + Δ) =
      batchOutput net X u + (batchDesign net X u Δ + stepRemainder net X u Δ) := by
  unfold stepRemainder
  abel

/-- Decomposition of the displaced residual:
`F(u + Δ) - y = (F(u) - y) + (J(u) Δ + stepRemainder)`. -/
theorem batchResidual_add_eq (net : TwoLayerNet d m) (X : Fin n → Fin d → ℝ)
    (y : Parameters n) (u Δ : Parameters (m * d)) :
    batchOutput net X (u + Δ) - y =
      (batchOutput net X u - y) + (batchDesign net X u Δ + stepRemainder net X u Δ) := by
  unfold stepRemainder
  abel

/-- The total output change `J(u) Δ + stepRemainder` is bounded by `mlpOutputLip * ‖Δ‖`. -/
theorem stepOutputChange_norm_le (net : TwoLayerNet d m) (X : Fin n → Fin d → ℝ)
    (u Δ : Parameters (m * d)) {A L E : ℝ}
    (hA : 0 ≤ A) (hL : 0 ≤ L) (hE : 0 ≤ E)
    (hW : ∀ i, |net.W₂ 0 i| ≤ A) (hσ : ∀ v, |deriv net.act.σ v| ≤ L)
    (hX : ∀ r, inputEnergy (X r) ≤ E) :
    ‖batchDesign net X u Δ + stepRemainder net X u Δ‖ ≤
      mlpOutputLip m n A L E * ‖Δ‖ := by
  have heq : batchDesign net X u Δ + stepRemainder net X u Δ =
      batchOutput (atDisplacement net (unflatten u)) X Δ -
        initialOutput (atDisplacement net (unflatten u)) X := by
    unfold stepRemainder
    rw [batchOutput_add, initialOutput_atDisplacement]
    abel
  rw [heq]
  have hchange := batchOutput_change_le (atDisplacement net (unflatten u)) X Δ hA hL hE hW hσ hX
  exact hchange.trans_eq (by unfold mlpOutputLip; ring)

/-- The nonlinear Taylor remainder `stepRemainder` is bounded by `mlpOutputCurv * ‖Δ‖ ^ 2`. -/
theorem stepRemainder_norm_le (net : TwoLayerNet d m) (X : Fin n → Fin d → ℝ)
    (u Δ : Parameters (m * d)) {A M E : ℝ}
    (hA : 0 ≤ A) (hM : 0 ≤ M) (hE : 0 ≤ E)
    (hW : ∀ i, |net.W₂ 0 i| ≤ A)
    (hLip : ∀ a b, |deriv net.act.σ a - deriv net.act.σ b| ≤ M * |a - b|)
    (hX : ∀ r, inputEnergy (X r) ≤ E) :
    ‖stepRemainder net X u Δ‖ ≤ mlpOutputCurv m n A M E * ‖Δ‖ ^ 2 := by
  let net_u := atDisplacement net (unflatten u)
  have heq : stepRemainder net X u Δ =
      output net_u X (unflatten Δ) - predict net_u X (flatten (unflatten Δ)) := by
    unfold stepRemainder predict batchDesign
    rw [batchOutput_add, flatten_unflatten]
    dsimp [net_u, batchOutput]
    abel
  rw [heq]
  have hD : 0 ≤ ‖Δ‖ * Real.sqrt E := by positivity
  have hshift : ∀ r i, |preShift (unflatten Δ) (X r) i| ≤ ‖Δ‖ * Real.sqrt E := by
    intro r i
    apply (preShift_unflatten_le_norm Δ (X r) i).trans
    exact mul_le_mul_of_nonneg_left (Real.sqrt_le_sqrt (hX r)) (norm_nonneg Δ)
  have hrem := output_remainder_le net_u X (unflatten Δ) hA hM hD hW hLip hshift
  apply hrem.trans_eq
  unfold mlpOutputCurv
  have hEsq : (‖Δ‖ * Real.sqrt E) ^ 2 = ‖Δ‖ ^ 2 * E := by
    rw [mul_pow, Real.sq_sqrt hE]
  rw [hEsq]
  ring

/-! ## 4. Exact Second-Order Taylor Upper Bound and Local-Sharpness Step Descent -/

/-- **Exact Second-Order Upper Bound for the Nonlinear MLP Objective
(`mlp_objective_quadratic_upper_bound`)**:
For **any** parameter vector `u` with residual `‖batchOutput net X u - y‖ ≤ R` and **any**
perturbation `Δ : Parameters (m * d)`, the nonlinear regularized batch objective satisfies the
local quadratic upper bound with sharpness `mlpLocalSharpness m n lam A L M E R`. -/
theorem mlp_objective_quadratic_upper_bound (net : TwoLayerNet d m) (X : Fin n → Fin d → ℝ)
    (y : Parameters n) (lam : ℝ) (u Δ : Parameters (m * d)) {A L M E R : ℝ}
    (hn : 0 < n) (hA : 0 ≤ A) (hL : 0 ≤ L) (hM : 0 ≤ M) (hE : 0 ≤ E)
    (hW : ∀ i, |net.W₂ 0 i| ≤ A) (hσ : ∀ v, |deriv net.act.σ v| ≤ L)
    (hLip : ∀ a b, |deriv net.act.σ a - deriv net.act.σ b| ≤ M * |a - b|)
    (hX : ∀ r, inputEnergy (X r) ≤ E)
    (hres : ‖batchOutput net X u - y‖ ≤ R) :
    batchRegularizedObjective net X y lam (u + Δ) ≤
      batchRegularizedObjective net X y lam u +
        ⟪batchGrad net X y lam u, Δ⟫ +
        (mlpLocalSharpness m n lam A L M E R / 2) * ‖Δ‖ ^ 2 := by
  have hn0 : (0 : ℝ) < n := Nat.cast_pos.mpr hn
  have hne_n : (n : ℝ) ≠ 0 := ne_of_gt hn0
  have hR : 0 ≤ R := (norm_nonneg _).trans hres
  let r_u := batchOutput net X u - y
  let δF := batchDesign net X u Δ + stepRemainder net X u Δ
  -- 1. Expand ‖r_{u+Δ}‖^2
  have hres_sq : ‖batchOutput net X (u + Δ) - y‖ ^ 2 =
      ‖r_u‖ ^ 2 + 2 * ⟪r_u, batchDesign net X u Δ⟫ +
        2 * ⟪r_u, stepRemainder net X u Δ⟫ + ‖δF‖ ^ 2 := by
    rw [batchResidual_add_eq net X y u Δ, norm_add_sq_real]
    dsimp [r_u, δF]
    rw [inner_add_right]
    ring
  -- 2. Bound ⟪r_u, stepRemainder⟫ ≤ R * mlpOutputCurv * ‖Δ‖^2
  have hrem_inner : ⟪r_u, stepRemainder net X u Δ⟫ ≤
      R * (mlpOutputCurv m n A M E * ‖Δ‖ ^ 2) := by
    calc
      ⟪r_u, stepRemainder net X u Δ⟫
        ≤ ‖r_u‖ * ‖stepRemainder net X u Δ‖ := real_inner_le_norm _ _
      _ ≤ R * (mlpOutputCurv m n A M E * ‖Δ‖ ^ 2) :=
        mul_le_mul hres (stepRemainder_norm_le net X u Δ hA hM hE hW hLip hX)
          (norm_nonneg _) hR
  -- 3. Bound ‖δF‖^2 ≤ mlpOutputLip^2 * ‖Δ‖^2
  have hchange_sq : ‖δF‖ ^ 2 ≤ mlpOutputLip m n A L E ^ 2 * ‖Δ‖ ^ 2 := by
    have hch := stepOutputChange_norm_le net X u Δ hA hL hE hW hσ hX
    have hnonneg : 0 ≤ mlpOutputLip m n A L E * ‖Δ‖ :=
      mul_nonneg (mlpOutputLip_nonneg m n hA hL hE) (norm_nonneg Δ)
    exact ((sq_le_sq₀ (norm_nonneg δF) hnonneg).mpr hch).trans_eq (by ring)
  -- 4. Expand ‖u + Δ‖^2
  have hu_sq : ‖u + Δ‖ ^ 2 = ‖u‖ ^ 2 + 2 * ⟪u, Δ⟫ + ‖Δ‖ ^ 2 :=
    norm_add_sq_real u Δ
  -- 5. Combine into batchRegularizedObjective
  unfold batchRegularizedObjective
  rw [hres_sq, hu_sq, inner_batchGrad_eq]
  have h1 : (1 / (2 * (n : ℝ))) *
      (‖r_u‖ ^ 2 + 2 * ⟪r_u, batchDesign net X u Δ⟫ +
        2 * ⟪r_u, stepRemainder net X u Δ⟫ + ‖δF‖ ^ 2) ≤
      (1 / (2 * (n : ℝ))) *
      (‖r_u‖ ^ 2 + 2 * ⟪r_u, batchDesign net X u Δ⟫ +
        2 * (R * (mlpOutputCurv m n A M E * ‖Δ‖ ^ 2)) +
        mlpOutputLip m n A L E ^ 2 * ‖Δ‖ ^ 2) := by
    apply mul_le_mul_of_nonneg_left _ (by positivity)
    linarith
  have hid : (1 / (2 * (n : ℝ))) *
      (‖r_u‖ ^ 2 + 2 * ⟪r_u, batchDesign net X u Δ⟫ +
        2 * (R * (mlpOutputCurv m n A M E * ‖Δ‖ ^ 2)) +
        mlpOutputLip m n A L E ^ 2 * ‖Δ‖ ^ 2) +
      lam / 2 * (‖u‖ ^ 2 + 2 * ⟪u, Δ⟫ + ‖Δ‖ ^ 2) =
      (1 / (2 * (n : ℝ)) * ‖r_u‖ ^ 2 + lam / 2 * ‖u‖ ^ 2) +
        ((n : ℝ)⁻¹ * ⟪r_u, batchDesign net X u Δ⟫ + lam * ⟪u, Δ⟫) +
        (mlpLocalSharpness m n lam A L M E R / 2) * ‖Δ‖ ^ 2 := by
    unfold mlpLocalSharpness
    field_simp [hne_n]
    ring
  dsimp [r_u] at h1 hid
  linarith

/-- Specialization of `mlp_objective_quadratic_upper_bound` to a gradient step `u - η • batchGrad(u)`. -/
theorem mlp_objective_grad_step_quadratic_le (net : TwoLayerNet d m) (X : Fin n → Fin d → ℝ)
    (y : Parameters n) (η lam : ℝ) (u : Parameters (m * d)) {A L M E R : ℝ}
    (hn : 0 < n) (hA : 0 ≤ A) (hL : 0 ≤ L) (hM : 0 ≤ M) (hE : 0 ≤ E)
    (hW : ∀ i, |net.W₂ 0 i| ≤ A) (hσ : ∀ v, |deriv net.act.σ v| ≤ L)
    (hLip : ∀ a b, |deriv net.act.σ a - deriv net.act.σ b| ≤ M * |a - b|)
    (hX : ∀ r, inputEnergy (X r) ≤ E)
    (hres : ‖batchOutput net X u - y‖ ≤ R) :
    let g := batchGrad net X y lam u
    let S_loc := mlpLocalSharpness m n lam A L M E R
    batchRegularizedObjective net X y lam (u - η • g) ≤
      batchRegularizedObjective net X y lam u -
        η * ‖g‖ ^ 2 + (S_loc / 2) * η ^ 2 * ‖g‖ ^ 2 := by
  intro g S_loc
  have hup := mlp_objective_quadratic_upper_bound net X y lam u ((-η) • g)
    hn hA hL hM hE hW hσ hLip hX hres
  have hsub : u + (-η) • g = u - η • g := by
    rw [neg_smul, ← sub_eq_add_neg]
  have hinner : ⟪g, (-η) • g⟫ = -η * ‖g‖ ^ 2 := by
    rw [inner_smul_right, real_inner_self_eq_norm_sq]
  have hnorm : ‖(-η) • g‖ ^ 2 = η ^ 2 * ‖g‖ ^ 2 := by
    rw [norm_smul, Real.norm_eq_abs, mul_pow, sq_abs, neg_sq]
  rw [hsub, hinner, hnorm] at hup
  linarith

/-- **Derived Local-Sharpness Descent for `nonlinearBatchStep` (`mlp_local_sharpness_step_descent`)**:
Whenever `‖batchOutput net X u - y‖ ≤ R`, a single step of `nonlinearBatchStep` with step size `η`
decreases `batchRegularizedObjective` by at least
`stepProgress η (mlpLocalSharpness m n lam A L M E R) (‖batchGrad net X y lam u‖ ^ 2)`. -/
theorem mlp_local_sharpness_step_descent (net : TwoLayerNet d m) (X : Fin n → Fin d → ℝ)
    (y : Parameters n) (η lam : ℝ) (u : Parameters (m * d)) {A L M E R : ℝ}
    (hn : 0 < n) (hA : 0 ≤ A) (hL : 0 ≤ L) (hM : 0 ≤ M) (hE : 0 ≤ E)
    (hW : ∀ i, |net.W₂ 0 i| ≤ A) (hσ : ∀ v, |deriv net.act.σ v| ≤ L)
    (hLip : ∀ a b, |deriv net.act.σ a - deriv net.act.σ b| ≤ M * |a - b|)
    (hX : ∀ r, inputEnergy (X r) ≤ E)
    (hres : ‖batchOutput net X u - y‖ ≤ R) :
    batchRegularizedObjective net X y lam (nonlinearBatchStep net X y η lam u) ≤
      batchRegularizedObjective net X y lam u -
        stepProgress η (mlpLocalSharpness m n lam A L M E R)
          (‖batchGrad net X y lam u‖ ^ 2) := by
  rw [nonlinearBatchStep_eq_sub_batchGrad]
  exact local_sharpness_descent (batchRegularizedObjective net X y lam)
    u (batchGrad net X y lam u) η (mlpLocalSharpness m n lam A L M E R)
    (mlp_objective_grad_step_quadratic_le net X y η lam u hn hA hL hM hE hW hσ hLip hX hres)

/-- **Optimal One-Step Adaptive Loss Drop (`mlp_adaptive_step_optimal_drop`)**:
Taking the local-sharpness adaptive step `η = adaptiveStep S_loc = 1 / S_loc` (where
`S_loc = mlpLocalSharpness m n lam A L M E R`) decreases `batchRegularizedObjective` by at least
`‖batchGrad net X y lam u‖ ^ 2 / (2 * S_loc)`. -/
theorem mlp_adaptive_step_optimal_drop (net : TwoLayerNet d m) (X : Fin n → Fin d → ℝ)
    (y : Parameters n) (lam : ℝ) (u : Parameters (m * d)) {A L M E R : ℝ}
    (hn : 0 < n) (hlam : 0 < lam) (hA : 0 ≤ A) (hL : 0 ≤ L) (hM : 0 ≤ M) (hE : 0 ≤ E)
    (hW : ∀ i, |net.W₂ 0 i| ≤ A) (hσ : ∀ v, |deriv net.act.σ v| ≤ L)
    (hLip : ∀ a b, |deriv net.act.σ a - deriv net.act.σ b| ≤ M * |a - b|)
    (hX : ∀ r, inputEnergy (X r) ≤ E)
    (hres : ‖batchOutput net X u - y‖ ≤ R) :
    let S_loc := mlpLocalSharpness m n lam A L M E R
    batchRegularizedObjective net X y lam
        (nonlinearBatchStep net X y (adaptiveStep S_loc) lam u) ≤
      batchRegularizedObjective net X y lam u -
        ‖batchGrad net X y lam u‖ ^ 2 / (2 * S_loc) := by
  intro S_loc
  have hR : 0 ≤ R := (norm_nonneg _).trans hres
  have hS_pos : 0 < S_loc := mlpLocalSharpness_pos m n hlam hA hM hE hR
  have hdesc := mlp_local_sharpness_step_descent net X y (adaptiveStep S_loc) lam u
    hn hA hL hM hE hW hσ hLip hX hres
  rwa [adaptive_step_progress_eq S_loc (‖batchGrad net X y lam u‖ ^ 2) (ne_of_gt hS_pos)] at hdesc

/-- For any safe step size `0 ≤ η ≤ 2 / S_loc` with `S_loc > 0`, `stepProgress η S_loc G ≥ 0`. -/
theorem stepProgress_nonneg_of_le_two_div {η S_loc G : ℝ}
    (hη0 : 0 ≤ η) (hS : 0 < S_loc) (hη_le : η ≤ 2 / S_loc) (hG : 0 ≤ G) :
    0 ≤ stepProgress η S_loc G := by
  unfold stepProgress
  have h1 : η * S_loc ≤ 2 := (le_div_iff₀ hS).mp hη_le
  have h2 : 0 ≤ 1 - η * S_loc / 2 := by linarith
  positivity

/-! ## 5. Horizon-Free Coercive Trajectory Without Weight-Decay Contraction (`q_nl < 1`) -/

/-- **Monotone Objective Descent Along the Nonlinear Batch Trajectory (`mlp_safe_step_objective_mono`)**:
Suppose at initialization `‖initialOutput net X - y‖ ≤ S`. Fix the safe local sharpness
`S_safe := mlpLocalSharpness m n lam A L M E S`. Then for **any** step size `0 ≤ η ≤ 2 / S_safe`
and **any** `lam ≥ 0` (with `S_safe > 0`), the actual nonlinear trajectory
`nonlinearBatchTrajectory net X y η lam` monotonically decreases `batchRegularizedObjective`
at every step `t : ℕ` and satisfies
`batchRegularizedObjective(u_{t+1}) ≤ batchRegularizedObjective(u_t) - stepProgress η S_safe ‖batchGrad(u_t)‖^2`. -/
theorem mlp_safe_step_objective_descent (net : TwoLayerNet d m) (X : Fin n → Fin d → ℝ)
    (y : Parameters n) (η lam : ℝ) {A L M E S : ℝ}
    (hn : 0 < n) (hlam : 0 ≤ lam)
    (hA : 0 ≤ A) (hL : 0 ≤ L) (hM : 0 ≤ M) (hE : 0 ≤ E) (hS : 0 ≤ S)
    (hS_safe : 0 < mlpLocalSharpness m n lam A L M E S)
    (hη0 : 0 ≤ η) (hη_le : η ≤ 2 / mlpLocalSharpness m n lam A L M E S)
    (hW : ∀ i, |net.W₂ 0 i| ≤ A) (hσ : ∀ v, |deriv net.act.σ v| ≤ L)
    (hLip : ∀ a b, |deriv net.act.σ a - deriv net.act.σ b| ≤ M * |a - b|)
    (hX : ∀ r, inputEnergy (X r) ≤ E)
    (hinit : ‖initialOutput net X - y‖ ≤ S) (t : ℕ) :
    let u := nonlinearBatchTrajectory net X y η lam
    let S_safe := mlpLocalSharpness m n lam A L M E S
    batchRegularizedObjective net X y lam (u t) ≤ batchRegularizedObjective net X y lam 0 ∧
      batchRegularizedObjective net X y lam (u (t + 1)) ≤
        batchRegularizedObjective net X y lam (u t) -
          stepProgress η S_safe (‖batchGrad net X y lam (u t)‖ ^ 2) := by
  intro u S_safe
  induction t with
  | zero =>
    have hu0 : u 0 = 0 := rfl
    have hres0 : ‖batchOutput net X (u 0) - y‖ ≤ S := by
      simpa [hu0] using hinit
    have hstep0 : u 1 = nonlinearBatchStep net X y η lam (u 0) := rfl
    refine ⟨le_rfl, ?_⟩
    rw [hstep0]
    exact mlp_local_sharpness_step_descent net X y η lam (u 0)
      hn hA hL hM hE hW hσ hLip hX hres0
  | succ k ih =>
    obtain ⟨h_le_zero, h_step_k⟩ := ih
    have hprog_nonneg : 0 ≤ stepProgress η S_safe (‖batchGrad net X y lam (u k)‖ ^ 2) :=
      stepProgress_nonneg_of_le_two_div hη0 hS_safe hη_le (sq_nonneg _)
    have h_next_le_zero : batchRegularizedObjective net X y lam (u (k + 1)) ≤
        batchRegularizedObjective net X y lam 0 := by
      linarith
    have hres_next : ‖batchOutput net X (u (k + 1)) - y‖ ≤ S :=
      regularized_loss_coercive_residual_le net X y lam (u (k + 1))
        hn hlam hS hinit h_next_le_zero
    have hstep_next : u (k + 2) = nonlinearBatchStep net X y η lam (u (k + 1)) := by
      simp only [u, nonlinearBatchTrajectory, Function.iterate_succ_apply']
    refine ⟨h_next_le_zero, ?_⟩
    rw [hstep_next]
    exact mlp_local_sharpness_step_descent net X y η lam (u (k + 1))
      hn hA hL hM hE hW hσ hLip hX hres_next

/-- **Breakthrough Theorem: Unconditional Horizon-Free Coercive Trajectory Under Safe/Adaptive Step Size
(`mlp_safe_step_coercive_trajectory`)**:
For **any** `lam > 0` (arbitrarily small, with **NO** weight-decay dominance condition `q_nl < 1`)
and **any** step size `0 ≤ η ≤ 2 / mlpLocalSharpness m n lam A L M E S` (in particular, the
optimal adaptive step `η = 1 / mlpLocalSharpness m n lam A L M E S`), the nonlinear batch GD
trajectory `u_t = nonlinearBatchTrajectory net X y η lam t` satisfies for **all** `t : ℕ`:
1. Monotone loss descent: `batchRegularizedObjective(u_{t+1}) ≤ batchRegularizedObjective(u_t) ≤ batchRegularizedObjective(0)`,
2. Non-expanding residual: `‖batchOutput net X u_t - y‖ ≤ S`,
3. Coercive ball confinement: `‖u_t‖ ≤ S / √(n * lam)`,
4. Uniform preactivation shift bound: `|preShift (unflatten u_t) (X r) i| ≤ (S / √(n * lam)) * √E`. -/
theorem mlp_safe_step_coercive_trajectory (net : TwoLayerNet d m) (X : Fin n → Fin d → ℝ)
    (y : Parameters n) (η lam : ℝ) {A L M E S : ℝ}
    (hn : 0 < n) (hlam : 0 < lam)
    (hA : 0 ≤ A) (hL : 0 ≤ L) (hM : 0 ≤ M) (hE : 0 ≤ E) (hS : 0 ≤ S)
    (hη0 : 0 ≤ η) (hη_le : η ≤ 2 / mlpLocalSharpness m n lam A L M E S)
    (hW : ∀ i, |net.W₂ 0 i| ≤ A) (hσ : ∀ v, |deriv net.act.σ v| ≤ L)
    (hLip : ∀ a b, |deriv net.act.σ a - deriv net.act.σ b| ≤ M * |a - b|)
    (hX : ∀ r, inputEnergy (X r) ≤ E)
    (hinit : ‖initialOutput net X - y‖ ≤ S) (t : ℕ) :
    let u := nonlinearBatchTrajectory net X y η lam
    batchRegularizedObjective net X y lam (u (t + 1)) ≤
        batchRegularizedObjective net X y lam (u t) ∧
      batchRegularizedObjective net X y lam (u t) ≤
        batchRegularizedObjective net X y lam 0 ∧
      ‖batchOutput net X (u t) - y‖ ≤ S ∧
      ‖u t‖ ≤ S / Real.sqrt (n * lam) ∧
      ∀ r i, |preShift (unflatten (u t)) (X r) i| ≤ (S / Real.sqrt (n * lam)) * Real.sqrt E := by
  intro u
  have hS_safe : 0 < mlpLocalSharpness m n lam A L M E S :=
    mlpLocalSharpness_pos m n hlam hA hM hE hS
  have hmono : ∀ k, batchRegularizedObjective net X y lam (u (k + 1)) ≤
      batchRegularizedObjective net X y lam (u k) := by
    intro k
    obtain ⟨_, hstep⟩ := mlp_safe_step_objective_descent net X y η lam
      hn hlam.le hA hL hM hE hS hS_safe hη0 hη_le hW hσ hLip hX hinit k
    have hprog : 0 ≤ stepProgress η (mlpLocalSharpness m n lam A L M E S)
        (‖batchGrad net X y lam (u k)‖ ^ 2) :=
      stepProgress_nonneg_of_le_two_div hη0 hS_safe hη_le (sq_nonneg _)
    linarith
  obtain ⟨h_le_zero, _⟩ := mlp_safe_step_objective_descent net X y η lam
    hn hlam.le hA hL hM hE hS hS_safe hη0 hη_le hW hσ hLip hX hinit t
  obtain ⟨hnorm, hres, hshift⟩ := descent_trajectory_horizon_free_bounds net X y lam u
    hn hlam hS hX hinit rfl hmono t
  exact ⟨hmono t, h_le_zero, hres, hnorm, hshift⟩

/-! ## 6. Stationarity and Cumulative Gradient-Norm Convergence Bound -/

/-- Nonnegativity of `batchRegularizedObjective` when `lam ≥ 0`. -/
theorem batchRegularizedObjective_nonneg (net : TwoLayerNet d m) (X : Fin n → Fin d → ℝ)
    (y : Parameters n) (lam : ℝ) (u : Parameters (m * d)) (hlam : 0 ≤ lam) :
    0 ≤ batchRegularizedObjective net X y lam u := by
  unfold batchRegularizedObjective
  positivity

/-- At the adaptive step size `η = 1 / S_safe`, every step `t : ℕ` along `nonlinearBatchTrajectory`
reduces `batchRegularizedObjective` by at least `‖batchGrad(u_t)‖^2 / (2 * S_safe)`. -/
theorem mlp_adaptive_trajectory_step_drop (net : TwoLayerNet d m) (X : Fin n → Fin d → ℝ)
    (y : Parameters n) (lam : ℝ) {A L M E S : ℝ}
    (hn : 0 < n) (hlam : 0 < lam)
    (hA : 0 ≤ A) (hL : 0 ≤ L) (hM : 0 ≤ M) (hE : 0 ≤ E) (hS : 0 ≤ S)
    (hW : ∀ i, |net.W₂ 0 i| ≤ A) (hσ : ∀ v, |deriv net.act.σ v| ≤ L)
    (hLip : ∀ a b, |deriv net.act.σ a - deriv net.act.σ b| ≤ M * |a - b|)
    (hX : ∀ r, inputEnergy (X r) ≤ E)
    (hinit : ‖initialOutput net X - y‖ ≤ S) (t : ℕ) :
    let S_safe := mlpLocalSharpness m n lam A L M E S
    let η := adaptiveStep S_safe
    let u := nonlinearBatchTrajectory net X y η lam
    batchRegularizedObjective net X y lam (u (t + 1)) ≤
      batchRegularizedObjective net X y lam (u t) -
        ‖batchGrad net X y lam (u t)‖ ^ 2 / (2 * S_safe) := by
  intro S_safe η u
  have hS_safe : 0 < S_safe := mlpLocalSharpness_pos m n hlam hA hM hE hS
  have hη0 : 0 ≤ η := by unfold η adaptiveStep; positivity
  have hη_le : η ≤ 2 / S_safe := by
    unfold η adaptiveStep
    exact div_le_div_of_nonneg_right (by norm_num) hS_safe.le
  obtain ⟨_, hstep⟩ := mlp_safe_step_objective_descent net X y η lam
    hn hlam.le hA hL hM hE hS hS_safe hη0 hη_le hW hσ hLip hX hinit t
  rwa [adaptive_step_progress_eq S_safe (‖batchGrad net X y lam (u t)‖ ^ 2) (ne_of_gt hS_safe)]
    at hstep

/-- **Telescoping Cumulative Gradient-Norm Bound (`mlp_adaptive_step_cumulative_grad_sq_le`)**:
Along the adaptive-step trajectory `u_t = nonlinearBatchTrajectory net X y (1 / S_safe) lam t`,
the sum of squared gradient norms over any horizon `T : ℕ` is bounded by
`∑_{t < T} ‖batchGrad(u_t)‖^2 ≤ 2 * S_safe * batchRegularizedObjective(0) ≤ S_safe * S^2 / n`. -/
theorem mlp_adaptive_step_cumulative_grad_sq_le (net : TwoLayerNet d m) (X : Fin n → Fin d → ℝ)
    (y : Parameters n) (lam : ℝ) {A L M E S : ℝ}
    (hn : 0 < n) (hlam : 0 < lam)
    (hA : 0 ≤ A) (hL : 0 ≤ L) (hM : 0 ≤ M) (hE : 0 ≤ E) (hS : 0 ≤ S)
    (hW : ∀ i, |net.W₂ 0 i| ≤ A) (hσ : ∀ v, |deriv net.act.σ v| ≤ L)
    (hLip : ∀ a b, |deriv net.act.σ a - deriv net.act.σ b| ≤ M * |a - b|)
    (hX : ∀ r, inputEnergy (X r) ≤ E)
    (hinit : ‖initialOutput net X - y‖ ≤ S) (T : ℕ) :
    let S_safe := mlpLocalSharpness m n lam A L M E S
    let η := adaptiveStep S_safe
    let u := nonlinearBatchTrajectory net X y η lam
    (∑ t ∈ Finset.range T, ‖batchGrad net X y lam (u t)‖ ^ 2) ≤
      2 * S_safe * (batchRegularizedObjective net X y lam 0 -
        batchRegularizedObjective net X y lam (u T)) ∧
      (∑ t ∈ Finset.range T, ‖batchGrad net X y lam (u t)‖ ^ 2) ≤
        S_safe * S ^ 2 / (n : ℝ) := by
  intro S_safe η u
  have hS_safe : 0 < S_safe := mlpLocalSharpness_pos m n hlam hA hM hE hS
  have h2S : 0 < 2 * S_safe := by positivity
  have htele : (∑ t ∈ Finset.range T, ‖batchGrad net X y lam (u t)‖ ^ 2) ≤
      2 * S_safe * (batchRegularizedObjective net X y lam 0 -
        batchRegularizedObjective net X y lam (u T)) := by
    induction T with
    | zero =>
      have hu0 : u 0 = 0 := rfl
      rw [Finset.sum_range_zero, hu0, sub_self, mul_zero]
    | succ k ih =>
      rw [Finset.sum_range_succ]
      have hdrop := mlp_adaptive_trajectory_step_drop net X y lam
        hn hlam hA hL hM hE hS hW hσ hLip hX hinit k
      have hstep_sq : ‖batchGrad net X y lam (u k)‖ ^ 2 ≤
          2 * S_safe * (batchRegularizedObjective net X y lam (u k) -
            batchRegularizedObjective net X y lam (u (k + 1))) := by
        have h1 : ‖batchGrad net X y lam (u k)‖ ^ 2 / (2 * S_safe) ≤
            batchRegularizedObjective net X y lam (u k) -
              batchRegularizedObjective net X y lam (u (k + 1)) := by linarith
        have h2 := (div_le_iff₀ h2S).mp h1
        linarith
      linarith
  refine ⟨htele, ?_⟩
  have hnonneg : 0 ≤ batchRegularizedObjective net X y lam (u T) :=
    batchRegularizedObjective_nonneg net X y lam (u T) hlam.le
  have h0_le := batchRegularizedObjective_zero_le net X y lam hS hinit
  have hn0 : (0 : ℝ) < n := Nat.cast_pos.mpr hn
  calc
    (∑ t ∈ Finset.range T, ‖batchGrad net X y lam (u t)‖ ^ 2)
      ≤ 2 * S_safe * (batchRegularizedObjective net X y lam 0 -
          batchRegularizedObjective net X y lam (u T)) := htele
    _ ≤ 2 * S_safe * (S ^ 2 / (2 * (n : ℝ))) := by
      apply mul_le_mul_of_nonneg_left _ h2S.le
      linarith
    _ = S_safe * S ^ 2 / (n : ℝ) := by
      field_simp [ne_of_gt hn0]

/-- **Sublinear Stationarity / Minimum Gradient-Norm Rate (`mlp_adaptive_step_min_grad_norm`)**:
Over any positive horizon `T > 0`, the adaptive-step nonlinear trajectory visits at least one
iterate `t < T` with squared gradient norm at most `(S_safe * S^2) / (T * n)`. -/
theorem mlp_adaptive_step_min_grad_norm (net : TwoLayerNet d m) (X : Fin n → Fin d → ℝ)
    (y : Parameters n) (lam : ℝ) {A L M E S : ℝ}
    (hn : 0 < n) (hlam : 0 < lam)
    (hA : 0 ≤ A) (hL : 0 ≤ L) (hM : 0 ≤ M) (hE : 0 ≤ E) (hS : 0 ≤ S)
    (hW : ∀ i, |net.W₂ 0 i| ≤ A) (hσ : ∀ v, |deriv net.act.σ v| ≤ L)
    (hLip : ∀ a b, |deriv net.act.σ a - deriv net.act.σ b| ≤ M * |a - b|)
    (hX : ∀ r, inputEnergy (X r) ≤ E)
    (hinit : ‖initialOutput net X - y‖ ≤ S) (T : ℕ) (hT : 0 < T) :
    let S_safe := mlpLocalSharpness m n lam A L M E S
    let η := adaptiveStep S_safe
    let u := nonlinearBatchTrajectory net X y η lam
    ∃ t < T, ‖batchGrad net X y lam (u t)‖ ^ 2 ≤ (S_safe * S ^ 2) / ((T : ℝ) * (n : ℝ)) := by
  intro S_safe η u
  have hT0 : (0 : ℝ) < T := Nat.cast_pos.mpr hT
  have hn0 : (0 : ℝ) < n := Nat.cast_pos.mpr hn
  obtain ⟨_, hsum⟩ := mlp_adaptive_step_cumulative_grad_sq_le net X y lam
    hn hlam hA hL hM hE hS hW hσ hLip hX hinit T
  by_contra! hgt
  have hlow : (T : ℝ) * ((S_safe * S ^ 2) / ((T : ℝ) * (n : ℝ))) <
      ∑ t ∈ Finset.range T, ‖batchGrad net X y lam (u t)‖ ^ 2 := by
    calc
      (T : ℝ) * ((S_safe * S ^ 2) / ((T : ℝ) * (n : ℝ)))
        = ∑ _t ∈ Finset.range T, ((S_safe * S ^ 2) / ((T : ℝ) * (n : ℝ))) := by
          simp
      _ < ∑ t ∈ Finset.range T, ‖batchGrad net X y lam (u t)‖ ^ 2 := by
        apply Finset.sum_lt_sum_of_nonempty (Finset.nonempty_range_iff.mpr (ne_of_gt hT))
        intro t ht
        exact hgt t (Finset.mem_range.mp ht)
  have heq : (T : ℝ) * ((S_safe * S ^ 2) / ((T : ℝ) * (n : ℝ))) =
      S_safe * S ^ 2 / (n : ℝ) := by
    field_simp [ne_of_gt hT0, ne_of_gt hn0]
  linarith

end OpenTheory.MLPLocalSharpness
