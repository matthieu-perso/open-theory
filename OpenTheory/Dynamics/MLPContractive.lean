import OpenTheory.Dynamics.MLPBatchTrajectory
import OpenTheory.Generalization.MLPBatchRisk

/-!
# Horizon-Free Contractive Trajectory and Statistical Risk Bounds for Nonlinear MLPs

In `OpenTheory/Dynamics/MLPBatchTrajectory.lean`, the parameter norm bound `batch_norm_le`
uses a conservative growth factor `batchGrowth ^ T` and accumulates step defects linearly in `T`.
When weight decay `lam` dominates the nonlinear Jacobian feedback (`nonlinearRate < 1`), and
using the strict contraction (`√q_gd < 1`) of the initial-Jacobian ridge GD comparator, all
trajectory envelopes (`uniformNormEnvelope`, `uniformShiftEnvelope`, `uniformResidualEnvelope`)
and the accumulated nonlinear-to-linear trajectory defect become **completely independent of `T`**.
Furthermore, the optimization term `q_gd ^ T` decays exponentially to `0` as `T → ∞`.
-/

noncomputable section

open MeasureTheory
open scoped BigOperators RealInnerProductSpace

namespace OpenTheory.MLPContractive

open OpenTheory.FeatureStep
open OpenTheory.MLPTrajectory
open OpenTheory.MLPFixedDesign
open OpenTheory.MLPBatchBounds
open OpenTheory.MLPResidual
open OpenTheory.MLPBatchTrajectory
open OpenTheory.MLPBatchRisk

variable {d m n : ℕ}

/-- One-step affine norm contraction rate for the nonlinear batch GD update. -/
def nonlinearRate (m n : ℕ) (η lam A L E : ℝ) : ℝ :=
  |1 - η * lam| +
    (η * (n : ℝ)⁻¹) * (Real.sqrt n * (Real.sqrt m * A * L * Real.sqrt E)) *
      (Real.sqrt n * (m * A * L * Real.sqrt E))

/-- One-step affine inhomogeneous norm coefficient for the initial residual bound `S`. -/
def nonlinearStepConst (m n : ℕ) (η A L E : ℝ) : ℝ :=
  (η * (n : ℝ)⁻¹) * (Real.sqrt n * (Real.sqrt m * A * L * Real.sqrt E))

/-- Horizon-free parameter norm envelope `(c_nl * S) / (1 - q_nl)`. -/
def uniformNormEnvelope (m n : ℕ) (η lam A L E S : ℝ) : ℝ :=
  (nonlinearStepConst m n η A L E * S) / (1 - nonlinearRate m n η lam A L E)

/-- Horizon-free preactivation shift envelope `uniformNormEnvelope * √E`. -/
def uniformShiftEnvelope (m n : ℕ) (η lam A L E S : ℝ) : ℝ :=
  uniformNormEnvelope m n η lam A L E S * Real.sqrt E

/-- Horizon-free batch output residual envelope `S + √n * (m * A * L * uniformShiftEnvelope)`. -/
def uniformResidualEnvelope (m n : ℕ) (η lam A L E S : ℝ) : ℝ :=
  S + Real.sqrt n * (m * A * L * uniformShiftEnvelope m n η lam A L E S)

theorem nonlinearRate_nonneg (m n : ℕ) (lam : ℝ) {η A L E : ℝ}
    (hη : 0 ≤ η) (hA : 0 ≤ A) (hL : 0 ≤ L) (_hE : 0 ≤ E) :
    0 ≤ nonlinearRate m n η lam A L E := by
  unfold nonlinearRate
  positivity

theorem nonlinearStepConst_nonneg (m n : ℕ) {η A L E : ℝ}
    (hη : 0 ≤ η) (hA : 0 ≤ A) (hL : 0 ≤ L) (_hE : 0 ≤ E) :
    0 ≤ nonlinearStepConst m n η A L E := by
  unfold nonlinearStepConst
  positivity

theorem uniformNormEnvelope_nonneg (m n : ℕ) {η lam A L E S : ℝ}
    (hη : 0 ≤ η) (hA : 0 ≤ A) (hL : 0 ≤ L) (hE : 0 ≤ E) (hS : 0 ≤ S)
    (hq : nonlinearRate m n η lam A L E < 1) :
    0 ≤ uniformNormEnvelope m n η lam A L E S := by
  unfold uniformNormEnvelope
  exact div_nonneg (mul_nonneg (nonlinearStepConst_nonneg m n hη hA hL hE) hS) (by linarith)

theorem uniformShiftEnvelope_nonneg (m n : ℕ) {η lam A L E S : ℝ}
    (hη : 0 ≤ η) (hA : 0 ≤ A) (hL : 0 ≤ L) (hE : 0 ≤ E) (hS : 0 ≤ S)
    (hq : nonlinearRate m n η lam A L E < 1) :
    0 ≤ uniformShiftEnvelope m n η lam A L E S := by
  unfold uniformShiftEnvelope
  exact mul_nonneg (uniformNormEnvelope_nonneg m n hη hA hL hE hS hq) (Real.sqrt_nonneg E)

theorem uniformResidualEnvelope_nonneg (m n : ℕ) {η lam A L E S : ℝ}
    (hη : 0 ≤ η) (hA : 0 ≤ A) (hL : 0 ≤ L) (hE : 0 ≤ E) (hS : 0 ≤ S)
    (hq : nonlinearRate m n η lam A L E < 1) :
    0 ≤ uniformResidualEnvelope m n η lam A L E S := by
  have hD := uniformShiftEnvelope_nonneg m n hη hA hL hE hS hq
  unfold uniformResidualEnvelope
  positivity

/-- For `n > 0`, the sample count `n` cancels in `nonlinearRate`, leaving
`|1 - η * lam| + η * m * √m * A^2 * L^2 * E`. -/
theorem nonlinearRate_eq_of_pos {m n : ℕ} (hn : 0 < n) (η lam A L E : ℝ) (hE : 0 ≤ E) :
    nonlinearRate m n η lam A L E =
      |1 - η * lam| + η * (m : ℝ) * Real.sqrt m * A ^ 2 * L ^ 2 * E := by
  have hn0 : (0 : ℝ) < n := Nat.cast_pos.mpr hn
  have hnsq : Real.sqrt n * Real.sqrt n = (n : ℝ) := Real.mul_self_sqrt hn0.le
  have hEsq : Real.sqrt E * Real.sqrt E = E := Real.mul_self_sqrt hE
  unfold nonlinearRate
  congr 1
  calc
    (η * (n : ℝ)⁻¹) * (Real.sqrt n * (Real.sqrt m * A * L * Real.sqrt E)) *
        (Real.sqrt n * (m * A * L * Real.sqrt E))
      = η * ((n : ℝ)⁻¹ * (Real.sqrt n * Real.sqrt n)) *
          (m : ℝ) * Real.sqrt m * A ^ 2 * L ^ 2 * (Real.sqrt E * Real.sqrt E) := by ring
    _ = η * ((n : ℝ)⁻¹ * (n : ℝ)) * (m : ℝ) * Real.sqrt m * A ^ 2 * L ^ 2 * E := by
          rw [hnsq, hEsq]
    _ = η * (m : ℝ) * Real.sqrt m * A ^ 2 * L ^ 2 * E := by
          rw [inv_mul_cancel₀ (ne_of_gt hn0)]
          ring

/-- Sufficient condition for `nonlinearRate < 1`: when `0 < η`, `η * lam ≤ 1`, and
`m * √m * A^2 * L^2 * E < lam`, the nonlinear step is strictly contractive. -/
theorem nonlinearRate_lt_one_of_lt_lam {m n : ℕ} (hn : 0 < n)
    {η lam A L E : ℝ} (hη : 0 < η) (hηlam : η * lam ≤ 1) (hE : 0 ≤ E)
    (hdom : (m : ℝ) * Real.sqrt m * A ^ 2 * L ^ 2 * E < lam) :
    nonlinearRate m n η lam A L E < 1 := by
  rw [nonlinearRate_eq_of_pos hn η lam A L E hE]
  have habs : |1 - η * lam| = 1 - η * lam := abs_of_nonneg (by linarith)
  rw [habs]
  nlinarith

/-- Residual norm at any displacement `u` bounded in terms of `‖u‖` and initial residual `S`. -/
theorem batchOutput_residual_le_of_norm (net : TwoLayerNet d m) (X : Fin n → Fin d → ℝ)
    (y : Parameters n) (u : Parameters (m * d)) {A L E S : ℝ}
    (hA : 0 ≤ A) (hL : 0 ≤ L) (hE : 0 ≤ E)
    (hW : ∀ i, |net.W₂ 0 i| ≤ A) (hσ : ∀ v, |deriv net.act.σ v| ≤ L)
    (hX : ∀ r, inputEnergy (X r) ≤ E)
    (hinit : ‖initialOutput net X - y‖ ≤ S) :
    ‖batchOutput net X u - y‖ ≤
      S + Real.sqrt n * (m * A * L * (‖u‖ * Real.sqrt E)) := by
  have hf := batchOutput_change_le net X u hA hL hE hW hσ hX
  have htri : ‖batchOutput net X u - y‖ ≤
      ‖batchOutput net X u - initialOutput net X‖ + ‖initialOutput net X - y‖ := by
    simpa using norm_add_le (batchOutput net X u - initialOutput net X) (initialOutput net X - y)
  linarith

/-- **One-step affine norm recurrence (`nonlinearBatchStep_norm_le`)**:
A single nonlinear batch GD step satisfies
`‖nonlinearBatchStep ... u‖ ≤ q_nl * ‖u‖ + c_nl * S`. -/
theorem nonlinearBatchStep_norm_le (net : TwoLayerNet d m) (X : Fin n → Fin d → ℝ)
    (y : Parameters n) (η lam : ℝ) (u : Parameters (m * d)) {A L E S : ℝ}
    (hη : 0 ≤ η) (hA : 0 ≤ A) (hL : 0 ≤ L) (hE : 0 ≤ E)
    (hW : ∀ i, |net.W₂ 0 i| ≤ A) (hσ : ∀ v, |deriv net.act.σ v| ≤ L)
    (hX : ∀ r, inputEnergy (X r) ≤ E)
    (hinit : ‖initialOutput net X - y‖ ≤ S) :
    ‖nonlinearBatchStep net X y η lam u‖ ≤
      nonlinearRate m n η lam A L E * ‖u‖ + nonlinearStepConst m n η A L E * S := by
  rw [nonlinearBatchStep]
  apply (norm_sub_le _ _).trans
  have hd : ‖(1 - η * lam) • u‖ = |1 - η * lam| * ‖u‖ := by
    rw [norm_smul, Real.norm_eq_abs]
  have hJ : ‖batchDesign net X u‖ ≤ Real.sqrt n * (Real.sqrt m * A * L * Real.sqrt E) :=
    design_opNorm_le (atDisplacement net (unflatten u)) X hA hL hE hW hσ hX
  have hres := batchOutput_residual_le_of_norm net X y u hA hL hE hW hσ hX hinit
  have hadj : ‖(batchDesign net X u).adjoint (batchOutput net X u - y)‖ ≤
      (Real.sqrt n * (Real.sqrt m * A * L * Real.sqrt E)) *
        (S + Real.sqrt n * (m * A * L * (‖u‖ * Real.sqrt E))) := by
    have hop := (batchDesign net X u).adjoint.le_opNorm (batchOutput net X u - y)
    simp only [LinearIsometryEquiv.norm_map] at hop
    exact hop.trans (mul_le_mul hJ hres (norm_nonneg _) (by positivity))
  have hb : ‖(η * (n : ℝ)⁻¹) • (batchDesign net X u).adjoint (batchOutput net X u - y)‖ ≤
      (η * (n : ℝ)⁻¹) * ((Real.sqrt n * (Real.sqrt m * A * L * Real.sqrt E)) *
        (S + Real.sqrt n * (m * A * L * (‖u‖ * Real.sqrt E)))) := by
    rw [norm_smul, Real.norm_eq_abs,
      abs_of_nonneg (mul_nonneg hη (inv_nonneg.mpr (Nat.cast_nonneg n)))]
    exact mul_le_mul_of_nonneg_left hadj (mul_nonneg hη (inv_nonneg.mpr (Nat.cast_nonneg n)))
  rw [hd]
  apply (add_le_add le_rfl hb).trans
  unfold nonlinearRate nonlinearStepConst
  linarith

/-- **Horizon-free parameter norm invariant (`contractive_norm_le`)**:
When `nonlinearRate < 1`, the nonlinear batch trajectory stays inside the fixed ball
of radius `uniformNormEnvelope` for **all** `t : ℕ`, with no exponential growth factor in `t`. -/
theorem contractive_norm_le (net : TwoLayerNet d m) (X : Fin n → Fin d → ℝ)
    (y : Parameters n) (η lam : ℝ) {A L E S : ℝ}
    (hη : 0 ≤ η) (hA : 0 ≤ A) (hL : 0 ≤ L) (hE : 0 ≤ E) (hS : 0 ≤ S)
    (hq : nonlinearRate m n η lam A L E < 1)
    (hW : ∀ i, |net.W₂ 0 i| ≤ A) (hσ : ∀ v, |deriv net.act.σ v| ≤ L)
    (hX : ∀ r, inputEnergy (X r) ≤ E)
    (hinit : ‖initialOutput net X - y‖ ≤ S) (t : ℕ) :
    ‖nonlinearBatchTrajectory net X y η lam t‖ ≤
      uniformNormEnvelope m n η lam A L E S := by
  have hq0 := nonlinearRate_nonneg m n lam hη hA hL hE
  have h1q : 0 < 1 - nonlinearRate m n η lam A L E := by linarith
  induction t with
  | zero =>
    simp only [nonlinearBatchTrajectory, Function.iterate_zero, id_eq, norm_zero]
    exact uniformNormEnvelope_nonneg m n hη hA hL hE hS hq
  | succ t ih =>
    have hstep_eq : nonlinearBatchTrajectory net X y η lam (t + 1) =
        nonlinearBatchStep net X y η lam (nonlinearBatchTrajectory net X y η lam t) := by
      simp only [nonlinearBatchTrajectory, Function.iterate_succ_apply']
    rw [hstep_eq]
    have hrec := nonlinearBatchStep_norm_le net X y η lam
      (nonlinearBatchTrajectory net X y η lam t) hη hA hL hE hW hσ hX hinit
    calc
      ‖nonlinearBatchStep net X y η lam (nonlinearBatchTrajectory net X y η lam t)‖
        ≤ nonlinearRate m n η lam A L E * ‖nonlinearBatchTrajectory net X y η lam t‖ +
            nonlinearStepConst m n η A L E * S := hrec
      _ ≤ nonlinearRate m n η lam A L E * uniformNormEnvelope m n η lam A L E S +
            nonlinearStepConst m n η A L E * S := by
        exact add_le_add_left (mul_le_mul_of_nonneg_left ih hq0) _
      _ = uniformNormEnvelope m n η lam A L E S := by
        unfold uniformNormEnvelope
        have hne : 1 - nonlinearRate m n η lam A L E ≠ 0 := ne_of_gt h1q
        field_simp [hne]
        ring

/-- **Horizon-free preactivation shift bound (`contractive_shift_le`)**:
For all `t : ℕ`, observation `r`, and hidden unit `i`, the hidden preactivation shift
is bounded by `uniformShiftEnvelope`. -/
theorem contractive_shift_le (net : TwoLayerNet d m) (X : Fin n → Fin d → ℝ)
    (y : Parameters n) (η lam : ℝ) {A L E S : ℝ}
    (hη : 0 ≤ η) (hA : 0 ≤ A) (hL : 0 ≤ L) (hE : 0 ≤ E) (hS : 0 ≤ S)
    (hq : nonlinearRate m n η lam A L E < 1)
    (hW : ∀ i, |net.W₂ 0 i| ≤ A) (hσ : ∀ v, |deriv net.act.σ v| ≤ L)
    (hX : ∀ r, inputEnergy (X r) ≤ E)
    (hinit : ‖initialOutput net X - y‖ ≤ S)
    (t : ℕ) (r : Fin n) (i : Fin m) :
    |preShift (unflatten (nonlinearBatchTrajectory net X y η lam t)) (X r) i| ≤
      uniformShiftEnvelope m n η lam A L E S := by
  have hnorm := contractive_norm_le net X y η lam hη hA hL hE hS hq hW hσ hX hinit t
  have huniv := uniformNormEnvelope_nonneg m n hη hA hL hE hS hq
  apply (preShift_unflatten_le_norm _ (X r) i).trans
  unfold uniformShiftEnvelope
  exact mul_le_mul hnorm (Real.sqrt_le_sqrt (hX r)) (Real.sqrt_nonneg _) huniv

/-- **Horizon-free batch output residual bound (`contractive_residual_le`)**:
For all `t : ℕ`, the batch residual norm is bounded by `uniformResidualEnvelope`. -/
theorem contractive_residual_le (net : TwoLayerNet d m) (X : Fin n → Fin d → ℝ)
    (y : Parameters n) (η lam : ℝ) {A L E S : ℝ}
    (hη : 0 ≤ η) (hA : 0 ≤ A) (hL : 0 ≤ L) (hE : 0 ≤ E) (hS : 0 ≤ S)
    (hq : nonlinearRate m n η lam A L E < 1)
    (hW : ∀ i, |net.W₂ 0 i| ≤ A) (hσ : ∀ v, |deriv net.act.σ v| ≤ L)
    (hX : ∀ r, inputEnergy (X r) ≤ E)
    (hinit : ‖initialOutput net X - y‖ ≤ S) (t : ℕ) :
    ‖batchOutput net X (nonlinearBatchTrajectory net X y η lam t) - y‖ ≤
      uniformResidualEnvelope m n η lam A L E S := by
  let ut := nonlinearBatchTrajectory net X y η lam t
  have hres := batchOutput_residual_le_of_norm net X y ut hA hL hE hW hσ hX hinit
  have hnorm := contractive_norm_le net X y η lam hη hA hL hE hS hq hW hσ hX hinit t
  have hshift : ‖ut‖ * Real.sqrt E ≤ uniformShiftEnvelope m n η lam A L E S :=
    mul_le_mul_of_nonneg_right hnorm (Real.sqrt_nonneg E)
  have hmul : Real.sqrt n * (m * A * L * (‖ut‖ * Real.sqrt E)) ≤
      Real.sqrt n * (m * A * L * uniformShiftEnvelope m n η lam A L E S) := by
    apply mul_le_mul_of_nonneg_left _ (Real.sqrt_nonneg n)
    exact mul_le_mul_of_nonneg_left hshift (by positivity)
  unfold uniformResidualEnvelope
  linarith

/-- **Contracted defect accumulation (`accumulated_contracting_vector_defect`)**:
In a strictly contractive vector recurrence (`‖F a - F b‖ ≤ q * ‖a - b‖` with `0 ≤ q < 1`),
bounded local step defects `≤ C` accumulate to at most `C / (1 - q)` for **all** horizons `T : ℕ`. -/
theorem accumulated_contracting_vector_defect {p : ℕ} (F : Parameters p → Parameters p)
    (u : ℕ → Parameters p) (q C : ℝ)
    (hq0 : 0 ≤ q) (hq1 : q < 1) (hC : 0 ≤ C)
    (hF : ∀ a b, ‖F a - F b‖ ≤ q * ‖a - b‖) (T : ℕ)
    (hdef : ∀ t < T, ‖u (t + 1) - F (u t)‖ ≤ C) :
    ‖u T - F^[T] (u 0)‖ ≤ C / (1 - q) := by
  have h1q : 0 < 1 - q := by linarith
  induction T with
  | zero =>
    simp only [Function.iterate_zero, id_eq, sub_self, norm_zero]
    exact div_nonneg hC h1q.le
  | succ T ih =>
    have hi := ih (fun t ht => hdef t (Nat.lt_succ_of_lt ht))
    have hd := hdef T (Nat.lt_succ_self T)
    rw [Function.iterate_succ_apply']
    have ht : ‖u (T + 1) - F (F^[T] (u 0))‖ ≤
        ‖u (T + 1) - F (u T)‖ + ‖F (u T) - F (F^[T] (u 0))‖ := by
      simpa using norm_add_le (u (T + 1) - F (u T)) (F (u T) - F (F^[T] (u 0)))
    have hl := hF (u T) (F^[T] (u 0))
    have hq_ih : q * ‖u T - F^[T] (u 0)‖ ≤ q * (C / (1 - q)) :=
      mul_le_mul_of_nonneg_left hi hq0
    have hcomb : C + q * (C / (1 - q)) = C / (1 - q) := by
      have hne : 1 - q ≠ 0 := ne_of_gt h1q
      field_simp [hne]
      ring
    linarith

/-- Under the ridge step-size condition, `linearBatchStep` is strictly contractive on
`Parameters (m * d)` with factor `√(Optim.GD.q η lam L_ridge) < 1`. -/
theorem linearBatchStep_contractive (net : TwoLayerNet d m) (X : Fin n → Fin d → ℝ)
    (y : Parameters n) (η lam : ℝ) (hlam : 0 < lam) (hη : 0 < η)
    (hstep : η * (Ridge.smoothness (design net X) (n : ℝ)⁻¹ lam : ℝ) ^ 2 < 2 * lam) :
    let q := Optim.GD.q η lam (Ridge.smoothness (design net X) (n : ℝ)⁻¹ lam)
    0 ≤ Real.sqrt q ∧ Real.sqrt q < 1 ∧
      ∀ a b : Parameters (m * d),
        ‖linearBatchStep net X y η lam a - linearBatchStep net X y η lam b‖ ≤
          Real.sqrt q * ‖a - b‖ := by
  intro q
  have hμL : lam ≤ (Ridge.smoothness (design net X) (n : ℝ)⁻¹ lam : ℝ) := by
    simp only [Ridge.smoothness, NNReal.coe_add, coe_nnnorm, Real.norm_eq_abs]
    linarith [norm_nonneg (Ridge.hessian (design net X) (n : ℝ)⁻¹ lam), le_abs_self lam]
  have hq0 : 0 ≤ q := Optim.GD.q_nonneg_of_le η lam
    (Ridge.smoothness (design net X) (n : ℝ)⁻¹ lam) hlam.le hμL
  have hq1 : q < 1 := Optim.GD.q_lt_one_of_mul_sq_lt η lam _ hη hstep
  have hsqrt1 : Real.sqrt q < 1 := by
    rw [← Real.sqrt_one]
    exact Real.sqrt_lt_sqrt hq0 hq1
  refine ⟨Real.sqrt_nonneg q, hsqrt1, fun a b => ?_⟩
  unfold linearBatchStep
  have hg : gradient (Ridge.objective (design net X) (y - initialOutput net X) (n : ℝ)⁻¹ lam) =
      Ridge.grad (design net X) (y - initialOutput net X) (n : ℝ)⁻¹ lam :=
    funext (Ridge.gradient_eq (design net X) (y - initialOutput net X) (n : ℝ)⁻¹ lam)
  rw [hg]
  have hmono := Ridge.stronglyMonotone (design net X) (y - initialOutput net X)
    (n : ℝ)⁻¹ lam (inv_nonneg.mpr (Nat.cast_nonneg n))
  have hlip : LipschitzWith (Ridge.smoothness (design net X) (n : ℝ)⁻¹ lam)
      (Ridge.grad (design net X) (y - initialOutput net X) (n : ℝ)⁻¹ lam) :=
    (Ridge.grad_lipschitz (design net X) (y - initialOutput net X) (n : ℝ)⁻¹ lam).weaken
      (by simp [Ridge.smoothness])
  have hsq := Optim.GD.step_norm_sq_le η lam hη.le
    (Ridge.grad (design net X) (y - initialOutput net X) (n : ℝ)⁻¹ lam) hmono hlip a b
  have hsq' : ‖Optim.GD.step η (Ridge.grad (design net X) (y - initialOutput net X) (n : ℝ)⁻¹ lam) a -
      Optim.GD.step η (Ridge.grad (design net X) (y - initialOutput net X) (n : ℝ)⁻¹ lam) b‖ ^ 2 ≤
      (Real.sqrt q * ‖a - b‖) ^ 2 := by
    rw [mul_pow, Real.sq_sqrt hq0]
    exact hsq
  exact (sq_le_sq₀ (norm_nonneg _) (mul_nonneg (Real.sqrt_nonneg q) (norm_nonneg _))).mp hsq'

/-- Horizon-free accumulated parameter distance between the `T`-step nonlinear batch trajectory
and independently trained initial-feature ridge GD. -/
theorem contractive_nonlinear_linear_param_gap (net : TwoLayerNet d m) (X : Fin n → Fin d → ℝ)
    (y : Parameters n) (η lam : ℝ) {A L M E S : ℝ}
    (hlam : 0 < lam) (hη : 0 < η)
    (hA : 0 ≤ A) (hL : 0 ≤ L) (hM : 0 ≤ M) (hE : 0 ≤ E) (hS : 0 ≤ S)
    (hq_nl : nonlinearRate m n η lam A L E < 1)
    (hstep : η * (Ridge.smoothness (design net X) (n : ℝ)⁻¹ lam : ℝ) ^ 2 < 2 * lam)
    (hW : ∀ i, |net.W₂ 0 i| ≤ A) (hσ : ∀ u, |deriv net.act.σ u| ≤ L)
    (hLip : ∀ a b, |deriv net.act.σ a - deriv net.act.σ b| ≤ M * |a - b|)
    (hX : ∀ r, inputEnergy (X r) ≤ E)
    (hinit : ‖initialOutput net X - y‖ ≤ S) (T : ℕ) :
    let D := uniformShiftEnvelope m n η lam A L E S
    let R := uniformResidualEnvelope m n η lam A L E S
    let B := Real.sqrt n * (M * m * A * D ^ 2)
    let G := Real.sqrt n * (Real.sqrt m * A * M * D * Real.sqrt E)
    let δ_step := (η * (n : ℝ)⁻¹) * (‖design net X‖ * B + G * R)
    let q_gd := Real.sqrt (Optim.GD.q η lam (Ridge.smoothness (design net X) (n : ℝ)⁻¹ lam))
    ‖nonlinearBatchTrajectory net X y η lam T - linearBatchTrajectory net X y η lam T‖ ≤
      δ_step / (1 - q_gd) := by
  dsimp only
  let D := uniformShiftEnvelope m n η lam A L E S
  let R := uniformResidualEnvelope m n η lam A L E S
  let B := Real.sqrt n * (M * m * A * D ^ 2)
  let G := Real.sqrt n * (Real.sqrt m * A * M * D * Real.sqrt E)
  let δ_step := (η * (n : ℝ)⁻¹) * (‖design net X‖ * B + G * R)
  let q_gd := Real.sqrt (Optim.GD.q η lam (Ridge.smoothness (design net X) (n : ℝ)⁻¹ lam))
  have hD : 0 ≤ D := uniformShiftEnvelope_nonneg m n hη.le hA hL hE hS hq_nl
  have hR : 0 ≤ R := uniformResidualEnvelope_nonneg m n hη.le hA hL hE hS hq_nl
  have hδ : 0 ≤ δ_step := by
    dsimp [δ_step, B, G]
    positivity
  obtain ⟨hq0, hq1, hF⟩ := linearBatchStep_contractive net X y η lam hlam hη hstep
  apply accumulated_contracting_vector_defect (linearBatchStep net X y η lam)
    (nonlinearBatchTrajectory net X y η lam) q_gd δ_step hq0 hq1 hδ hF T
  intro t _ht
  have hstep_eq : nonlinearBatchTrajectory net X y η lam (t + 1) =
      nonlinearBatchStep net X y η lam (nonlinearBatchTrajectory net X y η lam t) := by
    simp only [nonlinearBatchTrajectory, Function.iterate_succ_apply']
  rw [hstep_eq]
  exact batch_step_defect_norm_le net X y η lam
    (nonlinearBatchTrajectory net X y η lam t)
    hη.le hA hM hD hE hW hLip
    (contractive_shift_le net X y η lam hη.le hA hL hE hS hq_nl hW hσ hX hinit t)
    hX
    (contractive_residual_le net X y η lam hη.le hA hL hE hS hq_nl hW hσ hX hinit t)

/-- Horizon-free output-vector distance between the `T`-step nonlinear batch trajectory and
independently trained initial-feature ridge GD predictions. -/
theorem contractive_nonlinear_linear_output_gap (net : TwoLayerNet d m) (X : Fin n → Fin d → ℝ)
    (y : Parameters n) (η lam : ℝ) {A L M E S : ℝ}
    (hlam : 0 < lam) (hη : 0 < η)
    (hA : 0 ≤ A) (hL : 0 ≤ L) (hM : 0 ≤ M) (hE : 0 ≤ E) (hS : 0 ≤ S)
    (hq_nl : nonlinearRate m n η lam A L E < 1)
    (hstep : η * (Ridge.smoothness (design net X) (n : ℝ)⁻¹ lam : ℝ) ^ 2 < 2 * lam)
    (hW : ∀ i, |net.W₂ 0 i| ≤ A) (hσ : ∀ u, |deriv net.act.σ u| ≤ L)
    (hLip : ∀ a b, |deriv net.act.σ a - deriv net.act.σ b| ≤ M * |a - b|)
    (hX : ∀ r, inputEnergy (X r) ≤ E)
    (hinit : ‖initialOutput net X - y‖ ≤ S) (T : ℕ) :
    let D := uniformShiftEnvelope m n η lam A L E S
    let R := uniformResidualEnvelope m n η lam A L E S
    let B := Real.sqrt n * (M * m * A * D ^ 2)
    let G := Real.sqrt n * (Real.sqrt m * A * M * D * Real.sqrt E)
    let δ_step := (η * (n : ℝ)⁻¹) * (‖design net X‖ * B + G * R)
    let q_gd := Real.sqrt (Optim.GD.q η lam (Ridge.smoothness (design net X) (n : ℝ)⁻¹ lam))
    ‖batchOutput net X (nonlinearBatchTrajectory net X y η lam T) -
      predict net X (linearBatchTrajectory net X y η lam T)‖ ≤
      B + ‖design net X‖ * (δ_step / (1 - q_gd)) := by
  dsimp only
  let uT := nonlinearBatchTrajectory net X y η lam T
  let vT := linearBatchTrajectory net X y η lam T
  let D := uniformShiftEnvelope m n η lam A L E S
  let R := uniformResidualEnvelope m n η lam A L E S
  let B := Real.sqrt n * (M * m * A * D ^ 2)
  let G := Real.sqrt n * (Real.sqrt m * A * M * D * Real.sqrt E)
  let δ_step := (η * (n : ℝ)⁻¹) * (‖design net X‖ * B + G * R)
  let q_gd := Real.sqrt (Optim.GD.q η lam (Ridge.smoothness (design net X) (n : ℝ)⁻¹ lam))
  have hD : 0 ≤ D := uniformShiftEnvelope_nonneg m n hη.le hA hL hE hS hq_nl
  have hrem : ‖batchOutput net X uT - predict net X uT‖ ≤ B := by
    have h := output_remainder_le net X (unflatten uT) hA hM hD hW hLip
      (contractive_shift_le net X y η lam hη.le hA hL hE hS hq_nl hW hσ hX hinit T)
    simpa [batchOutput, flatten_unflatten, B] using h
  have hpred : ‖predict net X uT - predict net X vT‖ ≤
      ‖design net X‖ * (δ_step / (1 - q_gd)) := by
    have hsub : predict net X uT - predict net X vT = design net X (uT - vT) := by
      simp [predict, map_sub]
    rw [hsub]
    exact ((design net X).le_opNorm (uT - vT)).trans
      (mul_le_mul_of_nonneg_left
        (contractive_nonlinear_linear_param_gap net X y η lam hlam hη hA hL hM hE hS
          hq_nl hstep hW hσ hLip hX hinit T)
        (norm_nonneg _))
  have htri : ‖batchOutput net X uT - predict net X vT‖ ≤
      ‖batchOutput net X uT - predict net X uT‖ + ‖predict net X uT - predict net X vT‖ := by
    simpa using norm_add_le (batchOutput net X uT - predict net X uT) (predict net X uT - predict net X vT)
  exact htri.trans (add_le_add hrem hpred)

/-- Horizon-free prediction gap budget between `T`-step nonlinear batch GD and the
initial-Jacobian ridge prediction. Notice that `D`, `R`, `B`, `G`, and `δ_step` are
completely independent of `T`, while `q_gd ^ T` decays geometrically to `0` as `T → ∞`. -/
def uniformBatchGap (T m n : ℕ) (η lam A L M E J q S : ℝ) : ℝ :=
  let D := uniformShiftEnvelope m n η lam A L E S
  let R := uniformResidualEnvelope m n η lam A L E S
  let B := Real.sqrt n * (M * m * A * D ^ 2)
  let G := Real.sqrt n * (Real.sqrt m * A * M * D * Real.sqrt E)
  let δ_step := (η * (n : ℝ)⁻¹) * (J * B + G * R)
  let q_gd := Real.sqrt q
  B + J * (δ_step / (1 - q_gd)) +
    q_gd ^ T * (S / Real.sqrt (n * lam) * J)

/-- **Horizon-Free Nonlinear-to-Ridge Prediction Gap (`contractive_batch_trained_ridge_gap`)**:
When `nonlinearRate < 1` and `η * L_ridge^2 < 2 * lam`, the `T`-step nonlinear batch network
is within `uniformBatchGap` of `ridgePrediction` for **all** `T : ℕ`. -/
theorem contractive_batch_trained_ridge_gap (net : TwoLayerNet d m) (X : Fin n → Fin d → ℝ)
    (y : Parameters n) (η lam : ℝ) {A L M E S : ℝ}
    (hn : 0 < n) (hlam : 0 < lam) (hη : 0 < η)
    (hA : 0 ≤ A) (hL : 0 ≤ L) (hM : 0 ≤ M) (hE : 0 ≤ E) (hS : 0 ≤ S)
    (hq_nl : nonlinearRate m n η lam A L E < 1)
    (hstep : η * (Ridge.smoothness (design net X) (n : ℝ)⁻¹ lam : ℝ) ^ 2 < 2 * lam)
    (hW : ∀ i, |net.W₂ 0 i| ≤ A) (hσ : ∀ u, |deriv net.act.σ u| ≤ L)
    (hLip : ∀ a b, |deriv net.act.σ a - deriv net.act.σ b| ≤ M * |a - b|)
    (hX : ∀ r, inputEnergy (X r) ≤ E)
    (hinit : ‖initialOutput net X - y‖ ≤ S) (T : ℕ) :
    ‖batchOutput net X (nonlinearBatchTrajectory net X y η lam T) -
      ridgePrediction net X y lam hn hlam‖ ≤
      uniformBatchGap T m n η lam A L M E ‖design net X‖
        (Optim.GD.q η lam (Ridge.smoothness (design net X) (n : ℝ)⁻¹ lam)) S := by
  let uT := nonlinearBatchTrajectory net X y η lam T
  let vT := linearBatchTrajectory net X y η lam T
  let q := Optim.GD.q η lam (Ridge.smoothness (design net X) (n : ℝ)⁻¹ lam)
  let uStar := RidgeRisk.ridgeSolution (design net X) (y - initialOutput net X)
    (n : ℝ)⁻¹ lam (inv_pos.mpr (Nat.cast_pos.mpr hn)) hlam
  have hnl := contractive_nonlinear_linear_output_gap net X y η lam hlam hη hA hL hM hE hS
    hq_nl hstep hW hσ hLip hX hinit T
  obtain ⟨uStar', huStar', hgd⟩ := gd_to_ridge net X y lam hn hlam η hη hstep
  have hgd0 : ‖vT - uStar‖ ^ 2 ≤ q ^ T * ‖uStar‖ ^ 2 := by
    have h0 := hgd 0 T
    rw [huStar'] at h0
    simpa [vT, linearBatchTrajectory, linearBatchStep, zero_sub, norm_neg, uStar, q] using h0
  have hμL : lam ≤ (Ridge.smoothness (design net X) (n : ℝ)⁻¹ lam : ℝ) := by
    simp only [Ridge.smoothness, NNReal.coe_add, coe_nnnorm, Real.norm_eq_abs]
    linarith [norm_nonneg (Ridge.hessian (design net X) (n : ℝ)⁻¹ lam), le_abs_self lam]
  have hqnn : 0 ≤ q := Optim.GD.q_nonneg_of_le η lam
    (Ridge.smoothness (design net X) (n : ℝ)⁻¹ lam) hlam.le hμL
  have hstar_norm := ridgeSolution_norm_le_of_init net X y lam hn hlam hS hinit
  have hRnn : 0 ≤ S / Real.sqrt (n * lam) := div_nonneg hS (Real.sqrt_nonneg _)
  have hdist : ‖vT - uStar‖ ≤ Real.sqrt q ^ T * (S / Real.sqrt (n * lam)) := by
    have h1 : ‖vT - uStar‖ ^ 2 ≤ (Real.sqrt q ^ T * (S / Real.sqrt (n * lam))) ^ 2 := by
      rw [mul_pow, ← pow_mul, mul_comm T 2, pow_mul, Real.sq_sqrt hqnn]
      exact hgd0.trans (mul_le_mul_of_nonneg_left
        ((sq_le_sq₀ (norm_nonneg _) hRnn).mpr hstar_norm) (pow_nonneg hqnn T))
    exact (sq_le_sq₀ (norm_nonneg _) (mul_nonneg (pow_nonneg (Real.sqrt_nonneg q) T) hRnn)).mp h1
  have hopt_pred : ‖predict net X vT - ridgePrediction net X y lam hn hlam‖ ≤
      Real.sqrt q ^ T * (S / Real.sqrt (n * lam) * ‖design net X‖) := by
    have hsub : predict net X vT - ridgePrediction net X y lam hn hlam =
        design net X (vT - uStar) := by
      simp [ridgePrediction, predict, uStar, map_sub]
    rw [hsub]
    calc
      ‖design net X (vT - uStar)‖ ≤ ‖design net X‖ * ‖vT - uStar‖ :=
        (design net X).le_opNorm (vT - uStar)
      _ ≤ ‖design net X‖ * (Real.sqrt q ^ T * (S / Real.sqrt (n * lam))) :=
        mul_le_mul_of_nonneg_left hdist (norm_nonneg _)
      _ = Real.sqrt q ^ T * (S / Real.sqrt (n * lam) * ‖design net X‖) := by ring
  have htri : ‖batchOutput net X uT - ridgePrediction net X y lam hn hlam‖ ≤
      ‖batchOutput net X uT - predict net X vT‖ +
        ‖predict net X vT - ridgePrediction net X y lam hn hlam‖ := by
    simpa using norm_add_le
      (batchOutput net X uT - predict net X vT)
      (predict net X vT - ridgePrediction net X y lam hn hlam)
  exact htri.trans (by
    unfold uniformBatchGap
    exact add_le_add hnl hopt_pred)

/-- Pointwise excess squared error of the `T`-step contractive nonlinear batch GD network
relative to `predict net X uStar`, for any noise vector `z` with `‖z‖ ≤ Bz`. -/
theorem contractive_batch_pointwise_error_le (net : TwoLayerNet d m) (X : Fin n → Fin d → ℝ)
    (uStar : Parameters (m * d)) (z : Parameters n) (η lam : ℝ) {A L M E Bz : ℝ}
    (hn : 0 < n) (hlam : 0 < lam) (hη : 0 < η)
    (hA : 0 ≤ A) (hL : 0 ≤ L) (hM : 0 ≤ M) (hE : 0 ≤ E) (hBz : 0 ≤ Bz)
    (hq_nl : nonlinearRate m n η lam A L E < 1)
    (hstep : η * (Ridge.smoothness (design net X) (n : ℝ)⁻¹ lam : ℝ) ^ 2 < 2 * lam)
    (hW : ∀ i, |net.W₂ 0 i| ≤ A) (hσ : ∀ u, |deriv net.act.σ u| ≤ L)
    (hLip : ∀ a b, |deriv net.act.σ a - deriv net.act.σ b| ≤ M * |a - b|)
    (hX : ∀ r, inputEnergy (X r) ≤ E)
    (hz : ‖z‖ ≤ Bz) (T : ℕ) :
    let S := ‖design net X uStar‖ + Bz
    let q := Optim.GD.q η lam (Ridge.smoothness (design net X) (n : ℝ)⁻¹ lam)
    let δ := uniformBatchGap T m n η lam A L M E ‖design net X‖ q S
    ‖trainedBatchOutput net X (predict net X uStar + z) η lam T - predict net X uStar‖ ^ 2 ≤
      2 * ‖ridgePrediction net X (predict net X uStar + z) lam hn hlam -
        predict net X uStar‖ ^ 2 + 2 * δ ^ 2 := by
  intro S q δ
  have hS : 0 ≤ S := add_nonneg (norm_nonneg _) hBz
  have hinit : ‖initialOutput net X - (predict net X uStar + z)‖ ≤ S :=
    initial_residual_le_of_noise net X uStar hz
  have hgap : ‖trainedBatchOutput net X (predict net X uStar + z) η lam T -
      ridgePrediction net X (predict net X uStar + z) lam hn hlam‖ ≤ δ := by
    unfold trainedBatchOutput
    exact contractive_batch_trained_ridge_gap net X (predict net X uStar + z) η lam hn hlam hη
      hA hL hM hE hS hq_nl hstep hW hσ hLip hX hinit T
  have hδ : 0 ≤ δ := le_trans (norm_nonneg _) hgap
  exact PredictionGap.sq_norm_le_double
    (trainedBatchOutput net X (predict net X uStar + z) η lam T)
    (ridgePrediction net X (predict net X uStar + z) lam hn hlam)
    (predict net X uStar) hδ hgap

/-- **Horizon-Free Expected In-Sample MSE for Nonlinear MLP Training (`contractive_batch_expected_mse`)**:
Under white label noise `RidgeRisk.LabelNoise μ σ` with `‖z‖ ≤ Bz` almost surely, when
`nonlinearRate < 1` and `η * L_ridge^2 < 2 * lam`, the expected in-sample MSE of the `T`-step
nonlinear batch GD network is bounded by twice the initial-Jacobian ridge risk plus `2 δ^2 / n`,
where `δ = uniformBatchGap ...` has a horizon-free nonlinear floor and an exponentially decaying
optimization term `q_gd ^ T`. -/
theorem contractive_batch_expected_mse (net : TwoLayerNet d m) (X : Fin n → Fin d → ℝ)
    (uStar : Parameters (m * d)) (η lam : ℝ) {A L M E Bz : ℝ}
    (hn : 0 < n) (hlam : 0 < lam) (hη : 0 < η)
    (hA : 0 ≤ A) (hL : 0 ≤ L) (hM : 0 ≤ M) (hE : 0 ≤ E) (hBz : 0 ≤ Bz)
    (hq_nl : nonlinearRate m n η lam A L E < 1)
    (hstep : η * (Ridge.smoothness (design net X) (n : ℝ)⁻¹ lam : ℝ) ^ 2 < 2 * lam)
    (hW : ∀ i, |net.W₂ 0 i| ≤ A) (hσ : ∀ u, |deriv net.act.σ u| ≤ L)
    (hLip : ∀ a b, |deriv net.act.σ a - deriv net.act.σ b| ≤ M * |a - b|)
    (hX : ∀ r, inputEnergy (X r) ≤ E)
    {μ : Measure (Parameters n)} {σ : ℝ} [IsProbabilityMeasure μ]
    (hnoise : RidgeRisk.LabelNoise μ σ)
    (hz_ae : ∀ᵐ z ∂μ, ‖z‖ ≤ Bz)
    (T : ℕ)
    (hint_trained : Integrable (fun z =>
      ‖trainedBatchOutput net X (predict net X uStar + z) η lam T -
        predict net X uStar‖ ^ 2) μ) :
    let S := ‖design net X uStar‖ + Bz
    let q := Optim.GD.q η lam (Ridge.smoothness (design net X) (n : ℝ)⁻¹ lam)
    let δ := uniformBatchGap T m n η lam A L M E ‖design net X‖ q S
    (∫ z, ‖trainedBatchOutput net X (predict net X uStar + z) η lam T -
      predict net X uStar‖ ^ 2 ∂μ) / n ≤
      2 * ((lam / 4) * ‖uStar‖ ^ 2 +
        σ ^ 2 * RidgeRisk.degreesOfFreedom (design net X) (n : ℝ)⁻¹ lam
          (inv_pos.mpr (Nat.cast_pos.mpr hn)) hlam / n) +
        2 * δ ^ 2 / n := by
  intro S q δ
  have hpoint : ∀ᵐ z ∂μ,
      ‖trainedBatchOutput net X (predict net X uStar + z) η lam T - predict net X uStar‖ ^ 2 ≤
        2 * ‖ridgePrediction net X (predict net X uStar + z) lam hn hlam -
          predict net X uStar‖ ^ 2 + 2 * δ ^ 2 := by
    filter_upwards [hz_ae] with z hz
    exact contractive_batch_pointwise_error_le net X uStar z η lam hn hlam hη
      hA hL hM hE hBz hq_nl hstep hW hσ hLip hX hz T
  have hq_int := integrable_ridge_error_sq net X uStar lam hn hlam hnoise
  have hconst : Integrable (fun _ : Parameters n => 2 * δ ^ 2) μ := integrable_const _
  have hupper : Integrable (fun z =>
      2 * ‖ridgePrediction net X (predict net X uStar + z) lam hn hlam -
        predict net X uStar‖ ^ 2 + 2 * δ ^ 2) μ :=
    (hq_int.const_mul 2).add hconst
  have hint : ∫ z, ‖trainedBatchOutput net X (predict net X uStar + z) η lam T -
      predict net X uStar‖ ^ 2 ∂μ ≤
      ∫ z, 2 * ‖ridgePrediction net X (predict net X uStar + z) lam hn hlam -
        predict net X uStar‖ ^ 2 + 2 * δ ^ 2 ∂μ :=
    integral_mono_ae hint_trained hupper hpoint
  rw [integral_add (hq_int.const_mul 2) hconst, integral_const_mul, integral_const] at hint
  have hprob : μ.real Set.univ = 1 := by simp
  have hsmash : μ.real Set.univ • (2 * δ ^ 2) = 2 * δ ^ 2 := by rw [hprob, one_smul]
  have hmse := MLPFixedDesign.ridge_expected_mse net X uStar lam hn hlam hnoise
  have hn0 : (0 : ℝ) < n := Nat.cast_pos.mpr hn
  rw [div_le_iff₀ hn0]
  have hintq : ∫ z, ‖ridgePrediction net X (predict net X uStar + z) lam hn hlam -
      predict net X uStar‖ ^ 2 ∂μ ≤
      n * ((lam / 4) * ‖uStar‖ ^ 2 +
        σ ^ 2 * RidgeRisk.degreesOfFreedom (design net X) (n : ℝ)⁻¹ lam
          (inv_pos.mpr (Nat.cast_pos.mpr hn)) hlam / n) := by
    rw [div_le_iff₀ hn0, mul_comm] at hmse
    exact hmse
  calc
    ∫ z, ‖trainedBatchOutput net X (predict net X uStar + z) η lam T -
        predict net X uStar‖ ^ 2 ∂μ
        ≤ 2 * (∫ z, ‖ridgePrediction net X (predict net X uStar + z) lam hn hlam -
            predict net X uStar‖ ^ 2 ∂μ) + μ.real Set.univ • (2 * δ ^ 2) := hint
    _ = 2 * (∫ z, ‖ridgePrediction net X (predict net X uStar + z) lam hn hlam -
          predict net X uStar‖ ^ 2 ∂μ) + 2 * δ ^ 2 := by rw [hsmash]
    _ ≤ 2 * (n * ((lam / 4) * ‖uStar‖ ^ 2 +
          σ ^ 2 * RidgeRisk.degreesOfFreedom (design net X) (n : ℝ)⁻¹ lam
            (inv_pos.mpr (Nat.cast_pos.mpr hn)) hlam / n)) + 2 * δ ^ 2 := by linarith
    _ = (2 * ((lam / 4) * ‖uStar‖ ^ 2 +
          σ ^ 2 * RidgeRisk.degreesOfFreedom (design net X) (n : ℝ)⁻¹ lam
            (inv_pos.mpr (Nat.cast_pos.mpr hn)) hlam / n) +
        2 * δ ^ 2 / n) * n := by field_simp [ne_of_gt hn0]

/-- Rank-bounded corollary of `contractive_batch_expected_mse`. -/
theorem contractive_batch_expected_mse_of_rank (net : TwoLayerNet d m) (X : Fin n → Fin d → ℝ)
    (uStar : Parameters (m * d)) (η lam : ℝ) {A L M E Bz : ℝ}
    (hn : 0 < n) (hlam : 0 < lam) (hη : 0 < η)
    (hA : 0 ≤ A) (hL : 0 ≤ L) (hM : 0 ≤ M) (hE : 0 ≤ E) (hBz : 0 ≤ Bz)
    (hq_nl : nonlinearRate m n η lam A L E < 1)
    (hstep : η * (Ridge.smoothness (design net X) (n : ℝ)⁻¹ lam : ℝ) ^ 2 < 2 * lam)
    (hW : ∀ i, |net.W₂ 0 i| ≤ A) (hσ : ∀ u, |deriv net.act.σ u| ≤ L)
    (hLip : ∀ a b, |deriv net.act.σ a - deriv net.act.σ b| ≤ M * |a - b|)
    (hX : ∀ r, inputEnergy (X r) ≤ E)
    {μ : Measure (Parameters n)} {σ : ℝ} [IsProbabilityMeasure μ]
    (hnoise : RidgeRisk.LabelNoise μ σ)
    (hz_ae : ∀ᵐ z ∂μ, ‖z‖ ≤ Bz)
    (T : ℕ)
    (hint_trained : Integrable (fun z =>
      ‖trainedBatchOutput net X (predict net X uStar + z) η lam T -
        predict net X uStar‖ ^ 2) μ)
    {r : ℕ} (hr : Module.finrank ℝ (LinearMap.range (design net X).toLinearMap) ≤ r) :
    let S := ‖design net X uStar‖ + Bz
    let q := Optim.GD.q η lam (Ridge.smoothness (design net X) (n : ℝ)⁻¹ lam)
    let δ := uniformBatchGap T m n η lam A L M E ‖design net X‖ q S
    (∫ z, ‖trainedBatchOutput net X (predict net X uStar + z) η lam T -
      predict net X uStar‖ ^ 2 ∂μ) / n ≤
      2 * ((lam / 4) * ‖uStar‖ ^ 2 + σ ^ 2 * (r : ℝ) / n) + 2 * δ ^ 2 / n := by
  intro S q δ
  have hmse := contractive_batch_expected_mse net X uStar η lam hn hlam hη
    hA hL hM hE hBz hq_nl hstep hW hσ hLip hX hnoise hz_ae T hint_trained
  have hc : 0 < (n : ℝ)⁻¹ := inv_pos.mpr (Nat.cast_pos.mpr hn)
  have hdf := RidgeRisk.degreesOfFreedom_le_rank (design net X) (n : ℝ)⁻¹ lam hc hlam
  have hvar : σ ^ 2 * RidgeRisk.degreesOfFreedom (design net X) (n : ℝ)⁻¹ lam hc hlam / n ≤
      σ ^ 2 * (r : ℝ) / n := by
    have hσ : 0 ≤ σ ^ 2 := sq_nonneg σ
    have hn0 : (0 : ℝ) ≤ n := Nat.cast_nonneg _
    have hr' : RidgeRisk.degreesOfFreedom (design net X) (n : ℝ)⁻¹ lam hc hlam ≤ (r : ℝ) :=
      hdf.trans (Nat.cast_le.mpr hr)
    exact div_le_div_of_nonneg_right (mul_le_mul_of_nonneg_left hr' hσ) hn0
  linarith

end OpenTheory.MLPContractive
