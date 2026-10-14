import OpenTheory.Dynamics.TangentStep

/-! Finite-horizon, initialization-centered first-layer training for a scalar
two-layer MLP on one example. The readout is frozen. Regularization penalizes
displacement from initialization, not the initial weights themselves.

The remainder theorem compares the nonlinear output with the initial Taylor
model at the SAME displacement. It does not identify that displacement with
independent linearized GD, and does not discharge the ridge prediction gap.
The finite-horizon bound retains its residual-bound hypothesis explicitly. -/
noncomputable section
open scoped BigOperators

namespace OpenTheory.MLPTrajectory
open FeatureStep
variable {d m : ℕ}

def atDisplacement (net : TwoLayerNet d m) (U : Matrix (Fin m) (Fin d) ℝ) :
    TwoLayerNet d m := { net with W₁ := net.W₁ + U }

def preShift (U : Matrix (Fin m) (Fin d) ℝ) (x : Fin d → ℝ) (i : Fin m) : ℝ :=
  ∑ j, U i j * x j

theorem preactivation_displacement (net : TwoLayerNet d m) (U) (x : Fin d → ℝ) (i) :
    preactivation (atDisplacement net U) x i = preactivation net x i + preShift U x i := by
  simp [preactivation, atDisplacement, preShift, add_mul, Finset.sum_add_distrib]

def linearizedForward (net : TwoLayerNet d m) (U : Matrix (Fin m) (Fin d) ℝ)
    (x : Fin d → ℝ) : ℝ :=
  net.forward x + ∑ i, net.W₂ 0 i * deriv net.act.σ (preactivation net x i) * preShift U x i

/-- Global Taylor remainder, not limited to a single training step. -/
theorem forward_remainder_le (net : TwoLayerNet d m) (U) (x : Fin d → ℝ)
    {M : ℝ} (hM : 0 ≤ M)
    (hLip : ∀ a b, |deriv net.act.σ a - deriv net.act.σ b| ≤ M * |a - b|) :
    |(atDisplacement net U).forward x - linearizedForward net U x| ≤
      M * ∑ i, |net.W₂ 0 i| * |preShift U x i| ^ 2 := by
  have heq : (atDisplacement net U).forward x - linearizedForward net U x =
      ∑ i, net.W₂ 0 i *
        (net.act.σ (preactivation net x i + preShift U x i) -
          net.act.σ (preactivation net x i) -
          deriv net.act.σ (preactivation net x i) * preShift U x i) := by
    unfold linearizedForward
    rw [TangentStep.forward_eq_sigma_pre, TangentStep.forward_eq_sigma_pre,
      sub_add_eq_sub_sub, ← Finset.sum_sub_distrib, ← Finset.sum_sub_distrib]
    apply Finset.sum_congr rfl
    intro i _
    rw [preactivation_displacement]
    change net.W₂ 0 i * net.act.σ _ - net.W₂ 0 i * _ - _ = _
    ring
  rw [heq, Finset.mul_sum]
  apply (Finset.abs_sum_le_sum_abs _ _).trans
  apply Finset.sum_le_sum
  intro i _
  rw [abs_mul]
  have h := TangentStep.abs_quadratic_remainder net.act.σ net.act.differentiable
    hM hLip (preactivation net x i + preShift U x i) (preactivation net x i)
  simp only [add_sub_cancel_left] at h
  nlinarith [mul_le_mul_of_nonneg_left h (abs_nonneg (net.W₂ 0 i))]

def regularizedStep (net : TwoLayerNet d m) (x : Fin d → ℝ) (y η lam : ℝ)
    (U : Matrix (Fin m) (Fin d) ℝ) : Matrix (Fin m) (Fin d) ℝ :=
  fun i j => (1 - η * lam) * U i j -
    η * outputResidual (atDisplacement net U) x y * net.W₂ 0 i *
      deriv net.act.σ (preactivation (atDisplacement net U) x i) * x j

/-- Exact agreement with the previously checked squared-loss GD update,
plus decay of displacement from initialization. -/
theorem regularized_step_eq (net : TwoLayerNet d m) (x : Fin d → ℝ)
    (y η lam : ℝ) (U) (i : Fin m) (j : Fin d) :
    net.W₁ i j + regularizedStep net x y η lam U i j =
      firstLayerStep (atDisplacement net U) x y η i j - η * lam * U i j := by
  simp only [regularizedStep, firstLayerStep, atDisplacement, Matrix.add_apply]
  ring

def trajectory (net : TwoLayerNet d m) (x : Fin d → ℝ) (y η lam : ℝ) :
    ℕ → Matrix (Fin m) (Fin d) ℝ :=
  fun t => (regularizedStep net x y η lam)^[t] 0

theorem preShift_step (net : TwoLayerNet d m) (x : Fin d → ℝ) (y η lam : ℝ) (U) (i) :
    preShift (regularizedStep net x y η lam U) x i =
      (1 - η * lam) * preShift U x i -
        η * outputResidual (atDisplacement net U) x y * net.W₂ 0 i *
          deriv net.act.σ (preactivation (atDisplacement net U) x i) * inputEnergy x := by
  unfold preShift regularizedStep inputEnergy
  rw [Finset.mul_sum, Finset.mul_sum, ← Finset.sum_sub_distrib]
  apply Finset.sum_congr rfl
  intro j _
  ring

/-- An explicit T-step drift bound under a residual envelope and nonexpansive
regularization decay. No bounded-noise or residual-stability claim is hidden. -/
theorem iterate_preShift_le (net : TwoLayerNet d m) (x : Fin d → ℝ) (y η lam : ℝ)
    {A L R : ℝ} (hη : 0 ≤ η) (hA : 0 ≤ A) (hL : 0 ≤ L) (hR : 0 ≤ R)
    (hdecay : |1 - η * lam| ≤ 1)
    (hW : ∀ i, |net.W₂ 0 i| ≤ A) (hσ : ∀ u, |deriv net.act.σ u| ≤ L)
    (T : ℕ)
    (hres : ∀ t < T, |outputResidual (atDisplacement net (trajectory net x y η lam t)) x y| ≤ R)
    (i : Fin m) :
    |preShift (trajectory net x y η lam T) x i| ≤
      T * (η * R * A * L * inputEnergy x) := by
  induction T with
  | zero => simp [trajectory, preShift]
  | succ T ih =>
    have hi := ih (fun t ht => hres t (Nat.lt_succ_of_lt ht))
    have hr := hres T (Nat.lt_succ_self T)
    have hc : |η * outputResidual (atDisplacement net (trajectory net x y η lam T)) x y *
        net.W₂ 0 i * deriv net.act.σ
          (preactivation (atDisplacement net (trajectory net x y η lam T)) x i) *
        inputEnergy x| ≤ η * R * A * L * inputEnergy x := by
      simp only [abs_mul, abs_of_nonneg hη, abs_of_nonneg (inputEnergy_nonneg x)]
      gcongr
      · exact inputEnergy_nonneg x
      · exact hW i
      · exact hσ _
    have ht : trajectory net x y η lam (T + 1) =
        regularizedStep net x y η lam (trajectory net x y η lam T) := by
      simp only [trajectory, Function.iterate_succ_apply']
    rw [ht, preShift_step]
    calc
      _ ≤ |(1 - η * lam) * preShift (trajectory net x y η lam T) x i| +
          |η * outputResidual (atDisplacement net (trajectory net x y η lam T)) x y *
            net.W₂ 0 i * deriv net.act.σ
              (preactivation (atDisplacement net (trajectory net x y η lam T)) x i) *
            inputEnergy x| := abs_sub _ _
      _ ≤ |preShift (trajectory net x y η lam T) x i| +
          η * R * A * L * inputEnergy x := by
        rw [abs_mul]
        exact add_le_add (by nlinarith [abs_nonneg (preShift (trajectory net x y η lam T) x i)]) hc
      _ ≤ _ := by push_cast; linarith

/-- Finite-horizon MLP output remainder at its own trained displacement.
Independent linearized training and the ridge endpoint are separate gaps. -/
theorem iterate_remainder_le (net : TwoLayerNet d m) (x : Fin d → ℝ) (y η lam : ℝ)
    {A L R M : ℝ} (hη : 0 ≤ η) (hA : 0 ≤ A) (hL : 0 ≤ L) (hR : 0 ≤ R) (hM : 0 ≤ M)
    (hdecay : |1 - η * lam| ≤ 1)
    (hW : ∀ i, |net.W₂ 0 i| ≤ A) (hσ : ∀ u, |deriv net.act.σ u| ≤ L)
    (hLip : ∀ a b, |deriv net.act.σ a - deriv net.act.σ b| ≤ M * |a - b|)
    (T : ℕ)
    (hres : ∀ t < T, |outputResidual (atDisplacement net (trajectory net x y η lam t)) x y| ≤ R) :
    |(atDisplacement net (trajectory net x y η lam T)).forward x -
      linearizedForward net (trajectory net x y η lam T) x| ≤
      M * m * A * (T * (η * R * A * L * inputEnergy x)) ^ 2 := by
  apply (forward_remainder_le net _ x hM hLip).trans
  have hs : ∑ i, |net.W₂ 0 i| * |preShift (trajectory net x y η lam T) x i| ^ 2 ≤
      ∑ _ : Fin m, A * (T * (η * R * A * L * inputEnergy x)) ^ 2 := by
    apply Finset.sum_le_sum
    intro i _
    have hi := iterate_preShift_le net x y η lam hη hA hL hR hdecay hW hσ T hres i
    exact mul_le_mul (hW i)
      (by simpa only [pow_two] using mul_self_le_mul_self (abs_nonneg _) hi)
      (sq_nonneg _) hA
  have hb := mul_le_mul_of_nonneg_left hs hM
  simpa [Finset.sum_const, Finset.card_univ, nsmul_eq_mul, mul_assoc] using hb

end OpenTheory.MLPTrajectory
