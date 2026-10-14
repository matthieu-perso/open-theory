import OpenTheory.Dynamics.MLPTrajectory

/-! Comparison with independently trained initial-Jacobian predictions.
One fixed example, frozen readout, and initialization-centered regularization.
Residual and drift envelopes are explicit; no statistical risk claim is made. -/
noncomputable section
open scoped BigOperators
namespace OpenTheory.MLPComparison
open FeatureStep MLPTrajectory
variable {d m : ℕ}

def kernel (net : TwoLayerNet d m) (x : Fin d → ℝ) : ℝ :=
  ∑ i, net.W₂ 0 i ^ 2 * deriv net.act.σ (preactivation net x i) ^ 2 * inputEnergy x

def crossKernel (net : TwoLayerNet d m) (x : Fin d → ℝ)
    (U : Matrix (Fin m) (Fin d) ℝ) : ℝ :=
  ∑ i, net.W₂ 0 i ^ 2 * deriv net.act.σ (preactivation net x i) *
    deriv net.act.σ (preactivation (atDisplacement net U) x i) * inputEnergy x

def predictionStep (net : TwoLayerNet d m) (x : Fin d → ℝ) (y η lam z : ℝ) : ℝ :=
  z - η * ((z - y) * kernel net x + lam * (z - net.forward x))

def linearPrediction (net : TwoLayerNet d m) (x : Fin d → ℝ) (y η lam : ℝ)
    (T : ℕ) : ℝ := (predictionStep net x y η lam)^[T] (net.forward x)

/-- Independent GD on the affine initial Taylor model, in displacement coordinates. -/
def linearizedStep (net : TwoLayerNet d m) (x : Fin d → ℝ) (y η lam : ℝ)
    (U : Matrix (Fin m) (Fin d) ℝ) : Matrix (Fin m) (Fin d) ℝ :=
  fun i j => (1 - η * lam) * U i j - η * (linearizedForward net U x - y) *
    net.W₂ 0 i * deriv net.act.σ (preactivation net x i) * x j

private theorem linearized_update (net : TwoLayerNet d m) (x : Fin d → ℝ)
    (U : Matrix (Fin m) (Fin d) ℝ) (r η lam : ℝ) (g : Fin m → ℝ) :
    linearizedForward net
      (fun i j => (1 - η * lam) * U i j - η * r * net.W₂ 0 i * g i * x j) x =
      (1 - η * lam) * linearizedForward net U x + η * lam * net.forward x -
        η * r * ∑ i, net.W₂ 0 i ^ 2 * deriv net.act.σ (preactivation net x i) *
          g i * inputEnergy x := by
  have hp : ∀ i, preShift
      (fun i j => (1 - η * lam) * U i j - η * r * net.W₂ 0 i * g i * x j) x i =
      (1 - η * lam) * preShift U x i - η * r * net.W₂ 0 i * g i * inputEnergy x := by
    intro i
    unfold preShift inputEnergy
    rw [Finset.mul_sum, Finset.mul_sum, ← Finset.sum_sub_distrib]
    apply Finset.sum_congr rfl
    intro j _
    ring
  have hs : ∀ i, net.W₂ 0 i * deriv net.act.σ (preactivation net x i) *
      ((1 - η * lam) * preShift U x i - η * r * net.W₂ 0 i * g i * inputEnergy x) =
      (1 - η * lam) * (net.W₂ 0 i * deriv net.act.σ (preactivation net x i) * preShift U x i) -
      η * r * (net.W₂ 0 i ^ 2 * deriv net.act.σ (preactivation net x i) * g i * inputEnergy x) :=
    fun i => by ring
  simp only [linearizedForward, hp, hs, Finset.sum_sub_distrib, ← Finset.mul_sum]
  ring

theorem linearized_step_prediction (net : TwoLayerNet d m) (x : Fin d → ℝ)
    (y η lam : ℝ) (U) :
    linearizedForward net (linearizedStep net x y η lam U) x =
      predictionStep net x y η lam (linearizedForward net U x) := by
  unfold linearizedStep
  rw [linearized_update]
  simp only [predictionStep, kernel, pow_two, mul_assoc]
  ring

/-- The scalar recurrence is exactly the prediction of a separately trained
linearized parameter trajectory, not the Taylor model at nonlinear weights. -/
theorem linearized_trajectory_prediction (net : TwoLayerNet d m) (x : Fin d → ℝ)
    (y η lam : ℝ) (T : ℕ) :
    linearizedForward net ((linearizedStep net x y η lam)^[T] 0) x =
      linearPrediction net x y η lam T := by
  induction T with
  | zero => simp [linearizedForward, preShift, linearPrediction]
  | succ T ih =>
    simp only [Function.iterate_succ_apply', linearized_step_prediction, ih,
      linearPrediction]

/-- Exact local error driving divergence of the two trajectories. -/
theorem prediction_defect (net : TwoLayerNet d m) (x : Fin d → ℝ) (y η lam : ℝ) (U) :
    linearizedForward net (regularizedStep net x y η lam U) x -
      predictionStep net x y η lam (linearizedForward net U x) =
      -η * (((atDisplacement net U).forward x - linearizedForward net U x) * kernel net x +
        outputResidual (atDisplacement net U) x y * (crossKernel net x U - kernel net x)) := by
  unfold regularizedStep
  rw [linearized_update]
  simp only [predictionStep, crossKernel, outputResidual]
  ring

theorem predictionStep_nonexpansive (net : TwoLayerNet d m) (x : Fin d → ℝ)
    (y η lam : ℝ) (hstep : |1 - η * (kernel net x + lam)| ≤ 1) (a b : ℝ) :
    |predictionStep net x y η lam a - predictionStep net x y η lam b| ≤ |a - b| := by
  have he : predictionStep net x y η lam a - predictionStep net x y η lam b =
      (1 - η * (kernel net x + lam)) * (a - b) := by unfold predictionStep; ring
  rw [he, abs_mul]
  nlinarith [abs_nonneg (a - b)]

/-- Accumulation of bounded local defects in a nonexpansive recurrence. -/
theorem accumulated_defect (F : ℝ → ℝ) (u : ℕ → ℝ) (C : ℝ)
    (hF : ∀ a b, |F a - F b| ≤ |a - b|) (T : ℕ)
    (hdef : ∀ t < T, |u (t + 1) - F (u t)| ≤ C) :
    |u T - F^[T] (u 0)| ≤ T * C := by
  induction T with
  | zero => simp
  | succ T ih =>
    have hi := ih (fun t ht => hdef t (Nat.lt_succ_of_lt ht))
    have hd := hdef T (Nat.lt_succ_self T)
    rw [Function.iterate_succ_apply']
    have ht := abs_sub_le (u (T + 1)) (F (u T)) (F (F^[T] (u 0)))
    have hl := hF (u T) (F^[T] (u 0))
    push_cast
    linarith

/-- The actual nonlinear trajectory versus independently trained linearized GD.
B bounds Taylor error, G bounds cross-kernel drift, R bounds residuals.
All envelopes are required on the stated finite horizon, not assumed globally. -/
theorem nonlinear_linearized_gap (net : TwoLayerNet d m) (x : Fin d → ℝ)
    (y η lam : ℝ) (hη : 0 ≤ η)
    (hstep : |1 - η * (kernel net x + lam)| ≤ 1)
    {B G R : ℝ} (_hB : 0 ≤ B) (_hG : 0 ≤ G) (hR : 0 ≤ R) (T : ℕ)
    (hrem : ∀ t ≤ T, |(atDisplacement net (trajectory net x y η lam t)).forward x -
      linearizedForward net (trajectory net x y η lam t) x| ≤ B)
    (hcross : ∀ t < T, |crossKernel net x (trajectory net x y η lam t) - kernel net x| ≤ G)
    (hres : ∀ t < T, |outputResidual (atDisplacement net (trajectory net x y η lam t)) x y| ≤ R) :
    |(atDisplacement net (trajectory net x y η lam T)).forward x -
      linearPrediction net x y η lam T| ≤ B + T * η * (B * |kernel net x| + R * G) := by
  let u := fun t => linearizedForward net (trajectory net x y η lam t) x
  have hu0 : u 0 = net.forward x := by simp [u, trajectory, linearizedForward, preShift]
  have hd : ∀ t < T, |u (t + 1) - predictionStep net x y η lam (u t)| ≤
      η * (B * |kernel net x| + R * G) := by
    intro t ht
    change |linearizedForward net (trajectory net x y η lam (t + 1)) x - _| ≤ _
    have ht' : trajectory net x y η lam (t + 1) =
        regularizedStep net x y η lam (trajectory net x y η lam t) := by
      simp only [trajectory, Function.iterate_succ_apply']
    rw [ht', prediction_defect, abs_mul, abs_neg, abs_of_nonneg hη]
    apply mul_le_mul_of_nonneg_left _ hη
    apply (abs_add_le _ _).trans
    rw [abs_mul, abs_mul]
    exact add_le_add
      (mul_le_mul_of_nonneg_right (hrem t (Nat.le_of_lt ht)) (abs_nonneg _))
      (mul_le_mul (hres t ht) (hcross t ht) (abs_nonneg _) hR)
  have ha := accumulated_defect (predictionStep net x y η lam) u
    (η * (B * |kernel net x| + R * G)) (predictionStep_nonexpansive net x y η lam hstep) T hd
  rw [hu0] at ha
  have ht := abs_sub_le ((atDisplacement net (trajectory net x y η lam T)).forward x)
    (u T) (linearPrediction net x y η lam T)
  have hr := hrem T le_rfl
  change |u T - linearPrediction net x y η lam T| ≤ _ at ha
  dsimp [u] at ht ha
  nlinarith

/-- Initial/current Jacobian correlation changes at most linearly in hidden
preactivation displacement. This is the missing derivative-drift term. -/
theorem crossKernel_drift_le (net : TwoLayerNet d m) (x : Fin d → ℝ) (U)
    {A L M D : ℝ} (_hA : 0 ≤ A) (hL : 0 ≤ L) (hM : 0 ≤ M)
    (hW : ∀ i, |net.W₂ 0 i| ≤ A)
    (hσ : ∀ u, |deriv net.act.σ u| ≤ L)
    (hLip : ∀ a b, |deriv net.act.σ a - deriv net.act.σ b| ≤ M * |a - b|)
    (hD : ∀ i, |preShift U x i| ≤ D) :
    |crossKernel net x U - kernel net x| ≤ m * A ^ 2 * L * M * D * inputEnergy x := by
  have he : crossKernel net x U - kernel net x =
      ∑ i, net.W₂ 0 i ^ 2 * deriv net.act.σ (preactivation net x i) *
        (deriv net.act.σ (preactivation (atDisplacement net U) x i) -
          deriv net.act.σ (preactivation net x i)) * inputEnergy x := by
    unfold crossKernel kernel
    rw [← Finset.sum_sub_distrib]
    apply Finset.sum_congr rfl
    intro i _
    ring
  rw [he]
  apply (Finset.abs_sum_le_sum_abs _ _).trans
  calc
    _ ≤ ∑ _ : Fin m, A ^ 2 * L * (M * D) * inputEnergy x := by
      apply Finset.sum_le_sum
      intro i _
      have hd : |deriv net.act.σ (preactivation (atDisplacement net U) x i) -
          deriv net.act.σ (preactivation net x i)| ≤ M * D := by
        apply (hLip _ _).trans
        rw [preactivation_displacement, add_sub_cancel_left]
        exact mul_le_mul_of_nonneg_left (hD i) hM
      have hw : net.W₂ 0 i ^ 2 ≤ A ^ 2 := by
        simpa only [← pow_two, sq_abs] using mul_self_le_mul_self (abs_nonneg _) (hW i)
      rw [abs_mul, abs_mul, abs_mul, abs_of_nonneg (sq_nonneg _),
        abs_of_nonneg (inputEnergy_nonneg x)]
      apply mul_le_mul_of_nonneg_right _ (inputEnergy_nonneg x)
      exact mul_le_mul
        (mul_le_mul hw (hσ _) (abs_nonneg _) (sq_nonneg _)) hd (abs_nonneg _)
        (mul_nonneg (sq_nonneg _) hL)
    _ = _ := by simp [Finset.sum_const, Finset.card_univ, nsmul_eq_mul]; ring

/-- Fully explicit finite-horizon comparison, conditional only on stated
residual/derivative/readout envelopes and the two step conditions.
Unlike `iterate_remainder_le`, the comparator is independently trained GD. -/
theorem explicit_nonlinear_linearized_gap (net : TwoLayerNet d m) (x : Fin d → ℝ)
    (y η lam : ℝ) {A L R M : ℝ}
    (hη : 0 ≤ η) (hA : 0 ≤ A) (hL : 0 ≤ L) (hR : 0 ≤ R) (hM : 0 ≤ M)
    (hdecay : |1 - η * lam| ≤ 1)
    (hstep : |1 - η * (kernel net x + lam)| ≤ 1)
    (hW : ∀ i, |net.W₂ 0 i| ≤ A) (hσ : ∀ u, |deriv net.act.σ u| ≤ L)
    (hLip : ∀ a b, |deriv net.act.σ a - deriv net.act.σ b| ≤ M * |a - b|)
    (T : ℕ)
    (hres : ∀ t < T, |outputResidual (atDisplacement net (trajectory net x y η lam t)) x y| ≤ R) :
    let D : ℝ := T * (η * R * A * L * inputEnergy x)
    let B : ℝ := M * m * A * D ^ 2
    let G : ℝ := m * A ^ 2 * L * M * D * inputEnergy x
    |(atDisplacement net (trajectory net x y η lam T)).forward x -
      linearPrediction net x y η lam T| ≤ B + T * η * (B * |kernel net x| + R * G) := by
  dsimp only
  have hx := inputEnergy_nonneg x
  apply nonlinear_linearized_gap net x y η lam hη hstep
    (by positivity) (by positivity) hR T _ _ hres
  · intro t ht
    have hb := iterate_remainder_le net x y η lam hη hA hL hR hM hdecay hW hσ hLip t
      (fun s hs => hres s (lt_of_lt_of_le hs ht))
    apply hb.trans
    gcongr
  · intro t ht
    apply crossKernel_drift_le net x _ hA hL hM hW hσ hLip
    intro i
    apply (iterate_preShift_le net x y η lam hη hA hL hR hdecay hW hσ t
      (fun s hs => hres s (Nat.lt_trans hs ht)) i).trans
    gcongr

end OpenTheory.MLPComparison
