import OpenTheory.Dynamics.MLPEndpoint
import Mathlib.Analysis.Calculus.MeanValue

/-! Finite-horizon residual control derived from initial error, not assumed.
The geometric envelope is deliberately conservative; it is not a stability
or infinite-time convergence claim. -/
noncomputable section
open scoped BigOperators
namespace OpenTheory.MLPResidual
open FeatureStep MLPTrajectory
variable {d m : ℕ}

theorem forward_change_le (net : TwoLayerNet d m) (x : Fin d → ℝ) (U)
    {A L D : ℝ} (hA : 0 ≤ A) (hL : 0 ≤ L)
    (hW : ∀ i, |net.W₂ 0 i| ≤ A) (hσ : ∀ u, |deriv net.act.σ u| ≤ L)
    (hD : ∀ i, |preShift U x i| ≤ D) :
    |(atDisplacement net U).forward x - net.forward x| ≤ m * A * L * D := by
  have hact (a b : ℝ) : |net.act.σ a - net.act.σ b| ≤ L * |a - b| := by
    simpa only [Real.norm_eq_abs] using
      Convex.norm_image_sub_le_of_norm_deriv_le
        (fun u (_ : u ∈ Set.univ) => net.act.differentiable u)
        (fun u _ => show ‖deriv net.act.σ u‖ ≤ L from hσ u)
        (convex_univ : Convex ℝ (Set.univ : Set ℝ)) (Set.mem_univ b) (Set.mem_univ a)
  rw [TangentStep.forward_eq_sigma_pre, TangentStep.forward_eq_sigma_pre,
    ← Finset.sum_sub_distrib]
  apply (Finset.abs_sum_le_sum_abs _ _).trans
  calc
    _ ≤ ∑ _ : Fin m, A * (L * D) := by
      apply Finset.sum_le_sum
      intro i _
      change |net.W₂ 0 i * net.act.σ _ - net.W₂ 0 i * net.act.σ _| ≤ _
      rw [← mul_sub, abs_mul]
      apply mul_le_mul (hW i) _ (abs_nonneg _) hA
      apply (hact _ _).trans
      rw [preactivation_displacement, add_sub_cancel_left]
      exact mul_le_mul_of_nonneg_left (hD i) hL
    _ = _ := by simp [Finset.sum_const, Finset.card_univ, nsmul_eq_mul]; ring

def growth (m : ℕ) (η A L E : ℝ) : ℝ := 1 + (η * A * L * E) * (1 + m * A * L)
def envelope (m : ℕ) (η A L E S : ℝ) (T : ℕ) : ℝ :=
  (1 + m * A * L) * S * growth m η A L E ^ T

theorem growth_one_le (m : ℕ) {η A L E : ℝ}
    (hη : 0 ≤ η) (hA : 0 ≤ A) (hL : 0 ≤ L) (hE : 0 ≤ E) :
    1 ≤ growth m η A L E := by
  unfold growth
  exact le_add_of_nonneg_right (by positivity)

theorem shift_le (net : TwoLayerNet d m) (x : Fin d → ℝ) (y η lam : ℝ)
    {A L S : ℝ} (hη : 0 ≤ η) (hA : 0 ≤ A) (hL : 0 ≤ L) (hS : 0 ≤ S)
    (hdecay : |1 - η * lam| ≤ 1)
    (hW : ∀ i, |net.W₂ 0 i| ≤ A) (hσ : ∀ u, |deriv net.act.σ u| ≤ L)
    (hinit : |net.forward x - y| ≤ S) (T : ℕ) (i : Fin m) :
    |preShift (trajectory net x y η lam T) x i| ≤
      S * growth m η A L (inputEnergy x) ^ T := by
  have hE := inputEnergy_nonneg x
  have hq := growth_one_le m hη hA hL hE
  induction T generalizing i with
  | zero => simpa [trajectory, preShift] using hS
  | succ t ih =>
    have hpow : 1 ≤ growth m η A L (inputEnergy x) ^ t := one_le_pow₀ hq
    have hs : S ≤ S * growth m η A L (inputEnergy x) ^ t := by nlinarith
    have hr : |outputResidual (atDisplacement net (trajectory net x y η lam t)) x y| ≤
        envelope m η A L (inputEnergy x) S t := by
      have hf := forward_change_le net x (trajectory net x y η lam t) hA hL hW hσ ih
      have ht := abs_sub_le ((atDisplacement net (trajectory net x y η lam t)).forward x)
        (net.forward x) y
      unfold outputResidual envelope
      nlinarith
    have hu : trajectory net x y η lam (t + 1) =
        regularizedStep net x y η lam (trajectory net x y η lam t) := by
      simp only [trajectory, Function.iterate_succ_apply']
    rw [hu, preShift_step]
    apply (abs_sub _ _).trans
    rw [abs_mul]
    have hd : |1 - η * lam| * |preShift (trajectory net x y η lam t) x i| ≤
        S * growth m η A L (inputEnergy x) ^ t :=
      (mul_le_mul_of_nonneg_right hdecay (abs_nonneg _)).trans (by simpa using ih i)
    have hb : |η * outputResidual (atDisplacement net (trajectory net x y η lam t)) x y *
        net.W₂ 0 i * deriv net.act.σ (preactivation (atDisplacement net
          (trajectory net x y η lam t)) x i) * inputEnergy x| ≤
        η * envelope m η A L (inputEnergy x) S t * A * L * inputEnergy x := by
      have hR : 0 ≤ envelope m η A L (inputEnergy x) S t := by
        unfold envelope growth
        positivity
      simp only [abs_mul, abs_of_nonneg hη, abs_of_nonneg hE]
      gcongr
      · exact hW i
      · exact hσ _
    apply (add_le_add hd hb).trans
    simp only [envelope, pow_succ, growth]
    ring_nf
    exact le_rfl

/-- No trajectory-dependent residual hypothesis: only an initial error bound. -/
theorem residual_le (net : TwoLayerNet d m) (x : Fin d → ℝ) (y η lam : ℝ)
    {A L S : ℝ} (hη : 0 ≤ η) (hA : 0 ≤ A) (hL : 0 ≤ L) (hS : 0 ≤ S)
    (hdecay : |1 - η * lam| ≤ 1)
    (hW : ∀ i, |net.W₂ 0 i| ≤ A) (hσ : ∀ u, |deriv net.act.σ u| ≤ L)
    (hinit : |net.forward x - y| ≤ S) (T t : ℕ) (ht : t ≤ T) :
    |outputResidual (atDisplacement net (trajectory net x y η lam t)) x y| ≤
      envelope m η A L (inputEnergy x) S T := by
  have hq := growth_one_le m hη hA hL (inputEnergy_nonneg x)
  have hp : 1 ≤ growth m η A L (inputEnergy x) ^ t := one_le_pow₀ hq
  have hs : S ≤ S * growth m η A L (inputEnergy x) ^ t := by nlinarith
  have hf := forward_change_le net x _ hA hL hW hσ
    (shift_le net x y η lam hη hA hL hS hdecay hW hσ hinit t)
  have htri := abs_sub_le ((atDisplacement net (trajectory net x y η lam t)).forward x)
    (net.forward x) y
  have hr : |outputResidual (atDisplacement net (trajectory net x y η lam t)) x y| ≤
      envelope m η A L (inputEnergy x) S t := by
    unfold outputResidual envelope
    nlinarith
  apply hr.trans
  unfold envelope
  exact mul_le_mul_of_nonneg_left (pow_le_pow_right₀ hq ht) (by positivity)

end OpenTheory.MLPResidual
