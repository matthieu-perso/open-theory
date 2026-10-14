import NN.MLTheory.Optimization.StronglyConvexGD
import Mathlib.Analysis.InnerProductSpace.Adjoint
import Mathlib.Analysis.InnerProductSpace.Calculus
import Mathlib.Analysis.Calculus.Gradient.Basic
import Mathlib.LinearAlgebra.FiniteDimensional.Basic
import Mathlib.Tactic

/-! Exact-real ridge regression in operator form. `c = 1/n` gives mean squared error. -/
noncomputable section
open scoped RealInnerProductSpace
open ContinuousLinearMap

namespace OpenTheory.Ridge

variable {E F : Type} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]
  [NormedAddCommGroup F] [InnerProductSpace ℝ F] [CompleteSpace F]

def objective (A : E →L[ℝ] F) (y : F) (c lam : ℝ) (x : E) : ℝ :=
  c / 2 * ‖A x - y‖ ^ 2 + lam / 2 * ‖x‖ ^ 2

def grad (A : E →L[ℝ] F) (y : F) (c lam : ℝ) (x : E) : E :=
  c • A.adjoint (A x - y) + lam • x

theorem hasGradient (A : E →L[ℝ] F) (y : F) (c lam : ℝ) (x : E) :
    HasGradientAt (objective A y c lam) (grad A y c lam x) x := by
  apply hasGradientAt_iff_hasFDerivAt.mpr
  have h := (((A.hasFDerivAt.sub_const y).norm_sq).const_mul (c / 2)).add
    (((hasFDerivAt_id x).norm_sq).const_mul (lam / 2))
  have he : InnerProductSpace.toDual ℝ E (grad A y c lam x) =
      (c / 2) • (2 • (innerSL ℝ (A x - y)).comp A) +
      (lam / 2) • (2 • (innerSL ℝ x).comp (ContinuousLinearMap.id ℝ E)) := by
    ext v
    simp [grad, adjoint_inner_left]
    ring
  rw [he]
  convert! h using 1

theorem gradient_eq (A : E →L[ℝ] F) (y : F) (c lam : ℝ) (x : E) :
    gradient (objective A y c lam) x = grad A y c lam x :=
  (hasGradient A y c lam x).gradient

def hessian (A : E →L[ℝ] F) (c lam : ℝ) : E →L[ℝ] E :=
  c • A.adjoint.comp A + lam • ContinuousLinearMap.id ℝ E

theorem grad_sub (A : E →L[ℝ] F) (y : F) (c lam : ℝ) (x z : E) :
    grad A y c lam x - grad A y c lam z = hessian A c lam (x - z) := by
  simp [grad, hessian, map_sub, smul_sub]
  abel

theorem inner_hessian (A : E →L[ℝ] F) (c lam : ℝ) (v : E) :
    ⟪v, hessian A c lam v⟫ = c * ‖A v‖ ^ 2 + lam * ‖v‖ ^ 2 := by
  simp [hessian, inner_add_right, inner_smul_right, adjoint_inner_right]

theorem stronglyMonotone (A : E →L[ℝ] F) (y : F) (c lam : ℝ) (hc : 0 ≤ c) :
    Optim.GD.StrongMonotone lam (grad A y c lam) := by
  intro x z
  rw [grad_sub, inner_hessian]
  nlinarith [mul_nonneg hc (sq_nonneg ‖A (x - z)‖)]

theorem grad_lipschitz (A : E →L[ℝ] F) (y : F) (c lam : ℝ) :
    LipschitzWith ‖hessian A c lam‖₊ (grad A y c lam) := by
  apply LipschitzWith.of_dist_le_mul
  intro x z
  simp only [dist_eq_norm, grad_sub]
  exact (hessian A c lam).le_opNorm (x - z)

theorem exists_stationary [FiniteDimensional ℝ E]
    (A : E →L[ℝ] F) (y : F) (c lam : ℝ) (hc : 0 ≤ c) (hlam : 0 < lam) :
    ∃ x, grad A y c lam x = 0 := by
  have hi : Function.Injective (hessian A c lam) := by
    apply (hessian A c lam).toLinearMap.ker_eq_bot.mp
    rw [LinearMap.ker_eq_bot']
    intro v hv
    have he := inner_hessian A c lam v
    change hessian A c lam v = 0 at hv
    rw [hv, inner_zero_right] at he
    have hn : ‖v‖ ^ 2 = 0 := by
      nlinarith [mul_nonneg hc (sq_nonneg ‖A v‖), sq_nonneg ‖v‖]
    exact norm_eq_zero.mp ((sq_eq_zero_iff).mp hn)
  obtain ⟨x, hx⟩ := (LinearMap.injective_iff_surjective.mp hi) (c • A.adjoint y)
  refine ⟨x, ?_⟩
  change hessian A c lam x = c • A.adjoint y at hx
  change c • A.adjoint (A x) + lam • x = c • A.adjoint y at hx
  simp only [grad, map_sub, smul_sub]
  rw [sub_add_eq_add_sub, hx, sub_self]

theorem quadratic_expansion (A : E →L[ℝ] F) (y : F) (c lam : ℝ) (x z : E) :
    objective A y c lam z = objective A y c lam x + ⟪grad A y c lam x, z - x⟫ +
      c / 2 * ‖A (z - x)‖ ^ 2 + lam / 2 * ‖z - x‖ ^ 2 := by
  simp [objective, grad, map_sub, norm_sub_sq_real, inner_add_left, inner_smul_left,
    adjoint_inner_left, inner_sub_left, inner_sub_right, real_inner_comm]
  ring

theorem stationary_minimizes (A : E →L[ℝ] F) (y : F) (c lam : ℝ)
    (hc : 0 ≤ c) (hlam : 0 ≤ lam) {x : E} (hx : grad A y c lam x = 0) :
    ∀ z, objective A y c lam x ≤ objective A y c lam z := by
  intro z
  rw [quadratic_expansion A y c lam x z, hx, inner_zero_left]
  have h₁ := mul_nonneg (div_nonneg hc (by norm_num : (0 : ℝ) ≤ 2))
    (sq_nonneg ‖A (z - x)‖)
  have h₂ := mul_nonneg (div_nonneg hlam (by norm_num : (0 : ℝ) ≤ 2))
    (sq_nonneg ‖z - x‖)
  linarith

/-- Strong convexity makes every minimizer the unique stationary point. -/
theorem minimizer_grad_eq_zero [FiniteDimensional ℝ E]
    (A : E →L[ℝ] F) (y : F) (c lam : ℝ) (hc : 0 ≤ c) (hlam : 0 < lam)
    {x : E} (hmin : ∀ z, objective A y c lam x ≤ objective A y c lam z) :
    grad A y c lam x = 0 := by
  obtain ⟨s, hs⟩ := exists_stationary A y c lam hc hlam
  have hsmin := stationary_minimizes A y c lam hc hlam.le hs
  have hexp := quadratic_expansion A y c lam s x
  rw [hs, inner_zero_left] at hexp
  have hrest : c / 2 * ‖A (x - s)‖ ^ 2 + lam / 2 * ‖x - s‖ ^ 2 = 0 := by
    linarith [hmin s, hsmin x]
  have hnorm : ‖x - s‖ ^ 2 = 0 := by
    nlinarith [mul_nonneg (div_nonneg hc (by norm_num : (0 : ℝ) ≤ 2))
      (sq_nonneg ‖A (x - s)‖), hlam]
  rw [sub_eq_zero.mp (norm_eq_zero.mp ((sq_eq_zero_iff).mp hnorm)), hs]

/-- Convert distance to a stationary ridge minimizer into an objective gap. -/
theorem objective_gap_le (A : E →L[ℝ] F) (y : F) (c lam : ℝ)
    (hc : 0 ≤ c) {x : E} (hx : grad A y c lam x = 0) (z : E) :
    objective A y c lam z - objective A y c lam x ≤
      (c / 2 * ‖A‖ ^ 2 + lam / 2) * ‖z - x‖ ^ 2 := by
  rw [quadratic_expansion A y c lam x z, hx, inner_zero_left]
  have hA := A.le_opNorm (z - x)
  have hAsq : ‖A (z - x)‖ ^ 2 ≤ ‖A‖ ^ 2 * ‖z - x‖ ^ 2 := by
    nlinarith [norm_nonneg (A (z - x)), norm_nonneg A, norm_nonneg (z - x)]
  have hc2 : 0 ≤ c / 2 := div_nonneg hc (by norm_num)
  nlinarith [mul_le_mul_of_nonneg_left hAsq hc2]

/-- A computable-from-the-operator upper bound, enlarged to cover the convexity constant. -/
def smoothness (A : E →L[ℝ] F) (c lam : ℝ) : NNReal :=
  ‖hessian A c lam‖₊ + ‖lam‖₊

/-- A strictly positive admissible step derived from the operator and regularizer. -/
theorem admissible_step (A : E →L[ℝ] F) (c lam : ℝ) (hlam : 0 < lam) :
    0 < lam / (smoothness A c lam : ℝ) ^ 2 ∧
      (lam / (smoothness A c lam : ℝ) ^ 2) * (smoothness A c lam : ℝ) ^ 2 < 2 * lam := by
  have hL : 0 < (smoothness A c lam : ℝ) := by
    simp only [smoothness, NNReal.coe_add, coe_nnnorm, Real.norm_eq_abs]
    linarith [norm_nonneg (hessian A c lam), le_abs_self lam]
  have hsq := sq_pos_of_pos hL
  refine ⟨div_pos hlam hsq, ?_⟩
  rw [div_mul_cancel₀ _ (ne_of_gt hsq)]
  linarith

/-- Existence, actual-gradient iteration, and a geometric error bound in one theorem.
No minimizer or stationarity premise is left to the caller. -/
theorem training [FiniteDimensional ℝ E]
    (A : E →L[ℝ] F) (y : F) (c lam : ℝ) (hc : 0 ≤ c) (hlam : 0 < lam)
    (η : ℝ) (hη : 0 < η) (hstep : η * (smoothness A c lam : ℝ) ^ 2 < 2 * lam) :
    ∃ xStar, (∀ z, objective A y c lam xStar ≤ objective A y c lam z) ∧
      ∀ x k, ‖(Optim.GD.step η (gradient (objective A y c lam)))^[k] x - xStar‖ ^ 2 ≤
        (Optim.GD.q η lam (smoothness A c lam)) ^ k * ‖x - xStar‖ ^ 2 ∧
        Optim.GD.q η lam (smoothness A c lam) < 1 := by
  obtain ⟨xStar, hx⟩ := exists_stationary A y c lam hc hlam
  refine ⟨xStar, stationary_minimizes A y c lam hc hlam.le hx, ?_⟩
  intro x k
  have hg : gradient (objective A y c lam) = grad A y c lam :=
    funext (gradient_eq A y c lam)
  rw [hg]
  apply Optim.GD.dist_sq_iterate_le_of_step_size η lam (grad A y c lam)
    (stronglyMonotone A y c lam hc) _ hx hlam.le _ hη hstep k
  · exact (grad_lipschitz A y c lam).weaken (by simp [smoothness])
  · simp only [smoothness, NNReal.coe_add, coe_nnnorm, Real.norm_eq_abs]
    linarith [norm_nonneg (hessian A c lam), le_abs_self lam]

/-- The actual GD iterates also have a geometric regularized-objective gap. -/
theorem training_objective_gap [FiniteDimensional ℝ E]
    (A : E →L[ℝ] F) (y : F) (c lam : ℝ) (hc : 0 ≤ c) (hlam : 0 < lam)
    (η : ℝ) (hη : 0 < η) (hstep : η * (smoothness A c lam : ℝ) ^ 2 < 2 * lam) :
    ∃ xStar, (∀ z, objective A y c lam xStar ≤ objective A y c lam z) ∧
      ∀ x k, objective A y c lam
          ((Optim.GD.step η (gradient (objective A y c lam)))^[k] x) -
          objective A y c lam xStar ≤
        (c / 2 * ‖A‖ ^ 2 + lam / 2) *
          (Optim.GD.q η lam (smoothness A c lam)) ^ k * ‖x - xStar‖ ^ 2 := by
  obtain ⟨xStar, hx⟩ := exists_stationary A y c lam hc hlam
  refine ⟨xStar, stationary_minimizes A y c lam hc hlam.le hx, ?_⟩
  intro x k
  have hg : gradient (objective A y c lam) = grad A y c lam :=
    funext (gradient_eq A y c lam)
  have hlip : LipschitzWith (smoothness A c lam) (grad A y c lam) :=
    (grad_lipschitz A y c lam).weaken (by simp [smoothness])
  have hμL : lam ≤ (smoothness A c lam : ℝ) := by
    simp only [smoothness, NNReal.coe_add, coe_nnnorm, Real.norm_eq_abs]
    linarith [norm_nonneg (hessian A c lam), le_abs_self lam]
  have hd := Optim.GD.dist_sq_iterate_le_of_step_size η lam (grad A y c lam)
    (stronglyMonotone A y c lam hc) hlip (xStar := xStar) (x := x) hx hlam.le
    hμL hη hstep k
  rw [hg]
  calc
    objective A y c lam ((Optim.GD.step η (grad A y c lam))^[k] x) -
        objective A y c lam xStar ≤
        (c / 2 * ‖A‖ ^ 2 + lam / 2) *
          ‖(Optim.GD.step η (grad A y c lam))^[k] x - xStar‖ ^ 2 :=
      objective_gap_le A y c lam hc hx _
    _ ≤ (c / 2 * ‖A‖ ^ 2 + lam / 2) *
        ((Optim.GD.q η lam (smoothness A c lam)) ^ k * ‖x - xStar‖ ^ 2) := by
      apply mul_le_mul_of_nonneg_left hd.1
      positivity
    _ = _ := by ring

omit [CompleteSpace E] [CompleteSpace F] in
/-- Comparing with zero bounds every minimizer, independently of the design
operator. The squared-radius premise avoids a gratuitous square root. -/
theorem minimizer_norm_le (A : E →L[ℝ] F) (y : F) (c lam R : ℝ)
    (hc : 0 ≤ c) (hlam : 0 < lam) (hR : 0 ≤ R)
    (hbound : c * ‖y‖ ^ 2 ≤ lam * R ^ 2) {xStar : E}
    (hmin : ∀ z, objective A y c lam xStar ≤ objective A y c lam z) :
    ‖xStar‖ ≤ R := by
  have hz := hmin 0
  simp only [objective, map_zero, zero_sub, norm_neg, norm_zero, ne_eq,
    OfNat.ofNat_ne_zero, not_false_eq_true, zero_pow, mul_zero, add_zero] at hz
  have hs : lam * ‖xStar‖ ^ 2 ≤ lam * R ^ 2 := by
    nlinarith [mul_nonneg hc (sq_nonneg ‖A xStar - y‖)]
  have hs' : ‖xStar‖ ^ 2 ≤ R ^ 2 := by nlinarith
  nlinarith [norm_nonneg xStar]

/-- All exact-GD iterates started at zero lie in the fixed ball of radius
`2 R`, provided the response energy is at most `lam * R²`. -/
theorem training_radius [FiniteDimensional ℝ E]
    (A : E →L[ℝ] F) (y : F) (c lam R : ℝ)
    (hc : 0 ≤ c) (hlam : 0 < lam) (hR : 0 ≤ R)
    (hbound : c * ‖y‖ ^ 2 ≤ lam * R ^ 2)
    (η : ℝ) (hη : 0 < η)
    (hstep : η * (smoothness A c lam : ℝ) ^ 2 < 2 * lam) :
    ∃ xStar, (∀ z, objective A y c lam xStar ≤ objective A y c lam z) ∧
      ‖xStar‖ ≤ R ∧ ∀ k,
        ‖(Optim.GD.step η (gradient (objective A y c lam)))^[k] 0‖ ≤ 2 * R := by
  obtain ⟨xStar, hmin, hdist⟩ := training A y c lam hc hlam η hη hstep
  have hr := minimizer_norm_le A y c lam R hc hlam hR hbound hmin
  refine ⟨xStar, hmin, hr, fun k => ?_⟩
  have hμL : lam ≤ (smoothness A c lam : ℝ) := by
    simp only [smoothness, NNReal.coe_add, coe_nnnorm, Real.norm_eq_abs]
    linarith [norm_nonneg (hessian A c lam), le_abs_self lam]
  have hq := Optim.GD.q_nonneg_of_le η lam (smoothness A c lam) hlam.le hμL
  have hp := pow_le_one₀ (n := k) hq (hdist 0 k).2.le
  have hd := (hdist 0 k).1
  simp only [zero_sub, norm_neg] at hd
  have hs := mul_le_mul_of_nonneg_right hp (sq_nonneg ‖xStar‖)
  have hd' : ‖(Optim.GD.step η (gradient (objective A y c lam)))^[k] 0 - xStar‖ ≤
      ‖xStar‖ := by nlinarith [norm_nonneg xStar]
  have ht := norm_add_le
    ((Optim.GD.step η (gradient (objective A y c lam)))^[k] 0 - xStar) xStar
  rw [sub_add_cancel] at ht
  linarith

end OpenTheory.Ridge
