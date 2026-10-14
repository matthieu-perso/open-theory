import OpenTheory.Optimization.Ridge
import OpenTheory.Data.Foundation
import Mathlib.Analysis.InnerProductSpace.Spectrum
import Mathlib.Analysis.InnerProductSpace.Projection.FiniteDimensional
import Mathlib.LinearAlgebra.Eigenspace.Basic
import Mathlib.LinearAlgebra.FiniteDimensional.Lemmas
import Mathlib.MeasureTheory.Constructions.BorelSpace.Basic
import Mathlib.MeasureTheory.Integral.Bochner.ContinuousLinearMap
import Mathlib.Topology.Algebra.Module.FiniteDimension

/-! Fixed-design risk of the ridge solution computed by gradient descent.

The estimator is the unique minimizer of `Ridge.objective`. Exact gradient
descent converges to that minimizer; see `gd_approaches_ridgeSolution`.

Labels are `A θ* + ε` with the design `A` held fixed. White noise of scale `σ`
then gives an in-sample mean squared error bounded by the regularization bias
`(λ/4) ‖θ*‖²` plus `σ²` times the ridge degrees of freedom, divided by the
number of design rows. The degrees of freedom are at most the rank of `A`,
which is at most `min(d, m)`. Ambient parameter dimension enters only through
that rank.

This is the classical fixed-design bias–variance shape for ridge regression.
It is not a random-design generalization theorem and not a neural-network
theorem. It is the linear prototype of a kernel or NTK guarantee: the price of
fitting noise tracks the effective dimension of the design, not a uniform cover
of the parameter ball.
-/

noncomputable section

open MeasureTheory OpenTheory.Ridge
open scoped RealInnerProductSpace
open ContinuousLinearMap

namespace OpenTheory.RidgeRisk

variable {d m : ℕ}

instance : MeasurableSpace (Parameters m) := borel _
instance : BorelSpace (Parameters m) := ⟨rfl⟩

theorem hessian_bijective (A : Parameters d →L[ℝ] Parameters m) (c lam : ℝ)
    (hc : 0 ≤ c) (hlam : 0 < lam) : Function.Bijective (hessian A c lam) := by
  have hi : Function.Injective (hessian A c lam) := by
    apply (hessian A c lam).toLinearMap.ker_eq_bot.mp
    rw [LinearMap.ker_eq_bot']
    intro v hv
    have he := inner_hessian A c lam v
    change hessian A c lam v = 0 at hv
    rw [hv, inner_zero_right] at he
    have hn : ‖v‖ ^ 2 = 0 := by
      nlinarith [mul_nonneg hc (sq_nonneg ‖A v‖), sq_nonneg ‖v‖]
    exact norm_eq_zero.mp (sq_eq_zero_iff.mp hn)
  exact ⟨hi, LinearMap.injective_iff_surjective.mp hi⟩

/-- Inverse of `c A* A + λ I`. Finite dimension supplies continuity. -/
def hessianEquiv (A : Parameters d →L[ℝ] Parameters m) (c lam : ℝ)
    (hc : 0 ≤ c) (hlam : 0 < lam) : Parameters d ≃ₗ[ℝ] Parameters d :=
  let f := (hessian A c lam).toLinearMap
  { f, Equiv.ofBijective f (hessian_bijective A c lam hc hlam) with }

def hessianInv (A : Parameters d →L[ℝ] Parameters m) (c lam : ℝ)
    (hc : 0 ≤ c) (hlam : 0 < lam) : Parameters d →L[ℝ] Parameters d :=
  LinearMap.toContinuousLinearMap (hessianEquiv A c lam hc hlam).symm.toLinearMap

theorem hessianInv_apply (A : Parameters d →L[ℝ] Parameters m) (c lam : ℝ)
    (hc : 0 ≤ c) (hlam : 0 < lam) (x : Parameters d) :
    hessianInv A c lam hc hlam x = (hessianEquiv A c lam hc hlam).symm x :=
  rfl

theorem hessian_apply_hessianInv (A : Parameters d →L[ℝ] Parameters m) (c lam : ℝ)
    (hc : 0 ≤ c) (hlam : 0 < lam) (x : Parameters d) :
    hessian A c lam (hessianInv A c lam hc hlam x) = x := by
  rw [hessianInv_apply]
  exact (hessianEquiv A c lam hc hlam).apply_symm_apply x

theorem hessianInv_apply_hessian (A : Parameters d →L[ℝ] Parameters m) (c lam : ℝ)
    (hc : 0 ≤ c) (hlam : 0 < lam) (x : Parameters d) :
    hessianInv A c lam hc hlam (hessian A c lam x) = x := by
  rw [hessianInv_apply]
  exact (hessianEquiv A c lam hc hlam).symm_apply_apply x

theorem hessian_isSymmetric (A : Parameters d →L[ℝ] Parameters m) (c lam : ℝ) :
    (hessian A c lam).toLinearMap.IsSymmetric := by
  intro x y
  simp only [hessian, ContinuousLinearMap.toLinearMap_add, ContinuousLinearMap.toLinearMap_smul,
    ContinuousLinearMap.toLinearMap_comp, ContinuousLinearMap.coe_id, LinearMap.add_apply,
    LinearMap.smul_apply, LinearMap.comp_apply, LinearMap.id_apply, inner_add_left,
    inner_add_right, real_inner_smul_left, real_inner_smul_right, ContinuousLinearMap.coe_coe,
    adjoint_inner_left, adjoint_inner_right]

theorem hessianInv_symmetric (A : Parameters d →L[ℝ] Parameters m) (c lam : ℝ)
    (hc : 0 ≤ c) (hlam : 0 < lam) (x y : Parameters d) :
    ⟪hessianInv A c lam hc hlam x, y⟫ = ⟪x, hessianInv A c lam hc hlam y⟫ := by
  calc
    ⟪hessianInv A c lam hc hlam x, y⟫
        = ⟪hessianInv A c lam hc hlam x,
            hessian A c lam (hessianInv A c lam hc hlam y)⟫ := by
          rw [hessian_apply_hessianInv]
    _ = ⟪hessian A c lam (hessianInv A c lam hc hlam x),
          hessianInv A c lam hc hlam y⟫ :=
          ((hessian_isSymmetric A c lam)
            (hessianInv A c lam hc hlam x) (hessianInv A c lam hc hlam y)).symm
    _ = ⟪x, hessianInv A c lam hc hlam y⟫ := by rw [hessian_apply_hessianInv]

/-- The unique stationary point of the ridge objective, hence its minimizer. -/
def ridgeSolution (A : Parameters d →L[ℝ] Parameters m) (y : Parameters m) (c lam : ℝ)
    (hc : 0 < c) (hlam : 0 < lam) : Parameters d :=
  hessianInv A c lam hc.le hlam (c • A.adjoint y)

theorem hessian_ridgeSolution (A : Parameters d →L[ℝ] Parameters m) (y : Parameters m)
    (c lam : ℝ) (hc : 0 < c) (hlam : 0 < lam) :
    hessian A c lam (ridgeSolution A y c lam hc hlam) = c • A.adjoint y :=
  hessian_apply_hessianInv A c lam hc.le hlam _

theorem grad_eq_hessian_sub (A : Parameters d →L[ℝ] Parameters m) (y : Parameters m)
    (c lam : ℝ) (x : Parameters d) :
    grad A y c lam x = hessian A c lam x - c • A.adjoint y := by
  simp [grad, hessian, map_sub, smul_sub]
  abel

theorem grad_ridgeSolution (A : Parameters d →L[ℝ] Parameters m) (y : Parameters m)
    (c lam : ℝ) (hc : 0 < c) (hlam : 0 < lam) :
    grad A y c lam (ridgeSolution A y c lam hc hlam) = 0 := by
  rw [grad_eq_hessian_sub, hessian_ridgeSolution, sub_self]

theorem ridgeSolution_minimizes (A : Parameters d →L[ℝ] Parameters m) (y : Parameters m)
    (c lam : ℝ) (hc : 0 < c) (hlam : 0 < lam) :
    ∀ z, objective A y c lam (ridgeSolution A y c lam hc hlam) ≤
      objective A y c lam z :=
  stationary_minimizes A y c lam hc.le hlam.le (grad_ridgeSolution A y c lam hc hlam)

theorem eq_ridgeSolution_of_minimizer (A : Parameters d →L[ℝ] Parameters m)
    (y : Parameters m) (c lam : ℝ) (hc : 0 < c) (hlam : 0 < lam) {x : Parameters d}
    (hmin : ∀ z, objective A y c lam x ≤ objective A y c lam z) :
    x = ridgeSolution A y c lam hc hlam := by
  have hg := minimizer_grad_eq_zero A y c lam hc.le hlam hmin
  rw [grad_eq_hessian_sub] at hg
  have hx : hessian A c lam x = c • A.adjoint y := by
    exact sub_eq_zero.mp hg
  apply (hessian_bijective A c lam hc.le hlam).1
  rw [hx]
  exact (hessian_ridgeSolution A y c lam hc hlam).symm

/-- Exact gradient descent converges to `ridgeSolution`, the estimator below. -/
theorem gd_approaches_ridgeSolution (A : Parameters d →L[ℝ] Parameters m)
    (y : Parameters m) (c lam : ℝ) (hc : 0 < c) (hlam : 0 < lam) (η : ℝ) (hη : 0 < η)
    (hstep : η * (smoothness A c lam : ℝ) ^ 2 < 2 * lam) :
    ∃ xStar, xStar = ridgeSolution A y c lam hc hlam ∧
      ∀ x k, ‖(Optim.GD.step η (gradient (objective A y c lam)))^[k] x - xStar‖ ^ 2 ≤
        (Optim.GD.q η lam (smoothness A c lam)) ^ k * ‖x - xStar‖ ^ 2 := by
  obtain ⟨xStar, hmin, hdist⟩ := training A y c lam hc.le hlam η hη hstep
  refine ⟨xStar, eq_ridgeSolution_of_minimizer A y c lam hc hlam hmin, ?_⟩
  intro x k
  exact (hdist x k).1

/-- Hat operator `H = c A (c A* A + λ I)⁻¹ A*`. -/
def hat (A : Parameters d →L[ℝ] Parameters m) (c lam : ℝ) (hc : 0 < c) (hlam : 0 < lam) :
    Parameters m →L[ℝ] Parameters m :=
  c • (A.comp ((hessianInv A c lam hc.le hlam).comp A.adjoint))

theorem prediction_decomp (A : Parameters d →L[ℝ] Parameters m) (θStar : Parameters d)
    (ε : Parameters m) (c lam : ℝ) (hc : 0 < c) (hlam : 0 < lam) :
    A (ridgeSolution A (A θStar + ε) c lam hc hlam) - A θStar =
      - (lam • A (hessianInv A c lam hc.le hlam θStar)) + hat A c lam hc hlam ε := by
  let w : Parameters d := hessianInv A c lam hc.le hlam θStar
  have hw : hessian A c lam w = θStar := hessian_apply_hessianInv A c lam hc.le hlam θStar
  let S : Parameters d →L[ℝ] Parameters d := c • A.adjoint.comp A
  have hS : S θStar = hessian A c lam θStar - lam • θStar := by
    simp [S, hessian, sub_eq_add_neg]
  have hinvS : hessianInv A c lam hc.le hlam (S θStar) = θStar - lam • w := by
    apply (hessian_bijective A c lam hc.le hlam).1
    rw [hessian_apply_hessianInv, hS]
    simp [map_sub, map_smul, hw]
  have hy : c • A.adjoint (A θStar + ε) = S θStar + c • A.adjoint ε := by
    simp [S, map_add, smul_add]
  have hsol : ridgeSolution A (A θStar + ε) c lam hc hlam =
      (θStar - lam • w) + hessianInv A c lam hc.le hlam (c • A.adjoint ε) := by
    rw [ridgeSolution, hy, map_add, hinvS]
  rw [hsol, map_add, map_sub]
  have hhat : hat A c lam hc hlam ε =
      A (hessianInv A c lam hc.le hlam (c • A.adjoint ε)) := by
    simp [hat, map_smul]
  rw [hhat, map_smul]
  simp only [w]
  abel

theorem bias_sq_le (A : Parameters d →L[ℝ] Parameters m) (θStar : Parameters d)
    (c lam : ℝ) (hc : 0 < c) (hlam : 0 < lam) :
    ‖lam • A (hessianInv A c lam hc.le hlam θStar)‖ ^ 2 ≤
      (lam / (4 * c)) * ‖θStar‖ ^ 2 := by
  let w : Parameters d := hessianInv A c lam hc.le hlam θStar
  have hw : hessian A c lam w = θStar := hessian_apply_hessianInv A c lam hc.le hlam θStar
  let S : Parameters d →L[ℝ] Parameters d := c • A.adjoint.comp A
  have hsplit : θStar = S w + lam • w := by
    have : hessian A c lam w = S w + lam • w := by simp [S, hessian]
    simpa [hw] using this
  have hquad : ⟪S w, w⟫ = c * ‖A w‖ ^ 2 := by
    simp [S, inner_smul_left, adjoint_inner_left]
  have hdiff : ‖θStar‖ ^ 2 - ‖S w - lam • w‖ ^ 2 = 4 * lam * ⟪S w, w⟫ := by
    rw [hsplit, norm_add_sq_real, norm_sub_sq_real]
    simp [norm_smul, real_inner_smul_right, abs_of_pos hlam, mul_pow]
    ring
  have hinner : ⟪S w, w⟫ ≤ ‖θStar‖ ^ 2 / (4 * lam) := by
    have h4 : 4 * lam * ⟪S w, w⟫ ≤ ‖θStar‖ ^ 2 := by
      have := sq_nonneg ‖S w - lam • w‖
      linarith [hdiff]
    rw [le_div_iff₀ (mul_pos (by norm_num : (0 : ℝ) < 4) hlam)]
    linarith
  have hAw : ‖A w‖ ^ 2 ≤ ‖θStar‖ ^ 2 / (4 * lam * c) := by
    rw [le_div_iff₀ (mul_pos (mul_pos (by norm_num : (0 : ℝ) < 4) hlam) hc)]
    have hcq : c * ‖A w‖ ^ 2 = ⟪S w, w⟫ := hquad.symm
    nlinarith [hinner, hcq]
  calc
    ‖lam • A w‖ ^ 2 = lam ^ 2 * ‖A w‖ ^ 2 := by
      simp [norm_smul, abs_of_pos hlam, mul_pow]
    _ ≤ lam ^ 2 * (‖θStar‖ ^ 2 / (4 * lam * c)) := by
      gcongr
    _ = (lam / (4 * c)) * ‖θStar‖ ^ 2 := by
      field_simp

theorem hat_quadratic (A : Parameters d →L[ℝ] Parameters m) (c lam : ℝ)
    (hc : 0 < c) (hlam : 0 < lam) (v : Parameters m) :
    ‖hat A c lam hc hlam v‖ ^ 2 ≤ ⟪hat A c lam hc hlam v, v⟫ ∧
      ⟪hat A c lam hc hlam v, v⟫ ≤ ‖v‖ ^ 2 := by
  let z : Parameters d := hessianInv A c lam hc.le hlam (A.adjoint v)
  have hz : hessian A c lam z = A.adjoint v :=
    hessian_apply_hessianInv A c lam hc.le hlam (A.adjoint v)
  have hHz : hat A c lam hc hlam v = c • A z := by simp [hat, z]
  have hinner : ⟪hat A c lam hc hlam v, v⟫ =
      c * (c * ‖A z‖ ^ 2 + lam * ‖z‖ ^ 2) := by
    rw [hHz, inner_smul_left, starRingEnd_apply, star_trivial]
    rw [(adjoint_inner_right A z v).symm, ← hz, inner_hessian]
  have hnorm : ‖hat A c lam hc hlam v‖ ^ 2 = c ^ 2 * ‖A z‖ ^ 2 := by
    rw [hHz, norm_smul, Real.norm_eq_abs, abs_of_pos hc]
    ring
  have hle : ‖hat A c lam hc hlam v‖ ^ 2 ≤ ⟪hat A c lam hc hlam v, v⟫ := by
    nlinarith [hnorm, hinner, mul_nonneg hc.le (sq_nonneg ‖A z‖),
      mul_nonneg (mul_nonneg hc.le hlam.le) (sq_nonneg ‖z‖)]
  have hcs : ⟪hat A c lam hc hlam v, v⟫ ≤ ‖hat A c lam hc hlam v‖ * ‖v‖ := by
    rw [hHz, inner_smul_left]
    calc
      c * ⟪A z, v⟫ ≤ c * (‖A z‖ * ‖v‖) := by
        gcongr
        exact real_inner_le_norm _ _
      _ = ‖hat A c lam hc hlam v‖ * ‖v‖ := by
        rw [hHz, norm_smul, Real.norm_eq_abs, abs_of_pos hc]
        ring
  refine ⟨hle, ?_⟩
  by_cases hzero : ‖hat A c lam hc hlam v‖ = 0
  · have hvan : ⟪hat A c lam hc hlam v, v⟫ = 0 := by
      have hle0 : 0 ≤ ⟪hat A c lam hc hlam v, v⟫ := by simpa [hzero] using hle
      have hcs0 : ⟪hat A c lam hc hlam v, v⟫ ≤ 0 := by simpa [hzero] using hcs
      linarith
    rw [hvan]
    exact sq_nonneg _
  · have ht : 0 < ‖hat A c lam hc hlam v‖ :=
      lt_of_le_of_ne (norm_nonneg _) (Ne.symm hzero)
    have hstep : ‖hat A c lam hc hlam v‖ ≤ ‖v‖ := by
      have hmul : ‖hat A c lam hc hlam v‖ * ‖hat A c lam hc hlam v‖ ≤
          ‖hat A c lam hc hlam v‖ * ‖v‖ := by
        rw [← pow_two]
        exact hle.trans hcs
      exact le_of_mul_le_mul_left hmul ht
    calc
      ⟪hat A c lam hc hlam v, v⟫ ≤ ‖hat A c lam hc hlam v‖ * ‖v‖ := hcs
      _ ≤ ‖v‖ * ‖v‖ := by
        rw [mul_comm ‖hat A c lam hc hlam v‖ ‖v‖, mul_comm ‖v‖ ‖v‖]
        gcongr
      _ = ‖v‖ ^ 2 := by ring

theorem hat_isSymmetric (A : Parameters d →L[ℝ] Parameters m) (c lam : ℝ)
    (hc : 0 < c) (hlam : 0 < lam) :
    (hat A c lam hc hlam).toLinearMap.IsSymmetric := by
  intro x y
  have hcore : ⟪A (hessianInv A c lam hc.le hlam (A.adjoint x)), y⟫ =
      ⟪x, A (hessianInv A c lam hc.le hlam (A.adjoint y))⟫ := by
    calc
      ⟪A (hessianInv A c lam hc.le hlam (A.adjoint x)), y⟫
          = ⟪hessianInv A c lam hc.le hlam (A.adjoint x), A.adjoint y⟫ := by
            rw [← adjoint_inner_right]
      _ = ⟪A.adjoint x, hessianInv A c lam hc.le hlam (A.adjoint y)⟫ :=
            hessianInv_symmetric A c lam hc.le hlam _ _
      _ = ⟪x, A (hessianInv A c lam hc.le hlam (A.adjoint y))⟫ := by
            rw [adjoint_inner_left]
  simp only [hat, ContinuousLinearMap.toLinearMap_smul, ContinuousLinearMap.toLinearMap_comp,
    LinearMap.smul_apply, LinearMap.comp_apply, ContinuousLinearMap.coe_coe,
    real_inner_smul_left, real_inner_smul_right, hcore]

theorem output_finrank : Module.finrank ℝ (Parameters m) = m := by
  rw [finrank_euclideanSpace, Fintype.card_fin]

/-- Ridge degrees of freedom: the trace of the hat operator. -/
def degreesOfFreedom (A : Parameters d →L[ℝ] Parameters m) (c lam : ℝ)
    (hc : 0 < c) (hlam : 0 < lam) : ℝ :=
  ∑ i, (hat_isSymmetric A c lam hc hlam).eigenvalues output_finrank i

theorem design_rank_le_input (A : Parameters d →L[ℝ] Parameters m) :
    Module.finrank ℝ (LinearMap.range A.toLinearMap) ≤ d := by
  simpa [finrank_euclideanSpace, Fintype.card_fin] using
    (LinearMap.finrank_range_le A.toLinearMap)

theorem design_rank_le_output (A : Parameters d →L[ℝ] Parameters m) :
    Module.finrank ℝ (LinearMap.range A.toLinearMap) ≤ m := by
  have hinj : Function.Injective (LinearMap.range A.toLinearMap).subtype :=
    Subtype.val_injective
  have hle := LinearMap.finrank_le_finrank_of_injective hinj
  simpa [output_finrank] using hle

theorem degreesOfFreedom_nonneg (A : Parameters d →L[ℝ] Parameters m) (c lam : ℝ)
    (hc : 0 < c) (hlam : 0 < lam) : 0 ≤ degreesOfFreedom A c lam hc hlam := by
  apply Finset.sum_nonneg
  intro i _
  have hb := (hat_isSymmetric A c lam hc hlam).hasEigenvector_eigenvectorBasis output_finrank i
  have happ := (hat_isSymmetric A c lam hc hlam).apply_eigenvectorBasis output_finrank i
  let b := (hat_isSymmetric A c lam hc hlam).eigenvectorBasis output_finrank
  have hnorm : ‖b i‖ = 1 := b.orthonormal.norm_eq_one i
  have hquad := (hat_quadratic A c lam hc hlam (b i)).1
  have hinner : ⟪hat A c lam hc hlam (b i), b i⟫ =
      (hat_isSymmetric A c lam hc hlam).eigenvalues output_finrank i := by
    have happ' : hat A c lam hc hlam (b i) =
        (hat_isSymmetric A c lam hc hlam).eigenvalues output_finrank i • b i := by
      simp only [b]
      exact happ
    rw [happ', real_inner_smul_left, real_inner_self_eq_norm_sq, hnorm]
    ring
  nlinarith

theorem degreesOfFreedom_le_rank (A : Parameters d →L[ℝ] Parameters m) (c lam : ℝ)
    (hc : 0 < c) (hlam : 0 < lam) :
    degreesOfFreedom A c lam hc hlam ≤
      (Module.finrank ℝ (LinearMap.range A.toLinearMap) : ℝ) := by
  classical
  let T := (hat A c lam hc hlam).toLinearMap
  have hT : T.IsSymmetric := hat_isSymmetric A c lam hc hlam
  let μ : Fin m → ℝ := hT.eigenvalues output_finrank
  let b := hT.eigenvectorBasis output_finrank
  have hIcc : ∀ i, 0 ≤ μ i ∧ μ i ≤ 1 := by
    intro i
    have happ := hT.apply_eigenvectorBasis output_finrank i
    have hnorm : ‖b i‖ = 1 := b.orthonormal.norm_eq_one i
    have hinner : ⟪T (b i), b i⟫ = μ i := by
      have happ' : T (b i) = μ i • b i := by
        simp only [b, μ]
        exact happ
      rw [happ', real_inner_smul_left, real_inner_self_eq_norm_sq, hnorm]
      ring
    have hquad := hat_quadratic A c lam hc hlam (b i)
    have hco : ⟪hat A c lam hc hlam (b i), b i⟫ = ⟪T (b i), b i⟫ := rfl
    constructor
    · nlinarith [hquad.1, hinner, hco]
    · have := hquad.2
      rw [hco, hinner, hnorm] at this
      simpa using this
  let zeros := Finset.univ.filter fun i : Fin m => μ i = 0
  let nonzeros := Finset.univ.filter fun i : Fin m => ¬ μ i = 0
  have hcard : zeros.card + nonzeros.card = m := by
    simpa [zeros, nonzeros, Finset.card_univ, Fintype.card_fin] using
      Finset.card_filter_add_card_filter_not
        (s := Finset.univ) (p := fun i : Fin m => μ i = 0)
  have hzero_card : zeros.card = Module.finrank ℝ (LinearMap.ker T) := by
    have hcount := hT.card_filter_eigenvalues_eq output_finrank (0 : ℝ)
    rw [Module.End.eigenspace_zero T] at hcount
    simpa [zeros, μ] using hcount
  have horth : (LinearMap.range A.toLinearMap)ᗮ ≤ LinearMap.ker T := by
    intro v hv
    rw [Submodule.mem_orthogonal] at hv
    have hAd : A.adjoint v = 0 := by
      have hinner : ⟪A (A.adjoint v), v⟫ = 0 :=
        hv _ (LinearMap.mem_range_self A.toLinearMap (A.adjoint v))
      rw [← adjoint_inner_right] at hinner
      exact inner_self_eq_zero.mp hinner
    simp [T, hat, hAd]
  have horth_le : Module.finrank ℝ (LinearMap.range A.toLinearMap)ᗮ ≤ zeros.card := by
    rw [hzero_card]
    exact LinearMap.finrank_le_finrank_of_injective (Submodule.inclusion_injective horth)
  have hdim := Submodule.finrank_add_finrank_orthogonal
    (K := LinearMap.range A.toLinearMap)
  have hnz : nonzeros.card ≤ Module.finrank ℝ (LinearMap.range A.toLinearMap) := by
    have horth_dim : Module.finrank ℝ (LinearMap.range A.toLinearMap) +
        Module.finrank ℝ (LinearMap.range A.toLinearMap)ᗮ = m := by
      simpa [output_finrank] using hdim
    omega
  have hsum : degreesOfFreedom A c lam hc hlam ≤ (nonzeros.card : ℝ) := by
    have hμ : degreesOfFreedom A c lam hc hlam = ∑ i, μ i := by
      simp [degreesOfFreedom, μ, T]
    rw [hμ]
    have hsplit : ∑ i, μ i = ∑ i ∈ nonzeros, μ i := by
      rw [← Finset.sum_filter_add_sum_filter_not (s := Finset.univ)
        (f := μ) (fun i => μ i = 0)]
      have hzero : ∑ i ∈ Finset.univ.filter fun i => μ i = 0, μ i = 0 := by
        apply Finset.sum_eq_zero
        intro i hi
        simpa using hi
      simp [hzero, nonzeros]
    rw [hsplit]
    calc
      ∑ i ∈ nonzeros, μ i ≤ ∑ i ∈ nonzeros, (1 : ℝ) := by
        apply Finset.sum_le_sum
        intro i hi
        exact (hIcc i).2
      _ = nonzeros.card := by simp
  exact hsum.trans (Nat.cast_le.mpr hnz)

/-- Mean-zero isotropic label noise on the fixed design. `σ ^ 2` is the variance
of every one-dimensional marginal. -/
structure LabelNoise (μ : Measure (Parameters m)) (σ : ℝ) : Prop where
  integrable_id : Integrable (fun z : Parameters m => z) μ
  mean_zero : ∫ z, z ∂μ = 0
  second : ∀ u v : Parameters m, Integrable (fun z => ⟪u, z⟫ * ⟪v, z⟫) μ
  covar : ∀ u v : Parameters m, ∫ z, ⟪u, z⟫ * ⟪v, z⟫ ∂μ = σ ^ 2 * ⟪u, v⟫

theorem noise_energy_eq (A : Parameters d →L[ℝ] Parameters m) (c lam : ℝ)
    (hc : 0 < c) (hlam : 0 < lam) {μ : Measure (Parameters m)} {σ : ℝ}
    (h : LabelNoise μ σ) :
    ∫ z, ‖hat A c lam hc hlam z‖ ^ 2 ∂μ =
      σ ^ 2 * ∑ i, ((hat_isSymmetric A c lam hc hlam).eigenvalues output_finrank i) ^ 2 := by
  let T := (hat A c lam hc hlam).toLinearMap
  let hT : T.IsSymmetric := hat_isSymmetric A c lam hc hlam
  let b := hT.eigenvectorBasis output_finrank
  let eig := hT.eigenvalues output_finrank
  have hsq : ∀ z, ‖hat A c lam hc hlam z‖ ^ 2 = ∑ i, ⟪b i, hat A c lam hc hlam z⟫ ^ 2 :=
    fun z => (b.sum_sq_inner_right (hat A c lam hc hlam z)).symm
  have hcoord : ∀ z i, ⟪b i, hat A c lam hc hlam z⟫ = eig i * ⟪b i, z⟫ := by
    intro z i
    have happ := hT.apply_eigenvectorBasis output_finrank i
    have hsym : ⟪b i, hat A c lam hc hlam z⟫ = ⟪T (b i), z⟫ := (hT (b i) z).symm
    rw [hsym]
    have happ' : T (b i) = eig i • b i := by
      simp only [b, eig]
      exact happ
    rw [happ', real_inner_smul_left]
  rw [integral_congr_ae (ae_of_all μ hsq)]
  have hint : ∀ i, Integrable (fun z => ⟪b i, hat A c lam hc hlam z⟫ ^ 2) μ := by
    intro i
    have hrew : (fun z => ⟪b i, hat A c lam hc hlam z⟫ ^ 2) =
        fun z => (eig i) ^ 2 * (⟪b i, z⟫ * ⟪b i, z⟫) := by
      funext z
      rw [hcoord z i]
      ring
    rw [hrew]
    exact (h.second (b i) (b i)).const_mul _
  rw [integral_finsetSum _ (fun i _ => hint i)]
  have hterm : ∀ i, ∫ z, ⟪b i, hat A c lam hc hlam z⟫ ^ 2 ∂μ = σ ^ 2 * (eig i) ^ 2 := by
    intro i
    have hrew : (fun z => ⟪b i, hat A c lam hc hlam z⟫ ^ 2) =
        fun z => (eig i) ^ 2 * (⟪b i, z⟫ * ⟪b i, z⟫) := by
      funext z
      rw [hcoord z i]
      ring
    rw [hrew, integral_const_mul, h.covar (b i) (b i), real_inner_self_eq_norm_sq]
    have hnorm : ‖b i‖ = 1 := b.orthonormal.norm_eq_one i
    rw [hnorm]
    ring
  rw [Finset.sum_congr rfl (fun i _ => hterm i), Finset.mul_sum]

theorem noise_energy_le_df (A : Parameters d →L[ℝ] Parameters m) (c lam : ℝ)
    (hc : 0 < c) (hlam : 0 < lam) {μ : Measure (Parameters m)} {σ : ℝ}
    (h : LabelNoise μ σ) :
    ∫ z, ‖hat A c lam hc hlam z‖ ^ 2 ∂μ ≤ σ ^ 2 * degreesOfFreedom A c lam hc hlam := by
  rw [noise_energy_eq A c lam hc hlam h]
  have hsq : ∑ i, ((hat_isSymmetric A c lam hc hlam).eigenvalues output_finrank i) ^ 2 ≤
      degreesOfFreedom A c lam hc hlam := by
    unfold degreesOfFreedom
    apply Finset.sum_le_sum
    intro i _
    have hT := hat_isSymmetric A c lam hc hlam
    let b := hT.eigenvectorBasis output_finrank
    have happ := hT.apply_eigenvectorBasis output_finrank i
    have hnorm : ‖b i‖ = 1 := b.orthonormal.norm_eq_one i
    have hinner : ⟪hat A c lam hc hlam (b i), b i⟫ = hT.eigenvalues output_finrank i := by
      rw [show b i = hT.eigenvectorBasis output_finrank i from rfl]
      rw [show hat A c lam hc hlam (hT.eigenvectorBasis output_finrank i) =
          (hat A c lam hc hlam).toLinearMap (hT.eigenvectorBasis output_finrank i) from rfl]
      rw [happ, inner_smul_real_left, real_inner_self_eq_norm_sq, hnorm, smul_eq_mul, one_pow,
        mul_one]
    have hquad := hat_quadratic A c lam hc hlam (b i)
    have hμ0 : 0 ≤ hT.eigenvalues output_finrank i := by nlinarith [hquad.1, hinner]
    have hμ1 : hT.eigenvalues output_finrank i ≤ 1 := by
      have := hquad.2
      rw [hinner, hnorm] at this
      simpa using this
    nlinarith
  have hσ : 0 ≤ σ ^ 2 := sq_nonneg σ
  exact mul_le_mul_of_nonneg_left hsq hσ

theorem fixed_design_risk (A : Parameters d →L[ℝ] Parameters m) (θStar : Parameters d)
    (c lam : ℝ) (hc : 0 < c) (hlam : 0 < lam) {μ : Measure (Parameters m)} {σ : ℝ}
    [IsProbabilityMeasure μ] (h : LabelNoise μ σ) :
    ∫ z, ‖A (ridgeSolution A (A θStar + z) c lam hc hlam) - A θStar‖ ^ 2 ∂μ ≤
      (lam / (4 * c)) * ‖θStar‖ ^ 2 + σ ^ 2 * degreesOfFreedom A c lam hc hlam := by
  let H := hat A c lam hc hlam
  let bias : Parameters m := - (lam • A (hessianInv A c lam hc.le hlam θStar))
  have herr : ∀ z, A (ridgeSolution A (A θStar + z) c lam hc hlam) - A θStar = bias + H z := by
    intro z
    simpa [bias, H, sub_eq_add_neg] using prediction_decomp A θStar z c lam hc hlam
  have hbias : ‖bias‖ ^ 2 ≤ (lam / (4 * c)) * ‖θStar‖ ^ 2 := by
    simpa [bias, norm_neg] using bias_sq_le A θStar c lam hc hlam
  have hconst : Integrable (fun _ : Parameters m => ‖bias‖ ^ 2) μ :=
    integrable_const _
  have hvar_int : Integrable (fun z => ‖H z‖ ^ 2) μ := by
    have hsq : (fun z => ‖H z‖ ^ 2) = fun z =>
        ∑ i, ⟪((hat_isSymmetric A c lam hc hlam).eigenvectorBasis output_finrank) i, H z⟫ ^ 2 := by
      funext z
      exact ((hat_isSymmetric A c lam hc hlam).eigenvectorBasis output_finrank).sum_sq_inner_right
        (H z) |>.symm
    rw [hsq]
    refine integrable_finsetSum _ ?_
    intro i _
    let b := (hat_isSymmetric A c lam hc hlam).eigenvectorBasis output_finrank
    let eig := (hat_isSymmetric A c lam hc hlam).eigenvalues output_finrank
    have hcoord : ∀ z, ⟪b i, H z⟫ = eig i * ⟪b i, z⟫ := by
      intro z
      have happ := (hat_isSymmetric A c lam hc hlam).apply_eigenvectorBasis output_finrank i
      have hsym := ((hat_isSymmetric A c lam hc hlam) (b i) z).symm
      change ⟪b i, (hat A c lam hc hlam).toLinearMap z⟫ = eig i * ⟪b i, z⟫
      simp only [b] at hsym ⊢
      rw [hsym, happ, inner_smul_real_left, smul_eq_mul]
    have hrew : (fun z => ⟪b i, H z⟫ ^ 2) = fun z => (eig i) ^ 2 * (⟪b i, z⟫ * ⟪b i, z⟫) := by
      funext z
      rw [hcoord z]
      ring
    rw [hrew]
    exact (h.second (b i) (b i)).const_mul _
  have hcross_int : Integrable (fun z => ⟪bias, H z⟫) μ :=
    ((innerSL ℝ bias).comp H).integrable_comp h.integrable_id
  have hcross : ∫ z, ⟪bias, H z⟫ ∂μ = 0 := by
    have hfun : (fun z => ⟪bias, H z⟫) = fun z => (innerSL ℝ bias) (H z) := by
      funext z
      rw [innerSL_apply_apply]
    rw [hfun, ContinuousLinearMap.integral_comp_comm (innerSL ℝ bias)
        (H.integrable_comp h.integrable_id),
      ContinuousLinearMap.integral_comp_comm H h.integrable_id, h.mean_zero]
    simp [innerSL_apply_apply]
  have hsplit : ∀ z, ‖bias + H z‖ ^ 2 = ‖bias‖ ^ 2 + ‖H z‖ ^ 2 + 2 * ⟪bias, H z⟫ :=
    fun z => by rw [norm_add_sq_real]; ring
  rw [integral_congr_ae (ae_of_all μ (fun z => by rw [herr z, hsplit z]))]
  have hassoc : (fun a => ‖bias‖ ^ 2 + ‖H a‖ ^ 2 + 2 * ⟪bias, H a⟫) =
      fun a => ‖bias‖ ^ 2 + (‖H a‖ ^ 2 + 2 * ⟪bias, H a⟫) := by
    funext a
    ring
  rw [hassoc, integral_add (f := fun _ => ‖bias‖ ^ 2)
      (g := fun a => ‖H a‖ ^ 2 + 2 * ⟪bias, H a⟫) hconst
      (hvar_int.add (hcross_int.const_mul 2)),
    integral_add (f := fun a => ‖H a‖ ^ 2) (g := fun a => 2 * ⟪bias, H a⟫)
      hvar_int (hcross_int.const_mul 2),
    integral_const_mul, hcross, integral_const]
  simp
  linarith [hbias, noise_energy_le_df A c lam hc hlam h]

/-- In-sample mean squared error. With `c = 1/m`, the bias coefficient is `λ/4`
and the variance is the degrees of freedom over the number of rows. -/
theorem fixed_design_mse (A : Parameters d →L[ℝ] Parameters m) (θStar : Parameters d)
    (lam : ℝ) (hm : 0 < m) (hlam : 0 < lam) {μ : Measure (Parameters m)} {σ : ℝ}
    [IsProbabilityMeasure μ] (h : LabelNoise μ σ) :
    (∫ z, ‖A (ridgeSolution A (A θStar + z) (m : ℝ)⁻¹ lam
        (inv_pos.mpr (Nat.cast_pos.mpr hm)) hlam) - A θStar‖ ^ 2 ∂μ) / m ≤
      (lam / 4) * ‖θStar‖ ^ 2 +
        σ ^ 2 * degreesOfFreedom A ((m : ℝ)⁻¹) lam
          (inv_pos.mpr (Nat.cast_pos.mpr hm)) hlam / m := by
  have hc : 0 < (m : ℝ)⁻¹ := inv_pos.mpr (Nat.cast_pos.mpr hm)
  have hrisk := fixed_design_risk A θStar ((m : ℝ)⁻¹) lam hc hlam h
  have hm0 : (0 : ℝ) < m := Nat.cast_pos.mpr hm
  rw [div_le_iff₀ hm0]
  have hbias : lam / (4 * (m : ℝ)⁻¹) = (lam / 4) * m := by
    field_simp
  refine hrisk.trans ?_
  rw [hbias, add_mul, div_mul_cancel₀ _ (ne_of_gt hm0)]
  linarith

theorem fixed_design_mse_of_rank (A : Parameters d →L[ℝ] Parameters m)
    (θStar : Parameters d) (lam : ℝ) (hm : 0 < m) (hlam : 0 < lam)
    {μ : Measure (Parameters m)} {σ : ℝ} [IsProbabilityMeasure μ] (h : LabelNoise μ σ)
    {r : ℕ} (hr : Module.finrank ℝ (LinearMap.range A.toLinearMap) ≤ r) :
    (∫ z, ‖A (ridgeSolution A (A θStar + z) (m : ℝ)⁻¹ lam
        (inv_pos.mpr (Nat.cast_pos.mpr hm)) hlam) - A θStar‖ ^ 2 ∂μ) / m ≤
      (lam / 4) * ‖θStar‖ ^ 2 + σ ^ 2 * (r : ℝ) / m := by
  have hc : 0 < (m : ℝ)⁻¹ := inv_pos.mpr (Nat.cast_pos.mpr hm)
  have hdf := degreesOfFreedom_le_rank A ((m : ℝ)⁻¹) lam hc hlam
  have hmse := fixed_design_mse A θStar lam hm hlam h
  have hvar : σ ^ 2 * degreesOfFreedom A ((m : ℝ)⁻¹) lam hc hlam / m ≤
      σ ^ 2 * (r : ℝ) / m := by
    have hσ : 0 ≤ σ ^ 2 := sq_nonneg σ
    have hm0 : (0 : ℝ) ≤ m := Nat.cast_nonneg _
    have hr' : degreesOfFreedom A ((m : ℝ)⁻¹) lam hc hlam ≤ (r : ℝ) :=
      hdf.trans (Nat.cast_le.mpr hr)
    exact div_le_div_of_nonneg_right (mul_le_mul_of_nonneg_left hr' hσ) hm0
  linarith

end OpenTheory.RidgeRisk
