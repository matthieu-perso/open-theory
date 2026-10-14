import OpenTheory.Generalization.RidgeRisk

/-! Kernel form of the fixed-design ridge guarantee.

`kernelOp A c = c A A*` is the kernel operator of a fixed feature map `A`.
Exact gradient descent still trains the primal ridge parameter. Its predictions
equal the kernel smoother `K (K + λ I)⁻¹` applied to the labels, so those
predictions depend on the training labels only through this Gram. The degrees
of freedom in `fixed_design_mse` are the trace of that smoother. The rank of
`K` equals the rank of `A`, so the feature dimension cancels.

A frozen feature map, including the Jacobian features of a linearized model,
inherits the ridge bound with this kernel. This file does not prove that a
neural network stays close to its linearization, and it does not prove a
random-design guarantee.
-/

noncomputable section

open MeasureTheory OpenTheory.Ridge OpenTheory.RidgeRisk
open scoped RealInnerProductSpace
open ContinuousLinearMap

namespace OpenTheory.KernelRidge

variable {d m : ℕ}

/-- Kernel operator `K = c A A*` on the label space. -/
def kernelOp (A : Parameters d →L[ℝ] Parameters m) (c : ℝ) :
    Parameters m →L[ℝ] Parameters m :=
  c • A.comp A.adjoint

/-- `K + λ I` on the label space. -/
def kernelHessian (A : Parameters d →L[ℝ] Parameters m) (c lam : ℝ) :
    Parameters m →L[ℝ] Parameters m :=
  kernelOp A c + lam • ContinuousLinearMap.id ℝ (Parameters m)

theorem inner_kernelHessian (A : Parameters d →L[ℝ] Parameters m) (c lam : ℝ)
    (v : Parameters m) :
    ⟪kernelHessian A c lam v, v⟫ = c * ‖A.adjoint v‖ ^ 2 + lam * ‖v‖ ^ 2 := by
  have hK : ⟪(A.comp A.adjoint) v, v⟫ = ‖A.adjoint v‖ ^ 2 := by
    rw [comp_apply, ← adjoint_inner_right, real_inner_self_eq_norm_sq]
  have hsmul : ⟪kernelOp A c v, v⟫ = c * ‖A.adjoint v‖ ^ 2 := by
    rw [kernelOp, smul_apply, real_inner_smul_left, hK]
  rw [kernelHessian, add_apply, inner_add_left, hsmul, smul_apply, id_apply,
    real_inner_smul_left, real_inner_self_eq_norm_sq]

theorem kernel_bijective (A : Parameters d →L[ℝ] Parameters m) (c lam : ℝ)
    (hc : 0 ≤ c) (hlam : 0 < lam) : Function.Bijective (kernelHessian A c lam) := by
  have hi : Function.Injective (kernelHessian A c lam) := by
    apply (kernelHessian A c lam).toLinearMap.ker_eq_bot.mp
    rw [LinearMap.ker_eq_bot']
    intro v hv
    have he := inner_kernelHessian A c lam v
    change kernelHessian A c lam v = 0 at hv
    rw [hv, inner_zero_left] at he
    have hn : ‖v‖ ^ 2 = 0 := by
      nlinarith [mul_nonneg hc (sq_nonneg ‖A.adjoint v‖), sq_nonneg ‖v‖]
    exact norm_eq_zero.mp (sq_eq_zero_iff.mp hn)
  exact ⟨hi, LinearMap.injective_iff_surjective.mp hi⟩

def kernelEquiv (A : Parameters d →L[ℝ] Parameters m) (c lam : ℝ)
    (hc : 0 ≤ c) (hlam : 0 < lam) : Parameters m ≃ₗ[ℝ] Parameters m :=
  let f := (kernelHessian A c lam).toLinearMap
  { f, Equiv.ofBijective f (kernel_bijective A c lam hc hlam) with }

def kernelInv (A : Parameters d →L[ℝ] Parameters m) (c lam : ℝ)
    (hc : 0 ≤ c) (hlam : 0 < lam) : Parameters m →L[ℝ] Parameters m :=
  LinearMap.toContinuousLinearMap (kernelEquiv A c lam hc hlam).symm.toLinearMap

theorem kernelInv_apply (A : Parameters d →L[ℝ] Parameters m) (c lam : ℝ)
    (hc : 0 ≤ c) (hlam : 0 < lam) (x : Parameters m) :
    kernelInv A c lam hc hlam x = (kernelEquiv A c lam hc hlam).symm x :=
  rfl

theorem kernel_apply_kernelInv (A : Parameters d →L[ℝ] Parameters m) (c lam : ℝ)
    (hc : 0 ≤ c) (hlam : 0 < lam) (x : Parameters m) :
    kernelHessian A c lam (kernelInv A c lam hc hlam x) = x := by
  rw [kernelInv_apply]
  exact (kernelEquiv A c lam hc hlam).apply_symm_apply x

theorem kernelInv_apply_kernel (A : Parameters d →L[ℝ] Parameters m) (c lam : ℝ)
    (hc : 0 ≤ c) (hlam : 0 < lam) (x : Parameters m) :
    kernelInv A c lam hc hlam (kernelHessian A c lam x) = x := by
  rw [kernelInv_apply]
  exact (kernelEquiv A c lam hc hlam).symm_apply_apply x

/-- Pushing the design through the two ridge operators: `(K + λ I) A = A (c A* A + λ I)`. -/
theorem kernelHessian_comp_design (A : Parameters d →L[ℝ] Parameters m) (c lam : ℝ) :
    (kernelHessian A c lam).comp A = A.comp (hessian A c lam) := by
  apply ContinuousLinearMap.ext
  intro v
  simp only [comp_apply, kernelHessian, kernelOp, hessian, add_apply, smul_apply, id_apply,
    map_add, map_smul]

theorem design_comp_hessianInv (A : Parameters d →L[ℝ] Parameters m) (c lam : ℝ)
    (hc : 0 ≤ c) (hlam : 0 < lam) :
    A.comp (hessianInv A c lam hc hlam) = (kernelInv A c lam hc hlam).comp A := by
  apply ContinuousLinearMap.ext
  intro v
  apply (kernel_bijective A c lam hc hlam).1
  rw [comp_apply, comp_apply, kernel_apply_kernelInv]
  have hpt := congr_fun
    (congrArg (fun f : Parameters d →L[ℝ] Parameters m => (f : Parameters d → Parameters m))
      (kernelHessian_comp_design A c lam))
    (hessianInv A c lam hc hlam v)
  simp only [comp_apply] at hpt
  rw [hpt, hessian_apply_hessianInv]

/-- Kernel smoother `K (K + λ I)⁻¹`. -/
def kernelSmoother (A : Parameters d →L[ℝ] Parameters m) (c lam : ℝ)
    (hc : 0 ≤ c) (hlam : 0 < lam) : Parameters m →L[ℝ] Parameters m :=
  (kernelInv A c lam hc hlam).comp (kernelOp A c)

/-- Primal ridge predictions are the kernel smoother applied to the labels. -/
theorem prediction_eq_kernelRidge (A : Parameters d →L[ℝ] Parameters m) (y : Parameters m)
    (c lam : ℝ) (hc : 0 < c) (hlam : 0 < lam) :
    A (ridgeSolution A y c lam hc hlam) = kernelSmoother A c lam hc.le hlam y := by
  rw [ridgeSolution, kernelSmoother, ← comp_apply, design_comp_hessianInv, comp_apply, kernelOp]
  simp [comp_apply, smul_apply, map_smul]

/-- The ridge hat operator is the same kernel smoother. -/
theorem hat_eq_kernelSmoother (A : Parameters d →L[ℝ] Parameters m) (c lam : ℝ)
    (hc : 0 < c) (hlam : 0 < lam) :
    hat A c lam hc hlam = kernelSmoother A c lam hc.le hlam := by
  apply ContinuousLinearMap.ext
  intro ε
  simp only [hat, kernelSmoother, kernelOp, comp_apply, smul_apply, map_smul]
  rw [← comp_apply, design_comp_hessianInv, comp_apply]

/-- Equal kernel Grams give the same smoother, whatever feature coordinates produced them. -/
theorem kernelSmoother_eq_of_kernelOp {p : ℕ}
    (A : Parameters d →L[ℝ] Parameters m) (B : Parameters p →L[ℝ] Parameters m)
    (c lam : ℝ) (hc : 0 ≤ c) (hlam : 0 < lam)
    (hK : kernelOp A c = kernelOp B c) :
    kernelSmoother A c lam hc hlam = kernelSmoother B c lam hc hlam := by
  have hL : kernelHessian A c lam = kernelHessian B c lam := by
    simp [kernelHessian, hK]
  apply ContinuousLinearMap.ext
  intro y
  apply (kernel_bijective B c lam hc hlam).1
  have hAy : kernelHessian A c lam (kernelSmoother A c lam hc hlam y) = kernelOp A c y := by
    simp [kernelSmoother, comp_apply, kernel_apply_kernelInv]
  have hleft : kernelHessian B c lam (kernelSmoother A c lam hc hlam y) = kernelOp B c y := by
    rw [← hL, hAy, hK]
  have hright : kernelHessian B c lam (kernelSmoother B c lam hc hlam y) = kernelOp B c y := by
    simp [kernelSmoother, comp_apply, kernel_apply_kernelInv]
  rw [hleft, hright]

/-- Feature maps with the same kernel produce the same ridge predictions. -/
theorem prediction_eq_of_kernelOp {p : ℕ}
    (A : Parameters d →L[ℝ] Parameters m) (B : Parameters p →L[ℝ] Parameters m)
    (y : Parameters m) (c lam : ℝ) (hc : 0 < c) (hlam : 0 < lam)
    (hK : kernelOp A c = kernelOp B c) :
    A (ridgeSolution A y c lam hc hlam) = B (ridgeSolution B y c lam hc hlam) := by
  rw [prediction_eq_kernelRidge, prediction_eq_kernelRidge,
    kernelSmoother_eq_of_kernelOp A B c lam hc.le hlam hK]

/-- The kernel has the same rank as the feature map, so the feature dimension cancels. -/
theorem kernel_rank_eq (A : Parameters d →L[ℝ] Parameters m) (c : ℝ) (hc : c ≠ 0) :
    Module.finrank ℝ (LinearMap.range (kernelOp A c).toLinearMap) =
      Module.finrank ℝ (LinearMap.range A.toLinearMap) := by
  have hrange : LinearMap.range (kernelOp A c).toLinearMap =
      LinearMap.range A.toLinearMap := by
    simp only [kernelOp, ContinuousLinearMap.toLinearMap_smul,
      ContinuousLinearMap.toLinearMap_comp]
    rw [LinearMap.range_smul _ c hc, ← ContinuousLinearMap.adjoint_toLinearMap,
      LinearMap.range_self_comp_adjoint]
  rw [hrange]

/-- In-sample error of the kernel smoother. The variance term is the trace of
`K (K + λ I)⁻¹`, since that smoother is the ridge hat operator. -/
theorem fixed_design_kernel_mse (A : Parameters d →L[ℝ] Parameters m) (θStar : Parameters d)
    (lam : ℝ) (hm : 0 < m) (hlam : 0 < lam) {μ : Measure (Parameters m)} {σ : ℝ}
    [IsProbabilityMeasure μ] (h : LabelNoise μ σ) :
    (∫ z, ‖kernelSmoother A ((m : ℝ)⁻¹) lam (inv_pos.mpr (Nat.cast_pos.mpr hm)).le hlam
        (A θStar + z) - A θStar‖ ^ 2 ∂μ) / m ≤
      (lam / 4) * ‖θStar‖ ^ 2 +
        σ ^ 2 * degreesOfFreedom A ((m : ℝ)⁻¹) lam
          (inv_pos.mpr (Nat.cast_pos.mpr hm)) hlam / m := by
  have hc : 0 < (m : ℝ)⁻¹ := inv_pos.mpr (Nat.cast_pos.mpr hm)
  have hpred : ∀ z, kernelSmoother A ((m : ℝ)⁻¹) lam hc.le hlam (A θStar + z) =
      A (ridgeSolution A (A θStar + z) ((m : ℝ)⁻¹) lam hc hlam) :=
    fun z => (prediction_eq_kernelRidge A (A θStar + z) ((m : ℝ)⁻¹) lam hc hlam).symm
  rw [integral_congr_ae (ae_of_all μ (fun z => by rw [hpred z]))]
  exact fixed_design_mse A θStar lam hm hlam h

end OpenTheory.KernelRidge
