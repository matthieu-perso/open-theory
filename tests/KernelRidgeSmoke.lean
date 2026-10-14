import OpenTheory.Generalization.KernelRidge

open OpenTheory

-- Moving the primal-dual identity, the kernel rank, or the kernel risk bound must break these checks.
#check KernelRidge.prediction_eq_kernelRidge
#check KernelRidge.prediction_eq_of_kernelOp
#check KernelRidge.kernel_rank_eq
#check KernelRidge.fixed_design_kernel_mse

-- A zero feature map has kernel zero, so the smoother predicts zero in every feature dimension.
example (hlam : (0 : ℝ) < 1) (y : Parameters 3) :
    KernelRidge.kernelSmoother (0 : Parameters 8 →L[ℝ] Parameters 3) 1 1
      (by norm_num : (0 : ℝ) ≤ 1) hlam y = 0 := by
  simp [KernelRidge.kernelSmoother, KernelRidge.kernelOp, map_zero]

-- The kernel rank equals the feature-map rank, including when that rank is zero.
example (hc : (1 : ℝ) ≠ 0) :
    Module.finrank ℝ
        (LinearMap.range (KernelRidge.kernelOp (0 : Parameters 8 →L[ℝ] Parameters 3) 1).toLinearMap) =
      Module.finrank ℝ
        (LinearMap.range (0 : Parameters 8 →L[ℝ] Parameters 3).toLinearMap) :=
  KernelRidge.kernel_rank_eq (0 : Parameters 8 →L[ℝ] Parameters 3) 1 hc
