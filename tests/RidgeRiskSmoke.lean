import OpenTheory.Generalization.RidgeRisk

open OpenTheory

-- Moving the fixed-design statement, the rank comparison, or the GD link must break these checks.
#check RidgeRisk.fixed_design_mse
#check RidgeRisk.fixed_design_mse_of_rank
#check RidgeRisk.degreesOfFreedom_le_rank
#check RidgeRisk.gd_approaches_ridgeSolution

-- A zero design has rank 0, so the variance term vanishes while the parameter dimension stays 8.
example (hc : (0 : ℝ) < 1) (hlam : (0 : ℝ) < 1) :
    RidgeRisk.degreesOfFreedom (0 : Parameters 8 →L[ℝ] Parameters 3) 1 1 hc hlam = 0 := by
  have hle :=
    RidgeRisk.degreesOfFreedom_le_rank (0 : Parameters 8 →L[ℝ] Parameters 3) 1 1 hc hlam
  have hnn :=
    RidgeRisk.degreesOfFreedom_nonneg (0 : Parameters 8 →L[ℝ] Parameters 3) 1 1 hc hlam
  have hr :
      Module.finrank ℝ (LinearMap.range (0 : Parameters 8 →ₗ[ℝ] Parameters 3)) = 0 := by
    simp [LinearMap.range_zero]
  have hzero :
      RidgeRisk.degreesOfFreedom (0 : Parameters 8 →L[ℝ] Parameters 3) 1 1 hc hlam ≤ 0 := by
    simpa [ContinuousLinearMap.toLinearMap_zero, hr] using hle
  linarith

-- Output dimension caps the design rank, independently of the parameter dimension.
example (A : Parameters 100 →L[ℝ] Parameters 2) :
    Module.finrank ℝ (LinearMap.range A.toLinearMap) ≤ 2 :=
  RidgeRisk.design_rank_le_output A
