import OpenTheory.Representation.Alignment
import OpenTheory.Representation.FeatureLearningAdvantage

open OpenTheory
open OpenTheory.FeatureStep
open OpenTheory.MLPTrajectory
open OpenTheory.MLPComparison
open OpenTheory.MLPEndpoint
open OpenTheory.FeatureLearningAdvantage

-- 1. Alignment & Frobenius Pythagorean checks (`Alignment.lean`)
#check OpenTheory.frobeniusSq_sub_mul_orthoProjector
#check OpenTheory.frobeniusSq_mul_orthoProjector_le
#check OpenTheory.alignment_le_one
#check OpenTheory.alignment_feature_distance

-- 2. Dynamic feature-learning advantage (`FeatureLearningAdvantage.lean`)
#check FeatureLearningAdvantage.feature_learning_preactivation_shift_exact
#check FeatureLearningAdvantage.feature_learning_signed_preactivation_shift_exact
#check FeatureLearningAdvantage.standard_lr_preactivation_shift_suppressed
#check FeatureLearningAdvantage.feature_learning_vs_standard_shift_ratio
#check FeatureLearningAdvantage.feature_learning_preactivation_sq_strict_growth
#check FeatureLearningAdvantage.kernel_strict_growth_of_slope_growth

-- 3. Statistical ridge bias & expected risk improvement (`FeatureLearningAdvantage.lean`)
#check FeatureLearningAdvantage.ridgeWeights_objective_eq_minRidgeObj
#check FeatureLearningAdvantage.endpoint_expected_mse_eq_ridgeExpectedMSE
#check FeatureLearningAdvantage.minRidgeObj_sub_eq
#check FeatureLearningAdvantage.minRidgeObj_strict_anti
#check FeatureLearningAdvantage.biasSq_sub_eq
#check FeatureLearningAdvantage.biasSq_strict_anti
#check FeatureLearningAdvantage.biasSq_reduction_lower_bound
#check FeatureLearningAdvantage.ridgeExpectedMSE_sub_eq
#check FeatureLearningAdvantage.ridge_mse_strict_improvement_of_kernel_growth
#check FeatureLearningAdvantage.ridge_mse_strict_improvement_of_weighted_snr

-- 4. Frobenius subspace alignment advantage (`FeatureLearningAdvantage.lean`)
#check FeatureLearningAdvantage.frobeniusSq_residual_eq_one_sub_alignment
#check FeatureLearningAdvantage.residual_projection_strict_decrease_of_alignment_growth
#check FeatureLearningAdvantage.residual_projection_gap_of_alignment_growth

-- Concrete numerical verification of strict expected ridge MSE reduction:
-- With K₀ = 1, K₁ = 3, lam = 1, Δy = 2, σ = 1:
-- K₀ * σ² = 1 < 4 = lam * Δy², and K₁ * σ² = 3 ≤ 4 = lam * Δy².
-- ridgeExpectedMSE 1 1 2 1 = (1/2)^2 * 4 + (1/2)^2 * 1 = 5/4
-- ridgeExpectedMSE 3 1 2 1 = (1/4)^2 * 4 + (3/4)^2 * 1 = 4/16 + 9/16 = 13/16 < 20/16 = 5/4.
example : ridgeExpectedMSE 3 1 2 1 < ridgeExpectedMSE 1 1 2 1 := by
  exact ridge_mse_strict_improvement_of_kernel_growth
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)

-- Concrete numerical verification of strict bias reduction:
example : biasSq 3 1 2 < biasSq 1 1 2 := by
  exact biasSq_strict_anti (by norm_num) (by norm_num) (by norm_num) (by norm_num)

-- Concrete numerical verification of minRidgeObj strict reduction:
example : minRidgeObj 3 1 2 < minRidgeObj 1 1 2 := by
  exact minRidgeObj_strict_anti (by norm_num) (by norm_num) (by norm_num) (by norm_num)
