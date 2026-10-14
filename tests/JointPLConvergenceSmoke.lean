import OpenTheory.Optimization.JointPLConvergence

open OpenTheory
open OpenTheory.JointPLConvergence

#check JointPLConvergence.unregMSE
#check JointPLConvergence.unregGrad
#check JointPLConvergence.mlpUnregSharpness
#check JointPLConvergence.unregMSE_eq
#check JointPLConvergence.unregMSE_nonneg
#check JointPLConvergence.unregMSE_zero
#check JointPLConvergence.norm_residual_sq_eq_two_mul_n_mul_unregMSE
#check JointPLConvergence.unregGrad_eq
#check JointPLConvergence.norm_unregGrad_eq
#check JointPLConvergence.mlpUnregSharpness_pos
#check JointPLConvergence.mlp_unreg_quadratic_upper_bound
#check JointPLConvergence.mlp_unreg_step_descent
#check JointPLConvergence.plConstant
#check JointPLConvergence.jacobian_coercive_implies_pl
#check JointPLConvergence.joint_two_layer_pl_inequality
#check JointPLConvergence.joint_jacobian_coercive_implies_pl
#check JointPLConvergence.mlp_unreg_adaptive_pl_step
#check JointPLConvergence.mlp_global_geometric_convergence_to_zero_loss
#check JointPLConvergence.mlp_global_residual_sq_geometric_decay
#check JointPLConvergence.plContractionFactor
#check JointPLConvergence.feature_learning_contraction_rate_strict_improvement
#check JointPLConvergence.joint_vs_single_layer_contraction_strict_improvement
#check JointPLConvergence.spectral_amplification_strict_pl_acceleration

-- Smoke test 1: Positive unregularized local sharpness on concrete network dimensions
example : 0 < MLPLocalSharpness.mlpLocalSharpness 4 10 0 1 1 1 1 2 := by
  exact mlpUnregSharpness_pos (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)

-- Smoke test 2: Feature learning doubling the singular value (σ₀ = 1 → σ_feat = 2) strictly
-- improves both the 1-step contraction factor and multi-step loss bound.
example :
    1 - (2 ^ 2 / (10 : ℝ)) / 5 < 1 - (1 ^ 2 / (10 : ℝ)) / 5 ∧
      (1 - (2 ^ 2 / (10 : ℝ)) / 5) ^ 4 * (3 : ℝ) <
        (1 - (1 ^ 2 / (10 : ℝ)) / 5) ^ 4 * (3 : ℝ) := by
  exact feature_learning_contraction_rate_strict_improvement
    (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num)

-- Smoke test 3: Joint two-layer training with σ₁ = 1, σ₂ = 1 strictly accelerates
-- convergence compared to hidden-only training (σ₁ = 1).
example :
    1 - ((1 ^ 2 + 1 ^ 2) / (10 : ℝ)) / 5 < 1 - (1 ^ 2 / (10 : ℝ)) / 5 ∧
      (1 - ((1 ^ 2 + 1 ^ 2) / (10 : ℝ)) / 5) ^ 3 * (2 : ℝ) <
        (1 - (1 ^ 2 / (10 : ℝ)) / 5) ^ 3 * (2 : ℝ) := by
  exact joint_vs_single_layer_contraction_strict_improvement
    (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num)
