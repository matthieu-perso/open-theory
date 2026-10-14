import OpenTheory.Dynamics.MLPLocalSharpness

open OpenTheory

#check MLPLocalSharpness.unflatten_add
#check MLPLocalSharpness.atDisplacement_add
#check MLPLocalSharpness.batchOutput_add
#check MLPLocalSharpness.initialOutput_atDisplacement
#check MLPLocalSharpness.batchGrad
#check MLPLocalSharpness.nonlinearBatchStep_eq_sub_batchGrad
#check MLPLocalSharpness.inner_batchGrad_eq
#check MLPLocalSharpness.mlpOutputLip
#check MLPLocalSharpness.mlpOutputCurv
#check MLPLocalSharpness.mlpLocalSharpness
#check MLPLocalSharpness.mlpOutputLip_nonneg
#check MLPLocalSharpness.mlpOutputCurv_nonneg
#check MLPLocalSharpness.lam_le_mlpLocalSharpness
#check MLPLocalSharpness.mlpLocalSharpness_pos
#check MLPLocalSharpness.mlpLocalSharpness_mono_residual
#check MLPLocalSharpness.mlpLocalSharpness_eq_of_pos
#check MLPLocalSharpness.stepRemainder
#check MLPLocalSharpness.batchOutput_add_eq
#check MLPLocalSharpness.batchResidual_add_eq
#check MLPLocalSharpness.stepOutputChange_norm_le
#check MLPLocalSharpness.stepRemainder_norm_le
#check MLPLocalSharpness.mlp_objective_quadratic_upper_bound
#check MLPLocalSharpness.mlp_objective_grad_step_quadratic_le
#check MLPLocalSharpness.mlp_local_sharpness_step_descent
#check MLPLocalSharpness.mlp_adaptive_step_optimal_drop
#check MLPLocalSharpness.stepProgress_nonneg_of_le_two_div
#check MLPLocalSharpness.mlp_safe_step_objective_descent
#check MLPLocalSharpness.mlp_safe_step_coercive_trajectory
#check MLPLocalSharpness.batchRegularizedObjective_nonneg
#check MLPLocalSharpness.mlp_adaptive_trajectory_step_drop
#check MLPLocalSharpness.mlp_adaptive_step_cumulative_grad_sq_le
#check MLPLocalSharpness.mlp_adaptive_step_min_grad_norm

-- Smoke test 1: Verify positive local sharpness and safe adaptive step size on concrete parameters
example : 0 < MLPLocalSharpness.mlpLocalSharpness 4 10 0.01 1 1 1 1 2 := by
  exact MLPLocalSharpness.mlpLocalSharpness_pos 4 10 (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num)

-- Smoke test 2: Verify monotonicity of derived local sharpness in the residual radius R
example :
    MLPLocalSharpness.mlpLocalSharpness 4 10 0.01 1 1 1 1 1 ≤
      MLPLocalSharpness.mlpLocalSharpness 4 10 0.01 1 1 1 1 3 := by
  exact MLPLocalSharpness.mlpLocalSharpness_mono_residual 4 10 0.01 1 1 1 1
    (by norm_num) (by norm_num) (by norm_num) (by norm_num)
