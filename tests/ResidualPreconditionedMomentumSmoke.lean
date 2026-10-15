import OpenTheory.Optimization.ResidualPreconditionedMomentum

open OpenTheory
open OpenTheory.ResidualPreconditionedMomentum
open OpenTheory.PreconditionedDescent
open OpenTheory.MomentumAdamTrajectory
open OpenTheory.MLPLocalSharpness

-- Section 1: Coordinate-Wise Residual-Coupled Local Sharpness
#check coordResidualSharpness
#check lam_le_coordResidualSharpness
#check coordResidualSharpness_pos
#check coordResidualSharpness_sub_eq
#check coordResidualSharpness_mono_in_residual
#check coordResidualSharpness_strict_mono_in_residual
#check coordResidualSharpness_eq_mlpLocalSharpness
#check mlp_residual_drop_strictly_flattens_landscape

-- Section 2: Exact Variational Optimality of Residual-Preconditioned Momentum (RPM)
#check dampedCoordVar
#check dampedCoordVar_nonneg
#check dampedCoordVar_zero_noise
#check totalCoordMoment
#check totalCoordMoment_nonneg
#check totalCoordMoment_pos_of_grad_ne_zero
#check rpmOptimalStep
#check rpmResidualStep
#check optimizerNetProgress_eq_coord_sum
#check optimizerNetProgress_rpmOptimalStep_eq
#check rpm_variational_sub_eq_sum_sq
#check rpm_variational_global_optimality
#check rpm_variational_strict_optimality

-- Section 3: Systematic Regime Classification
#check rpmOptimalStep_zero_noise_eq_optimalDiagStep
#check rpmOptimalStep_isotropic_zero_noise_eq_scalar
#check rpm_progress_gap_from_sharpness_drop_eq
#check rpm_strict_speedup_from_residual_drop
#check rpmResidualStep_strict_speedup
#check rpm_momentum_strictly_increases_optimal_progress

-- Section 4: Dynamic Residual-Adapted PL Contraction Acceleration
#check rpm_dynamic_pl_contraction_strict_improvement
#check rpm_residual_drop_pl_contraction_strict_improvement

-- Smoke test 1: As prediction residual drops from R_0 = 10 to R_t = 2, coordinate local sharpness
-- strictly decreases.
example (i : Fin 4) :
    coordResidualSharpness 1 (fun _ => 2) (fun _ => 3) 9 2 i <
      coordResidualSharpness 1 (fun _ => 2) (fun _ => 3) 9 10 i := by
  exact coordResidualSharpness_strict_mono_in_residual 1 (fun _ => 2) (fun _ => 3) i
    (by norm_num) (by norm_num) (by decide)

-- Smoke test 2: Dynamic PL contraction factor strictly improves when local sharpness drops from
-- S_init = 8 to S_curr = 4 with μ = 1: 1 - 1/4 = 3/4 < 1 - 1/8 = 7/8.
example :
    0 ≤ (1 : ℝ) - 1 / 4 ∧
      (1 : ℝ) - 1 / 4 < 1 - 1 / 8 ∧
      ((1 : ℝ) - 1 / 8) - (1 - 1 / 4) = 1 * (8 - 4) / (4 * 8) := by
  exact rpm_dynamic_pl_contraction_strict_improvement (by norm_num) (by norm_num) (by norm_num)
