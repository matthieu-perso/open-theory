import OpenTheory.Optimization.StochasticSharpnessSGD

open OpenTheory

#check StochasticSharpnessSGD.sgdNetProgress
#check StochasticSharpnessSGD.sgdExpectedStepDrop
#check StochasticSharpnessSGD.sgdNetProgress_eq_linear_sub_quad
#check StochasticSharpnessSGD.sgdNetProgress_zero_noise
#check StochasticSharpnessSGD.sgd_local_sharpness_one_step
#check StochasticSharpnessSGD.sgd_pointwise_pythagorean_step
#check StochasticSharpnessSGD.noiseAwareStep
#check StochasticSharpnessSGD.sgd_optimal_lr_with_noise
#check StochasticSharpnessSGD.noiseAwareStep_zero_noise
#check StochasticSharpnessSGD.sgdNetProgress_noiseAwareStep_eq
#check StochasticSharpnessSGD.sgdNetProgress_sub_eq_sq
#check StochasticSharpnessSGD.noiseAwareStep_maximizes_sgdNetProgress
#check StochasticSharpnessSGD.sgdNetProgress_mono_batchSize
#check StochasticSharpnessSGD.criticalBatchRatio
#check StochasticSharpnessSGD.criticalBatchSize
#check StochasticSharpnessSGD.noiseAwareStep_eq_batch_efficiency
#check StochasticSharpnessSGD.optimal_sgd_progress_vs_full_batch
#check StochasticSharpnessSGD.critical_batch_half_efficiency
#check StochasticSharpnessSGD.small_batch_linear_scaling_lower_bound
#check StochasticSharpnessSGD.linear_scaling_regime_progress
#check StochasticSharpnessSGD.batch_size_doubling_speedup
#check StochasticSharpnessSGD.sgdNoiseFloor
#check StochasticSharpnessSGD.sgdNoiseFloor_nonneg
#check StochasticSharpnessSGD.sgd_pl_one_step_recurrence
#check StochasticSharpnessSGD.sgdNoiseFloor_fixed_point
#check StochasticSharpnessSGD.sgd_pl_noise_ball_convergence
#check StochasticSharpnessSGD.mlp_sgd_noise_floor_proportional_to_inv_batch

-- Smoke test 1: At the critical batch size B = σ_sq / G (e.g. G = 2, σ_sq = 64, B = 32),
-- the noise-aware optimal learning rate is half the full-batch adaptive step,
-- and expected per-step progress is 50% of full-batch progress.
example (S_loc : ℝ) (hS : 0 < S_loc) :
    StochasticSharpnessSGD.noiseAwareStep S_loc 2 64 32 = 1 / (2 * S_loc) ∧
      StochasticSharpnessSGD.sgdNetProgress
        (StochasticSharpnessSGD.noiseAwareStep S_loc 2 64 32) S_loc 2 64 32 =
        2 / (4 * S_loc) := by
  exact StochasticSharpnessSGD.critical_batch_half_efficiency S_loc 2 64 32
    (by decide) (by norm_num) hS (by norm_num)

-- Smoke test 2: Increasing batch size from B = 16 to B = 64 monotonically increases
-- expected mini-batch SGD progress for any learning rate η.
example (η : ℝ) :
    StochasticSharpnessSGD.sgdNetProgress η 4 2 32 16 ≤
      StochasticSharpnessSGD.sgdNetProgress η 4 2 32 64 := by
  exact StochasticSharpnessSGD.sgdNetProgress_mono_batchSize η 4 2 32
    (by decide) (by decide) (by norm_num) (by norm_num)

-- Smoke test 3: Multi-step PL convergence of mini-batch SGD to the variance noise horizon.
example (E_L : ℕ → ℝ) (T : ℕ)
    (hrec : ∀ t, E_L (t + 1) - 0 ≤
      (1 - (1 / 10 : ℝ) * 2) * (E_L t - 0) + ((1 / 10 : ℝ) ^ 2 * 4 / 2) * (16 / (32 : ℝ))) :
    E_L T - 0 ≤
      (1 - (1 / 10 : ℝ) * 2) ^ T * (E_L 0 - 0) +
        StochasticSharpnessSGD.sgdNoiseFloor (1 / 10) 4 2 16 32 := by
  exact StochasticSharpnessSGD.sgd_pl_noise_ball_convergence E_L 0 (1 / 10) 4 2 16 32
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by decide) hrec T
