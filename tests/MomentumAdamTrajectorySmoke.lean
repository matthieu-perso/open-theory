import OpenTheory.Optimization.MomentumAdamTrajectory

open OpenTheory
open OpenTheory.MomentumAdamTrajectory
open OpenTheory.PreconditionedDescent
open OpenTheory.StochasticSharpnessSGD
open OpenTheory.TrajectoryGeneralization

-- Section 1: EMA First-Moment Trajectory & Bias Correction
#check emaStep
#check emaTrajectory
#check emaTrajectory_zero
#check emaTrajectory_succ
#check emaWeightSum
#check emaWeightSum_zero
#check emaWeightSum_succ
#check emaWeightSum_nonneg
#check emaWeightSum_pos
#check emaWeightSum_le_one
#check biasCorrectedEMA
#check biasCorrectedEMA_eq_inv_weightSum
#check emaTrajectory_const_grad
#check biasCorrectedEMA_const_grad
#check emaStep_norm_le
#check emaTrajectory_norm_le
#check biasCorrectedEMA_norm_le

-- Section 2: EMA Variance Damping & PL Noise Floor Reduction
#check emaVarianceStep
#check emaStationaryVariance
#check emaVarianceTrajectory
#check emaVarianceTrajectory_zero
#check emaVarianceTrajectory_succ
#check emaVariance_fixed_point
#check emaVarianceTrajectory_eq
#check emaVarianceTrajectory_le_stationary
#check ema_damping_factor_lt_one
#check ema_variance_strictly_smaller_than_sgd
#check momentum_pl_noise_floor_eq
#check momentum_pl_noise_floor_strict_improvement
#check momentum_pl_noise_ball_convergence

-- Section 3: AdamW Decoupled Weight Decay & Out-of-Sample Generalization
#check adamwStep
#check adamwTrajectory
#check adamwTrajectory_zero
#check adamwTrajectory_succ
#check adamwRadius
#check adamwRadius_nonneg
#check adamw_step_norm_le
#check adamw_step_preserves_ball
#check adamw_trajectory_norm_envelope
#check adamw_trajectory_in_ball
#check adamw_sign_direction_norm_le
#check adamw_mlp_generalization_tail

-- Section 4: Optimizer Regime Selection
#check optimizerNoisePenalty
#check optimizerNoisePenalty_nonneg
#check optimizerNoisePenalty_pos
#check optimizerNetProgress
#check optimizerNetProgress_eq_sub_penalty
#check optimizerNetProgress_zero_momentum
#check adam_momentum_reduces_noise_penalty
#check adam_momentum_dominates_preconditioned_no_momentum
#check adam_momentum_strictly_beats_preconditioned_no_momentum
#check adam_momentum_strictly_beats_scalar_sgd

-- Smoke test 1: Bias-corrected EMA with β = 9/10 on a constant gradient g recovers g at any T ≥ 1.
example {p : ℕ} (g : Parameters p) (T : ℕ) (hT : 1 ≤ T) :
    biasCorrectedEMA (9 / 10 : ℝ) (fun _ => g) T = g := by
  exact biasCorrectedEMA_const_grad g (by norm_num) (by norm_num) hT

-- Smoke test 2: At β = 9/10, EMA momentum shrinks the stationary gradient variance and PL noise
-- floor by the exact factor (1 - 0.9)/(1 + 0.9) = 1/19 < 1.
example :
    emaStationaryVariance (9 / 10 : ℝ) 38 2 = 1 ∧
      sgdNoiseFloor (1 / 10) 4 2 (((1 - (9 / 10 : ℝ)) / (1 + (9 / 10 : ℝ))) * 38) 2 <
        sgdNoiseFloor (1 / 10) 4 2 38 2 := by
  refine ⟨?_, ?_⟩
  · unfold emaStationaryVariance
    norm_num
  · exact momentum_pl_noise_floor_strict_improvement 2
      (by norm_num) (by norm_num) (by norm_num)
      (by norm_num) (by norm_num) (by norm_num) (by decide)

-- Smoke test 3: AdamW with η = 1/100, lam_w = 1/10, and ‖d_t‖ ≤ 5 keeps all iterates in the ball of
-- radius D_max / lam_w = 50 for every horizon T.
example {p : ℕ} (d_seq : ℕ → Parameters p) (hd : ∀ t, ‖d_seq t‖ ≤ 5) (T : ℕ) :
    ‖adamwTrajectory (1 / 100) (1 / 10) 0 d_seq T‖ ≤ 50 := by
  have hmem := adamw_trajectory_in_ball (η := 1 / 100) (lam_w := 1 / 10) (D_max := 5) d_seq
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) hd T
  rw [mem_trajectoryBall] at hmem
  unfold adamwRadius at hmem
  linarith
