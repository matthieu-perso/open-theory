import OpenTheory.Optimization.GlobalSaddlePL

open OpenTheory
open OpenTheory.GlobalSaddlePL

-- Section 1: Endogenous Polyak-Łojasiewicz Constant μ(W_t) from Minimum Gram Eigenvalue
#check GlobalSaddlePL.endogenousPLConstant
#check GlobalSaddlePL.endogenousPLFromGram
#check GlobalSaddlePL.endogenousContractionFactor
#check GlobalSaddlePL.endogenousPLConstant_eq_plConstant
#check GlobalSaddlePL.endogenousPLConstant_pos
#check GlobalSaddlePL.endogenousPLFromGram_pos
#check GlobalSaddlePL.endogenous_pl_inequality_from_gram_margin
#check GlobalSaddlePL.endogenous_pl_inequality_sample_normalized
#check GlobalSaddlePL.endogenous_pl_increases_with_feature_conditioning
#check GlobalSaddlePL.dual_feature_and_residual_pl_acceleration
#check GlobalSaddlePL.mlp_dual_feature_and_residual_pl_acceleration

-- Section 2: Strict Saddle Points & Negative-Curvature Escape under Stochastic Noise
#check GlobalSaddlePL.saddleUnstableVarStep
#check GlobalSaddlePL.saddleUnstableVarTrajectory
#check GlobalSaddlePL.saddleUnstableVarTrajectory_zero
#check GlobalSaddlePL.saddleUnstableVarTrajectory_succ
#check GlobalSaddlePL.saddleAmplificationFactor
#check GlobalSaddlePL.one_lt_saddleAmplificationFactor
#check GlobalSaddlePL.saddle_deterministic_trapped
#check GlobalSaddlePL.saddleUnstableVarTrajectory_closed_form
#check GlobalSaddlePL.geom_sub_one_ge_linear
#check GlobalSaddlePL.geom_series_ratio_lower_bounds
#check GlobalSaddlePL.saddle_noise_exponential_escape
#check GlobalSaddlePL.saddleQuadraticLoss
#check GlobalSaddlePL.saddle_escape_loss_drop

-- Section 3: Two-Phase Global Non-Convex Convergence Theorem
#check GlobalSaddlePL.two_phase_global_nonconvex_convergence
#check GlobalSaddlePL.two_phase_global_nonconvex_from_recurrence
#check GlobalSaddlePL.optimal_two_phase_noise_schedule

-- Smoke test 1: Endogenous PL inequality from Gram eigenvalue lam_min_G = 3 on residual R_sq = 8
-- (L = 4) with squared gradient norm G_sq = 25 ≥ 3 * 8 = 24.
example :
    2 * (3 : ℝ) * 4 ≤ 25 ∧
      endogenousPLFromGram 3 * 4 ≤ 25 ∧
      (0 : ℝ) < 3 := by
  exact endogenous_pl_inequality_from_gram_margin
    (R_sq := 8) (L := 4) (G_sq := 25) (lam_min_G := 3)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num)

-- Smoke test 2: Dual feature conditioning (λ₀ = 1 → λ_t = 2) and residual flattening
-- (S₀ = 10 → S_t = 5) strictly improves the PL contraction factor from 9/10 to 3/5.
example :
    0 ≤ (1 : ℝ) - 2 / 5 ∧
      (1 : ℝ) - 2 / 5 < 1 - 2 / 10 ∧
      (1 : ℝ) - 2 / 10 < 1 - 1 / 10 ∧
      (1 : ℝ) - 2 / 5 < 1 - 1 / 5 ∧
      (1 : ℝ) - 1 / 5 < 1 - 1 / 10 ∧
      (1 : ℝ) - 2 / 5 < 1 - 1 / 10 := by
  exact dual_feature_and_residual_pl_acceleration
    (by norm_num) (by norm_num) (by norm_num) (by norm_num)

-- Smoke test 3: Deterministic GD at an exact saddle (V₀ = 0, σ_-^2 = 0) stays trapped at 0,
-- whereas stochastic noise σ_-^2 = 1 with η = 1, γ_neg = 1 (so q = (1 + 1)^2 = 4) drives
-- V_2 = 1 * (4^2 - 1) / (4 - 1) = 5 > 0.
example :
    saddleUnstableVarTrajectory 1 1 0 0 10 = 0 ∧
      saddleUnstableVarTrajectory 1 1 1 0 2 = 5 := by
  refine ⟨saddle_deterministic_trapped 1 1 10, ?_⟩
  simp [saddleUnstableVarTrajectory, saddleUnstableVarStep]
  norm_num

-- Smoke test 4: Two-phase noise schedule: batch size B₁ = 1 < B₂ = 4 gives strictly faster
-- saddle escape in Phase I and strictly smaller noise floor in Phase II with momentum β = 1/2.
example :
    saddleUnstableVarTrajectory 1 1 (4 / ((4 : ℕ) : ℝ)) 0 2 <
        saddleUnstableVarTrajectory 1 1 (4 / ((1 : ℕ) : ℝ)) 0 2 ∧
      ((1 - (1 / 2 : ℝ)) / (1 + 1 / 2)) *
          StochasticSharpnessSGD.sgdNoiseFloor 1 2 1 4 4 <
        StochasticSharpnessSGD.sgdNoiseFloor 1 2 1 4 4 ∧
      StochasticSharpnessSGD.sgdNoiseFloor 1 2 1 4 4 <
        StochasticSharpnessSGD.sgdNoiseFloor 1 2 1 4 1 := by
  exact optimal_two_phase_noise_schedule
    (η := 1) (γ_neg := 1) (σ_minus_sq := 4) (S := 2) (μ := 1) (σ_sq := 4)
    (β := 1 / 2) (B₁ := 1) (B₂ := 4) (T := 2)
    (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num)
    (by decide) (by decide) (by decide)
