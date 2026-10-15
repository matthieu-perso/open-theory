import OpenTheory.Optimization.CertiRPM

open OpenTheory
open OpenTheory.CertiRPM
open OpenTheory.ResidualPreconditionedMomentum
open OpenTheory.GlobalSaddlePL
open OpenTheory.StochasticSharpnessSGD

-- Section 1: Bridge 1A — Plug-In Estimator Quadratic Regret (Zero First-Order Sensitivity)
#check CertiRPM.plugin_step_quadratic_regret_identity
#check CertiRPM.plugin_step_uniform_error_regret_bound
#check CertiRPM.pluginSharpnessStep
#check CertiRPM.pluginSharpnessStep_eq_rpmOptimalStep
#check CertiRPM.pluginSharpnessStep_sub_oracle_eq
#check CertiRPM.pluginSharpness_exact_progress_ratio

-- Section 2: Bridge 1B — Upper-Confidence Sharpness Estimator Guarantees Safe Descent
#check CertiRPM.upperConfidenceSharpness
#check CertiRPM.upper_confidence_sharpness_interval
#check CertiRPM.upper_confidence_multiplier_bounds
#check CertiRPM.upper_confidence_sharpness_guarantees_positive_progress
#check CertiRPM.upper_confidence_progress_lower_bound

-- Section 3: System 1 — Zero-Hyperparameter Self-Tuning Optimizer (`CertiRPM`)
#check CertiRPM.certiRPMBatchSize
#check CertiRPM.certiRPMStep
#check CertiRPM.certiRPMBatchSize_saddle
#check CertiRPM.certiRPMBatchSize_basin
#check CertiRPM.certiRPMStep_saddle
#check CertiRPM.certiRPMStep_basin
#check CertiRPM.pluginSharpnessStep_strict_accel_from_sharpness_drop
#check CertiRPM.certiRPM_master_self_tuning_guarantee

-- Smoke test 1: A 10% upper-confidence sharpness estimation error (S_hat = 1.1 * S, e.g. S = 10,
-- S_hat = 11) incurs a multiplicative progress loss of ((11 - 10) / 11)^2 = 1 / 121 < 1 / 100
-- (less than 1% progress loss!) and retains 120 / 121 > 99% of oracle maximum progress.
example :
    ((11 : ℝ) - 10) ^ 2 / 11 ^ 2 = 1 / 121 ∧
      (1 : ℝ) / 121 < 1 / 100 ∧
      (2 * (11 : ℝ) * 10 - 10 ^ 2) / 11 ^ 2 = 120 / 121 ∧
      (10 : ℝ) / 11 ≤ (2 * 11 * 10 - 10 ^ 2) / 11 ^ 2 ∧
      (2 * (11 : ℝ) * 10 - 10 ^ 2) / 11 ^ 2 ≤ 1 := by
  have hbounds := upper_confidence_multiplier_bounds (s := 10) (s_hat := 11)
    (by norm_num) (by norm_num)
  refine ⟨by norm_num, by norm_num, by norm_num, hbounds.2.2.1, hbounds.2.2.2⟩

-- Smoke test 2: Underestimating sharpness by more than 2x (S = 10, S_hat = 4 < S / 2) causes
-- negative progress multiplier (2 * 4 * 10 - 10^2) / 4^2 = -20 / 16 = -5 / 4 < 0 (divergence!),
-- whereas the upper-confidence bound S_hat = 12 ≥ S = 10 guarantees positive multiplier 35 / 36 > 0.
example :
    (2 * (4 : ℝ) * 10 - 10 ^ 2) / 4 ^ 2 = -5 / 4 ∧
      (2 * (4 : ℝ) * 10 - 10 ^ 2) / 4 ^ 2 < 0 ∧
      (2 * (12 : ℝ) * 10 - 10 ^ 2) / 12 ^ 2 = 35 / 36 ∧
      0 < (2 * (12 : ℝ) * 10 - 10 ^ 2) / 12 ^ 2 := by
  norm_num

-- Smoke test 3: CertiRPM automatic mode switching between strict saddle (κ_hat = -2 < 0, selecting
-- small batch B_escape = 8 and escape step η_escape = 1/10) and PL basin (κ_hat = 3 ≥ 0, selecting
-- large batch B_basin = 64 and plug-in RPM step).
example (g : Parameters 2) (i : Fin 2) :
    certiRPMBatchSize (-2 : ℝ) 8 64 = 8 ∧
      certiRPMBatchSize (3 : ℝ) 8 64 = 64 ∧
      certiRPMStep (-2 : ℝ) (1 / 10) (fun _ => 5) g (9 / 10) (fun _ => 1) 64 i = 1 / 10 ∧
      certiRPMStep (3 : ℝ) (1 / 10) (fun _ => 5) g (9 / 10) (fun _ => 1) 64 i =
        pluginSharpnessStep (fun _ => 5) g (9 / 10) (fun _ => 1) 64 i := by
  refine ⟨by norm_num [certiRPMBatchSize], by norm_num [certiRPMBatchSize], ?_, ?_⟩
  · exact certiRPMStep_saddle (1 / 10) (fun _ => 5) g (9 / 10) (fun _ => 1) 64 i (by norm_num)
  · exact certiRPMStep_basin (1 / 10) (fun _ => 5) g (9 / 10) (fun _ => 1) 64 i (by norm_num)

-- Smoke test 4: Concrete evaluation of `pluginSharpnessStep` acceleration when empirical residual
-- drop reduces upper-confidence sharpness from S_hat_0 = 8 to S_hat_t = 4 at active coordinate g i = 2.
example (g : Parameters 1) (i : Fin 1) (hgi : g i = 2) :
    pluginSharpnessStep (fun _ => 8) g (1 / 2) (fun _ => 0) 16 i = 1 / 8 ∧
      pluginSharpnessStep (fun _ => 4) g (1 / 2) (fun _ => 0) 16 i = 1 / 4 ∧
      pluginSharpnessStep (fun _ => 8) g (1 / 2) (fun _ => 0) 16 i <
        pluginSharpnessStep (fun _ => 4) g (1 / 2) (fun _ => 0) 16 i := by
  unfold pluginSharpnessStep totalCoordMoment dampedCoordVar
  rw [hgi]
  norm_num
