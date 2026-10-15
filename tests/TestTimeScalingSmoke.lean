import OpenTheory.Generalization.TestTimeScaling

noncomputable section

open OpenTheory
open OpenTheory.ComputeOptimalSynthesis
open OpenTheory.AutoregressiveOOD
open OpenTheory.TestTimeScaling

-- Section 1: Best-of-K Verifier / Parallel Reasoning Pass@K Failure Exponential Decay
#check @bestOfKFailureProb
#check @bestOfKSuccessProb
#check @bestOfKFailureProb_Add_bestOfKSuccessProb
#check @bestOfKFailureProb_zero
#check @bestOfKFailureProb_one
#check @bestOfKFailureProb_nonneg
#check @bestOfKFailureProb_le_one
#check @bestOfKSuccessProb_bounds
#check @bestOfKFailureProb_strict_anti_in_K
#check @bestOfKSuccessProb_strict_mono_in_K
#check @bestOfK_exponential_bound
#check @bestOfK_sample_sufficiency

-- Section 2: Variational Tradeoff Between Model Size N and Test-Time Reasoning Horizon T_inf
#check @inferenceScalingRisk
#check @optimalInferenceModelSize
#check @optimalReasoningBudget
#check @paretoInferenceRisk
#check @inferenceScalingRisk_eq_scalingRisk
#check @optimalInferenceModelSize_mul_optimalReasoningBudget
#check @optimalInferenceModelSize_div_optimalReasoningBudget
#check @inferenceScalingRisk_at_optimal_eq_pareto
#check @test_time_scaling_completion_of_squares
#check @test_time_scaling_variational_optimality

-- Section 3: When a Smaller Reasoning Model Strictly Beats an M× Larger 1-Step Model
#check @giant_one_step_minus_small_reasoning_risk_gap
#check @small_reasoning_model_beats_giant_one_step_at_equal_inference_flops
#check @cot_best_of_K_joint_error_bound

-- Section 4: Total Lifecycle Compute Synthesis
#check @totalLifecycleCompute
#check @totalLifecycleCompute_factored
#check @lifecycleFeasibleModelSize
#check @lifecycleTokenToParamRatio
#check @totalLifecycleCompute_at_feasibleModelSize
#check @lifecycleTokenToParamRatio_eq
#check @deployment_queries_shift_optimal_model_size_down

-- Smoke test 1: A 1× small reasoning model (N_small = 1, T_inf = 10) strictly beats a 10× larger
-- 1-step model (N_giant = 10, T_inf = 1) at equal inference FLOPs (C_inf = 10) when α_model = 1
-- and β_reason = 5: the small reasoning model achieves risk 3/2 = 1.5 while the 10× giant 1-step
-- model suffers risk 51/10 = 5.1.
example :
    (1 : ℝ) * 10 = (10 * 1) * 1 ∧
      inferenceScalingRisk 0 1 5 1 10 = 3 / 2 ∧
      inferenceScalingRisk 0 1 5 (10 * 1) 1 = 51 / 10 ∧
      inferenceScalingRisk 0 1 5 1 10 < inferenceScalingRisk 0 1 5 (10 * 1) 1 := by
  have hbeat := (small_reasoning_model_beats_giant_one_step_at_equal_inference_flops
    0 1 5 (N_small := 1) (M := 10) (by norm_num) (by norm_num)).2.mpr (by norm_num)
  refine ⟨by norm_num, ?_, ?_, hbeat⟩
  · norm_num [inferenceScalingRisk]
  · norm_num [inferenceScalingRisk]

-- Smoke test 2: Best-of-K verifier exponential decay and CoT + Best-of-K joint error bound.
-- With single-trajectory failure p_fail = 1/2, K = 4 parallel reasoning rollouts reduce failure
-- probability from 1/2 to 1/16 and raise Pass@4 success rate to 15/16.
example :
    bestOfKFailureProb (1 / 2) 4 < bestOfKFailureProb (1 / 2) 1 ∧
      bestOfKFailureProb (1 / 2) 4 = 1 / 16 ∧
      bestOfKSuccessProb (1 / 2) 4 = 15 / 16 ∧
      bestOfKFailureProb (autoregressiveRolloutError (1 / 2) (1 / 4) 5) 4 ≤ 1 / 16 := by
  have hanti := (bestOfKFailureProb_strict_anti_in_K
    (p_fail := 1 / 2) (K₁ := 1) (K₂ := 4) (by norm_num) (by norm_num) (by decide)).1
  have hjoint := (cot_best_of_K_joint_error_bound
    (Λ := 1 / 2) (ε_token := 1 / 4) (by norm_num) (by norm_num) (by norm_num) (by norm_num) 5 4).1
  refine ⟨hanti, ?_, ?_, ?_⟩
  · norm_num [bestOfKFailureProb]
  · norm_num [bestOfKSuccessProb]
  · exact hjoint.trans (by norm_num)

-- Smoke test 3: Variational test-time scaling identity and lifecycle compute shift.
-- At α_model = 4, β_reason = 9, C_inf = 36, the optimal allocation is N* = 4, T_inf* = 9.
-- Under total lifecycle compute C_total = 1000 and pre-training tokens D = 100, increasing
-- deployment queries from Q₁ = 10 to Q₂ = 90 (at T_inf = 10) reduces feasible model size from
-- N₁ = 5 to N₂ = 1 and increases the token-to-parameter ratio from 20 to 100.
example (R_inf : ℝ) :
    optimalInferenceModelSize 4 9 36 * optimalReasoningBudget 4 9 36 = 36 ∧
      optimalInferenceModelSize 4 9 36 / optimalReasoningBudget 4 9 36 = 4 / 9 ∧
      inferenceScalingRisk R_inf 4 9
        (optimalInferenceModelSize 4 9 36) (optimalReasoningBudget 4 9 36) =
        paretoInferenceRisk R_inf 4 9 36 ∧
      lifecycleFeasibleModelSize 1000 100 90 10 = 1 ∧
      lifecycleFeasibleModelSize 1000 100 10 10 = 5 ∧
      lifecycleFeasibleModelSize 1000 100 90 10 < lifecycleFeasibleModelSize 1000 100 10 10 ∧
      lifecycleTokenToParamRatio 1000 100 10 10 < lifecycleTokenToParamRatio 1000 100 90 10 := by
  have hmul := optimalInferenceModelSize_mul_optimalReasoningBudget
    (α_model := 4) (β_reason := 9) (C_inf := 36) (by norm_num) (by norm_num) (by norm_num)
  have hdiv := optimalInferenceModelSize_div_optimalReasoningBudget
    (α_model := 4) (β_reason := 9) (C_inf := 36) (by norm_num) (by norm_num) (by norm_num)
  have hpar := inferenceScalingRisk_at_optimal_eq_pareto R_inf
    (α_model := 4) (β_reason := 9) (C_inf := 36) (by norm_num) (by norm_num) (by norm_num)
  have hshift := (deployment_queries_shift_optimal_model_size_down
    (C_total := 1000) (D := 100) (Q₁ := 10) (Q₂ := 90) (T_inf := 10)
    (Q := 10) (T₁ := 1) (T₂ := 10)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num)).1
  refine ⟨hmul, hdiv, hpar, ?_, ?_, hshift.1, hshift.2⟩
  · norm_num [lifecycleFeasibleModelSize]
  · norm_num [lifecycleFeasibleModelSize]
