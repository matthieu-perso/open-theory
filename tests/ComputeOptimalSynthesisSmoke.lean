import OpenTheory.Generalization.ComputeOptimalSynthesis

open OpenTheory
open OpenTheory.ComputeOptimalSynthesis
open OpenTheory.StochasticSharpnessSGD
open OpenTheory.MomentumAdamTrajectory

-- Section 1: Exact Variational Chinchilla / Compute-Optimal Allocation via Completion of Squares
#check scalingRisk
#check optimalModelCapacity
#check optimalDataBudget
#check paretoComputeRisk
#check scalingEndToEndError
#check populationRiskBound_scalingEndToEndError
#check optimalAllocation_pos
#check optimalCapacity_mul_optimalData_eq_compute
#check optimalCapacity_mul_optimalData_eq_compute_of_nonneg
#check scalingRisk_at_optimal_eq_pareto
#check chinchilla_completion_of_squares
#check chinchilla_variational_global_optimality

-- Section 2: How Architecture and Optimizer Improvements Shift the Allocation & Risk Frontier
#check paretoComputeRisk_mono_in_arch_and_opt
#check paretoComputeRisk_strict_mono_in_arch_and_opt
#check optimal_capacity_to_data_ratio_shift
#check optimalAllocation_directional_shifts

-- Section 3: Mini-Batch Compute-vs-Wall-Clock Pareto Frontier & Critical Batch Size Optimality
#check stepProgress
#check flopEfficiency
#check stepProgress_eq_optimal_sgd
#check flopEfficiency_eq_div
#check stepProgress_strict_mono_in_batch
#check flopEfficiency_strict_anti_in_batch
#check balancedBatchObjective
#check balancedBatchObjective_at_critical
#check critical_batch_variational_optimality
#check critical_batch_global_maximizer
#check momentumCriticalBatchRatio
#check momentum_reduces_critical_batch_size

-- Section 4: Master Co-Design Dominance Theorem
#check equivalentComputeBudget
#check codesign_compute_saving_factor
#check master_codesign_compute_dominance

-- Smoke test 1: At A = 4, B_opt = 9, C = 36, the compute-optimal allocation is N* = 4, D* = 9
-- with N* * D* = 36 and N* / D* = 4 / 9, achieving paretoComputeRisk R_inf 4 9 36 = R_inf + 2.
example (R_inf : ℝ) :
    optimalModelCapacity 4 9 36 * optimalDataBudget 4 9 36 = 36 ∧
      optimalModelCapacity 4 9 36 / optimalDataBudget 4 9 36 = 4 / 9 ∧
      scalingRisk R_inf 4 9 (optimalModelCapacity 4 9 36) (optimalDataBudget 4 9 36) =
        paretoComputeRisk R_inf 4 9 36 := by
  have h1 := optimalCapacity_mul_optimalData_eq_compute (A := 4) (B_opt := 9) (C := 36)
    (by norm_num) (by norm_num) (by norm_num)
  have h2 := scalingRisk_at_optimal_eq_pareto R_inf (A := 4) (B_opt := 9) (C := 36)
    (by norm_num) (by norm_num) (by norm_num)
  exact ⟨h1.1, h1.2, h2⟩

-- Smoke test 2: Critical batch size B_crit = 16 globally maximizes balancedBatchObjective
-- and strictly beats B = 48.
example :
    balancedBatchObjective 8 2 16 48 < balancedBatchObjective 8 2 16 16 ∧
      balancedBatchObjective 8 2 16 16 = 1 / 2 := by
  have hmax := critical_batch_global_maximizer (G := 8) (S_loc := 2) (B_crit := 16) (B := 48)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  have hcrit := balancedBatchObjective_at_critical 8 2 16 (by norm_num)
  refine ⟨hmax.2 (by norm_num), ?_⟩
  rw [hcrit]
  norm_num

-- Smoke test 3: Master co-design dominance chain across 4 architectures and 3 optimizers.
example (R_inf : ℝ) :
    paretoComputeRisk R_inf (1 * 2) 1 100 < paretoComputeRisk R_inf (1 * 4) 2 100 ∧
      paretoComputeRisk R_inf (1 * 4) 2 100 < paretoComputeRisk R_inf (1 * 8) 4 100 ∧
      paretoComputeRisk R_inf (1 * 8) 4 100 < paretoComputeRisk R_inf (1 * 16) 4 100 ∧
      paretoComputeRisk R_inf (1 * 2) 1 (((2 / 16) * (1 / 4)) * 100) =
        paretoComputeRisk R_inf (1 * 16) 4 100 := by
  have hm := master_codesign_compute_dominance R_inf
    (c_arch := 1) (p_SESC := 2) (p_Subspace := 4) (p_CNN := 8) (p_MLP := 16)
    (B_RPM := 1) (B_Adam := 2) (B_SGD := 4) (C := 100) (C_base := 100)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  exact ⟨hm.1.1, hm.1.2.1, hm.1.2.2, hm.2.2.2.1⟩
