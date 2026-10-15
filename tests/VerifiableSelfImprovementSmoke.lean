import OpenTheory.Optimization.VerifiableSelfImprovement

open OpenTheory
open OpenTheory.VerifiableSelfImprovement
open OpenTheory.CertiRPM
open OpenTheory.ComputeOptimalSynthesis
open OpenTheory.ResidualPreconditionedMomentum

-- Section 1: Symbolic Safety & Pareto-Dominance Filter for Candidate Self-Modifications
#check VerifiableSelfImprovement.IsVerifiablySafeImprovement
#check VerifiableSelfImprovement.verifiably_safe_improvement_guarantees_descent_and_pareto_drop

-- Section 2: Multi-Generation Recursive Self-Improvement Law
#check VerifiableSelfImprovement.generationComplexityProduct
#check VerifiableSelfImprovement.generationParetoRisk
#check VerifiableSelfImprovement.generationEquivalentCompute
#check VerifiableSelfImprovement.generationComplexityProduct_zero
#check VerifiableSelfImprovement.generationParetoRisk_zero
#check VerifiableSelfImprovement.generationParetoRisk_strict_anti_in_generation
#check VerifiableSelfImprovement.recursive_self_improvement_exponential_compute_compression

-- Section 3: Symbolic Verification vs. Empirical Grid-Search FLOP Savings
#check VerifiableSelfImprovement.empiricalSearchCompute
#check VerifiableSelfImprovement.symbolicSearchCompute
#check VerifiableSelfImprovement.symbolic_vs_empirical_search_flop_savings

-- Smoke test 1: 4 generations of γ_imp = 1/4 verified recursive self-improvement compress
-- the complexity product and required training compute by (1/4)^4 = 1/256x, so a generation-4
-- model trained with C = 1 FLOP achieves the exact same Pareto risk as a generation-0 baseline
-- trained with C₀ = 256 FLOPs.
example (R_inf : ℝ) :
    generationComplexityProduct 256 (1 / 4) 4 = 1 ∧
      generationEquivalentCompute 256 (1 / 4) 4 = 1 ∧
      (1 / 4 : ℝ) ^ 4 = 1 / 256 ∧
      paretoComputeRisk R_inf ((1 / 4 : ℝ) ^ 4 * 16) 16
          (generationEquivalentCompute 256 (1 / 4) 4) =
        paretoComputeRisk R_inf 16 16 256 := by
  have hcomp := recursive_self_improvement_exponential_compute_compression R_inf
    (A₀ := 16) (B₀ := 16) (γ_imp := 1 / 4) (C₀ := 256) 4
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  refine ⟨?_, ?_, by norm_num, hcomp.1⟩
  · unfold generationComplexityProduct
    norm_num
  · unfold generationEquivalentCompute
    norm_num

-- Smoke test 2: Symbolic verification vs. empirical brute-force grid search for M_cand = 100
-- candidate architectures/optimizers at C_train = 1000 GPU units and C_verify = 1 symbolic unit:
-- empirical search costs 100,000 units, symbolic search costs 1,100 units (saving 98,900 units),
-- and at C_verify = 0 symbolic filtering cuts search compute by 100x per generation and
-- 100^3 = 1,000,000x across 3 generations.
example :
    empiricalSearchCompute 100 1000 = 100000 ∧
      symbolicSearchCompute 100 1 1000 = 1100 ∧
      empiricalSearchCompute 100 1000 - symbolicSearchCompute 100 1 1000 = 98900 ∧
      symbolicSearchCompute 100 1 1000 < empiricalSearchCompute 100 1000 ∧
      symbolicSearchCompute 100 0 1000 / empiricalSearchCompute 100 1000 = 1 / 100 ∧
      (symbolicSearchCompute 100 0 1000 / empiricalSearchCompute 100 1000) ^ 3 =
        1 / 1000000 := by
  unfold empiricalSearchCompute symbolicSearchCompute
  norm_num

-- Smoke test 3: Concrete verification of `IsVerifiablySafeImprovement` and strict Pareto risk
-- reduction when a candidate self-modification halves both architectural complexity (A: 8 → 4)
-- and optimizer complexity (B: 8 → 4), yielding γ_imp = 1/4 contraction and cutting reducible
-- Pareto risk in half.
example (R_inf : ℝ) :
    IsVerifiablySafeImprovement (fun _ : Fin 2 => (10 : ℝ)) (fun _ : Fin 2 => (12 : ℝ))
      8 8 4 4 (1 / 4) ∧
    paretoComputeRisk R_inf 4 4 16 < paretoComputeRisk R_inf 8 8 16 ∧
    generationParetoRisk R_inf 64 (1 / 4) 16 1 <
      generationParetoRisk R_inf 64 (1 / 4) 16 0 := by
  refine ⟨⟨fun _ => by norm_num, by norm_num⟩, ?_, ?_⟩
  · exact paretoComputeRisk_strict_mono_in_arch_and_opt R_inf
      (by norm_num) (by norm_num) (by norm_num) (by norm_num)
      (Or.inl (by norm_num)) (by norm_num)
  · exact generationParetoRisk_strict_anti_in_generation R_inf
      (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by decide)
