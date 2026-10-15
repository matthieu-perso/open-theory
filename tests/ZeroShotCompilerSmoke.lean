import OpenTheory.Generalization.ZeroShotCompiler

noncomputable section

open OpenTheory
open OpenTheory.ComputeOptimalSynthesis
open OpenTheory.SpectralSparseCircuit
open OpenTheory.ResidualPreconditionedMomentum
open OpenTheory.PredictiveModelPrescription
open OpenTheory.ZeroShotCompiler

-- Section 1: Certified Zero-Shot Architecture Selection Under Empirical Estimation Noise
#check @certifiedScarceThreshold
#check @certifiedAbundantThreshold
#check @zero_shot_certified_threshold_brackets_oracle
#check @zero_shot_architecture_compiler_soundness
#check @zero_shot_plug_in_architecture_regret_le_two_delta

-- Section 2: Plug-In Chinchilla Compute Compiler & Quadratic Allocation Regret
#check @compiledCapacity
#check @compiledData
#check @compiledAllocation_pos
#check @compiledCapacity_mul_compiledData_eq_compute
#check @compiledAllocation_eq_optimal_of_exact_ratio
#check @compiled_chinchilla_exact_regret

-- Section 3: End-to-End Zero-Shot Compiler Specification & Verification
#check @ZeroShotCompiledSpec
#check @compileZeroShotSpec
#check @zero_shot_compiler_end_to_end_guarantee

-- Smoke test 1: Exact calculation of the certified scarce and abundant sample thresholds
-- [150, 225] bracketing the oracle threshold n_crit = 180 when s = 2, m = 8, r = 3, c_gen = 1,
-- ε_prune = 1 / 10, ε_hat = 1 / 10, and δ_ε = 1 / 50:
-- certifiedScarceThreshold = (24 - 6) / (1/10 + 1/50) = 18 / (6/50) = 150,
-- certifiedAbundantThreshold = (24 - 6) / (1/10 - 1/50) = 18 / (4/50) = 225.
example :
    certifiedScarceThreshold 2 8 3 1 (1 / 10) (1 / 50) = 150 ∧
    criticalSampleThreshold 2 8 3 1 (1 / 10) = 180 ∧
    certifiedAbundantThreshold 2 8 3 1 (1 / 10) (1 / 50) = 225 ∧
    certifiedScarceThreshold 2 8 3 1 (1 / 10) (1 / 50) ≤
      criticalSampleThreshold 2 8 3 1 (1 / 10) ∧
    criticalSampleThreshold 2 8 3 1 (1 / 10) ≤
      certifiedAbundantThreshold 2 8 3 1 (1 / 10) (1 / 50) := by
  have hbracket := zero_shot_certified_threshold_brackets_oracle
    (s := 2) (m := 8) (r := 3) (c_gen := 1)
    (ε_prune := 1 / 10) (ε_hat := 1 / 10) (δ_ε := 1 / 50)
    (by decide) (by decide) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num)
  refine ⟨?_, ?_, ?_, hbracket.2.1, hbracket.2.2⟩
  · norm_num [certifiedScarceThreshold, dimSubspace, dimSESC]
  · norm_num [criticalSampleThreshold, dimSubspace, dimSESC]
  · norm_num [certifiedAbundantThreshold, dimSubspace, dimSESC]

-- Smoke test 2: Certified architecture soundness outside [150, 225] and worst-case plug-in
-- regret bound ≤ 2 * horizonAmp * δ_ε = 2 * 2 * (1 / 50) = 2 / 25 inside the boundary zone.
example :
    predictiveTaskRisk (1 / 10) (dimSESC 2 3 : ℝ) 1 120 0 2 0 1 <
      predictiveTaskRisk 0 (dimSubspace 8 3 : ℝ) 1 120 0 2 0 1 ∧
    predictiveTaskRisk 0 (dimSubspace 8 3 : ℝ) 1 250 0 2 0 1 <
      predictiveTaskRisk (1 / 10) (dimSESC 2 3 : ℝ) 1 250 0 2 0 1 ∧
    predictiveTaskRisk (1 / 10) (dimSESC 2 3 : ℝ) 1 170 0 2 0 1 -
      predictiveTaskRisk 0 (dimSubspace 8 3 : ℝ) 1 170 0 2 0 1 ≤ 2 / 25 := by
  have hsound120 := zero_shot_architecture_compiler_soundness
    (s := 2) (m := 8) (r := 3) (c_gen := 1)
    (ε_prune := 1 / 10) (ε_hat := 1 / 10) (δ_ε := 1 / 50)
    (n := 120) (R_opt := 0) (horizonAmp := 2) (Δ_OOD_sq := 1)
    (by decide) (by decide) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  have hsound250 := zero_shot_architecture_compiler_soundness
    (s := 2) (m := 8) (r := 3) (c_gen := 1)
    (ε_prune := 1 / 10) (ε_hat := 1 / 10) (δ_ε := 1 / 50)
    (n := 250) (R_opt := 0) (horizonAmp := 2) (Δ_OOD_sq := 1)
    (by decide) (by decide) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  have hregret170 := zero_shot_plug_in_architecture_regret_le_two_delta
    (s := 2) (m := 8) (r := 3) (c_gen := 1)
    (ε_prune := 1 / 10) (ε_hat := 1 / 10) (δ_ε := 1 / 50)
    (n := 170) (R_opt := 0) (horizonAmp := 2) (Δ_OOD_sq := 1)
    (by decide) (by decide) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  refine ⟨?_, ?_, ?_⟩
  · apply hsound120.1
    norm_num [certifiedScarceThreshold, dimSubspace, dimSESC]
  · apply hsound250.2
    norm_num [certifiedAbundantThreshold, dimSubspace, dimSESC]
  · have h170_le : (170 : ℝ) ≤ criticalSampleThreshold 2 8 3 1 (1 / 10) := by
      norm_num [criticalSampleThreshold, dimSubspace, dimSESC]
    have hbound := hregret170.1 h170_le
    linarith

-- Smoke test 3: Compiled Chinchilla allocation budget feasibility, ratio consistency, and
-- end-to-end FLOP saving factor κ_flop_save = 3 / 256 for ρ_hat = 4 / 9, C = 36.
example (R_inf : ℝ) :
    compiledCapacity (4 / 9) 36 * compiledData (4 / 9) 36 = 36 ∧
    compiledCapacity (4 / 9) 36 / compiledData (4 / 9) 36 = 4 / 9 ∧
    0 ≤ scalingRisk R_inf 4 9 (compiledCapacity (4 / 9) 36) (compiledData (4 / 9) 36) -
      paretoComputeRisk R_inf 4 9 36 ∧
    (compileZeroShotSpec 2 8 3 16 1 (1 / 10) (1 / 50) (4 / 9) 1 4 120 36).κ_flop_save = 3 / 256 := by
  have hcomp := compiledCapacity_mul_compiledData_eq_compute (ρ_hat := 4 / 9) (C := 36)
    (by norm_num) (by norm_num)
  have hreg := compiled_chinchilla_exact_regret R_inf (ρ_hat := 4 / 9) (A := 4) (B_opt := 9) (C := 36)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  refine ⟨hcomp.1, hcomp.2, hreg.2.1, ?_⟩
  norm_num [compileZeroShotSpec, dimSESC, dimMLP]
