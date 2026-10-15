import OpenTheory.Generalization.PredictiveModelPrescription

noncomputable section

open OpenTheory
open OpenTheory.ComputeOptimalSynthesis
open OpenTheory.SpectralSparseCircuit
open OpenTheory.ResidualPreconditionedMomentum
open OpenTheory.PredictiveModelPrescription

-- Section 1: Formal Task Specification & Model Design Space
#check @TaskSignature
#check @ModelPrescription
#check @predictiveTaskRisk
#check @predictiveTaskRisk_nonneg
#check @predictiveTaskRisk_mono
#check @predictiveTaskRisk_strict_mono_in_each_factor

-- Section 2: Predictive Law 1 — Optimal Architecture Selection from Task Symmetries & Sample Budget
#check @prescribed_architecture_dominance_hierarchy
#check @criticalSampleThreshold
#check @adaptive_architecture_selection_by_sample_size

-- Section 3: Predictive Law 2 — Optimal Depth, Softmax vs. Linear Attention, and CoT Horizon Prescription
#check @compositionalReasoningRisk
#check @softmaxDistractorLeakage
#check @linearDistractorLeakage
#check @hierarchical_cot_softmax_strictly_dominates_shallow_linear

-- Section 4: The Master End-to-End Task-to-Model Synthesis Theorem
#check @endToEndComputeRisk
#check @prescribedParetoEndToEndRisk
#check @endToEndComputeRisk_at_optimal_eq_prescribed
#check @master_predictive_model_prescription

-- Smoke test 1: Exact calculation of the sample-size threshold n_crit where capacity should be
-- expanded from pruned SESC (s = 2, r = 3, dim = 6) to SubspaceNet (m = 8, r = 3, dim = 24)
-- at c_gen = 1 and pruning residual ε_prune = 1 / 10: n_crit = (24 - 6) * 1 / (1/10) = 180.
-- For n = 100 < 180 (data-scarce), SESC strictly beats SubspaceNet;
-- for n = 300 > 180 (data-abundant), SubspaceNet strictly beats SESC.
example :
    criticalSampleThreshold 2 8 3 1 (1 / 10) = 180 ∧
    predictiveTaskRisk (1 / 10) (dimSESC 2 3 : ℝ) 1 100 0 2 0 1 <
      predictiveTaskRisk 0 (dimSubspace 8 3 : ℝ) 1 100 0 2 0 1 ∧
    predictiveTaskRisk 0 (dimSubspace 8 3 : ℝ) 1 300 0 2 0 1 <
      predictiveTaskRisk (1 / 10) (dimSESC 2 3 : ℝ) 1 300 0 2 0 1 := by
  have hncrit : criticalSampleThreshold 2 8 3 1 (1 / 10) = 180 := by
    norm_num [criticalSampleThreshold, dimSubspace, dimSESC]
  have hadapt100 := adaptive_architecture_selection_by_sample_size
    (s := 2) (m := 8) (r := 3) (c_gen := 1) (ε_prune := 1 / 10)
    (n := 100) (R_opt := 0) (horizonAmp := 2) (Δ_OOD_sq := 1)
    (by decide) (by decide) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  have hadapt300 := adaptive_architecture_selection_by_sample_size
    (s := 2) (m := 8) (r := 3) (c_gen := 1) (ε_prune := 1 / 10)
    (n := 300) (R_opt := 0) (horizonAmp := 2) (Δ_OOD_sq := 1)
    (by decide) (by decide) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  refine ⟨hncrit, ?_, ?_⟩
  · apply hadapt100.2.1.mp
    rw [hncrit]
    norm_num
  · apply hadapt300.2.2.2.mp
    rw [hncrit]
    norm_num

-- Smoke test 2: Concrete 4-tier architecture dominance hierarchy on both in-distribution and
-- OOD risk for s = 2, m = 6, r = 3, k = 5, d = 16.
example :
    predictiveTaskRisk 0 (dimSESC 2 3 : ℝ) 1 100 0 2 0 5 <
      predictiveTaskRisk 0 (dimSubspace 6 3 : ℝ) 1 100 0 2 0 5 ∧
    predictiveTaskRisk 0 (dimSubspace 6 3 : ℝ) 1 100 0 2 0 5 <
      predictiveTaskRisk 0 (dimCNN 6 5 : ℝ) 1 100 0 2 1 5 ∧
    predictiveTaskRisk 0 (dimCNN 6 5 : ℝ) 1 100 0 2 1 5 <
      predictiveTaskRisk 0 (dimMLP 6 16 : ℝ) 1 100 0 2 3 5 := by
  have h := prescribed_architecture_dominance_hierarchy
    (s := 2) (m := 6) (r := 3) (k := 5) (d := 16)
    (c_gen := 1) (n := 100) (R_opt := 0) (horizonAmp := 2)
    (w_CNN := 1) (w_MLP := 3) (Δ_OOD_sq := 5)
    (by decide) (by decide) (by decide) (by decide) (by decide)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  exact h.2

-- Smoke test 3: Concrete end-to-end compute + OOD risk comparison and exact FLOP-saving factor
-- for s = 2, m = 8, r = 3, k = 6, d = 16 (so dimSESC = 6, dimMLP = 128) and B_RPM = 1, B_SGD = 4:
-- κ_save = (6 / 128) * (1 / 4) = 3 / 256 < 1.
example :
    let A_SESC : ℝ := 1 * (dimSESC 2 3 : ℝ)
    let A_MLP : ℝ := 1 * (dimMLP 8 16 : ℝ)
    let κ_save : ℝ := ((dimSESC 2 3 : ℝ) / (dimMLP 8 16 : ℝ)) * (1 / 4)
    κ_save = 3 / 256 ∧
    endToEndComputeRisk 0 A_SESC 1
        (optimalModelCapacity A_SESC 1 100) (optimalDataBudget A_SESC 1 100)
        (1 / (1 - 1 / 2)) 0 10 <
      endToEndComputeRisk 0 A_MLP 4 10 10 4 2 10 ∧
    prescribedParetoEndToEndRisk 0 A_SESC 1 (κ_save * 100) (1 / 2) 0 0 =
      prescribedParetoEndToEndRisk 0 A_MLP 4 100 (1 / 2) 2 0 := by
  intro A_SESC A_MLP κ_save
  have hκ : κ_save = 3 / 256 := by
    norm_num [κ_save, dimSESC, dimMLP]
  have hm := master_predictive_model_prescription 0
    (s := 2) (m := 8) (r := 3) (k := 6) (d := 16)
    (c_arch := 1) (B_RPM := 1) (B_SGD := 4) (C := 100) (N := 10) (D := 10)
    (Λ_CoT := 1 / 2) (amp_base := 4) (w_MLP := 2) (Δ_OOD_sq := 10)
    (by norm_num) (by norm_num)
    (by decide) (by decide) (by decide) (by decide) (by decide)
    (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  exact ⟨hκ, hm.1, hm.2.2.2.1⟩
