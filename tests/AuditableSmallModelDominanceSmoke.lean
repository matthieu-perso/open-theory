import OpenTheory.Generalization.AuditableSmallModelDominance

noncomputable section

open OpenTheory
open OpenTheory.ComputeOptimalSynthesis
open OpenTheory.SpectralSparseCircuit
open OpenTheory.TestTimeScaling
open OpenTheory.PredictiveModelPrescription
open OpenTheory.AuditableSmallModelDominance

-- Section 1: Mechanistic Auditability Complexity & Compression Ratio
#check @auditComplexitySESC
#check @auditComplexityGiant
#check @auditCompressionRatio
#check @auditComplexitySESC_eq
#check @auditComplexityGiant_eq
#check @auditCompressionRatio_pos_and_lt_one

-- Section 2: Small Auditable Reasoning Model vs. M× Monolithic Giant at Equal or Lower Inference FLOPs
#check @giantOneStepInferenceFLOPs
#check @smallAuditableInferenceFLOPs
#check @small_auditable_uses_le_inference_flops
#check @sesc_capacity_constant_lt_giant
#check @fullDeploymentRisk
#check @giant_minus_small_auditable_deployment_risk_gap
#check @auditable_small_model_strictly_beats_monolithic_giant

-- Section 3: The Special Case of Architecture-Matched Compression (M = (m * d) / (s * r))
#check @symmetry_matched_small_model_unconditional_dominance

-- Section 4: Master 4-Way Dominance Theorem
#check @master_auditable_small_vs_monolithic_giant_theorem

-- Smoke test 1: A 100× smaller SESC circuit (s = 2, r = 3 ⟹ 6 active circuit parameters) vs. a
-- 600-parameter dense MLP (m = 20, d = 30 ⟹ 600 parameters, M = 100):
-- Exact audit complexity compression ratio = 6 / 600 = 1 / 100 < 1, and with B_RPM = 1, B_SGD = 4,
-- pre-training compute compression factor κ_save = 1 / 400.
example :
    auditComplexitySESC 2 3 = 6 ∧
      auditComplexityGiant 20 30 = 600 ∧
      auditComplexitySESC 2 3 < auditComplexityGiant 20 30 ∧
      auditCompressionRatio 2 3 20 30 = 1 / 100 ∧
      0 < auditCompressionRatio 2 3 20 30 ∧
      auditCompressionRatio 2 3 20 30 < 1 := by
  have h := auditCompressionRatio_pos_and_lt_one
    (s := 2) (r := 3) (m := 20) (d := 30)
    (by decide) (by decide) (by decide) (by decide)
  refine ⟨by decide, by decide, h.2.2, ?_, h.1, h.2.1⟩
  norm_num [auditCompressionRatio, dimSESC, dimMLP]

-- Smoke test 2: Inference FLOPs and Deployment Risk for the 100× smaller auditable SESC model
-- (N_small = 6, α_small = 6, T_reason = 10 ≤ M = 100) vs. the 100× larger monolithic 1-step giant
-- (N_giant = 100 * 6 = 600, α_giant = 600, T_inf = 1):
-- The small model uses 6 * 10 = 60 inference FLOPs vs. 600 * 1 = 600 inference FLOPs (10× fewer
-- inference FLOPs!) while strictly reducing deployment risk from 8 to 3/2 = 1.5.
example :
    smallAuditableInferenceFLOPs 6 10 = 60 ∧
      giantOneStepInferenceFLOPs 100 6 = 600 ∧
      smallAuditableInferenceFLOPs 6 10 < giantOneStepInferenceFLOPs 100 6 ∧
      fullDeploymentRisk 0 6 5 6 10 0 2 = 3 / 2 ∧
      fullDeploymentRisk 0 (100 * 6) 5 (100 * 6) 1 1 2 = 8 ∧
      fullDeploymentRisk 0 6 5 6 10 0 2 < fullDeploymentRisk 0 (100 * 6) 5 (100 * 6) 1 1 2 := by
  have hdom := symmetry_matched_small_model_unconditional_dominance
    0 6 5 6 100 10 1 2
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  have hflops_lt := (small_auditable_uses_le_inference_flops
    (M := 100) (N_small := 6) (T_reason := 10) (by norm_num) (by norm_num)).2.1 (by norm_num)
  refine ⟨?_, ?_, hflops_lt, ?_, ?_, hdom.2.2.2.2⟩
  · norm_num [smallAuditableInferenceFLOPs]
  · norm_num [giantOneStepInferenceFLOPs]
  · norm_num [fullDeploymentRisk, inferenceScalingRisk]
  · norm_num [fullDeploymentRisk, inferenceScalingRisk]

-- Smoke test 3: Master 4-way dominance theorem instantiated on the 100× smaller SESC circuit
-- (s = 2, r = 3, m = 20, d = 30 ⟹ M = 100) with N_small = 6, T_reason = 10 ≤ 100,
-- B_RPM = 1, B_SGD = 4, β_reason = 5, w_giant = 1, Δ_OOD_sq = 2.
example :
    let κ_save : ℝ := auditCompressionRatio 2 3 20 30 * (1 / 4)
    κ_save = 1 / 400 ∧
      auditComplexitySESC 2 3 < auditComplexityGiant 20 30 ∧
      smallAuditableInferenceFLOPs 6 10 <
        giantOneStepInferenceFLOPs ((dimMLP 20 30 : ℝ) / (dimSESC 2 3 : ℝ)) 6 ∧
      fullDeploymentRisk 0 (1 * (dimSESC 2 3 : ℝ)) 5 6 10 0 2 <
        fullDeploymentRisk 0 (1 * (dimMLP 20 30 : ℝ)) 5
          (((dimMLP 20 30 : ℝ) / (dimSESC 2 3 : ℝ)) * 6) 1 1 2 := by
  intro κ_save
  have hm := master_auditable_small_vs_monolithic_giant_theorem 0
    (s := 2) (r := 3) (m := 20) (d := 30)
    (c_arch := 1) (B_RPM := 1) (B_SGD := 4) (C := 1000)
    (N_small := 6) (T_reason := 10) (β_reason := 5) (w_giant := 1) (Δ_OOD_sq := 2)
    (by norm_num)
    (by decide) (by decide) (by decide) (by decide)
    (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num)
    (by norm_num [dimMLP, dimSESC])
    (by norm_num) (by norm_num)
  have hκ : κ_save = 1 / 400 := by
    norm_num [κ_save, auditCompressionRatio, dimSESC, dimMLP]
  have hT_lt_M : (10 : ℝ) < (dimMLP 20 30 : ℝ) / (dimSESC 2 3 : ℝ) := by
    norm_num [dimMLP, dimSESC]
  exact ⟨hκ, hm.1.1, hm.2.2.1.2 hT_lt_M, hm.2.2.2.2⟩
