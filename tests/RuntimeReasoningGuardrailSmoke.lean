import OpenTheory.Architectures.RuntimeReasoningGuardrail

noncomputable section

open TorchLean MeasureTheory StatLean.StatisticalLearning
open OpenTheory
open OpenTheory.ConvolutionalGeneralization
open OpenTheory.SpectralSparseCircuit
open OpenTheory.AutoregressiveOOD
open OpenTheory.Attention
open OpenTheory.AttentionAblation
open OpenTheory.TransformerGeneralization
open OpenTheory.RuntimeReasoningGuardrail
open scoped BigOperators

variable {E F : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
  [NormedAddCommGroup F] [NormedSpace ℝ F]

-- Section 1: Runtime Contractive Jacobian Monitor & Dynamic Rollout Damping Guardrail
#check @dampedSensitivity
#check @requiredAnchorStrength
#check @dampedSensitivity_zero
#check @dampedSensitivity_one
#check @dampedSensitivity_nonneg
#check @dampedSensitivity_anti_in_anchor
#check @dampedSensitivity_at_requiredAnchorStrength
#check @requiredAnchorStrength_mem_unit_interval
#check @dampedSensitivity_le_target_iff
#check @autoregressiveRolloutError_mono_in_sensitivity
#check @runtime_damped_rollout_horizon_free_certificate

-- Section 2: Certified Upper-Confidence Runtime Contraction Under Noisy Jacobian Estimates
#check @upperConfidenceRolloutBound
#check @true_sensitivity_le_upper_confidence
#check @upperConfidenceRolloutBound_nonneg
#check @upper_confidence_runtime_rollout_soundness
#check @upperConfidenceRolloutBound_zero_noise

-- Section 3: Simultaneous Triple-Shield Runtime Guardrail
#check @multiHeadAttentionSubset
#check @multiHeadAttentionSubset_eq_ablatedMultiHeadOutput
#check @ablationExcessRiskShift
#check @ablationExcessRiskShift_nonneg
#check @master_runtime_triple_shield_certificate

-- Smoke test 1: Stabilizing an unstable Λ_raw = 2 rollout down to target contraction Λ_max = 1/2
-- requires anchor strength α_anchor = 3/4, restoring a horizon-free error bound of 2 * ε_token
-- for all H : ℕ (whereas the unguarded Λ_raw = 2 rollout blows up to 15 * ε_token at H = 4).
example (H : ℕ) :
    requiredAnchorStrength 2 (1 / 2) = 3 / 4 ∧
      dampedSensitivity (3 / 4) 2 = 1 / 2 ∧
      autoregressiveRolloutError (dampedSensitivity (3 / 4) 2) (1 / 10) H ≤ 1 / 5 ∧
      autoregressiveRolloutError 2 (1 / 10) 4 = 3 / 2 := by
  have hcert := (runtime_damped_rollout_horizon_free_certificate
    (α_anchor := 3 / 4) (Λ_raw := 2) (Λ_max := 1 / 2) (ε_token := 1 / 10)
    (by norm_num) (by norm_num) (by norm_num)
    (by norm_num [requiredAnchorStrength]) (by norm_num) (by norm_num) H).2.2.2.2
  refine ⟨?_, ?_, ?_, ?_⟩
  · norm_num [requiredAnchorStrength]
  · norm_num [dampedSensitivity]
  · norm_num at hcert
    exact hcert
  · norm_num [autoregressiveRolloutError, rolloutErrorStep]

-- Smoke test 2: Certified upper-confidence runtime rollout bound under noisy Jacobian estimation
-- (true Λ = 3/5, estimated Λ_hat = 1/2, estimation tolerance δ_Λ = 1/5, ε_token = 3/10).
-- The true infinite-horizon bound is 3/4, and the observable upper-confidence certificate is 1.
example (H : ℕ) :
    upperConfidenceRolloutBound (1 / 2) (1 / 5) (3 / 10) = 1 ∧
      autoregressiveRolloutError (3 / 5) (3 / 10) H ≤ 3 / 4 ∧
      autoregressiveRolloutError (3 / 5) (3 / 10) H ≤
        upperConfidenceRolloutBound (1 / 2) (1 / 5) (3 / 10) := by
  have hsound := upper_confidence_runtime_rollout_soundness
    (Λ := 3 / 5) (Λ_hat := 1 / 2) (δ_Λ := 1 / 5) (ε_token := 3 / 10)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) H
  refine ⟨?_, ?_, hsound.2.2.2.2⟩
  · norm_num [upperConfidenceRolloutBound]
  · have htrue := hsound.2.2.1
    norm_num at htrue
    exact htrue

-- Smoke test 3: Exact anchor calibration equivalence and monotonicity across anchor weights
-- for an expansive step Λ_raw = 5/2 and target contraction Λ_max = 3/4 (requiring α_anchor ≥ 7/10).
example :
    requiredAnchorStrength (5 / 2) (3 / 4) = 7 / 10 ∧
      dampedSensitivity (4 / 5) (5 / 2) = 1 / 2 ∧
      dampedSensitivity (4 / 5) (5 / 2) ≤ 3 / 4 ∧
      upperConfidenceRolloutBound (1 / 2) (1 / 4) 1 = 4 := by
  norm_num [requiredAnchorStrength, dampedSensitivity, upperConfidenceRolloutBound]

-- Axiom audits (ensuring zero sorryAx)
#print axioms OpenTheory.RuntimeReasoningGuardrail.dampedSensitivity_le_target_iff
#print axioms OpenTheory.RuntimeReasoningGuardrail.runtime_damped_rollout_horizon_free_certificate
#print axioms OpenTheory.RuntimeReasoningGuardrail.upper_confidence_runtime_rollout_soundness
#print axioms OpenTheory.RuntimeReasoningGuardrail.master_runtime_triple_shield_certificate
