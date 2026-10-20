import OpenTheory.Certification.AgentZeroKnowledgePassport

open OpenTheory.AgentZeroKnowledgePassport
open OpenTheory.ZeroKnowledgeCommitment
open OpenTheory.OptimalApplicationAlgorithms
open OpenTheory.DistillationAndSelfPlayRLVR

/-!
# Smoke Tests for `OpenTheory.AgentZeroKnowledgePassport`
-/

#check @isBooleanBitZ
#check @isBooleanBitZ_iff
#check @isBooleanBitZ_bounds
#check @BitWitness4Z
#check @evalBitWitness4Z
#check @verifyBitRange4Z
#check @verifyBitRange4Z_soundness
#check @verifySelfContainedR1CSIneqZ
#check @verifySelfContainedR1CSIneqZ_soundness
#check @PrivateAgentInventoryZ
#check @PublicInventoryEnvelopeZ
#check @agentInventoryFingerprintZ
#check @verifyAgentInventoryZ
#check @PrivateAgentCapabilityZ
#check @PublicCapabilityEnvelopeZ
#check @agentCapabilityFingerprintZ
#check @verifyAgentCapabilityZ
#check @PrivateExecutionTraceZ
#check @PublicExecutionReceiptZ
#check @executionTraceFingerprintZ
#check @verifyAgentExecutionZ
#check @verifyFullAgentZKPassportZ
#check @agent_zk_passport_certifies_has_can_did_over_reals

/-- Kernel `rfl` verification of 4-bit self-contained R1CS range proof (`slack = 11 = 1 + 2*1 + 4*0 + 8*1`). -/
example : verifyBitRange4Z 11 { b0 := 1, b1 := 1, b2 := 0, b3 := 1 } = true := rfl

/-- Kernel `rfl` verification of self-contained R1CS inequality (`42 ≤ 53` with `slack = 11`). -/
example : verifySelfContainedR1CSIneqZ 42 53 { b0 := 1, b1 := 1, b2 := 0, b3 := 1 } = true := rfl

/-- Concrete private inventory witness (what the Agent HAS). -/
def sampleInventoryZ : PrivateAgentInventoryZ where
  d_stem := 16
  k_active := 2
  E_experts := 8
  d_expert := 32
  d_head := 8
  b_quant := 4
  δ_prior_forget := 0
  n_clean_base := 100
  n_rlvr_verified := 400
  C_avail_flops := 1000000
  inventorySalt := 111222333

/-- Concrete public inventory envelope. -/
def samplePubInventoryZ : PublicInventoryEnvelopeZ where
  max_active_dim := 100
  max_bit_ops := 400
  min_verified_samples := 500
  min_required_flops := 500000
  inventoryCommitment :=
    pedersenLinearCommit 7 1 (agentInventoryFingerprintZ sampleInventoryZ) 111222333

/-- Concrete private capability witness (what the Agent CAN DO & GENERATE). -/
def sampleCapabilityZ : PrivateAgentCapabilityZ where
  S_sig_num := 10
  σ_base_sq_num := 2
  d_eff_num := 25
  proj_forbidden_subspace := 0
  conformal_miscoverage_num := 1
  conformal_miscoverage_den := 100
  capabilitySalt := 444555666

/-- Concrete public capability envelope. -/
def samplePubCapabilityZ : PublicCapabilityEnvelopeZ where
  target_risk_num := 1
  target_risk_den := 5
  max_miscoverage_num := 5
  max_miscoverage_den := 100
  capabilityCommitment :=
    pedersenLinearCommit 7 1 (agentCapabilityFingerprintZ sampleCapabilityZ) 444555666

/-- Concrete private execution trace witness (what the Agent DID). -/
def sampleExecutionTraceZ : PrivateExecutionTraceZ where
  cot_trace_hash := 99887766
  sys_prompt_hash := 55443322
  K_search_executed := 8
  verifier_score_num := 95
  verifier_score_den := 100
  step_error_num := 1
  step_error_den := 20
  executionSalt := 777888999

/-- Concrete public execution receipt. -/
def samplePubExecutionReceiptZ : PublicExecutionReceiptZ where
  prompt_output_hash := 12345678
  min_K_search := 4
  min_verifier_score_num := 90
  min_verifier_score_den := 100
  Λ_max_num := 1
  Λ_max_den := 2
  C_ceil_num := 1
  C_ceil_den := 5
  executionCommitment :=
    pedersenLinearCommit 7 1
      (executionTraceFingerprintZ sampleExecutionTraceZ
        { prompt_output_hash := 12345678
          min_K_search := 4
          min_verifier_score_num := 90
          min_verifier_score_den := 100
          Λ_max_num := 1
          Λ_max_den := 2
          C_ceil_num := 1
          C_ceil_den := 5
          executionCommitment := 0 })
      777888999

/-- Kernel `rfl` verification of the complete full-lifecycle Agent Zero-Knowledge Passport! -/
example :
    verifyFullAgentZKPassportZ 7
      sampleInventoryZ samplePubInventoryZ
      sampleCapabilityZ samplePubCapabilityZ
      sampleExecutionTraceZ samplePubExecutionReceiptZ = true := rfl

#print axioms verifyBitRange4Z_soundness
#print axioms verifySelfContainedR1CSIneqZ_soundness
#print axioms agent_zk_passport_certifies_has_can_did_over_reals
