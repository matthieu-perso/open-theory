import OpenTheory.Certification.ZeroKnowledgeCommitment

open OpenTheory.ZeroKnowledgeCommitment
open OpenTheory.ComputableTractableEngine

/-!
# Smoke Tests for `OpenTheory.ZeroKnowledgeCommitment`
-/

#check @pedersenLinearCommit
#check @pedersenBlindingEquiv
#check @pedersen_commitment_perfect_hiding
#check @pedersen_opening_binding
#check @SigmaTranscript
#check @sigmaRespond
#check @sigmaVerify
#check @sigmaSimulate
#check @sigma_protocol_completeness
#check @sigma_protocol_two_special_soundness
#check @sigma_protocol_field_witness_extractor
#check @sigma_protocol_hvzk_simulation
#check @BitOrTranscript
#check @verifyBitOrTranscript
#check @proveBitOrZero
#check @proveBitOrOne
#check @bit_or_protocol_completeness
#check @bit_or_protocol_two_special_soundness
#check @sigma_homomorphic_linear_batch_completeness
#check @PrivateSafetyWitnessZ
#check @PublicSafetyEnvelopeZ
#check @privateWitnessFingerprintZ
#check @verifyFlopSavingsZ
#check @verifyBarrierSafetyZ
#check @verifyRolloutCeilingZ
#check @verifyR1CSSlackGadgetZ
#check @verifyZeroKnowledgeSafetyCertZ
#check @verifyR1CSSlackGadgetZ_iff
#check @verifyFlopSavingsZ_iff_Q
#check @verifyRolloutCeilingZ_iff_Q
#check @verifyBarrierSafetyZ_unit_iff_Q
#check @zk_commitment_certifies_real_deployment_guarantees

/-- Concrete private telemetry witness (kept secret by Prover). -/
def samplePrivateWitness : PrivateSafetyWitnessZ where
  sr_active := 6
  md_dense := 128
  B_RPM_num := 1
  B_SGD_num := 4
  Λ_raw_num := 1
  Λ_raw_den := 2
  δ_Λ_num := 1
  δ_Λ_den := 20
  ε_token_num := 1
  ε_token_den := 10
  δ_ε_num := 1
  δ_ε_den := 50
  blindingSalt := 987654321

/-- Concrete public safety envelope (shared with Verifier). -/
def samplePublicEnvelope : PublicSafetyEnvelopeZ where
  Λ_max_num := 1
  Λ_max_den := 2
  B_safe_num := 2
  B_safe_den := 5
  C_ceil_num := 1
  C_ceil_den := 5
  commitment := pedersenLinearCommit 7 1 (privateWitnessFingerprintZ samplePrivateWitness) 987654321

/-- Kernel `rfl` verification of the complete division-free ZK safety certificate. -/
example : verifyZeroKnowledgeSafetyCertZ 7 samplePrivateWitness samplePublicEnvelope = true := rfl

/-- Kernel `rfl` verification of the 3-move Schnorr/Pedersen Σ-protocol over `ZMod 97`. -/
example :
    let g : ZMod 97 := 5
    let h : ZMod 97 := 11
    let w : ZMod 97 := 42
    let r : ZMod 97 := 19
    let C := pedersenLinearCommit g h w r
    let tr := sigmaRespond g h w r 13 29 7
    sigmaVerify g h C tr = true := rfl

/-- Kernel `rfl` verification of the HVZK Simulator over `ZMod 97`. -/
example :
    let g : ZMod 97 := 5
    let h : ZMod 97 := 11
    let C : ZMod 97 := 61
    let trSim := sigmaSimulate g h C 7 81 34
    sigmaVerify g h C trSim = true := rfl

/-- Kernel `rfl` verification of the 1-of-2 Disjunctive Bit OR-Proof (`b = 0` and `b = 1`) over `ZMod 97`. -/
example :
    let g : ZMod 97 := 5
    let h : ZMod 97 := 11
    let r : ZMod 97 := 19
    let c : ZMod 97 := 43
    verifyBitOrTranscript g h (pedersenLinearCommit g h 0 r) c
      (proveBitOrZero g h r 13 29 7 c) = true ∧
    verifyBitOrTranscript g h (pedersenLinearCommit g h 1 r) c
      (proveBitOrOne g h r 13 29 7 c) = true := ⟨rfl, rfl⟩

#print axioms zk_commitment_certifies_real_deployment_guarantees
#print axioms sigma_protocol_field_witness_extractor
#print axioms pedersen_commitment_perfect_hiding
#print axioms bit_or_protocol_completeness
#print axioms bit_or_protocol_two_special_soundness
#print axioms sigma_homomorphic_linear_batch_completeness
