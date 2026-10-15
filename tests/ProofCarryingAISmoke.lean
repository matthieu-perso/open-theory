import OpenTheory.Certification.ProofCarryingAI

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
open OpenTheory.ProofCarryingAI
open scoped BigOperators

variable {E F : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
  [NormedAddCommGroup F] [NormedSpace ℝ F]

-- Section 1: Closed-Loop Control Lyapunov / Safety Barrier Forward Invariance
#check @closedLoopStateError
#check @closedLoopStateError_zero
#check @closedLoopStateError_succ
#check @closedLoopStateError_zero_init_eq_autoregressiveRolloutError
#check @closedLoopStateError_nonneg
#check @IsSafeBarrierInvariant
#check @isSafeBarrierInvariant_iff
#check @safe_ball_step_invariant
#check @trajectory_stays_in_safe_ball_forever

-- Section 2: Observable Upper-Confidence Barrier Certificate from Finite-Sample Telemetry
#check @observableBarrierMargin
#check @observable_barrier_certificate_soundness

-- Section 3: Dynamic Anchor Stabilization into a Certified Safe Barrier
#check @barrierAnchorStrength
#check @barrierAnchorStrength_eq_requiredAnchorStrength
#check @damped_controller_satisfies_safe_barrier

-- Section 4: Master Proof-Carrying AI Deployment Certificate
#check @proof_carrying_ai_master_certificate

-- Smoke test 1: Observable upper-confidence telemetry barrier certificate on concrete flight-control
-- telemetry numbers (Λ_hat = 3/5, δ_Λ = 1/10, ε_hat = 1/10, δ_ε = 1/20, B_safe = 1, E₀ = 4/5).
-- The observable margin is 3/20 ≥ 0, certifying infinite-horizon safe envelope invariance in [0, 1].
example (t : ℕ) :
    observableBarrierMargin (3 / 5) (1 / 10) (1 / 10) (1 / 20) 1 = 3 / 20 ∧
      IsSafeBarrierInvariant (13 / 20) (3 / 25) 1 ∧
      0 ≤ closedLoopStateError (13 / 20) (3 / 25) (4 / 5) t ∧
      closedLoopStateError (13 / 20) (3 / 25) (4 / 5) t ≤ 1 := by
  have hcert := observable_barrier_certificate_soundness
    (Λ_ctrl := 13 / 20) (ε_ctrl := 3 / 25) (E₀ := 4 / 5) (B_safe := 1)
    (Λ_hat := 3 / 5) (δ_Λ := 1 / 10) (ε_hat := 1 / 10) (δ_ε := 1 / 20)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num [observableBarrierMargin])
  refine ⟨?_, hcert.1, (hcert.2 t).1, (hcert.2 t).2⟩
  norm_num [observableBarrierMargin]

-- Smoke test 2: Dynamic anchor stabilization of an unanchored expansive controller (Λ_upper = 2,
-- ε_upper = 1/5, B_safe = 1) using `barrierAnchorStrength 2 (1/5) 1 = 3/5`.
-- With α_anchor = 3/5, the damped controller satisfies the safe barrier condition and stays inside
-- [0, 1] for all t : ℕ starting from E₀ = 9/10.
example (t : ℕ) :
    barrierAnchorStrength 2 (1 / 5) 1 = 3 / 5 ∧
      dampedSensitivity (3 / 5) 2 = 4 / 5 ∧
      IsSafeBarrierInvariant (dampedSensitivity (3 / 5) 2) (1 / 5) 1 ∧
      0 ≤ closedLoopStateError (dampedSensitivity (3 / 5) 2) (1 / 5) (9 / 10) t ∧
      closedLoopStateError (dampedSensitivity (3 / 5) 2) (1 / 5) (9 / 10) t ≤ 1 := by
  have hdamp := damped_controller_satisfies_safe_barrier
    (α_anchor := 3 / 5) (Λ_ctrl := 2) (Λ_upper := 2)
    (ε_ctrl := 1 / 5) (ε_upper := 1 / 5) (E₀ := 9 / 10) (B_safe := 1)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num [barrierAnchorStrength]) (by norm_num)
  refine ⟨?_, ?_, hdamp.1, (hdamp.2 t).1, (hdamp.2 t).2⟩
  · norm_num [barrierAnchorStrength]
  · norm_num [dampedSensitivity]

-- Smoke test 3: Exact step-by-step trajectory check and zero-init reduction to
-- `autoregressiveRolloutError` for Λ_ctrl = 1/2, ε_ctrl = 1/4, B_safe = 1/2.
example :
    IsSafeBarrierInvariant (1 / 2) (1 / 4) (1 / 2) ∧
      closedLoopStateError (1 / 2) (1 / 4) 0 3 = autoregressiveRolloutError (1 / 2) (1 / 4) 3 ∧
      closedLoopStateError (1 / 2) (1 / 4) 0 3 = 7 / 16 ∧
      closedLoopStateError (1 / 2) (1 / 4) (1 / 2) 3 = 1 / 2 := by
  refine ⟨?_, ?_, ?_, ?_⟩
  · norm_num [IsSafeBarrierInvariant]
  · exact closedLoopStateError_zero_init_eq_autoregressiveRolloutError (1 / 2) (1 / 4) 3
  · norm_num [closedLoopStateError, rolloutErrorStep]
  · norm_num [closedLoopStateError, rolloutErrorStep]

-- Axiom audits (confirming zero sorryAx)
#print axioms OpenTheory.ProofCarryingAI.closedLoopStateError_zero_init_eq_autoregressiveRolloutError
#print axioms OpenTheory.ProofCarryingAI.isSafeBarrierInvariant_iff
#print axioms OpenTheory.ProofCarryingAI.safe_ball_step_invariant
#print axioms OpenTheory.ProofCarryingAI.trajectory_stays_in_safe_ball_forever
#print axioms OpenTheory.ProofCarryingAI.observable_barrier_certificate_soundness
#print axioms OpenTheory.ProofCarryingAI.damped_controller_satisfies_safe_barrier
#print axioms OpenTheory.ProofCarryingAI.proof_carrying_ai_master_certificate
