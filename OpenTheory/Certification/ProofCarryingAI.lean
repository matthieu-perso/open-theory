import OpenTheory.Architectures.RuntimeReasoningGuardrail

/-!
# Proof-Carrying Neural Systems for Safety-Critical Industries

This module formalizes **Proof-Carrying Neural Systems (`OpenTheory.ProofCarryingAI`)** for
safety-critical domains (aviation, medicine, autonomous control, and power grids), establishing
infinite-horizon closed-loop control barrier invariance, observable upper-confidence telemetry
certificates, dynamic anchor stabilization into certified safe envelopes, and the Master
Proof-Carrying AI Deployment Theorem.

## Main Definitions & Theorems

1. **Section 1: Closed-Loop Control Lyapunov / Safety Barrier Forward Invariance**:
   - `closedLoopStateError Λ_ctrl ε_ctrl E₀ t`: recursive state deviation / control tracking error
     at step `t : ℕ` starting from initial state deviation `E₀ : ℝ` under step sensitivity `Λ_ctrl`
     and per-step policy disturbance `ε_ctrl`.
   - `closedLoopStateError_zero_init_eq_autoregressiveRolloutError`: when `E₀ = 0`,
     `closedLoopStateError Λ_ctrl ε_ctrl 0 H = autoregressiveRolloutError Λ_ctrl ε_ctrl H`.
   - `IsSafeBarrierInvariant Λ_ctrl ε_ctrl B_safe`: the safety barrier condition
     `Λ_ctrl * B_safe + ε_ctrl ≤ B_safe`.
   - `isSafeBarrierInvariant_iff`: for `0 ≤ Λ_ctrl < 1`,
     `IsSafeBarrierInvariant Λ_ctrl ε_ctrl B_safe ↔ ε_ctrl / (1 - Λ_ctrl) ≤ B_safe`.
   - `safe_ball_step_invariant`: one-step safe ball preservation
     (`rolloutErrorStep Λ_ctrl ε_ctrl E ≤ B_safe`).
   - `trajectory_stays_in_safe_ball_forever`: infinite-horizon safe ball forward invariance
     (`0 ≤ closedLoopStateError Λ_ctrl ε_ctrl E₀ t ∧ closedLoopStateError Λ_ctrl ε_ctrl E₀ t ≤ B_safe`
     for all `t : ℕ`).

2. **Section 2: Observable Upper-Confidence Barrier Certificate from Finite-Sample Telemetry**:
   - `observableBarrierMargin Λ_hat δ_Λ ε_hat δ_ε B_safe = B_safe - ((Λ_hat + δ_Λ) * B_safe + (ε_hat + δ_ε))`:
     observable upper-confidence safety barrier margin from telemetry estimates `(Λ_hat, ε_hat)` and
     confidence radii `(δ_Λ, δ_ε)`.
   - `observable_barrier_certificate_soundness`: soundness of the observable telemetry certificate,
     guaranteeing `IsSafeBarrierInvariant Λ_ctrl ε_ctrl B_safe` and
     `0 ≤ closedLoopStateError Λ_ctrl ε_ctrl E₀ t ≤ B_safe` for all `t : ℕ`.

3. **Section 3: Dynamic Anchor Stabilization into a Certified Safe Barrier**:
   - `barrierAnchorStrength Λ_upper ε_upper B_safe = 1 - (B_safe - ε_upper) / (Λ_upper * B_safe)`:
     minimum anchoring strength needed to stabilize an unanchored controller into the safe barrier
     `B_safe`.
   - `damped_controller_satisfies_safe_barrier`: whenever `0 < Λ_ctrl ≤ Λ_upper`,
     `0 ≤ ε_ctrl ≤ ε_upper < B_safe`, and
     `barrierAnchorStrength Λ_upper ε_upper B_safe ≤ α_anchor ≤ 1`, the damped closed-loop
     controller satisfies `IsSafeBarrierInvariant (dampedSensitivity α_anchor Λ_ctrl) ε_ctrl B_safe`
     and stays inside `[0, B_safe]` for all `t : ℕ`.

4. **Section 4: Master Proof-Carrying AI Deployment Certificate**:
   - `proof_carrying_ai_master_certificate`: bundles all 4 safety-critical deployment guarantees:
     (1) infinite-horizon closed-loop barrier invariance in `[0, B_safe]`,
     (2) zero OOD spurious shift error on `SESC` circuits under orthogonal bulk perturbations,
     (3) hard eradication of the unsafe actuator subspace (`P_unsafe (...) = 0`), and
     (4) bounded benign task performance shift (`2 * P.risk (full) + ablationExcessRiskShift`).
-/

noncomputable section

open TorchLean MeasureTheory StatLean.StatisticalLearning
open scoped BigOperators

namespace OpenTheory.ProofCarryingAI

open OpenTheory.ConvolutionalGeneralization
open OpenTheory.SpectralSparseCircuit
open OpenTheory.AutoregressiveOOD
open OpenTheory.Attention
open OpenTheory.AttentionAblation
open OpenTheory.TransformerGeneralization
open OpenTheory.RuntimeReasoningGuardrail

variable {E F : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
  [NormedAddCommGroup F] [NormedSpace ℝ F]

/-! ## 1. Closed-Loop Control Lyapunov / Safety Barrier Forward Invariance -/

/-- **Closed-Loop State Deviation / Control Tracking Error (`closedLoopStateError`)**:
In a safety-critical closed-loop control system driven by a neural policy, the state deviation
at step `t : ℕ` starting from initial deviation `E₀ : ℝ` evolves under step sensitivity `Λ_ctrl`
and per-step policy disturbance `ε_ctrl` via `rolloutErrorStep Λ_ctrl ε_ctrl`. -/
def closedLoopStateError (Λ_ctrl ε_ctrl E₀ : ℝ) : ℕ → ℝ
  | 0 => E₀
  | t + 1 => rolloutErrorStep Λ_ctrl ε_ctrl (closedLoopStateError Λ_ctrl ε_ctrl E₀ t)

@[simp]
theorem closedLoopStateError_zero (Λ_ctrl ε_ctrl E₀ : ℝ) :
    closedLoopStateError Λ_ctrl ε_ctrl E₀ 0 = E₀ :=
  rfl

@[simp]
theorem closedLoopStateError_succ (Λ_ctrl ε_ctrl E₀ : ℝ) (t : ℕ) :
    closedLoopStateError Λ_ctrl ε_ctrl E₀ (t + 1) =
      rolloutErrorStep Λ_ctrl ε_ctrl (closedLoopStateError Λ_ctrl ε_ctrl E₀ t) :=
  rfl

/-- **Zero-Initial-Error Reduction (`closedLoopStateError_zero_init_eq_autoregressiveRolloutError`)**:
When the initial state deviation is `E₀ = 0`, `closedLoopStateError Λ_ctrl ε_ctrl 0 H` coincides
identically with `autoregressiveRolloutError Λ_ctrl ε_ctrl H`. -/
theorem closedLoopStateError_zero_init_eq_autoregressiveRolloutError
    (Λ_ctrl ε_ctrl : ℝ) :
    ∀ H : ℕ, closedLoopStateError Λ_ctrl ε_ctrl 0 H = autoregressiveRolloutError Λ_ctrl ε_ctrl H
  | 0 => rfl
  | H + 1 => by
      simp only [closedLoopStateError_succ, autoregressiveRolloutError,
        closedLoopStateError_zero_init_eq_autoregressiveRolloutError Λ_ctrl ε_ctrl H]

/-- Nonnegativity of `closedLoopStateError` at all steps `t : ℕ` whenever `0 ≤ Λ_ctrl`,
`0 ≤ ε_ctrl`, and `0 ≤ E₀`. -/
theorem closedLoopStateError_nonneg {Λ_ctrl ε_ctrl E₀ : ℝ}
    (hΛ : 0 ≤ Λ_ctrl) (hε : 0 ≤ ε_ctrl) (hE₀ : 0 ≤ E₀) :
    ∀ t : ℕ, 0 ≤ closedLoopStateError Λ_ctrl ε_ctrl E₀ t
  | 0 => hE₀
  | t + 1 => by
      rw [closedLoopStateError_succ, rolloutErrorStep]
      exact add_nonneg (mul_nonneg hΛ (closedLoopStateError_nonneg hΛ hε hE₀ t)) hε

/-- **Safety Barrier Condition (`IsSafeBarrierInvariant`)**:
A safety envelope radius `B_safe : ℝ` is a control barrier invariant under step sensitivity
`Λ_ctrl` and per-step policy disturbance `ε_ctrl` if `Λ_ctrl * B_safe + ε_ctrl ≤ B_safe`. -/
def IsSafeBarrierInvariant (Λ_ctrl ε_ctrl B_safe : ℝ) : Prop :=
  Λ_ctrl * B_safe + ε_ctrl ≤ B_safe

/-- **Contractive Barrier Fixed-Point Equivalence (`isSafeBarrierInvariant_iff`)**:
In the contractive regime `0 ≤ Λ_ctrl < 1`, the safety barrier condition
`IsSafeBarrierInvariant Λ_ctrl ε_ctrl B_safe` holds if and only if the asymptotic fixed-point
radius `ε_ctrl / (1 - Λ_ctrl)` is at most `B_safe`. -/
theorem isSafeBarrierInvariant_iff {Λ_ctrl ε_ctrl B_safe : ℝ}
    (_hΛ_nonneg : 0 ≤ Λ_ctrl) (hΛ_lt : Λ_ctrl < 1) :
    IsSafeBarrierInvariant Λ_ctrl ε_ctrl B_safe ↔ ε_ctrl / (1 - Λ_ctrl) ≤ B_safe := by
  unfold IsSafeBarrierInvariant
  have h1_sub_pos : 0 < 1 - Λ_ctrl := sub_pos.mpr hΛ_lt
  rw [div_le_iff₀ h1_sub_pos]
  constructor <;> intro h <;> linarith

/-- **One-Step Safe Ball Preservation (`safe_ball_step_invariant`)**:
If `0 ≤ Λ_ctrl`, `IsSafeBarrierInvariant Λ_ctrl ε_ctrl B_safe`, and the current state error
satisfies `E ≤ B_safe`, then the next-step state error also satisfies
`rolloutErrorStep Λ_ctrl ε_ctrl E ≤ B_safe`. -/
theorem safe_ball_step_invariant {Λ_ctrl ε_ctrl B_safe E : ℝ}
    (hΛ : 0 ≤ Λ_ctrl)
    (hbar : IsSafeBarrierInvariant Λ_ctrl ε_ctrl B_safe)
    (hE : E ≤ B_safe) :
    rolloutErrorStep Λ_ctrl ε_ctrl E ≤ B_safe := by
  unfold rolloutErrorStep
  unfold IsSafeBarrierInvariant at hbar
  have hmul : Λ_ctrl * E ≤ Λ_ctrl * B_safe := mul_le_mul_of_nonneg_left hE hΛ
  linarith

/-- **Infinite-Horizon Safe Ball Forward Invariance (`trajectory_stays_in_safe_ball_forever`)**:
If `0 ≤ Λ_ctrl`, `0 ≤ ε_ctrl`, `0 ≤ E₀`, `E₀ ≤ B_safe`, and
`IsSafeBarrierInvariant Λ_ctrl ε_ctrl B_safe`, then for **every** time step `t : ℕ`:
`0 ≤ closedLoopStateError Λ_ctrl ε_ctrl E₀ t ∧ closedLoopStateError Λ_ctrl ε_ctrl E₀ t ≤ B_safe`.
The closed-loop neural controller is mathematically guaranteed never to leave the safe state
envelope `[0, B_safe]` over an infinite operational lifetime `t ∈ [0, ∞)`! -/
theorem trajectory_stays_in_safe_ball_forever {Λ_ctrl ε_ctrl E₀ B_safe : ℝ}
    (hΛ : 0 ≤ Λ_ctrl) (hε : 0 ≤ ε_ctrl)
    (hE₀_nonneg : 0 ≤ E₀) (hE₀_le : E₀ ≤ B_safe)
    (hbar : IsSafeBarrierInvariant Λ_ctrl ε_ctrl B_safe) (t : ℕ) :
    0 ≤ closedLoopStateError Λ_ctrl ε_ctrl E₀ t ∧
      closedLoopStateError Λ_ctrl ε_ctrl E₀ t ≤ B_safe := by
  refine ⟨closedLoopStateError_nonneg hΛ hε hE₀_nonneg t, ?_⟩
  induction t with
  | zero => exact hE₀_le
  | succ k ih =>
      rw [closedLoopStateError_succ]
      exact safe_ball_step_invariant hΛ hbar ih

/-! ## 2. Observable Upper-Confidence Barrier Certificate from Finite-Sample Telemetry -/

/-- **Observable Upper-Confidence Barrier Margin (`observableBarrierMargin`)**:
Given empirical telemetry estimates `(Λ_hat, ε_hat)` with confidence radii `(δ_Λ, δ_ε)` and
safety envelope `B_safe`, the observable barrier margin is
`observableBarrierMargin Λ_hat δ_Λ ε_hat δ_ε B_safe = B_safe - ((Λ_hat + δ_Λ) * B_safe + (ε_hat + δ_ε))`. -/
def observableBarrierMargin (Λ_hat δ_Λ ε_hat δ_ε B_safe : ℝ) : ℝ :=
  B_safe - ((Λ_hat + δ_Λ) * B_safe + (ε_hat + δ_ε))

/-- **Soundness of the Observable Telemetry Barrier Certificate
(`observable_barrier_certificate_soundness`)**:
Whenever `0 ≤ Λ_ctrl`, `0 ≤ ε_ctrl`, `0 ≤ E₀ ≤ B_safe`, the telemetry bounds
`|Λ_hat - Λ_ctrl| ≤ δ_Λ` and `|ε_hat - ε_ctrl| ≤ δ_ε` hold, and the observable margin satisfies
`0 ≤ observableBarrierMargin Λ_hat δ_Λ ε_hat δ_ε B_safe`:
1. The true unknown system satisfies `IsSafeBarrierInvariant Λ_ctrl ε_ctrl B_safe`, and
2. For all `t : ℕ`, `0 ≤ closedLoopStateError Λ_ctrl ε_ctrl E₀ t ≤ B_safe`. -/
theorem observable_barrier_certificate_soundness
    {Λ_ctrl ε_ctrl E₀ B_safe Λ_hat δ_Λ ε_hat δ_ε : ℝ}
    (hΛ : 0 ≤ Λ_ctrl) (hε : 0 ≤ ε_ctrl)
    (hE₀_nonneg : 0 ≤ E₀) (hE₀_le : E₀ ≤ B_safe)
    (hest_Λ : |Λ_hat - Λ_ctrl| ≤ δ_Λ)
    (hest_ε : |ε_hat - ε_ctrl| ≤ δ_ε)
    (hmargin : 0 ≤ observableBarrierMargin Λ_hat δ_Λ ε_hat δ_ε B_safe) :
    IsSafeBarrierInvariant Λ_ctrl ε_ctrl B_safe ∧
      ∀ t : ℕ,
        0 ≤ closedLoopStateError Λ_ctrl ε_ctrl E₀ t ∧
          closedLoopStateError Λ_ctrl ε_ctrl E₀ t ≤ B_safe := by
  have hB_nonneg : 0 ≤ B_safe := hE₀_nonneg.trans hE₀_le
  have hΛ_ucb : Λ_ctrl ≤ Λ_hat + δ_Λ := true_sensitivity_le_upper_confidence hest_Λ
  have hε_ucb : ε_ctrl ≤ ε_hat + δ_ε := true_sensitivity_le_upper_confidence hest_ε
  have hmul : Λ_ctrl * B_safe ≤ (Λ_hat + δ_Λ) * B_safe :=
    mul_le_mul_of_nonneg_right hΛ_ucb hB_nonneg
  have hbar : IsSafeBarrierInvariant Λ_ctrl ε_ctrl B_safe := by
    unfold IsSafeBarrierInvariant
    unfold observableBarrierMargin at hmargin
    linarith
  exact ⟨hbar, trajectory_stays_in_safe_ball_forever hΛ hε hE₀_nonneg hE₀_le hbar⟩

/-! ## 3. Dynamic Anchor Stabilization into a Certified Safe Barrier -/

/-- **Minimum Barrier-Stabilizing Anchor Strength (`barrierAnchorStrength`)**:
Given upper confidence bounds `Λ_upper` on step sensitivity and `ε_upper` on per-step disturbance,
the minimum anchor strength `α_anchor` required to force the closed-loop system into the safe
barrier `B_safe` is:
`barrierAnchorStrength Λ_upper ε_upper B_safe = 1 - (B_safe - ε_upper) / (Λ_upper * B_safe)`. -/
def barrierAnchorStrength (Λ_upper ε_upper B_safe : ℝ) : ℝ :=
  1 - (B_safe - ε_upper) / (Λ_upper * B_safe)

/-- Connection between `barrierAnchorStrength` and `requiredAnchorStrength`:
`barrierAnchorStrength Λ_upper ε_upper B_safe = requiredAnchorStrength Λ_upper ((B_safe - ε_upper) / B_safe)`. -/
theorem barrierAnchorStrength_eq_requiredAnchorStrength (Λ_upper ε_upper B_safe : ℝ) :
    barrierAnchorStrength Λ_upper ε_upper B_safe =
      requiredAnchorStrength Λ_upper ((B_safe - ε_upper) / B_safe) := by
  unfold barrierAnchorStrength requiredAnchorStrength
  rw [div_div, mul_comm B_safe Λ_upper]

/-- **Dynamic Anchor Stabilization into a Certified Safe Barrier
(`damped_controller_satisfies_safe_barrier`)**:
Whenever `0 < Λ_ctrl ≤ Λ_upper`, `0 ≤ ε_ctrl ≤ ε_upper < B_safe`, `0 ≤ E₀ ≤ B_safe`, and the
anchoring strength satisfies `barrierAnchorStrength Λ_upper ε_upper B_safe ≤ α_anchor ≤ 1`,
the damped closed-loop controller satisfies
`IsSafeBarrierInvariant (dampedSensitivity α_anchor Λ_ctrl) ε_ctrl B_safe` and stays inside
`[0, B_safe]` for all `t : ℕ`! -/
theorem damped_controller_satisfies_safe_barrier
    {α_anchor Λ_ctrl Λ_upper ε_ctrl ε_upper E₀ B_safe : ℝ}
    (hΛ_pos : 0 < Λ_ctrl) (hΛ_le : Λ_ctrl ≤ Λ_upper)
    (hε_nonneg : 0 ≤ ε_ctrl) (hε_le : ε_ctrl ≤ ε_upper)
    (hε_lt_B : ε_upper < B_safe)
    (hE₀_nonneg : 0 ≤ E₀) (hE₀_le : E₀ ≤ B_safe)
    (hα_req : barrierAnchorStrength Λ_upper ε_upper B_safe ≤ α_anchor)
    (hα_le_one : α_anchor ≤ 1) :
    IsSafeBarrierInvariant (dampedSensitivity α_anchor Λ_ctrl) ε_ctrl B_safe ∧
      ∀ t : ℕ,
        0 ≤ closedLoopStateError (dampedSensitivity α_anchor Λ_ctrl) ε_ctrl E₀ t ∧
          closedLoopStateError (dampedSensitivity α_anchor Λ_ctrl) ε_ctrl E₀ t ≤ B_safe := by
  have hB_pos : 0 < B_safe := lt_of_le_of_lt (hε_nonneg.trans hε_le) hε_lt_B
  have hΛ_upper_pos : 0 < Λ_upper := lt_of_lt_of_le hΛ_pos hΛ_le
  have hprod_pos : 0 < Λ_upper * B_safe := mul_pos hΛ_upper_pos hB_pos
  have h1_sub_nonneg : 0 ≤ 1 - α_anchor := sub_nonneg.mpr hα_le_one
  have hdamp_nonneg : 0 ≤ dampedSensitivity α_anchor Λ_ctrl :=
    dampedSensitivity_nonneg hα_le_one hΛ_pos.le
  have hsub_le_div : 1 - α_anchor ≤ (B_safe - ε_upper) / (Λ_upper * B_safe) := by
    unfold barrierAnchorStrength at hα_req
    linarith
  have hmul_upper : (1 - α_anchor) * (Λ_upper * B_safe) ≤ B_safe - ε_upper :=
    (le_div_iff₀ hprod_pos).mp hsub_le_div
  have hmul_ctrl :
      dampedSensitivity α_anchor Λ_ctrl * B_safe ≤ (1 - α_anchor) * (Λ_upper * B_safe) := by
    unfold dampedSensitivity
    have hle_B : Λ_ctrl * B_safe ≤ Λ_upper * B_safe :=
      mul_le_mul_of_nonneg_right hΛ_le hB_pos.le
    calc
      (1 - α_anchor) * Λ_ctrl * B_safe = (1 - α_anchor) * (Λ_ctrl * B_safe) := by ring
      _ ≤ (1 - α_anchor) * (Λ_upper * B_safe) :=
        mul_le_mul_of_nonneg_left hle_B h1_sub_nonneg
  have hbar : IsSafeBarrierInvariant (dampedSensitivity α_anchor Λ_ctrl) ε_ctrl B_safe := by
    unfold IsSafeBarrierInvariant
    linarith
  exact ⟨hbar, trajectory_stays_in_safe_ball_forever hdamp_nonneg hε_nonneg hE₀_nonneg hE₀_le hbar⟩

/-! ## 4. Master Proof-Carrying AI Deployment Certificate -/

/-- **Master Proof-Carrying AI Deployment Certificate (`proof_carrying_ai_master_certificate`)**:
Unifies all four machine-checked certification pillars required to deploy a neural system in
safety-critical infrastructure (aviation, medicine, autonomous control, power grids):
1. **Infinite-Horizon Closed-Loop Barrier Invariance**:
   `IsSafeBarrierInvariant (dampedSensitivity α_anchor Λ_ctrl) ε_ctrl B_safe` holds, and for every
   time step `t : ℕ`,
   `0 ≤ closedLoopStateError (dampedSensitivity α_anchor Λ_ctrl) ε_ctrl E₀ t ≤ B_safe`.
2. **Zero OOD Spurious Shift Error**:
   Under any orthogonal spectral-bulk OOD perturbation `η_bulk`,
   `sescCircuitForward net embed B_spec (x + η_bulk) = sescCircuitForward net embed B_spec x` and
   `(sescCircuitForward net embed B_spec (x + η_bulk) - sescCircuitForward net embed B_spec x) ^ 2 = 0`.
3. **Hard Eradication of Unsafe Actuator Subspace**:
   For every observation `z : Z`,
   `P_unsafe (multiHeadAttentionSubset S_keep (scores z) (mask z) (values z) proj) = 0`.
4. **Bounded Benign Performance Shift**:
   The ablated multi-head controller has integrable benign loss and population risk bounded by
   `2 * P.risk (full) + ablationExcessRiskShift proj S_keep B_V`. -/
theorem proof_carrying_ai_master_certificate
    {Z : Type*} [MeasurableSpace Z]
    {d k r m H_heads n_ctx : ℕ} [NeZero d]
    -- Pillar 1 parameters (Closed-Loop Control Barrier Invariance under Anchor Damping)
    {α_anchor Λ_ctrl Λ_upper ε_ctrl ε_upper E₀ B_safe : ℝ}
    (hΛ_pos : 0 < Λ_ctrl) (hΛ_le : Λ_ctrl ≤ Λ_upper)
    (hε_nonneg : 0 ≤ ε_ctrl) (hε_le : ε_ctrl ≤ ε_upper)
    (hε_lt_B : ε_upper < B_safe)
    (hE₀_nonneg : 0 ≤ E₀) (hE₀_le : E₀ ≤ B_safe)
    (hα_req : barrierAnchorStrength Λ_upper ε_upper B_safe ≤ α_anchor)
    (hα_le_one : α_anchor ≤ 1)
    -- Pillar 2 parameters (SESC Orthogonal Bulk OOD Immunity)
    (net : SpectralSparseCircuit r m)
    (embed : Fin k → Fin d)
    (B_spec : Matrix (Fin r) (Fin k) ℝ)
    (x η_bulk : Fin d → ℝ)
    (h_ortho : ∀ p : Fin d, ∀ l : Fin r, patchProject B_spec (cyclicPatch embed η_bulk p) l = 0)
    -- Pillars 3 & 4 parameters (Unsafe Actuator Subspace Eradication & Benign Risk Certificate)
    (P : Population Z)
    (P_unsafe : F →L[ℝ] F)
    (scores : Z → Fin H_heads → Tensor ℝ [n_ctx])
    (mask : Z → Fin H_heads → Tensor Bool [n_ctx])
    (values : Z → Fin H_heads → Fin n_ctx → E)
    (proj : Fin H_heads → (E →L[ℝ] F))
    (S_keep : Finset (Fin H_heads))
    (B_V : Fin H_heads → ℝ)
    (label : Z → F)
    (horth : ∀ (z : Z), ∀ h ∈ S_keep,
      P_unsafe (headContribution (scores z) (mask z) (values z) proj h) = 0)
    (hB : ∀ h ∈ S_keepᶜ, 0 ≤ B_V h)
    (hmeas_ablated : AEStronglyMeasurable
      (fun z =>
        ‖multiHeadAttentionSubset S_keep (scores z) (mask z) (values z) proj - label z‖ ^ 2)
      P.law)
    (hint_full : P.IntegrableLoss
      (fun z => ‖multiHeadOutput (scores z) (mask z) (values z) proj - label z‖ ^ 2))
    (hae : ∀ᵐ z ∂P.law,
      (∀ h ∈ S_keepᶜ, ∃ i, (mask z h).getScalar i = true) ∧
      (∀ h ∈ S_keepᶜ, ∀ i, ‖values z h i‖ ≤ B_V h)) :
    -- (1) Infinite-Horizon Closed-Loop Barrier Invariance
    (IsSafeBarrierInvariant (dampedSensitivity α_anchor Λ_ctrl) ε_ctrl B_safe ∧
     ∀ t : ℕ,
       0 ≤ closedLoopStateError (dampedSensitivity α_anchor Λ_ctrl) ε_ctrl E₀ t ∧
         closedLoopStateError (dampedSensitivity α_anchor Λ_ctrl) ε_ctrl E₀ t ≤ B_safe) ∧
    -- (2) Zero OOD Spurious Shift Error
    (sescCircuitForward net embed B_spec (x + η_bulk) =
      sescCircuitForward net embed B_spec x ∧
     (sescCircuitForward net embed B_spec (x + η_bulk) -
      sescCircuitForward net embed B_spec x) ^ 2 = 0) ∧
    -- (3) Hard Eradication of Unsafe Actuator Subspace
    (∀ z : Z,
      P_unsafe (multiHeadAttentionSubset S_keep (scores z) (mask z) (values z) proj) = 0) ∧
    -- (4) Bounded Benign Performance Shift
    (P.IntegrableLoss
      (fun z =>
        ‖multiHeadAttentionSubset S_keep (scores z) (mask z) (values z) proj - label z‖ ^ 2) ∧
     P.risk
      (fun z =>
        ‖multiHeadAttentionSubset S_keep (scores z) (mask z) (values z) proj - label z‖ ^ 2) ≤
      2 * P.risk
        (fun z => ‖multiHeadOutput (scores z) (mask z) (values z) proj - label z‖ ^ 2) +
        ablationExcessRiskShift proj S_keep B_V) := by
  have hpillar1 :=
    damped_controller_satisfies_safe_barrier
      hΛ_pos hΛ_le hε_nonneg hε_le hε_lt_B hE₀_nonneg hE₀_le hα_req hα_le_one
  have hpillar2_eq :
      sescCircuitForward net embed B_spec (x + η_bulk) =
        sescCircuitForward net embed B_spec x :=
    (sesc_orthogonal_bulk_invariance net embed B_spec x η_bulk h_ortho).1
  have hpillar2_sq :
      (sescCircuitForward net embed B_spec (x + η_bulk) -
        sescCircuitForward net embed B_spec x) ^ 2 = 0 :=
    (sesc_zero_ood_shift_penalty net embed B_spec x η_bulk h_ortho).1
  obtain ⟨hpillar3, hpillar4_int, hpillar4_risk⟩ :=
    orthogonal_circuit_ablation_zero_unsafe_and_risk_bound
      P P_unsafe scores mask values proj S_keep B_V label
      horth hB hmeas_ablated hint_full hae
  exact ⟨hpillar1, ⟨hpillar2_eq, hpillar2_sq⟩, hpillar3, ⟨hpillar4_int, hpillar4_risk⟩⟩

end OpenTheory.ProofCarryingAI
