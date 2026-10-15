import OpenTheory.Generalization.AutoregressiveOOD
import OpenTheory.Generalization.TransformerGeneralization
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Ring

/-!
# Certified Runtime Reasoning & Out-of-Distribution (OOD) Safety Guardrail

This module formalizes **System 3: Certified Runtime Reasoning & Out-of-Distribution (OOD) Safety
Guardrail** in `OpenTheory.RuntimeReasoningGuardrail`—a runtime monitor and projection layer that
wraps an autoregressive Transformer / `SESC` agent during live inference to provide machine-checked
runtime certificates across three simultaneous failure modes:
1. **Long-Horizon Chain-of-Thought (CoT) Drift** (`Λ_raw > 1`),
2. **Adversarial / Out-of-Distribution (OOD) Spurious Feature Shift** (`η_bulk`), and
3. **Unsafe Output Subspace Activation** (`P_unsafe`).

## Main Definitions & Theorems

1. **Section 1: Runtime Contractive Jacobian Monitor & Dynamic Rollout Damping Guardrail**:
   - `dampedSensitivity α_anchor Λ_raw = (1 - α_anchor) * Λ_raw`: runtime anchor / verification
     damping operator mixing the raw step-to-step Jacobian sensitivity `Λ_raw > 0` against verified
     context with anchoring strength `α_anchor ∈ [0, 1]`.
   - `requiredAnchorStrength Λ_raw Λ_max = 1 - Λ_max / Λ_raw`: minimum certified anchor strength
     needed to enforce a target contraction margin `Λ_max ∈ [0, 1)`.
   - `dampedSensitivity_at_requiredAnchorStrength`: exact calibration identity
     `dampedSensitivity (requiredAnchorStrength Λ_raw Λ_max) Λ_raw = Λ_max`.
   - `dampedSensitivity_le_target_iff`: for `0 < Λ_raw`,
     `dampedSensitivity α_anchor Λ_raw ≤ Λ_max ↔ requiredAnchorStrength Λ_raw Λ_max ≤ α_anchor`.
   - `runtime_damped_rollout_horizon_free_certificate`: whenever
     `α_anchor ∈ [requiredAnchorStrength Λ_raw Λ_max, 1]` with `0 < Λ_raw`, `0 ≤ Λ_max < 1`, and
     `0 ≤ ε_token`, the damped sensitivity satisfies `0 ≤ dampedSensitivity α_anchor Λ_raw ≤ Λ_max < 1`
     and for **every** reasoning horizon `H : ℕ`,
     `autoregressiveRolloutError (dampedSensitivity α_anchor Λ_raw) ε_token H ≤ ε_token / (1 - Λ_max)`.

2. **Section 2: Certified Upper-Confidence Runtime Contraction Under Noisy Jacobian Estimates**:
   - `upperConfidenceRolloutBound Λ_hat δ_Λ ε_token = ε_token / (1 - (Λ_hat + δ_Λ))`: observable
     upper-confidence runtime error certificate when the true step sensitivity `Λ` is estimated by
     `Λ_hat` within tolerance `|Λ_hat - Λ| ≤ δ_Λ`.
   - `upper_confidence_runtime_rollout_soundness`: whenever `0 ≤ Λ`, `|Λ_hat - Λ| ≤ δ_Λ`,
     `Λ_hat + δ_Λ < 1`, and `0 ≤ ε_token`, the true `H`-step rollout error is certifiably bounded
     for all `H : ℕ` by:
     `autoregressiveRolloutError Λ ε_token H ≤ ε_token / (1 - Λ) ≤ upperConfidenceRolloutBound Λ_hat δ_Λ ε_token`.

3. **Section 3: Simultaneous Triple-Shield Runtime Guardrail (Horizon Stability + Zero OOD Bulk Shift + Zero Unsafe Subspace)**:
   - `ablationExcessRiskShift proj S_keep B_V = 2 * (∑ h ∈ S_keepᶜ, ‖proj h‖ * B_V h) ^ 2`:
     explicit benign MSE shift bound induced by ablating unsafe attention heads `S_keepᶜ`.
   - `master_runtime_triple_shield_certificate`: unified runtime safety and reliability theorem
     certifying simultaneously:
     1. **Horizon-Free Reasoning Stability**: `autoregressiveRolloutError (dampedSensitivity α_anchor Λ_raw) ε_token H ≤ ε_token / (1 - Λ_max)` for all `H : ℕ`,
     2. **Exact OOD Bulk Immunity**: `(sescCircuitForward net embed B_spec (x + η_bulk) - sescCircuitForward net embed B_spec x) ^ 2 = 0`, and
     3. **Zero Unsafe Subspace Activation with Bounded Benign Drift**:
        `P_unsafe (ablatedMultiHeadOutput (scores z) (mask z) (values z) proj S_keep) = 0` for all `z : Z`
        and benign population MSE bounded by `2 * P.risk (full) + ablationExcessRiskShift proj S_keep B_V`.
-/

noncomputable section

open TorchLean MeasureTheory StatLean.StatisticalLearning
open scoped BigOperators

namespace OpenTheory.RuntimeReasoningGuardrail

open OpenTheory.ConvolutionalGeneralization
open OpenTheory.SpectralSparseCircuit
open OpenTheory.AutoregressiveOOD
open OpenTheory.Attention
open OpenTheory.AttentionAblation
open OpenTheory.TransformerGeneralization

variable {E F : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
  [NormedAddCommGroup F] [NormedSpace ℝ F]

/-! ## 1. Runtime Contractive Jacobian Monitor & Dynamic Rollout Damping Guardrail -/

/-- **Runtime Anchor / Verification Damping Operator (`dampedSensitivity`)**:
Given an unguarded step-to-step Jacobian sensitivity `Λ_raw : ℝ` and an induction-head /
context-verification anchoring weight `α_anchor ∈ [0, 1]`, the damped step sensitivity is
`dampedSensitivity α_anchor Λ_raw = (1 - α_anchor) * Λ_raw`. -/
def dampedSensitivity (α_anchor Λ_raw : ℝ) : ℝ :=
  (1 - α_anchor) * Λ_raw

/-- **Minimum Certified Anchor Strength (`requiredAnchorStrength`)**:
The minimum anchoring weight `α_anchor` needed to contract a raw step sensitivity `Λ_raw > 0`
down to a target contraction factor `Λ_max`:
`requiredAnchorStrength Λ_raw Λ_max = 1 - Λ_max / Λ_raw`. -/
def requiredAnchorStrength (Λ_raw Λ_max : ℝ) : ℝ :=
  1 - Λ_max / Λ_raw

/-- At zero anchoring (`α_anchor = 0`), `dampedSensitivity` equals the raw sensitivity `Λ_raw`. -/
@[simp]
theorem dampedSensitivity_zero (Λ_raw : ℝ) :
    dampedSensitivity 0 Λ_raw = Λ_raw := by
  simp [dampedSensitivity]

/-- At full anchoring (`α_anchor = 1`), `dampedSensitivity` vanishes identically. -/
@[simp]
theorem dampedSensitivity_one (Λ_raw : ℝ) :
    dampedSensitivity 1 Λ_raw = 0 := by
  simp [dampedSensitivity]

/-- Nonnegativity of `dampedSensitivity` when `α_anchor ≤ 1` and `0 ≤ Λ_raw`. -/
theorem dampedSensitivity_nonneg {α_anchor Λ_raw : ℝ}
    (hα_le_one : α_anchor ≤ 1) (hΛ_nonneg : 0 ≤ Λ_raw) :
    0 ≤ dampedSensitivity α_anchor Λ_raw := by
  unfold dampedSensitivity
  have h1 : 0 ≤ 1 - α_anchor := sub_nonneg.mpr hα_le_one
  exact mul_nonneg h1 hΛ_nonneg

/-- Antitonicity of `dampedSensitivity` in the anchoring strength `α_anchor` when `0 ≤ Λ_raw`. -/
theorem dampedSensitivity_anti_in_anchor {α₁ α₂ Λ_raw : ℝ}
    (hα : α₁ ≤ α₂) (hΛ_nonneg : 0 ≤ Λ_raw) :
    dampedSensitivity α₂ Λ_raw ≤ dampedSensitivity α₁ Λ_raw := by
  unfold dampedSensitivity
  have hsub : 1 - α₂ ≤ 1 - α₁ := by linarith
  exact mul_le_mul_of_nonneg_right hsub hΛ_nonneg

/-- **Exact Calibration Identity (`dampedSensitivity_at_requiredAnchorStrength`)**:
For any `Λ_raw ≠ 0`, setting `α_anchor = requiredAnchorStrength Λ_raw Λ_max` achieves
`dampedSensitivity (requiredAnchorStrength Λ_raw Λ_max) Λ_raw = Λ_max`. -/
theorem dampedSensitivity_at_requiredAnchorStrength {Λ_raw : ℝ} (Λ_max : ℝ)
    (hΛ_ne : Λ_raw ≠ 0) :
    dampedSensitivity (requiredAnchorStrength Λ_raw Λ_max) Λ_raw = Λ_max := by
  unfold dampedSensitivity requiredAnchorStrength
  field_simp [hΛ_ne]
  ring

/-- Feasibility of `requiredAnchorStrength` in `[0, 1]` when `0 < Λ_raw`, `0 ≤ Λ_max`, and
`Λ_max ≤ Λ_raw`:
`0 ≤ requiredAnchorStrength Λ_raw Λ_max ≤ 1`. -/
theorem requiredAnchorStrength_mem_unit_interval {Λ_raw Λ_max : ℝ}
    (hΛ_pos : 0 < Λ_raw) (hΛ_max_nonneg : 0 ≤ Λ_max) (hΛ_le : Λ_max ≤ Λ_raw) :
    0 ≤ requiredAnchorStrength Λ_raw Λ_max ∧ requiredAnchorStrength Λ_raw Λ_max ≤ 1 := by
  unfold requiredAnchorStrength
  have hdiv_nonneg : 0 ≤ Λ_max / Λ_raw := div_nonneg hΛ_max_nonneg hΛ_pos.le
  have hdiv_le_one : Λ_max / Λ_raw ≤ 1 := (div_le_one hΛ_pos).mpr hΛ_le
  exact ⟨by linarith, by linarith⟩

/-- **Exact Anchor Calibration Equivalence (`dampedSensitivity_le_target_iff`)**:
For `0 < Λ_raw`, the damped sensitivity is at most the target contraction bound `Λ_max`
if and only if the anchoring strength `α_anchor` is at least `requiredAnchorStrength Λ_raw Λ_max`. -/
theorem dampedSensitivity_le_target_iff {α_anchor Λ_raw Λ_max : ℝ}
    (hΛ_pos : 0 < Λ_raw) :
    dampedSensitivity α_anchor Λ_raw ≤ Λ_max ↔
      requiredAnchorStrength Λ_raw Λ_max ≤ α_anchor := by
  unfold dampedSensitivity requiredAnchorStrength
  rw [← le_div_iff₀ hΛ_pos]
  constructor <;> intro h <;> linarith

/-- Monotonicity of `autoregressiveRolloutError` in the transition sensitivity `Λ` when
`0 ≤ Λ₁ ≤ Λ₂` and `0 ≤ ε_token`. -/
theorem autoregressiveRolloutError_mono_in_sensitivity {Λ₁ Λ₂ ε_token : ℝ}
    (hΛ₁ : 0 ≤ Λ₁) (hΛ_le : Λ₁ ≤ Λ₂) (hε : 0 ≤ ε_token) :
    ∀ H : ℕ, autoregressiveRolloutError Λ₁ ε_token H ≤ autoregressiveRolloutError Λ₂ ε_token H
  | 0 => le_rfl
  | H + 1 => by
      rw [autoregressiveRolloutError_succ, autoregressiveRolloutError_succ]
      have ih := autoregressiveRolloutError_mono_in_sensitivity hΛ₁ hΛ_le hε H
      have hE₁_nonneg : 0 ≤ autoregressiveRolloutError Λ₁ ε_token H :=
        autoregressiveRolloutError_nonneg hΛ₁ hε H
      have hΛ₂ : 0 ≤ Λ₂ := hΛ₁.trans hΛ_le
      have hmul :
          Λ₁ * autoregressiveRolloutError Λ₁ ε_token H ≤
            Λ₂ * autoregressiveRolloutError Λ₂ ε_token H :=
        mul_le_mul hΛ_le ih hE₁_nonneg hΛ₂
      linarith

/-- **Runtime Damped Rollout Horizon-Free Stability Certificate
(`runtime_damped_rollout_horizon_free_certificate`)**:
Whenever the anchoring strength satisfies `requiredAnchorStrength Λ_raw Λ_max ≤ α_anchor ≤ 1`
with `0 < Λ_raw`, `0 ≤ Λ_max < 1`, and `0 ≤ ε_token`, the damped step sensitivity satisfies
`0 ≤ dampedSensitivity α_anchor Λ_raw ≤ Λ_max < 1`, and for **every** reasoning horizon `H : ℕ`:
`autoregressiveRolloutError (dampedSensitivity α_anchor Λ_raw) ε_token H ≤ ε_token / (1 - Λ_max)`.
Even when the unguarded model would blow up exponentially (`Λ_raw > 1`), the runtime anchor
guardrail restores **horizon-free stability** across arbitrarily long reasoning chains! -/
theorem runtime_damped_rollout_horizon_free_certificate {α_anchor Λ_raw Λ_max ε_token : ℝ}
    (hΛ_pos : 0 < Λ_raw)
    (_hΛ_max_nonneg : 0 ≤ Λ_max)
    (hΛ_max_lt : Λ_max < 1)
    (hα_req : requiredAnchorStrength Λ_raw Λ_max ≤ α_anchor)
    (hα_le_one : α_anchor ≤ 1)
    (hε : 0 ≤ ε_token) (H : ℕ) :
    0 ≤ dampedSensitivity α_anchor Λ_raw ∧
      dampedSensitivity α_anchor Λ_raw ≤ Λ_max ∧
      dampedSensitivity α_anchor Λ_raw < 1 ∧
      autoregressiveRolloutError (dampedSensitivity α_anchor Λ_raw) ε_token H ≤
        ε_token / (1 - dampedSensitivity α_anchor Λ_raw) ∧
      autoregressiveRolloutError (dampedSensitivity α_anchor Λ_raw) ε_token H ≤
        ε_token / (1 - Λ_max) := by
  have hdamp_nonneg : 0 ≤ dampedSensitivity α_anchor Λ_raw :=
    dampedSensitivity_nonneg hα_le_one hΛ_pos.le
  have hdamp_le_max : dampedSensitivity α_anchor Λ_raw ≤ Λ_max :=
    (dampedSensitivity_le_target_iff hΛ_pos).mpr hα_req
  have hdamp_lt_one : dampedSensitivity α_anchor Λ_raw < 1 :=
    hdamp_le_max.trans_lt hΛ_max_lt
  have hcontr_damp :=
    (cot_contractive_rollout_horizon_free_bound hdamp_nonneg hdamp_lt_one hε H).2
  have h1_sub_max_pos : 0 < 1 - Λ_max := sub_pos.mpr hΛ_max_lt
  have h1_sub_le : 1 - Λ_max ≤ 1 - dampedSensitivity α_anchor Λ_raw := by linarith
  have hdiv_le :
      ε_token / (1 - dampedSensitivity α_anchor Λ_raw) ≤ ε_token / (1 - Λ_max) :=
    div_le_div_of_nonneg_left hε h1_sub_max_pos h1_sub_le
  exact ⟨hdamp_nonneg, hdamp_le_max, hdamp_lt_one, hcontr_damp, hcontr_damp.trans hdiv_le⟩

/-! ## 2. Certified Upper-Confidence Runtime Contraction Under Noisy Jacobian Estimates -/

/-- **Upper-Confidence Runtime Error Certificate (`upperConfidenceRolloutBound`)**:
When the true step-to-step sensitivity `Λ` is estimated at runtime by `Λ_hat` with estimation
error tolerance `|Λ_hat - Λ| ≤ δ_Λ` and upper confidence bound `Λ_hat + δ_Λ < 1`, the observable
upper-confidence rollout error certificate is
`upperConfidenceRolloutBound Λ_hat δ_Λ ε_token = ε_token / (1 - (Λ_hat + δ_Λ))`. -/
def upperConfidenceRolloutBound (Λ_hat δ_Λ ε_token : ℝ) : ℝ :=
  ε_token / (1 - (Λ_hat + δ_Λ))

/-- Under `|Λ_hat - Λ| ≤ δ_Λ`, the true sensitivity `Λ` is upper-bounded by the observable
upper confidence limit `Λ_hat + δ_Λ`. -/
theorem true_sensitivity_le_upper_confidence {Λ Λ_hat δ_Λ : ℝ}
    (hest : |Λ_hat - Λ| ≤ δ_Λ) :
    Λ ≤ Λ_hat + δ_Λ := by
  have h := (abs_le.mp hest).1
  linarith

/-- Nonnegativity of `upperConfidenceRolloutBound` when `Λ_hat + δ_Λ < 1` and `0 ≤ ε_token`. -/
theorem upperConfidenceRolloutBound_nonneg {Λ_hat δ_Λ ε_token : ℝ}
    (hucb_lt : Λ_hat + δ_Λ < 1) (hε : 0 ≤ ε_token) :
    0 ≤ upperConfidenceRolloutBound Λ_hat δ_Λ ε_token := by
  unfold upperConfidenceRolloutBound
  have hden_pos : 0 < 1 - (Λ_hat + δ_Λ) := sub_pos.mpr hucb_lt
  exact div_nonneg hε hden_pos.le

/-- **Soundness of the Upper-Confidence Runtime Rollout Certificate
(`upper_confidence_runtime_rollout_soundness`)**:
Whenever `0 ≤ Λ`, `|Λ_hat - Λ| ≤ δ_Λ`, `Λ_hat + δ_Λ < 1`, and `0 ≤ ε_token`, the true
(unobserved) `H`-step rollout error is certifiably bounded by the observable upper-confidence
certificate for **all** horizons `H : ℕ`:
`autoregressiveRolloutError Λ ε_token H ≤ ε_token / (1 - Λ) ≤ upperConfidenceRolloutBound Λ_hat δ_Λ ε_token`. -/
theorem upper_confidence_runtime_rollout_soundness {Λ Λ_hat δ_Λ ε_token : ℝ}
    (hΛ_nonneg : 0 ≤ Λ)
    (hest : |Λ_hat - Λ| ≤ δ_Λ)
    (hucb_lt : Λ_hat + δ_Λ < 1)
    (hε : 0 ≤ ε_token) (H : ℕ) :
    Λ ≤ Λ_hat + δ_Λ ∧
      Λ < 1 ∧
      autoregressiveRolloutError Λ ε_token H ≤ ε_token / (1 - Λ) ∧
      ε_token / (1 - Λ) ≤ upperConfidenceRolloutBound Λ_hat δ_Λ ε_token ∧
      autoregressiveRolloutError Λ ε_token H ≤
        upperConfidenceRolloutBound Λ_hat δ_Λ ε_token := by
  have hΛ_le_ucb : Λ ≤ Λ_hat + δ_Λ := true_sensitivity_le_upper_confidence hest
  have hΛ_lt_one : Λ < 1 := hΛ_le_ucb.trans_lt hucb_lt
  have hrollout_le : autoregressiveRolloutError Λ ε_token H ≤ ε_token / (1 - Λ) :=
    (cot_contractive_rollout_horizon_free_bound hΛ_nonneg hΛ_lt_one hε H).2
  have hden_ucb_pos : 0 < 1 - (Λ_hat + δ_Λ) := sub_pos.mpr hucb_lt
  have hden_le : 1 - (Λ_hat + δ_Λ) ≤ 1 - Λ := by linarith
  have hucb_bound : ε_token / (1 - Λ) ≤ upperConfidenceRolloutBound Λ_hat δ_Λ ε_token := by
    unfold upperConfidenceRolloutBound
    exact div_le_div_of_nonneg_left hε hden_ucb_pos hden_le
  exact ⟨hΛ_le_ucb, hΛ_lt_one, hrollout_le, hucb_bound, hrollout_le.trans hucb_bound⟩

/-- Tightness of `upperConfidenceRolloutBound` as estimation noise vanishes (`δ_Λ = 0`):
when `Λ_hat = Λ` and `δ_Λ = 0`, `upperConfidenceRolloutBound Λ 0 ε_token = ε_token / (1 - Λ)`. -/
@[simp]
theorem upperConfidenceRolloutBound_zero_noise (Λ ε_token : ℝ) :
    upperConfidenceRolloutBound Λ 0 ε_token = ε_token / (1 - Λ) := by
  simp [upperConfidenceRolloutBound]

/-! ## 3. Simultaneous Triple-Shield Runtime Guardrail (Horizon Stability + Zero OOD Bulk Shift + Zero Unsafe Subspace) -/

/-- Alias for `ablatedMultiHeadOutput` matching `multiHeadAttentionSubset S_keep` notation:
retains only the certified-safe attention head subset `S_keep : Finset (Fin H_heads)`. -/
def multiHeadAttentionSubset {H_heads n_ctx : ℕ}
    (S_keep : Finset (Fin H_heads))
    (scores : Fin H_heads → Tensor ℝ [n_ctx])
    (mask : Fin H_heads → Tensor Bool [n_ctx])
    (values : Fin H_heads → Fin n_ctx → E)
    (proj : Fin H_heads → (E →L[ℝ] F)) : F :=
  ablatedMultiHeadOutput scores mask values proj S_keep

@[simp]
theorem multiHeadAttentionSubset_eq_ablatedMultiHeadOutput {H_heads n_ctx : ℕ}
    (S_keep : Finset (Fin H_heads))
    (scores : Fin H_heads → Tensor ℝ [n_ctx])
    (mask : Fin H_heads → Tensor Bool [n_ctx])
    (values : Fin H_heads → Fin n_ctx → E)
    (proj : Fin H_heads → (E →L[ℝ] F)) :
    multiHeadAttentionSubset S_keep scores mask values proj =
      ablatedMultiHeadOutput scores mask values proj S_keep :=
  rfl

/-- **Benign Ablation Excess Risk Shift (`ablationExcessRiskShift`)**:
The additive population MSE shift bound `2 * (∑ h ∈ S_keepᶜ, ‖proj h‖ * B_V h) ^ 2` incurred on the
benign task when ablating unsafe attention heads `S_keepᶜ`. -/
def ablationExcessRiskShift {H_heads : ℕ}
    (proj : Fin H_heads → (E →L[ℝ] F))
    (S_keep : Finset (Fin H_heads))
    (B_V : Fin H_heads → ℝ) : ℝ :=
  2 * (∑ h ∈ S_keepᶜ, ‖proj h‖ * B_V h) ^ 2

/-- Nonnegativity of `ablationExcessRiskShift`. -/
theorem ablationExcessRiskShift_nonneg {H_heads : ℕ}
    (proj : Fin H_heads → (E →L[ℝ] F))
    (S_keep : Finset (Fin H_heads))
    (B_V : Fin H_heads → ℝ) :
    0 ≤ ablationExcessRiskShift proj S_keep B_V := by
  unfold ablationExcessRiskShift
  positivity

/-- **Master Runtime Triple-Shield Certificate (`master_runtime_triple_shield_certificate`)**:
In a single unified theorem, certifies all three runtime safety and reliability shields
simultaneously for an autoregressive `SESC` / Multi-Head Transformer agent:
1. **Shield 1 — Horizon-Free Reasoning Stability**:
   Under anchor strength `requiredAnchorStrength Λ_raw Λ_max ≤ α_anchor ≤ 1` (`0 < Λ_raw`,
   `0 ≤ Λ_max < 1`, `0 ≤ ε_token`), for **every** reasoning horizon `H : ℕ`,
   `autoregressiveRolloutError (dampedSensitivity α_anchor Λ_raw) ε_token H ≤ ε_token / (1 - Λ_max)`.
2. **Shield 2 — Exact OOD Bulk Immunity**:
   For any orthogonal spectral-bulk OOD shift `η_bulk : Fin d → ℝ`
   (`∀ p l, patchProject B_spec (cyclicPatch embed η_bulk p) l = 0`), the `SESC` circuit forward
   pass suffers identically zero OOD shift:
   `sescCircuitForward net embed B_spec (x + η_bulk) = sescCircuitForward net embed B_spec x` and
   `(sescCircuitForward net embed B_spec (x + η_bulk) - sescCircuitForward net embed B_spec x) ^ 2 = 0`.
3. **Shield 3 — Zero Unsafe Subspace Activation with Bounded Benign Drift**:
   When attention heads outside `S_keep` are ablated and every retained head `h ∈ S_keep` is
   orthogonal to `P_unsafe : F →L[ℝ] F`, the unsafe projection vanishes identically for all `z : Z`
   (`P_unsafe (multiHeadAttentionSubset S_keep (scores z) (mask z) (values z) proj) = 0`), while
   benign task population MSE is bounded by
   `2 * P.risk (full) + ablationExcessRiskShift proj S_keep B_V`. -/
theorem master_runtime_triple_shield_certificate
    {Z : Type*} [MeasurableSpace Z]
    {d k r m H_heads n_ctx : ℕ} [NeZero d]
    -- Shield 1 parameters (Runtime Contractive CoT Rollout Guardrail)
    {α_anchor Λ_raw Λ_max ε_token : ℝ}
    (hΛ_pos : 0 < Λ_raw)
    (hΛ_max_nonneg : 0 ≤ Λ_max)
    (hΛ_max_lt : Λ_max < 1)
    (hα_req : requiredAnchorStrength Λ_raw Λ_max ≤ α_anchor)
    (hα_le_one : α_anchor ≤ 1)
    (hε : 0 ≤ ε_token)
    -- Shield 2 parameters (SESC Orthogonal Bulk OOD Filter)
    (net : SpectralSparseCircuit r m)
    (embed : Fin k → Fin d)
    (B_spec : Matrix (Fin r) (Fin k) ℝ)
    (x η_bulk : Fin d → ℝ)
    (h_ortho : ∀ p : Fin d, ∀ l : Fin r, patchProject B_spec (cyclicPatch embed η_bulk p) l = 0)
    -- Shield 3 parameters (Mechanistic Unsafe-Subspace Head Ablation Shield)
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
    -- (1) Horizon-Free Reasoning Stability across all H : ℕ
    (∀ H : ℕ,
      autoregressiveRolloutError (dampedSensitivity α_anchor Λ_raw) ε_token H ≤
        ε_token / (1 - Λ_max)) ∧
    -- (2) Exact OOD Bulk Immunity
    (sescCircuitForward net embed B_spec (x + η_bulk) =
      sescCircuitForward net embed B_spec x ∧
     (sescCircuitForward net embed B_spec (x + η_bulk) -
      sescCircuitForward net embed B_spec x) ^ 2 = 0) ∧
    -- (3) Zero Unsafe Subspace Activation & Bounded Benign Risk Drift
    ((∀ z : Z,
      P_unsafe (multiHeadAttentionSubset S_keep (scores z) (mask z) (values z) proj) = 0) ∧
     P.IntegrableLoss
      (fun z =>
        ‖multiHeadAttentionSubset S_keep (scores z) (mask z) (values z) proj - label z‖ ^ 2) ∧
     P.risk
      (fun z =>
        ‖multiHeadAttentionSubset S_keep (scores z) (mask z) (values z) proj - label z‖ ^ 2) ≤
      2 * P.risk
        (fun z => ‖multiHeadOutput (scores z) (mask z) (values z) proj - label z‖ ^ 2) +
        ablationExcessRiskShift proj S_keep B_V) := by
  have hshield1 : ∀ H : ℕ,
      autoregressiveRolloutError (dampedSensitivity α_anchor Λ_raw) ε_token H ≤
        ε_token / (1 - Λ_max) :=
    fun H =>
      (runtime_damped_rollout_horizon_free_certificate
        hΛ_pos hΛ_max_nonneg hΛ_max_lt hα_req hα_le_one hε H).2.2.2.2
  have hshield2_eq :
      sescCircuitForward net embed B_spec (x + η_bulk) =
        sescCircuitForward net embed B_spec x :=
    (sesc_orthogonal_bulk_invariance net embed B_spec x η_bulk h_ortho).1
  have hshield2_sq :
      (sescCircuitForward net embed B_spec (x + η_bulk) -
        sescCircuitForward net embed B_spec x) ^ 2 = 0 :=
    (sesc_zero_ood_shift_penalty net embed B_spec x η_bulk h_ortho).1
  have hshield3 :=
    orthogonal_circuit_ablation_zero_unsafe_and_risk_bound
      P P_unsafe scores mask values proj S_keep B_V label
      horth hB hmeas_ablated hint_full hae
  exact ⟨hshield1, ⟨hshield2_eq, hshield2_sq⟩, hshield3⟩

end OpenTheory.RuntimeReasoningGuardrail
