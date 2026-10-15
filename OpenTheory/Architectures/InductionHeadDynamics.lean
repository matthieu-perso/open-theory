import OpenTheory.Architectures.Attention
import OpenTheory.Architectures.AttentionLogitStability
import OpenTheory.Architectures.InContextRetrieval
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.Complex.ExponentialBounds

/-!
# True Nonlinear Softmax Attention Training Dynamics & Spontaneous Sparsification into 1-Hot Induction Heads

This module formalizes the exact nonlinear softmax attention probability laws, the nonlinear
gradient dynamics of the query-key logit margin during induction head formation, the expressivity
and signal-to-noise ratio (SNR) separation between nonlinear Softmax Attention and Linear/Kernel
Attention on long contexts, and the end-to-end value retrieval error along training.

1. **Exact Softmax Target-vs-Distractor Probability & Margin Law**:
   - Defines `softmaxTargetProb m_dist Δ = 1 / (1 + m_dist * exp(-Δ))` and
     `softmaxDistractorMass m_dist Δ = 1 - softmaxTargetProb m_dist Δ` for a context of length
     `n_ctx = 1 + m_dist` consisting of 1 target key and `m_dist` distractor keys separated by
     logit margin `Δ = z_target - z_dist`.
   - Proves `softmaxTargetProb_pos`, `softmaxTargetProb_le_one`, `softmaxDistractorMass_eq`,
     `softmaxDistractorMass_nonneg`, `softmaxDistractorMass_lt_one`,
     `softmaxDistractorMass_le_exp_neg_margin`, and `softmaxTargetProb_strict_mono_in_margin`
     (plus `softmaxDistractorMass_strict_anti_in_margin`).

2. **Nonlinear Softmax Gradient Dynamics on the Query-Key Logit Margin**:
   - Defines `logitMarginLogLossStep η g_ret m_dist Δ` (the single-step log-loss / cross-entropy
     gradient update on the query-key logit margin) and `logitMarginSqLossStep η g_ret m_dist Δ`
     (the squared-error softmax gradient update proportional to `p * (1 - p)`), together with the
     multi-step trajectory `logitMarginTrajectory Δ₀ δ_step T = Δ₀ + T * δ_step`.
   - Proves `logitMarginLogLossStep_strict_increase` (and `logitMarginSqLossStep_strict_increase`):
     for `η > 0`, `g_ret > 0`, and `m_dist ≥ 1`, the logit margin strictly increases at every
     training step.
   - Proves `induction_head_exponential_sparsification`: along `logitMarginTrajectory Δ₀ δ_step T`,
     the distractor attention mass decays exponentially in `T` as
     `m_dist * exp(-Δ₀ - T * δ_step)` and the target attention weight converges exponentially to a
     1-hot induction head (`1 - m_dist * exp(-Δ₀ - T * δ_step) ≤ softmaxTargetProb m_dist Δ_T ≤ 1`).

3. **Softmax Attention vs. Linear / Kernel Attention Expressivity & SNR Separation**:
   - Defines `linearAttentionTargetWeight m_dist Δ = Δ / (Δ + m_dist)` and
     `linearAttentionDistractorMass m_dist Δ = m_dist / (Δ + m_dist)`.
   - Proves `softmax_vs_linear_attention_distractor_suppression`: for any `Δ > 0` and `m_dist ≥ 1`
     (and in particular `Δ ≥ 1`), using `Δ < exp(Δ)`, nonlinear Softmax Attention achieves strictly
     lower distractor leakage and strictly higher target retrieval weight than Linear Attention.
   - Proves `softmax_logarithmic_margin_sufficiency`, `linear_attention_requires_linear_margin`,
     and the combined `long_context_margin_scaling`: to suppress distractor leakage below
     `ε ∈ (0, 1)` as `m_dist` grows, Softmax Attention only requires a **logarithmic** margin
     `Δ ≥ log(m_dist / ε)`, whereas Linear Attention requires a **linear** margin
     `Δ ≥ m_dist * (1 - ε) / ε`.

4. **End-to-End Trained Induction Head Retrieval Error**:
   - Defines `inductionHeadOutput p_target v_target v_dist` for the canonical target-vs-distractor
     value configuration and connects both the exact canonical output and general `Attention.rowOutput`
     on `Fin (1 + m_dist)` to `softmaxDistractorMass m_dist (logitMarginTrajectory Δ₀ δ_step T)`.
   - Proves `trained_induction_head_retrieval_error` and
     `trained_induction_head_rowOutput_retrieval_error`: `T` steps of QK training exponentially
     eliminate in-context retrieval bias with bound
     `R_V * softmaxDistractorMass m_dist (logitMarginTrajectory Δ₀ δ_step T) ≤ R_V * m_dist * exp(-Δ₀ - T * δ_step)`.
-/

noncomputable section

open TorchLean TorchLean.Tensor Spec
open scoped BigOperators

namespace OpenTheory.InductionHeadDynamics

open OpenTheory.Attention
open OpenTheory.AttentionLogitStability
open OpenTheory.InContextRetrieval

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

/-! ## Section 1: Exact Softmax Target-vs-Distractor Probability & Margin Law -/

/-- Canonical uniform-distractor softmax target attention probability for a context with
1 target key and `m_dist` distractor keys at logit margin `Δ = z_target - z_dist`:
`softmaxTargetProb m_dist Δ = 1 / (1 + m_dist * exp(-Δ))`. -/
def softmaxTargetProb (m_dist : ℕ) (Δ : ℝ) : ℝ :=
  1 / (1 + (m_dist : ℝ) * Real.exp (-Δ))

/-- Total distractor probability mass under canonical softmax attention with `m_dist`
distractors and logit margin `Δ`:
`softmaxDistractorMass m_dist Δ = 1 - softmaxTargetProb m_dist Δ`. -/
def softmaxDistractorMass (m_dist : ℕ) (Δ : ℝ) : ℝ :=
  1 - softmaxTargetProb m_dist Δ

/-- Helper lemma: the denominator `1 + (m_dist : ℝ) * Real.exp (-Δ)` of `softmaxTargetProb` is
strictly positive for every `m_dist : ℕ` and `Δ : ℝ`. -/
theorem softmaxDenom_pos (m_dist : ℕ) (Δ : ℝ) :
    0 < 1 + (m_dist : ℝ) * Real.exp (-Δ) := by
  have hm : 0 ≤ (m_dist : ℝ) := Nat.cast_nonneg m_dist
  have hexp : 0 < Real.exp (-Δ) := Real.exp_pos (-Δ)
  have hmul : 0 ≤ (m_dist : ℝ) * Real.exp (-Δ) := mul_nonneg hm hexp.le
  linarith

/-- `softmaxTargetProb m_dist Δ` is strictly positive for all `m_dist : ℕ` and `Δ : ℝ`. -/
theorem softmaxTargetProb_pos (m_dist : ℕ) (Δ : ℝ) :
    0 < softmaxTargetProb m_dist Δ := by
  unfold softmaxTargetProb
  exact div_pos one_pos (softmaxDenom_pos m_dist Δ)

/-- `softmaxTargetProb m_dist Δ` is nonnegative for all `m_dist : ℕ` and `Δ : ℝ`. -/
theorem softmaxTargetProb_nonneg (m_dist : ℕ) (Δ : ℝ) :
    0 ≤ softmaxTargetProb m_dist Δ :=
  (softmaxTargetProb_pos m_dist Δ).le

/-- `softmaxTargetProb m_dist Δ ≤ 1` for all `m_dist : ℕ` and `Δ : ℝ`. -/
theorem softmaxTargetProb_le_one (m_dist : ℕ) (Δ : ℝ) :
    softmaxTargetProb m_dist Δ ≤ 1 := by
  unfold softmaxTargetProb
  rw [div_le_iff₀ (softmaxDenom_pos m_dist Δ)]
  have hm : 0 ≤ (m_dist : ℝ) := Nat.cast_nonneg m_dist
  have hexp : 0 ≤ Real.exp (-Δ) := (Real.exp_pos (-Δ)).le
  have hmul : 0 ≤ (m_dist : ℝ) * Real.exp (-Δ) := mul_nonneg hm hexp
  linarith

/-- For `m_dist ≥ 1`, `softmaxTargetProb m_dist Δ < 1`. -/
theorem softmaxTargetProb_lt_one {m_dist : ℕ} (hm : 1 ≤ m_dist) (Δ : ℝ) :
    softmaxTargetProb m_dist Δ < 1 := by
  unfold softmaxTargetProb
  rw [div_lt_iff₀ (softmaxDenom_pos m_dist Δ)]
  have hm_pos : (0 : ℝ) < (m_dist : ℝ) := by
    have : (1 : ℝ) ≤ (m_dist : ℝ) := by exact_mod_cast hm
    linarith
  have hmul : 0 < (m_dist : ℝ) * Real.exp (-Δ) := mul_pos hm_pos (Real.exp_pos (-Δ))
  linarith

/-- Exact closed-form identity for the total distractor probability mass:
`softmaxDistractorMass m_dist Δ = ((m_dist : ℝ) * Real.exp (-Δ)) / (1 + (m_dist : ℝ) * Real.exp (-Δ))`. -/
theorem softmaxDistractorMass_eq (m_dist : ℕ) (Δ : ℝ) :
    softmaxDistractorMass m_dist Δ =
      ((m_dist : ℝ) * Real.exp (-Δ)) / (1 + (m_dist : ℝ) * Real.exp (-Δ)) := by
  unfold softmaxDistractorMass softmaxTargetProb
  have hne : 1 + (m_dist : ℝ) * Real.exp (-Δ) ≠ 0 := ne_of_gt (softmaxDenom_pos m_dist Δ)
  rw [one_sub_div hne]
  ring

/-- `softmaxDistractorMass m_dist Δ` is nonnegative for all `m_dist : ℕ` and `Δ : ℝ`. -/
theorem softmaxDistractorMass_nonneg (m_dist : ℕ) (Δ : ℝ) :
    0 ≤ softmaxDistractorMass m_dist Δ := by
  unfold softmaxDistractorMass
  linarith [softmaxTargetProb_le_one m_dist Δ]

/-- For `m_dist ≥ 1`, `softmaxDistractorMass m_dist Δ` is strictly positive. -/
theorem softmaxDistractorMass_pos {m_dist : ℕ} (hm : 1 ≤ m_dist) (Δ : ℝ) :
    0 < softmaxDistractorMass m_dist Δ := by
  unfold softmaxDistractorMass
  linarith [softmaxTargetProb_lt_one hm Δ]

/-- `softmaxDistractorMass m_dist Δ < 1` for all `m_dist : ℕ` and `Δ : ℝ`. -/
theorem softmaxDistractorMass_lt_one (m_dist : ℕ) (Δ : ℝ) :
    softmaxDistractorMass m_dist Δ < 1 := by
  unfold softmaxDistractorMass
  linarith [softmaxTargetProb_pos m_dist Δ]

/-- **Exponential Upper Bound on Softmax Distractor Mass (`softmaxDistractorMass_le_exp_neg_margin`)**:
For any `m_dist : ℕ` (and in particular `m_dist ≥ 1`) and any margin `Δ : ℝ`,
`softmaxDistractorMass m_dist Δ ≤ (m_dist : ℝ) * Real.exp (-Δ)`. -/
theorem softmaxDistractorMass_le_exp_neg_margin (m_dist : ℕ) (Δ : ℝ) :
    softmaxDistractorMass m_dist Δ ≤ (m_dist : ℝ) * Real.exp (-Δ) := by
  rw [softmaxDistractorMass_eq]
  rw [div_le_iff₀ (softmaxDenom_pos m_dist Δ)]
  have hnum_nonneg : 0 ≤ (m_dist : ℝ) * Real.exp (-Δ) :=
    mul_nonneg (Nat.cast_nonneg m_dist) (Real.exp_pos (-Δ)).le
  have hdenom_ge_one : 1 ≤ 1 + (m_dist : ℝ) * Real.exp (-Δ) := by linarith
  calc
    (m_dist : ℝ) * Real.exp (-Δ) = ((m_dist : ℝ) * Real.exp (-Δ)) * 1 := (mul_one _).symm
    _ ≤ ((m_dist : ℝ) * Real.exp (-Δ)) * (1 + (m_dist : ℝ) * Real.exp (-Δ)) :=
      mul_le_mul_of_nonneg_left hdenom_ge_one hnum_nonneg

/-- **Strict Monotonicity of Softmax Target Probability in the Logit Margin
(`softmaxTargetProb_strict_mono_in_margin`)**:
For `m_dist ≥ 1` and `Δ₁ < Δ₂`,
`softmaxTargetProb m_dist Δ₁ < softmaxTargetProb m_dist Δ₂`. -/
theorem softmaxTargetProb_strict_mono_in_margin {m_dist : ℕ} (hm : 1 ≤ m_dist)
    {Δ₁ Δ₂ : ℝ} (hΔ : Δ₁ < Δ₂) :
    softmaxTargetProb m_dist Δ₁ < softmaxTargetProb m_dist Δ₂ := by
  unfold softmaxTargetProb
  have hm_pos : (0 : ℝ) < (m_dist : ℝ) := by
    have : (1 : ℝ) ≤ (m_dist : ℝ) := by exact_mod_cast hm
    linarith
  have hexp_lt : Real.exp (-Δ₂) < Real.exp (-Δ₁) :=
    Real.exp_lt_exp.mpr (neg_lt_neg hΔ)
  have hmul_lt : (m_dist : ℝ) * Real.exp (-Δ₂) < (m_dist : ℝ) * Real.exp (-Δ₁) :=
    mul_lt_mul_of_pos_left hexp_lt hm_pos
  have hdenom_lt : 1 + (m_dist : ℝ) * Real.exp (-Δ₂) < 1 + (m_dist : ℝ) * Real.exp (-Δ₁) := by
    linarith
  exact one_div_lt_one_div_of_lt (softmaxDenom_pos m_dist Δ₂) hdenom_lt

/-- Weak monotonicity of `softmaxTargetProb` in the logit margin `Δ₁ ≤ Δ₂`. -/
theorem softmaxTargetProb_mono_in_margin (m_dist : ℕ) {Δ₁ Δ₂ : ℝ} (hΔ : Δ₁ ≤ Δ₂) :
    softmaxTargetProb m_dist Δ₁ ≤ softmaxTargetProb m_dist Δ₂ := by
  unfold softmaxTargetProb
  have hm_nonneg : (0 : ℝ) ≤ (m_dist : ℝ) := Nat.cast_nonneg m_dist
  have hexp_le : Real.exp (-Δ₂) ≤ Real.exp (-Δ₁) :=
    Real.exp_le_exp.mpr (neg_le_neg hΔ)
  have hdenom_le : 1 + (m_dist : ℝ) * Real.exp (-Δ₂) ≤ 1 + (m_dist : ℝ) * Real.exp (-Δ₁) := by
    linarith [mul_le_mul_of_nonneg_left hexp_le hm_nonneg]
  exact one_div_le_one_div_of_le (softmaxDenom_pos m_dist Δ₂) hdenom_le

/-- Strict antotonicity of `softmaxDistractorMass` in the logit margin `Δ₁ < Δ₂` for `m_dist ≥ 1`. -/
theorem softmaxDistractorMass_strict_anti_in_margin {m_dist : ℕ} (hm : 1 ≤ m_dist)
    {Δ₁ Δ₂ : ℝ} (hΔ : Δ₁ < Δ₂) :
    softmaxDistractorMass m_dist Δ₂ < softmaxDistractorMass m_dist Δ₁ := by
  unfold softmaxDistractorMass
  linarith [softmaxTargetProb_strict_mono_in_margin hm hΔ]

/-! ## Section 2: Nonlinear Softmax Gradient Dynamics on the Query-Key Logit Margin -/

/-- **Log-Loss Softmax Margin Step (`logitMarginLogLossStep`)**:
Single gradient ascent step on the query-key logit margin `Δ` under cross-entropy / log-likelihood
retrieval loss with learning rate `η` and target alignment signal `g_ret`:
since `-∂/∂Δ log(softmaxTargetProb m_dist Δ) = 1 - p = softmaxDistractorMass m_dist Δ`,
the updated logit margin is `Δ + η * g_ret * softmaxDistractorMass m_dist Δ`. -/
def logitMarginLogLossStep (η g_ret : ℝ) (m_dist : ℕ) (Δ : ℝ) : ℝ :=
  Δ + η * g_ret * softmaxDistractorMass m_dist Δ

/-- **Squared-Error Softmax Margin Step (`logitMarginSqLossStep`)**:
Single gradient step on the query-key logit margin `Δ` under squared-error retrieval loss,
where the softmax Jacobian factor is `p * (1 - p) = softmaxTargetProb m_dist Δ * softmaxDistractorMass m_dist Δ`. -/
def logitMarginSqLossStep (η g_ret : ℝ) (m_dist : ℕ) (Δ : ℝ) : ℝ :=
  Δ + η * g_ret * (softmaxTargetProb m_dist Δ * softmaxDistractorMass m_dist Δ)

/-- **Direct Feature-Aligned Margin Growth Trajectory (`logitMarginTrajectory`)**:
When query-key inner product alignment increases the logit margin by `δ_step` per training step
starting from initial margin `Δ₀`, after `T : ℕ` steps the logit margin is
`Δ₀ + (T : ℝ) * δ_step`. -/
def logitMarginTrajectory (Δ₀ δ_step : ℝ) (T : ℕ) : ℝ :=
  Δ₀ + (T : ℝ) * δ_step

@[simp] theorem logitMarginTrajectory_zero (Δ₀ δ_step : ℝ) :
    logitMarginTrajectory Δ₀ δ_step 0 = Δ₀ := by
  simp [logitMarginTrajectory]

@[simp] theorem logitMarginTrajectory_succ (Δ₀ δ_step : ℝ) (T : ℕ) :
    logitMarginTrajectory Δ₀ δ_step (T + 1) =
      logitMarginTrajectory Δ₀ δ_step T + δ_step := by
  unfold logitMarginTrajectory
  push_cast
  ring

/-- **Strict Margin Increase Under Log-Loss Softmax Gradient (`logitMarginLogLossStep_strict_increase`)**:
For any learning rate `η > 0`, alignment signal `g_ret > 0`, and `m_dist ≥ 1`, the logit margin
strictly increases at every step:
`Δ < logitMarginLogLossStep η g_ret m_dist Δ`. -/
theorem logitMarginLogLossStep_strict_increase {η g_ret : ℝ} {m_dist : ℕ}
    (hη : 0 < η) (hg : 0 < g_ret) (hm : 1 ≤ m_dist) (Δ : ℝ) :
    Δ < logitMarginLogLossStep η g_ret m_dist Δ := by
  unfold logitMarginLogLossStep
  have hmass_pos : 0 < softmaxDistractorMass m_dist Δ := softmaxDistractorMass_pos hm Δ
  have hprod_pos : 0 < η * g_ret * softmaxDistractorMass m_dist Δ :=
    mul_pos (mul_pos hη hg) hmass_pos
  linarith

/-- **Strict Margin Increase Under Squared-Error Softmax Gradient (`logitMarginSqLossStep_strict_increase`)**:
For any learning rate `η > 0`, alignment signal `g_ret > 0`, and `m_dist ≥ 1`, the squared-loss
softmax margin step also strictly increases the logit margin at every step:
`Δ < logitMarginSqLossStep η g_ret m_dist Δ`. -/
theorem logitMarginSqLossStep_strict_increase {η g_ret : ℝ} {m_dist : ℕ}
    (hη : 0 < η) (hg : 0 < g_ret) (hm : 1 ≤ m_dist) (Δ : ℝ) :
    Δ < logitMarginSqLossStep η g_ret m_dist Δ := by
  unfold logitMarginSqLossStep
  have hp_pos : 0 < softmaxTargetProb m_dist Δ := softmaxTargetProb_pos m_dist Δ
  have hq_pos : 0 < softmaxDistractorMass m_dist Δ := softmaxDistractorMass_pos hm Δ
  have hprod_pos :
      0 < η * g_ret * (softmaxTargetProb m_dist Δ * softmaxDistractorMass m_dist Δ) :=
    mul_pos (mul_pos hη hg) (mul_pos hp_pos hq_pos)
  linarith

/-- One step of `logitMarginLogLossStep` strictly increases the target attention probability
`softmaxTargetProb` whenever `η > 0`, `g_ret > 0`, and `m_dist ≥ 1`. -/
theorem softmaxTargetProb_strict_increase_after_logLossStep {η g_ret : ℝ} {m_dist : ℕ}
    (hη : 0 < η) (hg : 0 < g_ret) (hm : 1 ≤ m_dist) (Δ : ℝ) :
    softmaxTargetProb m_dist Δ <
      softmaxTargetProb m_dist (logitMarginLogLossStep η g_ret m_dist Δ) :=
  softmaxTargetProb_strict_mono_in_margin hm
    (logitMarginLogLossStep_strict_increase hη hg hm Δ)

/-- **Spontaneous Sparsification into a 1-Hot Induction Head
(`induction_head_exponential_sparsification`)**:
Along `logitMarginTrajectory Δ₀ δ_step T`, the distractor attention mass decays exponentially in
training steps `T`:
1. `softmaxDistractorMass m_dist (logitMarginTrajectory Δ₀ δ_step T) ≤ (m_dist : ℝ) * Real.exp (-Δ₀ - (T : ℝ) * δ_step)`,
2. `1 - (m_dist : ℝ) * Real.exp (-Δ₀ - (T : ℝ) * δ_step) ≤ softmaxTargetProb m_dist (logitMarginTrajectory Δ₀ δ_step T)`, and
3. `softmaxTargetProb m_dist (logitMarginTrajectory Δ₀ δ_step T) ≤ 1`. -/
theorem induction_head_exponential_sparsification (m_dist : ℕ) (Δ₀ δ_step : ℝ) (T : ℕ) :
    softmaxDistractorMass m_dist (logitMarginTrajectory Δ₀ δ_step T) ≤
      (m_dist : ℝ) * Real.exp (-Δ₀ - (T : ℝ) * δ_step) ∧
    1 - (m_dist : ℝ) * Real.exp (-Δ₀ - (T : ℝ) * δ_step) ≤
      softmaxTargetProb m_dist (logitMarginTrajectory Δ₀ δ_step T) ∧
    softmaxTargetProb m_dist (logitMarginTrajectory Δ₀ δ_step T) ≤ 1 := by
  have hle := softmaxDistractorMass_le_exp_neg_margin m_dist (logitMarginTrajectory Δ₀ δ_step T)
  have hneg : -(logitMarginTrajectory Δ₀ δ_step T) = -Δ₀ - (T : ℝ) * δ_step := by
    unfold logitMarginTrajectory
    ring
  rw [hneg] at hle
  refine ⟨hle, ?_, softmaxTargetProb_le_one m_dist (logitMarginTrajectory Δ₀ δ_step T)⟩
  unfold softmaxDistractorMass at hle
  linarith

/-! ## Section 3: Softmax Attention vs. Linear / Kernel Attention Expressivity & SNR Separation -/

/-- Normalized target retrieval weight in Linear Attention when the target key has inner product
`Δ` and each of `m_dist` distractor keys has baseline inner product `1`:
`linearAttentionTargetWeight m_dist Δ = Δ / (Δ + (m_dist : ℝ))`. -/
def linearAttentionTargetWeight (m_dist : ℕ) (Δ : ℝ) : ℝ :=
  Δ / (Δ + (m_dist : ℝ))

/-- Total distractor leakage mass in Linear Attention when the target key has inner product `Δ`
and each of `m_dist` distractor keys has baseline inner product `1`:
`linearAttentionDistractorMass m_dist Δ = (m_dist : ℝ) / (Δ + (m_dist : ℝ))`. -/
def linearAttentionDistractorMass (m_dist : ℕ) (Δ : ℝ) : ℝ :=
  (m_dist : ℝ) / (Δ + (m_dist : ℝ))

/-- For `Δ + (m_dist : ℝ) ≠ 0`, the linear attention target weight and distractor mass sum to `1`. -/
theorem linearAttention_weights_sum_one (m_dist : ℕ) (Δ : ℝ)
    (hdenom : Δ + (m_dist : ℝ) ≠ 0) :
    linearAttentionTargetWeight m_dist Δ + linearAttentionDistractorMass m_dist Δ = 1 := by
  unfold linearAttentionTargetWeight linearAttentionDistractorMass
  rw [← add_div]
  exact div_self hdenom

/-- Fundamental strict exponential-vs-linear separation: for every `Δ : ℝ`, `Δ < Real.exp Δ`. -/
theorem self_lt_exp (Δ : ℝ) : Δ < Real.exp Δ := by
  have h := Real.add_one_le_exp Δ
  linarith

/-- Equivalently, `softmaxTargetProb m_dist Δ` can be written with positive exponent `Real.exp Δ`
by multiplying numerator and denominator by `Real.exp Δ`:
`softmaxTargetProb m_dist Δ = Real.exp Δ / (Real.exp Δ + (m_dist : ℝ))`. -/
theorem softmaxTargetProb_eq_exp_div (m_dist : ℕ) (Δ : ℝ) :
    softmaxTargetProb m_dist Δ = Real.exp Δ / (Real.exp Δ + (m_dist : ℝ)) := by
  unfold softmaxTargetProb
  have h1 : 1 + (m_dist : ℝ) * Real.exp (-Δ) ≠ 0 := ne_of_gt (softmaxDenom_pos m_dist Δ)
  have h2 : Real.exp Δ + (m_dist : ℝ) ≠ 0 :=
    ne_of_gt (add_pos_of_pos_of_nonneg (Real.exp_pos Δ) (Nat.cast_nonneg m_dist))
  have hprod : Real.exp (-Δ) * Real.exp Δ = 1 := by
    rw [← Real.exp_add, neg_add_cancel, Real.exp_zero]
  rw [div_eq_div_iff h1 h2]
  calc
    1 * (Real.exp Δ + (m_dist : ℝ)) = Real.exp Δ + (m_dist : ℝ) * 1 := by ring
    _ = Real.exp Δ + (m_dist : ℝ) * (Real.exp (-Δ) * Real.exp Δ) := by rw [hprod]
    _ = Real.exp Δ * (1 + (m_dist : ℝ) * Real.exp (-Δ)) := by ring

/-- Equivalently, `softmaxDistractorMass m_dist Δ = (m_dist : ℝ) / (Real.exp Δ + (m_dist : ℝ))`. -/
theorem softmaxDistractorMass_eq_div_exp (m_dist : ℕ) (Δ : ℝ) :
    softmaxDistractorMass m_dist Δ = (m_dist : ℝ) / (Real.exp Δ + (m_dist : ℝ)) := by
  unfold softmaxDistractorMass
  rw [softmaxTargetProb_eq_exp_div]
  have hpos : 0 < Real.exp Δ + (m_dist : ℝ) :=
    add_pos_of_pos_of_nonneg (Real.exp_pos Δ) (Nat.cast_nonneg m_dist)
  rw [one_sub_div (ne_of_gt hpos)]
  ring

/-- **Softmax vs. Linear Attention Distractor Suppression & Target Amplification
(`softmax_vs_linear_attention_distractor_suppression`)**:
For any `Δ > 0` (in particular `Δ ≥ 1`) and `m_dist ≥ 1`, because `Δ < Real.exp Δ`
(by `Real.add_one_le_exp`), nonlinear Softmax Attention achieves:
1. Strictly lower distractor leakage than Linear Attention:
   `softmaxDistractorMass m_dist Δ < linearAttentionDistractorMass m_dist Δ`, and
2. Strictly higher target retrieval probability than Linear Attention:
   `linearAttentionTargetWeight m_dist Δ < softmaxTargetProb m_dist Δ`. -/
theorem softmax_vs_linear_attention_distractor_suppression {m_dist : ℕ} {Δ : ℝ}
    (hm : 1 ≤ m_dist) (hΔ : 0 < Δ) :
    softmaxDistractorMass m_dist Δ < linearAttentionDistractorMass m_dist Δ ∧
    linearAttentionTargetWeight m_dist Δ < softmaxTargetProb m_dist Δ := by
  have hm_pos : (0 : ℝ) < (m_dist : ℝ) := by
    have : (1 : ℝ) ≤ (m_dist : ℝ) := by exact_mod_cast hm
    linarith
  have hlt_exp : Δ < Real.exp Δ := self_lt_exp Δ
  have hlin_denom_pos : 0 < Δ + (m_dist : ℝ) := add_pos hΔ hm_pos
  have hsoft_denom_pos : 0 < Real.exp Δ + (m_dist : ℝ) := add_pos (Real.exp_pos Δ) hm_pos
  have hdist_lt : softmaxDistractorMass m_dist Δ < linearAttentionDistractorMass m_dist Δ := by
    rw [softmaxDistractorMass_eq_div_exp, linearAttentionDistractorMass]
    rw [div_lt_div_iff₀ hsoft_denom_pos hlin_denom_pos]
    nlinarith
  have htarget_gt : linearAttentionTargetWeight m_dist Δ < softmaxTargetProb m_dist Δ := by
    have hsum_lin := linearAttention_weights_sum_one m_dist Δ (ne_of_gt hlin_denom_pos)
    unfold softmaxDistractorMass at hdist_lt
    linarith
  exact ⟨hdist_lt, htarget_gt⟩

/-- **Logarithmic Margin Sufficiency for Softmax Long-Context Retrieval
(`softmax_logarithmic_margin_sufficiency`)**:
For any context distractor count `m_dist ≥ 1` and tolerance `ε > 0`, if the logit margin satisfies
`Δ ≥ Real.log ((m_dist : ℝ) / ε)`, then Softmax Attention suppresses total distractor leakage
below `ε`:
`softmaxDistractorMass m_dist Δ ≤ ε`. -/
theorem softmax_logarithmic_margin_sufficiency {m_dist : ℕ} {Δ ε : ℝ}
    (hm : 1 ≤ m_dist) (hε : 0 < ε)
    (hΔ : Real.log ((m_dist : ℝ) / ε) ≤ Δ) :
    softmaxDistractorMass m_dist Δ ≤ ε := by
  have hm_pos : (0 : ℝ) < (m_dist : ℝ) := by
    have : (1 : ℝ) ≤ (m_dist : ℝ) := by exact_mod_cast hm
    linarith
  have hratio_pos : 0 < (m_dist : ℝ) / ε := div_pos hm_pos hε
  have hneg_le : -Δ ≤ -Real.log ((m_dist : ℝ) / ε) := neg_le_neg hΔ
  have hexp_le : Real.exp (-Δ) ≤ Real.exp (-Real.log ((m_dist : ℝ) / ε)) :=
    Real.exp_le_exp.mpr hneg_le
  have hexp_eq : Real.exp (-Real.log ((m_dist : ℝ) / ε)) = ε / (m_dist : ℝ) := by
    rw [Real.exp_neg, Real.exp_log hratio_pos, inv_div]
  rw [hexp_eq] at hexp_le
  calc
    softmaxDistractorMass m_dist Δ ≤ (m_dist : ℝ) * Real.exp (-Δ) :=
      softmaxDistractorMass_le_exp_neg_margin m_dist Δ
    _ ≤ (m_dist : ℝ) * (ε / (m_dist : ℝ)) :=
      mul_le_mul_of_nonneg_left hexp_le hm_pos.le
    _ = ε := mul_div_cancel₀ ε (ne_of_gt hm_pos)

/-- **Linear Margin Necessity & Sufficiency for Linear Attention (`linear_attention_requires_linear_margin`)**:
For any `m_dist ≥ 1`, `Δ > 0`, and `ε ∈ (0, 1)`, Linear Attention achieves distractor leakage
`linearAttentionDistractorMass m_dist Δ ≤ ε` if and only if the margin grows **linearly** in the
context distractor count `m_dist`:
`linearAttentionDistractorMass m_dist Δ ≤ ε ↔ (m_dist : ℝ) * (1 - ε) / ε ≤ Δ`. -/
theorem linear_attention_requires_linear_margin {m_dist : ℕ} {Δ ε : ℝ}
    (hm : 1 ≤ m_dist) (hΔ : 0 < Δ) (hε_pos : 0 < ε) :
    linearAttentionDistractorMass m_dist Δ ≤ ε ↔
      (m_dist : ℝ) * (1 - ε) / ε ≤ Δ := by
  unfold linearAttentionDistractorMass
  have hm_pos : (0 : ℝ) < (m_dist : ℝ) := by
    have : (1 : ℝ) ≤ (m_dist : ℝ) := by exact_mod_cast hm
    linarith
  have hdenom_pos : 0 < Δ + (m_dist : ℝ) := add_pos hΔ hm_pos
  rw [div_le_iff₀ hdenom_pos, div_le_iff₀ hε_pos]
  constructor <;> intro h <;> linarith

/-- **Long-Context Margin Scaling Law (`long_context_margin_scaling`)**:
Combined long-context scaling comparison for `m_dist ≥ 1` and `ε ∈ (0, 1)`:
1. Softmax Attention achieves `softmaxDistractorMass m_dist Δ ≤ ε` whenever
   `Δ ≥ Real.log ((m_dist : ℝ) / ε)` (logarithmic scaling in `m_dist`), whereas
2. Linear Attention with `Δ > 0` satisfies `linearAttentionDistractorMass m_dist Δ ≤ ε`
   if and only if `Δ ≥ (m_dist : ℝ) * (1 - ε) / ε` (linear scaling in `m_dist`). -/
theorem long_context_margin_scaling {m_dist : ℕ} {Δ ε : ℝ}
    (hm : 1 ≤ m_dist) (hΔ : 0 < Δ) (hε_pos : 0 < ε) :
    (Real.log ((m_dist : ℝ) / ε) ≤ Δ → softmaxDistractorMass m_dist Δ ≤ ε) ∧
    (linearAttentionDistractorMass m_dist Δ ≤ ε ↔ (m_dist : ℝ) * (1 - ε) / ε ≤ Δ) :=
  ⟨softmax_logarithmic_margin_sufficiency hm hε_pos,
   linear_attention_requires_linear_margin hm hΔ hε_pos⟩

/-! ## Section 4: End-to-End Trained Induction Head Retrieval Error -/

/-- Canonical convex combination output of an attention head assigning probability `p_target`
to the target value `v_target : E` and complementary probability `1 - p_target` to the aggregate
distractor value `v_dist : E`. -/
def inductionHeadOutput (p_target : ℝ) (v_target v_dist : E) : E :=
  p_target • v_target + (1 - p_target) • v_dist

/-- Exact error identity for `inductionHeadOutput`:
`inductionHeadOutput p_target v_target v_dist - v_target = (1 - p_target) • (v_dist - v_target)`. -/
theorem inductionHeadOutput_sub_target (p_target : ℝ) (v_target v_dist : E) :
    inductionHeadOutput p_target v_target v_dist - v_target =
      (1 - p_target) • (v_dist - v_target) := by
  unfold inductionHeadOutput
  rw [smul_sub, sub_smul, sub_smul, one_smul, one_smul]
  abel

/-- **End-to-End Trained Induction Head Retrieval Error (`trained_induction_head_retrieval_error`)**:
After `T : ℕ` steps along `logitMarginTrajectory Δ₀ δ_step T`, for any target value `v_target : E`
and aggregate distractor value `v_dist : E` with `‖v_dist - v_target‖ ≤ R_V` (`R_V ≥ 0`), the
trained induction head output `v_out(T) = inductionHeadOutput (softmaxTargetProb m_dist Δ_T) v_target v_dist`
satisfies both the exact distractor-mass bound and the exponential decay bound:
1. `‖v_out(T) - v_target‖ ≤ R_V * softmaxDistractorMass m_dist (logitMarginTrajectory Δ₀ δ_step T)`, and
2. `R_V * softmaxDistractorMass m_dist (logitMarginTrajectory Δ₀ δ_step T) ≤ R_V * (m_dist : ℝ) * Real.exp (-Δ₀ - (T : ℝ) * δ_step)`. -/
theorem trained_induction_head_retrieval_error
    (m_dist : ℕ) (Δ₀ δ_step R_V : ℝ) (v_target v_dist : E) (T : ℕ)
    (hR_nonneg : 0 ≤ R_V)
    (hR : ‖v_dist - v_target‖ ≤ R_V) :
    let Δ_T := logitMarginTrajectory Δ₀ δ_step T
    let v_out := inductionHeadOutput (softmaxTargetProb m_dist Δ_T) v_target v_dist
    ‖v_out - v_target‖ ≤ R_V * softmaxDistractorMass m_dist Δ_T ∧
    R_V * softmaxDistractorMass m_dist Δ_T ≤
      R_V * (m_dist : ℝ) * Real.exp (-Δ₀ - (T : ℝ) * δ_step) := by
  intro Δ_T v_out
  have hsub : v_out - v_target = softmaxDistractorMass m_dist Δ_T • (v_dist - v_target) := by
    unfold v_out softmaxDistractorMass
    exact inductionHeadOutput_sub_target (softmaxTargetProb m_dist Δ_T) v_target v_dist
  have hmass_nonneg : 0 ≤ softmaxDistractorMass m_dist Δ_T :=
    softmaxDistractorMass_nonneg m_dist Δ_T
  have hnorm_le : ‖v_out - v_target‖ ≤ R_V * softmaxDistractorMass m_dist Δ_T := by
    rw [hsub, norm_smul, Real.norm_eq_abs, abs_of_nonneg hmass_nonneg, mul_comm]
    exact mul_le_mul_of_nonneg_right hR hmass_nonneg
  have hsparse := (induction_head_exponential_sparsification m_dist Δ₀ δ_step T).1
  have hexp_le :
      R_V * softmaxDistractorMass m_dist Δ_T ≤
        R_V * (m_dist : ℝ) * Real.exp (-Δ₀ - (T : ℝ) * δ_step) := by
    calc
      R_V * softmaxDistractorMass m_dist Δ_T ≤
          R_V * ((m_dist : ℝ) * Real.exp (-Δ₀ - (T : ℝ) * δ_step)) :=
        mul_le_mul_of_nonneg_left hsparse hR_nonneg
      _ = R_V * (m_dist : ℝ) * Real.exp (-Δ₀ - (T : ℝ) * δ_step) := by ring
  exact ⟨hnorm_le, hexp_le⟩

/-- **Full-Context `Attention.rowOutput` Induction Head Retrieval Error
(`trained_induction_head_rowOutput_retrieval_error`)**:
In a general context of length `1 + m_dist` where unmasked target key `i_star : Fin (1 + m_dist)`
has logit margin at least `logitMarginTrajectory Δ₀ δ_step T` over all other unmasked keys
`j ≠ i_star` and all unmasked values satisfy `‖values j - values i_star‖ ≤ R_V` (`R_V ≥ 0`),
the actual `TorchLean` attention output `Attention.rowOutput scores_T mask values` satisfies:
`‖Attention.rowOutput scores_T mask values - values i_star‖ ≤ R_V * (m_dist : ℝ) * Real.exp (-Δ₀ - (T : ℝ) * δ_step)`. -/
theorem trained_induction_head_rowOutput_retrieval_error {m_dist : ℕ}
    (Δ₀ δ_step R_V : ℝ)
    (scores_T : Tensor ℝ [1 + m_dist])
    (mask : Tensor Bool [1 + m_dist])
    (values : Fin (1 + m_dist) → E)
    (i_star : Fin (1 + m_dist))
    (T : ℕ)
    (hi_star : mask.getScalar i_star = true)
    (hmargin : ∀ j ≠ i_star, mask.getScalar j = true →
      scores_T.getScalar j ≤ scores_T.getScalar i_star - logitMarginTrajectory Δ₀ δ_step T)
    (hR_nonneg : 0 ≤ R_V)
    (hR : ∀ j, mask.getScalar j = true → ‖values j - values i_star‖ ≤ R_V) :
    ‖Attention.rowOutput scores_T mask values - values i_star‖ ≤
      R_V * (m_dist : ℝ) * Real.exp (-Δ₀ - (T : ℝ) * δ_step) := by
  have hstab := (AttentionLogitStability.logit_margin_output_stability
    scores_T mask values i_star (logitMarginTrajectory Δ₀ δ_step T) R_V
    hi_star hmargin hR_nonneg hR).1
  have hsub : (1 + m_dist - 1 : ℕ) = m_dist := by omega
  have hneg : -(logitMarginTrajectory Δ₀ δ_step T) = -Δ₀ - (T : ℝ) * δ_step := by
    unfold logitMarginTrajectory
    ring
  rw [hsub, hneg] at hstab
  linarith

end OpenTheory.InductionHeadDynamics
