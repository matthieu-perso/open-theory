import OpenTheory.Architectures.InductionHeadDynamics

noncomputable section

open TorchLean OpenTheory OpenTheory.Attention
open OpenTheory.AttentionLogitStability OpenTheory.InContextRetrieval
open OpenTheory.InductionHeadDynamics
open scoped BigOperators

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

-- #check all definitions and theorems in OpenTheory.InductionHeadDynamics
#check @softmaxTargetProb
#check @softmaxDistractorMass
#check @softmaxDenom_pos
#check @softmaxTargetProb_pos
#check @softmaxTargetProb_nonneg
#check @softmaxTargetProb_le_one
#check @softmaxTargetProb_lt_one
#check @softmaxDistractorMass_eq
#check @softmaxDistractorMass_nonneg
#check @softmaxDistractorMass_pos
#check @softmaxDistractorMass_lt_one
#check @softmaxDistractorMass_le_exp_neg_margin
#check @softmaxTargetProb_strict_mono_in_margin
#check @softmaxTargetProb_mono_in_margin
#check @softmaxDistractorMass_strict_anti_in_margin
#check @logitMarginLogLossStep
#check @logitMarginSqLossStep
#check @logitMarginTrajectory
#check @logitMarginTrajectory_zero
#check @logitMarginTrajectory_succ
#check @logitMarginLogLossStep_strict_increase
#check @logitMarginSqLossStep_strict_increase
#check @softmaxTargetProb_strict_increase_after_logLossStep
#check @induction_head_exponential_sparsification
#check @linearAttentionTargetWeight
#check @linearAttentionDistractorMass
#check @linearAttention_weights_sum_one
#check @self_lt_exp
#check @softmaxTargetProb_eq_exp_div
#check @softmaxDistractorMass_eq_div_exp
#check @softmax_vs_linear_attention_distractor_suppression
#check @softmax_logarithmic_margin_sufficiency
#check @linear_attention_requires_linear_margin
#check @long_context_margin_scaling
#check @inductionHeadOutput
#check @inductionHeadOutput_sub_target
#check @trained_induction_head_retrieval_error
#check @trained_induction_head_rowOutput_retrieval_error

-- Concrete numerical example 1: At zero margin Δ = 0 with m_dist = 3 distractors (context length 4),
-- softmaxTargetProb 3 0 = 1/4 and softmaxDistractorMass 3 0 = 3/4.
example :
    softmaxTargetProb 3 0 = 1 / 4 ∧
    softmaxDistractorMass 3 0 = 3 / 4 := by
  norm_num [softmaxTargetProb, softmaxDistractorMass]

-- Concrete numerical example 2: At Δ = 0 with m_dist = 1 distractor, η = 2, g_ret = 1,
-- one step of logitMarginLogLossStep increases the margin from 0 to 1, and
-- logitMarginTrajectory 0 (1 / 2) 10 = 5.
example :
    logitMarginLogLossStep 2 1 1 0 = 1 ∧
    logitMarginSqLossStep 4 1 1 0 = 1 ∧
    logitMarginTrajectory 0 (1 / 2) 10 = 5 := by
  norm_num [logitMarginLogLossStep, logitMarginSqLossStep, logitMarginTrajectory,
    softmaxDistractorMass, softmaxTargetProb]

-- Concrete numerical example 3: Linear attention weights at Δ = 4 and m_dist = 16 (context length 17)
-- give target weight 4/20 = 1/5 and distractor leakage 16/20 = 4/5, whereas to achieve ε = 1/10
-- distractor leakage with m_dist = 16 distractors, linear attention requires margin Δ ≥ 144.
example :
    linearAttentionTargetWeight 16 4 = 1 / 5 ∧
    linearAttentionDistractorMass 16 4 = 4 / 5 ∧
    ((16 : ℝ) * (1 - 1 / 10) / (1 / 10) = 144) := by
  norm_num [linearAttentionTargetWeight, linearAttentionDistractorMass]

-- Concrete numerical example 4: Positivity and strict bounds for softmax probabilities and
-- linear-margin trajectory.
example (Δ : ℝ) :
    0 < softmaxTargetProb 5 Δ ∧
    0 < 1 + (5 : ℝ) * Real.exp (-Δ) := by
  exact ⟨softmaxTargetProb_pos 5 Δ, by positivity⟩
