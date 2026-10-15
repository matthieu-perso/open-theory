import OpenTheory.Generalization.AutoregressiveOOD

noncomputable section

open OpenTheory
open OpenTheory.ConvolutionalGeneralization
open OpenTheory.SpectralSparseCircuit
open OpenTheory.AutoregressiveOOD

-- Section 1: Autoregressive Rollout & Long-Horizon Error Compounding
#check @rolloutErrorStep
#check @autoregressiveRolloutError
#check @autoregressiveRolloutError_zero
#check @autoregressiveRolloutError_succ
#check @autoregressiveRolloutError_one
#check @autoregressiveRolloutError_nonneg
#check @autoregressiveRolloutError_unit_lip
#check @autoregressiveRolloutError_closed_form
#check @autoregressiveRolloutError_closed_form_contractive
#check @cot_contractive_rollout_horizon_free_bound
#check @unstable_rollout_strict_mono_and_exceeds_contractive

-- Section 2: Chain-of-Thought (CoT) Decomposition vs. Direct 1-Step Prediction
#check @directPredictionRisk
#check @cotRolloutRisk
#check @cotContractiveRolloutRisk
#check @cotRolloutRisk_eq_autoregressiveRolloutError
#check @cot_strictly_dominates_direct_prediction
#check @cot_contractive_dominates_direct_for_all_horizons

-- Section 3: Out-of-Distribution (OOD) Compositional Generalization & Spurious Shift Robustness
#check @oodExcessRisk
#check @oodExcessRisk_zero_spurious
#check @oodExcessRisk_eq_archTotalExcessRisk
#check @sesc_zero_ood_shift_penalty
#check @sesc_ood_strictly_dominates_dense_mlp
#check @dense_mlp_ood_risk_floor_as_n_infty

-- Smoke test 1: Contractive CoT rollout (Λ = 1/2, ε_token = 1) vs. Unstable rollout (Λ = 2, ε_token = 1).
-- At H = 4, contractive rollout error is 15/8 ≤ 2 (horizon-free bound), while unstable rollout error is 15 > 4.
example (H : ℕ) :
    autoregressiveRolloutError (1 / 2) 1 H ≤ 2 ∧
      autoregressiveRolloutError (1 / 2) 1 4 = 15 / 8 ∧
      (4 : ℝ) * 1 < autoregressiveRolloutError 2 1 4 ∧
      autoregressiveRolloutError 2 1 4 = 15 := by
  have hcontr := (cot_contractive_rollout_horizon_free_bound
    (Λ := 1 / 2) (ε_token := 1) (by norm_num) (by norm_num) (by norm_num) H).2
  have hunst := (unstable_rollout_strict_mono_and_exceeds_contractive
    (Λ := 2) (ε_token := 1) (by norm_num) (by norm_num)).2.2 4 (by decide)
  refine ⟨?_, ?_, hunst, ?_⟩
  · norm_num at hcontr
    exact hcontr
  · norm_num [autoregressiveRolloutError, rolloutErrorStep]
  · norm_num [autoregressiveRolloutError, rolloutErrorStep]

-- Smoke test 2: 4-step Chain-of-Thought (H = 4, s_step = 2) strictly dominates direct 1-step prediction
-- (s_total = 8) on vocabulary/feature dimension d = 2, achieving exact risk ratio 4 / 2^6 = 1 / 16 < 1.
example :
    cotRolloutRisk 4 1 2 2 100 < directPredictionRisk 1 2 8 100 ∧
      cotRolloutRisk 4 1 2 2 100 / directPredictionRisk 1 2 8 100 = 1 / 16 := by
  have hcot := cot_strictly_dominates_direct_prediction
    (H := 4) (s_step := 2) (s_total := 8) (c := 1) (d := 2) (n := 100)
    (by norm_num) (by norm_num) (by norm_num) (by decide) (by norm_num)
  refine ⟨hcot.1, ?_⟩
  rw [hcot.2.1]
  norm_num

-- Smoke test 3: OOD generalization dominance of SESC (p_SESC = 6, w_spurious = 0) over a dense MLP
-- (p_MLP = 96, w_spurious = 1/2) under OOD shift energy Δ_OOD_sq = 4 at sample size n = 100.
example :
    oodExcessRisk 0 6 1 100 0 4 < oodExcessRisk 0 96 1 100 (1 / 2) 4 ∧
      oodExcessRisk 0 6 1 100 0 4 < (1 / 2) * 4 ∧
      (1 / 2) * 4 ≤ oodExcessRisk 0 96 1 100 (1 / 2) 4 := by
  have hood := dense_mlp_ood_risk_floor_as_n_infty
    (p_SESC := 6) (p_MLP := 96) (ε_struct := 0) (c_gen := 1) (n := 100)
    (w_spurious := 1 / 2) (Δ_OOD_sq := 4)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  exact ⟨hood.2.2, hood.2.1, hood.1⟩
