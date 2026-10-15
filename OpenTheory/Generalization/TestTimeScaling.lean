import OpenTheory.Generalization.ComputeOptimalSynthesis
import OpenTheory.Generalization.AutoregressiveOOD
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.Real.Sqrt
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Ring

/-!
# Joint Pre-Training vs. Test-Time Inference Compute Scaling & Best-of-`K` Verification Frontier

This module formalizes **Bridge 2: Joint Pre-Training vs. Test-Time Inference Compute Scaling &
Best-of-`K` Verification Frontier (The "Reasoning Model" Frontier)** in `OpenTheory.TestTimeScaling`:

1. **Best-of-`K` Verifier / Parallel Reasoning Pass@`K` Failure Exponential Decay
   (`bestOfKFailureProb`, `bestOfKSuccessProb`, `bestOfKFailureProb_nonneg`,
   `bestOfKFailureProb_le_one`, `bestOfKFailureProb_strict_anti_in_K`,
   `bestOfKSuccessProb_strict_mono_in_K`, `bestOfK_exponential_bound`,
   `bestOfK_sample_sufficiency`)**:
   - Models `K` independent reasoning rollouts scored by an exact or calibrated verifier, where a
     single trajectory has residual failure probability `p_fail ∈ [0, 1]`.
   - Proves strict monotonic decay of `bestOfKFailureProb p_fail K = p_fail ^ K` in `K` for
     `0 < p_fail < 1`, the exponential upper bound `bestOfKFailureProb p_fail K ≤ exp(-K * (1 - p_fail))`
     via `Real.add_one_le_exp`, and the sample-sufficiency threshold `K * (1 - p_fail) ≥ log(1 / ε)`
     guaranteeing `bestOfKFailureProb p_fail K ≤ ε`.

2. **Variational Tradeoff Between Model Size `N` and Test-Time Reasoning Horizon `T_inf` at Fixed Inference Budget
   (`inferenceScalingRisk`, `optimalInferenceModelSize`, `optimalReasoningBudget`,
   `paretoInferenceRisk`, `optimalInferenceModelSize_mul_optimalReasoningBudget`,
   `optimalInferenceModelSize_div_optimalReasoningBudget`, `inferenceScalingRisk_at_optimal_eq_pareto`,
   `test_time_scaling_completion_of_squares`, `test_time_scaling_variational_optimality`)**:
   - Under per-query test-time FLOP budget `C_inf = N * T_inf > 0`, decomposes deployment risk into
     `inferenceScalingRisk R_inf α_model β_reason N T_inf = R_inf + α_model / N + β_reason / T_inf`.
   - Proves the exact completion-of-squares variational identity:
     `inferenceScalingRisk R_inf α_model β_reason N T_inf - paretoInferenceRisk R_inf α_model β_reason C_inf = (√(α_model * T_inf) - √(β_reason * N))^2 / C_inf ≥ 0`,
     and establishes `optimalInferenceModelSize * optimalReasoningBudget = C_inf` and
     `optimalInferenceModelSize / optimalReasoningBudget = α_model / β_reason`.

3. **When a Smaller Reasoning Model Strictly Beats an `M×` Larger 1-Step Model
   (`small_reasoning_model_beats_giant_one_step_at_equal_inference_flops`,
   `cot_best_of_K_joint_error_bound`)**:
   - Compares a giant 1-step model of size `M * N_small` (`T_inf = 1`) against a small reasoning
     model of size `N_small` with test-time budget `T_inf = M` at **identical inference FLOPs**
     `C_inf = M * N_small`.
   - Proves the exact iff criterion: the small reasoning model strictly outperforms the `M×` larger
     1-step model if and only if `α_model / N_small < β_reason`.
   - Combines `H`-step contractive Chain-of-Thought rollout (`autoregressiveRolloutError Λ ε_token H ≤ ε_token / (1 - Λ)`)
     with Best-of-`K` verification to bound joint end-to-end failure by `(ε_token / (1 - Λ)) ^ K`.

4. **Total Lifecycle Compute Synthesis: Pre-Training FLOPs + Deployment Inference FLOPs over `Q` Queries
   (`totalLifecycleCompute`, `lifecycleFeasibleModelSize`, `lifecycleTokenToParamRatio`,
   `deployment_queries_shift_optimal_model_size_down`)**:
   - Models total lifecycle compute `C_total = N * D + Q * N * T_inf = N * (D + Q * T_inf)` across
     pre-training (`D` tokens) and deployment (`Q` queries at reasoning horizon `T_inf`).
   - Proves that at fixed `D > 0` and `C_total > 0`, increasing query volume `Q₂ > Q₁` or reasoning
     horizon `T₂ > T₁` strictly reduces the feasible model size `N = C_total / (D + Q * T_inf)` and
     strictly increases the token-to-parameter ratio `D / N` ("over-training relative to Chinchilla").
-/

noncomputable section

namespace OpenTheory.TestTimeScaling

open OpenTheory
open OpenTheory.ComputeOptimalSynthesis
open OpenTheory.AutoregressiveOOD

/-! ## 1. Best-of-`K` Verifier / Parallel Reasoning Pass@`K` Failure Exponential Decay -/

/-- Probability (or normalized residual failure rate) that all `K` independent reasoning rollouts
fail when each single trajectory has failure probability `p_fail`:
`bestOfKFailureProb p_fail K = p_fail ^ K`. -/
def bestOfKFailureProb (p_fail : ℝ) (K : ℕ) : ℝ :=
  p_fail ^ K

/-- Best-of-`K` (Pass@`K` with verifier selection) success probability:
`bestOfKSuccessProb p_fail K = 1 - p_fail ^ K`. -/
def bestOfKSuccessProb (p_fail : ℝ) (K : ℕ) : ℝ :=
  1 - p_fail ^ K

@[simp]
theorem bestOfKFailureProb_Add_bestOfKSuccessProb (p_fail : ℝ) (K : ℕ) :
    bestOfKFailureProb p_fail K + bestOfKSuccessProb p_fail K = 1 := by
  unfold bestOfKFailureProb bestOfKSuccessProb
  ring

@[simp]
theorem bestOfKFailureProb_zero (p_fail : ℝ) :
    bestOfKFailureProb p_fail 0 = 1 :=
  rfl

@[simp]
theorem bestOfKFailureProb_one (p_fail : ℝ) :
    bestOfKFailureProb p_fail 1 = p_fail :=
  pow_one p_fail

/-- Nonnegativity of `bestOfKFailureProb p_fail K` for `0 ≤ p_fail`. -/
theorem bestOfKFailureProb_nonneg {p_fail : ℝ} (hp : 0 ≤ p_fail) (K : ℕ) :
    0 ≤ bestOfKFailureProb p_fail K :=
  pow_nonneg hp K

/-- Upper bound `bestOfKFailureProb p_fail K ≤ 1` for `0 ≤ p_fail ≤ 1`. -/
theorem bestOfKFailureProb_le_one {p_fail : ℝ} (hp0 : 0 ≤ p_fail) (hp1 : p_fail ≤ 1) (K : ℕ) :
    bestOfKFailureProb p_fail K ≤ 1 :=
  pow_le_one₀ hp0 hp1

/-- Bounds `0 ≤ bestOfKSuccessProb p_fail K ≤ 1` for `0 ≤ p_fail ≤ 1`. -/
theorem bestOfKSuccessProb_bounds {p_fail : ℝ} (hp0 : 0 ≤ p_fail) (hp1 : p_fail ≤ 1) (K : ℕ) :
    0 ≤ bestOfKSuccessProb p_fail K ∧ bestOfKSuccessProb p_fail K ≤ 1 := by
  unfold bestOfKSuccessProb
  have h0 : 0 ≤ p_fail ^ K := pow_nonneg hp0 K
  have h1 : p_fail ^ K ≤ 1 := pow_le_one₀ hp0 hp1
  exact ⟨by linarith, by linarith⟩

/-- **Strict Decay of Best-of-`K` Failure Probability (`bestOfKFailureProb_strict_anti_in_K`)**:
When `0 < p_fail < 1` and `K₁ < K₂`, generating more candidate reasoning trajectories strictly
reduces the Best-of-`K` failure probability and strictly increases the Best-of-`K` success rate:
`bestOfKFailureProb p_fail K₂ < bestOfKFailureProb p_fail K₁` and
`bestOfKSuccessProb p_fail K₁ < bestOfKSuccessProb p_fail K₂`. -/
theorem bestOfKFailureProb_strict_anti_in_K {p_fail : ℝ} {K₁ K₂ : ℕ}
    (hp0 : 0 < p_fail) (hp1 : p_fail < 1) (hK : K₁ < K₂) :
    bestOfKFailureProb p_fail K₂ < bestOfKFailureProb p_fail K₁ ∧
      bestOfKSuccessProb p_fail K₁ < bestOfKSuccessProb p_fail K₂ := by
  unfold bestOfKFailureProb bestOfKSuccessProb
  have hpow : p_fail ^ K₂ < p_fail ^ K₁ := pow_lt_pow_right_of_lt_one₀ hp0 hp1 hK
  exact ⟨hpow, by linarith⟩

/-- Strict monotonicity of `bestOfKSuccessProb` in `K` for `0 < p_fail < 1`. -/
theorem bestOfKSuccessProb_strict_mono_in_K {p_fail : ℝ} {K₁ K₂ : ℕ}
    (hp0 : 0 < p_fail) (hp1 : p_fail < 1) (hK : K₁ < K₂) :
    bestOfKSuccessProb p_fail K₁ < bestOfKSuccessProb p_fail K₂ :=
  (bestOfKFailureProb_strict_anti_in_K hp0 hp1 hK).2

/-- **Exponential Upper Bound on Best-of-`K` Failure (`bestOfK_exponential_bound`)**:
For any `0 ≤ p_fail ≤ 1` and any `K : ℕ`, using `1 - x ≤ exp(-x)` for `x = 1 - p_fail`
(`p_fail ≤ exp(-(1 - p_fail))`), the `K`-sample failure rate decays exponentially in `K`:
`bestOfKFailureProb p_fail K ≤ Real.exp (-(K : ℝ) * (1 - p_fail))`. -/
theorem bestOfK_exponential_bound {p_fail : ℝ} (hp0 : 0 ≤ p_fail) (_hp1 : p_fail ≤ 1) (K : ℕ) :
    bestOfKFailureProb p_fail K ≤ Real.exp (-(K : ℝ) * (1 - p_fail)) := by
  unfold bestOfKFailureProb
  have h1 : p_fail ≤ Real.exp (-(1 - p_fail)) := by
    have hexp := Real.add_one_le_exp (-(1 - p_fail))
    linarith
  calc
    p_fail ^ K ≤ (Real.exp (-(1 - p_fail))) ^ K :=
      pow_le_pow_left₀ hp0 h1 K
    _ = Real.exp ((K : ℝ) * (-(1 - p_fail))) := by
      rw [← Real.exp_nat_mul]
    _ = Real.exp (-(K : ℝ) * (1 - p_fail)) := by
      congr 1
      ring

/-- **Best-of-`K` Sample Complexity Sufficiency (`bestOfK_sample_sufficiency`)**:
For any target failure tolerance `ε > 0` and single-rollout failure probability `0 ≤ p_fail ≤ 1`,
if the test-time sample count `K` satisfies `(K : ℝ) * (1 - p_fail) ≥ Real.log (1 / ε)`, then:
`bestOfKFailureProb p_fail K ≤ ε` and `1 - ε ≤ bestOfKSuccessProb p_fail K`. -/
theorem bestOfK_sample_sufficiency {p_fail ε : ℝ} {K : ℕ}
    (hp0 : 0 ≤ p_fail) (hp1 : p_fail ≤ 1) (hε : 0 < ε)
    (hK : (K : ℝ) * (1 - p_fail) ≥ Real.log (1 / ε)) :
    bestOfKFailureProb p_fail K ≤ ε ∧ 1 - ε ≤ bestOfKSuccessProb p_fail K := by
  have hexp_bound := bestOfK_exponential_bound hp0 hp1 K
  have hneg_le : -(K : ℝ) * (1 - p_fail) ≤ -Real.log (1 / ε) := by linarith
  have hexp_le : Real.exp (-(K : ℝ) * (1 - p_fail)) ≤ Real.exp (-Real.log (1 / ε)) :=
    Real.exp_le_exp.mpr hneg_le
  have hinv_pos : 0 < 1 / ε := one_div_pos.mpr hε
  have hexp_log : Real.exp (-Real.log (1 / ε)) = ε := by
    rw [Real.exp_neg, Real.exp_log hinv_pos, one_div, inv_inv]
  have hfail : bestOfKFailureProb p_fail K ≤ ε := by
    linarith [hexp_bound, hexp_le, hexp_log]
  refine ⟨hfail, ?_⟩
  unfold bestOfKSuccessProb
  unfold bestOfKFailureProb at hfail
  linarith

/-! ## 2. Variational Tradeoff Between Model Size `N` and Test-Time Reasoning Horizon `T_inf` -/

/-- Joint Model-vs-Inference Risk under per-query parameter count `N` and test-time compute
multiplier `T_inf` (reasoning tokens or verification rollouts per query):
`inferenceScalingRisk R_inf α_model β_reason N T_inf = R_inf + α_model / N + β_reason / T_inf`. -/
def inferenceScalingRisk (R_inf α_model β_reason N T_inf : ℝ) : ℝ :=
  R_inf + α_model / N + β_reason / T_inf

/-- Optimal inference-aware model size `N*(C_inf) = √(α_model * C_inf / β_reason)` minimizing
`inferenceScalingRisk` under fixed per-query inference FLOP budget `N * T_inf = C_inf`. -/
def optimalInferenceModelSize (α_model β_reason C_inf : ℝ) : ℝ :=
  Real.sqrt (α_model * C_inf / β_reason)

/-- Optimal test-time reasoning budget `T_inf*(C_inf) = √(β_reason * C_inf / α_model)` minimizing
`inferenceScalingRisk` under fixed per-query inference FLOP budget `N * T_inf = C_inf`. -/
def optimalReasoningBudget (α_model β_reason C_inf : ℝ) : ℝ :=
  Real.sqrt (β_reason * C_inf / α_model)

/-- Pareto-optimal test-time inference risk frontier at per-query FLOP budget `C_inf`:
`paretoInferenceRisk R_inf α_model β_reason C_inf = R_inf + 2 * √(α_model * β_reason) / √C_inf`. -/
def paretoInferenceRisk (R_inf α_model β_reason C_inf : ℝ) : ℝ :=
  R_inf + 2 * Real.sqrt (α_model * β_reason) / Real.sqrt C_inf

/-- Bridge to `ComputeOptimalSynthesis`: `inferenceScalingRisk`, `optimalInferenceModelSize`,
`optimalReasoningBudget`, and `paretoInferenceRisk` are structurally isomorphic to the Chinchilla
pre-training variational synthesis (`scalingRisk`, `optimalModelCapacity`, `optimalDataBudget`,
`paretoComputeRisk`). -/
@[simp]
theorem inferenceScalingRisk_eq_scalingRisk (R_inf α_model β_reason N T_inf : ℝ) :
    inferenceScalingRisk R_inf α_model β_reason N T_inf =
      scalingRisk R_inf α_model β_reason N T_inf :=
  rfl

/-- Exact product identity: `optimalInferenceModelSize α_model β_reason C_inf * optimalReasoningBudget α_model β_reason C_inf = C_inf`. -/
theorem optimalInferenceModelSize_mul_optimalReasoningBudget {α_model β_reason C_inf : ℝ}
    (hα : 0 < α_model) (hβ : 0 < β_reason) (hC : 0 < C_inf) :
    optimalInferenceModelSize α_model β_reason C_inf *
      optimalReasoningBudget α_model β_reason C_inf = C_inf :=
  (optimalCapacity_mul_optimalData_eq_compute hα hβ hC).1

/-- Exact ratio identity: `optimalInferenceModelSize α_model β_reason C_inf / optimalReasoningBudget α_model β_reason C_inf = α_model / β_reason`. -/
theorem optimalInferenceModelSize_div_optimalReasoningBudget {α_model β_reason C_inf : ℝ}
    (hα : 0 < α_model) (hβ : 0 < β_reason) (hC : 0 < C_inf) :
    optimalInferenceModelSize α_model β_reason C_inf /
      optimalReasoningBudget α_model β_reason C_inf = α_model / β_reason :=
  (optimalCapacity_mul_optimalData_eq_compute hα hβ hC).2

/-- Evaluating `inferenceScalingRisk` at `(optimalInferenceModelSize, optimalReasoningBudget)`
achieves the exact `paretoInferenceRisk` frontier. -/
theorem inferenceScalingRisk_at_optimal_eq_pareto (R_inf : ℝ) {α_model β_reason C_inf : ℝ}
    (hα : 0 < α_model) (hβ : 0 < β_reason) (hC : 0 < C_inf) :
    inferenceScalingRisk R_inf α_model β_reason
      (optimalInferenceModelSize α_model β_reason C_inf)
      (optimalReasoningBudget α_model β_reason C_inf) =
      paretoInferenceRisk R_inf α_model β_reason C_inf :=
  scalingRisk_at_optimal_eq_pareto R_inf hα hβ hC

/-- **Exact Completion-of-Squares Variational Identity for Test-Time Scaling
(`test_time_scaling_completion_of_squares`)**:
For all `N > 0`, `T_inf > 0` with per-query inference FLOP budget `N * T_inf = C_inf > 0` and
`0 ≤ α_model`, `0 ≤ β_reason`, the excess risk above the Pareto inference frontier is the exact
nonnegative square:
`inferenceScalingRisk R_inf α_model β_reason N T_inf - paretoInferenceRisk R_inf α_model β_reason C_inf = (√(α_model * T_inf) - √(β_reason * N))^2 / C_inf`. -/
theorem test_time_scaling_completion_of_squares (R_inf : ℝ) {α_model β_reason N T_inf C_inf : ℝ}
    (hα : 0 ≤ α_model) (hβ : 0 ≤ β_reason) (hN : 0 < N) (hT : 0 < T_inf) (hC : 0 < C_inf)
    (hbudget : N * T_inf = C_inf) :
    inferenceScalingRisk R_inf α_model β_reason N T_inf -
        paretoInferenceRisk R_inf α_model β_reason C_inf =
      (Real.sqrt (α_model * T_inf) - Real.sqrt (β_reason * N)) ^ 2 / C_inf :=
  chinchilla_completion_of_squares R_inf hα hβ hN hT hC hbudget

/-- **Global Variational Optimality of the Test-Time Scaling Frontier
(`test_time_scaling_variational_optimality`)**:
For all `N > 0`, `T_inf > 0` with `N * T_inf = C_inf > 0` and `0 < α_model`, `0 < β_reason`:
1. The excess risk is the exact square `(√(α_model * T_inf) - √(β_reason * N))^2 / C_inf ≥ 0`.
2. `paretoInferenceRisk R_inf α_model β_reason C_inf ≤ inferenceScalingRisk R_inf α_model β_reason N T_inf`.
3. `optimalInferenceModelSize * optimalReasoningBudget = C_inf` and
   `optimalInferenceModelSize / optimalReasoningBudget = α_model / β_reason`. -/
theorem test_time_scaling_variational_optimality (R_inf : ℝ) {α_model β_reason N T_inf C_inf : ℝ}
    (hα : 0 < α_model) (hβ : 0 < β_reason) (hN : 0 < N) (hT : 0 < T_inf) (hC : 0 < C_inf)
    (hbudget : N * T_inf = C_inf) :
    inferenceScalingRisk R_inf α_model β_reason N T_inf -
        paretoInferenceRisk R_inf α_model β_reason C_inf =
      (Real.sqrt (α_model * T_inf) - Real.sqrt (β_reason * N)) ^ 2 / C_inf ∧
    0 ≤ (Real.sqrt (α_model * T_inf) - Real.sqrt (β_reason * N)) ^ 2 / C_inf ∧
    paretoInferenceRisk R_inf α_model β_reason C_inf ≤
      inferenceScalingRisk R_inf α_model β_reason N T_inf ∧
    optimalInferenceModelSize α_model β_reason C_inf *
      optimalReasoningBudget α_model β_reason C_inf = C_inf ∧
    optimalInferenceModelSize α_model β_reason C_inf /
      optimalReasoningBudget α_model β_reason C_inf = α_model / β_reason := by
  have hsq := test_time_scaling_completion_of_squares R_inf hα.le hβ.le hN hT hC hbudget
  have hnonneg : 0 ≤ (Real.sqrt (α_model * T_inf) - Real.sqrt (β_reason * N)) ^ 2 / C_inf := by
    positivity
  have hopt := chinchilla_variational_global_optimality R_inf hα.le hβ.le hN hT hC hbudget
  exact ⟨hsq, hnonneg, hopt,
    optimalInferenceModelSize_mul_optimalReasoningBudget hα hβ hC,
    optimalInferenceModelSize_div_optimalReasoningBudget hα hβ hC⟩

/-! ## 3. When a Smaller Reasoning Model Strictly Beats an `M×` Larger 1-Step Model -/

/-- Exact risk gap between an `M×` larger 1-step model `(M * N_small, T_inf = 1)` and a small
reasoning model `(N_small, T_inf = M)` at equal inference FLOPs `M * N_small`:
`inferenceScalingRisk R_inf α_model β_reason (M * N_small) 1 - inferenceScalingRisk R_inf α_model β_reason N_small M = (1 - 1 / M) * (β_reason - α_model / N_small)`. -/
theorem giant_one_step_minus_small_reasoning_risk_gap (R_inf α_model β_reason N_small M : ℝ)
    (hN : N_small ≠ 0) (hM : M ≠ 0) :
    inferenceScalingRisk R_inf α_model β_reason (M * N_small) 1 -
        inferenceScalingRisk R_inf α_model β_reason N_small M =
      (1 - 1 / M) * (β_reason - α_model / N_small) := by
  unfold inferenceScalingRisk
  field_simp [hN, hM]
  ring

/-- **When a Smaller Reasoning Model Strictly Beats an `M×` Larger 1-Step Model at Equal Inference FLOPs
(`small_reasoning_model_beats_giant_one_step_at_equal_inference_flops`)**:
Compare two systems with **identical per-query inference FLOPs** `C_inf = M * N_small` (`M > 1`, `N_small > 0`):
1. **Small Reasoning Model**: parameter count `N_small`, test-time reasoning multiplier `T_inf = M`
   (inference FLOPs `N_small * M = M * N_small`).
2. **Giant 1-Step Model**: parameter count `M * N_small`, direct 1-step inference `T_inf = 1`
   (inference FLOPs `(M * N_small) * 1 = M * N_small`).

Then:
`inferenceScalingRisk R_inf α_model β_reason N_small M < inferenceScalingRisk R_inf α_model β_reason (M * N_small) 1 ↔ α_model / N_small < β_reason`.
Whenever the reasoning bottleneck `β_reason` exceeds the base model's single-step capacity gap
`α_model / N_small`, allocating the `M×` compute multiplier to test-time reasoning strictly beats
scaling model parameters by `M×` with 1-step generation! -/
theorem small_reasoning_model_beats_giant_one_step_at_equal_inference_flops
    (R_inf α_model β_reason : ℝ) {N_small M : ℝ}
    (hN : 0 < N_small) (hM : 1 < M) :
    (N_small * M = (M * N_small) * 1) ∧
    (inferenceScalingRisk R_inf α_model β_reason N_small M <
        inferenceScalingRisk R_inf α_model β_reason (M * N_small) 1 ↔
      α_model / N_small < β_reason) := by
  have hM_pos : 0 < M := zero_lt_one.trans hM
  have hN_ne : N_small ≠ 0 := ne_of_gt hN
  have hM_ne : M ≠ 0 := ne_of_gt hM_pos
  have hgap := giant_one_step_minus_small_reasoning_risk_gap
    R_inf α_model β_reason N_small M hN_ne hM_ne
  have h1_div_lt : 1 / M < 1 := (div_lt_one hM_pos).mpr hM
  have hfac_pos : 0 < 1 - 1 / M := sub_pos.mpr h1_div_lt
  refine ⟨by ring, ?_⟩
  constructor
  · intro hlt
    have hdiff_pos : 0 < (1 - 1 / M) * (β_reason - α_model / N_small) := by linarith
    have hsub_pos : 0 < β_reason - α_model / N_small :=
      pos_of_mul_pos_right hdiff_pos hfac_pos.le
    linarith
  · intro hlt
    have hsub_pos : 0 < β_reason - α_model / N_small := sub_pos.mpr hlt
    have hdiff_pos : 0 < (1 - 1 / M) * (β_reason - α_model / N_small) :=
      mul_pos hfac_pos hsub_pos
    linarith

/-- **Joint `H`-Step Contractive Chain-of-Thought + Best-of-`K` Verification Error Bound
(`cot_best_of_K_joint_error_bound`)**:
Combining an `H`-step contractive Chain-of-Thought rollout (`0 ≤ Λ < 1`, `0 ≤ ε_token`, where
single-trajectory error satisfies `autoregressiveRolloutError Λ ε_token H ≤ ε_token / (1 - Λ) ≤ 1`)
with Best-of-`K` verifier selection across `K` parallel trajectories bounds the joint `K`-trajectory
failure probability by `(ε_token / (1 - Λ)) ^ K`, and exponentially by
`Real.exp (-(K : ℝ) * (1 - ε_token / (1 - Λ)))`. -/
theorem cot_best_of_K_joint_error_bound {Λ ε_token : ℝ}
    (hΛ0 : 0 ≤ Λ) (hΛ1 : Λ < 1) (hε0 : 0 ≤ ε_token)
    (hε_lt : ε_token / (1 - Λ) ≤ 1) (H K : ℕ) :
    bestOfKFailureProb (autoregressiveRolloutError Λ ε_token H) K ≤
        (ε_token / (1 - Λ)) ^ K ∧
      (ε_token / (1 - Λ)) ^ K ≤ Real.exp (-(K : ℝ) * (1 - ε_token / (1 - Λ))) ∧
      bestOfKFailureProb (autoregressiveRolloutError Λ ε_token H) K ≤
        Real.exp (-(K : ℝ) * (1 - ε_token / (1 - Λ))) := by
  unfold bestOfKFailureProb
  have hroll_nonneg : 0 ≤ autoregressiveRolloutError Λ ε_token H :=
    autoregressiveRolloutError_nonneg hΛ0 hε0 H
  have hroll_le : autoregressiveRolloutError Λ ε_token H ≤ ε_token / (1 - Λ) :=
    (cot_contractive_rollout_horizon_free_bound hΛ0 hΛ1 hε0 H).2
  have h1_sub_pos : 0 < 1 - Λ := sub_pos.mpr hΛ1
  have hdiv_nonneg : 0 ≤ ε_token / (1 - Λ) := div_nonneg hε0 h1_sub_pos.le
  have hpow_le : (autoregressiveRolloutError Λ ε_token H) ^ K ≤ (ε_token / (1 - Λ)) ^ K :=
    pow_le_pow_left₀ hroll_nonneg hroll_le K
  have hexp_le : (ε_token / (1 - Λ)) ^ K ≤ Real.exp (-(K : ℝ) * (1 - ε_token / (1 - Λ))) :=
    bestOfK_exponential_bound hdiv_nonneg hε_lt K
  exact ⟨hpow_le, hexp_le, hpow_le.trans hexp_le⟩

/-! ## 4. Total Lifecycle Compute Synthesis (Pre-Training FLOPs + Deployment Inference FLOPs over `Q` Queries) -/

/-- Total lifecycle compute (pre-training + deployment inference) for a model of size `N` trained on
`D` pre-training tokens and deployed across `Q` queries each of reasoning length `T_inf`:
`totalLifecycleCompute N D Q T_inf = N * D + Q * N * T_inf`. -/
def totalLifecycleCompute (N D Q T_inf : ℝ) : ℝ :=
  N * D + Q * N * T_inf

/-- Factored form of total lifecycle compute:
`totalLifecycleCompute N D Q T_inf = N * (D + Q * T_inf)`. -/
theorem totalLifecycleCompute_factored (N D Q T_inf : ℝ) :
    totalLifecycleCompute N D Q T_inf = N * (D + Q * T_inf) := by
  unfold totalLifecycleCompute
  ring

/-- Feasible model parameter count `N` for a given total lifecycle compute budget `C_total`,
pre-training token budget `D`, deployment query volume `Q`, and test-time reasoning horizon `T_inf`:
`lifecycleFeasibleModelSize C_total D Q T_inf = C_total / (D + Q * T_inf)`. -/
def lifecycleFeasibleModelSize (C_total D Q T_inf : ℝ) : ℝ :=
  C_total / (D + Q * T_inf)

/-- Effective training-token-to-parameter ratio `D / N` ("over-training multiplier") when model size
is chosen to satisfy the lifecycle compute budget `C_total`:
`lifecycleTokenToParamRatio C_total D Q T_inf = D / lifecycleFeasibleModelSize C_total D Q T_inf`. -/
def lifecycleTokenToParamRatio (C_total D Q T_inf : ℝ) : ℝ :=
  D / lifecycleFeasibleModelSize C_total D Q T_inf

/-- Exact identity showing that `lifecycleFeasibleModelSize C_total D Q T_inf` satisfies the total
lifecycle compute budget `totalLifecycleCompute N D Q T_inf = C_total` whenever `D + Q * T_inf ≠ 0`. -/
theorem totalLifecycleCompute_at_feasibleModelSize {C_total D Q T_inf : ℝ}
    (hden : D + Q * T_inf ≠ 0) :
    totalLifecycleCompute (lifecycleFeasibleModelSize C_total D Q T_inf) D Q T_inf = C_total := by
  rw [totalLifecycleCompute_factored]
  unfold lifecycleFeasibleModelSize
  exact div_mul_cancel₀ C_total hden

/-- Closed-form expression for the lifecycle token-to-parameter ratio:
`lifecycleTokenToParamRatio C_total D Q T_inf = D * (D + Q * T_inf) / C_total`. -/
theorem lifecycleTokenToParamRatio_eq (C_total D Q T_inf : ℝ) :
    lifecycleTokenToParamRatio C_total D Q T_inf = D * (D + Q * T_inf) / C_total := by
  unfold lifecycleTokenToParamRatio lifecycleFeasibleModelSize
  rw [div_div_eq_mul_div]

/-- **Deployment Queries & Test-Time Reasoning Shift Optimal Model Size Down and Over-Training Ratio Up
(`deployment_queries_shift_optimal_model_size_down`)**:
For a fixed pre-training token budget `D > 0` and total lifecycle compute budget `C_total > 0`:
1. Increasing deployment query volume `0 ≤ Q₁ < Q₂` (at `T_inf > 0`) **strictly reduces** the
   feasible model size `lifecycleFeasibleModelSize C_total D Q₂ T_inf < lifecycleFeasibleModelSize C_total D Q₁ T_inf`
   and **strictly increases** the token-to-parameter over-training ratio
   `lifecycleTokenToParamRatio C_total D Q₁ T_inf < lifecycleTokenToParamRatio C_total D Q₂ T_inf`.
2. Increasing test-time reasoning horizon `0 ≤ T₁ < T₂` (at `Q > 0`) likewise **strictly reduces**
   the feasible model size `lifecycleFeasibleModelSize C_total D Q T₂ < lifecycleFeasibleModelSize C_total D Q T₁`
   and **strictly increases** the token-to-parameter ratio. -/
theorem deployment_queries_shift_optimal_model_size_down
    {C_total D Q₁ Q₂ T_inf Q T₁ T₂ : ℝ}
    (hC : 0 < C_total) (hD : 0 < D)
    (hQ₁ : 0 ≤ Q₁) (hQ_lt : Q₁ < Q₂) (hT : 0 < T_inf)
    (hQ : 0 < Q) (hT₁ : 0 ≤ T₁) (hT_lt : T₁ < T₂) :
    (lifecycleFeasibleModelSize C_total D Q₂ T_inf <
        lifecycleFeasibleModelSize C_total D Q₁ T_inf ∧
      lifecycleTokenToParamRatio C_total D Q₁ T_inf <
        lifecycleTokenToParamRatio C_total D Q₂ T_inf) ∧
    (lifecycleFeasibleModelSize C_total D Q T₂ <
        lifecycleFeasibleModelSize C_total D Q T₁ ∧
      lifecycleTokenToParamRatio C_total D Q T₁ <
        lifecycleTokenToParamRatio C_total D Q T₂) := by
  have hden_Q1_pos : 0 < D + Q₁ * T_inf := by positivity
  have hden_Q_lt : D + Q₁ * T_inf < D + Q₂ * T_inf := by
    have hmul : Q₁ * T_inf < Q₂ * T_inf := mul_lt_mul_of_pos_right hQ_lt hT
    linarith
  have hden_T1_pos : 0 < D + Q * T₁ := by positivity
  have hden_T_lt : D + Q * T₁ < D + Q * T₂ := by
    have hmul : Q * T₁ < Q * T₂ := mul_lt_mul_of_pos_left hT_lt hQ
    linarith
  refine ⟨⟨?_, ?_⟩, ⟨?_, ?_⟩⟩
  · unfold lifecycleFeasibleModelSize
    exact div_lt_div_of_pos_left hC hden_Q1_pos hden_Q_lt
  · rw [lifecycleTokenToParamRatio_eq, lifecycleTokenToParamRatio_eq]
    have hnum : D * (D + Q₁ * T_inf) < D * (D + Q₂ * T_inf) :=
      mul_lt_mul_of_pos_left hden_Q_lt hD
    exact div_lt_div_of_pos_right hnum hC
  · unfold lifecycleFeasibleModelSize
    exact div_lt_div_of_pos_left hC hden_T1_pos hden_T_lt
  · rw [lifecycleTokenToParamRatio_eq, lifecycleTokenToParamRatio_eq]
    have hnum : D * (D + Q * T₁) < D * (D + Q * T₂) :=
      mul_lt_mul_of_pos_left hden_T_lt hD
    exact div_lt_div_of_pos_right hnum hC

end OpenTheory.TestTimeScaling
