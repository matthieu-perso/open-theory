import OpenTheory.Architectures.SpectralSparseCircuit
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Ring

/-!
# Autoregressive Next-Token Rollout, Chain-of-Thought (CoT) State-Tracking, and Out-of-Distribution (OOD) Compositional Generalization

This module formalizes two frontier pillars in `OpenTheory.AutoregressiveOOD`:

1. **Autoregressive Rollout & Long-Horizon Error Compounding**:
   - Defines the single-step autoregressive error transition `rolloutErrorStep Λ ε_token E = Λ * E + ε_token`
     and the `H`-step autoregressive rollout error `autoregressiveRolloutError Λ ε_token H` starting from `0`.
   - Proves `autoregressiveRolloutError_zero`, `autoregressiveRolloutError_succ`, and nonnegativity
     `autoregressiveRolloutError_nonneg`.
   - Proves **Unit-Sensitivity Linear Compounding (`autoregressiveRolloutError_unit_lip`)**:
     when `Λ = 1`, `autoregressiveRolloutError 1 ε_token H = (H : ℝ) * ε_token`.
   - Proves the **Closed-Form Geometric Series (`autoregressiveRolloutError_closed_form`)**:
     when `Λ ≠ 1`, `autoregressiveRolloutError Λ ε_token H = ε_token * ((Λ ^ H - 1) / (Λ - 1))`.
   - Establishes the **Stability Dichotomy Between Contractive (`Λ < 1`) and Unstable (`Λ > 1`) Rollouts**:
     - **Horizon-Free Uniform Bound (`cot_contractive_rollout_horizon_free_bound`)**: when `0 ≤ Λ < 1` and
       `0 ≤ ε_token` (e.g., under intermediate Chain-of-Thought state-verification / induction-head
       anchoring), `autoregressiveRolloutError Λ ε_token H ≤ ε_token / (1 - Λ)` uniformly for all `H : ℕ`.
     - **Unstable Super-Linear Drift (`unstable_rollout_strict_mono_and_exceeds_contractive`)**: when `1 < Λ`
       and `0 < ε_token`, the rollout error is strictly increasing in `H` and strictly exceeds linear
       compounding `(H : ℝ) * ε_token` for every `H ≥ 2`.

2. **Chain-of-Thought (CoT) Decomposition vs. Direct 1-Step Prediction**:
   - Defines `directPredictionRisk c d s_total n = c * d ^ s_total / n` for direct 1-step generation of a
     compositional task of complexity `s_total`, and `cotRolloutRisk H c d s_step n = (H : ℝ) * (c * d ^ s_step / n)`
     for an `H`-step Chain-of-Thought decomposition where each step has local complexity `s_step`.
   - Also defines `cotContractiveRolloutRisk Λ c d s_step n = (c * d ^ s_step / n) / (1 - Λ)` for contractive
     CoT state-tracking.
   - Proves `cot_strictly_dominates_direct_prediction`: whenever `0 < c`, `0 < n`, `0 < d`, `s_step ≤ s_total`,
     and `(H : ℝ) * d ^ s_step < d ^ s_total`, `H`-step CoT achieves strictly lower end-to-end risk than
     direct 1-step prediction, with exact risk ratio `(H : ℝ) / d ^ (s_total - s_step) < 1`.

3. **Out-of-Distribution (OOD) Compositional Generalization & Spurious Shift Robustness**:
   - Models a data distribution with an invariant causal spectral-sparse circuit subspace and an orthogonal
     spurious bulk subspace shifting by energy `Δ_OOD_sq > 0` at test time.
   - Defines `oodExcessRisk ε_struct p c_gen n w_spurious Δ_OOD_sq = ε_struct + (p : ℝ) * c_gen / n + w_spurious * Δ_OOD_sq`.
   - Connects directly to `SpectralSparseCircuit.sesc_orthogonal_bulk_invariance`:
     - Proves `sesc_zero_ood_shift_penalty`: for any `SESC` model, any orthogonal spectral-bulk OOD shift
       `η_bulk` produces identically zero squared prediction shift on both `sescCircuitForward` and
       `sescFullForward`.
     - Proves `sesc_ood_strictly_dominates_dense_mlp`: for any `Δ_OOD_sq > 0`, `w_spurious > 0`, and
       `p_SESC ≤ p_MLP`, the OOD excess risk of `SESC` (`w_spurious = 0`) is strictly smaller than that of
       an unconstrained dense MLP by the exact sum of the sample-variance reduction
       `((p_MLP : ℝ) - (p_SESC : ℝ)) * c_gen / n ≥ 0` and the OOD shift penalty `w_spurious * Δ_OOD_sq > 0`.
     - Proves `dense_mlp_ood_risk_floor_as_n_infty`: any model with `w_spurious > 0` suffers an irreducible
       OOD error floor `≥ w_spurious * Δ_OOD_sq > 0` for all sample sizes `n > 0`, whereas `SESC` achieves
       OOD excess risk `(p_SESC : ℝ) * c_gen / n < w_spurious * Δ_OOD_sq` for all
       `n > (p_SESC : ℝ) * c_gen / (w_spurious * Δ_OOD_sq)`.
-/

noncomputable section

open scoped BigOperators

namespace OpenTheory.AutoregressiveOOD

open OpenTheory.ConvolutionalGeneralization
open OpenTheory.SpectralSparseCircuit

/-! ## 1. Autoregressive Rollout & Long-Horizon Error Compounding -/

/-- Single-step autoregressive error recursion: at each rollout step, the accumulated state error `E`
is amplified by the transition sensitivity / Lipschitz factor `Λ` and incremented by the local
single-token prediction error `ε_token`. -/
def rolloutErrorStep (Λ ε_token : ℝ) (E : ℝ) : ℝ :=
  Λ * E + ε_token

/-- `H`-step autoregressive rollout error starting from zero initial error (`E_0 = 0`). -/
def autoregressiveRolloutError (Λ ε_token : ℝ) : ℕ → ℝ
  | 0 => 0
  | H + 1 => rolloutErrorStep Λ ε_token (autoregressiveRolloutError Λ ε_token H)

@[simp]
theorem autoregressiveRolloutError_zero (Λ ε_token : ℝ) :
    autoregressiveRolloutError Λ ε_token 0 = 0 :=
  rfl

@[simp]
theorem autoregressiveRolloutError_succ (Λ ε_token : ℝ) (H : ℕ) :
    autoregressiveRolloutError Λ ε_token (H + 1) =
      Λ * autoregressiveRolloutError Λ ε_token H + ε_token :=
  rfl

@[simp]
theorem autoregressiveRolloutError_one (Λ ε_token : ℝ) :
    autoregressiveRolloutError Λ ε_token 1 = ε_token := by
  simp [autoregressiveRolloutError, rolloutErrorStep]

/-- Nonnegativity of the `H`-step autoregressive rollout error when `0 ≤ Λ` and `0 ≤ ε_token`. -/
theorem autoregressiveRolloutError_nonneg {Λ ε_token : ℝ} (hΛ : 0 ≤ Λ) (hε : 0 ≤ ε_token) :
    ∀ H : ℕ, 0 ≤ autoregressiveRolloutError Λ ε_token H
  | 0 => le_rfl
  | H + 1 => by
      rw [autoregressiveRolloutError_succ]
      exact add_nonneg (mul_nonneg hΛ (autoregressiveRolloutError_nonneg hΛ hε H)) hε

/-- **Unit-Sensitivity Linear Error Compounding (`autoregressiveRolloutError_unit_lip`)**:
When the state-transition Lipschitz factor is `Λ = 1`, the `H`-step autoregressive rollout error
compounds linearly: `autoregressiveRolloutError 1 ε_token H = (H : ℝ) * ε_token`. -/
theorem autoregressiveRolloutError_unit_lip (ε_token : ℝ) :
    ∀ H : ℕ, autoregressiveRolloutError 1 ε_token H = (H : ℝ) * ε_token
  | 0 => by simp
  | H + 1 => by
      rw [autoregressiveRolloutError_succ, autoregressiveRolloutError_unit_lip ε_token H]
      push_cast
      ring

/-- **Closed-Form Geometric Series for Autoregressive Rollout Error
(`autoregressiveRolloutError_closed_form`)**:
When `Λ ≠ 1`, the `H`-step autoregressive rollout error satisfies the exact geometric closed form
`autoregressiveRolloutError Λ ε_token H = ε_token * ((Λ ^ H - 1) / (Λ - 1))`. -/
theorem autoregressiveRolloutError_closed_form {Λ : ℝ} (ε_token : ℝ) (H : ℕ) (hΛ : Λ ≠ 1) :
    autoregressiveRolloutError Λ ε_token H = ε_token * ((Λ ^ H - 1) / (Λ - 1)) := by
  have hsub : Λ - 1 ≠ 0 := sub_ne_zero.mpr hΛ
  induction H with
  | zero =>
      simp
  | succ k ih =>
      rw [autoregressiveRolloutError_succ, ih, pow_succ]
      field_simp [hsub]
      ring

/-- Alternative closed-form expression written with denominator `1 - Λ` when `Λ ≠ 1`
(particularly natural in the contractive regime `Λ < 1`):
`autoregressiveRolloutError Λ ε_token H = ε_token * ((1 - Λ ^ H) / (1 - Λ))`. -/
theorem autoregressiveRolloutError_closed_form_contractive {Λ : ℝ} (ε_token : ℝ) (H : ℕ)
    (hΛ : Λ ≠ 1) :
    autoregressiveRolloutError Λ ε_token H = ε_token * ((1 - Λ ^ H) / (1 - Λ)) := by
  rw [autoregressiveRolloutError_closed_form ε_token H hΛ]
  have hsub1 : Λ - 1 ≠ 0 := sub_ne_zero.mpr hΛ
  have hsub2 : 1 - Λ ≠ 0 := sub_ne_zero.mpr (Ne.symm hΛ)
  field_simp [hsub1, hsub2]
  ring

/-- **Horizon-Free Uniform Error Bound for Contractive Chain-of-Thought Rollouts
(`cot_contractive_rollout_horizon_free_bound`)**:
When intermediate Chain-of-Thought state verification or induction-head anchoring makes the state
transition contractive (`0 ≤ Λ < 1`) with `0 ≤ ε_token`, the `H`-step rollout error satisfies the
exact formula `(ε_token / (1 - Λ)) * (1 - Λ ^ H)` and is uniformly bounded by `ε_token / (1 - Λ)`
for **all** rollout horizons `H : ℕ` (preventing exponential error blowup as `H → ∞`). -/
theorem cot_contractive_rollout_horizon_free_bound {Λ ε_token : ℝ}
    (hΛ_nonneg : 0 ≤ Λ) (hΛ_lt : Λ < 1) (hε : 0 ≤ ε_token) (H : ℕ) :
    autoregressiveRolloutError Λ ε_token H = (ε_token / (1 - Λ)) * (1 - Λ ^ H) ∧
      autoregressiveRolloutError Λ ε_token H ≤ ε_token / (1 - Λ) := by
  have hΛ_ne : Λ ≠ 1 := ne_of_lt hΛ_lt
  have h1_sub_pos : 0 < 1 - Λ := sub_pos.mpr hΛ_lt
  have hpow_nonneg : 0 ≤ Λ ^ H := pow_nonneg hΛ_nonneg H
  have hclosed := autoregressiveRolloutError_closed_form_contractive ε_token H hΛ_ne
  have heq : autoregressiveRolloutError Λ ε_token H = (ε_token / (1 - Λ)) * (1 - Λ ^ H) := by
    rw [hclosed]
    ring
  refine ⟨heq, ?_⟩
  rw [heq]
  have hfactor_le : 1 - Λ ^ H ≤ 1 := by linarith
  have hdiv_nonneg : 0 ≤ ε_token / (1 - Λ) := div_nonneg hε h1_sub_pos.le
  calc
    (ε_token / (1 - Λ)) * (1 - Λ ^ H) ≤ (ε_token / (1 - Λ)) * 1 :=
      mul_le_mul_of_nonneg_left hfactor_le hdiv_nonneg
    _ = ε_token / (1 - Λ) := mul_one _

/-- **Unstable Rollout Strict Monotonicity & Super-Linear Error Blowup
(`unstable_rollout_strict_mono_and_exceeds_contractive`)**:
When the state transition is expansive (`1 < Λ`) and `0 < ε_token` (unanchored latent drift):
1. The rollout error grows strictly at every step:
   `autoregressiveRolloutError Λ ε_token H < autoregressiveRolloutError Λ ε_token (H + 1)`.
2. For every horizon `H`, `(H : ℝ) * ε_token ≤ autoregressiveRolloutError Λ ε_token H`.
3. For every horizon `2 ≤ H`, the rollout error strictly exceeds linear compounding:
   `(H : ℝ) * ε_token < autoregressiveRolloutError Λ ε_token H` (and hence also strictly exceeds
   any contractive rollout `0 ≤ Λ_c < 1`). -/
theorem unstable_rollout_strict_mono_and_exceeds_contractive {Λ ε_token : ℝ}
    (hΛ : 1 < Λ) (hε : 0 < ε_token) :
    (∀ H : ℕ, autoregressiveRolloutError Λ ε_token H <
      autoregressiveRolloutError Λ ε_token (H + 1)) ∧
    (∀ H : ℕ, (H : ℝ) * ε_token ≤ autoregressiveRolloutError Λ ε_token H) ∧
    (∀ H : ℕ, 2 ≤ H → (H : ℝ) * ε_token < autoregressiveRolloutError Λ ε_token H) := by
  have hlin_le : ∀ H : ℕ, (H : ℝ) * ε_token ≤ autoregressiveRolloutError Λ ε_token H := by
    intro H
    induction H with
    | zero => simp
    | succ k ih =>
        rw [autoregressiveRolloutError_succ]
        have hk_nonneg : 0 ≤ autoregressiveRolloutError Λ ε_token k :=
          autoregressiveRolloutError_nonneg (by linarith) hε.le k
        have hmul : autoregressiveRolloutError Λ ε_token k ≤
            Λ * autoregressiveRolloutError Λ ε_token k := by
          nlinarith
        push_cast
        linarith
  refine ⟨?_, hlin_le, ?_⟩
  · intro H
    rw [autoregressiveRolloutError_succ]
    have hE_nonneg : 0 ≤ autoregressiveRolloutError Λ ε_token H :=
      autoregressiveRolloutError_nonneg (by linarith) hε.le H
    have hmul : autoregressiveRolloutError Λ ε_token H ≤
        Λ * autoregressiveRolloutError Λ ε_token H := by
      nlinarith
    linarith
  · intro H hH
    obtain ⟨m, rfl⟩ : ∃ m : ℕ, H = m + 2 := Nat.exists_eq_add_of_le' hH
    clear hH
    induction m with
    | zero =>
        simp only [Nat.zero_add, autoregressiveRolloutError_succ, autoregressiveRolloutError_zero]
        push_cast
        nlinarith
    | succ k ih =>
        rw [Nat.add_right_comm k 1 2, autoregressiveRolloutError_succ (H := k + 2)]
        have hk_pos : 0 < autoregressiveRolloutError Λ ε_token (k + 2) := by
          have h2 : (0 : ℝ) < ((k + 2 : ℕ) : ℝ) * ε_token := by positivity
          linarith
        have hmul_strict :
            autoregressiveRolloutError Λ ε_token (k + 2) <
              Λ * autoregressiveRolloutError Λ ε_token (k + 2) := by
          nlinarith
        push_cast at ih ⊢
        linarith

/-! ## 2. Chain-of-Thought (CoT) Decomposition vs. Direct 1-Step Prediction -/

/-- Population excess risk of **Direct 1-Step Generation** for a compositional reasoning task of
total complexity `s_total`: `directPredictionRisk c d s_total n = c * d ^ s_total / n`. -/
def directPredictionRisk (c d : ℝ) (s_total : ℕ) (n : ℝ) : ℝ :=
  c * d ^ s_total / n

/-- End-to-end population excess risk of **`H`-Step Chain-of-Thought (CoT) Generation** with linear
step-error accumulation (`Λ = 1`), where each intermediate reasoning step has local complexity
`s_step`: `cotRolloutRisk H c d s_step n = (H : ℝ) * (c * d ^ s_step / n)`. -/
def cotRolloutRisk (H : ℕ) (c d : ℝ) (s_step : ℕ) (n : ℝ) : ℝ :=
  (H : ℝ) * (c * d ^ s_step / n)

/-- End-to-end population excess risk of **Contractive Chain-of-Thought Generation** (`0 ≤ Λ < 1`),
using the horizon-free error envelope `(c * d ^ s_step / n) / (1 - Λ)`. -/
def cotContractiveRolloutRisk (Λ c d : ℝ) (s_step : ℕ) (n : ℝ) : ℝ :=
  (c * d ^ s_step / n) / (1 - Λ)

/-- Bridge connecting `cotRolloutRisk` to `autoregressiveRolloutError 1`:
`cotRolloutRisk H c d s_step n` is literally the `H`-step unit-Lipschitz autoregressive rollout
error when each step has local token risk `ε_step = c * d ^ s_step / n`. -/
theorem cotRolloutRisk_eq_autoregressiveRolloutError (H : ℕ) (c d : ℝ) (s_step : ℕ) (n : ℝ) :
    cotRolloutRisk H c d s_step n =
      autoregressiveRolloutError 1 (c * d ^ s_step / n) H := by
  rw [autoregressiveRolloutError_unit_lip]
  rfl

/-- **Chain-of-Thought Strictly Dominates Direct 1-Step Prediction
(`cot_strictly_dominates_direct_prediction`)**:
Suppose a compositional task of total complexity `s_total` is decomposed into `H` intermediate
Chain-of-Thought steps of local complexity `s_step ≤ s_total`. Whenever `0 < c`, `0 < n`, `0 < d`,
and `(H : ℝ) * d ^ s_step < d ^ s_total`:
1. `H`-step Chain-of-Thought achieves **strictly lower** end-to-end risk than direct 1-step
   prediction: `cotRolloutRisk H c d s_step n < directPredictionRisk c d s_total n`.
2. The exact risk ratio equals `(H : ℝ) / d ^ (s_total - s_step)` and is strictly less than `1`. -/
theorem cot_strictly_dominates_direct_prediction {H s_step s_total : ℕ} {c d n : ℝ}
    (hc : 0 < c) (hn : 0 < n) (hd : 0 < d) (hs : s_step ≤ s_total)
    (hdom : (H : ℝ) * d ^ s_step < d ^ s_total) :
    cotRolloutRisk H c d s_step n < directPredictionRisk c d s_total n ∧
      cotRolloutRisk H c d s_step n / directPredictionRisk c d s_total n =
        (H : ℝ) / d ^ (s_total - s_step) ∧
      (H : ℝ) / d ^ (s_total - s_step) < 1 := by
  unfold cotRolloutRisk directPredictionRisk
  have hc_ne : c ≠ 0 := ne_of_gt hc
  have hn_ne : n ≠ 0 := ne_of_gt hn
  have hd_step_pos : 0 < d ^ s_step := pow_pos hd s_step
  have hd_step_ne : d ^ s_step ≠ 0 := ne_of_gt hd_step_pos
  have hd_diff_pos : 0 < d ^ (s_total - s_step) := pow_pos hd (s_total - s_step)
  have hd_diff_ne : d ^ (s_total - s_step) ≠ 0 := ne_of_gt hd_diff_pos
  have hpow_split : d ^ s_total = d ^ s_step * d ^ (s_total - s_step) := by
    rw [← pow_add, Nat.add_sub_of_le hs]
  have h_strict : (H : ℝ) * (c * d ^ s_step / n) < c * d ^ s_total / n := by
    have hnum : (H : ℝ) * (c * d ^ s_step) < c * d ^ s_total := by
      calc
        (H : ℝ) * (c * d ^ s_step) = c * ((H : ℝ) * d ^ s_step) := by ring
        _ < c * d ^ s_total := mul_lt_mul_of_pos_left hdom hc
    calc
      (H : ℝ) * (c * d ^ s_step / n) = ((H : ℝ) * (c * d ^ s_step)) / n := by ring
      _ < (c * d ^ s_total) / n := div_lt_div_of_pos_right hnum hn
  have h_ratio :
      ((H : ℝ) * (c * d ^ s_step / n)) / (c * d ^ s_total / n) =
        (H : ℝ) / d ^ (s_total - s_step) := by
    rw [hpow_split]
    field_simp [hc_ne, hn_ne, hd_step_ne, hd_diff_ne]
  have h_ratio_lt_one : (H : ℝ) / d ^ (s_total - s_step) < 1 := by
    rw [div_lt_one hd_diff_pos]
    have hdom' : (H : ℝ) * d ^ s_step < d ^ (s_total - s_step) * d ^ s_step := by
      linarith [hpow_split]
    exact lt_of_mul_lt_mul_right hdom' hd_step_pos.le
  exact ⟨h_strict, h_ratio, h_ratio_lt_one⟩

/-- **Contractive CoT Uniformly Dominates Direct Prediction Across All Horizons
(`cot_contractive_dominates_direct_for_all_horizons`)**:
When `0 ≤ Λ < 1`, `0 < c`, `0 < n`, `0 < d`, and `d ^ s_step / (1 - Λ) < d ^ s_total`, an `H`-step
contractive CoT rollout achieves end-to-end error `autoregressiveRolloutError Λ (c * d ^ s_step / n) H`
strictly smaller than `directPredictionRisk c d s_total n` for **every** rollout horizon `H : ℕ`. -/
theorem cot_contractive_dominates_direct_for_all_horizons {Λ c d n : ℝ} {s_step s_total : ℕ}
    (hΛ_nonneg : 0 ≤ Λ) (hΛ_lt : Λ < 1)
    (hc : 0 < c) (hn : 0 < n) (hd : 0 < d)
    (hdom : d ^ s_step / (1 - Λ) < d ^ s_total) (H : ℕ) :
    autoregressiveRolloutError Λ (c * d ^ s_step / n) H ≤
      cotContractiveRolloutRisk Λ c d s_step n ∧
    cotContractiveRolloutRisk Λ c d s_step n <
      directPredictionRisk c d s_total n := by
  have hε_pos : 0 < c * d ^ s_step / n := by positivity
  have hbound := (cot_contractive_rollout_horizon_free_bound hΛ_nonneg hΛ_lt hε_pos.le H).2
  refine ⟨hbound, ?_⟩
  unfold cotContractiveRolloutRisk directPredictionRisk
  have hnum : c * (d ^ s_step / (1 - Λ)) < c * d ^ s_total :=
    mul_lt_mul_of_pos_left hdom hc
  calc
    (c * d ^ s_step / n) / (1 - Λ) = (c * (d ^ s_step / (1 - Λ))) / n := by ring
    _ < (c * d ^ s_total) / n := div_lt_div_of_pos_right hnum hn

/-! ## 3. Out-of-Distribution (OOD) Compositional Generalization & Spurious Shift Robustness -/

/-- Out-of-Distribution (OOD) excess population risk for an architecture of effective dimension `p`
with structural approximation bias `ε_struct`, finite-sample variance `(p : ℝ) * c_gen / n`, and
spurious-subspace sensitivity `w_spurious` under OOD bulk shift energy `Δ_OOD_sq`:
`oodExcessRisk ε_struct p c_gen n w_spurious Δ_OOD_sq = ε_struct + (p : ℝ) * c_gen / n + w_spurious * Δ_OOD_sq`. -/
def oodExcessRisk (ε_struct : ℝ) (p : ℕ) (c_gen n : ℝ) (w_spurious Δ_OOD_sq : ℝ) : ℝ :=
  ε_struct + (p : ℝ) * c_gen / n + w_spurious * Δ_OOD_sq

/-- When `Δ_OOD_sq = 0` (in-distribution evaluation) or `w_spurious = 0` (exact orthogonal-bulk
invariance, as in `SESC`), `oodExcessRisk` reduces to the in-distribution excess risk
`ε_struct + (p : ℝ) * c_gen / n`. -/
@[simp]
theorem oodExcessRisk_zero_spurious (ε_struct : ℝ) (p : ℕ) (c_gen n Δ_OOD_sq : ℝ) :
    oodExcessRisk ε_struct p c_gen n 0 Δ_OOD_sq = ε_struct + (p : ℝ) * c_gen / n := by
  simp [oodExcessRisk]

/-- Bridge to `SpectralSparseCircuit.archTotalExcessRisk` when `n : ℕ` and `w_spurious = 0`. -/
theorem oodExcessRisk_eq_archTotalExcessRisk (ε_struct : ℝ) (p : ℕ) (c_gen : ℝ) (n : ℕ)
    (Δ_OOD_sq : ℝ) :
    oodExcessRisk ε_struct p c_gen (n : ℝ) 0 Δ_OOD_sq =
      archTotalExcessRisk ε_struct p c_gen n := by
  simp [oodExcessRisk, archTotalExcessRisk]

/-- **Zero Pointwise OOD Shift Penalty for Spectral-Equivariant Sparse Circuits
(`sesc_zero_ood_shift_penalty`)**:
By `SpectralSparseCircuit.sesc_orthogonal_bulk_invariance`, for any `SESC` network `net`, any
input signal `x : Fin d → ℝ`, and any OOD distribution shift `η_bulk : Fin d → ℝ` supported in
the orthogonal spectral bulk (`∀ p l, patchProject B_spec (cyclicPatch embed η_bulk p) l = 0`),
the squared OOD output shift is identically zero for both the sparse circuit output
`sescCircuitForward` and the full output `sescFullForward`. -/
theorem sesc_zero_ood_shift_penalty {d k r m : ℕ} [NeZero d]
    (net : SpectralSparseCircuit r m) (embed : Fin k → Fin d)
    (B_spec : Matrix (Fin r) (Fin k) ℝ) (x η_bulk : Fin d → ℝ)
    (h_ortho : ∀ p : Fin d, ∀ l : Fin r, patchProject B_spec (cyclicPatch embed η_bulk p) l = 0) :
    (sescCircuitForward net embed B_spec (x + η_bulk) -
      sescCircuitForward net embed B_spec x) ^ 2 = 0 ∧
    (sescFullForward net embed B_spec (x + η_bulk) -
      sescFullForward net embed B_spec x) ^ 2 = 0 := by
  obtain ⟨hcirc, hfull⟩ := sesc_orthogonal_bulk_invariance net embed B_spec x η_bulk h_ortho
  rw [hcirc, hfull]
  simp

/-- **Strict OOD Dominance of `SESC` Over Unconstrained Dense MLPs
(`sesc_ood_strictly_dominates_dense_mlp`)**:
When both `SESC` (`w_spurious = 0`, effective dimension `p_SESC`) and an unconstrained dense MLP
(`w_spurious > 0`, effective dimension `p_MLP`) are structurally well-specified on the causal
subspace (`ε_struct = 0`) with `p_SESC ≤ p_MLP`, `0 ≤ c_gen`, `0 < n`, `0 < w_spurious`, and
`0 < Δ_OOD_sq`:
1. The exact OOD risk gap equals the sum of the finite-sample variance reduction and the OOD shift
   penalty:
   `oodExcessRisk 0 p_MLP c_gen n w_spurious Δ_OOD_sq - oodExcessRisk 0 p_SESC c_gen n 0 Δ_OOD_sq = ((p_MLP : ℝ) - (p_SESC : ℝ)) * c_gen / n + w_spurious * Δ_OOD_sq`.
2. This OOD risk gap is at least `w_spurious * Δ_OOD_sq > 0`.
3. `SESC` achieves strictly lower OOD excess risk than the dense MLP:
   `oodExcessRisk 0 p_SESC c_gen n 0 Δ_OOD_sq < oodExcessRisk 0 p_MLP c_gen n w_spurious Δ_OOD_sq`. -/
theorem sesc_ood_strictly_dominates_dense_mlp {p_SESC p_MLP : ℕ}
    {c_gen n w_spurious Δ_OOD_sq : ℝ}
    (hp : p_SESC ≤ p_MLP) (hc : 0 ≤ c_gen) (hn : 0 < n)
    (hw : 0 < w_spurious) (hΔ : 0 < Δ_OOD_sq) :
    oodExcessRisk 0 p_MLP c_gen n w_spurious Δ_OOD_sq -
        oodExcessRisk 0 p_SESC c_gen n 0 Δ_OOD_sq =
      ((p_MLP : ℝ) - (p_SESC : ℝ)) * c_gen / n + w_spurious * Δ_OOD_sq ∧
    w_spurious * Δ_OOD_sq ≤
      oodExcessRisk 0 p_MLP c_gen n w_spurious Δ_OOD_sq -
        oodExcessRisk 0 p_SESC c_gen n 0 Δ_OOD_sq ∧
    oodExcessRisk 0 p_SESC c_gen n 0 Δ_OOD_sq <
      oodExcessRisk 0 p_MLP c_gen n w_spurious Δ_OOD_sq := by
  unfold oodExcessRisk
  have hp_real : (p_SESC : ℝ) ≤ (p_MLP : ℝ) := Nat.cast_le.mpr hp
  have hp_sub_nonneg : 0 ≤ (p_MLP : ℝ) - (p_SESC : ℝ) := sub_nonneg.mpr hp_real
  have hvar_nonneg : 0 ≤ ((p_MLP : ℝ) - (p_SESC : ℝ)) * c_gen / n :=
    div_nonneg (mul_nonneg hp_sub_nonneg hc) hn.le
  have hshift_pos : 0 < w_spurious * Δ_OOD_sq := mul_pos hw hΔ
  have hgap :
      (0 + (p_MLP : ℝ) * c_gen / n + w_spurious * Δ_OOD_sq) -
        (0 + (p_SESC : ℝ) * c_gen / n + 0 * Δ_OOD_sq) =
      ((p_MLP : ℝ) - (p_SESC : ℝ)) * c_gen / n + w_spurious * Δ_OOD_sq := by
    ring
  refine ⟨hgap, by linarith, by linarith⟩

/-- **Irreducible Dense MLP OOD Risk Floor vs. Vanishing `SESC` OOD Risk as `n → ∞`
(`dense_mlp_ood_risk_floor_as_n_infty`)**:
For any `0 ≤ ε_struct`, `0 ≤ c_gen`, `0 < n`, `0 < w_spurious`, and `0 < Δ_OOD_sq`:
1. **Irreducible Spurious Risk Floor**: Even with arbitrarily large training sample size `n`, any
   architecture with spurious sensitivity `w_spurious > 0` suffers an OOD excess risk floor
   `w_spurious * Δ_OOD_sq ≤ oodExcessRisk ε_struct p_MLP c_gen n w_spurious Δ_OOD_sq`.
2. **Sample-Size Threshold where `SESC` Beats the Infinite-Data MLP Floor**: Whenever
   `n > (p_SESC : ℝ) * c_gen / (w_spurious * Δ_OOD_sq)`, a finite-sample `SESC` model achieves
   total OOD excess risk strictly below the infinite-data OOD floor `w_spurious * Δ_OOD_sq` of the
   spurious-sensitive model! -/
theorem dense_mlp_ood_risk_floor_as_n_infty {p_SESC p_MLP : ℕ}
    {ε_struct c_gen n w_spurious Δ_OOD_sq : ℝ}
    (hε : 0 ≤ ε_struct) (hc : 0 ≤ c_gen) (hn : 0 < n)
    (hw : 0 < w_spurious) (hΔ : 0 < Δ_OOD_sq)
    (hn_thresh : (p_SESC : ℝ) * c_gen / (w_spurious * Δ_OOD_sq) < n) :
    w_spurious * Δ_OOD_sq ≤ oodExcessRisk ε_struct p_MLP c_gen n w_spurious Δ_OOD_sq ∧
    oodExcessRisk 0 p_SESC c_gen n 0 Δ_OOD_sq < w_spurious * Δ_OOD_sq ∧
    oodExcessRisk 0 p_SESC c_gen n 0 Δ_OOD_sq <
      oodExcessRisk ε_struct p_MLP c_gen n w_spurious Δ_OOD_sq := by
  unfold oodExcessRisk
  have hshift_pos : 0 < w_spurious * Δ_OOD_sq := mul_pos hw hΔ
  have hvar_mlp_nonneg : 0 ≤ (p_MLP : ℝ) * c_gen / n := by positivity
  have hfloor :
      w_spurious * Δ_OOD_sq ≤ ε_struct + (p_MLP : ℝ) * c_gen / n + w_spurious * Δ_OOD_sq := by
    linarith
  have hmul_lt : (p_SESC : ℝ) * c_gen < n * (w_spurious * Δ_OOD_sq) := by
    have h := (div_lt_iff₀ hshift_pos).mp hn_thresh
    linarith
  have hsesc_lt : (p_SESC : ℝ) * c_gen / n < w_spurious * Δ_OOD_sq := by
    rw [div_lt_iff₀ hn]
    linarith
  have hsesc_lt_floor :
      0 + (p_SESC : ℝ) * c_gen / n + 0 * Δ_OOD_sq < w_spurious * Δ_OOD_sq := by
    linarith
  exact ⟨hfloor, hsesc_lt_floor, lt_of_lt_of_le hsesc_lt_floor hfloor⟩

end OpenTheory.AutoregressiveOOD
