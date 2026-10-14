import OpenTheory.Architectures.Attention
import OpenTheory.Architectures.AttentionLogitStability
import OpenTheory.Architectures.AttentionAblation
import OpenTheory.Architectures.Torch
import Mathlib.Analysis.SpecialFunctions.Log.Basic

/-!
# Training & Margin-Growth Dynamics of In-Context Attention Retrieval

One of the defining phenomena of Transformer language models is **In-Context Learning (ICL)**
and **Associative Retrieval**: how an attention head adapts during training (or via prompt
alignment / gradient updates) to focus sharply on a relevant context token `i_star : Fin n` and
retrieve its value `values i_star` with exponentially decaying error.

In `OpenTheory.AttentionLogitStability`, we proved static concentration certificates:
- `logit_margin_implies_focused`: a logit margin `γ` for an unmasked token `i_star` bounds the
  complementary tail weight by `((n - 1 : ℕ) : ℝ) * Real.exp (-γ)`.
- `logit_margin_output_stability`: under value dispersion `D_V`, a logit margin `γ` bounds the
  attention retrieval error `‖rowOutput s mask values - values i_star‖` by
  `((n - 1 : ℕ) : ℝ) * Real.exp (-γ) * D_V`.

This module closes the dynamical loop by formalizing the **margin-growth training trajectory**
of an attention head and proving four core pillars of In-Context Retrieval:

1. **Score Update Dynamics & Linear Logit-Margin Growth**:
   - Defines `scoreStep η scores g_score` and the `T`-step score trajectory
     `scoreTrajectory η scores₀ g_seq T` for both function-represented score vectors `Fin n → ℝ`
     and `TorchLean` score tensors `Tensor ℝ [n]` (bridged via `scoreTensor` and `maskTensor`).
   - Proves `logitMargin_after_step` and `logitMargin_trajectory_ge`: if the initial scores have
     logit margin `γ₀` at `i_star` and each step's score gradient separates `i_star` from all other
     unmasked keys `j ≠ i_star` by at least `Δ_g`, then after `T` steps with learning rate `η ≥ 0`,
     the logit margin grows linearly to at least `γ₀ + (T : ℝ) * η * Δ_g`.

2. **Bilinear Query-Key Gradient Step Separation**:
   - Defines the scaled dot-product bilinear score `bilinearScore scale q K` between a query
     vector `q : Fin d_k → ℝ` and keys `K : Fin n → Fin d_k → ℝ`.
   - Proves `bilinearScore_step_eq`: a linear query update `q ↦ q + η • d_q` induces an exact
     score step `scoreStep η (bilinearScore scale q K) (bilinearScore scale d_q K)`.
   - Proves `queryStep_score_separation`: when the query update direction `d_q = K i_star` aligns
     with the target key and `K i_star` has correlation margin `Δ_K` over all other unmasked keys
     `j ≠ i_star`, the induced score gradient separates `i_star` from every other unmasked key by at
     least `scale * Δ_K`.

3. **Exponential Convergence of In-Context Retrieval Error Along Training**:
   - Proves `icl_attention_weight_convergence`: along a `T`-step alignment trajectory, the
     complementary attention weight `1 - (rowWeights ... i_star)` decays exponentially as
     `((n - 1 : ℕ) : ℝ) * Real.exp (-γ₀) * Real.exp (-(T : ℝ) * η * Δ_g)` (and with `((n : ℝ) - 1)`
     when `1 ≤ n`).
   - Proves `icl_retrieval_error_exponential_decay`: in any normed space `E`, under value
     dispersion `‖values j - values i_star‖ ≤ D_V`, the attention head's output converges
     exponentially to the target value `values i_star`:
     `‖rowOutput ... - values i_star‖ ≤ ((n - 1 : ℕ) : ℝ) * D_V * Real.exp (-γ₀) * Real.exp (-(T : ℝ) * η * Δ_g)`.

4. **Explicit Step Complexity & Population Risk of In-Context Retrieval**:
   - Proves `icl_step_complexity_certificate`: for any target retrieval accuracy `ε > 0`, once
     `(T : ℝ) * η * Δ_g ≥ Real.log (((n : ℝ) - 1) * D_V / ε) - γ₀`, the trained attention head is
     guaranteed to retrieve `values i_star` within norm error `ε`.
   - Proves `icl_population_risk_bound`: if an ideal oracle predictor uses the exact target value
     `values z (i_star z)` with population MSE `P.risk (oracle)`, then after `T` steps of in-context
     alignment, the trained attention layer's population MSE is bounded by
     `2 * P.risk (oracle) + 2 * (((n - 1 : ℕ) : ℝ) * D_V * Real.exp (-γ₀ - (T : ℝ) * η * Δ_g)) ^ 2`.
-/

noncomputable section

open TorchLean TorchLean.Tensor Spec MeasureTheory
open scoped BigOperators

namespace OpenTheory.InContextRetrieval

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

/-! ## Bridge Between Function Representation `Fin n → ℝ` and `Tensor ℝ [n]` -/

/-- Convert a function-represented score vector `s : Fin n → ℝ` into a 1D `TorchLean` tensor
`Tensor ℝ [n]`. -/
def scoreTensor {n : ℕ} (s : Fin n → ℝ) : Tensor ℝ [n] :=
  TorchLean.Tensor.Internal.Rep.ofFn (fun idx => s idx.1)

/-- Convert a boolean mask predicate `m : Fin n → Bool` into a 1D `TorchLean` boolean tensor
`Tensor Bool [n]`. -/
def maskTensor {n : ℕ} (m : Fin n → Bool) : Tensor Bool [n] :=
  TorchLean.Tensor.Internal.Rep.ofFn (fun idx => m idx.1)

@[simp] theorem getScalar_scoreTensor {n : ℕ} (s : Fin n → ℝ) (j : Fin n) :
    (scoreTensor s).getScalar j = s j := by
  simp [scoreTensor, getScalar_eq_apply]

@[simp] theorem getScalar_maskTensor {n : ℕ} (m : Fin n → Bool) (j : Fin n) :
    (maskTensor m).getScalar j = m j := by
  simp [maskTensor, getScalar_eq_apply]

/-- Attention weight on token `j : Fin n` for function-represented scores `s : Fin n → ℝ`
and boolean mask `m : Fin n → Bool`, directly wrapping `Attention.rowWeights`. -/
def iclRowWeights {n : ℕ} (s : Fin n → ℝ) (m : Fin n → Bool) (j : Fin n) : ℝ :=
  (Attention.rowWeights (scoreTensor s) (maskTensor m)).getScalar j

/-- Attention row output in a normed space `E` for function-represented scores `s : Fin n → ℝ`
and boolean mask `m : Fin n → Bool`, directly wrapping `Attention.rowOutput`. -/
def iclRowOutput {n : ℕ} (s : Fin n → ℝ) (m : Fin n → Bool) (values : Fin n → E) : E :=
  Attention.rowOutput (scoreTensor s) (maskTensor m) values

/-! ## 1. Score Update Dynamics & Linear Logit-Margin Growth -/

/-- Single gradient / alignment step on an attention score vector `scores : Fin n → ℝ`
with learning rate `η` along a score-gradient direction `g_score : Fin n → ℝ`. -/
def scoreStep {n : ℕ} (η : ℝ) (scores g_score : Fin n → ℝ) : Fin n → ℝ :=
  fun j => scores j + η * g_score j

/-- `T`-step score trajectory starting from `scores₀ : Fin n → ℝ` driven by a sequence of
score-gradient directions `g_seq : ℕ → Fin n → ℝ` with learning rate `η`. -/
def scoreTrajectory {n : ℕ} (η : ℝ) (scores₀ : Fin n → ℝ) (g_seq : ℕ → Fin n → ℝ) :
    ℕ → (Fin n → ℝ)
  | 0 => scores₀
  | t + 1 => scoreStep η (scoreTrajectory η scores₀ g_seq t) (g_seq t)

@[simp] theorem scoreTrajectory_zero {n : ℕ} (η : ℝ) (scores₀ : Fin n → ℝ)
    (g_seq : ℕ → Fin n → ℝ) :
    scoreTrajectory η scores₀ g_seq 0 = scores₀ :=
  rfl

@[simp] theorem scoreTrajectory_succ {n : ℕ} (η : ℝ) (scores₀ : Fin n → ℝ)
    (g_seq : ℕ → Fin n → ℝ) (t : ℕ) :
    scoreTrajectory η scores₀ g_seq (t + 1) =
      scoreStep η (scoreTrajectory η scores₀ g_seq t) (g_seq t) :=
  rfl

/-- Single-step logit-margin growth: if `scores` has logit margin `γ` at target key `i_star`
over all other unmasked keys `j ≠ i_star`, and the score-gradient `g_score` separates `i_star`
from all other unmasked keys by at least `Δ_g`, then for `η ≥ 0`, `scoreStep η scores g_score`
has logit margin `γ + η * Δ_g` at `i_star`. -/
theorem logitMargin_after_step {n : ℕ}
    (η γ Δ_g : ℝ)
    (scores g_score : Fin n → ℝ)
    (mask : Fin n → Bool)
    (i_star : Fin n)
    (hη : 0 ≤ η)
    (hmargin : ∀ j : Fin n, mask j = true → j ≠ i_star → scores j ≤ scores i_star - γ)
    (hsep : ∀ j : Fin n, mask j = true → j ≠ i_star → g_score j ≤ g_score i_star - Δ_g) :
    ∀ j : Fin n, mask j = true → j ≠ i_star →
      scoreStep η scores g_score j ≤ scoreStep η scores g_score i_star - (γ + η * Δ_g) := by
  intro j hmj hj_ne
  unfold scoreStep
  have hs := hmargin j hmj hj_ne
  have hg := hsep j hmj hj_ne
  have hηg : η * g_score j ≤ η * (g_score i_star - Δ_g) :=
    mul_le_mul_of_nonneg_left hg hη
  linarith

/-- Multi-step linear logit-margin growth by induction on `T : ℕ`:
if `scores₀` has initial logit margin `γ₀` at `i_star` on `mask`, and at every step `t`,
`g_seq t` has key separation at least `Δ_g` at `i_star` on `mask`, then after `T` steps with
learning rate `η ≥ 0`, `scoreTrajectory η scores₀ g_seq T` has logit margin at least
`γ₀ + (T : ℝ) * η * Δ_g` at `i_star`. -/
theorem logitMargin_trajectory_ge {n : ℕ}
    (η γ₀ Δ_g : ℝ)
    (scores₀ : Fin n → ℝ)
    (g_seq : ℕ → Fin n → ℝ)
    (mask : Fin n → Bool)
    (i_star : Fin n)
    (hη : 0 ≤ η)
    (hmargin₀ : ∀ j : Fin n, mask j = true → j ≠ i_star → scores₀ j ≤ scores₀ i_star - γ₀)
    (hsep : ∀ (t : ℕ) (j : Fin n), mask j = true → j ≠ i_star →
      g_seq t j ≤ g_seq t i_star - Δ_g)
    (T : ℕ) :
    ∀ j : Fin n, mask j = true → j ≠ i_star →
      scoreTrajectory η scores₀ g_seq T j ≤
        scoreTrajectory η scores₀ g_seq T i_star - (γ₀ + (T : ℝ) * η * Δ_g) := by
  induction T with
  | zero =>
      intro j hmj hj_ne
      simp only [scoreTrajectory_zero, Nat.cast_zero, zero_mul, add_zero]
      exact hmargin₀ j hmj hj_ne
  | succ t ih =>
      intro j hmj hj_ne
      have hstep := logitMargin_after_step η (γ₀ + (t : ℝ) * η * Δ_g) Δ_g
        (scoreTrajectory η scores₀ g_seq t) (g_seq t) mask i_star hη ih (hsep t) j hmj hj_ne
      simp only [scoreTrajectory_succ, Nat.cast_add, Nat.cast_one]
      linarith

/-- Tensor-indexed version of `logitMargin_trajectory_ge` for masks given directly as
`mask : Tensor Bool [n]`: after `T` steps, `scoreTensor (scoreTrajectory η scores₀ g_seq T)`
satisfies the logit margin condition of `AttentionLogitStability` with margin
`γ₀ + (T : ℝ) * η * Δ_g`. -/
theorem logitMargin_scoreTensor_trajectory_ge {n : ℕ}
    (η γ₀ Δ_g : ℝ)
    (scores₀ : Fin n → ℝ)
    (g_seq : ℕ → Fin n → ℝ)
    (mask : Tensor Bool [n])
    (i_star : Fin n)
    (hη : 0 ≤ η)
    (hmargin₀ : ∀ j : Fin n, mask.getScalar j = true → j ≠ i_star → scores₀ j ≤ scores₀ i_star - γ₀)
    (hsep : ∀ (t : ℕ) (j : Fin n), mask.getScalar j = true → j ≠ i_star →
      g_seq t j ≤ g_seq t i_star - Δ_g)
    (T : ℕ) :
    ∀ j ≠ i_star, mask.getScalar j = true →
      (scoreTensor (scoreTrajectory η scores₀ g_seq T)).getScalar j ≤
        (scoreTensor (scoreTrajectory η scores₀ g_seq T)).getScalar i_star -
          (γ₀ + (T : ℝ) * η * Δ_g) := by
  intro j hj_ne hmj
  simp only [getScalar_scoreTensor]
  exact logitMargin_trajectory_ge η γ₀ Δ_g scores₀ g_seq
    (fun k => mask.getScalar k) i_star hη hmargin₀ hsep T j hmj hj_ne

/-! ## 2. Bilinear Query-Key Gradient Step Separation -/

/-- Scaled dot-product bilinear score between a query vector `q : Fin d_k → ℝ` and a key
matrix `K : Fin n → Fin d_k → ℝ` with scaling factor `scale : ℝ` (typically `1 / √d_k`). -/
def bilinearScore {n d_k : ℕ} (scale : ℝ) (q : Fin d_k → ℝ) (K : Fin n → Fin d_k → ℝ) :
    Fin n → ℝ :=
  fun j => scale * ∑ l : Fin d_k, q l * K j l

/-- Bilinear Decomposition of Query Updates: updating the query vector `q` linearly along
direction `d_q` with step size `η` (`q_next l = q l + η * d_q l`) updates the attention scores
by `scoreStep η (bilinearScore scale q K) (bilinearScore scale d_q K)`. -/
theorem bilinearScore_step_eq {n d_k : ℕ}
    (scale η : ℝ) (q d_q : Fin d_k → ℝ) (K : Fin n → Fin d_k → ℝ) :
    bilinearScore scale (fun l => q l + η * d_q l) K =
      scoreStep η (bilinearScore scale q K) (bilinearScore scale d_q K) := by
  funext j
  unfold bilinearScore scoreStep
  have hsum :
      (∑ l : Fin d_k, (q l + η * d_q l) * K j l) =
        (∑ l : Fin d_k, q l * K j l) + η * (∑ l : Fin d_k, d_q l * K j l) := by
    calc
      (∑ l : Fin d_k, (q l + η * d_q l) * K j l) =
          ∑ l : Fin d_k, (q l * K j l + η * (d_q l * K j l)) := by
        refine Finset.sum_congr rfl fun l _ => ?_
        ring
      _ = (∑ l : Fin d_k, q l * K j l) + ∑ l : Fin d_k, η * (d_q l * K j l) :=
        Finset.sum_add_distrib
      _ = (∑ l : Fin d_k, q l * K j l) + η * (∑ l : Fin d_k, d_q l * K j l) := by
        rw [← Finset.mul_sum]
  rw [hsum]
  ring

/-- **Bilinear Query-Key Gradient Step Separation (`queryStep_score_separation`)**:
When a query step moves along the target key `K i_star` (or more generally any direction `d_q`),
if the target key `K i_star` has correlation margin `Δ_K` over all other unmasked keys `j ≠ i_star`:
`∑ l, K i_star l * (K i_star l - K j l) ≥ Δ_K`,
and `scale ≥ 0` (in particular for `scale > 0`), then the induced score gradient
`bilinearScore scale (K i_star) K` separates `i_star` from every other unmasked key `j ≠ i_star` by at
least `scale * Δ_K`. -/
theorem queryStep_score_separation {n d_k : ℕ}
    (scale Δ_K : ℝ)
    (K : Fin n → Fin d_k → ℝ)
    (mask : Fin n → Bool)
    (i_star : Fin n)
    (hscale : 0 ≤ scale)
    (hcorr : ∀ j : Fin n, mask j = true → j ≠ i_star →
      ∑ l : Fin d_k, K i_star l * (K i_star l - K j l) ≥ Δ_K) :
    ∀ j : Fin n, mask j = true → j ≠ i_star →
      bilinearScore scale (K i_star) K j ≤
        bilinearScore scale (K i_star) K i_star - scale * Δ_K := by
  intro j hmj hj_ne
  unfold bilinearScore
  have hdiff :
      scale * (∑ l : Fin d_k, K i_star l * K i_star l) -
        scale * (∑ l : Fin d_k, K i_star l * K j l) =
      scale * (∑ l : Fin d_k, K i_star l * (K i_star l - K j l)) := by
    rw [← mul_sub, ← Finset.sum_sub_distrib]
    congr 1
    refine Finset.sum_congr rfl fun l _ => ?_
    ring
  have hmul :
      scale * Δ_K ≤ scale * (∑ l : Fin d_k, K i_star l * (K i_star l - K j l)) :=
    mul_le_mul_of_nonneg_left (hcorr j hmj hj_ne) hscale
  linarith

/-- One-step bilinear query alignment margin growth: updating `q` to `q + η • K i_star` increases
the attention logit margin of `bilinearScore scale q K` at `i_star` from `γ` to
`γ + η * (scale * Δ_K)`. -/
theorem bilinearQueryStep_margin_growth {n d_k : ℕ}
    (scale η γ Δ_K : ℝ)
    (q : Fin d_k → ℝ)
    (K : Fin n → Fin d_k → ℝ)
    (mask : Fin n → Bool)
    (i_star : Fin n)
    (hscale : 0 ≤ scale)
    (hη : 0 ≤ η)
    (hmargin : ∀ j : Fin n, mask j = true → j ≠ i_star →
      bilinearScore scale q K j ≤ bilinearScore scale q K i_star - γ)
    (hcorr : ∀ j : Fin n, mask j = true → j ≠ i_star →
      ∑ l : Fin d_k, K i_star l * (K i_star l - K j l) ≥ Δ_K) :
    ∀ j : Fin n, mask j = true → j ≠ i_star →
      bilinearScore scale (fun l => q l + η * K i_star l) K j ≤
        bilinearScore scale (fun l => q l + η * K i_star l) K i_star - (γ + η * (scale * Δ_K)) := by
  rw [bilinearScore_step_eq]
  exact logitMargin_after_step η γ (scale * Δ_K)
    (bilinearScore scale q K) (bilinearScore scale (K i_star) K) mask i_star hη hmargin
    (queryStep_score_separation scale Δ_K K mask i_star hscale hcorr)

/-! ## 3. Exponential Convergence of In-Context Retrieval Error Along Training -/

/-- Helper lemma factoring `Real.exp (-(γ₀ + (T : ℝ) * η * Δ_g))` into
`Real.exp (-γ₀) * Real.exp (-(T : ℝ) * η * Δ_g)`. -/
theorem exp_neg_trajectory_margin_eq (γ₀ η Δ_g : ℝ) (T : ℕ) :
    Real.exp (-(γ₀ + (T : ℝ) * η * Δ_g)) =
      Real.exp (-γ₀) * Real.exp (-(T : ℝ) * η * Δ_g) := by
  have hneg : -(γ₀ + (T : ℝ) * η * Δ_g) = (-γ₀) + (-(T : ℝ) * η * Δ_g) := by ring
  rw [hneg, Real.exp_add]

/-- **Exponential Convergence of Target Attention Weight (`icl_attention_weight_convergence`)**:
Under initial logit margin `γ₀` at unmasked target token `i_star` and per-step score separation
`Δ_g` with learning rate `η ≥ 0`, after `T` steps of training, the complementary attention
weight on all non-target tokens decays exponentially:
`1 - (Attention.rowWeights (scoreTensor (scoreTrajectory η scores₀ g_seq T)) mask).getScalar i_star ≤ ((n - 1 : ℕ) : ℝ) * Real.exp (-γ₀) * Real.exp (-(T : ℝ) * η * Δ_g)`. -/
theorem icl_attention_weight_convergence {n : ℕ}
    (η γ₀ Δ_g : ℝ)
    (scores₀ : Fin n → ℝ)
    (g_seq : ℕ → Fin n → ℝ)
    (mask : Tensor Bool [n])
    (i_star : Fin n)
    (hη : 0 ≤ η)
    (hi_star : mask.getScalar i_star = true)
    (hmargin₀ : ∀ j : Fin n, mask.getScalar j = true → j ≠ i_star → scores₀ j ≤ scores₀ i_star - γ₀)
    (hsep : ∀ (t : ℕ) (j : Fin n), mask.getScalar j = true → j ≠ i_star →
      g_seq t j ≤ g_seq t i_star - Δ_g)
    (T : ℕ) :
    1 - (Attention.rowWeights (scoreTensor (scoreTrajectory η scores₀ g_seq T)) mask).getScalar i_star ≤
      ((n - 1 : ℕ) : ℝ) * Real.exp (-γ₀) * Real.exp (-(T : ℝ) * η * Δ_g) := by
  have hmarginT := logitMargin_scoreTensor_trajectory_ge η γ₀ Δ_g scores₀ g_seq mask i_star
    hη hmargin₀ hsep T
  have hfocus := (AttentionLogitStability.logit_margin_implies_focused
    (scoreTensor (scoreTrajectory η scores₀ g_seq T)) mask i_star
    (γ₀ + (T : ℝ) * η * Δ_g) hi_star hmarginT).2.1
  rw [exp_neg_trajectory_margin_eq, ← mul_assoc] at hfocus
  exact hfocus

/-- Real-cast `((n : ℝ) - 1)` version of `icl_attention_weight_convergence` (since `i_star : Fin n`
automatically implies `1 ≤ n`, so `((n - 1 : ℕ) : ℝ) = (n : ℝ) - 1`). -/
theorem icl_attention_weight_convergence_real_sub {n : ℕ}
    (η γ₀ Δ_g : ℝ)
    (scores₀ : Fin n → ℝ)
    (g_seq : ℕ → Fin n → ℝ)
    (mask : Tensor Bool [n])
    (i_star : Fin n)
    (hη : 0 ≤ η)
    (hi_star : mask.getScalar i_star = true)
    (hmargin₀ : ∀ j : Fin n, mask.getScalar j = true → j ≠ i_star → scores₀ j ≤ scores₀ i_star - γ₀)
    (hsep : ∀ (t : ℕ) (j : Fin n), mask.getScalar j = true → j ≠ i_star →
      g_seq t j ≤ g_seq t i_star - Δ_g)
    (T : ℕ) :
    1 - (Attention.rowWeights (scoreTensor (scoreTrajectory η scores₀ g_seq T)) mask).getScalar i_star ≤
      ((n : ℝ) - 1) * Real.exp (-γ₀) * Real.exp (-(T : ℝ) * η * Δ_g) := by
  have hn1 : 1 ≤ n := by have := i_star.isLt; omega
  have hcast : ((n - 1 : ℕ) : ℝ) = (n : ℝ) - 1 := by
    rw [Nat.cast_sub hn1, Nat.cast_one]
  rw [← hcast]
  exact icl_attention_weight_convergence η γ₀ Δ_g scores₀ g_seq mask i_star hη hi_star hmargin₀ hsep T

/-- **Exponential Convergence of In-Context Retrieval Error (`icl_retrieval_error_exponential_decay`)**:
In any normed space `E`, if all unmasked values satisfy `‖values j - values i_star‖ ≤ D_V` (`D_V ≥ 0`),
then after `T : ℕ` steps of alignment training, the attention head's output converges exponentially
to the exact target value `values i_star`:
`‖Attention.rowOutput (scoreTensor (scoreTrajectory η scores₀ g_seq T)) mask values - values i_star‖ ≤ ((n : ℝ) - 1) * D_V * Real.exp (-γ₀) * Real.exp (-(T : ℝ) * η * Δ_g)`. -/
theorem icl_retrieval_error_exponential_decay {n : ℕ}
    (η γ₀ Δ_g D_V : ℝ)
    (scores₀ : Fin n → ℝ)
    (g_seq : ℕ → Fin n → ℝ)
    (mask : Tensor Bool [n])
    (values : Fin n → E)
    (i_star : Fin n)
    (hη : 0 ≤ η)
    (hi_star : mask.getScalar i_star = true)
    (hmargin₀ : ∀ j : Fin n, mask.getScalar j = true → j ≠ i_star → scores₀ j ≤ scores₀ i_star - γ₀)
    (hsep : ∀ (t : ℕ) (j : Fin n), mask.getScalar j = true → j ≠ i_star →
      g_seq t j ≤ g_seq t i_star - Δ_g)
    (hD_nonneg : 0 ≤ D_V)
    (hD : ∀ j : Fin n, mask.getScalar j = true → ‖values j - values i_star‖ ≤ D_V)
    (T : ℕ) :
    ‖Attention.rowOutput (scoreTensor (scoreTrajectory η scores₀ g_seq T)) mask values -
      values i_star‖ ≤
      ((n : ℝ) - 1) * D_V * Real.exp (-γ₀) * Real.exp (-(T : ℝ) * η * Δ_g) := by
  have hn1 : 1 ≤ n := by have := i_star.isLt; omega
  have hcast : ((n - 1 : ℕ) : ℝ) = (n : ℝ) - 1 := by
    rw [Nat.cast_sub hn1, Nat.cast_one]
  have hmarginT := logitMargin_scoreTensor_trajectory_ge η γ₀ Δ_g scores₀ g_seq mask i_star
    hη hmargin₀ hsep T
  have hstab := (AttentionLogitStability.logit_margin_output_stability
    (scoreTensor (scoreTrajectory η scores₀ g_seq T)) mask values i_star
    (γ₀ + (T : ℝ) * η * Δ_g) D_V hi_star hmarginT hD_nonneg hD).1
  rw [hcast, exp_neg_trajectory_margin_eq] at hstab
  calc
    ‖Attention.rowOutput (scoreTensor (scoreTrajectory η scores₀ g_seq T)) mask values -
        values i_star‖ ≤
        ((n : ℝ) - 1) * (Real.exp (-γ₀) * Real.exp (-(T : ℝ) * η * Δ_g)) * D_V := hstab
    _ = ((n : ℝ) - 1) * D_V * Real.exp (-γ₀) * Real.exp (-(T : ℝ) * η * Δ_g) := by ring

/-! ## 4. Explicit Step Complexity & Population Risk of In-Context Retrieval -/

/-- **Explicit Step Complexity Certificate for `ε`-Accurate In-Context Retrieval
(`icl_step_complexity_certificate`)**:
Given any target retrieval accuracy `ε > 0`, whenever `T : ℕ` satisfies
`(T : ℝ) * η * Δ_g ≥ Real.log (((n : ℝ) - 1) * D_V / ε) - γ₀`
(with `0 < (n : ℝ) - 1` and `0 < D_V`), the attention head is guaranteed to retrieve `values i_star`
within error `ε`:
`‖Attention.rowOutput (scoreTensor (scoreTrajectory η scores₀ g_seq T)) mask values - values i_star‖ ≤ ε`. -/
theorem icl_step_complexity_certificate {n : ℕ}
    (η γ₀ Δ_g D_V ε : ℝ)
    (scores₀ : Fin n → ℝ)
    (g_seq : ℕ → Fin n → ℝ)
    (mask : Tensor Bool [n])
    (values : Fin n → E)
    (i_star : Fin n)
    (hη : 0 ≤ η)
    (hi_star : mask.getScalar i_star = true)
    (hmargin₀ : ∀ j : Fin n, mask.getScalar j = true → j ≠ i_star → scores₀ j ≤ scores₀ i_star - γ₀)
    (hsep : ∀ (t : ℕ) (j : Fin n), mask.getScalar j = true → j ≠ i_star →
      g_seq t j ≤ g_seq t i_star - Δ_g)
    (hn : 0 < (n : ℝ) - 1)
    (hD_pos : 0 < D_V)
    (hε_pos : 0 < ε)
    (hD : ∀ j : Fin n, mask.getScalar j = true → ‖values j - values i_star‖ ≤ D_V)
    (T : ℕ)
    (hT : (T : ℝ) * η * Δ_g ≥ Real.log (((n : ℝ) - 1) * D_V / ε) - γ₀) :
    ‖Attention.rowOutput (scoreTensor (scoreTrajectory η scores₀ g_seq T)) mask values -
      values i_star‖ ≤ ε := by
  have hdecay := icl_retrieval_error_exponential_decay η γ₀ Δ_g D_V scores₀ g_seq mask values i_star
    hη hi_star hmargin₀ hsep hD_pos.le hD T
  have hC_pos : 0 < ((n : ℝ) - 1) * D_V := mul_pos hn hD_pos
  have hratio_pos : 0 < ((n : ℝ) - 1) * D_V / ε := div_pos hC_pos hε_pos
  have hlog_le : -(γ₀ + (T : ℝ) * η * Δ_g) ≤ -Real.log (((n : ℝ) - 1) * D_V / ε) := by
    linarith
  have hexp_le :
      Real.exp (-(γ₀ + (T : ℝ) * η * Δ_g)) ≤
        Real.exp (-Real.log (((n : ℝ) - 1) * D_V / ε)) :=
    Real.exp_le_exp.mpr hlog_le
  have hexp_inv :
      Real.exp (-Real.log (((n : ℝ) - 1) * D_V / ε)) =
        ε / (((n : ℝ) - 1) * D_V) := by
    rw [Real.exp_neg, Real.exp_log hratio_pos, inv_div]
  rw [exp_neg_trajectory_margin_eq, hexp_inv] at hexp_le
  calc
    ‖Attention.rowOutput (scoreTensor (scoreTrajectory η scores₀ g_seq T)) mask values -
        values i_star‖ ≤
        ((n : ℝ) - 1) * D_V * (Real.exp (-γ₀) * Real.exp (-(T : ℝ) * η * Δ_g)) := by
      linarith
    _ ≤ ((n : ℝ) - 1) * D_V * (ε / (((n : ℝ) - 1) * D_V)) :=
      mul_le_mul_of_nonneg_left hexp_le hC_pos.le
    _ = ε := mul_div_cancel₀ ε (ne_of_gt hC_pos)

/-- Pointwise squared-error comparison between the `T`-step trained in-context attention head
and the ideal oracle retriever `values z (i_star z)`. -/
theorem icl_pointwise_sq_error_le_oracle {Z : Type*} {n : ℕ}
    (η γ₀ Δ_g D_V : ℝ)
    (scores₀ : Z → Fin n → ℝ)
    (g_seq : Z → ℕ → Fin n → ℝ)
    (mask : Z → Tensor Bool [n])
    (values : Z → Fin n → E)
    (i_star : Z → Fin n)
    (label : Z → E)
    (z : Z)
    (hη : 0 ≤ η)
    (hi_star : (mask z).getScalar (i_star z) = true)
    (hmargin₀ : ∀ j : Fin n, (mask z).getScalar j = true → j ≠ i_star z →
      scores₀ z j ≤ scores₀ z (i_star z) - γ₀)
    (hsep : ∀ (t : ℕ) (j : Fin n), (mask z).getScalar j = true → j ≠ i_star z →
      g_seq z t j ≤ g_seq z t (i_star z) - Δ_g)
    (hD_nonneg : 0 ≤ D_V)
    (hD : ∀ j : Fin n, (mask z).getScalar j = true →
      ‖values z j - values z (i_star z)‖ ≤ D_V)
    (T : ℕ) :
    ‖Attention.rowOutput (scoreTensor (scoreTrajectory η (scores₀ z) (g_seq z) T))
        (mask z) (values z) - label z‖ ^ 2 ≤
      2 * ‖values z (i_star z) - label z‖ ^ 2 +
        2 * (((n : ℝ) - 1) * D_V * Real.exp (-γ₀ - (T : ℝ) * η * Δ_g)) ^ 2 := by
  have hn1 : 1 ≤ n := by have := (i_star z).isLt; omega
  have hn_nonneg : 0 ≤ (n : ℝ) - 1 := by
    have : (1 : ℝ) ≤ (n : ℝ) := by exact_mod_cast hn1
    linarith
  have hδ_nonneg : 0 ≤ ((n : ℝ) - 1) * D_V * Real.exp (-γ₀ - (T : ℝ) * η * Δ_g) :=
    mul_nonneg (mul_nonneg hn_nonneg hD_nonneg) (Real.exp_pos _).le
  have hdecay := icl_retrieval_error_exponential_decay η γ₀ Δ_g D_V
    (scores₀ z) (g_seq z) (mask z) (values z) (i_star z)
    hη hi_star hmargin₀ hsep hD_nonneg hD T
  have hexp_eq :
      ((n : ℝ) - 1) * D_V * Real.exp (-γ₀) * Real.exp (-(T : ℝ) * η * Δ_g) =
        ((n : ℝ) - 1) * D_V * Real.exp (-γ₀ - (T : ℝ) * η * Δ_g) := by
    have h1 : -γ₀ - (T : ℝ) * η * Δ_g = (-γ₀) + (-(T : ℝ) * η * Δ_g) := by ring
    rw [h1, Real.exp_add, mul_assoc]
  rw [hexp_eq] at hdecay
  exact PredictionGap.sq_norm_le_double
    (Attention.rowOutput (scoreTensor (scoreTrajectory η (scores₀ z) (g_seq z) T))
      (mask z) (values z))
    (values z (i_star z))
    (label z)
    hδ_nonneg
    hdecay

/-- **Population Risk of In-Context Retrieval (`icl_population_risk_bound`)**:
Given a population `P : Population Z`, suppose an ideal oracle regressor uses the exact
retrieved context value `values z (i_star z)` with integrable squared-error loss and population
MSE `P.risk (fun z => ‖values z (i_star z) - label z‖ ^ 2)`. If almost surely under `P.law` the
target token `i_star z` is unmasked, initial scores have margin `γ₀`, score gradients have key
separation `Δ_g`, and value dispersion is at most `D_V ≥ 0`, then after `T` steps of in-context
alignment (`η ≥ 0`), the attention head has integrable squared-error loss and its population MSE
satisfies:
`P.risk (trained) ≤ 2 * P.risk (oracle) + 2 * (((n : ℝ) - 1) * D_V * Real.exp (-γ₀ - (T : ℝ) * η * Δ_g)) ^ 2`. -/
theorem icl_population_risk_bound {Z : Type*} [MeasurableSpace Z] {n : ℕ}
    (P : Population Z)
    (η γ₀ Δ_g D_V : ℝ)
    (scores₀ : Z → Fin n → ℝ)
    (g_seq : Z → ℕ → Fin n → ℝ)
    (mask : Z → Tensor Bool [n])
    (values : Z → Fin n → E)
    (i_star : Z → Fin n)
    (label : Z → E)
    (T : ℕ)
    (hη : 0 ≤ η)
    (hD_nonneg : 0 ≤ D_V)
    (hmeas_icl : AEStronglyMeasurable
      (fun z =>
        ‖Attention.rowOutput (scoreTensor (scoreTrajectory η (scores₀ z) (g_seq z) T))
          (mask z) (values z) - label z‖ ^ 2)
      P.law)
    (hint_oracle : P.IntegrableLoss
      (fun z => ‖values z (i_star z) - label z‖ ^ 2))
    (hae : ∀ᵐ z ∂P.law,
      (mask z).getScalar (i_star z) = true ∧
      (∀ j : Fin n, (mask z).getScalar j = true → j ≠ i_star z →
        scores₀ z j ≤ scores₀ z (i_star z) - γ₀) ∧
      (∀ (t : ℕ) (j : Fin n), (mask z).getScalar j = true → j ≠ i_star z →
        g_seq z t j ≤ g_seq z t (i_star z) - Δ_g) ∧
      (∀ j : Fin n, (mask z).getScalar j = true →
        ‖values z j - values z (i_star z)‖ ≤ D_V)) :
    P.IntegrableLoss
      (fun z =>
        ‖Attention.rowOutput (scoreTensor (scoreTrajectory η (scores₀ z) (g_seq z) T))
          (mask z) (values z) - label z‖ ^ 2) ∧
    P.risk
      (fun z =>
        ‖Attention.rowOutput (scoreTensor (scoreTrajectory η (scores₀ z) (g_seq z) T))
          (mask z) (values z) - label z‖ ^ 2) ≤
      2 * P.risk (fun z => ‖values z (i_star z) - label z‖ ^ 2) +
        2 * (((n : ℝ) - 1) * D_V * Real.exp (-γ₀ - (T : ℝ) * η * Δ_g)) ^ 2 := by
  let δ := ((n : ℝ) - 1) * D_V * Real.exp (-γ₀ - (T : ℝ) * η * Δ_g)
  let lossOracle : Z → ℝ := fun z => ‖values z (i_star z) - label z‖ ^ 2
  let lossICL : Z → ℝ := fun z =>
    ‖Attention.rowOutput (scoreTensor (scoreTrajectory η (scores₀ z) (g_seq z) T))
      (mask z) (values z) - label z‖ ^ 2
  let envelope : Z → ℝ := fun z => 2 * lossOracle z + 2 * δ ^ 2
  have hconst : Integrable (fun _ : Z => 2 * δ ^ 2) P.law := integrable_const _
  have henv_int : Integrable envelope P.law := (hint_oracle.const_mul 2).add hconst
  have hae_bound : ∀ᵐ z ∂P.law, 0 ≤ lossICL z ∧ lossICL z ≤ envelope z :=
    hae.mono fun z hz =>
      ⟨sq_nonneg _,
        icl_pointwise_sq_error_le_oracle η γ₀ Δ_g D_V scores₀ g_seq mask values i_star label z
          hη hz.1 hz.2.1 hz.2.2.1 hD_nonneg hz.2.2.2 T⟩
  obtain ⟨hi_icl, hle⟩ := P.risk_le_envelope lossICL envelope hmeas_icl henv_int hae_bound
  refine ⟨hi_icl, ?_⟩
  unfold Population.risk at hle ⊢
  rw [integral_add (hint_oracle.const_mul 2) hconst, integral_const_mul, integral_const] at hle
  have hprob : P.law.real Set.univ = 1 := by simp
  rw [hprob, one_smul] at hle
  exact hle

end OpenTheory.InContextRetrieval
