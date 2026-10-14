import OpenTheory.Optimization.LocalSharpnessDescent
import OpenTheory.Dynamics.MLPLocalSharpness

/-!
# Mini-Batch SGD under Local Sharpness and the Linear Scaling / Critical Batch Size Law

In practical deep learning, models are trained with **Stochastic Mini-Batch Gradient Descent (SGD)**
using batch size `B ≥ 1`, where the stochastic mini-batch gradient is `g_batch = g + ξ` (`g` is the
true batch gradient with squared norm `G = ‖g‖^2` and `ξ` is zero-mean mini-batch noise with
variance `σ_sq / B`), so that the expected squared norm of the stochastic gradient decomposes as
`E[‖g_batch‖^2] ≤ G + σ_sq / B`.

This module develops the exact optimization and scaling theory of mini-batch SGD under local
sharpness `S_loc > 0`:

1. **One-Step Mini-Batch SGD Descent with Variance Decomposition (`sgdNetProgress`,
   `sgdExpectedStepDrop`, `sgd_local_sharpness_one_step`, `noiseAwareStep`,
   `sgd_optimal_lr_with_noise`, `sgdNetProgress_noiseAwareStep_eq`, `sgdNetProgress_sub_eq_sq`,
   `noiseAwareStep_maximizes_sgdNetProgress`, `sgdNetProgress_mono_batchSize`)**:
   Under local sharpness `S_loc > 0`, stepping by `-η • g_batch` guarantees an expected loss drop of
   `sgdNetProgress η S_loc G σ_sq B = stepProgress η S_loc G - (η^2 * S_loc / 2) * (σ_sq / B)`.
   Completing the square proves that the **exact noise-aware optimal learning rate**
   `noiseAwareStep S_loc G σ_sq B = G / (S_loc * (G + σ_sq / B))` uniquely maximizes
   `sgdNetProgress` over all `η : ℝ`, achieving `G^2 / (2 * S_loc * (G + σ_sq / B))`.

2. **The Critical Batch Size & Linear Scaling Law (`criticalBatchRatio`, `criticalBatchSize`,
   `optimal_sgd_progress_vs_full_batch`, `critical_batch_half_efficiency`,
   `small_batch_linear_scaling_lower_bound`, `linear_scaling_regime_progress`,
   `batch_size_doubling_speedup`)**:
   Defining the critical batch size ratio `B_crit = σ_sq / G`, the optimal mini-batch SGD progress
   factors into full-batch progress `G / (2 * S_loc)` times the batch efficiency ratio
   `B / (B + B_crit)`. At `B = B_crit`, the optimal learning rate is half the full-batch learning
   rate and achieves 50% of full-batch per-step progress; for `B ≤ B_crit`, progress scales at
   least linearly with `B`.

3. **Multi-Step PL Convergence of Mini-Batch SGD to the Variance Noise Horizon (`sgdNoiseFloor`,
   `sgd_pl_one_step_recurrence`, `sgd_pl_noise_ball_convergence`,
   `mlp_sgd_noise_floor_proportional_to_inv_batch`)**:
   Under the Polyak-Łojasiewicz condition `G_t ≥ 2 * μ * (E_L t - floor)` with constant step size
   `0 < η ≤ 1 / S_loc`, expected optimality gaps contract geometrically at rate `(1 - η * μ)^T`
   to the explicit variance noise horizon `sgdNoiseFloor η S_loc μ σ_sq B = (η * S_loc * σ_sq) / (2 * μ * B)`,
   which specializes directly to two-layer MLPs via `mlpLocalSharpness`.
-/

noncomputable section

open scoped BigOperators RealInnerProductSpace

namespace OpenTheory.StochasticSharpnessSGD

open OpenTheory.LocalSharpnessDescent
open OpenTheory.MLPLocalSharpness

/-! ## 1. One-Step Mini-Batch SGD Descent and Noise-Aware Optimal Learning Rate -/

/-- Guaranteed expected one-step loss reduction of mini-batch SGD with step size `η`, local
sharpness `S_loc`, squared true gradient norm `G = ‖g‖^2`, single-sample gradient variance `σ_sq`,
and mini-batch size `B`:
`sgdNetProgress η S_loc G σ_sq B = stepProgress η S_loc G - (η^2 * S_loc / 2) * (σ_sq / B)`. -/
def sgdNetProgress (η S_loc G σ_sq : ℝ) (B : ℕ) : ℝ :=
  stepProgress η S_loc G - (η ^ 2 * S_loc / 2) * (σ_sq / (B : ℝ))

/-- Alias for `sgdNetProgress` representing the expected one-step objective drop of mini-batch SGD. -/
abbrev sgdExpectedStepDrop (η S_loc G σ_sq : ℝ) (B : ℕ) : ℝ :=
  sgdNetProgress η S_loc G σ_sq B

/-- Expanded linear-quadratic form of `sgdNetProgress`:
`η * G - (η^2 * S_loc / 2) * (G + σ_sq / B)`. -/
theorem sgdNetProgress_eq_linear_sub_quad (η S_loc G σ_sq : ℝ) (B : ℕ) :
    sgdNetProgress η S_loc G σ_sq B =
      η * G - (η ^ 2 * S_loc / 2) * (G + σ_sq / (B : ℝ)) := by
  unfold sgdNetProgress stepProgress
  ring

/-- When `σ_sq = 0` (zero mini-batch noise / full-batch gradient descent), `sgdNetProgress`
reduces to the deterministic `stepProgress η S_loc G`. -/
@[simp]
theorem sgdNetProgress_zero_noise (η S_loc G : ℝ) (B : ℕ) :
    sgdNetProgress η S_loc G 0 B = stepProgress η S_loc G := by
  unfold sgdNetProgress
  simp

/-- **One-Step Mini-Batch SGD Descent under Local Sharpness (`sgd_local_sharpness_one_step`)**:
If the expected next-step loss `E_L_next` satisfies the local-sharpness variance-decomposed upper
bound `E_L_next ≤ E_L - η * G + (η^2 * S_loc / 2) * (G + σ_sq / B)`, then
`E_L_next ≤ E_L - sgdNetProgress η S_loc G σ_sq B`. -/
theorem sgd_local_sharpness_one_step (E_L E_L_next η S_loc G σ_sq : ℝ) (B : ℕ)
    (hstep : E_L_next ≤ E_L - η * G + (η ^ 2 * S_loc / 2) * (G + σ_sq / (B : ℝ))) :
    E_L_next ≤ E_L - sgdNetProgress η S_loc G σ_sq B := by
  rw [sgdNetProgress_eq_linear_sub_quad]
  linarith

/-- Pointwise inner-product and norm-square identity for a mini-batch gradient `g_batch = g + ξ`
under a local-sharpness quadratic upper bound: when `⟪g, ξ⟫ = 0` and `‖g + ξ‖^2 = ‖g‖^2 + ‖ξ‖^2`,
the one-step loss drop decomposes into deterministic progress `stepProgress η S_loc (‖g‖^2)` minus
the variance penalty `(η^2 * S_loc / 2) * ‖ξ‖^2`. -/
theorem sgd_pointwise_pythagorean_step {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
    (L : E → ℝ) (u g ξ : E) (η S_loc : ℝ)
    (hquad : L (u - η • (g + ξ)) ≤
      L u - η * ⟪g, g + ξ⟫ + (S_loc / 2) * η ^ 2 * ‖g + ξ‖ ^ 2)
    (horth : ⟪g, ξ⟫ = 0) :
    L (u - η • (g + ξ)) ≤
      L u - stepProgress η S_loc (‖g‖ ^ 2) + (η ^ 2 * S_loc / 2) * ‖ξ‖ ^ 2 := by
  have hinner : ⟪g, g + ξ⟫ = ‖g‖ ^ 2 := by
    rw [inner_add_right, real_inner_self_eq_norm_sq, horth, add_zero]
  have hpyth : ‖g + ξ‖ ^ 2 = ‖g‖ ^ 2 + ‖ξ‖ ^ 2 := by
    rw [norm_add_sq_real, horth]
    ring
  rw [hinner, hpyth] at hquad
  unfold stepProgress
  linarith

/-- **Noise-Aware Optimal Learning Rate (`noiseAwareStep`)**:
The exact step size that maximizes expected one-step mini-batch SGD progress under local sharpness
`S_loc`, signal `G = ‖g‖^2`, and mini-batch noise variance `σ_sq / B`:
`η_opt = G / (S_loc * (G + σ_sq / B))`. -/
def noiseAwareStep (S_loc G σ_sq : ℝ) (B : ℕ) : ℝ :=
  G / (S_loc * (G + σ_sq / (B : ℝ)))

/-- Equivalent factored form of `noiseAwareStep`:
`noiseAwareStep S_loc G σ_sq B = adaptiveStep S_loc * (G / (G + σ_sq / B))`, scaling the
deterministic adaptive step `1 / S_loc` by the signal-to-total-variance ratio `G / (G + σ_sq / B)`. -/
theorem sgd_optimal_lr_with_noise (S_loc G σ_sq : ℝ) (B : ℕ) :
    noiseAwareStep S_loc G σ_sq B =
      adaptiveStep S_loc * (G / (G + σ_sq / (B : ℝ))) := by
  unfold noiseAwareStep adaptiveStep
  rw [one_div_mul_eq_div, div_div, mul_comm]

/-- When `σ_sq = 0` and `G ≠ 0`, `noiseAwareStep` recovers the deterministic optimal step
`adaptiveStep S_loc = 1 / S_loc`. -/
theorem noiseAwareStep_zero_noise (S_loc G : ℝ) (B : ℕ) (hG : G ≠ 0) :
    noiseAwareStep S_loc G 0 B = adaptiveStep S_loc := by
  rw [sgd_optimal_lr_with_noise]
  simp [div_self hG]

/-- **Maximum Expected Progress at the Noise-Aware Step (`sgdNetProgress_noiseAwareStep_eq`)**:
When `0 < S_loc` and `0 < G + σ_sq / B`, evaluating `sgdNetProgress` at `noiseAwareStep` yields
`G ^ 2 / (2 * S_loc * (G + σ_sq / B))`. -/
theorem sgdNetProgress_noiseAwareStep_eq (S_loc G σ_sq : ℝ) (B : ℕ)
    (hS : 0 < S_loc) (hV : 0 < G + σ_sq / (B : ℝ)) :
    sgdNetProgress (noiseAwareStep S_loc G σ_sq B) S_loc G σ_sq B =
      G ^ 2 / (2 * S_loc * (G + σ_sq / (B : ℝ))) := by
  have hS_ne : S_loc ≠ 0 := ne_of_gt hS
  have hV_ne : G + σ_sq / (B : ℝ) ≠ 0 := ne_of_gt hV
  rw [sgdNetProgress_eq_linear_sub_quad]
  unfold noiseAwareStep
  field_simp [hS_ne, hV_ne]
  ring

/-- **Completion-of-Squares Identity for Mini-Batch SGD Progress (`sgdNetProgress_sub_eq_sq`)**:
For any learning rate `η : ℝ`, when `0 < S_loc` and `0 < G + σ_sq / B`, the suboptimality gap
between `sgdNetProgress` at `noiseAwareStep` and at `η` is the exact square
`(S_loc * (G + σ_sq / B) / 2) * (η - noiseAwareStep S_loc G σ_sq B) ^ 2`. -/
theorem sgdNetProgress_sub_eq_sq (η S_loc G σ_sq : ℝ) (B : ℕ)
    (hS : 0 < S_loc) (hV : 0 < G + σ_sq / (B : ℝ)) :
    sgdNetProgress (noiseAwareStep S_loc G σ_sq B) S_loc G σ_sq B -
        sgdNetProgress η S_loc G σ_sq B =
      (S_loc * (G + σ_sq / (B : ℝ)) / 2) * (η - noiseAwareStep S_loc G σ_sq B) ^ 2 := by
  have hS_ne : S_loc ≠ 0 := ne_of_gt hS
  have hV_ne : G + σ_sq / (B : ℝ) ≠ 0 := ne_of_gt hV
  rw [sgdNetProgress_noiseAwareStep_eq S_loc G σ_sq B hS hV,
    sgdNetProgress_eq_linear_sub_quad]
  unfold noiseAwareStep
  field_simp [hS_ne, hV_ne]
  ring

/-- **Global Optimality of the Noise-Aware Learning Rate (`noiseAwareStep_maximizes_sgdNetProgress`)**:
For any `η : ℝ`, when `0 < S_loc` and `0 < G + σ_sq / B`, `noiseAwareStep S_loc G σ_sq B`
achieves at least as much expected one-step loss reduction as `η`. -/
theorem noiseAwareStep_maximizes_sgdNetProgress (η S_loc G σ_sq : ℝ) (B : ℕ)
    (hS : 0 < S_loc) (hV : 0 < G + σ_sq / (B : ℝ)) :
    sgdNetProgress η S_loc G σ_sq B ≤
      sgdNetProgress (noiseAwareStep S_loc G σ_sq B) S_loc G σ_sq B := by
  have hdiff := sgdNetProgress_sub_eq_sq η S_loc G σ_sq B hS hV
  have hnonneg : 0 ≤ (S_loc * (G + σ_sq / (B : ℝ)) / 2) *
      (η - noiseAwareStep S_loc G σ_sq B) ^ 2 := by positivity
  linarith

/-- **Monotonicity of Mini-Batch SGD Progress in Batch Size (`sgdNetProgress_mono_batchSize`)**:
Increasing the mini-batch size `0 < B₁ ≤ B₂` with `0 ≤ S_loc` and `0 ≤ σ_sq` monotonically
increases `sgdNetProgress η S_loc G σ_sq B` for any learning rate `η`. -/
theorem sgdNetProgress_mono_batchSize (η S_loc G σ_sq : ℝ) {B₁ B₂ : ℕ}
    (hB₁ : 0 < B₁) (hB : B₁ ≤ B₂) (hS : 0 ≤ S_loc) (hσ : 0 ≤ σ_sq) :
    sgdNetProgress η S_loc G σ_sq B₁ ≤ sgdNetProgress η S_loc G σ_sq B₂ := by
  unfold sgdNetProgress
  have hB₁_pos : (0 : ℝ) < B₁ := Nat.cast_pos.mpr hB₁
  have hB_le : (B₁ : ℝ) ≤ B₂ := Nat.cast_le.mpr hB
  have hdiv : σ_sq / (B₂ : ℝ) ≤ σ_sq / (B₁ : ℝ) :=
    div_le_div_of_nonneg_left hσ hB₁_pos hB_le
  have hcoef : 0 ≤ η ^ 2 * S_loc / 2 := by positivity
  have hmul : (η ^ 2 * S_loc / 2) * (σ_sq / (B₂ : ℝ)) ≤
      (η ^ 2 * S_loc / 2) * (σ_sq / (B₁ : ℝ)) :=
    mul_le_mul_of_nonneg_left hdiv hcoef
  linarith

/-! ## 2. The Critical Batch Size & Linear Scaling Law -/

/-- **Critical Batch Size Ratio (`criticalBatchRatio`)**:
The noise-to-signal ratio `B_crit = σ_sq / G` at which mini-batch gradient variance `σ_sq / B`
equals the squared true gradient norm `G = ‖g‖^2`. -/
def criticalBatchRatio (G σ_sq : ℝ) : ℝ :=
  σ_sq / G

/-- Alias for `criticalBatchRatio` representing the critical batch size `B_crit = σ_sq / G`. -/
abbrev criticalBatchSize (G σ_sq : ℝ) : ℝ :=
  criticalBatchRatio G σ_sq

/-- **Linear Scaling Decomposition of Optimal Learning Rate (`noiseAwareStep_eq_batch_efficiency`)**:
When `0 < B` and `0 < G` (with `0 ≤ σ_sq`), the optimal mini-batch learning rate decomposes into
the full-batch learning rate `1 / S_loc` times the batch efficiency factor `B / (B + σ_sq / G)`. -/
theorem noiseAwareStep_eq_batch_efficiency (S_loc G σ_sq : ℝ) (B : ℕ)
    (hB : 0 < B) (hG : 0 < G) (hS : 0 < S_loc) (hσ : 0 ≤ σ_sq) :
    noiseAwareStep S_loc G σ_sq B =
      (1 / S_loc) * ((B : ℝ) / ((B : ℝ) + σ_sq / G)) := by
  have hB0 : (0 : ℝ) < B := Nat.cast_pos.mpr hB
  have hB_ne : (B : ℝ) ≠ 0 := ne_of_gt hB0
  have hG_ne : G ≠ 0 := ne_of_gt hG
  have hS_ne : S_loc ≠ 0 := ne_of_gt hS
  have hden1 : G + σ_sq / (B : ℝ) ≠ 0 := ne_of_gt (by positivity)
  have hden2 : (B : ℝ) + σ_sq / G ≠ 0 := ne_of_gt (by positivity)
  unfold noiseAwareStep
  field_simp [hB_ne, hG_ne, hS_ne, hden1, hden2]

/-- **Optimal Mini-Batch Progress vs. Full-Batch Progress (`optimal_sgd_progress_vs_full_batch`)**:
When `0 < B`, `0 < G`, `0 < S_loc`, and `0 ≤ σ_sq`, the optimal mini-batch SGD loss reduction is
the exact full-batch loss reduction `G / (2 * S_loc)` multiplied by the batch efficiency factor
`B / (B + σ_sq / G)`. -/
theorem optimal_sgd_progress_vs_full_batch (S_loc G σ_sq : ℝ) (B : ℕ)
    (hB : 0 < B) (hG : 0 < G) (hS : 0 < S_loc) (hσ : 0 ≤ σ_sq) :
    sgdNetProgress (noiseAwareStep S_loc G σ_sq B) S_loc G σ_sq B =
      (G / (2 * S_loc)) * ((B : ℝ) / ((B : ℝ) + σ_sq / G)) := by
  have hB0 : (0 : ℝ) < B := Nat.cast_pos.mpr hB
  have hV : 0 < G + σ_sq / (B : ℝ) := by positivity
  rw [sgdNetProgress_noiseAwareStep_eq S_loc G σ_sq B hS hV]
  have hB_ne : (B : ℝ) ≠ 0 := ne_of_gt hB0
  have hG_ne : G ≠ 0 := ne_of_gt hG
  have hS_ne : S_loc ≠ 0 := ne_of_gt hS
  have hden1 : G + σ_sq / (B : ℝ) ≠ 0 := ne_of_gt hV
  have hden2 : (B : ℝ) + σ_sq / G ≠ 0 := ne_of_gt (by positivity)
  field_simp [hB_ne, hG_ne, hS_ne, hden1, hden2]

/-- **50% Efficiency at the Critical Batch Size (`critical_batch_half_efficiency`)**:
At the critical batch size `(B : ℝ) = σ_sq / G` (with `0 < B`, `0 < G`, `0 < S_loc`), the optimal
mini-batch learning rate is `1 / (2 * S_loc)` (half the full-batch learning rate) and mini-batch
SGD achieves **exactly 50% of the full-batch per-step loss reduction** (`G / (4 * S_loc)`). -/
theorem critical_batch_half_efficiency (S_loc G σ_sq : ℝ) (B : ℕ)
    (hB : 0 < B) (hG : 0 < G) (hS : 0 < S_loc)
    (hcrit : (B : ℝ) = σ_sq / G) :
    noiseAwareStep S_loc G σ_sq B = 1 / (2 * S_loc) ∧
      sgdNetProgress (noiseAwareStep S_loc G σ_sq B) S_loc G σ_sq B = G / (4 * S_loc) := by
  have hB0 : (0 : ℝ) < B := Nat.cast_pos.mpr hB
  have hB_ne : (B : ℝ) ≠ 0 := ne_of_gt hB0
  have hS_ne : S_loc ≠ 0 := ne_of_gt hS
  have hσ_pos : 0 < σ_sq := by
    have hdiv_pos : 0 < σ_sq / G := by linarith
    exact (div_pos_iff_of_pos_right hG).mp hdiv_pos
  have hlr := noiseAwareStep_eq_batch_efficiency S_loc G σ_sq B hB hG hS hσ_pos.le
  have hprog := optimal_sgd_progress_vs_full_batch S_loc G σ_sq B hB hG hS hσ_pos.le
  rw [← hcrit] at hlr hprog
  have hhalf : (B : ℝ) / ((B : ℝ) + (B : ℝ)) = 1 / 2 := by
    field_simp [hB_ne]
    ring
  rw [hhalf] at hlr hprog
  refine ⟨?_, ?_⟩
  · rw [hlr]
    ring
  · rw [hprog]
    ring

/-- **Small-Batch Linear Scaling Lower Bound (`small_batch_linear_scaling_lower_bound`)**:
In the small-batch / noise-dominated regime `(B : ℝ) ≤ σ_sq / G` (`0 < B`, `0 < G`, `0 < S_loc`,
`0 < σ_sq`), optimal mini-batch SGD progress is at least `(B : ℝ) * G ^ 2 / (4 * S_loc * σ_sq)`,
scaling **linearly with the batch size `B`**. -/
theorem small_batch_linear_scaling_lower_bound (S_loc G σ_sq : ℝ) (B : ℕ)
    (hB : 0 < B) (hG : 0 < G) (hS : 0 < S_loc) (hσ : 0 < σ_sq)
    (hsmall : (B : ℝ) ≤ σ_sq / G) :
    (B : ℝ) * G ^ 2 / (4 * S_loc * σ_sq) ≤
      sgdNetProgress (noiseAwareStep S_loc G σ_sq B) S_loc G σ_sq B := by
  have hB0 : (0 : ℝ) < B := Nat.cast_pos.mpr hB
  rw [optimal_sgd_progress_vs_full_batch S_loc G σ_sq B hB hG hS hσ.le]
  have hden_le : (B : ℝ) + σ_sq / G ≤ 2 * (σ_sq / G) := by linarith
  have hden_pos : 0 < (B : ℝ) + σ_sq / G := by positivity
  have hfrac : (B : ℝ) / (2 * (σ_sq / G)) ≤ (B : ℝ) / ((B : ℝ) + σ_sq / G) :=
    div_le_div_of_nonneg_left hB0.le hden_pos hden_le
  have hmul : (G / (2 * S_loc)) * ((B : ℝ) / (2 * (σ_sq / G))) ≤
      (G / (2 * S_loc)) * ((B : ℝ) / ((B : ℝ) + σ_sq / G)) :=
    mul_le_mul_of_nonneg_left hfrac (by positivity)
  have hid : (G / (2 * S_loc)) * ((B : ℝ) / (2 * (σ_sq / G))) =
      (B : ℝ) * G ^ 2 / (4 * S_loc * σ_sq) := by
    have hG_ne : G ≠ 0 := ne_of_gt hG
    have hS_ne : S_loc ≠ 0 := ne_of_gt hS
    have hσ_ne : σ_sq ≠ 0 := ne_of_gt hσ
    field_simp [hG_ne, hS_ne, hσ_ne]
    ring
  linarith

/-- **Linear Scaling Regime Upper and Lower Sandwich (`linear_scaling_regime_progress`)**:
For any batch size `0 < B` in the linear-scaling regime `(B : ℝ) ≤ criticalBatchRatio G σ_sq`,
optimal mini-batch SGD progress is sandwiched within a factor of 2 of the exact linear-in-`B`
noise-limited rate `(B : ℝ) * G^2 / (2 * S_loc * σ_sq)`:
`(B * G^2) / (4 * S_loc * σ_sq) ≤ progress(B) ≤ (B * G^2) / (2 * S_loc * σ_sq)`. -/
theorem linear_scaling_regime_progress (S_loc G σ_sq : ℝ) (B : ℕ)
    (hB : 0 < B) (hG : 0 < G) (hS : 0 < S_loc) (hσ : 0 < σ_sq)
    (hsmall : (B : ℝ) ≤ criticalBatchRatio G σ_sq) :
    (B : ℝ) * G ^ 2 / (4 * S_loc * σ_sq) ≤
        sgdNetProgress (noiseAwareStep S_loc G σ_sq B) S_loc G σ_sq B ∧
      sgdNetProgress (noiseAwareStep S_loc G σ_sq B) S_loc G σ_sq B ≤
        (B : ℝ) * G ^ 2 / (2 * S_loc * σ_sq) := by
  refine ⟨small_batch_linear_scaling_lower_bound S_loc G σ_sq B hB hG hS hσ hsmall, ?_⟩
  have hB0 : (0 : ℝ) < B := Nat.cast_pos.mpr hB
  rw [optimal_sgd_progress_vs_full_batch S_loc G σ_sq B hB hG hS hσ.le]
  have hcrit_pos : 0 < σ_sq / G := by positivity
  have hden_ge : σ_sq / G ≤ (B : ℝ) + σ_sq / G := by linarith
  have hfrac : (B : ℝ) / ((B : ℝ) + σ_sq / G) ≤ (B : ℝ) / (σ_sq / G) :=
    div_le_div_of_nonneg_left hB0.le hcrit_pos hden_ge
  have hmul : (G / (2 * S_loc)) * ((B : ℝ) / ((B : ℝ) + σ_sq / G)) ≤
      (G / (2 * S_loc)) * ((B : ℝ) / (σ_sq / G)) :=
    mul_le_mul_of_nonneg_left hfrac (by positivity)
  have hid : (G / (2 * S_loc)) * ((B : ℝ) / (σ_sq / G)) =
      (B : ℝ) * G ^ 2 / (2 * S_loc * σ_sq) := by
    have hG_ne : G ≠ 0 := ne_of_gt hG
    have hS_ne : S_loc ≠ 0 := ne_of_gt hS
    have hσ_ne : σ_sq ≠ 0 := ne_of_gt hσ
    field_simp [hG_ne, hS_ne, hσ_ne]
  linarith

/-- **Exact Batch-Size Doubling Speedup Law (`batch_size_doubling_speedup`)**:
Doubling the batch size from `B` to `2 * B` (`0 < B`, `0 < G`, `0 < S_loc`, `0 ≤ σ_sq`) increases
the optimal per-step SGD progress by the exact factor
`2 * (B + B_crit) / (2 * B + B_crit) ∈ [1, 2]`, which approaches `2` (ideal linear speedup) when
`B ≪ B_crit` and approaches `1` (diminishing returns) when `B ≫ B_crit`. -/
theorem batch_size_doubling_speedup (S_loc G σ_sq : ℝ) (B : ℕ)
    (hB : 0 < B) (hG : 0 < G) (hS : 0 < S_loc) (hσ : 0 ≤ σ_sq) :
    sgdNetProgress (noiseAwareStep S_loc G σ_sq (2 * B)) S_loc G σ_sq (2 * B) =
      ((2 * ((B : ℝ) + σ_sq / G)) / (2 * (B : ℝ) + σ_sq / G)) *
        sgdNetProgress (noiseAwareStep S_loc G σ_sq B) S_loc G σ_sq B := by
  have h2B : 0 < 2 * B := Nat.mul_pos (by decide) hB
  rw [optimal_sgd_progress_vs_full_batch S_loc G σ_sq (2 * B) h2B hG hS hσ,
    optimal_sgd_progress_vs_full_batch S_loc G σ_sq B hB hG hS hσ]
  push_cast
  have hB0 : (0 : ℝ) < B := Nat.cast_pos.mpr hB
  have hden1 : (B : ℝ) + σ_sq / G ≠ 0 := ne_of_gt (by positivity)
  have hden2 : 2 * (B : ℝ) + σ_sq / G ≠ 0 := ne_of_gt (by positivity)
  have hS_ne : S_loc ≠ 0 := ne_of_gt hS
  field_simp [hden1, hden2, hS_ne]

/-! ## 3. Multi-Step PL Convergence of Mini-Batch SGD to the Variance Noise Horizon -/

/-- **SGD Variance Noise Horizon (`sgdNoiseFloor`)**:
Under local sharpness `S_loc`, PL constant `μ`, constant learning rate `η`, gradient variance
`σ_sq`, and batch size `B`, the asymptotic noise ball radius of mini-batch SGD is
`sgdNoiseFloor η S_loc μ σ_sq B = (η * S_loc * σ_sq) / (2 * μ * B)`. -/
def sgdNoiseFloor (η S_loc μ σ_sq : ℝ) (B : ℕ) : ℝ :=
  (η * S_loc * σ_sq) / (2 * μ * (B : ℝ))

/-- Nonnegativity of `sgdNoiseFloor` when `0 ≤ η`, `0 ≤ S_loc`, `0 ≤ μ`, and `0 ≤ σ_sq`. -/
theorem sgdNoiseFloor_nonneg {η S_loc μ σ_sq : ℝ} (B : ℕ)
    (hη : 0 ≤ η) (hS : 0 ≤ S_loc) (hμ : 0 ≤ μ) (hσ : 0 ≤ σ_sq) :
    0 ≤ sgdNoiseFloor η S_loc μ σ_sq B := by
  unfold sgdNoiseFloor
  positivity

/-- **One-Step PL Contraction with Variance Penalty (`sgd_pl_one_step_recurrence`)**:
If at step `t`, `E_L_next ≤ E_L - sgdNetProgress η S_loc G σ_sq B`, the step size satisfies
`0 ≤ η ≤ 1 / S_loc` (`0 < S_loc`), and the Polyak-Łojasiewicz condition `2 * μ * (E_L - floor) ≤ G`
holds (with `0 ≤ G`), then `1 - η * S_loc / 2 ≥ 1 / 2` implies:
`E_L_next - floor ≤ (1 - η * μ) * (E_L - floor) + (η^2 * S_loc / 2) * (σ_sq / B)`. -/
theorem sgd_pl_one_step_recurrence (E_L E_L_next floor η S_loc μ G σ_sq : ℝ) (B : ℕ)
    (hS : 0 < S_loc) (hη0 : 0 ≤ η) (hη_le : η ≤ 1 / S_loc)
    (hG : 0 ≤ G) (hPL : 2 * μ * (E_L - floor) ≤ G)
    (hstep : E_L_next ≤ E_L - sgdNetProgress η S_loc G σ_sq B) :
    E_L_next - floor ≤
      (1 - η * μ) * (E_L - floor) + (η ^ 2 * S_loc / 2) * (σ_sq / (B : ℝ)) := by
  have h1 : η * S_loc ≤ 1 := (le_div_iff₀ hS).mp hη_le
  have hhalf : 1 / 2 ≤ 1 - η * S_loc / 2 := by linarith
  have hcoeff : η / 2 ≤ η * (1 - η * S_loc / 2) := by
    have := mul_le_mul_of_nonneg_left hhalf hη0
    linarith
  have hprog_ge : (η / 2) * G ≤ stepProgress η S_loc G := by
    unfold stepProgress
    exact mul_le_mul_of_nonneg_right hcoeff hG
  have hpl_mul : (η / 2) * (2 * μ * (E_L - floor)) ≤ (η / 2) * G :=
    mul_le_mul_of_nonneg_left hPL (by positivity)
  unfold sgdNetProgress at hstep
  linarith

/-- Fixed-point identity for `sgdNoiseFloor`:
`(1 - η * μ) * sgdNoiseFloor + (η^2 * S_loc / 2) * (σ_sq / B) = sgdNoiseFloor`. -/
theorem sgdNoiseFloor_fixed_point (η S_loc μ σ_sq : ℝ) (B : ℕ)
    (hμ : μ ≠ 0) (hB : (B : ℝ) ≠ 0) :
    (1 - η * μ) * sgdNoiseFloor η S_loc μ σ_sq B +
        (η ^ 2 * S_loc / 2) * (σ_sq / (B : ℝ)) =
      sgdNoiseFloor η S_loc μ σ_sq B := by
  unfold sgdNoiseFloor
  field_simp [hμ, hB]
  ring

/-- **Multi-Step PL Convergence of Mini-Batch SGD to the Variance Noise Horizon
(`sgd_pl_noise_ball_convergence`)**:
Along any mini-batch SGD trajectory where each step satisfies the PL variance recurrence
`E_L (t + 1) - floor ≤ (1 - η * μ) * (E_L t - floor) + (η^2 * S_loc / 2) * (σ_sq / B)`
with `0 < η`, `0 < μ`, `η * μ < 1`, `0 ≤ S_loc`, `0 ≤ σ_sq`, and `0 < B`, the expected
optimality gap at horizon `T : ℕ` satisfies:
`E_L T - floor ≤ (1 - η * μ) ^ T * (E_L 0 - floor) + sgdNoiseFloor η S_loc μ σ_sq B`. -/
theorem sgd_pl_noise_ball_convergence (E_L : ℕ → ℝ) (floor η S_loc μ σ_sq : ℝ) (B : ℕ)
    (hη : 0 < η) (hμ : 0 < μ) (hημ : η * μ < 1)
    (hS : 0 ≤ S_loc) (hσ : 0 ≤ σ_sq) (hB : 0 < B)
    (hrec : ∀ t, E_L (t + 1) - floor ≤
      (1 - η * μ) * (E_L t - floor) + (η ^ 2 * S_loc / 2) * (σ_sq / (B : ℝ)))
    (T : ℕ) :
    E_L T - floor ≤
      (1 - η * μ) ^ T * (E_L 0 - floor) + sgdNoiseFloor η S_loc μ σ_sq B := by
  have hB0 : (0 : ℝ) < B := Nat.cast_pos.mpr hB
  have hrate_nonneg : 0 ≤ 1 - η * μ := by linarith
  have hfloor_nonneg : 0 ≤ sgdNoiseFloor η S_loc μ σ_sq B :=
    sgdNoiseFloor_nonneg B hη.le hS hμ.le hσ
  have hfp := sgdNoiseFloor_fixed_point η S_loc μ σ_sq B (ne_of_gt hμ) (ne_of_gt hB0)
  have h_excess : E_L T - floor - sgdNoiseFloor η S_loc μ σ_sq B ≤
      (1 - η * μ) ^ T * (E_L 0 - floor - sgdNoiseFloor η S_loc μ σ_sq B) := by
    induction T with
    | zero =>
      simp
    | succ t ih =>
      have hstep := hrec t
      calc
        E_L (t + 1) - floor - sgdNoiseFloor η S_loc μ σ_sq B
          ≤ (1 - η * μ) * (E_L t - floor) + (η ^ 2 * S_loc / 2) * (σ_sq / (B : ℝ)) -
              sgdNoiseFloor η S_loc μ σ_sq B := by linarith
        _ = (1 - η * μ) * (E_L t - floor - sgdNoiseFloor η S_loc μ σ_sq B) := by
          linarith [hfp]
        _ ≤ (1 - η * μ) *
              ((1 - η * μ) ^ t * (E_L 0 - floor - sgdNoiseFloor η S_loc μ σ_sq B)) :=
          mul_le_mul_of_nonneg_left ih hrate_nonneg
        _ = (1 - η * μ) ^ (t + 1) * (E_L 0 - floor - sgdNoiseFloor η S_loc μ σ_sq B) := by
          rw [pow_succ', mul_assoc]
  have h_pow_floor : 0 ≤ (1 - η * μ) ^ T * sgdNoiseFloor η S_loc μ σ_sq B :=
    mul_nonneg (pow_nonneg hrate_nonneg T) hfloor_nonneg
  linarith

/-- **Explicit Two-Layer MLP Mini-Batch SGD Noise Horizon (`mlp_sgd_noise_floor_proportional_to_inv_batch`)**:
Plugging the derived two-layer MLP local sharpness `S_loc = mlpLocalSharpness m n lam A L M E R`
into `sgdNoiseFloor` shows that the MLP mini-batch SGD noise floor is directly proportional to
`(B : ℝ)⁻¹` and satisfies the multi-step PL convergence bound. -/
theorem mlp_sgd_noise_floor_proportional_to_inv_batch (m n : ℕ)
    (E_L : ℕ → ℝ) (floor η lam μ A L M E R σ_sq : ℝ) (B : ℕ)
    (hη : 0 < η) (hlam : 0 < lam) (hμ : 0 < μ) (hημ : η * μ < 1)
    (hA : 0 ≤ A) (hM : 0 ≤ M) (hE : 0 ≤ E) (hR : 0 ≤ R)
    (hσ : 0 ≤ σ_sq) (hB : 0 < B)
    (hrec : ∀ t, E_L (t + 1) - floor ≤
      (1 - η * μ) * (E_L t - floor) +
        (η ^ 2 * mlpLocalSharpness m n lam A L M E R / 2) * (σ_sq / (B : ℝ)))
    (T : ℕ) :
    let S_mlp := mlpLocalSharpness m n lam A L M E R
    sgdNoiseFloor η S_mlp μ σ_sq B =
        ((η * S_mlp * σ_sq) / (2 * μ)) * (B : ℝ)⁻¹ ∧
      E_L T - floor ≤
        (1 - η * μ) ^ T * (E_L 0 - floor) +
          ((η * S_mlp * σ_sq) / (2 * μ)) * (B : ℝ)⁻¹ := by
  intro S_mlp
  have hS_pos : 0 < S_mlp := mlpLocalSharpness_pos m n hlam hA hM hE hR
  have hinv : sgdNoiseFloor η S_mlp μ σ_sq B =
      ((η * S_mlp * σ_sq) / (2 * μ)) * (B : ℝ)⁻¹ := by
    unfold sgdNoiseFloor
    ring
  refine ⟨hinv, ?_⟩
  rw [← hinv]
  exact sgd_pl_noise_ball_convergence E_L floor η S_mlp μ σ_sq B
    hη hμ hημ hS_pos.le hσ hB hrec T

end OpenTheory.StochasticSharpnessSGD
