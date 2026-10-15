import OpenTheory.Optimization.ResidualPreconditionedMomentum
import OpenTheory.Optimization.GlobalSaddlePL

/-!
# Observable-to-Oracle Plug-In Estimator Gap (Pure Quadratic Regret & Safe Upper-Confidence Descent) and the Self-Tuning `CertiRPM` Optimizer

In `OpenTheory/Optimization/ResidualPreconditionedMomentum.lean` and
`OpenTheory/Optimization/GlobalSaddlePL.lean`, we established that the Residual-Preconditioned
Momentum (RPM) coordinate learning rate
`rpmOptimalStep S g β σ_coord B i = (g i)^2 / (S i * totalCoordMoment g β σ_coord B i)`
uniquely maximizes one-step expected net progress in a Polyak-Łojasiewicz (PL) basin, while a
small-batch noisy step escapes strict saddles along negative-curvature directions.

In practice, however, an optimizer does not have access to the exact oracle sharpness `S_i` or
exact population moments; it must plug in finite-sample empirical observables `(κ_hat, S_hat, M_hat)`
estimated from live mini-batches. Two fundamental questions arise:
1. **How sensitive is net optimization progress to estimation error in the plug-in step size or
   plug-in sharpness?**
2. **How can an autonomous optimizer guarantee 100% safe non-divergent descent under noisy
   sharpness estimates while automatically switching between saddle escape and PL basin acceleration
   with zero manual hyperparameter schedules?**

This module answers both questions from first principles with zero `sorry`:

1. **Bridge 1A — Plug-In Estimator Quadratic Regret & Zero First-Order Sensitivity
   (`plugin_step_quadratic_regret_identity`, `plugin_step_uniform_error_regret_bound`,
   `pluginSharpnessStep`, `pluginSharpnessStep_sub_oracle_eq`,
   `pluginSharpness_exact_progress_ratio`)**:
   - Because `rpmOptimalStep` is the interior stationary point of the quadratic progress functional
     `optimizerNetProgress`, the first-order sensitivity to step-size estimation error
     `δ_i = η_hat_i - η*_i` vanishes identically! The suboptimality gap is a **pure quadratic form**
     `∑ i, (S_i * M_i / 2) * (η_hat_i - η*_i)^2`, which is `O(δ_η^2)` under uniform error
     `|η_hat_i - η*_i| ≤ δ_η`.
   - When a practitioner plugs an estimated sharpness profile `S_hat : Fin p → ℝ` into the RPM rule
     (`pluginSharpnessStep`), the relative step error is `(S_i - S_hat_i) / S_hat_i`, yielding the
     **Exact Multiplicative Sharpness Regret Law**: each coordinate achieves fraction
     `(2 * S_hat_i * S_i - S_i^2) / S_hat_i^2 = 1 - ((S_hat_i - S_i) / S_hat_i)^2` of the oracle
     maximum progress. For example, a `10%` sharpness estimation error (`S_hat_i = 1.1 * S_i`)
     loses only `(0.1 / 1.1)^2 = 1 / 121 < 0.83%` of oracle progress!

2. **Bridge 1B — Upper-Confidence Sharpness Estimator Guarantees 100% Safe Non-Divergent Descent
   (`upper_confidence_sharpness_interval`, `upper_confidence_multiplier_bounds`,
   `upper_confidence_sharpness_guarantees_positive_progress`,
   `upper_confidence_progress_lower_bound`)**:
   - Underestimating sharpness (`S_hat_i < S_i / 2`) causes `2 * S_hat_i * S_i - S_i^2 < 0`
     (divergent overshoot!).
   - Conversely, using an **Upper-Confidence Sharpness Bound** `S_hat_i = S_emp_i + δ_{S,i} ≥ S_i`
     guarantees `S_i / S_hat_i ≤ (2 * S_hat_i * S_i - S_i^2) / S_hat_i^2 ≤ 1` in every coordinate,
     ensuring **100% safe non-divergent positive descent** (`0 ≤ optimizerNetProgress`) and
     retaining at least `c_min = min_i (S_i / S_hat_i)` of the oracle maximum progress.

3. **System 1 — Zero-Hyperparameter Self-Tuning Optimizer (`CertiRPM`)
   (`certiRPMBatchSize`, `certiRPMStep`, `pluginSharpnessStep_strict_accel_from_sharpness_drop`,
   `certiRPM_master_self_tuning_guarantee`)**:
   - Formalizes the two-regime state machine `CertiRPM` driven by live empirical curvature
     `κ_hat : ℝ`, residual norm `R_hat : ℝ`, and upper-confidence sharpness `S_hat(R_hat)`:
     * When `κ_hat < 0` (strict saddle detected), `CertiRPM` selects small batch
       `B_escape < B_basin` and step `η_escape`, driving strictly faster exponential unstable-mode
       escape than large-batch or deterministic updates.
     * When `κ_hat ≥ 0` (PL basin detected), `CertiRPM` switches to large batch `B_basin`,
       EMA momentum `β ∈ (0, 1)`, and `pluginSharpnessStep`, guaranteeing safe positive progress,
       automatic step-size acceleration `η_hat_i(R_t) > η_hat_i(R_0)` as residual `R_t < R_0`
       drops, and a strictly smaller stochastic PL noise floor.
-/

noncomputable section

open scoped BigOperators RealInnerProductSpace

namespace OpenTheory.CertiRPM

open OpenTheory.LocalSharpnessDescent
open OpenTheory.PreconditionedDescent
open OpenTheory.StochasticSharpnessSGD
open OpenTheory.MomentumAdamTrajectory
open OpenTheory.MLPLocalSharpness
open OpenTheory.ResidualPreconditionedMomentum
open OpenTheory.GlobalSaddlePL

variable {p : ℕ}

/-! ## Section 1: Bridge 1A — Plug-In Estimator Quadratic Regret (Zero First-Order Sensitivity) -/

/-- **Plug-In Step Quadratic Regret Identity (`plugin_step_quadratic_regret_identity`)**:
For any finite-sample plug-in coordinate step vector `η_hat : Fin p → ℝ`, the suboptimality gap
relative to the oracle `rpmOptimalStep S g β σ_coord B` has **zero first-order (linear) term** in
the step estimation error `δ_i = η_hat i - rpmOptimalStep S g β σ_coord B i` and equals the exact
pure quadratic form:
`∑ i : Fin p, (S i * totalCoordMoment g β σ_coord B i / 2) * (η_hat i - rpmOptimalStep S g β σ_coord B i) ^ 2`. -/
theorem plugin_step_quadratic_regret_identity (η_hat S : Fin p → ℝ) (g : Parameters p) (β : ℝ)
    (σ_coord : Fin p → ℝ) (B : ℝ)
    (hS : ∀ i, 0 < S i) (hM : ∀ i, 0 < totalCoordMoment g β σ_coord B i) :
    optimizerNetProgress (rpmOptimalStep S g β σ_coord B) S g β σ_coord B -
        optimizerNetProgress η_hat S g β σ_coord B =
      ∑ i : Fin p,
        (S i * totalCoordMoment g β σ_coord B i / 2) *
          (η_hat i - rpmOptimalStep S g β σ_coord B i) ^ 2 :=
  rpm_variational_sub_eq_sum_sq η_hat S g β σ_coord B hS hM

/-- **Uniform Step-Error Quadratic Regret Bound (`plugin_step_uniform_error_regret_bound`)**:
If every coordinate step estimate `η_hat i` satisfies `|η_hat i - rpmOptimalStep S g β σ_coord B i| ≤ δ_η`
(with `0 ≤ δ_η`), then the total net progress regret relative to the oracle `rpmOptimalStep` is
bounded by `(δ_η ^ 2 / 2) * ∑ i : Fin p, S i * totalCoordMoment g β σ_coord B i`, vanishing
quadratically as `O(δ_η ^ 2)`. -/
theorem plugin_step_uniform_error_regret_bound (η_hat S : Fin p → ℝ) (g : Parameters p) (β : ℝ)
    (σ_coord : Fin p → ℝ) (B δ_η : ℝ)
    (hS : ∀ i, 0 < S i) (hM : ∀ i, 0 < totalCoordMoment g β σ_coord B i)
    (_hδ : 0 ≤ δ_η)
    (herr : ∀ i, |η_hat i - rpmOptimalStep S g β σ_coord B i| ≤ δ_η) :
    optimizerNetProgress (rpmOptimalStep S g β σ_coord B) S g β σ_coord B -
        optimizerNetProgress η_hat S g β σ_coord B ≤
      (δ_η ^ 2 / 2) * ∑ i : Fin p, S i * totalCoordMoment g β σ_coord B i := by
  rw [plugin_step_quadratic_regret_identity η_hat S g β σ_coord B hS hM, Finset.mul_sum]
  refine Finset.sum_le_sum fun i _ => ?_
  have hSi : 0 ≤ S i := (hS i).le
  have hMi : 0 ≤ totalCoordMoment g β σ_coord B i := (hM i).le
  have hsq : (η_hat i - rpmOptimalStep S g β σ_coord B i) ^ 2 ≤ δ_η ^ 2 := by
    have habs := herr i
    have h1 : |η_hat i - rpmOptimalStep S g β σ_coord B i| ^ 2 ≤ δ_η ^ 2 :=
      pow_le_pow_left₀ (abs_nonneg _) habs 2
    simpa [sq_abs] using h1
  have hcoeff : 0 ≤ S i * totalCoordMoment g β σ_coord B i / 2 := by positivity
  calc
    (S i * totalCoordMoment g β σ_coord B i / 2) *
        (η_hat i - rpmOptimalStep S g β σ_coord B i) ^ 2
      ≤ (S i * totalCoordMoment g β σ_coord B i / 2) * δ_η ^ 2 :=
        mul_le_mul_of_nonneg_left hsq hcoeff
    _ = (δ_η ^ 2 / 2) * (S i * totalCoordMoment g β σ_coord B i) := by ring

/-- **Plug-In Sharpness Coordinate Step (`pluginSharpnessStep`)**:
The practical RPM coordinate learning rate obtained by plugging an empirical or upper-confidence
sharpness estimate `S_hat : Fin p → ℝ` into the variational optimal formula:
`η_hat_i = (g i) ^ 2 / (S_hat i * totalCoordMoment g β σ_coord B i)`. -/
def pluginSharpnessStep (S_hat : Fin p → ℝ) (g : Parameters p) (β : ℝ)
    (σ_coord : Fin p → ℝ) (B : ℝ) (i : Fin p) : ℝ :=
  (g i) ^ 2 / (S_hat i * totalCoordMoment g β σ_coord B i)

/-- Equivalence between `pluginSharpnessStep` and `rpmOptimalStep` applied to `S_hat`. -/
@[simp]
theorem pluginSharpnessStep_eq_rpmOptimalStep (S_hat : Fin p → ℝ) (g : Parameters p) (β : ℝ)
    (σ_coord : Fin p → ℝ) (B : ℝ) :
    pluginSharpnessStep S_hat g β σ_coord B = rpmOptimalStep S_hat g β σ_coord B :=
  rfl

/-- **Exact Relative Step-Error Identity for Plug-In Sharpness (`pluginSharpnessStep_sub_oracle_eq`)**:
When `S_hat i ≠ 0`, `S i ≠ 0`, and `totalCoordMoment g β σ_coord B i ≠ 0`, the estimation error of
`pluginSharpnessStep` relative to the oracle `rpmOptimalStep` is proportional to the relative
sharpness error `(S i - S_hat i) / S_hat i`:
`pluginSharpnessStep S_hat g β σ_coord B i - rpmOptimalStep S g β σ_coord B i = ((S i - S_hat i) / S_hat i) * rpmOptimalStep S g β σ_coord B i`. -/
theorem pluginSharpnessStep_sub_oracle_eq (S_hat S : Fin p → ℝ) (g : Parameters p) (β : ℝ)
    (σ_coord : Fin p → ℝ) (B : ℝ) (i : Fin p)
    (hS_hat : S_hat i ≠ 0) (hS : S i ≠ 0)
    (hM : totalCoordMoment g β σ_coord B i ≠ 0) :
    pluginSharpnessStep S_hat g β σ_coord B i - rpmOptimalStep S g β σ_coord B i =
      ((S i - S_hat i) / S_hat i) * rpmOptimalStep S g β σ_coord B i := by
  unfold pluginSharpnessStep rpmOptimalStep
  field_simp [hS_hat, hS, hM]

/-- **Exact Multiplicative Sharpness Regret Law (`pluginSharpness_exact_progress_ratio`)**:
When `∀ i, 0 < S_hat i`, `∀ i, 0 < S i`, and `∀ i, 0 < totalCoordMoment g β σ_coord B i`:
1. The exact suboptimality gap (regret) of `pluginSharpnessStep S_hat` relative to the oracle
   `rpmOptimalStep S` is
   `∑ i : Fin p, (((g i) ^ 2) ^ 2 / (2 * S i * totalCoordMoment g β σ_coord B i)) * ((S_hat i - S i) / S_hat i) ^ 2`.
2. The exact net progress achieved by `pluginSharpnessStep S_hat` on the true landscape `S` is
   `∑ i : Fin p, (((g i) ^ 2) ^ 2 / (2 * S i * totalCoordMoment g β σ_coord B i)) * ((2 * S_hat i * S i - S i ^ 2) / S_hat i ^ 2)`. -/
theorem pluginSharpness_exact_progress_ratio (S_hat S : Fin p → ℝ) (g : Parameters p) (β : ℝ)
    (σ_coord : Fin p → ℝ) (B : ℝ)
    (hS_hat : ∀ i, 0 < S_hat i) (hS : ∀ i, 0 < S i)
    (hM : ∀ i, 0 < totalCoordMoment g β σ_coord B i) :
    optimizerNetProgress (rpmOptimalStep S g β σ_coord B) S g β σ_coord B -
        optimizerNetProgress (pluginSharpnessStep S_hat g β σ_coord B) S g β σ_coord B =
      ∑ i : Fin p,
        (((g i) ^ 2) ^ 2 / (2 * S i * totalCoordMoment g β σ_coord B i)) *
          ((S_hat i - S i) / S_hat i) ^ 2 ∧
    optimizerNetProgress (pluginSharpnessStep S_hat g β σ_coord B) S g β σ_coord B =
      ∑ i : Fin p,
        (((g i) ^ 2) ^ 2 / (2 * S i * totalCoordMoment g β σ_coord B i)) *
          ((2 * S_hat i * S i - S i ^ 2) / S_hat i ^ 2) := by
  refine ⟨?_, ?_⟩
  · rw [plugin_step_quadratic_regret_identity (pluginSharpnessStep S_hat g β σ_coord B)
      S g β σ_coord B hS hM]
    refine Finset.sum_congr rfl fun i _ => ?_
    have hS_hat_i : S_hat i ≠ 0 := ne_of_gt (hS_hat i)
    have hS_i : S i ≠ 0 := ne_of_gt (hS i)
    have hM_i : totalCoordMoment g β σ_coord B i ≠ 0 := ne_of_gt (hM i)
    rw [pluginSharpnessStep_sub_oracle_eq S_hat S g β σ_coord B i hS_hat_i hS_i hM_i]
    unfold rpmOptimalStep
    field_simp [hS_hat_i, hS_i, hM_i]
    ring
  · rw [optimizerNetProgress_eq_coord_sum]
    unfold pluginSharpnessStep
    refine Finset.sum_congr rfl fun i _ => ?_
    have hS_hat_i : S_hat i ≠ 0 := ne_of_gt (hS_hat i)
    have hS_i : S i ≠ 0 := ne_of_gt (hS i)
    have hM_i : totalCoordMoment g β σ_coord B i ≠ 0 := ne_of_gt (hM i)
    field_simp [hS_hat_i, hS_i, hM_i]

/-! ## Section 2: Bridge 1B — Upper-Confidence Sharpness Estimator Guarantees 100% Safe Non-Divergent Descent -/

/-- **Upper-Confidence Sharpness Estimator (`upperConfidenceSharpness`)**:
Given an empirical sharpness estimate `S_emp : Fin p → ℝ` and confidence radius `δ_S : Fin p → ℝ`,
the upper-confidence sharpness estimator is `S_hat⁺_i = S_emp i + δ_S i`. -/
def upperConfidenceSharpness (S_emp δ_S : Fin p → ℝ) (i : Fin p) : ℝ :=
  S_emp i + δ_S i

/-- **Upper-Confidence Sharpness Sandwich Interval (`upper_confidence_sharpness_interval`)**:
Whenever the empirical sharpness estimate satisfies `|S_emp i - S i| ≤ δ_S i`, the upper-confidence
estimator `upperConfidenceSharpness S_emp δ_S i = S_emp i + δ_S i` brackets the true sharpness:
`S i ≤ upperConfidenceSharpness S_emp δ_S i ≤ S i + 2 * δ_S i`. -/
theorem upper_confidence_sharpness_interval (S_emp S δ_S : Fin p → ℝ) (i : Fin p)
    (herr : |S_emp i - S i| ≤ δ_S i) :
    S i ≤ upperConfidenceSharpness S_emp δ_S i ∧
      upperConfidenceSharpness S_emp δ_S i ≤ S i + 2 * δ_S i := by
  unfold upperConfidenceSharpness
  have h := abs_le.mp herr
  refine ⟨by linarith, by linarith⟩

/-- **Pointwise Multiplier Bounds Under Upper-Confidence Sharpness (`upper_confidence_multiplier_bounds`)**:
Whenever `0 < s ≤ s_hat`, the multiplicative progress factor `(2 * s_hat * s - s ^ 2) / s_hat ^ 2`
satisfies the exact algebraic identity
`1 - ((s_hat - s) / s_hat) ^ 2 = (2 * s_hat * s - s ^ 2) / s_hat ^ 2`
and is sandwiched between the conservative linear ratio `s / s_hat` and `1`:
`0 < s / s_hat ≤ (2 * s_hat * s - s ^ 2) / s_hat ^ 2 ≤ 1`. -/
theorem upper_confidence_multiplier_bounds {s s_hat : ℝ}
    (hs : 0 < s) (hle : s ≤ s_hat) :
    (2 * s_hat * s - s ^ 2) / s_hat ^ 2 = 1 - ((s_hat - s) / s_hat) ^ 2 ∧
      0 < s / s_hat ∧
      s / s_hat ≤ (2 * s_hat * s - s ^ 2) / s_hat ^ 2 ∧
      (2 * s_hat * s - s ^ 2) / s_hat ^ 2 ≤ 1 := by
  have hs_hat : 0 < s_hat := hs.trans_le hle
  have hs_hat_ne : s_hat ≠ 0 := ne_of_gt hs_hat
  have hid : (2 * s_hat * s - s ^ 2) / s_hat ^ 2 = 1 - ((s_hat - s) / s_hat) ^ 2 := by
    field_simp [hs_hat_ne]
    ring
  have hpos_ratio : 0 < s / s_hat := div_pos hs hs_hat
  have hdiff_lower :
      (2 * s_hat * s - s ^ 2) / s_hat ^ 2 - s / s_hat =
        s * (s_hat - s) / s_hat ^ 2 := by
    field_simp [hs_hat_ne]
    ring
  have hsub_nonneg : 0 ≤ s_hat - s := sub_nonneg.mpr hle
  have hlower_nonneg : 0 ≤ s * (s_hat - s) / s_hat ^ 2 := by positivity
  have hsq_nonneg : 0 ≤ ((s_hat - s) / s_hat) ^ 2 := sq_nonneg _
  refine ⟨hid, hpos_ratio, by linarith, by linarith⟩

/-- **Upper-Confidence Sharpness Guarantees Positive Non-Divergent Descent
(`upper_confidence_sharpness_guarantees_positive_progress`)**:
Whenever `0 < S i ≤ S_hat i` and `0 < totalCoordMoment g β σ_coord B i` for all `i : Fin p`:
1. Every coordinate multiplier satisfies
   `S i / S_hat i ≤ (2 * S_hat i * S i - S i ^ 2) / S_hat i ^ 2 ≤ 1`,
2. `pluginSharpnessStep` is **guaranteed non-divergent**:
   `0 ≤ optimizerNetProgress (pluginSharpnessStep S_hat g β σ_coord B) S g β σ_coord B`,
3. `pluginSharpnessStep` achieves at least the coordinate-weighted linear ratio lower bound:
   `∑ i : Fin p, (((g i) ^ 2) ^ 2 / (2 * S i * totalCoordMoment g β σ_coord B i)) * (S i / S_hat i) ≤ optimizerNetProgress (pluginSharpnessStep S_hat g β σ_coord B) S g β σ_coord B`. -/
theorem upper_confidence_sharpness_guarantees_positive_progress
    (S_hat S : Fin p → ℝ) (g : Parameters p) (β : ℝ)
    (σ_coord : Fin p → ℝ) (B : ℝ)
    (hS : ∀ i, 0 < S i) (hle : ∀ i, S i ≤ S_hat i)
    (hM : ∀ i, 0 < totalCoordMoment g β σ_coord B i) :
    (∀ i, S i / S_hat i ≤ (2 * S_hat i * S i - S i ^ 2) / S_hat i ^ 2 ∧
      (2 * S_hat i * S i - S i ^ 2) / S_hat i ^ 2 ≤ 1) ∧
    ∑ i : Fin p,
        (((g i) ^ 2) ^ 2 / (2 * S i * totalCoordMoment g β σ_coord B i)) *
          (S i / S_hat i) ≤
      optimizerNetProgress (pluginSharpnessStep S_hat g β σ_coord B) S g β σ_coord B ∧
    0 ≤ optimizerNetProgress (pluginSharpnessStep S_hat g β σ_coord B) S g β σ_coord B := by
  have hS_hat : ∀ i, 0 < S_hat i := fun i => (hS i).trans_le (hle i)
  have hmult : ∀ i, S i / S_hat i ≤ (2 * S_hat i * S i - S i ^ 2) / S_hat i ^ 2 ∧
      (2 * S_hat i * S i - S i ^ 2) / S_hat i ^ 2 ≤ 1 := by
    intro i
    obtain ⟨_, _, h1, h2⟩ := upper_confidence_multiplier_bounds (hS i) (hle i)
    exact ⟨h1, h2⟩
  have hprog := (pluginSharpness_exact_progress_ratio S_hat S g β σ_coord B hS_hat hS hM).2
  have hsum_le :
      ∑ i : Fin p,
          (((g i) ^ 2) ^ 2 / (2 * S i * totalCoordMoment g β σ_coord B i)) *
            (S i / S_hat i) ≤
        optimizerNetProgress (pluginSharpnessStep S_hat g β σ_coord B) S g β σ_coord B := by
    rw [hprog]
    refine Finset.sum_le_sum fun i _ => ?_
    have hSi : 0 < S i := hS i
    have hMi : 0 < totalCoordMoment g β σ_coord B i := hM i
    have hcoeff : 0 ≤ ((g i) ^ 2) ^ 2 / (2 * S i * totalCoordMoment g β σ_coord B i) := by
      positivity
    exact mul_le_mul_of_nonneg_left (hmult i).1 hcoeff
  have hsum_nonneg :
      0 ≤ ∑ i : Fin p,
          (((g i) ^ 2) ^ 2 / (2 * S i * totalCoordMoment g β σ_coord B i)) *
            (S i / S_hat i) := by
    refine Finset.sum_nonneg fun i _ => ?_
    have hSi : 0 < S i := hS i
    have hS_hati : 0 < S_hat i := hS_hat i
    have hMi : 0 < totalCoordMoment g β σ_coord B i := hM i
    positivity
  exact ⟨hmult, hsum_le, hsum_nonneg.trans hsum_le⟩

/-- **Uniform Lower Bound on Progress Ratio Under Upper-Confidence Sharpness
(`upper_confidence_progress_lower_bound`)**:
If `0 < S i ≤ S_hat i` and `c_min ≤ S i / S_hat i` for all `i : Fin p`, then `pluginSharpnessStep`
achieves at least fraction `c_min` of the oracle maximum net progress:
`c_min * optimizerNetProgress (rpmOptimalStep S g β σ_coord B) S g β σ_coord B ≤ optimizerNetProgress (pluginSharpnessStep S_hat g β σ_coord B) S g β σ_coord B`. -/
theorem upper_confidence_progress_lower_bound
    (S_hat S : Fin p → ℝ) (g : Parameters p) (β : ℝ)
    (σ_coord : Fin p → ℝ) (B c_min : ℝ)
    (hS : ∀ i, 0 < S i) (hle : ∀ i, S i ≤ S_hat i)
    (hM : ∀ i, 0 < totalCoordMoment g β σ_coord B i)
    (hc : ∀ i, c_min ≤ S i / S_hat i) :
    c_min * optimizerNetProgress (rpmOptimalStep S g β σ_coord B) S g β σ_coord B ≤
      optimizerNetProgress (pluginSharpnessStep S_hat g β σ_coord B) S g β σ_coord B := by
  have h1 := (upper_confidence_sharpness_guarantees_positive_progress
    S_hat S g β σ_coord B hS hle hM).2.1
  have horacle := optimizerNetProgress_rpmOptimalStep_eq S g β σ_coord B hS hM
  have h2 :
      c_min * optimizerNetProgress (rpmOptimalStep S g β σ_coord B) S g β σ_coord B ≤
        ∑ i : Fin p,
          (((g i) ^ 2) ^ 2 / (2 * S i * totalCoordMoment g β σ_coord B i)) *
            (S i / S_hat i) := by
    rw [horacle, Finset.mul_sum]
    refine Finset.sum_le_sum fun i _ => ?_
    have hSi : 0 < S i := hS i
    have hMi : 0 < totalCoordMoment g β σ_coord B i := hM i
    have hcoeff : 0 ≤ ((g i) ^ 2) ^ 2 / (2 * S i * totalCoordMoment g β σ_coord B i) := by
      positivity
    calc
      c_min * (((g i) ^ 2) ^ 2 / (2 * S i * totalCoordMoment g β σ_coord B i))
        = (((g i) ^ 2) ^ 2 / (2 * S i * totalCoordMoment g β σ_coord B i)) * c_min := by ring
      _ ≤ (((g i) ^ 2) ^ 2 / (2 * S i * totalCoordMoment g β σ_coord B i)) *
            (S i / S_hat i) := mul_le_mul_of_nonneg_left (hc i) hcoeff
  exact h2.trans h1

/-! ## Section 3: System 1 — Zero-Hyperparameter Self-Tuning Optimizer (`CertiRPM`) -/

/-- **CertiRPM Two-Regime Self-Tuning Mini-Batch Size (`certiRPMBatchSize`)**:
Given empirical directional curvature `κ_hat : ℝ` at step `t`:
- If `κ_hat < 0` (strict saddle regime with negative curvature), `CertiRPM` selects the small,
  high-variance escape batch size `B_escape`.
- If `κ_hat ≥ 0` (local PL basin regime), `CertiRPM` selects the large, low-variance basin batch
  size `B_basin`. -/
def certiRPMBatchSize (κ_hat : ℝ) (B_escape B_basin : ℕ) : ℕ :=
  if κ_hat < 0 then B_escape else B_basin

/-- **CertiRPM Two-Regime Self-Tuning Coordinate Learning Rate (`certiRPMStep`)**:
Given empirical directional curvature `κ_hat : ℝ` at step `t`:
- If `κ_hat < 0` (strict saddle regime), `CertiRPM` applies the negative-curvature escape step
  size `η_escape`.
- If `κ_hat ≥ 0` (local PL basin regime), `CertiRPM` automatically switches to the
  upper-confidence plug-in RPM step `pluginSharpnessStep S_hat g β σ_coord B_basin i`. -/
def certiRPMStep (κ_hat : ℝ) (η_escape : ℝ) (S_hat : Fin p → ℝ) (g : Parameters p)
    (β : ℝ) (σ_coord : Fin p → ℝ) (B_basin : ℝ) (i : Fin p) : ℝ :=
  if κ_hat < 0 then η_escape else pluginSharpnessStep S_hat g β σ_coord B_basin i

@[simp]
theorem certiRPMBatchSize_saddle {κ_hat : ℝ} (B_escape B_basin : ℕ) (hκ : κ_hat < 0) :
    certiRPMBatchSize κ_hat B_escape B_basin = B_escape := by
  simp [certiRPMBatchSize, hκ]

@[simp]
theorem certiRPMBatchSize_basin {κ_hat : ℝ} (B_escape B_basin : ℕ) (hκ : 0 ≤ κ_hat) :
    certiRPMBatchSize κ_hat B_escape B_basin = B_basin := by
  simp [certiRPMBatchSize, not_lt.mpr hκ]

@[simp]
theorem certiRPMStep_saddle {κ_hat : ℝ} (η_escape : ℝ) (S_hat : Fin p → ℝ) (g : Parameters p)
    (β : ℝ) (σ_coord : Fin p → ℝ) (B_basin : ℝ) (i : Fin p) (hκ : κ_hat < 0) :
    certiRPMStep κ_hat η_escape S_hat g β σ_coord B_basin i = η_escape := by
  simp [certiRPMStep, hκ]

@[simp]
theorem certiRPMStep_basin {κ_hat : ℝ} (η_escape : ℝ) (S_hat : Fin p → ℝ) (g : Parameters p)
    (β : ℝ) (σ_coord : Fin p → ℝ) (B_basin : ℝ) (i : Fin p) (hκ : 0 ≤ κ_hat) :
    certiRPMStep κ_hat η_escape S_hat g β σ_coord B_basin i =
      pluginSharpnessStep S_hat g β σ_coord B_basin i := by
  simp [certiRPMStep, not_lt.mpr hκ]

/-- **Automatic Step-Size Acceleration of `pluginSharpnessStep` as Sharpness Drops
(`pluginSharpnessStep_strict_accel_from_sharpness_drop`)**:
As empirical residual reduction lowers the upper-confidence sharpness estimate from `S_hat_0 i` to
`S_hat_t i` (`0 < S_hat_t i < S_hat_0 i`) on an active coordinate `g i ≠ 0` with
`0 < totalCoordMoment g β σ_coord B i`, the plug-in step size strictly accelerates:
`pluginSharpnessStep S_hat_0 g β σ_coord B i < pluginSharpnessStep S_hat_t g β σ_coord B i`. -/
theorem pluginSharpnessStep_strict_accel_from_sharpness_drop
    (S_hat_t S_hat_0 : Fin p → ℝ) (g : Parameters p) (β : ℝ)
    (σ_coord : Fin p → ℝ) (B : ℝ) (i : Fin p)
    (hSt_pos : 0 < S_hat_t i) (hS_lt : S_hat_t i < S_hat_0 i)
    (hM : 0 < totalCoordMoment g β σ_coord B i) (hgi : g i ≠ 0) :
    pluginSharpnessStep S_hat_0 g β σ_coord B i <
      pluginSharpnessStep S_hat_t g β σ_coord B i := by
  unfold pluginSharpnessStep
  have hg2 : 0 < (g i) ^ 2 := sq_pos_of_ne_zero hgi
  have hden_pos : 0 < S_hat_t i * totalCoordMoment g β σ_coord B i := mul_pos hSt_pos hM
  have hden_lt :
      S_hat_t i * totalCoordMoment g β σ_coord B i <
        S_hat_0 i * totalCoordMoment g β σ_coord B i :=
    mul_lt_mul_of_pos_right hS_lt hM
  exact div_lt_div_of_pos_left hg2 hden_pos hden_lt

/-- **Master CertiRPM Self-Tuning Guarantee (`certiRPM_master_self_tuning_guarantee`)**:
Unifies the two self-tuning operating modes of `CertiRPM` into a single end-to-end guarantee:
1. **Automatic Saddle-Escape Mode (`κ_hat_saddle < 0`)**:
   `certiRPMBatchSize` selects `B_escape < B_basin` and `certiRPMStep` selects `η_escape`, driving
   strictly faster exponential unstable-mode variance growth
   `0 = saddleUnstableVarTrajectory η_escape γ_neg 0 0 T < saddleUnstableVarTrajectory η_escape γ_neg (σ_minus_sq / B_basin) 0 T < saddleUnstableVarTrajectory η_escape γ_neg (σ_minus_sq / B_escape) 0 T`.
2. **Automatic PL Basin Mode (`0 ≤ κ_hat_basin`) with Upper-Confidence Sharpness
   `0 < S i ≤ S_hat_t i < S_hat_0 i`**:
   `certiRPMBatchSize` switches to `B_basin` and `certiRPMStep` switches to `pluginSharpnessStep`,
   simultaneously guaranteeing:
   - **Safe Non-Divergent Descent**: `0 ≤ optimizerNetProgress (certiRPMStep κ_hat_basin ...) S g β σ_coord B_basin`,
   - **Exact Multiplicative Quadratic Regret**: suboptimality gap equals
     `∑ i, (((g i)^2)^2 / (2 * S i * M_i)) * ((S_hat_t i - S i) / S_hat_t i)^2`,
   - **Automatic Step-Size Acceleration from Residual Flattening**:
     `certiRPMStep κ_hat_basin η_escape S_hat_0 g β σ_coord B_basin i₀ < certiRPMStep κ_hat_basin η_escape S_hat_t g β σ_coord B_basin i₀`,
   - **Strictly Lowered Stochastic PL Noise Floor**: via `B_basin > B_escape` and `β ∈ (0, 1)`. -/
theorem certiRPM_master_self_tuning_guarantee
    {κ_hat_saddle κ_hat_basin η_escape γ_neg σ_minus_sq S_scalar μ σ_sq β : ℝ}
    {B_escape B_basin T : ℕ}
    (S S_hat_t S_hat_0 : Fin p → ℝ) (g : Parameters p) (σ_coord : Fin p → ℝ) (i₀ : Fin p)
    (hκ_saddle : κ_hat_saddle < 0) (hκ_basin : 0 ≤ κ_hat_basin)
    (hη_esc : 0 < η_escape) (hγ : 0 < γ_neg) (hσ_minus : 0 < σ_minus_sq)
    (hS_scalar : 0 < S_scalar) (hμ : 0 < μ) (hσ : 0 < σ_sq)
    (hβ0 : 0 < β) (hβ1 : β < 1)
    (hB_esc : 0 < B_escape) (hB_lt : B_escape < B_basin) (hT : 1 ≤ T)
    (hS : ∀ i, 0 < S i) (hle : ∀ i, S i ≤ S_hat_t i)
    (hM : ∀ i, 0 < totalCoordMoment g β σ_coord (B_basin : ℝ) i)
    (hS_drop : S_hat_t i₀ < S_hat_0 i₀) (hgi₀ : g i₀ ≠ 0) :
    -- (1) Mode selection identities
    (certiRPMBatchSize κ_hat_saddle B_escape B_basin = B_escape ∧
      certiRPMBatchSize κ_hat_basin B_escape B_basin = B_basin ∧
      certiRPMStep κ_hat_basin η_escape S_hat_t g β σ_coord (B_basin : ℝ) =
        pluginSharpnessStep S_hat_t g β σ_coord (B_basin : ℝ)) ∧
    -- (2) Saddle escape: deterministic trapped at 0, while B_escape strictly beats B_basin > 0
    (saddleUnstableVarTrajectory η_escape γ_neg 0 0 T = 0 ∧
      0 < saddleUnstableVarTrajectory η_escape γ_neg (σ_minus_sq / (B_basin : ℝ)) 0 T ∧
      saddleUnstableVarTrajectory η_escape γ_neg (σ_minus_sq / (B_basin : ℝ)) 0 T <
        saddleUnstableVarTrajectory η_escape γ_neg (σ_minus_sq / (B_escape : ℝ)) 0 T) ∧
    -- (3) Safe positive descent & exact quadratic regret in the PL basin
    (0 ≤ optimizerNetProgress
        (certiRPMStep κ_hat_basin η_escape S_hat_t g β σ_coord (B_basin : ℝ))
        S g β σ_coord (B_basin : ℝ) ∧
      optimizerNetProgress (rpmOptimalStep S g β σ_coord (B_basin : ℝ)) S g β σ_coord (B_basin : ℝ) -
          optimizerNetProgress
            (certiRPMStep κ_hat_basin η_escape S_hat_t g β σ_coord (B_basin : ℝ))
            S g β σ_coord (B_basin : ℝ) =
        ∑ i : Fin p,
          (((g i) ^ 2) ^ 2 / (2 * S i * totalCoordMoment g β σ_coord (B_basin : ℝ) i)) *
            ((S_hat_t i - S i) / S_hat_t i) ^ 2) ∧
    -- (4) Automatic step acceleration as residual/sharpness drops & lower PL noise floor
    (certiRPMStep κ_hat_basin η_escape S_hat_0 g β σ_coord (B_basin : ℝ) i₀ <
        certiRPMStep κ_hat_basin η_escape S_hat_t g β σ_coord (B_basin : ℝ) i₀ ∧
      ((1 - β) / (1 + β)) * sgdNoiseFloor η_escape S_scalar μ σ_sq B_basin <
        sgdNoiseFloor η_escape S_scalar μ σ_sq B_basin ∧
      sgdNoiseFloor η_escape S_scalar μ σ_sq B_basin <
        sgdNoiseFloor η_escape S_scalar μ σ_sq B_escape) := by
  have hstep_eq :
      certiRPMStep κ_hat_basin η_escape S_hat_t g β σ_coord (B_basin : ℝ) =
        pluginSharpnessStep S_hat_t g β σ_coord (B_basin : ℝ) := by
    ext i
    exact certiRPMStep_basin η_escape S_hat_t g β σ_coord (B_basin : ℝ) i hκ_basin
  have hS_hat_t : ∀ i, 0 < S_hat_t i := fun i => (hS i).trans_le (hle i)
  have hB_basin_pos : (0 : ℝ) < (B_basin : ℝ) :=
    Nat.cast_pos.mpr (hB_esc.trans hB_lt)
  have hσ_basin_pos : 0 < σ_minus_sq / (B_basin : ℝ) := div_pos hσ_minus hB_basin_pos
  have hesc_pos :=
    (saddle_noise_exponential_escape hη_esc hγ hσ_basin_pos T hT).2.2.2.2
  obtain ⟨hvar_lt, hmom_lt, hbatch_lt⟩ :=
    optimal_two_phase_noise_schedule hη_esc hγ hσ_minus hS_scalar hμ hσ hβ0 hβ1
      hB_esc hB_lt hT
  have hsafe :=
    (upper_confidence_sharpness_guarantees_positive_progress
      S_hat_t S g β σ_coord (B_basin : ℝ) hS hle hM).2.2
  have hregret :=
    (pluginSharpness_exact_progress_ratio
      S_hat_t S g β σ_coord (B_basin : ℝ) hS_hat_t hS hM).1
  have haccel :
      certiRPMStep κ_hat_basin η_escape S_hat_0 g β σ_coord (B_basin : ℝ) i₀ <
        certiRPMStep κ_hat_basin η_escape S_hat_t g β σ_coord (B_basin : ℝ) i₀ := by
    rw [certiRPMStep_basin _ _ _ _ _ _ _ hκ_basin,
      certiRPMStep_basin _ _ _ _ _ _ _ hκ_basin]
    exact pluginSharpnessStep_strict_accel_from_sharpness_drop
      S_hat_t S_hat_0 g β σ_coord (B_basin : ℝ) i₀ (hS_hat_t i₀) hS_drop (hM i₀) hgi₀
  refine ⟨⟨certiRPMBatchSize_saddle B_escape B_basin hκ_saddle,
      certiRPMBatchSize_basin B_escape B_basin hκ_basin, hstep_eq⟩,
    ⟨saddle_deterministic_trapped η_escape γ_neg T, hesc_pos, hvar_lt⟩,
    ⟨?_, ?_⟩,
    ⟨haccel, hmom_lt, hbatch_lt⟩⟩
  · rw [hstep_eq]
    exact hsafe
  · rw [hstep_eq]
    exact hregret

end OpenTheory.CertiRPM
