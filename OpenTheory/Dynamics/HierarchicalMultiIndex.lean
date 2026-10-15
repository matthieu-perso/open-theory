import OpenTheory.Data.Spectrum
import OpenTheory.Dynamics.MultiStepSpectralBias
import OpenTheory.Dynamics.GradientSpectralEmergence
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Ring

/-! # Non-Linear Multi-Index & Hierarchical Feature Learning (Beyond the Linearized Gram Matrix)

This module formalizes the theory of non-linear multi-index and compositional feature learning
with **information exponent** `s ≥ 1`, explaining why first-order Gram / NTK dynamics fail when
`s ≥ 2`, why higher-order tensor power iteration exhibits long saddle plateaus followed by
sudden emergence ("grokking"), and why deep hierarchical architectures exponentially beat
shallow networks on compositional targets:

1. **Information Exponent `s` and Vanishing Low-Order Correlations**:
   Defines `MultiIndexTarget` and the order-`m` Hermite / tensor alignment drift
   `tensorAlignmentDrift s m c_s α`, proving that all correlation orders `m < s` (in particular
   the linear Gram order `m = 1` when `s ≥ 2`) vanish identically
   (`tensorAlignmentDrift_eq_zero_of_lt`, `linear_gram_drift_zero_of_exponent_ge_two`), while
   the leading non-zero drift at `m = s` scales as `c_s * α ^ (s - 1)`
   (`tensorAlignmentDrift_at_exponent`). Defines the 1-step non-linear alignment update
   `multiIndexStep η c_s s α = α + η * c_s * α ^ s` and the `T`-step trajectory
   `multiIndexTrajectory η c_s s α₀ T`, proving non-negativity, positivity, and strict
   monotonicity along the trajectory.

2. **Information Exponent Dichotomy (`s = 1` Linear Fast Escape vs. `s ≥ 2` Saddle Plateau & Grokking)**:
   - For `s = 1` (linear Gram signal), the trajectory reduces to exact exponential / geometric
     growth `multiIndexTrajectory η c₁ 1 α₀ T = (1 + η * c₁) ^ T * α₀`
     (`multiIndexTrajectory_exponent_one_eq`), matching `MultiStepSpectralBias.modeAmpDisc`.
   - For `s ≥ 2` and small initial overlap `0 < α < 1`, the non-linear tensor drift `η * c_s * α ^ s`
     is strictly smaller than the linear drift `η * c_s * α` (`nonlinear_drift_lt_linear_drift_on_plateau`).
   - Defines the characteristic saddle-escape time scale
     `grokkingEscapeSteps η c_s α₀ s = 1 / (η * c_s * α₀ ^ (s - 1))`, proving strict monotonicity
     in `s` (`grokking_escape_time_strict_mono_in_exponent`) and the exact ratio law
     `(1 / α₀) ^ (s₂ - s₁) > 1` (`grokking_escape_time_ratio_eq`).
   - Establishes a rigorous **Two-Stage Grokking Trajectory Bound**: slow linear-in-`T` drift
     inside the saddle plateau (`multiIndex_slow_plateau_upper_bound`) followed by rapid
     post-threshold geometric blowup (`multiIndex_post_threshold_exponential_growth`).

3. **Compositional Hierarchy vs. Shallow Representation Bottleneck**:
   Defines `shallowSampleComplexity d s_total = d ^ s_total` and
   `hierarchicalSampleComplexity L d s_local = L * d ^ s_local`, together with
   `hierarchicalEscapeSteps L η c_s α₀ s_local = L * grokkingEscapeSteps η c_s α₀ s_local`.
   Proves strict sample-complexity separation (`hierarchical_vs_shallow_sample_complexity_separation`)
   and strict stagewise saddle-escape time separation (`hierarchical_stagewise_escape_time_lt_shallow`)
   when `L < (1 / α₀) ^ (s_total - s_local)`.

4. **Practical Prescription: Depth & Warmup Schedule for High-Information-Exponent Data**:
   Synthesizes the three-regime comparison (`optimal_depth_and_horizon_for_multi_index`) between
   linearized/NTK dynamics (zero first-order drift for `s ≥ 2`), shallow non-linear feature
   learning (cost `d ^ s_total` and escape horizon `∝ α₀ ^ -(s_total - 1)`), and deep hierarchical
   feature learning (cost `L * d ^ s_local` and escape horizon `L * α₀ ^ -(s_local - 1)`).
-/

noncomputable section

namespace OpenTheory.HierarchicalMultiIndex

open OpenTheory
open OpenTheory.MultiStepSpectralBias

/-! ### Section 1: Information Exponent `s` and Vanishing Low-Order Correlations -/

/-- Specification of a non-linear single/multi-index target with information exponent `s ≥ 1`
    and leading order-`s` Hermite / tensor coefficient `c_s > 0`. -/
structure MultiIndexTarget where
  /-- Leading non-vanishing Hermite / tensor correlation order (information exponent). -/
  s : ℕ
  /-- Leading Hermite / tensor coupling coefficient at order `s`. -/
  c_s : ℝ
  /-- The information exponent is at least `1`. -/
  s_pos : 1 ≤ s
  /-- The leading coupling coefficient is strictly positive. -/
  c_s_pos : 0 < c_s

/-- Order-`m` Hermite / tensor alignment drift for a target with information exponent `s`
    and leading coefficient `c_s` at signal overlap `α`:
    all tensor correlations of order `m < s` vanish identically (`0`), while at order `m ≥ s`
    the leading order-`s` tensor contraction contributes `c_s * α ^ (s - 1)`. -/
def tensorAlignmentDrift (s m : ℕ) (c_s : ℝ) (α : ℝ) : ℝ :=
  if m < s then 0 else c_s * α ^ (s - 1)

/-- **Vanishing Low-Order Tensor Correlations (`tensorAlignmentDrift_eq_zero_of_lt`)**:
    For any correlation order `m < s` below the information exponent `s`, the order-`m`
    alignment drift is identically zero. -/
@[simp]
theorem tensorAlignmentDrift_eq_zero_of_lt {s m : ℕ} (c_s α : ℝ) (hm : m < s) :
    tensorAlignmentDrift s m c_s α = 0 := by
  unfold tensorAlignmentDrift
  simp [hm]

/-- **Vanishing First-Order / Linear Gram Drift for `s ≥ 2`
    (`linear_gram_drift_zero_of_exponent_ge_two`)**:
    When a target has information exponent `s ≥ 2` (such as parity, XOR, or phase retrieval),
    the first-order / linearized Gram correlation (`m = 1`) is identically zero for every `α`. -/
theorem linear_gram_drift_zero_of_exponent_ge_two {s : ℕ} (c_s α : ℝ) (hs : 2 ≤ s) :
    tensorAlignmentDrift s 1 c_s α = 0 := by
  have hm : 1 < s := by omega
  exact tensorAlignmentDrift_eq_zero_of_lt c_s α hm

/-- **Leading Non-Vanishing Tensor Drift at Order `s` (`tensorAlignmentDrift_at_exponent`)**:
    At order `m = s`, the tensor alignment drift equals `c_s * α ^ (s - 1)`. -/
@[simp]
theorem tensorAlignmentDrift_at_exponent (s : ℕ) (c_s α : ℝ) :
    tensorAlignmentDrift s s c_s α = c_s * α ^ (s - 1) := by
  unfold tensorAlignmentDrift
  simp

/-- One-step non-linear alignment update on signal overlap `α` for a target with information
    exponent `s`, coupling `c_s`, and step size `η`:
    `multiIndexStep η c_s s α = α + η * c_s * α ^ s`. -/
def multiIndexStep (η c_s : ℝ) (s : ℕ) (α : ℝ) : ℝ :=
  α + η * c_s * α ^ s

/-- Equivalence between `multiIndexStep` and the leading order-`s` tensor alignment drift
    when `s ≥ 1`: `multiIndexStep η c_s s α = α + η * α * tensorAlignmentDrift s s c_s α`. -/
theorem multiIndexStep_eq_add_drift (η c_s : ℝ) {s : ℕ} (hs : 1 ≤ s) (α : ℝ) :
    multiIndexStep η c_s s α = α + η * α * tensorAlignmentDrift s s c_s α := by
  unfold multiIndexStep
  rw [tensorAlignmentDrift_at_exponent]
  have hs_eq : s = (s - 1) + 1 := (Nat.sub_add_cancel hs).symm
  conv_lhs => rw [hs_eq, pow_succ']
  ring

/-- `T`-step non-linear alignment trajectory starting from initial signal overlap `α₀`
    under repeated application of `multiIndexStep η c_s s`. -/
def multiIndexTrajectory (η c_s : ℝ) (s : ℕ) (α₀ : ℝ) : ℕ → ℝ
  | 0 => α₀
  | t + 1 => multiIndexStep η c_s s (multiIndexTrajectory η c_s s α₀ t)

@[simp]
lemma multiIndexTrajectory_zero (η c_s : ℝ) (s : ℕ) (α₀ : ℝ) :
    multiIndexTrajectory η c_s s α₀ 0 = α₀ :=
  rfl

@[simp]
lemma multiIndexTrajectory_succ (η c_s : ℝ) (s : ℕ) (α₀ : ℝ) (t : ℕ) :
    multiIndexTrajectory η c_s s α₀ (t + 1) =
      multiIndexStep η c_s s (multiIndexTrajectory η c_s s α₀ t) :=
  rfl

/-- Non-negativity of `multiIndexStep` when `0 ≤ α`, `0 ≤ η`, and `0 ≤ c_s`. -/
lemma multiIndexStep_nonneg {η c_s α : ℝ} (s : ℕ)
    (hα : 0 ≤ α) (hη : 0 ≤ η) (hc : 0 ≤ c_s) :
    0 ≤ multiIndexStep η c_s s α := by
  unfold multiIndexStep
  positivity

/-- Strict positivity of `multiIndexStep` when `0 < α`, `0 ≤ η`, and `0 ≤ c_s`. -/
lemma multiIndexStep_pos {η c_s α : ℝ} (s : ℕ)
    (hα : 0 < α) (hη : 0 ≤ η) (hc : 0 ≤ c_s) :
    0 < multiIndexStep η c_s s α := by
  unfold multiIndexStep
  have hterm : 0 ≤ η * c_s * α ^ s := by positivity
  linarith

/-- Non-negativity along the `T`-step trajectory `multiIndexTrajectory`. -/
theorem multiIndexTrajectory_nonneg {η c_s α₀ : ℝ} (s : ℕ)
    (hα₀ : 0 ≤ α₀) (hη : 0 ≤ η) (hc : 0 ≤ c_s) (T : ℕ) :
    0 ≤ multiIndexTrajectory η c_s s α₀ T := by
  induction T with
  | zero => exact hα₀
  | succ t ih =>
    rw [multiIndexTrajectory_succ]
    exact multiIndexStep_nonneg s ih hη hc

/-- **Positivity of the Multi-Index Trajectory (`multiIndexTrajectory_pos`)**:
    If `0 < α₀`, `0 ≤ η`, and `0 ≤ c_s`, then `0 < multiIndexTrajectory η c_s s α₀ T` for all `T`. -/
theorem multiIndexTrajectory_pos {η c_s α₀ : ℝ} (s : ℕ)
    (hα₀ : 0 < α₀) (hη : 0 ≤ η) (hc : 0 ≤ c_s) (T : ℕ) :
    0 < multiIndexTrajectory η c_s s α₀ T := by
  induction T with
  | zero => exact hα₀
  | succ t ih =>
    rw [multiIndexTrajectory_succ]
    exact multiIndexStep_pos s ih hη hc

/-- **Strict Stepwise Growth of the Multi-Index Trajectory (`multiIndexTrajectory_lt_succ`)**:
    When `0 < η`, `0 < c_s`, and `0 < α₀`, each step strictly increases the alignment overlap:
    `multiIndexTrajectory η c_s s α₀ t < multiIndexTrajectory η c_s s α₀ (t + 1)`. -/
theorem multiIndexTrajectory_lt_succ {η c_s α₀ : ℝ} (s : ℕ)
    (hα₀ : 0 < α₀) (hη : 0 < η) (hc : 0 < c_s) (t : ℕ) :
    multiIndexTrajectory η c_s s α₀ t < multiIndexTrajectory η c_s s α₀ (t + 1) := by
  rw [multiIndexTrajectory_succ]
  unfold multiIndexStep
  have hpos : 0 < multiIndexTrajectory η c_s s α₀ t :=
    multiIndexTrajectory_pos s hα₀ (le_of_lt hη) (le_of_lt hc) t
  have hdrift : 0 < η * c_s * (multiIndexTrajectory η c_s s α₀ t) ^ s := by positivity
  linarith

/-- **Strict Monotonicity of the Multi-Index Trajectory (`multiIndexTrajectory_strictMono`)**:
    When `0 < η`, `0 < c_s`, and `0 < α₀`, `t ↦ multiIndexTrajectory η c_s s α₀ t` is strictly
    increasing in `t : ℕ`. -/
theorem multiIndexTrajectory_strictMono {η c_s α₀ : ℝ} (s : ℕ)
    (hα₀ : 0 < α₀) (hη : 0 < η) (hc : 0 < c_s) :
    StrictMono (multiIndexTrajectory η c_s s α₀) :=
  strictMono_nat_of_lt_succ fun t => multiIndexTrajectory_lt_succ s hα₀ hη hc t

/-- Weak monotonicity of `multiIndexTrajectory` across steps `t₁ ≤ t₂` when `0 ≤ α₀`, `0 ≤ η`, `0 ≤ c_s`. -/
theorem multiIndexTrajectory_mono {η c_s α₀ : ℝ} (s : ℕ)
    (hα₀ : 0 ≤ α₀) (hη : 0 ≤ η) (hc : 0 ≤ c_s) {t₁ t₂ : ℕ} (ht : t₁ ≤ t₂) :
    multiIndexTrajectory η c_s s α₀ t₁ ≤ multiIndexTrajectory η c_s s α₀ t₂ := by
  induction ht with
  | refl => exact le_refl _
  | @step m _ ih =>
    rw [multiIndexTrajectory_succ]
    unfold multiIndexStep
    have hm_nonneg : 0 ≤ multiIndexTrajectory η c_s s α₀ m :=
      multiIndexTrajectory_nonneg s hα₀ hη hc m
    have hdrift : 0 ≤ η * c_s * (multiIndexTrajectory η c_s s α₀ m) ^ s := by positivity
    linarith

/-! ### Section 2: Information Exponent Dichotomy (`s = 1` Linear Fast Escape vs. `s ≥ 2` Saddle Plateau & Grokking) -/

/-- **Exact Closed Form for Information Exponent `s = 1` (`multiIndexTrajectory_exponent_one_eq`)**:
    For a target with information exponent `s = 1` (non-vanishing first-order Gram correlation),
    the `T`-step trajectory reduces to exact geometric power-iteration amplification
    `(1 + η * c₁) ^ T * α₀` (matching `modeAmpDisc η c₁ T α₀`). -/
theorem multiIndexTrajectory_exponent_one_eq (η c₁ α₀ : ℝ) (T : ℕ) :
    multiIndexTrajectory η c₁ 1 α₀ T = (1 + η * c₁) ^ T * α₀ := by
  induction T with
  | zero => simp [multiIndexTrajectory]
  | succ t ih =>
    rw [multiIndexTrajectory_succ]
    unfold multiIndexStep
    rw [ih, pow_one, pow_succ']
    ring

/-- Connection between `multiIndexTrajectory` at `s = 1` and `MultiStepSpectralBias.modeAmpDisc`. -/
theorem multiIndexTrajectory_exponent_one_eq_modeAmpDisc (η c₁ α₀ : ℝ) (T : ℕ) :
    multiIndexTrajectory η c₁ 1 α₀ T = modeAmpDisc η c₁ T α₀ := by
  rw [multiIndexTrajectory_exponent_one_eq]
  rfl

/-- **Saddle Plateau Drift Suppression for `s ≥ 2` (`nonlinear_drift_lt_linear_drift_on_plateau`)**:
    In the small-alignment regime `0 < α < 1`, any higher-order information exponent `s ≥ 2`
    satisfies `α ^ s < α`, so the non-linear tensor drift `η * c_s * α ^ s` is strictly smaller
    than the linear (`s = 1`) drift `η * c_s * α`. -/
theorem nonlinear_drift_lt_linear_drift_on_plateau {η c_s α : ℝ} {s : ℕ}
    (hη : 0 < η) (hc : 0 < c_s) (hα_pos : 0 < α) (hα_lt : α < 1) (hs : 2 ≤ s) :
    α ^ s < α ∧ η * c_s * α ^ s < η * c_s * α := by
  have hs_gt : 1 < s := by omega
  have hpow : α ^ s < α ^ 1 := pow_lt_pow_right_of_lt_one₀ hα_pos hα_lt hs_gt
  rw [pow_one] at hpow
  have hηc : 0 < η * c_s := mul_pos hη hc
  exact ⟨hpow, mul_lt_mul_of_pos_left hpow hηc⟩

/-- Characteristic saddle-plateau escape time scale (`grokkingEscapeSteps`) for information
    exponent `s` starting from initial overlap `α₀`:
    `grokkingEscapeSteps η c_s α₀ s = 1 / (η * c_s * α₀ ^ (s - 1))`. -/
def grokkingEscapeSteps (η c_s α₀ : ℝ) (s : ℕ) : ℝ :=
  1 / (η * c_s * α₀ ^ (s - 1))

/-- Positivity of `grokkingEscapeSteps` when `0 < η`, `0 < c_s`, and `0 < α₀`. -/
lemma grokkingEscapeSteps_pos {η c_s α₀ : ℝ} (s : ℕ)
    (hη : 0 < η) (hc : 0 < c_s) (hα₀ : 0 < α₀) :
    0 < grokkingEscapeSteps η c_s α₀ s := by
  unfold grokkingEscapeSteps
  positivity

/-- **Strict Monotonicity of Grokking Escape Time in the Information Exponent
    (`grokking_escape_time_strict_mono_in_exponent`)**:
    For small initial overlap `0 < α₀ < 1` (e.g., `α₀ = 1 / √d`) and `1 ≤ s₁ < s₂`, the
    characteristic saddle-escape time scale strictly grows with the information exponent:
    `grokkingEscapeSteps η c_s α₀ s₁ < grokkingEscapeSteps η c_s α₀ s₂`. -/
theorem grokking_escape_time_strict_mono_in_exponent {η c_s α₀ : ℝ} {s₁ s₂ : ℕ}
    (hη : 0 < η) (hc : 0 < c_s) (hα₀_pos : 0 < α₀) (hα₀_lt : α₀ < 1)
    (hs₁ : 1 ≤ s₁) (hs : s₁ < s₂) :
    grokkingEscapeSteps η c_s α₀ s₁ < grokkingEscapeSteps η c_s α₀ s₂ := by
  unfold grokkingEscapeSteps
  have hexp_lt : s₁ - 1 < s₂ - 1 := by omega
  have hpow_lt : α₀ ^ (s₂ - 1) < α₀ ^ (s₁ - 1) :=
    pow_lt_pow_right_of_lt_one₀ hα₀_pos hα₀_lt hexp_lt
  have hηc : 0 < η * c_s := mul_pos hη hc
  have hden₂_pos : 0 < η * c_s * α₀ ^ (s₂ - 1) := by positivity
  have hden_lt : η * c_s * α₀ ^ (s₂ - 1) < η * c_s * α₀ ^ (s₁ - 1) :=
    mul_lt_mul_of_pos_left hpow_lt hηc
  exact one_div_lt_one_div_of_lt hden₂_pos hden_lt

/-- **Exact Grokking Escape Time Ratio Law (`grokking_escape_time_ratio_eq`)**:
    For `1 ≤ s₁ < s₂` and `0 < α₀ < 1`, the ratio of characteristic escape times between
    information exponents `s₂` and `s₁` equals `(1 / α₀) ^ (s₂ - s₁)`, which is strictly
    greater than `1`. -/
theorem grokking_escape_time_ratio_eq {η c_s α₀ : ℝ} {s₁ s₂ : ℕ}
    (hη : 0 < η) (hc : 0 < c_s) (hα₀_pos : 0 < α₀) (hα₀_lt : α₀ < 1)
    (hs₁ : 1 ≤ s₁) (hs : s₁ < s₂) :
    grokkingEscapeSteps η c_s α₀ s₂ / grokkingEscapeSteps η c_s α₀ s₁ =
      (1 / α₀) ^ (s₂ - s₁) ∧
    1 < (1 / α₀) ^ (s₂ - s₁) := by
  have hη_ne : η ≠ 0 := ne_of_gt hη
  have hc_ne : c_s ≠ 0 := ne_of_gt hc
  have hα₀_ne : α₀ ≠ 0 := ne_of_gt hα₀_pos
  have hexp_split : s₂ - 1 = (s₁ - 1) + (s₂ - s₁) := by omega
  have hratio : grokkingEscapeSteps η c_s α₀ s₂ / grokkingEscapeSteps η c_s α₀ s₁ =
      (1 / α₀) ^ (s₂ - s₁) := by
    unfold grokkingEscapeSteps
    rw [hexp_split, pow_add, one_div_pow]
    field_simp [hη_ne, hc_ne, hα₀_ne]
  have hinv_gt : 1 < 1 / α₀ := (one_lt_div hα₀_pos).mpr hα₀_lt
  have hdiff_ne : s₂ - s₁ ≠ 0 := by omega
  have hpow_gt : 1 < (1 / α₀) ^ (s₂ - s₁) := one_lt_pow₀ hinv_gt hdiff_ne
  exact ⟨hratio, hpow_gt⟩

/-- **Stage 1 of Grokking — Slow Saddle Plateau Upper Bound
    (`multiIndex_slow_plateau_upper_bound`)**:
    While the trajectory remains inside the saddle plateau
    (`∀ t < T, multiIndexTrajectory η c_s s α₀ t ≤ α_plateau`), each step increases the overlap
    by at most `η * c_s * α_plateau ^ s`, so after `T` steps the cumulative gain is bounded by
    `multiIndexTrajectory η c_s s α₀ T - α₀ ≤ (T : ℝ) * (η * c_s * α_plateau ^ s)`. -/
theorem multiIndex_slow_plateau_upper_bound {η c_s α₀ α_plateau : ℝ} (s : ℕ)
    (hα₀ : 0 ≤ α₀) (hη : 0 ≤ η) (hc : 0 ≤ c_s) (T : ℕ)
    (hplat : ∀ t < T, multiIndexTrajectory η c_s s α₀ t ≤ α_plateau) :
    multiIndexTrajectory η c_s s α₀ T ≤ α₀ + (T : ℝ) * (η * c_s * α_plateau ^ s) ∧
    multiIndexTrajectory η c_s s α₀ T - α₀ ≤ (T : ℝ) * (η * c_s * α_plateau ^ s) := by
  have hbound : multiIndexTrajectory η c_s s α₀ T ≤
      α₀ + (T : ℝ) * (η * c_s * α_plateau ^ s) := by
    induction T with
    | zero =>
      simp [multiIndexTrajectory]
    | succ t ih =>
      have hplat_prev : ∀ k < t, multiIndexTrajectory η c_s s α₀ k ≤ α_plateau :=
        fun k hk => hplat k (Nat.lt_succ_of_lt hk)
      have ih_t := ih hplat_prev
      have ht_le : multiIndexTrajectory η c_s s α₀ t ≤ α_plateau :=
        hplat t (Nat.lt_succ_self t)
      have ht_nonneg : 0 ≤ multiIndexTrajectory η c_s s α₀ t :=
        multiIndexTrajectory_nonneg s hα₀ hη hc t
      have hpow_le : (multiIndexTrajectory η c_s s α₀ t) ^ s ≤ α_plateau ^ s :=
        pow_le_pow_left₀ ht_nonneg ht_le s
      have hηc : 0 ≤ η * c_s := mul_nonneg hη hc
      have hstep_le : η * c_s * (multiIndexTrajectory η c_s s α₀ t) ^ s ≤
          η * c_s * α_plateau ^ s :=
        mul_le_mul_of_nonneg_left hpow_le hηc
      rw [multiIndexTrajectory_succ]
      unfold multiIndexStep
      push_cast
      linarith
  exact ⟨hbound, by linarith⟩

/-- One-step lower bound once the overlap crosses a positive threshold `α_thresh ≤ α`:
    `multiIndexStep η c_s s α ≥ (1 + η * c_s * α_thresh ^ (s - 1)) * α`. -/
lemma multiIndexStep_ge_of_threshold {η c_s α_thresh α : ℝ} {s : ℕ}
    (hs : 1 ≤ s) (hη : 0 ≤ η) (hc : 0 ≤ c_s)
    (hthresh_pos : 0 ≤ α_thresh) (hα : α_thresh ≤ α) :
    (1 + η * c_s * α_thresh ^ (s - 1)) * α ≤ multiIndexStep η c_s s α := by
  unfold multiIndexStep
  have hs_eq : s = (s - 1) + 1 := (Nat.sub_add_cancel hs).symm
  have hpow_le : α_thresh ^ (s - 1) ≤ α ^ (s - 1) :=
    pow_le_pow_left₀ hthresh_pos hα (s - 1)
  have hα_nonneg : 0 ≤ α := le_trans hthresh_pos hα
  have hηcα : 0 ≤ η * c_s * α := by positivity
  calc
    (1 + η * c_s * α_thresh ^ (s - 1)) * α
        = α + (η * c_s * α) * α_thresh ^ (s - 1) := by ring
    _ ≤ α + (η * c_s * α) * α ^ (s - 1) := by
          have := mul_le_mul_of_nonneg_left hpow_le hηcα
          linarith
    _ = α + η * c_s * α ^ s := by
          conv_rhs => rw [hs_eq, pow_succ']
          ring

/-- **Stage 2 of Grokking — Rapid Post-Threshold Exponential Growth
    (`multiIndex_post_threshold_exponential_growth`)**:
    Once the trajectory reaches a threshold `α_thresh > 0` at step `T₀`
    (`α_thresh ≤ multiIndexTrajectory η c_s s α₀ T₀`), every subsequent step `T₀ + K`
    grows at least geometrically with factor `(1 + η * c_s * α_thresh ^ (s - 1))`:
    `(1 + η * c_s * α_thresh ^ (s - 1)) ^ K * multiIndexTrajectory η c_s s α₀ T₀ ≤ multiIndexTrajectory η c_s s α₀ (T₀ + K)`. -/
theorem multiIndex_post_threshold_exponential_growth {η c_s α₀ α_thresh : ℝ} {s : ℕ}
    (hs : 1 ≤ s) (hα₀ : 0 ≤ α₀) (hη : 0 ≤ η) (hc : 0 ≤ c_s)
    (hthresh_nonneg : 0 ≤ α_thresh) (T₀ : ℕ)
    (hcross : α_thresh ≤ multiIndexTrajectory η c_s s α₀ T₀) (K : ℕ) :
    (1 + η * c_s * α_thresh ^ (s - 1)) ^ K * multiIndexTrajectory η c_s s α₀ T₀ ≤
      multiIndexTrajectory η c_s s α₀ (T₀ + K) := by
  induction K with
  | zero =>
    simp
  | succ k ih =>
    have hmono : multiIndexTrajectory η c_s s α₀ T₀ ≤
        multiIndexTrajectory η c_s s α₀ (T₀ + k) :=
      multiIndexTrajectory_mono s hα₀ hη hc (Nat.le_add_right T₀ k)
    have hthresh_k : α_thresh ≤ multiIndexTrajectory η c_s s α₀ (T₀ + k) :=
      le_trans hcross hmono
    have hstep := multiIndexStep_ge_of_threshold hs hη hc hthresh_nonneg hthresh_k
    have hfactor_nonneg : 0 ≤ 1 + η * c_s * α_thresh ^ (s - 1) := by positivity
    rw [Nat.add_succ, multiIndexTrajectory_succ, pow_succ']
    calc
      (1 + η * c_s * α_thresh ^ (s - 1)) *
          (1 + η * c_s * α_thresh ^ (s - 1)) ^ k *
          multiIndexTrajectory η c_s s α₀ T₀
          = (1 + η * c_s * α_thresh ^ (s - 1)) *
              ((1 + η * c_s * α_thresh ^ (s - 1)) ^ k *
                multiIndexTrajectory η c_s s α₀ T₀) := by ring
      _ ≤ (1 + η * c_s * α_thresh ^ (s - 1)) *
            multiIndexTrajectory η c_s s α₀ (T₀ + k) :=
          mul_le_mul_of_nonneg_left ih hfactor_nonneg
      _ ≤ multiIndexStep η c_s s (multiIndexTrajectory η c_s s α₀ (T₀ + k)) := hstep

/-! ### Section 3: Compositional Hierarchy vs. Shallow Representation Bottleneck -/

/-- Sample complexity scale `d ^ s_total` required by a shallow (single-layer) architecture
    to resolve a compositional target of total information exponent `s_total` in dimension `d`. -/
def shallowSampleComplexity (d : ℝ) (s_total : ℕ) : ℝ :=
  d ^ s_total

/-- Cumulative sample complexity scale `L * d ^ s_local` required by an `L`-layer hierarchical
    architecture where each layer resolves a local composition of information exponent `s_local`. -/
def hierarchicalSampleComplexity (L : ℕ) (d : ℝ) (s_local : ℕ) : ℝ :=
  (L : ℝ) * d ^ s_local

/-- Cumulative saddle-escape time `L * grokkingEscapeSteps η c_s α₀ s_local` across `L`
    sequential hierarchical stages, each of local information exponent `s_local`. -/
def hierarchicalEscapeSteps (L : ℕ) (η c_s α₀ : ℝ) (s_local : ℕ) : ℝ :=
  (L : ℝ) * grokkingEscapeSteps η c_s α₀ s_local

/-- **Hierarchical vs. Shallow Sample Complexity Separation
    (`hierarchical_vs_shallow_sample_complexity_separation`)**:
    Whenever the depth `L` satisfies `(L : ℝ) < d ^ (s_total - s_local)` (with `s_local ≤ s_total`
    and `0 < d`), the `L`-layer hierarchical architecture achieves strictly smaller sample
    complexity than a shallow architecture learning the end-to-end exponent `s_total` in one step:
    `hierarchicalSampleComplexity L d s_local < shallowSampleComplexity d s_total`. -/
theorem hierarchical_vs_shallow_sample_complexity_separation
    {L s_local s_total : ℕ} {d : ℝ}
    (hd : 0 < d) (hs : s_local ≤ s_total)
    (hL : (L : ℝ) < d ^ (s_total - s_local)) :
    hierarchicalSampleComplexity L d s_local < shallowSampleComplexity d s_total := by
  unfold hierarchicalSampleComplexity shallowSampleComplexity
  have hsplit : s_total = (s_total - s_local) + s_local := (Nat.sub_add_cancel hs).symm
  rw [hsplit, pow_add]
  have hd_pow_pos : 0 < d ^ s_local := pow_pos hd s_local
  exact mul_lt_mul_of_pos_right hL hd_pow_pos

/-- Concrete binary compositional hierarchy (`L = 2`, `s_local = 2`, `s_total = 4`):
    for any ambient dimension `d ≥ 2`, `2 * d ^ 2 < d ^ 4`. -/
theorem binary_hierarchy_sample_complexity_lt_shallow {d : ℝ} (hd : 2 ≤ d) :
    hierarchicalSampleComplexity 2 d 2 < shallowSampleComplexity d 4 := by
  apply hierarchical_vs_shallow_sample_complexity_separation (by linarith) (by omega)
  norm_num
  nlinarith

/-- **Hierarchical Stagewise Escape Time Beats Shallow Escape Time
    (`hierarchical_stagewise_escape_time_lt_shallow`)**:
    The sum of `L` stagewise grokking escape times
    `∑ _l : Fin L, grokkingEscapeSteps η c_s α₀ s_local = hierarchicalEscapeSteps L η c_s α₀ s_local`
    is strictly smaller than the single-stage shallow escape time
    `grokkingEscapeSteps η c_s α₀ s_total` whenever `1 ≤ s_local ≤ s_total` and
    `(L : ℝ) < (1 / α₀) ^ (s_total - s_local)`. -/
theorem hierarchical_stagewise_escape_time_lt_shallow
    {L s_local s_total : ℕ} {η c_s α₀ : ℝ}
    (hη : 0 < η) (hc : 0 < c_s) (hα₀ : 0 < α₀)
    (hs_loc : 1 ≤ s_local) (hs : s_local ≤ s_total)
    (hL : (L : ℝ) < (1 / α₀) ^ (s_total - s_local)) :
    (∑ _l : Fin L, grokkingEscapeSteps η c_s α₀ s_local) =
      hierarchicalEscapeSteps L η c_s α₀ s_local ∧
    hierarchicalEscapeSteps L η c_s α₀ s_local <
      grokkingEscapeSteps η c_s α₀ s_total := by
  have hsum : (∑ _l : Fin L, grokkingEscapeSteps η c_s α₀ s_local) =
      hierarchicalEscapeSteps L η c_s α₀ s_local := by
    unfold hierarchicalEscapeSteps
    simp
  refine ⟨hsum, ?_⟩
  unfold hierarchicalEscapeSteps grokkingEscapeSteps
  have hη_ne : η ≠ 0 := ne_of_gt hη
  have hc_ne : c_s ≠ 0 := ne_of_gt hc
  have hα₀_ne : α₀ ≠ 0 := ne_of_gt hα₀
  have hexp_split : s_total - 1 = (s_local - 1) + (s_total - s_local) := by omega
  have hbase_pos : 0 < 1 / (η * c_s * α₀ ^ (s_local - 1)) := by positivity
  have hscale_eq :
      1 / (η * c_s * α₀ ^ (s_total - 1)) =
        (1 / α₀) ^ (s_total - s_local) * (1 / (η * c_s * α₀ ^ (s_local - 1))) := by
    rw [hexp_split, pow_add, one_div_pow]
    field_simp [hη_ne, hc_ne, hα₀_ne]
  rw [hscale_eq]
  exact mul_lt_mul_of_pos_right hL hbase_pos

/-! ### Section 4: Practical Prescription: Depth & Warmup Schedule for High-Information-Exponent Data -/

/-- **Unified Three-Regime Prescription for Compositional Multi-Index Targets
    (`optimal_depth_and_horizon_for_multi_index`)**:
    Given an `L`-level compositional target with local exponent `s_local ≥ 2` and end-to-end
    shallow exponent `s_total > s_local`, step size `η > 0`, coupling `c_s > 0`, initial overlap
    `0 < α₀ < 1`, ambient dimension `d > 0`, and `(L : ℝ) < min (d ^ (s_total - s_local)) ((1 / α₀) ^ (s_total - s_local))`:
    1. **Linearized / NTK Regime (`m = 1`) Fails**: Both local (`s_local ≥ 2`) and end-to-end
       (`s_total ≥ 2`) targets have identically zero first-order Gram drift
       (`tensorAlignmentDrift s_local 1 c_s α₀ = 0` and `tensorAlignmentDrift s_total 1 c_s α₀ = 0`).
    2. **Shallow Non-Linear Feature Learning Bottleneck**: Resolving `s_total` in a single layer
       incurs sample complexity `d ^ s_total` and saddle escape time `grokkingEscapeSteps η c_s α₀ s_total`,
       strictly larger than the `s_local` single-stage scale.
    3. **Deep Hierarchical Feature Learning Advantage**: Decomposing the target across `L` stages
       of exponent `s_local` strictly reduces both total sample complexity
       (`hierarchicalSampleComplexity L d s_local < shallowSampleComplexity d s_total`) and
       cumulative grokking escape horizon
       (`hierarchicalEscapeSteps L η c_s α₀ s_local < grokkingEscapeSteps η c_s α₀ s_total`). -/
theorem optimal_depth_and_horizon_for_multi_index
    {L s_local s_total : ℕ} {η c_s α₀ d : ℝ}
    (hη : 0 < η) (hc : 0 < c_s) (hα₀_pos : 0 < α₀) (hα₀_lt : α₀ < 1)
    (hd : 0 < d) (hs_local : 2 ≤ s_local) (hs_lt : s_local < s_total)
    (hL_sample : (L : ℝ) < d ^ (s_total - s_local))
    (hL_escape : (L : ℝ) < (1 / α₀) ^ (s_total - s_local)) :
    tensorAlignmentDrift s_local 1 c_s α₀ = 0 ∧
    tensorAlignmentDrift s_total 1 c_s α₀ = 0 ∧
    grokkingEscapeSteps η c_s α₀ s_local < grokkingEscapeSteps η c_s α₀ s_total ∧
    hierarchicalSampleComplexity L d s_local < shallowSampleComplexity d s_total ∧
    hierarchicalEscapeSteps L η c_s α₀ s_local < grokkingEscapeSteps η c_s α₀ s_total := by
  have hs_total_ge_two : 2 ≤ s_total := by omega
  have hs_local_ge_one : 1 ≤ s_local := by omega
  have hs_le : s_local ≤ s_total := le_of_lt hs_lt
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  · exact linear_gram_drift_zero_of_exponent_ge_two c_s α₀ hs_local
  · exact linear_gram_drift_zero_of_exponent_ge_two c_s α₀ hs_total_ge_two
  · exact grokking_escape_time_strict_mono_in_exponent
      hη hc hα₀_pos hα₀_lt hs_local_ge_one hs_lt
  · exact hierarchical_vs_shallow_sample_complexity_separation hd hs_le hL_sample
  · exact (hierarchical_stagewise_escape_time_lt_shallow
      hη hc hα₀_pos hs_local_ge_one hs_le hL_escape).2

end OpenTheory.HierarchicalMultiIndex
