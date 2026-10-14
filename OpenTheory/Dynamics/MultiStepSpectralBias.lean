import OpenTheory.Data.Spectrum
import OpenTheory.Data.Model
import OpenTheory.Representation.Alignment
import OpenTheory.Representation.FeatureLearningAdvantage
import OpenTheory.Parameterization.MuP
import Mathlib.Analysis.SpecialFunctions.ExpDeriv
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Ring
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity

/-! # Multi-Step & Continuous-Time µP Spectral Bias, Signal-to-Bulk Amplification, and Subspace Alignment Convergence

This module establishes the quantitative dynamics of feature modes under maximal-update (µP)
spectral evolution, proving exponential / geometric signal-to-bulk amplification, subspace
alignment convergence to `1`, and multi-step feature-learning risk advantage:

1. **Continuous-Time & Discrete-Time Eigenmode Dynamics**:
   Along an eigenspace of the data Gram matrix with eigenvalue `γ ≥ 0`, a feature mode evolves
   in continuous time as `modeAmpCont γ t a₀ = Real.exp (γ * t) * a₀` and in discrete time over
   `T : ℕ` steps with normalized step size `η > 0` as `modeAmpDisc η γ T a₀ = (1 + η * γ) ^ T * a₀`.
   We establish exact ODE derivatives (`hasDerivAt_modeAmpCont`), one-step recurrences
   (`modeAmpDisc_step`), and strict monotonicity across time/steps and eigenvalues.

2. **Signal-to-Bulk Amplification under `SpectralGap`**:
   Given a `SpectralGap` (`sg.leading ≥ sg.bulk + sg.gap`, `sg.gap > 0`, `sg.bulk ≥ 0`), signal
   energy growing at rate `2 * sg.leading` outpaces bulk energy growing at rate `2 * sg.bulk`:
   - In continuous time (`signal_to_bulk_ratio_cont`):
     `B(t) * S₀ ≤ Real.exp (-2 * sg.gap * t) * (S(t) * B₀)`
     (and `B(t) / S(t) ≤ (B₀ / S₀) * Real.exp (-2 * sg.gap * t)`).
   - In discrete time (`signal_to_bulk_ratio_disc`):
     `((1 + η * sg.leading) / (1 + η * sg.bulk)) ^ (2 * T) * (B_T * S₀) ≤ S_T * B₀`,
     where the per-step growth ratio satisfies
     `1 < 1 + (η * sg.gap) / (1 + η * sg.bulk) ≤ (1 + η * sg.leading) / (1 + η * sg.bulk)`.

3. **Subspace Alignment Convergence to `1`**:
   Using the Frobenius Pythagorean identity `OpenTheory.frobeniusSq_sub_mul_orthoProjector`,
   if `P` is an orthogonal projector (`IsOrthoProjector P`), `frobeniusSq (H * P) ≥ S > 0`,
   and `frobeniusSq (H - H * P) ≤ B` (`B ≥ 0`), then:
   `alignment H P ≥ S / (S + B) = 1 - B / (S + B) ≥ 1 - B / S`.
   Combining this with the continuous and discrete signal-to-bulk bounds gives:
   - `spectral_bias_alignment_cont_ge`:
     `alignment H_t P ≥ 1 - (B₀ / S₀) * Real.exp (-2 * sg.gap * t)`
     and relative subspace distance `frobeniusSq (H_t - H_t * P) ≤ ((B₀ / S₀) * Real.exp (-2 * sg.gap * t)) * frobeniusSq H_t`.
   - `spectral_bias_alignment_disc_ge`:
     `alignment H_T P ≥ 1 - (B₀ / S₀) * ((1 + η * sg.bulk) / (1 + η * sg.leading)) ^ (2 * T)`
     and relative subspace distance `frobeniusSq (H_T - H_T * P) ≤ ((B₀ / S₀) * ((1 + η * sg.bulk) / (1 + η * sg.leading)) ^ (2 * T)) * frobeniusSq H_T`.

4. **Multi-Step Feature Learning Risk Advantage**:
   When `T ≥ 1` steps of µP feature learning amplify the target-aligned kernel energy from
   `K₀ > 0` to `K_T ≥ (1 + η * sg.leading) ^ (2 * T) * K₀`, we have `K₀ < K_T`, which strictly
   decreases both the ridge regularization bias `biasSq` and, in the signal-dominant regime,
   the total expected ridge MSE `ridgeExpectedMSE`.
-/

noncomputable section

namespace OpenTheory.MultiStepSpectralBias

open OpenTheory
open OpenTheory.FeatureLearningAdvantage

/-! ### Part 1: Continuous-Time & Discrete-Time Eigenmode Dynamics -/

/-- Effective first-layer feature learning rate from a `MuPScaling d m` configuration. -/
def mupEffectiveFeatureLR {d m : ℕ} (scale : MuPScaling d m) : ℝ :=
  scale.lr_W1

/-- Continuous-time feature mode amplitude along a Gram eigenspace with eigenvalue `γ`
    at time `t`, starting from initial amplitude `a₀`:
    `modeAmpCont γ t a₀ = Real.exp (γ * t) * a₀`. -/
def modeAmpCont (γ t a₀ : ℝ) : ℝ :=
  Real.exp (γ * t) * a₀

/-- Discrete-time feature mode amplitude after `T : ℕ` steps with step size `η` along a Gram
    eigenspace with eigenvalue `γ`, starting from initial amplitude `a₀`:
    `modeAmpDisc η γ T a₀ = (1 + η * γ) ^ T * a₀`. -/
def modeAmpDisc (η γ : ℝ) (T : ℕ) (a₀ : ℝ) : ℝ :=
  (1 + η * γ) ^ T * a₀

@[simp]
lemma modeAmpCont_zero (γ a₀ : ℝ) : modeAmpCont γ 0 a₀ = a₀ := by
  simp [modeAmpCont]

@[simp]
lemma modeAmpDisc_zero (η γ a₀ : ℝ) : modeAmpDisc η γ 0 a₀ = a₀ := by
  simp [modeAmpDisc]

/-- Exact time derivative of `modeAmpCont`:
    `d/dt (Real.exp (γ * t) * a₀) = γ * (Real.exp (γ * t) * a₀)`. -/
theorem hasDerivAt_modeAmpCont (γ a₀ t : ℝ) :
    HasDerivAt (fun s => Real.exp (γ * s) * a₀) (γ * (Real.exp (γ * t) * a₀)) t := by
  have hlin : HasDerivAt (fun s => γ * s) γ t := by
    simpa using (hasDerivAt_id t).const_mul γ
  have hexp : HasDerivAt (fun s => Real.exp (γ * s)) (Real.exp (γ * t) * γ) t :=
    hlin.exp
  have hmul : HasDerivAt (fun s => Real.exp (γ * s) * a₀) ((Real.exp (γ * t) * γ) * a₀) t :=
    hexp.mul_const a₀
  convert hmul using 1
  ring

/-- Named wrapper for the derivative of `modeAmpCont γ · a₀`. -/
theorem hasDerivAt_modeAmpCont' (γ a₀ t : ℝ) :
    HasDerivAt (fun s => modeAmpCont γ s a₀) (γ * modeAmpCont γ t a₀) t :=
  hasDerivAt_modeAmpCont γ a₀ t

/-- Exact one-step recurrence for `modeAmpDisc`:
    `modeAmpDisc η γ (T + 1) a₀ = (1 + η * γ) * modeAmpDisc η γ T a₀`. -/
theorem modeAmpDisc_step (η γ : ℝ) (T : ℕ) (a₀ : ℝ) :
    modeAmpDisc η γ (T + 1) a₀ = (1 + η * γ) * modeAmpDisc η γ T a₀ := by
  unfold modeAmpDisc
  rw [pow_succ']
  ring

/-- Nonnegativity of `modeAmpCont` for nonnegative initial amplitude `a₀ ≥ 0`. -/
lemma modeAmpCont_nonneg (γ t : ℝ) {a₀ : ℝ} (ha₀ : 0 ≤ a₀) :
    0 ≤ modeAmpCont γ t a₀ :=
  mul_nonneg (Real.exp_nonneg _) ha₀

/-- Strict positivity of `modeAmpCont` for positive initial amplitude `a₀ > 0`. -/
lemma modeAmpCont_pos (γ t : ℝ) {a₀ : ℝ} (ha₀ : 0 < a₀) :
    0 < modeAmpCont γ t a₀ :=
  mul_pos (Real.exp_pos _) ha₀

/-- Monotonicity of `modeAmpCont` in time `t` when `γ ≥ 0` and `a₀ ≥ 0`. -/
theorem modeAmpCont_mono_time {γ t₁ t₂ a₀ : ℝ}
    (hγ : 0 ≤ γ) (ht : t₁ ≤ t₂) (ha₀ : 0 ≤ a₀) :
    modeAmpCont γ t₁ a₀ ≤ modeAmpCont γ t₂ a₀ := by
  unfold modeAmpCont
  exact mul_le_mul_of_nonneg_right
    (Real.exp_le_exp.mpr (mul_le_mul_of_nonneg_left ht hγ)) ha₀

/-- Monotonicity of `modeAmpCont` in the eigenvalue `γ` when `t ≥ 0` and `a₀ ≥ 0`. -/
theorem modeAmpCont_mono_eigenvalue {γ₁ γ₂ t a₀ : ℝ}
    (hγ : γ₁ ≤ γ₂) (ht : 0 ≤ t) (ha₀ : 0 ≤ a₀) :
    modeAmpCont γ₁ t a₀ ≤ modeAmpCont γ₂ t a₀ := by
  unfold modeAmpCont
  exact mul_le_mul_of_nonneg_right
    (Real.exp_le_exp.mpr (mul_le_mul_of_nonneg_right hγ ht)) ha₀

/-- Monotonicity of `modeAmpDisc` in the eigenvalue `γ` when `η ≥ 0` and `a₀ ≥ 0`. -/
theorem modeAmpDisc_mono_eigenvalue {η γ₁ γ₂ a₀ : ℝ} (T : ℕ)
    (hη : 0 ≤ η) (hγ₁ : 0 ≤ γ₁) (hγ : γ₁ ≤ γ₂) (ha₀ : 0 ≤ a₀) :
    modeAmpDisc η γ₁ T a₀ ≤ modeAmpDisc η γ₂ T a₀ := by
  unfold modeAmpDisc
  apply mul_le_mul_of_nonneg_right _ ha₀
  exact pow_le_pow_left₀ (by positivity) (by nlinarith) T

/-- Strict step-wise growth of `modeAmpDisc` when `η > 0`, `γ > 0`, `T ≥ 1`, and `a₀ > 0`. -/
theorem modeAmpDisc_lt_of_pos {η γ a₀ : ℝ} {T : ℕ}
    (hη : 0 < η) (hγ : 0 < γ) (hT : 1 ≤ T) (ha₀ : 0 < a₀) :
    a₀ < modeAmpDisc η γ T a₀ := by
  unfold modeAmpDisc
  have hbase : 1 < 1 + η * γ := by linarith [mul_pos hη hγ]
  have hpow : 1 < (1 + η * γ) ^ T := one_lt_pow₀ hbase (Nat.ne_of_gt hT)
  calc
    a₀ = 1 * a₀ := (one_mul a₀).symm
    _ < (1 + η * γ) ^ T * a₀ := mul_lt_mul_of_pos_right hpow ha₀

/-! ### Part 2: Signal-to-Bulk Amplification under `SpectralGap` -/

/-- Basic positivity of the leading eigenvalue under a `SpectralGap`:
    `0 < sg.leading` since `sg.leading ≥ sg.bulk + sg.gap > 0`. -/
lemma leading_pos_of_spectralGap (sg : SpectralGap) : 0 < sg.leading := by
  have hgap := sg.gap_pos
  have hbulk := sg.bulk_nonneg
  have hsep := sg.separated
  linarith

/-- **Continuous-Time Signal-to-Bulk Amplification (`signal_to_bulk_ratio_cont`)**:
    Under a `SpectralGap` (`sg.leading ≥ sg.bulk + sg.gap`), if at time `t ≥ 0` the signal energy
    `S_t` grows at least as `Real.exp (2 * sg.leading * t) * S₀` (`S₀ > 0`) and the bulk energy
    `B_t` grows at most as `Real.exp (2 * sg.bulk * t) * B₀` (`B₀ ≥ 0`), then:
    `B_t * S₀ ≤ Real.exp (-2 * sg.gap * t) * (S_t * B₀)`. -/
theorem signal_to_bulk_ratio_cont (sg : SpectralGap)
    {t S₀ B₀ S_t B_t : ℝ}
    (ht : 0 ≤ t) (hS₀ : 0 < S₀) (hB₀ : 0 ≤ B₀)
    (hS : Real.exp (2 * sg.leading * t) * S₀ ≤ S_t)
    (hB : B_t ≤ Real.exp (2 * sg.bulk * t) * B₀) :
    B_t * S₀ ≤ Real.exp (-2 * sg.gap * t) * (S_t * B₀) := by
  have hsep := sg.separated
  have hexp_le : Real.exp (2 * sg.bulk * t) ≤
      Real.exp (-2 * sg.gap * t) * Real.exp (2 * sg.leading * t) := by
    rw [← Real.exp_add]
    apply Real.exp_le_exp.mpr
    nlinarith
  calc
    B_t * S₀ ≤ (Real.exp (2 * sg.bulk * t) * B₀) * S₀ :=
      mul_le_mul_of_nonneg_right hB (le_of_lt hS₀)
    _ = Real.exp (2 * sg.bulk * t) * (S₀ * B₀) := by ring
    _ ≤ (Real.exp (-2 * sg.gap * t) * Real.exp (2 * sg.leading * t)) * (S₀ * B₀) :=
      mul_le_mul_of_nonneg_right hexp_le (mul_nonneg (le_of_lt hS₀) hB₀)
    _ = Real.exp (-2 * sg.gap * t) * ((Real.exp (2 * sg.leading * t) * S₀) * B₀) := by ring
    _ ≤ Real.exp (-2 * sg.gap * t) * (S_t * B₀) := by
      apply mul_le_mul_of_nonneg_left _ (Real.exp_nonneg _)
      exact mul_le_mul_of_nonneg_right hS hB₀

/-- Ratio form of continuous-time signal-to-bulk decay:
    `B_t / S_t ≤ (B₀ / S₀) * Real.exp (-2 * sg.gap * t)`. -/
theorem bulk_div_signal_cont_le (sg : SpectralGap)
    {t S₀ B₀ S_t B_t : ℝ}
    (ht : 0 ≤ t) (hS₀ : 0 < S₀) (hB₀ : 0 ≤ B₀)
    (hS : Real.exp (2 * sg.leading * t) * S₀ ≤ S_t)
    (hB : B_t ≤ Real.exp (2 * sg.bulk * t) * B₀) :
    B_t / S_t ≤ (B₀ / S₀) * Real.exp (-2 * sg.gap * t) := by
  have hSt_pos : 0 < S_t :=
    lt_of_lt_of_le (mul_pos (Real.exp_pos _) hS₀) hS
  have hprod := signal_to_bulk_ratio_cont sg ht hS₀ hB₀ hS hB
  rw [div_le_iff₀ hSt_pos]
  calc
    B_t = (B_t * S₀) / S₀ := by field_simp [ne_of_gt hS₀]
    _ ≤ (Real.exp (-2 * sg.gap * t) * (S_t * B₀)) / S₀ :=
      div_le_div_of_nonneg_right hprod (le_of_lt hS₀)
    _ = (B₀ / S₀) * Real.exp (-2 * sg.gap * t) * S_t := by
      field_simp [ne_of_gt hS₀]

/-- Lower bound and strict exceeding-one property of the discrete per-step signal-to-bulk
    growth ratio `(1 + η * sg.leading) / (1 + η * sg.bulk)` under `sg : SpectralGap` and `η > 0`. -/
theorem disc_growth_ratio_bounds (sg : SpectralGap) {η : ℝ} (hη : 0 < η) :
    1 + (η * sg.gap) / (1 + η * sg.bulk) ≤
      (1 + η * sg.leading) / (1 + η * sg.bulk) ∧
    1 < (1 + η * sg.leading) / (1 + η * sg.bulk) := by
  have hbulk := sg.bulk_nonneg
  have hgap := sg.gap_pos
  have hsep := sg.separated
  have hden : 0 < 1 + η * sg.bulk := by positivity
  have hle : 1 + (η * sg.gap) / (1 + η * sg.bulk) ≤
      (1 + η * sg.leading) / (1 + η * sg.bulk) := by
    rw [one_add_div (ne_of_gt hden)]
    apply div_le_div_of_nonneg_right _ (le_of_lt hden)
    nlinarith
  have hgt : 1 < 1 + (η * sg.gap) / (1 + η * sg.bulk) := by
    have hpos : 0 < (η * sg.gap) / (1 + η * sg.bulk) := by positivity
    linarith
  exact ⟨hle, lt_of_lt_of_le hgt hle⟩

/-- **Discrete-Time Signal-to-Bulk Amplification (`signal_to_bulk_ratio_disc`)**:
    Over `T : ℕ` discrete steps with step size `η > 0` and `sg : SpectralGap`, if
    `S_T ≥ (1 + η * sg.leading) ^ (2 * T) * S₀` (`S₀ > 0`) and
    `B_T ≤ (1 + η * sg.bulk) ^ (2 * T) * B₀` (`B₀ ≥ 0`), then:
    1. `((1 + η * sg.leading) / (1 + η * sg.bulk)) ^ (2 * T) * (B_T * S₀) ≤ S_T * B₀`,
    2. `1 + (η * sg.gap) / (1 + η * sg.bulk) ≤ (1 + η * sg.leading) / (1 + η * sg.bulk)`,
    3. `1 < (1 + η * sg.leading) / (1 + η * sg.bulk)`. -/
theorem signal_to_bulk_ratio_disc (sg : SpectralGap)
    {η S₀ B₀ S_T B_T : ℝ} (T : ℕ)
    (hη : 0 < η) (hS₀ : 0 < S₀) (hB₀ : 0 ≤ B₀)
    (hS : (1 + η * sg.leading) ^ (2 * T) * S₀ ≤ S_T)
    (hB : B_T ≤ (1 + η * sg.bulk) ^ (2 * T) * B₀) :
    ((1 + η * sg.leading) / (1 + η * sg.bulk)) ^ (2 * T) * (B_T * S₀) ≤ S_T * B₀ ∧
    1 + (η * sg.gap) / (1 + η * sg.bulk) ≤
      (1 + η * sg.leading) / (1 + η * sg.bulk) ∧
    1 < (1 + η * sg.leading) / (1 + η * sg.bulk) := by
  have hbounds := disc_growth_ratio_bounds sg hη
  refine ⟨?_, hbounds.1, hbounds.2⟩
  have hbulk := sg.bulk_nonneg
  have hlead := leading_pos_of_spectralGap sg
  have hden_pos : 0 < 1 + η * sg.bulk := by positivity
  have hnum_pos : 0 < 1 + η * sg.leading := by positivity
  have hden_pow_pos : 0 < (1 + η * sg.bulk) ^ (2 * T) := pow_pos hden_pos (2 * T)
  have hden_pow_ne : (1 + η * sg.bulk) ^ (2 * T) ≠ 0 := ne_of_gt hden_pow_pos
  rw [div_pow]
  calc
    (1 + η * sg.leading) ^ (2 * T) / (1 + η * sg.bulk) ^ (2 * T) * (B_T * S₀)
        ≤ (1 + η * sg.leading) ^ (2 * T) / (1 + η * sg.bulk) ^ (2 * T) *
            (((1 + η * sg.bulk) ^ (2 * T) * B₀) * S₀) := by
          apply mul_le_mul_of_nonneg_left
          · exact mul_le_mul_of_nonneg_right hB (le_of_lt hS₀)
          · positivity
    _ = ((1 + η * sg.leading) ^ (2 * T) * S₀) * B₀ := by
          field_simp [hden_pow_ne]
    _ ≤ S_T * B₀ := mul_le_mul_of_nonneg_right hS hB₀

/-- Ratio form of discrete-time signal-to-bulk decay:
    `B_T / S_T ≤ (B₀ / S₀) * ((1 + η * sg.bulk) / (1 + η * sg.leading)) ^ (2 * T)`. -/
theorem bulk_div_signal_disc_le (sg : SpectralGap)
    {η S₀ B₀ S_T B_T : ℝ} (T : ℕ)
    (hη : 0 < η) (hS₀ : 0 < S₀) (hB₀ : 0 ≤ B₀)
    (hS : (1 + η * sg.leading) ^ (2 * T) * S₀ ≤ S_T)
    (hB : B_T ≤ (1 + η * sg.bulk) ^ (2 * T) * B₀) :
    B_T / S_T ≤ (B₀ / S₀) * ((1 + η * sg.bulk) / (1 + η * sg.leading)) ^ (2 * T) := by
  have hlead := leading_pos_of_spectralGap sg
  have hnum_pos : 0 < 1 + η * sg.leading := by positivity
  have hpow_pos : 0 < (1 + η * sg.leading) ^ (2 * T) := pow_pos hnum_pos (2 * T)
  have hST_pos : 0 < S_T := lt_of_lt_of_le (mul_pos hpow_pos hS₀) hS
  rw [div_le_iff₀ hST_pos, div_pow]
  calc
    B_T ≤ (1 + η * sg.bulk) ^ (2 * T) * B₀ := hB
    _ = (B₀ / S₀) * ((1 + η * sg.bulk) ^ (2 * T) / (1 + η * sg.leading) ^ (2 * T)) *
          ((1 + η * sg.leading) ^ (2 * T) * S₀) := by
          field_simp [ne_of_gt hS₀, ne_of_gt hpow_pos]
    _ ≤ (B₀ / S₀) * ((1 + η * sg.bulk) ^ (2 * T) / (1 + η * sg.leading) ^ (2 * T)) * S_T := by
          apply mul_le_mul_of_nonneg_left hS
          have hbulk := sg.bulk_nonneg
          positivity

/-! ### Part 3: Subspace Alignment Convergence to `1` -/

/-- **Subspace Alignment Lower Bound from Signal & Bulk Energy (`alignment_lower_bound_of_signal_bulk`)**:
    Suppose `P : Matrix (Fin m) (Fin m) ℝ` is an orthogonal projector (`IsOrthoProjector P`),
    and the feature matrix `H : Matrix (Fin n) (Fin m) ℝ` has signal energy
    `frobeniusSq (H * P) ≥ S` (`S > 0`) and residual bulk energy
    `frobeniusSq (H - H * P) ≤ B` (`B ≥ 0`).
    Then `frobeniusSq H ≠ 0` and:
    1. `S / (S + B) ≤ alignment H P`,
    2. `S / (S + B) = 1 - B / (S + B)`,
    3. `1 - B / S ≤ 1 - B / (S + B)`,
    4. `1 - B / S ≤ alignment H P`. -/
theorem alignment_lower_bound_of_signal_bulk {n m : ℕ}
    (H : Matrix (Fin n) (Fin m) ℝ) (P : Matrix (Fin m) (Fin m) ℝ)
    (hP : IsOrthoProjector P)
    {S B : ℝ} (hS_pos : 0 < S) (hB_nonneg : 0 ≤ B)
    (hS : S ≤ frobeniusSq (H * P))
    (hB : frobeniusSq (H - H * P) ≤ B) :
    frobeniusSq H ≠ 0 ∧
    S / (S + B) ≤ alignment H P ∧
    S / (S + B) = 1 - B / (S + B) ∧
    1 - B / S ≤ 1 - B / (S + B) ∧
    1 - B / S ≤ alignment H P := by
  have hpyth := frobeniusSq_sub_mul_orthoProjector H P hP
  -- Note: frobeniusSq H = frobeniusSq (H * P) + frobeniusSq (H - H * P)
  have hH_eq : frobeniusSq H = frobeniusSq (H * P) + frobeniusSq (H - H * P) := by linarith
  have hHP_pos : 0 < frobeniusSq (H * P) := lt_of_lt_of_le hS_pos hS
  have hres_nonneg : 0 ≤ frobeniusSq (H - H * P) := frobeniusSq_nonneg (H - H * P)
  have hH_pos : 0 < frobeniusSq H := by linarith
  have hH_ne : frobeniusSq H ≠ 0 := ne_of_gt hH_pos
  have hSB_pos : 0 < S + B := by linarith
  -- Prove S / (S + B) ≤ alignment H P
  have halign_ge : S / (S + B) ≤ alignment H P := by
    have ha_eq : alignment H P = frobeniusSq (H * P) / frobeniusSq H := by
      simp [alignment, hH_ne]
    rw [ha_eq, div_le_div_iff₀ hSB_pos hH_pos]
    nlinarith
  have h_eq : S / (S + B) = 1 - B / (S + B) := by
    field_simp [ne_of_gt hSB_pos]
    ring
  have h_step : 1 - B / S ≤ 1 - B / (S + B) := by
    have hdiv : B / (S + B) ≤ B / S :=
      div_le_div_of_nonneg_left hB_nonneg hS_pos (by linarith)
    linarith
  have h_final : 1 - B / S ≤ alignment H P := by linarith
  exact ⟨hH_ne, halign_ge, h_eq, h_step, h_final⟩

/-- **Continuous-Time Spectral Bias Subspace Alignment Theorem (`spectral_bias_alignment_cont_ge`)**:
    Under continuous-time µP spectral evolution with `sg : SpectralGap`, at any time `t ≥ 0`,
    if the feature matrix `H_t` has signal energy `frobeniusSq (H_t * P) ≥ Real.exp (2 * sg.leading * t) * S₀`
    (`S₀ > 0`) and residual bulk energy `frobeniusSq (H_t - H_t * P) ≤ Real.exp (2 * sg.bulk * t) * B₀`
    (`B₀ ≥ 0`), then:
    1. `alignment H_t P ≥ 1 - (B₀ / S₀) * Real.exp (-2 * sg.gap * t)`,
    2. `frobeniusSq (H_t - H_t * P) ≤ ((B₀ / S₀) * Real.exp (-2 * sg.gap * t)) * frobeniusSq H_t`. -/
theorem spectral_bias_alignment_cont_ge {n m : ℕ}
    (sg : SpectralGap)
    (H_t : Matrix (Fin n) (Fin m) ℝ) (P : Matrix (Fin m) (Fin m) ℝ)
    (hP : IsOrthoProjector P)
    {t S₀ B₀ : ℝ} (ht : 0 ≤ t) (hS₀ : 0 < S₀) (hB₀ : 0 ≤ B₀)
    (hS : Real.exp (2 * sg.leading * t) * S₀ ≤ frobeniusSq (H_t * P))
    (hB : frobeniusSq (H_t - H_t * P) ≤ Real.exp (2 * sg.bulk * t) * B₀) :
    1 - (B₀ / S₀) * Real.exp (-2 * sg.gap * t) ≤ alignment H_t P ∧
    frobeniusSq (H_t - H_t * P) ≤
      ((B₀ / S₀) * Real.exp (-2 * sg.gap * t)) * frobeniusSq H_t := by
  set S_t := Real.exp (2 * sg.leading * t) * S₀
  set B_t := Real.exp (2 * sg.bulk * t) * B₀
  have hSt_pos : 0 < S_t := mul_pos (Real.exp_pos _) hS₀
  have hBt_nonneg : 0 ≤ B_t := mul_nonneg (Real.exp_nonneg _) hB₀
  obtain ⟨hH_ne, _, _, _, halign_StBt⟩ :=
    alignment_lower_bound_of_signal_bulk H_t P hP hSt_pos hBt_nonneg hS hB
  have hratio : B_t / S_t ≤ (B₀ / S₀) * Real.exp (-2 * sg.gap * t) :=
    bulk_div_signal_cont_le sg ht hS₀ hB₀ (le_refl S_t) (le_refl B_t)
  have halign : 1 - (B₀ / S₀) * Real.exp (-2 * sg.gap * t) ≤ alignment H_t P := by
    linarith
  have hε_nonneg : 0 ≤ (B₀ / S₀) * Real.exp (-2 * sg.gap * t) := by positivity
  have hdist := alignment_feature_distance H_t P hP
    ((B₀ / S₀) * Real.exp (-2 * sg.gap * t)) hε_nonneg hH_ne halign
  exact ⟨halign, hdist⟩

/-- **Discrete-Time Spectral Bias Subspace Alignment Theorem (`spectral_bias_alignment_disc_ge`)**:
    Under discrete `T`-step µP spectral evolution with `sg : SpectralGap` and step size `η > 0`,
    if the feature matrix `H_T` after `T` steps has signal energy
    `frobeniusSq (H_T * P) ≥ (1 + η * sg.leading) ^ (2 * T) * S₀` (`S₀ > 0`) and residual bulk energy
    `frobeniusSq (H_T - H_T * P) ≤ (1 + η * sg.bulk) ^ (2 * T) * B₀` (`B₀ ≥ 0`), then:
    1. `alignment H_T P ≥ 1 - (B₀ / S₀) * ((1 + η * sg.bulk) / (1 + η * sg.leading)) ^ (2 * T)`,
    2. `frobeniusSq (H_T - H_T * P) ≤ ((B₀ / S₀) * ((1 + η * sg.bulk) / (1 + η * sg.leading)) ^ (2 * T)) * frobeniusSq H_T`. -/
theorem spectral_bias_alignment_disc_ge {n m : ℕ}
    (sg : SpectralGap)
    (H_T : Matrix (Fin n) (Fin m) ℝ) (P : Matrix (Fin m) (Fin m) ℝ)
    (hP : IsOrthoProjector P)
    {η S₀ B₀ : ℝ} (T : ℕ) (hη : 0 < η) (hS₀ : 0 < S₀) (hB₀ : 0 ≤ B₀)
    (hS : (1 + η * sg.leading) ^ (2 * T) * S₀ ≤ frobeniusSq (H_T * P))
    (hB : frobeniusSq (H_T - H_T * P) ≤ (1 + η * sg.bulk) ^ (2 * T) * B₀) :
    1 - (B₀ / S₀) * ((1 + η * sg.bulk) / (1 + η * sg.leading)) ^ (2 * T) ≤ alignment H_T P ∧
    frobeniusSq (H_T - H_T * P) ≤
      ((B₀ / S₀) * ((1 + η * sg.bulk) / (1 + η * sg.leading)) ^ (2 * T)) * frobeniusSq H_T := by
  set S_T := (1 + η * sg.leading) ^ (2 * T) * S₀
  set B_T := (1 + η * sg.bulk) ^ (2 * T) * B₀
  have hlead := leading_pos_of_spectralGap sg
  have hbulk := sg.bulk_nonneg
  have hST_pos : 0 < S_T := mul_pos (pow_pos (by positivity) (2 * T)) hS₀
  have hBT_nonneg : 0 ≤ B_T := mul_nonneg (pow_nonneg (by positivity) (2 * T)) hB₀
  obtain ⟨hH_ne, _, _, _, halign_STBT⟩ :=
    alignment_lower_bound_of_signal_bulk H_T P hP hST_pos hBT_nonneg hS hB
  have hratio : B_T / S_T ≤
      (B₀ / S₀) * ((1 + η * sg.bulk) / (1 + η * sg.leading)) ^ (2 * T) :=
    bulk_div_signal_disc_le sg T hη hS₀ hB₀ (le_refl S_T) (le_refl B_T)
  have halign :
      1 - (B₀ / S₀) * ((1 + η * sg.bulk) / (1 + η * sg.leading)) ^ (2 * T) ≤
        alignment H_T P := by
    linarith
  have hε_nonneg :
      0 ≤ (B₀ / S₀) * ((1 + η * sg.bulk) / (1 + η * sg.leading)) ^ (2 * T) := by
    positivity
  have hdist := alignment_feature_distance H_T P hP
    ((B₀ / S₀) * ((1 + η * sg.bulk) / (1 + η * sg.leading)) ^ (2 * T))
    hε_nonneg hH_ne halign
  exact ⟨halign, hdist⟩

/-- Specialization of `spectral_bias_alignment_cont_ge` to a `SpikedModel d` via `model.gap`. -/
theorem spiked_model_alignment_cont_ge {d n m : ℕ}
    (model : SpikedModel d)
    (H_t : Matrix (Fin n) (Fin m) ℝ) (P : Matrix (Fin m) (Fin m) ℝ)
    (hP : IsOrthoProjector P)
    {t S₀ B₀ : ℝ} (ht : 0 ≤ t) (hS₀ : 0 < S₀) (hB₀ : 0 ≤ B₀)
    (hS : Real.exp (2 * model.gap.leading * t) * S₀ ≤ frobeniusSq (H_t * P))
    (hB : frobeniusSq (H_t - H_t * P) ≤ Real.exp (2 * model.gap.bulk * t) * B₀) :
    1 - (B₀ / S₀) * Real.exp (-2 * model.gap.gap * t) ≤ alignment H_t P ∧
    frobeniusSq (H_t - H_t * P) ≤
      ((B₀ / S₀) * Real.exp (-2 * model.gap.gap * t)) * frobeniusSq H_t :=
  spectral_bias_alignment_cont_ge model.gap H_t P hP ht hS₀ hB₀ hS hB

/-! ### Part 4: Multi-Step Feature Learning Risk Advantage -/

/-- Strict growth of target-aligned kernel energy after `T ≥ 1` steps of µP feature learning:
    if `K_T ≥ (1 + η * γ) ^ (2 * T) * K₀` with `η > 0`, `γ > 0`, and `K₀ > 0`, then `K₀ < K_T`. -/
theorem multistep_kernel_strict_growth {η γ K₀ K_T : ℝ} {T : ℕ}
    (hη : 0 < η) (hγ : 0 < γ) (hT : 1 ≤ T) (hK₀ : 0 < K₀)
    (hK_T : (1 + η * γ) ^ (2 * T) * K₀ ≤ K_T) :
    K₀ < K_T := by
  have h2T : 1 ≤ 2 * T := by omega
  have hlt := modeAmpDisc_lt_of_pos hη hγ h2T hK₀
  unfold modeAmpDisc at hlt
  exact lt_of_lt_of_le hlt hK_T

/-- **Multi-Step Feature Learning Risk Advantage (`multistep_kernel_growth_improves_ridge_mse`)**:
    If `T ≥ 1` steps of µP feature learning with step size `η > 0` and `sg : SpectralGap`
    amplify the target-aligned kernel energy from `K₀ > 0` to
    `K_T ≥ (1 + η * sg.leading) ^ (2 * T) * K₀`, then:
    1. `K₀ < K_T`,
    2. For any `lam > 0` and nonzero target offset `Δy ≠ 0`, the squared ridge regularization
       bias strictly decreases: `biasSq K_T lam Δy < biasSq K₀ lam Δy`,
    3. In the signal-dominant regime (`K₀ * σ ^ 2 < lam * Δy ^ 2` and `K_T * σ ^ 2 ≤ lam * Δy ^ 2`),
       the total expected ridge MSE strictly decreases:
       `ridgeExpectedMSE K_T lam Δy σ < ridgeExpectedMSE K₀ lam Δy σ`. -/
theorem multistep_kernel_growth_improves_ridge_mse
    (sg : SpectralGap) {η K₀ K_T lam Δy σ : ℝ} {T : ℕ}
    (hη : 0 < η) (hT : 1 ≤ T) (hK₀ : 0 < K₀)
    (hK_T : (1 + η * sg.leading) ^ (2 * T) * K₀ ≤ K_T)
    (hlam : 0 < lam) (hΔy : Δy ≠ 0)
    (hsnr₀ : K₀ * σ ^ 2 < lam * Δy ^ 2)
    (hsnr_T : K_T * σ ^ 2 ≤ lam * Δy ^ 2) :
    K₀ < K_T ∧
    biasSq K_T lam Δy < biasSq K₀ lam Δy ∧
    ridgeExpectedMSE K_T lam Δy σ < ridgeExpectedMSE K₀ lam Δy σ := by
  have hK_lt : K₀ < K_T :=
    multistep_kernel_strict_growth hη (leading_pos_of_spectralGap sg) hT hK₀ hK_T
  have hbias : biasSq K_T lam Δy < biasSq K₀ lam Δy :=
    biasSq_strict_anti (le_of_lt hK₀) hK_lt hlam hΔy
  have hmse : ridgeExpectedMSE K_T lam Δy σ < ridgeExpectedMSE K₀ lam Δy σ :=
    ridge_mse_strict_improvement_of_kernel_growth (le_of_lt hK₀) hK_lt hlam hsnr₀ hsnr_T
  exact ⟨hK_lt, hbias, hmse⟩

end OpenTheory.MultiStepSpectralBias
