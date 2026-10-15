import OpenTheory.Data.Spectrum
import OpenTheory.Data.Model
import OpenTheory.Parameterization.MuP
import OpenTheory.Parameterization.NTK
import OpenTheory.Representation.FeatureLearningAdvantage
import OpenTheory.Dynamics.FeatureAlignment
import OpenTheory.Generalization.RidgeRisk
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Ring

/-! Spiked-Covariance Spectral Risk Decomposition, Effective Dimension Collapse,
and Sample Complexity Separation between µP Feature Learning and Frozen-Kernel NTK.

This module connects spectral subspace alignment (`OpenTheory.mup_spectral_alignment_of_ge_horizon`
and `OpenTheory.FeatureLearningAdvantage`) with ridge effective degrees of freedom and
expected test risk on spiked-covariance data (`SpikedModel` / `SpectralGap`):

1. **Spiked-Covariance Spectral Risk Decomposition**:
   Decomposes total feature kernel energy `K_total` into effective signal kernel
   `signalKernel K_total α_align = α_align * K_total` on the `r`-dimensional spike subspace
   and bulk leakage kernel `bulkKernel K_total α_align = (1 - α_align) * K_total` across the
   `d_bulk`-dimensional isotropic bulk. Defines `spikedExpectedMSE` and proves monotonicity
   and strict decrease under bulk leakage suppression and signal amplification.

2. **Effective Degrees of Freedom & Ambient Dimension Independence via µP Alignment**:
   Defines `spikedEffectiveDim K_sig K_bulk r d_bulk lam` and proves that frozen isotropic
   NTK initialization (`K_bulk = c_0 > 0`) suffers effective dimension scaling linearly with
   `d_bulk`, whereas µP feature alignment (`K_bulk ≤ ε * K_total`) collapses the effective
   dimension to `r + d_bulk * ε * (K_total / lam)` (and below `r + δ_df` once `ε` is small),
   removing ambient dimension dependence.

3. **Explicit Sample Complexity Separation (µP vs. Frozen NTK)**:
   Defines `spikedVariance` and `requiredSampleScale` for achieving target variance `ε_var > 0`,
   proves sample size sufficiency (`spikedVariance_le_of_sample_ge`), and establishes a strict
   sample complexity separation between µP feature learning and frozen NTK, collapsing to pure
   intrinsic-rank scaling `O(r)` at full subspace alignment (`K_bulk = 0`).

4. **End-to-End Spiked-Covariance µP Alignment + Risk Master Guarantee**:
   Combines `mup_spectral_alignment_of_ge_horizon` with `spikedExpectedMSE` and
   `spikedEffectiveDim` to certify joint subspace alignment (`1 - ε ≤ alignment H_t P`),
   off-subspace residual suppression (`frobeniusSq (H_t - H_t * P) ≤ ε * frobeniusSq H_t`),
   and quantitative bulk variance / effective dimension bounds after `alignmentHorizon`.
-/

noncomputable section

namespace OpenTheory.SpikedSampleComplexity

open OpenTheory
open OpenTheory.FeatureLearningAdvantage

/-! ### Part 1: Spiked-Covariance Spectral Risk Decomposition -/

/-- Effective feature kernel energy projected onto the leading `r`-dimensional signal spike
    subspace at alignment level `α_align ∈ [0, 1]`. -/
def signalKernel (K_total α_align : ℝ) : ℝ :=
  α_align * K_total

/-- Residual feature kernel energy leaking into the `d_bulk`-dimensional orthogonal bulk
    subspace at alignment level `α_align ∈ [0, 1]`. -/
def bulkKernel (K_total α_align : ℝ) : ℝ :=
  (1 - α_align) * K_total

/-- Conservation of total kernel energy across the signal spike and orthogonal bulk subspaces. -/
theorem signalKernel_add_bulkKernel (K_total α_align : ℝ) :
    signalKernel K_total α_align + bulkKernel K_total α_align = K_total := by
  unfold signalKernel bulkKernel
  ring

/-- Nonnegativity of `signalKernel` and `bulkKernel` when `0 ≤ K_total` and `α_align ∈ [0, 1]`. -/
theorem signalKernel_bulkKernel_nonneg {K_total α_align : ℝ}
    (hK : 0 ≤ K_total) (hα0 : 0 ≤ α_align) (hα1 : α_align ≤ 1) :
    0 ≤ signalKernel K_total α_align ∧ 0 ≤ bulkKernel K_total α_align := by
  unfold signalKernel bulkKernel
  have h1 : 0 ≤ 1 - α_align := sub_nonneg.mpr hα1
  exact ⟨mul_nonneg hα0 hK, mul_nonneg h1 hK⟩

/-- Exact correspondence between `bulkKernel` and off-subspace Frobenius residual
    `frobeniusSq (H - H * P)` for any orthogonal projector `P`. -/
theorem bulkKernel_eq_frobeniusSq_residual {n m : ℕ}
    (H : Matrix (Fin n) (Fin m) ℝ) (P : Matrix (Fin m) (Fin m) ℝ)
    (hP : IsOrthoProjector P) :
    bulkKernel (frobeniusSq H) (alignment H P) = frobeniusSq (H - H * P) := by
  unfold bulkKernel
  exact (frobeniusSq_residual_eq_one_sub_alignment H P hP).symm

/-- Exact correspondence between `signalKernel` and projected Frobenius energy
    `frobeniusSq (H * P)` for any orthogonal projector `P`. -/
theorem signalKernel_eq_frobeniusSq_proj {n m : ℕ}
    (H : Matrix (Fin n) (Fin m) ℝ) (P : Matrix (Fin m) (Fin m) ℝ)
    (hP : IsOrthoProjector P) :
    signalKernel (frobeniusSq H) (alignment H P) = frobeniusSq (H * P) := by
  have hsum := signalKernel_add_bulkKernel (frobeniusSq H) (alignment H P)
  have hbulk := bulkKernel_eq_frobeniusSq_residual H P hP
  have hpyth := frobeniusSq_sub_mul_orthoProjector H P hP
  linarith

/-- Variance contribution across `r` signal eigenmodes (with effective kernel `K_sig`) and
    `d_bulk` bulk noise eigenmodes (with leakage kernel `K_bulk`) over `n` samples. -/
def spikedVariance (K_sig K_bulk : ℝ) (r d_bulk : ℕ) (lam σ : ℝ) (n : ℕ) : ℝ :=
  ((r : ℝ) * (K_sig / (K_sig + lam)) ^ 2 +
    (d_bulk : ℝ) * (K_bulk / (K_bulk + lam)) ^ 2) * (σ ^ 2 / (n : ℝ))

/-- Expected test MSE on spiked-covariance data with signal rank `r`, orthogonal bulk
    dimension `d_bulk`, regularization `lam`, target energy `Δy²` supported on the signal spike,
    and label noise variance `σ²` over `n` samples. -/
def spikedExpectedMSE (K_sig K_bulk : ℝ) (r d_bulk : ℕ) (lam Δy σ : ℝ) (n : ℕ) : ℝ :=
  biasSq K_sig lam Δy +
    ((r : ℝ) * (K_sig / (K_sig + lam)) ^ 2 +
      (d_bulk : ℝ) * (K_bulk / (K_bulk + lam)) ^ 2) * (σ ^ 2 / (n : ℝ))

/-- `spikedExpectedMSE` decomposes as regularization bias on the signal spike plus `spikedVariance`. -/
theorem spikedExpectedMSE_eq_biasSq_add_spikedVariance
    (K_sig K_bulk : ℝ) (r d_bulk : ℕ) (lam Δy σ : ℝ) (n : ℕ) :
    spikedExpectedMSE K_sig K_bulk r d_bulk lam Δy σ n =
      biasSq K_sig lam Δy + spikedVariance K_sig K_bulk r d_bulk lam σ n :=
  rfl

/-- In the single-mode, single-sample case (`r = 1, d_bulk = 0, n = 1`), `spikedExpectedMSE`
    reduces identically to `FeatureLearningAdvantage.ridgeExpectedMSE`. -/
theorem spikedExpectedMSE_rank_one_zero_bulk (K_sig K_bulk lam Δy σ : ℝ) :
    spikedExpectedMSE K_sig K_bulk 1 0 lam Δy σ 1 =
      ridgeExpectedMSE K_sig lam Δy σ := by
  unfold spikedExpectedMSE ridgeExpectedMSE varianceTerm
  simp

/-- Nonnegativity of `spikedVariance`. -/
theorem spikedVariance_nonneg (K_sig K_bulk : ℝ) (r d_bulk : ℕ) (lam σ : ℝ) (n : ℕ) :
    0 ≤ spikedVariance K_sig K_bulk r d_bulk lam σ n := by
  unfold spikedVariance
  positivity

/-- Nonnegativity of `spikedExpectedMSE`. -/
theorem spikedExpectedMSE_nonneg (K_sig K_bulk : ℝ) (r d_bulk : ℕ) (lam Δy σ : ℝ) (n : ℕ) :
    0 ≤ spikedExpectedMSE K_sig K_bulk r d_bulk lam Δy σ n := by
  unfold spikedExpectedMSE biasSq
  positivity

/-- Helper lemma: the shrinkage ratio `K / (K + lam)` is monotone increasing on `0 ≤ K` for `lam > 0`. -/
lemma shrinkage_ratio_le_of_le {K₁ K₀ lam : ℝ}
    (hK₁ : 0 ≤ K₁) (hle : K₁ ≤ K₀) (hlam : 0 < lam) :
    K₁ / (K₁ + lam) ≤ K₀ / (K₀ + lam) := by
  have hd₁ : 0 < K₁ + lam := by linarith
  have hd₀ : 0 < K₀ + lam := by linarith
  rw [div_le_div_iff₀ hd₁ hd₀]
  nlinarith

/-- Helper lemma: the shrinkage ratio `K / (K + lam)` is strictly increasing on `0 ≤ K` for `lam > 0`. -/
lemma shrinkage_ratio_lt_of_lt {K₁ K₀ lam : ℝ}
    (hK₁ : 0 ≤ K₁) (hlt : K₁ < K₀) (hlam : 0 < lam) :
    K₁ / (K₁ + lam) < K₀ / (K₀ + lam) := by
  have hd₁ : 0 < K₁ + lam := by linarith
  have hd₀ : 0 < K₀ + lam := by linarith
  rw [div_lt_div_iff₀ hd₁ hd₀]
  nlinarith

/-- Helper lemma: the squared shrinkage factor `(K / (K + lam)) ^ 2` is monotone increasing. -/
lemma shrinkage_sq_le_of_le {K₁ K₀ lam : ℝ}
    (hK₁ : 0 ≤ K₁) (hle : K₁ ≤ K₀) (hlam : 0 < lam) :
    (K₁ / (K₁ + lam)) ^ 2 ≤ (K₀ / (K₀ + lam)) ^ 2 := by
  have hratio := shrinkage_ratio_le_of_le hK₁ hle hlam
  have hnonneg : 0 ≤ K₁ / (K₁ + lam) := div_nonneg hK₁ (by linarith)
  nlinarith

/-- Helper lemma: the squared shrinkage factor `(K / (K + lam)) ^ 2` is strictly increasing. -/
lemma shrinkage_sq_lt_of_lt {K₁ K₀ lam : ℝ}
    (hK₁ : 0 ≤ K₁) (hlt : K₁ < K₀) (hlam : 0 < lam) :
    (K₁ / (K₁ + lam)) ^ 2 < (K₀ / (K₀ + lam)) ^ 2 := by
  have hratio := shrinkage_ratio_lt_of_lt hK₁ hlt hlam
  have hnonneg : 0 ≤ K₁ / (K₁ + lam) := div_nonneg hK₁ (by linarith)
  nlinarith

/-- **Monotonicity and Strict Decrease of Spiked Risk in Bulk Kernel Leakage
    (`spikedExpectedMSE_mono_in_bulk`)**:
    Reducing bulk kernel leakage from `K_bulk₀` to `K_bulk₁ ≤ K_bulk₀` (`0 ≤ K_bulk₁`, `0 < lam`)
    monotonically reduces `spikedExpectedMSE`, and **strictly** reduces it whenever
    `0 < d_bulk`, `0 < σ`, `0 < n`, and `K_bulk₁ < K_bulk₀`. -/
theorem spikedExpectedMSE_mono_in_bulk
    {K_sig K_bulk₁ K_bulk₀ : ℝ} {r d_bulk : ℕ} {lam Δy σ : ℝ} {n : ℕ}
    (hK₁ : 0 ≤ K_bulk₁) (hle : K_bulk₁ ≤ K_bulk₀) (hlam : 0 < lam) :
    spikedExpectedMSE K_sig K_bulk₁ r d_bulk lam Δy σ n ≤
      spikedExpectedMSE K_sig K_bulk₀ r d_bulk lam Δy σ n ∧
    (0 < d_bulk → 0 < σ → 0 < n → K_bulk₁ < K_bulk₀ →
      spikedExpectedMSE K_sig K_bulk₁ r d_bulk lam Δy σ n <
        spikedExpectedMSE K_sig K_bulk₀ r d_bulk lam Δy σ n) := by
  unfold spikedExpectedMSE
  constructor
  · have hsq := shrinkage_sq_le_of_le hK₁ hle hlam
    have hdb : 0 ≤ (d_bulk : ℝ) := Nat.cast_nonneg d_bulk
    have hsn : 0 ≤ σ ^ 2 / (n : ℝ) := by positivity
    have hmul1 : (d_bulk : ℝ) * (K_bulk₁ / (K_bulk₁ + lam)) ^ 2 ≤
        (d_bulk : ℝ) * (K_bulk₀ / (K_bulk₀ + lam)) ^ 2 :=
      mul_le_mul_of_nonneg_left hsq hdb
    have hmul2 : ((r : ℝ) * (K_sig / (K_sig + lam)) ^ 2 +
          (d_bulk : ℝ) * (K_bulk₁ / (K_bulk₁ + lam)) ^ 2) * (σ ^ 2 / (n : ℝ)) ≤
        ((r : ℝ) * (K_sig / (K_sig + lam)) ^ 2 +
          (d_bulk : ℝ) * (K_bulk₀ / (K_bulk₀ + lam)) ^ 2) * (σ ^ 2 / (n : ℝ)) :=
      mul_le_mul_of_nonneg_right (by linarith) hsn
    linarith
  · intro hdb hσ hn hlt
    have hsq := shrinkage_sq_lt_of_lt hK₁ hlt hlam
    have hdb_pos : 0 < (d_bulk : ℝ) := Nat.cast_pos.mpr hdb
    have hn_pos : 0 < (n : ℝ) := Nat.cast_pos.mpr hn
    have hsn_pos : 0 < σ ^ 2 / (n : ℝ) := div_pos (sq_pos_of_pos hσ) hn_pos
    have hmul1 : (d_bulk : ℝ) * (K_bulk₁ / (K_bulk₁ + lam)) ^ 2 <
        (d_bulk : ℝ) * (K_bulk₀ / (K_bulk₀ + lam)) ^ 2 :=
      mul_lt_mul_of_pos_left hsq hdb_pos
    have hmul2 : ((r : ℝ) * (K_sig / (K_sig + lam)) ^ 2 +
          (d_bulk : ℝ) * (K_bulk₁ / (K_bulk₁ + lam)) ^ 2) * (σ ^ 2 / (n : ℝ)) <
        ((r : ℝ) * (K_sig / (K_sig + lam)) ^ 2 +
          (d_bulk : ℝ) * (K_bulk₀ / (K_bulk₀ + lam)) ^ 2) * (σ ^ 2 / (n : ℝ)) :=
      mul_lt_mul_of_pos_right (by linarith) hsn_pos
    linarith

/-- **Joint Signal Amplification and Bulk Suppression Risk Reduction
    (`spikedExpectedMSE_strict_decrease_of_signal_and_bulk`)**:
    When feature learning increases the signal kernel from `K_sig₀` to `K_sig₁ > K_sig₀`
    in the signal-dominant regime (`K_sig₀ * ((r : ℝ) * (σ ^ 2 / (n : ℝ))) < lam * Δy ^ 2`
    and `K_sig₁ * ((r : ℝ) * (σ ^ 2 / (n : ℝ))) ≤ lam * Δy ^ 2`) while simultaneously
    reducing bulk leakage `0 ≤ K_bulk₁ ≤ K_bulk₀`, the total spiked expected test MSE
    strictly decreases. -/
theorem spikedExpectedMSE_strict_decrease_of_signal_and_bulk
    {K_sig₀ K_sig₁ K_bulk₁ K_bulk₀ : ℝ} {r d_bulk : ℕ} {lam Δy σ : ℝ} {n : ℕ}
    (hK_sig₀ : 0 ≤ K_sig₀) (hK_sig : K_sig₀ < K_sig₁)
    (hK_bulk₁ : 0 ≤ K_bulk₁) (hK_bulk : K_bulk₁ ≤ K_bulk₀)
    (hlam : 0 < lam)
    (hsnr₀ : K_sig₀ * ((r : ℝ) * (σ ^ 2 / (n : ℝ))) < lam * Δy ^ 2)
    (hsnr₁ : K_sig₁ * ((r : ℝ) * (σ ^ 2 / (n : ℝ))) ≤ lam * Δy ^ 2) :
    spikedExpectedMSE K_sig₁ K_bulk₁ r d_bulk lam Δy σ n <
      spikedExpectedMSE K_sig₀ K_bulk₀ r d_bulk lam Δy σ n := by
  have hbulk_mono := (spikedExpectedMSE_mono_in_bulk
    (K_sig := K_sig₁) (r := r) (d_bulk := d_bulk) (Δy := Δy) (σ := σ) (n := n)
    hK_bulk₁ hK_bulk hlam).1
  suffices hsig_strict :
      spikedExpectedMSE K_sig₁ K_bulk₀ r d_bulk lam Δy σ n <
        spikedExpectedMSE K_sig₀ K_bulk₀ r d_bulk lam Δy σ n by
    linarith
  unfold spikedExpectedMSE
  let σ_eff_sq : ℝ := (r : ℝ) * (σ ^ 2 / (n : ℝ))
  have hd₀ : 0 < K_sig₀ + lam := by linarith
  have hd₁ : 0 < K_sig₁ + lam := by linarith
  have hdiff :
      (biasSq K_sig₀ lam Δy + (K_sig₀ / (K_sig₀ + lam)) ^ 2 * σ_eff_sq) -
        (biasSq K_sig₁ lam Δy + (K_sig₁ / (K_sig₁ + lam)) ^ 2 * σ_eff_sq) =
      lam * (K_sig₁ - K_sig₀) *
        ((K_sig₁ + lam) * (lam * Δy ^ 2 - K_sig₀ * σ_eff_sq) +
          (K_sig₀ + lam) * (lam * Δy ^ 2 - K_sig₁ * σ_eff_sq)) /
        ((K_sig₀ + lam) ^ 2 * (K_sig₁ + lam) ^ 2) := by
    unfold biasSq
    field_simp [ne_of_gt hd₀, ne_of_gt hd₁]
    ring
  have hsub : 0 < K_sig₁ - K_sig₀ := sub_pos.mpr hK_sig
  have hterm₀ : 0 < (K_sig₁ + lam) * (lam * Δy ^ 2 - K_sig₀ * σ_eff_sq) :=
    mul_pos hd₁ (sub_pos.mpr hsnr₀)
  have hterm₁ : 0 ≤ (K_sig₀ + lam) * (lam * Δy ^ 2 - K_sig₁ * σ_eff_sq) :=
    mul_nonneg (le_of_lt hd₀) (sub_nonneg.mpr hsnr₁)
  have hsum : 0 < (K_sig₁ + lam) * (lam * Δy ^ 2 - K_sig₀ * σ_eff_sq) +
      (K_sig₀ + lam) * (lam * Δy ^ 2 - K_sig₁ * σ_eff_sq) := by linarith
  have hpos : 0 < lam * (K_sig₁ - K_sig₀) *
      ((K_sig₁ + lam) * (lam * Δy ^ 2 - K_sig₀ * σ_eff_sq) +
        (K_sig₀ + lam) * (lam * Δy ^ 2 - K_sig₁ * σ_eff_sq)) /
      ((K_sig₀ + lam) ^ 2 * (K_sig₁ + lam) ^ 2) := by positivity
  have hsig_part :
      biasSq K_sig₁ lam Δy + (K_sig₁ / (K_sig₁ + lam)) ^ 2 * σ_eff_sq <
        biasSq K_sig₀ lam Δy + (K_sig₀ / (K_sig₀ + lam)) ^ 2 * σ_eff_sq := by
    linarith
  have hexpand₀ :
      ((r : ℝ) * (K_sig₀ / (K_sig₀ + lam)) ^ 2 +
        (d_bulk : ℝ) * (K_bulk₀ / (K_bulk₀ + lam)) ^ 2) * (σ ^ 2 / (n : ℝ)) =
      (K_sig₀ / (K_sig₀ + lam)) ^ 2 * σ_eff_sq +
        (d_bulk : ℝ) * (K_bulk₀ / (K_bulk₀ + lam)) ^ 2 * (σ ^ 2 / (n : ℝ)) := by
    dsimp [σ_eff_sq]
    ring
  have hexpand₁ :
      ((r : ℝ) * (K_sig₁ / (K_sig₁ + lam)) ^ 2 +
        (d_bulk : ℝ) * (K_bulk₀ / (K_bulk₀ + lam)) ^ 2) * (σ ^ 2 / (n : ℝ)) =
      (K_sig₁ / (K_sig₁ + lam)) ^ 2 * σ_eff_sq +
        (d_bulk : ℝ) * (K_bulk₀ / (K_bulk₀ + lam)) ^ 2 * (σ ^ 2 / (n : ℝ)) := by
    dsimp [σ_eff_sq]
    ring
  linarith

/-! ### Part 2: Effective Degrees of Freedom & Ambient Dimension Independence via µP Alignment -/

/-- Ridge effective degrees of freedom on a spiked spectrum with `r` signal eigenmodes
    at kernel scale `K_sig` and `d_bulk` bulk eigenmodes at leakage scale `K_bulk`:
    `df(λ) = r * K_sig / (K_sig + λ) + d_bulk * K_bulk / (K_bulk + λ)`. -/
def spikedEffectiveDim (K_sig K_bulk : ℝ) (r d_bulk : ℕ) (lam : ℝ) : ℝ :=
  (r : ℝ) * (K_sig / (K_sig + lam)) + (d_bulk : ℝ) * (K_bulk / (K_bulk + lam))

/-- Fundamental bounds on `spikedEffectiveDim`: for `0 ≤ K_sig`, `0 ≤ K_bulk`, and `0 < lam`,
    the effective dimension lies in `[0, r + d_bulk]`. -/
theorem spikedEffectiveDim_bounds {K_sig K_bulk : ℝ} (r d_bulk : ℕ) {lam : ℝ}
    (hK_sig : 0 ≤ K_sig) (hK_bulk : 0 ≤ K_bulk) (hlam : 0 < lam) :
    0 ≤ spikedEffectiveDim K_sig K_bulk r d_bulk lam ∧
    spikedEffectiveDim K_sig K_bulk r d_bulk lam ≤ (r : ℝ) + (d_bulk : ℝ) := by
  unfold spikedEffectiveDim
  have hd_sig : 0 < K_sig + lam := by linarith
  have hd_bulk : 0 < K_bulk + lam := by linarith
  constructor
  · positivity
  · have h1 : K_sig / (K_sig + lam) ≤ 1 := by
      rw [div_le_one hd_sig]
      linarith
    have h2 : K_bulk / (K_bulk + lam) ≤ 1 := by
      rw [div_le_one hd_bulk]
      linarith
    have hr : (r : ℝ) * (K_sig / (K_sig + lam)) ≤ (r : ℝ) * 1 :=
      mul_le_mul_of_nonneg_left h1 (Nat.cast_nonneg r)
    have hdb : (d_bulk : ℝ) * (K_bulk / (K_bulk + lam)) ≤ (d_bulk : ℝ) * 1 :=
      mul_le_mul_of_nonneg_left h2 (Nat.cast_nonneg d_bulk)
    linarith

/-- At perfect subspace alignment (`K_bulk = 0`), `spikedEffectiveDim` is bounded above by the
    intrinsic spike rank `r`, completely independent of the bulk dimension `d_bulk`. -/
theorem spikedEffectiveDim_zero_bulk_le_rank {K_sig : ℝ} (r d_bulk : ℕ) {lam : ℝ}
    (hK_sig : 0 ≤ K_sig) (hlam : 0 < lam) :
    spikedEffectiveDim K_sig 0 r d_bulk lam = (r : ℝ) * (K_sig / (K_sig + lam)) ∧
    spikedEffectiveDim K_sig 0 r d_bulk lam ≤ (r : ℝ) := by
  unfold spikedEffectiveDim
  have hd_sig : 0 < K_sig + lam := by linarith
  have h1 : K_sig / (K_sig + lam) ≤ 1 := by
    rw [div_le_one hd_sig]
    linarith
  have hr : (r : ℝ) * (K_sig / (K_sig + lam)) ≤ (r : ℝ) * 1 :=
    mul_le_mul_of_nonneg_left h1 (Nat.cast_nonneg r)
  refine ⟨by simp, by simp; linarith⟩

/-- **Isotropic / Unaligned NTK Effective Dimension (`unaligned_effectiveDim_scales_with_d_bulk`)**:
    When the bulk kernel stays at a frozen isotropic scale `K_bulk = c_0 > 0` (`0 ≤ K_sig`, `0 < lam`),
    the effective dimension grows at least linearly with the bulk dimension `d_bulk`:
    `spikedEffectiveDim K_sig c_0 r d_bulk lam ≥ d_bulk * (c_0 / (c_0 + lam)) > 0` (for `d_bulk > 0`). -/
theorem unaligned_effectiveDim_scales_with_d_bulk
    {K_sig c_0 : ℝ} (r d_bulk : ℕ) {lam : ℝ}
    (hK_sig : 0 ≤ K_sig) (hc_0 : 0 < c_0) (hlam : 0 < lam) :
    (d_bulk : ℝ) * (c_0 / (c_0 + lam)) ≤ spikedEffectiveDim K_sig c_0 r d_bulk lam ∧
    (0 < d_bulk → 0 < spikedEffectiveDim K_sig c_0 r d_bulk lam) := by
  unfold spikedEffectiveDim
  have hd_sig : 0 < K_sig + lam := by linarith
  have hd_c : 0 < c_0 + lam := by linarith
  have hsig_nonneg : 0 ≤ (r : ℝ) * (K_sig / (K_sig + lam)) := by positivity
  constructor
  · linarith
  · intro hdb
    have hdb_pos : 0 < (d_bulk : ℝ) := Nat.cast_pos.mpr hdb
    have hbulk_pos : 0 < (d_bulk : ℝ) * (c_0 / (c_0 + lam)) := by positivity
    linarith

/-- **µP Feature-Aligned Effective Dimension Collapse (`mup_aligned_effectiveDim_le_rank_add_eps`)**:
    When µP feature learning suppresses the bulk kernel leakage to `0 ≤ K_bulk ≤ ε * K_total`
    (`0 ≤ K_sig`, `0 ≤ K_total`, `0 ≤ ε`, `0 < lam`), the effective dimension satisfies:
    1. `spikedEffectiveDim K_sig K_bulk r d_bulk lam ≤ (r : ℝ) + (d_bulk : ℝ) * ε * (K_total / lam)`, and
    2. For any target tolerance `δ_df ≥ 0`, if `(d_bulk : ℝ) * ε * (K_total / lam) ≤ δ_df`, then
       `spikedEffectiveDim K_sig K_bulk r d_bulk lam ≤ (r : ℝ) + δ_df`, independent of ambient dimension. -/
theorem mup_aligned_effectiveDim_le_rank_add_eps
    {K_sig K_bulk K_total ε : ℝ} (r d_bulk : ℕ) {lam : ℝ}
    (hK_sig : 0 ≤ K_sig) (hK_bulk : 0 ≤ K_bulk)
    (hlam : 0 < lam) (hbulk_le : K_bulk ≤ ε * K_total) :
    spikedEffectiveDim K_sig K_bulk r d_bulk lam ≤
      (r : ℝ) + (d_bulk : ℝ) * ε * (K_total / lam) ∧
    ∀ δ_df : ℝ, (d_bulk : ℝ) * ε * (K_total / lam) ≤ δ_df →
      spikedEffectiveDim K_sig K_bulk r d_bulk lam ≤ (r : ℝ) + δ_df := by
  unfold spikedEffectiveDim
  have hd_sig : 0 < K_sig + lam := by linarith
  have hd_bulk : 0 < K_bulk + lam := by linarith
  have hsig_le : (r : ℝ) * (K_sig / (K_sig + lam)) ≤ (r : ℝ) := by
    have h1 : K_sig / (K_sig + lam) ≤ 1 := by
      rw [div_le_one hd_sig]
      linarith
    have hr := mul_le_mul_of_nonneg_left h1 (Nat.cast_nonneg r)
    linarith
  have hbulk_ratio : K_bulk / (K_bulk + lam) ≤ ε * (K_total / lam) := by
    calc
      K_bulk / (K_bulk + lam) ≤ K_bulk / lam :=
        div_le_div_of_nonneg_left hK_bulk hlam (by linarith)
      _ ≤ (ε * K_total) / lam :=
        div_le_div_of_nonneg_right hbulk_le (le_of_lt hlam)
      _ = ε * (K_total / lam) := mul_div_assoc ε K_total lam
  have hdb_le : (d_bulk : ℝ) * (K_bulk / (K_bulk + lam)) ≤
      (d_bulk : ℝ) * ε * (K_total / lam) := by
    calc
      (d_bulk : ℝ) * (K_bulk / (K_bulk + lam)) ≤
          (d_bulk : ℝ) * (ε * (K_total / lam)) :=
        mul_le_mul_of_nonneg_left hbulk_ratio (Nat.cast_nonneg d_bulk)
      _ = (d_bulk : ℝ) * ε * (K_total / lam) := by ring
  have hmain : (r : ℝ) * (K_sig / (K_sig + lam)) +
      (d_bulk : ℝ) * (K_bulk / (K_bulk + lam)) ≤
      (r : ℝ) + (d_bulk : ℝ) * ε * (K_total / lam) := by linarith
  refine ⟨hmain, fun δ_df hδ => by linarith⟩

/-! ### Part 3: Explicit Sample Complexity Separation Between µP Feature Learning and Frozen NTK -/

/-- Sample complexity scale required to drive `spikedVariance` below target tolerance `ε_var > 0`. -/
def requiredSampleScale (K_sig K_bulk : ℝ) (r d_bulk : ℕ) (lam σ ε_var : ℝ) : ℝ :=
  ((r : ℝ) * (K_sig / (K_sig + lam)) ^ 2 +
    (d_bulk : ℝ) * (K_bulk / (K_bulk + lam)) ^ 2) * σ ^ 2 / ε_var

/-- Bulk sample complexity overhead arising from kernel leakage `K_bulk` across `d_bulk` noise modes. -/
def bulkSampleOverhead (K_bulk : ℝ) (d_bulk : ℕ) (lam σ ε_var : ℝ) : ℝ :=
  (d_bulk : ℝ) * (K_bulk / (K_bulk + lam)) ^ 2 * σ ^ 2 / ε_var

/-- Exact decomposition of `requiredSampleScale` into intrinsic spike sample complexity
    plus `bulkSampleOverhead`. -/
theorem requiredSampleScale_eq_signal_add_bulk
    (K_sig K_bulk : ℝ) (r d_bulk : ℕ) (lam σ ε_var : ℝ) :
    requiredSampleScale K_sig K_bulk r d_bulk lam σ ε_var =
      (r : ℝ) * (K_sig / (K_sig + lam)) ^ 2 * σ ^ 2 / ε_var +
        bulkSampleOverhead K_bulk d_bulk lam σ ε_var := by
  unfold requiredSampleScale bulkSampleOverhead
  ring

/-- Nonnegativity of `requiredSampleScale` when `0 < ε_var`. -/
theorem requiredSampleScale_nonneg (K_sig K_bulk : ℝ) (r d_bulk : ℕ) (lam σ : ℝ)
    {ε_var : ℝ} (hε : 0 < ε_var) :
    0 ≤ requiredSampleScale K_sig K_bulk r d_bulk lam σ ε_var := by
  unfold requiredSampleScale
  positivity

/-- **Sample Size Sufficiency (`spikedVariance_le_of_sample_ge`)**:
    Whenever the sample size `n > 0` satisfies `(n : ℝ) ≥ requiredSampleScale K_sig K_bulk r d_bulk lam σ ε_var`
    for target tolerance `ε_var > 0`, the variance contribution satisfies
    `spikedVariance K_sig K_bulk r d_bulk lam σ n ≤ ε_var`. -/
theorem spikedVariance_le_of_sample_ge
    {K_sig K_bulk : ℝ} {r d_bulk : ℕ} {lam σ ε_var : ℝ} {n : ℕ}
    (hn : 0 < n) (hε : 0 < ε_var)
    (hsample : requiredSampleScale K_sig K_bulk r d_bulk lam σ ε_var ≤ (n : ℝ)) :
    spikedVariance K_sig K_bulk r d_bulk lam σ n ≤ ε_var := by
  have hn_pos : 0 < (n : ℝ) := Nat.cast_pos.mpr hn
  unfold requiredSampleScale at hsample
  unfold spikedVariance
  have hmul : ((r : ℝ) * (K_sig / (K_sig + lam)) ^ 2 +
      (d_bulk : ℝ) * (K_bulk / (K_bulk + lam)) ^ 2) * σ ^ 2 ≤ ε_var * (n : ℝ) := by
    have h1 := (div_le_iff₀ hε).mp hsample
    linarith
  calc
    ((r : ℝ) * (K_sig / (K_sig + lam)) ^ 2 +
        (d_bulk : ℝ) * (K_bulk / (K_bulk + lam)) ^ 2) * (σ ^ 2 / (n : ℝ))
      = (((r : ℝ) * (K_sig / (K_sig + lam)) ^ 2 +
          (d_bulk : ℝ) * (K_bulk / (K_bulk + lam)) ^ 2) * σ ^ 2) / (n : ℝ) := by
        ring
    _ ≤ (ε_var * (n : ℝ)) / (n : ℝ) :=
        div_le_div_of_nonneg_right hmul (le_of_lt hn_pos)
    _ = ε_var := mul_div_cancel_right₀ ε_var (ne_of_gt hn_pos)

/-- **Strict Sample Complexity Separation Between µP Feature Learning and Frozen NTK
    (`mup_vs_ntk_sample_complexity_separation`)**:
    For `0 < d_bulk`, `0 < σ`, `0 < ε_var`, `0 < lam`, and `0 ≤ K_bulk_mup < K_bulk_ntk`:
    1. The bulk sample complexity overhead under µP (`bulkSampleOverhead K_bulk_mup d_bulk lam σ ε_var`)
       is **strictly smaller** than under frozen NTK (`bulkSampleOverhead K_bulk_ntk d_bulk lam σ ε_var`).
    2. For matching signal kernel scale `K_sig`, total `requiredSampleScale` is strictly smaller
       under µP than under frozen NTK.
    3. At perfect subspace alignment (`K_bulk_mup = 0`), `requiredSampleScale K_sig 0 r d_bulk lam σ ε_var`
       equals `(r : ℝ) * (K_sig / (K_sig + lam)) ^ 2 * σ ^ 2 / ε_var`, which depends **solely on the
       intrinsic spike rank `r`** and has zero dependence on `d_bulk` (bounded above by `r * σ² / ε_var`). -/
theorem mup_vs_ntk_sample_complexity_separation
    {K_sig K_bulk_mup K_bulk_ntk : ℝ} {r d_bulk : ℕ} {lam σ ε_var : ℝ}
    (hK_sig : 0 ≤ K_sig) (hK_mup : 0 ≤ K_bulk_mup) (hK_lt : K_bulk_mup < K_bulk_ntk)
    (hdb : 0 < d_bulk) (hlam : 0 < lam) (hσ : 0 < σ) (hε : 0 < ε_var) :
    bulkSampleOverhead K_bulk_mup d_bulk lam σ ε_var <
      bulkSampleOverhead K_bulk_ntk d_bulk lam σ ε_var ∧
    requiredSampleScale K_sig K_bulk_mup r d_bulk lam σ ε_var <
      requiredSampleScale K_sig K_bulk_ntk r d_bulk lam σ ε_var ∧
    requiredSampleScale K_sig 0 r d_bulk lam σ ε_var =
      (r : ℝ) * (K_sig / (K_sig + lam)) ^ 2 * σ ^ 2 / ε_var ∧
    requiredSampleScale K_sig 0 r d_bulk lam σ ε_var ≤
      (r : ℝ) * σ ^ 2 / ε_var := by
  have hsq_lt := shrinkage_sq_lt_of_lt hK_mup hK_lt hlam
  have hdb_pos : 0 < (d_bulk : ℝ) := Nat.cast_pos.mpr hdb
  have hσ2_pos : 0 < σ ^ 2 := sq_pos_of_pos hσ
  have hbulk_lt :
      bulkSampleOverhead K_bulk_mup d_bulk lam σ ε_var <
        bulkSampleOverhead K_bulk_ntk d_bulk lam σ ε_var := by
    unfold bulkSampleOverhead
    have hm1 : (d_bulk : ℝ) * (K_bulk_mup / (K_bulk_mup + lam)) ^ 2 <
        (d_bulk : ℝ) * (K_bulk_ntk / (K_bulk_ntk + lam)) ^ 2 :=
      mul_lt_mul_of_pos_left hsq_lt hdb_pos
    have hm2 : (d_bulk : ℝ) * (K_bulk_mup / (K_bulk_mup + lam)) ^ 2 * σ ^ 2 <
        (d_bulk : ℝ) * (K_bulk_ntk / (K_bulk_ntk + lam)) ^ 2 * σ ^ 2 :=
      mul_lt_mul_of_pos_right hm1 hσ2_pos
    exact div_lt_div_of_pos_right hm2 hε
  have hreq_lt :
      requiredSampleScale K_sig K_bulk_mup r d_bulk lam σ ε_var <
        requiredSampleScale K_sig K_bulk_ntk r d_bulk lam σ ε_var := by
    rw [requiredSampleScale_eq_signal_add_bulk K_sig K_bulk_mup,
      requiredSampleScale_eq_signal_add_bulk K_sig K_bulk_ntk]
    linarith
  have hreq_zero :
      requiredSampleScale K_sig 0 r d_bulk lam σ ε_var =
        (r : ℝ) * (K_sig / (K_sig + lam)) ^ 2 * σ ^ 2 / ε_var := by
    unfold requiredSampleScale
    simp
  have hreq_zero_le :
      requiredSampleScale K_sig 0 r d_bulk lam σ ε_var ≤ (r : ℝ) * σ ^ 2 / ε_var := by
    rw [hreq_zero]
    have hd_sig : 0 < K_sig + lam := by linarith
    have hratio_le : K_sig / (K_sig + lam) ≤ 1 := by
      rw [div_le_one hd_sig]
      linarith
    have hratio_nonneg : 0 ≤ K_sig / (K_sig + lam) := div_nonneg hK_sig (le_of_lt hd_sig)
    have hsq_le : (K_sig / (K_sig + lam)) ^ 2 ≤ 1 := by nlinarith
    have hr_le : (r : ℝ) * (K_sig / (K_sig + lam)) ^ 2 * σ ^ 2 ≤ (r : ℝ) * 1 * σ ^ 2 :=
      mul_le_mul_of_nonneg_right
        (mul_le_mul_of_nonneg_left hsq_le (Nat.cast_nonneg r)) (le_of_lt hσ2_pos)
    rw [mul_one] at hr_le
    exact div_le_div_of_nonneg_right hr_le (le_of_lt hε)
  exact ⟨hbulk_lt, hreq_lt, hreq_zero, hreq_zero_le⟩

/-! ### Part 4: End-to-End Spiked-Covariance µP Alignment + Risk Master Guarantee -/

/-- **End-to-End Spiked-Covariance µP Alignment + Risk Guarantee
    (`mup_spiked_covariance_master_guarantee`)**:
    Given a `MuPScaling d m` with `0 < scale.lr_W1`, a `sg : SpectralGap`, initial signal energy
    `S₀ > 0`, bulk energy `B₀ ≥ 0`, precision `ε > 0`, signal rank `r`, bulk dimension `d_bulk`,
    regularization `lam > 0`, target amplitude `Δy`, noise scale `σ`, and sample count `n`:
    Whenever `t ≥ alignmentHorizon scale.lr_W1 sg.gap S₀ B₀ ε` and the feature matrix `H_t`
    evolves under the µP spectral energy bounds, setting `K_total = frobeniusSq H_t`,
    `K_sig = frobeniusSq (H_t * P)`, and `K_bulk = frobeniusSq (H_t - H_t * P)`:
    1. Subspace alignment reaches `1 - ε ≤ alignment H_t P`,
    2. Off-subspace residual energy satisfies `K_bulk ≤ ε * K_total`,
    3. Spiked effective dimension is bounded by `(r : ℝ) + (d_bulk : ℝ) * ε * (K_total / lam)`, and
    4. Expected test MSE satisfies
       `spikedExpectedMSE K_sig K_bulk r d_bulk lam Δy σ n ≤ biasSq K_sig lam Δy + ((r : ℝ) * (K_sig / (K_sig + lam)) ^ 2 + (d_bulk : ℝ) * (ε * K_total / lam) ^ 2) * (σ ^ 2 / (n : ℝ))`. -/
theorem mup_spiked_covariance_master_guarantee {d n_feat m : ℕ}
    (scale : MuPScaling d m) (hlr : 0 < scale.lr_W1)
    (sg : SpectralGap)
    (H_t : Matrix (Fin n_feat) (Fin m) ℝ) (P : Matrix (Fin m) (Fin m) ℝ)
    (hP : IsOrthoProjector P)
    {t S₀ B₀ ε : ℝ} (hS₀ : 0 < S₀) (hB₀ : 0 ≤ B₀) (hε : 0 < ε)
    (ht : alignmentHorizon scale.lr_W1 sg.gap S₀ B₀ ε ≤ t)
    (hS : Real.exp (2 * sg.leading * (scale.lr_W1 * t)) * S₀ ≤ frobeniusSq (H_t * P))
    (hB : frobeniusSq (H_t - H_t * P) ≤ Real.exp (2 * sg.bulk * (scale.lr_W1 * t)) * B₀)
    (r d_bulk : ℕ) {lam Δy σ : ℝ} (hlam : 0 < lam) (n : ℕ) :
    1 - ε ≤ alignment H_t P ∧
    frobeniusSq (H_t - H_t * P) ≤ ε * frobeniusSq H_t ∧
    spikedEffectiveDim (frobeniusSq (H_t * P)) (frobeniusSq (H_t - H_t * P)) r d_bulk lam ≤
      (r : ℝ) + (d_bulk : ℝ) * ε * (frobeniusSq H_t / lam) ∧
    spikedExpectedMSE (frobeniusSq (H_t * P)) (frobeniusSq (H_t - H_t * P)) r d_bulk lam Δy σ n ≤
      biasSq (frobeniusSq (H_t * P)) lam Δy +
        ((r : ℝ) * (frobeniusSq (H_t * P) / (frobeniusSq (H_t * P) + lam)) ^ 2 +
          (d_bulk : ℝ) * (ε * frobeniusSq H_t / lam) ^ 2) * (σ ^ 2 / (n : ℝ)) := by
  obtain ⟨halign, hbulk_le⟩ :=
    mup_spectral_alignment_of_ge_horizon scale hlr sg H_t P hP hS₀ hB₀ hε ht hS hB
  have hK_sig_nonneg : 0 ≤ frobeniusSq (H_t * P) := frobeniusSq_nonneg _
  have hK_bulk_nonneg : 0 ≤ frobeniusSq (H_t - H_t * P) := frobeniusSq_nonneg _
  have hdf := (mup_aligned_effectiveDim_le_rank_add_eps
    r d_bulk hK_sig_nonneg hK_bulk_nonneg hlam hbulk_le).1
  refine ⟨halign, hbulk_le, hdf, ?_⟩
  unfold spikedExpectedMSE
  have hratio_le :
      frobeniusSq (H_t - H_t * P) / (frobeniusSq (H_t - H_t * P) + lam) ≤
        ε * frobeniusSq H_t / lam := by
    calc
      frobeniusSq (H_t - H_t * P) / (frobeniusSq (H_t - H_t * P) + lam)
        ≤ frobeniusSq (H_t - H_t * P) / lam :=
          div_le_div_of_nonneg_left hK_bulk_nonneg hlam (by linarith)
      _ ≤ (ε * frobeniusSq H_t) / lam :=
          div_le_div_of_nonneg_right hbulk_le (le_of_lt hlam)
  have hratio_nonneg :
      0 ≤ frobeniusSq (H_t - H_t * P) / (frobeniusSq (H_t - H_t * P) + lam) :=
    div_nonneg hK_bulk_nonneg (by linarith)
  have hsq_le :
      (frobeniusSq (H_t - H_t * P) / (frobeniusSq (H_t - H_t * P) + lam)) ^ 2 ≤
        (ε * frobeniusSq H_t / lam) ^ 2 := by
    nlinarith
  have hdb_le :
      (d_bulk : ℝ) * (frobeniusSq (H_t - H_t * P) / (frobeniusSq (H_t - H_t * P) + lam)) ^ 2 ≤
        (d_bulk : ℝ) * (ε * frobeniusSq H_t / lam) ^ 2 :=
    mul_le_mul_of_nonneg_left hsq_le (Nat.cast_nonneg d_bulk)
  have hsn_nonneg : 0 ≤ σ ^ 2 / (n : ℝ) := by positivity
  have hvar_le :
      ((r : ℝ) * (frobeniusSq (H_t * P) / (frobeniusSq (H_t * P) + lam)) ^ 2 +
        (d_bulk : ℝ) * (frobeniusSq (H_t - H_t * P) / (frobeniusSq (H_t - H_t * P) + lam)) ^ 2) *
          (σ ^ 2 / (n : ℝ)) ≤
      ((r : ℝ) * (frobeniusSq (H_t * P) / (frobeniusSq (H_t * P) + lam)) ^ 2 +
        (d_bulk : ℝ) * (ε * frobeniusSq H_t / lam) ^ 2) * (σ ^ 2 / (n : ℝ)) :=
    mul_le_mul_of_nonneg_right (by linarith) hsn_nonneg
  linarith

end OpenTheory.SpikedSampleComplexity
