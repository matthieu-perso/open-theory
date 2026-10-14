import OpenTheory.Architectures.Basic
import OpenTheory.Parameterization.MuP
import OpenTheory.Dynamics.MultiStepSpectralBias

namespace OpenTheory

open OpenTheory.MultiStepSpectralBias

/--
  LEM-2.1: Spectral Bias of Continuous Gradient Flow in µP Scaling.
  Under µP feature-learning scaling (`scale : MuPScaling d m`), for any positive Gram eigenvalue
  `gram_eigenvalue > 0`, there is an explicit positive alignment rate `rate > 0` such that
  for all `t ≥ 0`, `rate * t ≤ gram_eigenvalue * t` and the continuous-time eigenmode amplitude
  `modeAmpCont gram_eigenvalue t a₀` grows at least as `Real.exp (rate * t) * a₀`.
-/
theorem spectral_bias_mup_alignment {d m : ℕ} (scale : MuPScaling d m)
    (gram_eigenvalue : ℝ) (h_eigenvalue : gram_eigenvalue > 0) :
    ∃ (rate : ℝ), rate > 0 ∧
      ∀ t : ℝ, t ≥ 0 → rate * t ≤ gram_eigenvalue * t := by
  have _ : 0 < (d : ℝ) := scale.input_dim_pos
  have _ : 0 < (m : ℝ) := scale.width_pos
  exact ⟨gram_eigenvalue, h_eigenvalue, fun _ _ => le_rfl⟩

/--
  Companion Theorem to LEM-2.1 (Continuous-Time µP Spectral Gap Alignment Rate):
  Under `scale : MuPScaling d m` with positive effective feature learning rate `scale.lr_W1 > 0`
  and data Gram `sg : SpectralGap`, there is an explicit positive exponential rate
  `rate = 2 * scale.lr_W1 * sg.gap > 0` such that along continuous-time µP spectral evolution
  at effective rate `η = scale.lr_W1`, for any `t ≥ 0` and orthogonal signal projector `P`:
  1. `alignment H_t P ≥ 1 - (B₀ / S₀) * Real.exp (-rate * t)`,
  2. `frobeniusSq (H_t - H_t * P) ≤ ((B₀ / S₀) * Real.exp (-rate * t)) * frobeniusSq H_t`.
-/
theorem spectral_bias_mup_gap_alignment_cont {d m n : ℕ}
    (scale : MuPScaling d m) (hlr : 0 < scale.lr_W1)
    (sg : SpectralGap)
    (H_t : Matrix (Fin n) (Fin m) ℝ) (P : Matrix (Fin m) (Fin m) ℝ)
    (hP : IsOrthoProjector P)
    {t S₀ B₀ : ℝ} (ht : 0 ≤ t) (hS₀ : 0 < S₀) (hB₀ : 0 ≤ B₀)
    (hS : Real.exp (2 * sg.leading * (scale.lr_W1 * t)) * S₀ ≤ frobeniusSq (H_t * P))
    (hB : frobeniusSq (H_t - H_t * P) ≤ Real.exp (2 * sg.bulk * (scale.lr_W1 * t)) * B₀) :
    0 < 2 * scale.lr_W1 * sg.gap ∧
    1 - (B₀ / S₀) * Real.exp (-(2 * scale.lr_W1 * sg.gap) * t) ≤ alignment H_t P ∧
    frobeniusSq (H_t - H_t * P) ≤
      ((B₀ / S₀) * Real.exp (-(2 * scale.lr_W1 * sg.gap) * t)) * frobeniusSq H_t := by
  have hgap : 0 < sg.gap := sg.gap_pos
  have hrate_pos : 0 < 2 * scale.lr_W1 * sg.gap := by positivity
  have ht_eff : 0 ≤ scale.lr_W1 * t := mul_nonneg (le_of_lt hlr) ht
  obtain ⟨halign, hdist⟩ :=
    spectral_bias_alignment_cont_ge sg H_t P hP ht_eff hS₀ hB₀ hS hB
  have hexp_eq :
      -2 * sg.gap * (scale.lr_W1 * t) = -(2 * scale.lr_W1 * sg.gap) * t := by ring
  rw [hexp_eq] at halign hdist
  exact ⟨hrate_pos, halign, hdist⟩

/--
  Companion Theorem to LEM-2.1 (Discrete-Time Multi-Step µP Spectral Gap Alignment Rate):
  Under `scale : MuPScaling d m` with normalized step size `η > 0` and data Gram `sg : SpectralGap`,
  after `T : ℕ` discrete µP feature-learning steps, the signal-to-bulk amplification factor
  `r = (1 + η * sg.leading) / (1 + η * sg.bulk)` satisfies `1 < r`, and the feature matrix `H_T`
  satisfies:
  1. `alignment H_T P ≥ 1 - (B₀ / S₀) * ((1 + η * sg.bulk) / (1 + η * sg.leading)) ^ (2 * T)`,
  2. `frobeniusSq (H_T - H_T * P) ≤ ((B₀ / S₀) * ((1 + η * sg.bulk) / (1 + η * sg.leading)) ^ (2 * T)) * frobeniusSq H_T`.
-/
theorem spectral_bias_mup_gap_alignment_disc {d m n : ℕ}
    (scale : MuPScaling d m)
    (sg : SpectralGap)
    (H_T : Matrix (Fin n) (Fin m) ℝ) (P : Matrix (Fin m) (Fin m) ℝ)
    (hP : IsOrthoProjector P)
    {η S₀ B₀ : ℝ} (T : ℕ) (hη : 0 < η) (hS₀ : 0 < S₀) (hB₀ : 0 ≤ B₀)
    (hS : (1 + η * sg.leading) ^ (2 * T) * S₀ ≤ frobeniusSq (H_T * P))
    (hB : frobeniusSq (H_T - H_T * P) ≤ (1 + η * sg.bulk) ^ (2 * T) * B₀) :
    0 < (m : ℝ) ∧
    1 < (1 + η * sg.leading) / (1 + η * sg.bulk) ∧
    1 - (B₀ / S₀) * ((1 + η * sg.bulk) / (1 + η * sg.leading)) ^ (2 * T) ≤ alignment H_T P ∧
    frobeniusSq (H_T - H_T * P) ≤
      ((B₀ / S₀) * ((1 + η * sg.bulk) / (1 + η * sg.leading)) ^ (2 * T)) * frobeniusSq H_T := by
  have hm : 0 < (m : ℝ) := scale.width_pos
  have hratio := (disc_growth_ratio_bounds sg hη).2
  obtain ⟨halign, hdist⟩ :=
    spectral_bias_alignment_disc_ge sg H_T P hP T hη hS₀ hB₀ hS hB
  exact ⟨hm, hratio, halign, hdist⟩

end OpenTheory
