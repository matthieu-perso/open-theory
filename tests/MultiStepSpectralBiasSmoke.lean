import OpenTheory.Dynamics.MultiStepSpectralBias
import OpenTheory.Dynamics.SpectralBias

open OpenTheory
open OpenTheory.MultiStepSpectralBias
open OpenTheory.FeatureLearningAdvantage

-- 1. Continuous-Time & Discrete-Time Eigenmode Dynamics (`MultiStepSpectralBias.lean`)
#check OpenTheory.MultiStepSpectralBias.mupEffectiveFeatureLR
#check OpenTheory.MultiStepSpectralBias.modeAmpCont
#check OpenTheory.MultiStepSpectralBias.modeAmpDisc
#check OpenTheory.MultiStepSpectralBias.modeAmpCont_zero
#check OpenTheory.MultiStepSpectralBias.modeAmpDisc_zero
#check OpenTheory.MultiStepSpectralBias.hasDerivAt_modeAmpCont
#check OpenTheory.MultiStepSpectralBias.hasDerivAt_modeAmpCont'
#check OpenTheory.MultiStepSpectralBias.modeAmpDisc_step
#check OpenTheory.MultiStepSpectralBias.modeAmpCont_nonneg
#check OpenTheory.MultiStepSpectralBias.modeAmpCont_pos
#check OpenTheory.MultiStepSpectralBias.modeAmpCont_mono_time
#check OpenTheory.MultiStepSpectralBias.modeAmpCont_mono_eigenvalue
#check OpenTheory.MultiStepSpectralBias.modeAmpDisc_mono_eigenvalue
#check OpenTheory.MultiStepSpectralBias.modeAmpDisc_lt_of_pos

-- 2. Signal-to-Bulk Amplification under `SpectralGap` (`MultiStepSpectralBias.lean`)
#check OpenTheory.MultiStepSpectralBias.leading_pos_of_spectralGap
#check OpenTheory.MultiStepSpectralBias.signal_to_bulk_ratio_cont
#check OpenTheory.MultiStepSpectralBias.bulk_div_signal_cont_le
#check OpenTheory.MultiStepSpectralBias.disc_growth_ratio_bounds
#check OpenTheory.MultiStepSpectralBias.signal_to_bulk_ratio_disc
#check OpenTheory.MultiStepSpectralBias.bulk_div_signal_disc_le

-- 3. Subspace Alignment Convergence to `1` (`MultiStepSpectralBias.lean`)
#check OpenTheory.MultiStepSpectralBias.alignment_lower_bound_of_signal_bulk
#check OpenTheory.MultiStepSpectralBias.spectral_bias_alignment_cont_ge
#check OpenTheory.MultiStepSpectralBias.spectral_bias_alignment_disc_ge
#check OpenTheory.MultiStepSpectralBias.spiked_model_alignment_cont_ge

-- 4. Multi-Step Feature Learning Risk Advantage (`MultiStepSpectralBias.lean`)
#check OpenTheory.MultiStepSpectralBias.multistep_kernel_strict_growth
#check OpenTheory.MultiStepSpectralBias.multistep_kernel_growth_improves_ridge_mse

-- 5. Verified LEM-2.1 & Companion Spectral Bias Theorems (`SpectralBias.lean`)
#check OpenTheory.spectral_bias_mup_alignment
#check OpenTheory.spectral_bias_mup_gap_alignment_cont
#check OpenTheory.spectral_bias_mup_gap_alignment_disc

-- Concrete verification of `modeAmpDisc_step`:
example : modeAmpDisc (1 / 2) 2 1 3 = (1 + (1 / 2) * 2) * modeAmpDisc (1 / 2) 2 0 3 := by
  exact modeAmpDisc_step (1 / 2) 2 0 3

-- Concrete verification of `disc_growth_ratio_bounds`:
noncomputable def sampleSpectralGap : SpectralGap where
  leading := 3
  bulk := 1
  gap := 2
  gap_pos := by norm_num
  bulk_nonneg := by norm_num
  separated := by norm_num

example :
    1 + (1 * sampleSpectralGap.gap) / (1 + 1 * sampleSpectralGap.bulk) ≤
      (1 + 1 * sampleSpectralGap.leading) / (1 + 1 * sampleSpectralGap.bulk) ∧
    1 < (1 + 1 * sampleSpectralGap.leading) / (1 + 1 * sampleSpectralGap.bulk) := by
  exact disc_growth_ratio_bounds sampleSpectralGap (by norm_num)

-- Concrete verification of `multistep_kernel_growth_improves_ridge_mse`:
-- With sampleSpectralGap (leading = 3), η = 1/3, T = 1, K₀ = 1, K_T = 4:
-- (1 + η * leading)^(2*T) * K₀ = (1 + 1)^2 * 1 = 4 ≤ K_T.
-- With lam = 1, Δy = 3, σ = 1: K₀ * σ² = 1 < 9 = lam * Δy² and K_T * σ² = 4 ≤ 9.
example :
    (1 : ℝ) < 4 ∧
    biasSq 4 1 3 < biasSq 1 1 3 ∧
    ridgeExpectedMSE 4 1 3 1 < ridgeExpectedMSE 1 1 3 1 := by
  exact multistep_kernel_growth_improves_ridge_mse
    sampleSpectralGap (η := (1 : ℝ) / 3) (T := 1)
    (by norm_num) (by decide) (by norm_num)
    (by simp [sampleSpectralGap]; norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num)
