import OpenTheory.Generalization.SpikedSampleComplexity

open OpenTheory
open OpenTheory.FeatureLearningAdvantage
open OpenTheory.SpikedSampleComplexity

-- 1. Spiked-Covariance Spectral Risk Decomposition
#check SpikedSampleComplexity.signalKernel
#check SpikedSampleComplexity.bulkKernel
#check SpikedSampleComplexity.signalKernel_add_bulkKernel
#check SpikedSampleComplexity.signalKernel_bulkKernel_nonneg
#check SpikedSampleComplexity.bulkKernel_eq_frobeniusSq_residual
#check SpikedSampleComplexity.signalKernel_eq_frobeniusSq_proj
#check SpikedSampleComplexity.spikedVariance
#check SpikedSampleComplexity.spikedExpectedMSE
#check SpikedSampleComplexity.spikedExpectedMSE_eq_biasSq_add_spikedVariance
#check SpikedSampleComplexity.spikedExpectedMSE_rank_one_zero_bulk
#check SpikedSampleComplexity.spikedVariance_nonneg
#check SpikedSampleComplexity.spikedExpectedMSE_nonneg
#check SpikedSampleComplexity.spikedExpectedMSE_mono_in_bulk
#check SpikedSampleComplexity.spikedExpectedMSE_strict_decrease_of_signal_and_bulk

-- 2. Effective Degrees of Freedom & Ambient Dimension Independence via µP Alignment
#check SpikedSampleComplexity.spikedEffectiveDim
#check SpikedSampleComplexity.spikedEffectiveDim_bounds
#check SpikedSampleComplexity.spikedEffectiveDim_zero_bulk_le_rank
#check SpikedSampleComplexity.unaligned_effectiveDim_scales_with_d_bulk
#check SpikedSampleComplexity.mup_aligned_effectiveDim_le_rank_add_eps

-- 3. Explicit Sample Complexity Separation Between µP Feature Learning and Frozen NTK
#check SpikedSampleComplexity.requiredSampleScale
#check SpikedSampleComplexity.bulkSampleOverhead
#check SpikedSampleComplexity.requiredSampleScale_eq_signal_add_bulk
#check SpikedSampleComplexity.requiredSampleScale_nonneg
#check SpikedSampleComplexity.spikedVariance_le_of_sample_ge
#check SpikedSampleComplexity.mup_vs_ntk_sample_complexity_separation

-- 4. End-to-End Spiked-Covariance µP Alignment + Risk Guarantee
#check SpikedSampleComplexity.mup_spiked_covariance_master_guarantee

-- Concrete numerical verification 1: Strict decrease of spikedExpectedMSE when bulk leakage drops
-- With K_sig = 2, K_bulk₁ = 0, K_bulk₀ = 1, r = 2, d_bulk = 10, lam = 1, Δy = 3, σ = 1, n = 5
example : spikedExpectedMSE 2 0 2 10 1 3 1 5 < spikedExpectedMSE 2 1 2 10 1 3 1 5 := by
  exact (spikedExpectedMSE_mono_in_bulk (by norm_num) (by norm_num) (by norm_num)).2
    (by decide) (by norm_num) (by decide) (by norm_num)

-- Concrete numerical verification 2: Joint signal amplification and bulk suppression
-- K_sig₀ = 1, K_sig₁ = 3, K_bulk₁ = 0, K_bulk₀ = 1, r = 1, d_bulk = 10, lam = 1, Δy = 2, σ = 1, n = 1
example : spikedExpectedMSE 3 0 1 10 1 2 1 1 < spikedExpectedMSE 1 1 1 10 1 2 1 1 := by
  exact spikedExpectedMSE_strict_decrease_of_signal_and_bulk
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num)

-- Concrete numerical verification 3: Strict sample complexity separation between µP and NTK
example :
    requiredSampleScale 2 0 2 100 1 1 (1 / 10) <
      requiredSampleScale 2 1 2 100 1 1 (1 / 10) ∧
    requiredSampleScale 2 0 2 100 1 1 (1 / 10) = 80 / 9 := by
  have hsep := mup_vs_ntk_sample_complexity_separation
    (K_sig := 2) (K_bulk_mup := 0) (K_bulk_ntk := 1) (r := 2) (d_bulk := 100)
    (lam := 1) (σ := 1) (ε_var := 1 / 10)
    (by norm_num) (by norm_num) (by norm_num) (by decide) (by norm_num) (by norm_num) (by norm_num)
  refine ⟨hsep.2.1, ?_⟩
  rw [hsep.2.2.1]
  norm_num
