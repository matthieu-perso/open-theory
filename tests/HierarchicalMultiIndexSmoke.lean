import OpenTheory.Dynamics.HierarchicalMultiIndex

open OpenTheory
open OpenTheory.MultiStepSpectralBias
open OpenTheory.HierarchicalMultiIndex

-- 1. Section 1: Information Exponent s and Vanishing Low-Order Correlations
#check HierarchicalMultiIndex.MultiIndexTarget
#check HierarchicalMultiIndex.tensorAlignmentDrift
#check HierarchicalMultiIndex.tensorAlignmentDrift_eq_zero_of_lt
#check HierarchicalMultiIndex.linear_gram_drift_zero_of_exponent_ge_two
#check HierarchicalMultiIndex.tensorAlignmentDrift_at_exponent
#check HierarchicalMultiIndex.multiIndexStep
#check HierarchicalMultiIndex.multiIndexStep_eq_add_drift
#check HierarchicalMultiIndex.multiIndexTrajectory
#check HierarchicalMultiIndex.multiIndexTrajectory_zero
#check HierarchicalMultiIndex.multiIndexTrajectory_succ
#check HierarchicalMultiIndex.multiIndexTrajectory_nonneg
#check HierarchicalMultiIndex.multiIndexTrajectory_pos
#check HierarchicalMultiIndex.multiIndexTrajectory_lt_succ
#check HierarchicalMultiIndex.multiIndexTrajectory_strictMono
#check HierarchicalMultiIndex.multiIndexTrajectory_mono

-- 2. Section 2: Information Exponent Dichotomy (s = 1 Linear Fast Escape vs. s >= 2 Saddle Plateau & Grokking)
#check HierarchicalMultiIndex.multiIndexTrajectory_exponent_one_eq
#check HierarchicalMultiIndex.multiIndexTrajectory_exponent_one_eq_modeAmpDisc
#check HierarchicalMultiIndex.nonlinear_drift_lt_linear_drift_on_plateau
#check HierarchicalMultiIndex.grokkingEscapeSteps
#check HierarchicalMultiIndex.grokkingEscapeSteps_pos
#check HierarchicalMultiIndex.grokking_escape_time_strict_mono_in_exponent
#check HierarchicalMultiIndex.grokking_escape_time_ratio_eq
#check HierarchicalMultiIndex.multiIndex_slow_plateau_upper_bound
#check HierarchicalMultiIndex.multiIndexStep_ge_of_threshold
#check HierarchicalMultiIndex.multiIndex_post_threshold_exponential_growth

-- 3. Section 3: Compositional Hierarchy vs. Shallow Representation Bottleneck
#check HierarchicalMultiIndex.shallowSampleComplexity
#check HierarchicalMultiIndex.hierarchicalSampleComplexity
#check HierarchicalMultiIndex.hierarchicalEscapeSteps
#check HierarchicalMultiIndex.hierarchical_vs_shallow_sample_complexity_separation
#check HierarchicalMultiIndex.binary_hierarchy_sample_complexity_lt_shallow
#check HierarchicalMultiIndex.hierarchical_stagewise_escape_time_lt_shallow

-- 4. Section 4: Practical Prescription: Depth & Warmup Schedule for High-Information-Exponent Data
#check HierarchicalMultiIndex.optimal_depth_and_horizon_for_multi_index

-- Concrete numerical verifications
section ConcreteExamples

-- Example 1: Comparing grokking escape steps across s = 1, s = 2, and s = 4 at α₀ = 1/10, η = 1, c_s = 1
-- s = 1 → 1 step scale; s = 2 → 10 step scale; s = 4 → 1000 step scale
example :
    grokkingEscapeSteps 1 1 (1 / 10) 1 = 1 ∧
    grokkingEscapeSteps 1 1 (1 / 10) 2 = 10 ∧
    grokkingEscapeSteps 1 1 (1 / 10) 4 = 1000 ∧
    grokkingEscapeSteps 1 1 (1 / 10) 1 < grokkingEscapeSteps 1 1 (1 / 10) 2 ∧
    grokkingEscapeSteps 1 1 (1 / 10) 2 < grokkingEscapeSteps 1 1 (1 / 10) 4 := by
  unfold grokkingEscapeSteps
  norm_num

-- Example 2: Vanishing first-order Gram drift for s = 2 and s = 4 vs. non-zero drift at m = s
example :
    tensorAlignmentDrift 2 1 1 (1 / 10) = 0 ∧
    tensorAlignmentDrift 4 1 1 (1 / 10) = 0 ∧
    tensorAlignmentDrift 2 2 1 (1 / 10) = 1 / 10 ∧
    tensorAlignmentDrift 4 4 1 (1 / 10) = 1 / 1000 := by
  unfold tensorAlignmentDrift
  norm_num

-- Example 3: Hierarchical L = 2, s_local = 2 vs. shallow s_total = 4 on ambient dimension d = 10
-- Sample complexity: hierarchical = 2 * 10^2 = 200 vs. shallow = 10^4 = 10000
-- Escape time at α₀ = 1/10: hierarchical = 2 * 10 = 20 vs. shallow = 1000
example :
    hierarchicalSampleComplexity 2 10 2 = 200 ∧
    shallowSampleComplexity 10 4 = 10000 ∧
    hierarchicalSampleComplexity 2 10 2 < shallowSampleComplexity 10 4 ∧
    hierarchicalEscapeSteps 2 1 1 (1 / 10) 2 = 20 ∧
    grokkingEscapeSteps 1 1 (1 / 10) 4 = 1000 ∧
    hierarchicalEscapeSteps 2 1 1 (1 / 10) 2 < grokkingEscapeSteps 1 1 (1 / 10) 4 := by
  unfold hierarchicalSampleComplexity shallowSampleComplexity
    hierarchicalEscapeSteps grokkingEscapeSteps
  norm_num

-- Example 4: 1-step trajectory comparison at α₀ = 1/10, η = 1, c_s = 1 across s = 1, 2, 4
example :
    multiIndexTrajectory 1 1 1 (1 / 10) 1 = 2 / 10 ∧
    multiIndexTrajectory 1 1 2 (1 / 10) 1 = 11 / 100 ∧
    multiIndexTrajectory 1 1 4 (1 / 10) 1 = 1001 / 10000 := by
  unfold multiIndexTrajectory multiIndexStep
  norm_num

end ConcreteExamples
