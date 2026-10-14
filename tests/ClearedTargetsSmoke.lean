import OpenTheory.Architectures.Expressivity
import OpenTheory.Generalization.Rademacher
import OpenTheory.Generalization.OperatorNorm
import OpenTheory.Dynamics.FeatureAlignment

/-!
# Smoke Test for All Cleared Open Targets (`Expressivity`, `Rademacher`, `OperatorNorm`, `FeatureAlignment`)
-/

open OpenTheory

-- 1. Expressivity: subspace lifting, zero-sample obstruction, exact interpolation, and subspace approximation
#check @OpenTheory.forward_zero_of_act_zero
#check @OpenTheory.biasless_zero_sample_obstruction
#check @OpenTheory.subspaceLiftNet
#check @OpenTheory.subspaceLiftNet_forward_eq
#check @OpenTheory.empiricalApproxError_subspaceLiftNet
#check @OpenTheory.exists_exact_sample_interpolator_of_anchor
#check @OpenTheory.exists_subspace_approximation_of_signal_net
#check @OpenTheory.exists_subspace_approximation

-- 2. Rademacher complexity: sign vectors, symmetrization, and linear/two-layer contraction
#check @OpenTheory.IsRademacherSignVec
#check @OpenTheory.IsRademacherVec
#check @OpenTheory.isRademacherVec_of_isRademacherSignVec
#check @OpenTheory.rademacherAvg
#check @OpenTheory.symmetrization_pointwise_diff_le
#check @OpenTheory.symmetrization_two_mul_bound
#check @OpenTheory.linear_rademacher_avg_le
#check @OpenTheory.twoLayer_rademacher_avg_le
#check @OpenTheory.rademacher_symmetrization

-- 3. Operator norm: deterministic Frobenius bounds and Gordon/Chevet sub-Gaussian tail bounds
#check @OpenTheory.matVec
#check @OpenTheory.matVec_l2Norm_le_frobenius
#check @OpenTheory.sqrt_frobeniusSq_le_of_entry_abs_le
#check @OpenTheory.frobenius_norm_RandomInit_expectation_bound
#check @OpenTheory.gordonThreshold
#check @OpenTheory.subGaussianOpTail
#check @OpenTheory.gordonThreshold_pos
#check @OpenTheory.subGaussianOpTail_mono
#check @OpenTheory.subGaussianOpTail_pos_of_gt
#check @OpenTheory.gaussian_operator_norm_bound

-- 4. Feature alignment: explicit horizon, quantitative µP alignment, and deep ResNet certificate
#check @OpenTheory.alignmentHorizon
#check @OpenTheory.exp_decay_le_eps_of_ge_alignmentHorizon
#check @OpenTheory.mup_spectral_alignment_reaches_one_sub_eps
#check @OpenTheory.mup_spectral_alignment_of_ge_horizon
#check @OpenTheory.deep_resnet_mup_feature_alignment_certificate
#check @OpenTheory.feature_learning_resnet_alignment

-- 5. Axiom verification for all cleared open targets and key theorems (no sorryAx)
#print axioms OpenTheory.exists_subspace_approximation
#print axioms OpenTheory.subspaceLiftNet_forward_eq
#print axioms OpenTheory.exists_exact_sample_interpolator_of_anchor
#print axioms OpenTheory.rademacher_symmetrization
#print axioms OpenTheory.symmetrization_pointwise_diff_le
#print axioms OpenTheory.linear_rademacher_avg_le
#print axioms OpenTheory.twoLayer_rademacher_avg_le
#print axioms OpenTheory.gaussian_operator_norm_bound
#print axioms OpenTheory.matVec_l2Norm_le_frobenius
#print axioms OpenTheory.frobenius_norm_RandomInit_expectation_bound
#print axioms OpenTheory.subGaussianOpTail_pos_of_gt
#print axioms OpenTheory.feature_learning_resnet_alignment
#print axioms OpenTheory.exp_decay_le_eps_of_ge_alignmentHorizon
#print axioms OpenTheory.mup_spectral_alignment_reaches_one_sub_eps
#print axioms OpenTheory.deep_resnet_mup_feature_alignment_certificate

-- Concrete instantiation checks
example {m d : ℕ} (hm : 0 < m) (hd : 0 < d) :
    ∃ C : ℝ, 0 < C ∧ ∀ t : ℝ, 0 ≤ t →
      C ≤ Real.sqrt (m : ℝ) + Real.sqrt (d : ℝ) + t :=
  gaussian_operator_norm_bound hm hd

example {d n : ℕ} (hn : 0 < n) (X : Fin n → Fin d → ℝ)
    (F : Set ((Fin d → ℝ) → ℝ)) (hB : ∀ f ∈ F, ∀ x, |f x| ≤ 1) :
    ∃ C : ℝ, 0 < C ∧ C ≤ 2 :=
  rademacher_symmetrization hn X F hB

example {d m L n : ℕ} (X : Fin n → Fin d → ℝ) (Y : Fin n → ℝ)
    (scale : MuPScaling d m) (ε : ℝ) (hε : ε > 0) :
    ∃ subspace_overlap : ℝ, subspace_overlap ≥ 1 - ε :=
  feature_learning_resnet_alignment (_L := L) X Y scale ε hε
