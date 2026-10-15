import OpenTheory.Architectures.ConvolutionalGeneralization

noncomputable section

open MeasureTheory StatLean.StatisticalLearning OpenTheory OpenTheory.ConvolutionalGeneralization
open scoped BigOperators

-- Family 1: Patch-Based / Cyclic Convolutional Layer Semantics & Exact Translation Equivariance
#check @spatialShift
#check @spatialShift_apply
#check @spatialShift_zero
#check @spatialShift_add
#check @extractPatch
#check @cyclicPatch
#check @cyclicPatch_apply
#check @cyclicPatch_shift
#check @Activation
#check @ConvNet
#check @convPreactivation
#check @convFeatureMap
#check @convFeatureMap_eq_sigma_pre
#check @convForward
#check @convPreactivation_shift_equivariant
#check @convFeatureMap_shift_equivariant
#check @convFeatureMap_shift_equivariant_fun
#check @convForward_shift_invariant

-- Family 2: Receptive-Field Locality & Parameter Perturbation Lipschitz Bound
#check @patchEnergy
#check @patchEnergy_eq_inputEnergy_patch
#check @patchEnergy_nonneg
#check @patchEnergy_le_inputEnergy
#check @flattenConv
#check @unflattenConv
#check @unflattenConv_flattenConv
#check @flattenConv_unflattenConv
#check @unflattenConv_sub
#check @convAtDisplacement
#check @convAtDisplacement_zero
#check @convAtDisplacement_add
#check @convAtDisplacement_sub_cancel
#check @convPreShift
#check @convPreactivation_displacement
#check @convPreShift_unflattenConv_le_norm
#check @convPosition_change_le
#check @conv_param_diff_le

-- Family 3: High-Probability Out-of-Sample Generalization Bound for CNNs
#check @convParamLoss
#check @convLossLipConst
#check @convLossLipConst_eq_paramLossLipConst
#check @convLossLipConst_nonneg
#check @convParamLoss_lipschitz
#check @convBall_loss_cover
#check @conv_uniform_deviation_tail
#check @conv_approx_erm_excess_risk_tail

-- Family 4: Strict Architectural Generalization & Sample Complexity Separation (CNN vs. Dense MLP)
#check @cnn_lip_le_mlp_lip
#check @cnn_lip_lt_mlp_lip
#check @coverAxis_mono
#check @cnn_covering_strictly_smaller_than_mlp
#check @archComplexity
#check @archComplexity_eq_log_covering
#check @cnn_vs_mlp_sample_complexity_ratio

-- Smoke test 1: Exact translation invariance of global-pooled CNN output on Fin 8 with kernel k = 3.
example (net : ConvNet 3 4) (embed : Fin 3 → Fin 8) (s : Fin 8) (x : Fin 8 → ℝ) :
    convForward net embed (spatialShift s x) = convForward net embed x :=
  convForward_shift_invariant net embed s x

-- Smoke test 2: Zero local patch energy (E_patch = 0) implies zero parameter-perturbation output change.
example {d k c : ℕ} [NeZero d] (net : ConvNet k c) (embed : Fin k → Fin d)
    (x : Fin d → ℝ) (u v : Parameters (c * k)) {A L : ℝ}
    (hA : 0 ≤ A) (hL : 0 ≤ L)
    (hα : ∀ ch, |net.α ch| ≤ A) (hσ : ∀ s, |deriv net.act.σ s| ≤ L)
    (hpatch : ∀ p : Fin d, patchEnergy embed x p ≤ 0) :
    convForward (convAtDisplacement net (unflattenConv u)) embed x =
      convForward (convAtDisplacement net (unflattenConv v)) embed x := by
  have h := conv_param_diff_le net embed x u v hA hL le_rfl hα hσ hpatch
  have hzero : |convForward (convAtDisplacement net (unflattenConv u)) embed x -
      convForward (convAtDisplacement net (unflattenConv v)) embed x| ≤ 0 := by
    simpa using h
  exact sub_eq_zero.mp (abs_nonpos_iff.mp hzero)

-- Smoke test 3: Concrete strict covering number separation for c = 2, k = 3, d = 8, rad = 1, δ = 1.
example :
    (BallCover.coverAxis 1 1 (2 * 3) + 1) ^ (2 * 3) <
      (BallCover.coverAxis 1 1 (2 * 8) + 1) ^ (2 * 8) :=
  cnn_covering_strictly_smaller_than_mlp (c := 2) (k := 3) (d := 8)
    (by decide) (by decide) (by decide) (by norm_num) (by norm_num)

-- Smoke test 4: Concrete sample complexity ratio for k = 3, d = 12 is 3 / 12 < 1.
example (c : ℕ) (hc : 0 < c) :
    archComplexity (c * 3) 4 / archComplexity (c * 12) 4 = (3 : ℝ) / (12 : ℝ) ∧
      (3 : ℝ) / (12 : ℝ) < 1 :=
  cnn_vs_mlp_sample_complexity_ratio hc (by decide) (by decide) (by norm_num)
