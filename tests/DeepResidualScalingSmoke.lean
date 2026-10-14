import OpenTheory.Dynamics.Contraction
import OpenTheory.Dynamics.DeepResidualScaling

open OpenTheory
open OpenTheory.DeepResidualScaling

-- Section 1: l2Norm algebra and triangle inequalities
#check DeepResidualScaling.l2Norm_sq
#check DeepResidualScaling.l2Norm_nonneg
#check DeepResidualScaling.l2Norm_zero
#check DeepResidualScaling.l2Norm_smul
#check DeepResidualScaling.l2Norm_neg
#check DeepResidualScaling.l2Norm_sub_comm
#check DeepResidualScaling.dotProduct_le_l2Norm_mul_l2Norm
#check DeepResidualScaling.l2Norm_add_le
#check DeepResidualScaling.l2Norm_sub_le
#check DeepResidualScaling.l2Norm_add_ge

-- Section 2: Residual branch and step Lipschitz bounds
#check DeepResidualScaling.branchOutput
#check DeepResidualScaling.residualStep_eq_add_branchOutput
#check DeepResidualScaling.branchOutput_l2Norm_le
#check DeepResidualScaling.residualStep_l2Norm_le
#check DeepResidualScaling.residualStep_l2Norm_le_of_bound
#check DeepResidualScaling.residualStep_sqrt_depth_scaling

-- Section 3: Deep L-layer residual network and depth-independent exp(c) stability
#check DeepResidualScaling.deepResidual
#check DeepResidualScaling.deepResidual_l2Norm_le
#check DeepResidualScaling.one_add_div_pow_le_exp
#check DeepResidualScaling.deepResidualChain_depth_independent_exp_bound

-- Section 4: Signal propagation and representation norm window
#check DeepResidualScaling.residualStep_zero_of_branchOutput_zero
#check DeepResidualScaling.deepResidual_zero_input
#check DeepResidualScaling.residualStep_l2Norm_ge
#check DeepResidualScaling.deepResidual_norm_window

-- Section 5: Contraction.lean exports
#check OpenTheory.resnet_step_contraction_le
#check OpenTheory.resnet_representation_contraction_lipschitz
#check OpenTheory.resnet_representation_contraction

-- Smoke test 1: With zero branch scaling (scale = 0), an L-layer residual network is an exact isometry.
example {d : ℕ} (L : ℕ) (branches : ℕ → ResidualBranch d) (x : Fin d → ℝ) :
    deepResidual L branches 0 x = x := by
  induction L with
  | zero => rfl
  | succ t ih =>
    simp only [deepResidual_succ, ih]
    ext i
    simp [residualStep]

-- Smoke test 2: Depth-independent exp(c) Lipschitz bound holds at arbitrary depth L ≥ 1.
example {d L : ℕ} (hL : 0 < L) (branch : ResidualBranch d) {L_σ c : ℝ}
    (hL_σ : 0 ≤ L_σ) (hc : 0 ≤ c)
    (hLip : ∀ a b : ℝ, |branch.act.σ a - branch.act.σ b| ≤ L_σ * |a - b|)
    (hW : L_σ * Real.sqrt (frobeniusSq branch.W) ≤ c)
    (x y : Fin d → ℝ) :
    l2Norm (fun i =>
      deepResidual L (fun _ => branch) (1 / (L : ℝ)) x i -
        deepResidual L (fun _ => branch) (1 / (L : ℝ)) y i) ≤
      Real.exp c * l2Norm (fun j => x j - y j) :=
  deepResidualChain_depth_independent_exp_bound hL (fun _ => branch) (fun _ => L_σ) hc
    (fun _ _ => hL_σ) (fun _ _ => hLip) (fun _ _ => hW) x y

-- Smoke test 3: Axiom audit on all primary theorems (no sorryAx).
#print axioms DeepResidualScaling.branchOutput_l2Norm_le
#print axioms DeepResidualScaling.residualStep_l2Norm_le
#print axioms DeepResidualScaling.residualStep_sqrt_depth_scaling
#print axioms DeepResidualScaling.deepResidualChain_depth_independent_exp_bound
#print axioms DeepResidualScaling.deepResidual_norm_window
#print axioms OpenTheory.resnet_step_contraction_le
#print axioms OpenTheory.resnet_representation_contraction
