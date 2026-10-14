import OpenTheory.Generalization.Bounds
import OpenTheory.Generalization.TrajectoryGeneralization

open MeasureTheory StatLean.StatisticalLearning OpenTheory

-- Part A: Input Lipschitz certificates in OpenTheory/Generalization/Bounds.lean
#check OpenTheory.l2Norm_nonneg
#check OpenTheory.l2Norm_sq
#check OpenTheory.l2Norm_eq_euclideanNorm
#check OpenTheory.dotProduct_sub_right
#check OpenTheory.abs_dotProduct_le
#check OpenTheory.abs_dotProduct_sub_le
#check OpenTheory.lipschitz_const_nonneg
#check OpenTheory.layer_lipschitz_of_lip
#check OpenTheory.layer_lipschitz
#check OpenTheory.forward_lipschitz_explicit
#check OpenTheory.forward_lipschitz

-- Part B: Trajectory-localized out-of-sample generalization bounds
#check TrajectoryGeneralization.unflatten_sub
#check TrajectoryGeneralization.atDisplacement_add
#check TrajectoryGeneralization.atDisplacement_sub_cancel
#check TrajectoryGeneralization.forward_param_diff_le
#check TrajectoryGeneralization.paramLoss
#check TrajectoryGeneralization.paramLossLipConst
#check TrajectoryGeneralization.paramLossLipConst_nonneg
#check TrajectoryGeneralization.paramLoss_lipschitz
#check TrajectoryGeneralization.clippedSqLoss
#check TrajectoryGeneralization.clippedSqLoss_mem_Icc
#check TrajectoryGeneralization.abs_min_one_sq_sub_le
#check TrajectoryGeneralization.clippedSqLoss_lipschitz
#check TrajectoryGeneralization.clippedSqLoss_param_lipschitz
#check TrajectoryGeneralization.trajectoryBall
#check TrajectoryGeneralization.mem_trajectoryBall
#check TrajectoryGeneralization.trajectoryBall_loss_cover
#check TrajectoryGeneralization.trajectory_uniform_deviation_tail
#check TrajectoryGeneralization.contractive_trajectory_in_ball
#check TrajectoryGeneralization.contractive_trajectory_generalization_tail
#check TrajectoryGeneralization.approx_erm_trajectory_excess_risk_tail

-- Smoke test 1: Identical inputs give zero difference under forward_lipschitz_explicit.
example {d m : ℕ} (net : TwoLayerNet d m) (x : Fin d → ℝ)
    {L : ℝ} (hL : ∀ a b : ℝ, |net.act.σ a - net.act.σ b| ≤ L * |a - b|) :
    |net.forward x - net.forward x| ≤
      (∑ i : Fin m, |net.W₂ 0 i| * L * l2Norm (net.W₁ i)) *
        l2Norm (fun j => x j - x j) :=
  forward_lipschitz_explicit net x x hL

-- Smoke test 2: Zero input energy (E = 0) implies zero parameter-perturbation output change.
example {d m : ℕ} (net : TwoLayerNet d m) (x : Fin d → ℝ)
    (u v : Parameters (m * d)) {A L : ℝ}
    (hA : 0 ≤ A) (hL : 0 ≤ L)
    (hW : ∀ i, |net.W₂ 0 i| ≤ A) (hσ : ∀ s, |deriv net.act.σ s| ≤ L)
    (hx : FeatureStep.inputEnergy x ≤ 0) :
    (MLPTrajectory.atDisplacement net (MLPBatchTrajectory.unflatten u)).forward x =
      (MLPTrajectory.atDisplacement net (MLPBatchTrajectory.unflatten v)).forward x := by
  have h := TrajectoryGeneralization.forward_param_diff_le net x u v hA hL le_rfl hW hσ hx
  have hzero : |(MLPTrajectory.atDisplacement net (MLPBatchTrajectory.unflatten u)).forward x -
      (MLPTrajectory.atDisplacement net (MLPBatchTrajectory.unflatten v)).forward x| ≤ 0 := by
    simpa using h
  exact sub_eq_zero.mp (abs_nonpos_iff.mp hzero)

-- Smoke test 3: Clipped squared loss with scale B = 2 is 1-Lipschitz in the prediction.
example (p₁ p₂ y : ℝ) :
    |TrajectoryGeneralization.clippedSqLoss 2 p₁ y -
      TrajectoryGeneralization.clippedSqLoss 2 p₂ y| ≤ |p₁ - p₂| := by
  have h := TrajectoryGeneralization.clippedSqLoss_lipschitz (B := 2) (by norm_num) p₁ p₂ y
  simpa using h

-- Smoke test 4: Concrete cover cardinality check at radius 0 (coverAxis 0 δ p = 1).
example (δ : ℝ) (p : ℕ) :
    BallCover.coverAxis 0 δ p = 1 := by
  simp [BallCover.coverAxis]
