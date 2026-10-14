import OpenTheory.Optimization.FiniteRidge

open OpenTheory

-- With one observation x=1, y=2 and lambda=1, the ridge objective at theta=1 is 1.
example : Ridge.objective (ContinuousLinearMap.id ℝ ℝ) (2 : ℝ) 1 1 1 = 1 := by
  norm_num [Ridge.objective]

example : Ridge.grad (ContinuousLinearMap.id ℝ ℝ) (2 : ℝ) 1 1 1 = 0 := by
  norm_num [Ridge.grad, ContinuousLinearMap.adjoint_id]

-- Catches a missing factor of one half or an invalid distance-to-objective conversion.
example (z : ℝ) :
    Ridge.objective (ContinuousLinearMap.id ℝ ℝ) 2 1 1 z -
      Ridge.objective (ContinuousLinearMap.id ℝ ℝ) 2 1 1 1 ≤
        (1 / 2 * ‖ContinuousLinearMap.id ℝ ℝ‖ ^ 2 + 1 / 2) * ‖z - 1‖ ^ 2 := by
  have hx : Ridge.grad (ContinuousLinearMap.id ℝ ℝ) 2 1 1 1 = 0 := by
    norm_num [Ridge.grad, ContinuousLinearMap.adjoint_id]
  exact Ridge.objective_gap_le _ _ _ _ (by norm_num) hx z

example (η : ℝ) (hη : 0 < η)
    (hstep : η * (Ridge.smoothness (ContinuousLinearMap.id ℝ ℝ) 1 1 : ℝ) ^ 2 < 2) :
    ∃ xStar : ℝ, (∀ z, Ridge.objective (ContinuousLinearMap.id ℝ ℝ) 2 1 1 xStar ≤
      Ridge.objective (ContinuousLinearMap.id ℝ ℝ) 2 1 1 z) ∧
      ∀ x k, Ridge.objective (ContinuousLinearMap.id ℝ ℝ) 2 1 1
          ((Optim.GD.step η (gradient (Ridge.objective
            (ContinuousLinearMap.id ℝ ℝ) 2 1 1)))^[k] x) -
          Ridge.objective (ContinuousLinearMap.id ℝ ℝ) 2 1 1 xStar ≤
        (1 / 2 * ‖ContinuousLinearMap.id ℝ ℝ‖ ^ 2 + 1 / 2) *
          (Optim.GD.q η 1 (Ridge.smoothness (ContinuousLinearMap.id ℝ ℝ) 1 1)) ^ k *
          ‖x - xStar‖ ^ 2 := by
  have hstep' : η * (Ridge.smoothness (ContinuousLinearMap.id ℝ ℝ) 1 1 : ℝ) ^ 2 <
      2 * 1 := by simpa using hstep
  exact Ridge.training_objective_gap (ContinuousLinearMap.id ℝ ℝ) 2 1 1
    (by norm_num) (by norm_num) η hη hstep'
