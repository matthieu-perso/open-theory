import OpenTheory.Generalization.TrainedRidge

open OpenTheory

-- B = 0 and λ = 1 give smoothness 2, step 1/4, and contraction 3/4.
example : Dataset.uniformStep 0 1 = 1 / 4 := by
  unfold Dataset.uniformStep Dataset.uniformSmoothness
  norm_num

example : Dataset.uniformFactor 0 1 = 3 / 4 := by
  unfold Dataset.uniformFactor Dataset.uniformSmoothness
  norm_num

-- Moving the grid count or the trained-iterate statement must break these checks.
#check BallCover.euclideanBall_net
#check TrainedRidge.trained_ridge_tail
#check Ridge.minimizer_grad_eq_zero
