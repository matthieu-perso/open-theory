import OpenTheory.Optimization.Optimizers

open TorchLean OpenTheory

example {s : Spec.Shape} (p : Tensor ℝ s) :
    (Optim.SGD.update (Optim.SGD.init 0 p) p p).parameters = p := by
  simp [Optimizer.sgd_eq_step, Optim.GD.step, Optim.SGD.init]
