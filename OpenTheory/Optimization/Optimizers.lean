import NN.MLTheory.Optimization.FirstOrder
import NN.MLTheory.Optimization.StronglyConvexGD
import NN.Proofs.Tensor.Euclidean
import NN.Proofs.Analysis.Lipschitz.Norm

/-! Reuse the upstream state machines directly: `Optim.SGD`, `MomentumSGD`,
`Adam`, and `AdamW`. Gradient estimation is a separate caller-supplied operation.
These equations do not assert unbiasedness or stochastic convergence. -/
noncomputable section
open TorchLean Spec

namespace OpenTheory.Optimizer

/-- The tensor SGD update is the same update used by the Hilbert-space GD theorem. -/
theorem sgd_eq_step {s : Shape} (state : Optim.SGD.State ℝ s)
    (parameters gradients : Tensor ℝ s) :
    (Optim.SGD.update state parameters gradients).parameters =
      Optim.GD.step state.learningRate (fun _ => gradients) parameters := by
  simp [Optim.SGD.update, Optim.GD.step, TorchLean.Tensor.subSpec_eq_sub,
    Proofs.scaleSpec_eq_smul]

/-- Bias correction uses the next step index, starting with one after initialization. -/
theorem adam_first_step {s : Shape} (lr b₁ b₂ ε : ℝ) (p g : Tensor ℝ s) :
    (Optim.Adam.update (Optim.Adam.init lr b₁ b₂ ε p) p g).optimizerState.stepCount = 1 := by
  simp [Optim.Adam.init]

end OpenTheory.Optimizer
