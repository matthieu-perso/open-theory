import Mathlib.Analysis.Calculus.Deriv.Basic
import OpenTheory.Optimization.Loss

namespace OpenTheory

/-- A differentiable trajectory of networks whose empirical risk has derivative `derivL`.
    `descends` says that derivative is non-positive.
    This is the descent consequence of gradient flow. The identification of `derivL`
    with `-learningRate • ∇loss` is not yet in the library. -/
structure DescentPath {d m n : ℕ} (sample : Sample d n) (path : ℝ → TwoLayerNet d m) where
  derivL : ℝ → ℝ
  hasDeriv : ∀ t, HasDerivAt (fun s => EmpiricalRisk sample (path s)) (derivL t) t
  descends : ∀ t, derivL t ≤ 0

end OpenTheory
