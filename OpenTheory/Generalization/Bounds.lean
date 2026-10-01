import Mathlib.Algebra.BigOperators.Group.Finset
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import OpenTheory.Architectures.Basic

namespace OpenTheory

/-- Euclidean inner product on `ℝ^d` indexed by `Fin d`. -/
def dotProduct {d : ℕ} (u v : Fin d → ℝ) : ℝ := ∑ j : Fin d, u j * v j

/-- Euclidean norm. -/
noncomputable def l2Norm {d : ℕ} (u : Fin d → ℝ) : ℝ :=
  Real.sqrt (∑ j : Fin d, u j ^ 2)

/-- One hidden unit is Lipschitz in the input, with constant `L ‖w‖₂`. -/
theorem layer_lipschitz (act : ActivationFunction) {d : ℕ} (w x y : Fin d → ℝ) :
    ∃ L : ℝ,
      |act.σ (dotProduct w x) - act.σ (dotProduct w y)| ≤
        L * l2Norm w * l2Norm (fun j => x j - y j) := by
  sorry

/-- The scalar network `forward` is Lipschitz in the input. -/
theorem forward_lipschitz {d m : ℕ} (net : TwoLayerNet d m) (x y : Fin d → ℝ) :
    ∃ C : ℝ, |net.forward x - net.forward y| ≤ C * l2Norm (fun j => x j - y j) := by
  sorry

end OpenTheory
