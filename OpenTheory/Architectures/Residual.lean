import Mathlib.Algebra.BigOperators.Group.Finset
import Mathlib.Data.Matrix.Basic
import OpenTheory.Architectures.Basic

namespace OpenTheory

/-- One residual branch: a square weight matrix and an admissible activation. -/
structure ResidualBranch (d : ℕ) where
  W : Matrix (Fin d) (Fin d) ℝ
  act : ActivationFunction

/-- Residual step `h ↦ h + scale • σ(W h)`.
    The contraction lemma quantifies `scale` of order `1/√L`. -/
def residualStep {d : ℕ} (h : Fin d → ℝ) (branch : ResidualBranch d) (scale : ℝ) :
    Fin d → ℝ :=
  fun i => h i + scale * branch.act.σ (∑ j : Fin d, branch.W i j * h j)

end OpenTheory
