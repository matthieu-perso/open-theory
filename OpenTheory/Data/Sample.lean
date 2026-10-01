import Mathlib.Algebra.BigOperators.Group.Finset
import Mathlib.Data.Real.Basic

namespace OpenTheory

/-- A finite labeled sample.
    Inputs are Euclidean-bounded by 1 and labels are bounded by 1.
    These are the standing hypotheses for empirical risk and generalization. -/
structure Sample (d n : ℕ) where
  X : Fin n → Fin d → ℝ
  Y : Fin n → ℝ
  n_pos : 0 < n
  input_bound : ∀ i, ∑ j : Fin d, (X i j) ^ 2 ≤ 1
  label_bound : ∀ i, |Y i| ≤ 1

end OpenTheory
