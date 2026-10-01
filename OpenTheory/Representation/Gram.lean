import Mathlib.Algebra.BigOperators.Group.Finset
import Mathlib.Data.Matrix.Basic
import OpenTheory.Data.Sample

namespace OpenTheory

/-- Empirical un-normalized Gram matrix of the inputs, `(Xᵀ X)_{ij} = ∑_s X_si X_sj`. -/
def dataGram {d n : ℕ} (sample : Sample d n) : Matrix (Fin d) (Fin d) ℝ :=
  fun i j => ∑ s : Fin n, sample.X s i * sample.X s j

end OpenTheory
