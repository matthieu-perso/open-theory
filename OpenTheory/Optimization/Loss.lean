import Mathlib.Algebra.BigOperators.Group.Finset
import OpenTheory.Architectures.Basic
import OpenTheory.Data.Sample

namespace OpenTheory

/-- Mean squared error of a two-layer net on a sample. -/
noncomputable def EmpiricalRisk {d m n : ℕ} (sample : Sample d n)
    (net : TwoLayerNet d m) : ℝ :=
  ((n : ℝ)⁻¹) * ∑ i : Fin n, (net.forward (sample.X i) - sample.Y i) ^ 2

end OpenTheory
