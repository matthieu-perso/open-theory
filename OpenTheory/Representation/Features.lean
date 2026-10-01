import Mathlib.Algebra.BigOperators.Group.Finset
import Mathlib.Data.Matrix.Basic
import OpenTheory.Architectures.Basic
import OpenTheory.Data.Sample

namespace OpenTheory

/-- Hidden representation of a two-layer net on one input: `σ(W₁ x) ∈ ℝ^m`. -/
def hiddenFeatures {d m : ℕ} (net : TwoLayerNet d m) (x : Fin d → ℝ) : Fin m → ℝ :=
  fun i => net.act.σ (∑ j : Fin d, net.W₁ i j * x j)

/-- Feature matrix of a sample. Row `s` is `hiddenFeatures` on `X s`.
    Columns are feature coordinates. -/
def featureMatrix {d m n : ℕ} (net : TwoLayerNet d m) (sample : Sample d n) :
    Matrix (Fin n) (Fin m) ℝ :=
  fun s i => hiddenFeatures net (sample.X s) i

lemma forward_eq_readout {d m : ℕ} (net : TwoLayerNet d m) (x : Fin d → ℝ) :
    net.forward x = ∑ k : Fin m, net.W₂ 0 k * hiddenFeatures net x k := by
  unfold TwoLayerNet.forward hiddenFeatures
  rfl

end OpenTheory
