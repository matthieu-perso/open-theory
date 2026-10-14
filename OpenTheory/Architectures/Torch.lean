import NN.Spec.Layers.Linear
import NN.Proofs.Analysis.Lipschitz.Network
import OpenTheory.Architectures.Basic

/-! Neural semantics reuse TorchLean; metric bounds use its Euclidean tensor metric. -/
noncomputable section
open TorchLean TorchLean.Tensor Spec

namespace OpenTheory.Torch

theorem affine_coordinate {d m : ℕ} (layer : LinearSpec ℝ d m)
    (x : Tensor ℝ [d]) (i : Fin m) :
    getScalar (linearSpec layer x) i =
      (∑ j, get2 layer.weights i j * getScalar x j) + getScalar layer.bias i := by
  simp [linearSpec, Spec.getScalar_mat_vec_mul_spec, Spec.getScalar_add_spec]

def layerBound {d m : ℕ} (layer : LinearSpec ℝ d m) : NNReal :=
  ⟨Proofs.matrixFrobeniusNorm layer.weights, Real.sqrt_nonneg _⟩

theorem affine_lipschitz {d m : ℕ} (layer : LinearSpec ℝ d m) :
    LipschitzWith (layerBound layer) (linearSpec layer) := by
  apply LipschitzWith.of_dist_le_mul
  intro x y
  simp only [linearSpec, addSpec_eq_add, dist_add_right]
  have h := Proofs.linear_op_norm_bound layer.weights x y
  rw [Proofs.tensorL2Dist_eq_dist, Proofs.tensorL2Dist_eq_dist] at h
  convert! h using 1

/-- Configuration is fixed by the architecture; the two upstream layers carry weights and biases. -/
structure MLP (d m o : ℕ) where
  first : LinearSpec ℝ d m
  second : LinearSpec ℝ m o

def MLP.forward {d m o : ℕ} (net : MLP d m o) (x : Tensor ℝ [d]) : Tensor ℝ [o] :=
  linearSpec net.second (Activation.reluSpec (linearSpec net.first x))

theorem MLP.lipschitz {d m o : ℕ} (net : MLP d m o) :
    LipschitzWith (layerBound net.second * layerBound net.first) net.forward := by
  convert! (show LipschitzWith (layerBound net.second * (1 * layerBound net.first))
    (fun x => linearSpec net.second (Activation.reluSpec (linearSpec net.first x))) from
    (affine_lipschitz net.second).comp
      (Proofs.reluSpec_lipschitzWith.comp (affine_lipschitz net.first))) using 1
  simp

def MLP.residual {d m : ℕ} (net : MLP d m d) (x : Tensor ℝ [d]) : Tensor ℝ [d] :=
  x + net.forward x

theorem MLP.residual_lipschitz {d m : ℕ} (net : MLP d m d) :
    LipschitzWith (1 + layerBound net.second * layerBound net.first) net.residual :=
  LipschitzWith.id.add net.lipschitz

def vector {d : ℕ} (x : Fin d → ℝ) : Tensor ℝ [d] :=
  TorchLean.Tensor.Internal.Rep.ofFn (fun i => x i.1)

def matrix {d m : ℕ} (W : Matrix (Fin m) (Fin d) ℝ) : Tensor ℝ [m, d] :=
  TorchLean.Tensor.Internal.Rep.ofFn (fun i => W i.1 i.2.1)

/-- Preserve the legacy activation explicitly; the legacy structure excludes ReLU at zero. -/
def legacyForward {d m : ℕ} (net : TwoLayerNet d m) (x : Tensor ℝ [d]) : Tensor ℝ [1] :=
  linearSpec (⟨matrix net.W₂, 0⟩ : LinearSpec ℝ m 1)
    (Tensor.map net.act.σ (linearSpec (⟨matrix net.W₁, 0⟩ : LinearSpec ℝ d m) x))

theorem legacy_correspondence {d m : ℕ} (net : TwoLayerNet d m) (x : Fin d → ℝ) :
    getScalar (legacyForward net (vector x)) 0 = net.forward x := by
  have hmap (v : Tensor ℝ [m]) (i : Fin m) :
      getScalar (Tensor.map net.act.σ v) i = net.act.σ (getScalar v i) := by
    simp [getScalar_eq_apply, Tensor.map]
  simp only [legacyForward, affine_coordinate, hmap]
  simp [vector, matrix, TwoLayerNet.forward, getScalar_eq_apply, get2_eq]

end OpenTheory.Torch
