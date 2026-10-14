import OpenTheory.Architectures.Torch

open OpenTheory TorchLean

-- Biases must be included: a zero-weight layer is a constant map to its bias.
example {d m : ℕ} (b : Tensor ℝ [m]) (x : Tensor ℝ [d]) :
    Spec.linearSpec (⟨0, b⟩ : Spec.LinearSpec ℝ d m) x = b := by
  apply Tensor.ext_vector
  intro i
  simp [Torch.affine_coordinate, Spec.get2_eq_apply]
