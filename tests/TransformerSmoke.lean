import OpenTheory.Architectures.Transformer

open OpenTheory.Transformer TorchLean

-- The configurable block must preserve the upstream encoder at its default epsilon.
example {n h d f : ℕ} (cfg : Config n h d f)
    (layer : Spec.TransformerEncoderLayer h d f ℝ) (x : Tensor ℝ [n, d])
    (hε : cfg.epsilon = TorchLean.normalizationEpsilon) :
    forward cfg layer x none = Spec.TransformerEncoderLayer.forward layer x cfg.seq_pos cfg.dim_pos :=
  forward_eq_upstream cfg layer x hε
