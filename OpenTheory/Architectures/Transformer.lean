import NN.Spec.Models.Transformer
import NN.Spec.Core.Context.Real

/-! A masked post-norm transformer block reusing upstream parameters and operations.
This is deterministic inference semantics: no dropout, sampling, or training-state claims. -/
noncomputable section
open TorchLean TorchLean.Tensor Spec

namespace OpenTheory.Transformer

/-- Configuration is separate from learned parameters. Require full head coverage and positive epsilon. -/
structure Config (seqLen heads dim hidden : ℕ) where
  seq_pos : 0 < seqLen
  heads_pos : 0 < heads
  dim_pos : 0 < dim
  hidden_pos : 0 < hidden
  heads_divide : heads ∣ dim
  epsilon : ℝ
  epsilon_pos : 0 < epsilon

theorem Config.head_coverage {n h d f : ℕ} (cfg : Config n h d f) : h * (d / h) = d :=
  Nat.mul_div_cancel' cfg.heads_divide

theorem Config.head_dim_pos {n h d f : ℕ} (cfg : Config n h d f) : 0 < d / h :=
  Nat.div_pos (Nat.le_of_dvd cfg.dim_pos cfg.heads_divide) cfg.heads_pos

def forward {n h d f : ℕ} (cfg : Config n h d f)
    (layer : TransformerEncoderLayer h d f ℝ) (x : Tensor ℝ [n, d])
    (mask : Option (Tensor Bool [n, n])) : Tensor ℝ [n, d] :=
  let attn := MultiHeadAttention.forward n (Nat.ne_of_gt cfg.seq_pos) layer.mha x mask
  let normAttn := layerNorm (addSpec x attn) layer.norm1Scale layer.norm1Bias
    cfg.seq_pos cfg.dim_pos cfg.epsilon
  let ff := FeedForward.forward layer.ffn normAttn
  layerNorm (addSpec normAttn ff) layer.norm2Scale layer.norm2Bias
    cfg.seq_pos cfg.dim_pos cfg.epsilon

theorem forward_eq_upstream {n h d f : ℕ} (cfg : Config n h d f)
    (layer : TransformerEncoderLayer h d f ℝ) (x : Tensor ℝ [n, d])
    (hε : cfg.epsilon = TorchLean.normalizationEpsilon) :
    forward cfg layer x none = TransformerEncoderLayer.forward layer x cfg.seq_pos cfg.dim_pos := by
  simp only [forward, TransformerEncoderLayer.forward, hε]

def causalForward {n h d f : ℕ} (cfg : Config n h d f)
    (layer : TransformerEncoderLayer h d f ℝ) (x : Tensor ℝ [n, d]) : Tensor ℝ [n, d] :=
  forward cfg layer x (some (causalMask n))

end OpenTheory.Transformer
