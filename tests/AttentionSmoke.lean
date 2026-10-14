import OpenTheory.Architectures.Attention

open TorchLean OpenTheory.Attention

-- Equal scores and two allowed keys must average values, not sum them.
example : rowOutput (Tensor.full [2] (0 : ℝ)) (Tensor.full [2] true)
    (fun i : Fin 2 => if i = 0 then (2 : ℝ) else 4) = 3 := by
  norm_num [rowOutput, rowWeights, weight_formula,
    Proofs.Autograd.HardMaskedSoftmax.numerator,
    Proofs.Autograd.HardMaskedSoftmax.denominator, Fin.sum_univ_two]

-- A fully blocked row must produce zero, not uniform attention.
example (scores : Tensor ℝ [2]) (values : Fin 2 → ℝ) :
    rowOutput scores (Tensor.full [2] false) values = 0 := by
  exact rowOutput_blocked scores _ values (by intro i; simp)

-- Causal masking includes the present token; normalization is valid even in row zero.
example (scores : Tensor ℝ [3, 3]) :
    ∑ j, Spec.get2 (Spec.hardMaskedSoftmaxSpec scores (Spec.causalMask 3)) 0 j = 1 :=
  causal_row_sum_one scores 0
