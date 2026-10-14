import NN.Proofs.Autograd.FDeriv.HardMaskedSoftmax
import NN.Proofs.Models.Attention.CausalMask
import Mathlib.Analysis.Convex.Combination

/-! Exact hard-masked attention. Bounds are deterministic and impose no data law.
Empty allowed sets return zero; normalization requires at least one allowed key. -/
noncomputable section
open TorchLean TorchLean.Tensor Spec
open scoped BigOperators

namespace OpenTheory.Attention

abbrev rowWeights {n : ℕ} := @Spec.hardMaskedSoftmaxVecSpec ℝ _ _ _ n

theorem weight_formula {n : ℕ} (scores : Tensor ℝ [n]) (mask : Tensor Bool [n]) (i : Fin n) :
    (rowWeights scores mask).getScalar i =
      Proofs.Autograd.HardMaskedSoftmax.numerator mask (Proofs.Autograd.getScalarE scores) i /
      Proofs.Autograd.HardMaskedSoftmax.denominator mask (Proofs.Autograd.getScalarE scores) :=
  Proofs.Autograd.HardMaskedSoftmax.getScalar_hardMaskedSoftmaxVecSpec scores mask i

theorem weight_nonneg {n : ℕ} (scores : Tensor ℝ [n]) (mask : Tensor Bool [n]) (i : Fin n) :
    0 ≤ (rowWeights scores mask).getScalar i := by
  rw [weight_formula]
  apply div_nonneg
  · unfold Proofs.Autograd.HardMaskedSoftmax.numerator
    split <;> positivity
  · apply Finset.sum_nonneg
    intro j _
    unfold Proofs.Autograd.HardMaskedSoftmax.numerator
    split <;> positivity

theorem weights_sum_one {n : ℕ} (scores : Tensor ℝ [n]) (mask : Tensor Bool [n])
    (h : ∃ i, mask.getScalar i = true) :
    ∑ i, (rowWeights scores mask).getScalar i = 1 := by
  simp_rw [weight_formula]
  rw [← Finset.sum_div]
  exact div_self (ne_of_gt (Proofs.Autograd.HardMaskedSoftmax.denominator_pos mask _ h))

theorem weight_blocked {n : ℕ} (scores : Tensor ℝ [n]) (mask : Tensor Bool [n])
    (i : Fin n) (h : mask.getScalar i = false) : (rowWeights scores mask).getScalar i = 0 :=
  NN.Proofs.Models.Attention.hardMaskedSoftmaxVecSpec_blocked_eq_zero scores mask i h

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

def rowOutput {n : ℕ} (scores : Tensor ℝ [n]) (mask : Tensor Bool [n]) (values : Fin n → E) : E :=
  ∑ i, (rowWeights scores mask).getScalar i • values i

theorem rowOutput_blocked {n : ℕ} (scores : Tensor ℝ [n]) (mask : Tensor Bool [n])
    (values : Fin n → E) (h : ∀ i, mask.getScalar i = false) : rowOutput scores mask values = 0 := by
  simp [rowOutput, weight_blocked scores mask _ (h _)]

theorem rowOutput_mem_convexHull {n : ℕ} (scores : Tensor ℝ [n]) (mask : Tensor Bool [n])
    (values : Fin n → E) (h : ∃ i, mask.getScalar i = true) :
    rowOutput scores mask values ∈ convexHull ℝ (Set.range values) := by
  exact (convex_convexHull ℝ _).sum_mem
    (fun i _ => weight_nonneg scores mask i) (weights_sum_one scores mask h)
    (fun i _ => subset_convexHull ℝ _ (Set.mem_range_self i))

theorem rowOutput_norm_le {n : ℕ} (scores : Tensor ℝ [n]) (mask : Tensor Bool [n])
    (values : Fin n → E) (h : ∃ i, mask.getScalar i = true) (B : ℝ)
    (hB : ∀ i, ‖values i‖ ≤ B) : ‖rowOutput scores mask values‖ ≤ B := by
  calc
    ‖rowOutput scores mask values‖ ≤ ∑ i, ‖(rowWeights scores mask).getScalar i • values i‖ :=
      norm_sum_le _ _
    _ = ∑ i, (rowWeights scores mask).getScalar i * ‖values i‖ := by
      simp [norm_smul, Real.norm_eq_abs, abs_of_nonneg (weight_nonneg scores mask _)]
    _ ≤ ∑ i, (rowWeights scores mask).getScalar i * B :=
      Finset.sum_le_sum fun i _ => mul_le_mul_of_nonneg_left (hB i) (weight_nonneg scores mask i)
    _ = B := by rw [← Finset.sum_mul, weights_sum_one scores mask h, one_mul]

/-- Nonexpansiveness in values with scores and mask fixed, not a global Q/K/V Lipschitz claim. -/
theorem rowOutput_perturbation {n : ℕ} (scores : Tensor ℝ [n]) (mask : Tensor Bool [n])
    (v w : Fin n → E) (h : ∃ i, mask.getScalar i = true) (δ : ℝ)
    (hδ : ∀ i, ‖v i - w i‖ ≤ δ) :
    ‖rowOutput scores mask v - rowOutput scores mask w‖ ≤ δ := by
  have he : rowOutput scores mask v - rowOutput scores mask w =
      rowOutput scores mask (fun i => v i - w i) := by
    simp [rowOutput, smul_sub, Finset.sum_sub_distrib]
  rw [he]
  exact rowOutput_norm_le scores mask _ h δ hδ

def scores {nQ nK d : ℕ} {hQ : nQ ≠ 0} {hK : nK ≠ 0}
    (ctx : AttentionContext ℝ nQ nK d hQ hK) : Tensor ℝ [nQ, nK] :=
  scaleSpec (matMulSpec ctx.Q (swapAdjacentAxes ctx.K 0)) (1 / attentionScaleDenom (α := ℝ) d)

/-- The convex-combination bounds apply to the actual upstream scaled-dot-product output. -/
theorem masked_output_coordinate {nQ nK d : ℕ} {hQ : nQ ≠ 0} {hK : nK ≠ 0}
    (ctx : AttentionContext ℝ nQ nK d hQ hK) (mask : Tensor Bool [nQ, nK])
    (i : Fin nQ) (k : Fin d) :
    get2 (scaledDotProductAttention (h1 := hQ) (h2 := hK) {ctx with mask := some mask}) i k =
      rowOutput (get (scores ctx) i) (get mask i) (fun j => get2 ctx.V j k) := by
  change get2 (matMulSpec (hardMaskedSoftmaxSpec (scores ctx) mask) ctx.V) i k = _
  rw [get2_mat_mul_spec]
  simp [rowOutput, rowWeights, get2_eq_getScalar_get, hardMaskedSoftmaxSpec,
    Spec.get_eq, smul_eq_mul]

theorem causal_future_zero {n : ℕ} (s : Tensor ℝ [n, n]) (i j : Fin n) (h : i.val < j.val) :
    get2 (hardMaskedSoftmaxSpec s (causalMask n)) i j = 0 :=
  NN.Proofs.Models.Attention.hardMaskedSoftmaxSpec_causal_future_zero s i j h

theorem masked_output_abs_le {nQ nK d : ℕ} {hQ : nQ ≠ 0} {hK : nK ≠ 0}
    (ctx : AttentionContext ℝ nQ nK d hQ hK) (mask : Tensor Bool [nQ, nK])
    (i : Fin nQ) (k : Fin d) (h : ∃ j, get2 mask i j = true)
    (B : ℝ) (hv : ∀ j, |get2 ctx.V j k| ≤ B) :
    |get2 (scaledDotProductAttention (h1 := hQ) (h2 := hK) {ctx with mask := some mask}) i k| ≤ B := by
  rw [masked_output_coordinate]
  simpa only [Real.norm_eq_abs] using rowOutput_norm_le
    (get (scores ctx) i) (get mask i) (fun j => get2 ctx.V j k) h B hv

theorem causal_row_sum_one {n : ℕ} (s : Tensor ℝ [n, n]) (i : Fin n) :
    ∑ j, get2 (hardMaskedSoftmaxSpec s (causalMask n)) i j = 1 := by
  have h : ∃ j, (get (causalMask n) i).getScalar j = true :=
    ⟨i, NN.Proofs.Models.Attention.causalMask_allows_past i i (le_refl _)⟩
  have hw := weights_sum_one (get s i) (get (causalMask n) i) h
  simpa [rowWeights, get2_eq_getScalar_get, hardMaskedSoftmaxSpec, Spec.get_eq] using hw

end OpenTheory.Attention
