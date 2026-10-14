import OpenTheory.Architectures.Attention
import OpenTheory.Architectures.AttentionAblation

/-!
# Attention Score/Logit Perturbation, Joint Q/K/V Control, and Sharp Concentration Stability

In real Transformers, upstream interventions, quantization, or adversarial perturbations
change queries `Q` and keys `K`, perturbing the attention logits/scores `s` and shifting
the masked softmax distribution `rowWeights s mask`.

This module formalizes three core pillars of attention logit stability:

1. **Attention Weight Shift & Total Variation (`l₁`) Output Control**:
   - Defines `weightL1Diff s₁ s₂ mask`, the `l₁` (total variation) distance between two
     masked attention weight vectors.
   - Proves `rowOutput_score_perturbation_l1`: raw value norm control
     `‖rowOutput s₁ mask v - rowOutput s₂ mask v‖ ≤ B_V * weightL1Diff s₁ s₂ mask`.
   - Proves `rowOutput_score_perturbation_centered`: because `rowWeights s₁ mask` and
     `rowWeights s₂ mask` both sum to `1` on unmasked keys and vanish on masked keys,
     `∑ i, (p₁ i - p₂ i) • v_bar = 0` for any reference vector `v_bar : E`. Thus score
     perturbation sensitivity scales with the **value dispersion** `Δ_V` around `v_bar`
     (`‖v i - v_bar‖ ≤ Δ_V` on unmasked keys), rather than the raw value norm `B_V`.
   - Proves joint score + value (`Q/K/V`) perturbation bounds (`rowOutput_joint_score_value_perturbation`,
     `rowOutput_joint_qkv_perturbation`, and `multiHead_joint_qkv_perturbation`) for single-row
     and `H`-head attention layers.

2. **Sharp / Focused Attention Concentration & Dominant-Token Stability**:
   - For an attention row focusing at least `1 - ε` of its mass on a single dominant key `i_star`,
     proves `tail_mass_eq_of_focused`: the complementary tail mass equals `1 - p(i_star) ≤ ε`.
   - Proves `focused_rowOutput_approx_dominant` and `focused_rowOutput_approx_dominant_norm`:
     the attention output is within `ε * D_V` (or `2 * ε * B_V`) of the dominant token value `v i_star`.
   - Proves `focused_weight_l1_bound` and `focused_score_perturbation_stability`: if two score
     vectors `s₁, s₂` both concentrate on `i_star` with tail masses `ε₁, ε₂`, then
     `weightL1Diff s₁ s₂ mask ≤ 2 * (ε₁ + ε₂)` and their attention outputs differ by at most
     `(ε₁ + ε₂) * D_V` (without factor 2, via triangle inequality through `v i_star`, and also
     bounded by `2 * (ε₁ + ε₂) * Δ_V` via centered L1 control).

3. **Logit Margin → Exponential Concentration Certificate**:
   - Using `Attention.weight_formula` and `Proofs.Autograd.HardMaskedSoftmax.numerator`, proves
     that if `mask.getScalar i_star = true` and `i_star` has a logit margin `γ` over all other
     unmasked keys (`s.getScalar j ≤ s.getScalar i_star - γ`), then every other key `j ≠ i_star`
     satisfies `(rowWeights s mask).getScalar j ≤ Real.exp (-γ) * (rowWeights s mask).getScalar i_star`.
   - Proves `logit_margin_implies_focused`: the tail mass `1 - (rowWeights s mask).getScalar i_star`
     is bounded by `((n - 1 : ℕ) : ℝ) * Real.exp (-γ)` and by `(n : ℝ) * Real.exp (-γ)`.
   - Proves `logit_margin_output_stability`: a logit margin `γ` guarantees that `rowOutput s mask v`
     is within `((n - 1 : ℕ) : ℝ) * Real.exp (-γ) * D_V` (and `(n : ℝ) * Real.exp (-γ) * D_V`)
     of the dominant value `v i_star`.
-/

noncomputable section

open TorchLean TorchLean.Tensor Spec
open scoped BigOperators

namespace OpenTheory.AttentionLogitStability

variable {E F : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
  [NormedAddCommGroup F] [NormedSpace ℝ F]

/-! ## 1. Attention Weight Shift & Total Variation (`l₁`) Output Control -/

/-- The `l₁` (total variation) distance between the attention weight distributions induced
by two score vectors `s₁, s₂ : Tensor ℝ [n]` under a common boolean mask `mask`. -/
def weightL1Diff {n : ℕ} (s₁ s₂ : Tensor ℝ [n]) (mask : Tensor Bool [n]) : ℝ :=
  ∑ i : Fin n,
    |(Attention.rowWeights s₁ mask).getScalar i - (Attention.rowWeights s₂ mask).getScalar i|

/-- `weightL1Diff` is always nonnegative. -/
theorem weightL1Diff_nonneg {n : ℕ} (s₁ s₂ : Tensor ℝ [n]) (mask : Tensor Bool [n]) :
    0 ≤ weightL1Diff s₁ s₂ mask :=
  Finset.sum_nonneg fun _ _ => abs_nonneg _

/-- `weightL1Diff` vanishes when the two score vectors are identical. -/
@[simp] theorem weightL1Diff_self {n : ℕ} (s : Tensor ℝ [n]) (mask : Tensor Bool [n]) :
    weightL1Diff s s mask = 0 := by
  simp [weightL1Diff]

/-- `weightL1Diff` is symmetric in `s₁` and `s₂`. -/
theorem weightL1Diff_comm {n : ℕ} (s₁ s₂ : Tensor ℝ [n]) (mask : Tensor Bool [n]) :
    weightL1Diff s₁ s₂ mask = weightL1Diff s₂ s₁ mask := by
  unfold weightL1Diff
  refine Finset.sum_congr rfl fun i _ => ?_
  exact abs_sub_comm _ _

/-- `weightL1Diff` satisfies the triangle inequality across three score vectors. -/
theorem weightL1Diff_triangle {n : ℕ} (s₁ s₂ s₃ : Tensor ℝ [n]) (mask : Tensor Bool [n]) :
    weightL1Diff s₁ s₃ mask ≤ weightL1Diff s₁ s₂ mask + weightL1Diff s₂ s₃ mask := by
  unfold weightL1Diff
  rw [← Finset.sum_add_distrib]
  refine Finset.sum_le_sum fun i _ => ?_
  exact abs_sub_le _ _ _

/-- Exact algebraic decomposition of the attention row output difference under score perturbation. -/
theorem rowOutput_sub_eq_sum {n : ℕ} (s₁ s₂ : Tensor ℝ [n]) (mask : Tensor Bool [n])
    (v : Fin n → E) :
    Attention.rowOutput s₁ mask v - Attention.rowOutput s₂ mask v =
      ∑ i : Fin n,
        ((Attention.rowWeights s₁ mask).getScalar i -
          (Attention.rowWeights s₂ mask).getScalar i) • v i := by
  unfold Attention.rowOutput
  rw [← Finset.sum_sub_distrib]
  refine Finset.sum_congr rfl fun i _ => ?_
  rw [sub_smul]

/-- **Score perturbation `l₁` output bound**: for any normed space `E`, score vectors `s₁, s₂`,
mask `mask`, and values `v : Fin n → E` bounded in norm by `B_V`, the change in attention output
is bounded by `B_V * weightL1Diff s₁ s₂ mask`. -/
theorem rowOutput_score_perturbation_l1 {n : ℕ}
    (s₁ s₂ : Tensor ℝ [n]) (mask : Tensor Bool [n])
    (v : Fin n → E) (B_V : ℝ) (hv : ∀ i, ‖v i‖ ≤ B_V) :
    ‖Attention.rowOutput s₁ mask v - Attention.rowOutput s₂ mask v‖ ≤
      B_V * weightL1Diff s₁ s₂ mask := by
  rw [rowOutput_sub_eq_sum]
  calc
    ‖∑ i : Fin n,
        ((Attention.rowWeights s₁ mask).getScalar i -
          (Attention.rowWeights s₂ mask).getScalar i) • v i‖ ≤
        ∑ i : Fin n,
          ‖((Attention.rowWeights s₁ mask).getScalar i -
            (Attention.rowWeights s₂ mask).getScalar i) • v i‖ :=
      norm_sum_le _ _
    _ = ∑ i : Fin n,
          |(Attention.rowWeights s₁ mask).getScalar i -
            (Attention.rowWeights s₂ mask).getScalar i| * ‖v i‖ := by
      refine Finset.sum_congr rfl fun i _ => ?_
      rw [norm_smul, Real.norm_eq_abs]
    _ ≤ ∑ i : Fin n,
          |(Attention.rowWeights s₁ mask).getScalar i -
            (Attention.rowWeights s₂ mask).getScalar i| * B_V :=
      Finset.sum_le_sum fun i _ =>
        mul_le_mul_of_nonneg_left (hv i) (abs_nonneg _)
    _ = B_V * weightL1Diff s₁ s₂ mask := by
      unfold weightL1Diff
      rw [← Finset.sum_mul, mul_comm]

/-- When at least one key is unmasked, the weight differences `(p₁ i - p₂ i)` sum to zero,
so subtracting any constant reference vector `v_bar : E` from every value leaves the
attention output difference unchanged. -/
theorem rowOutput_sub_eq_centered_sum {n : ℕ}
    (s₁ s₂ : Tensor ℝ [n]) (mask : Tensor Bool [n])
    (v : Fin n → E) (v_bar : E)
    (hmask : ∃ i, mask.getScalar i = true) :
    Attention.rowOutput s₁ mask v - Attention.rowOutput s₂ mask v =
      ∑ i : Fin n,
        ((Attention.rowWeights s₁ mask).getScalar i -
          (Attention.rowWeights s₂ mask).getScalar i) • (v i - v_bar) := by
  rw [rowOutput_sub_eq_sum]
  have hsum_zero :
      ∑ i : Fin n,
        ((Attention.rowWeights s₁ mask).getScalar i -
          (Attention.rowWeights s₂ mask).getScalar i) • v_bar = 0 := by
    rw [← Finset.sum_smul, Finset.sum_sub_distrib,
      Attention.weights_sum_one s₁ mask hmask,
      Attention.weights_sum_one s₂ mask hmask, sub_self, zero_smul]
  calc
    ∑ i : Fin n,
        ((Attention.rowWeights s₁ mask).getScalar i -
          (Attention.rowWeights s₂ mask).getScalar i) • v i =
      (∑ i : Fin n,
        ((Attention.rowWeights s₁ mask).getScalar i -
          (Attention.rowWeights s₂ mask).getScalar i) • v i) -
      ∑ i : Fin n,
        ((Attention.rowWeights s₁ mask).getScalar i -
          (Attention.rowWeights s₂ mask).getScalar i) • v_bar := by
      rw [hsum_zero, sub_zero]
    _ = ∑ i : Fin n,
        ((Attention.rowWeights s₁ mask).getScalar i -
          (Attention.rowWeights s₂ mask).getScalar i) • (v i - v_bar) := by
      rw [← Finset.sum_sub_distrib]
      refine Finset.sum_congr rfl fun i _ => ?_
      rw [← smul_sub]

/-- **Centered value score-perturbation bound**: if `∃ i, mask.getScalar i = true`, then for
ANY reference vector `v_bar : E` (e.g. mean value or dominant value) such that every unmasked key
satisfies `‖v i - v_bar‖ ≤ Δ_V`, the score-perturbation output shift is bounded by
`Δ_V * weightL1Diff s₁ s₂ mask`.
In particular, attention logit sensitivity scales with the **value dispersion** `Δ_V` rather
than the raw value norm `B_V`. -/
theorem rowOutput_score_perturbation_centered {n : ℕ}
    (s₁ s₂ : Tensor ℝ [n]) (mask : Tensor Bool [n])
    (v : Fin n → E) (v_bar : E) (Δ_V : ℝ)
    (hmask : ∃ i, mask.getScalar i = true)
    (hΔ : ∀ i, mask.getScalar i = true → ‖v i - v_bar‖ ≤ Δ_V) :
    ‖Attention.rowOutput s₁ mask v - Attention.rowOutput s₂ mask v‖ ≤
      Δ_V * weightL1Diff s₁ s₂ mask := by
  obtain ⟨i₀, hi₀⟩ := hmask
  have hΔ_nonneg : 0 ≤ Δ_V := le_trans (norm_nonneg _) (hΔ i₀ hi₀)
  rw [rowOutput_sub_eq_centered_sum s₁ s₂ mask v v_bar ⟨i₀, hi₀⟩]
  calc
    ‖∑ i : Fin n,
        ((Attention.rowWeights s₁ mask).getScalar i -
          (Attention.rowWeights s₂ mask).getScalar i) • (v i - v_bar)‖ ≤
        ∑ i : Fin n,
          ‖((Attention.rowWeights s₁ mask).getScalar i -
            (Attention.rowWeights s₂ mask).getScalar i) • (v i - v_bar)‖ :=
      norm_sum_le _ _
    _ = ∑ i : Fin n,
          |(Attention.rowWeights s₁ mask).getScalar i -
            (Attention.rowWeights s₂ mask).getScalar i| * ‖v i - v_bar‖ := by
      refine Finset.sum_congr rfl fun i _ => ?_
      rw [norm_smul, Real.norm_eq_abs]
    _ ≤ ∑ i : Fin n,
          |(Attention.rowWeights s₁ mask).getScalar i -
            (Attention.rowWeights s₂ mask).getScalar i| * Δ_V := by
      refine Finset.sum_le_sum fun i _ => ?_
      cases hmi : mask.getScalar i
      · rw [Attention.weight_blocked s₁ mask i hmi,
          Attention.weight_blocked s₂ mask i hmi, sub_self, abs_zero, zero_mul, zero_mul]
      · exact mul_le_mul_of_nonneg_left (hΔ i hmi) (abs_nonneg _)
    _ = Δ_V * weightL1Diff s₁ s₂ mask := by
      unfold weightL1Diff
      rw [← Finset.sum_mul, mul_comm]

/-- **Joint Score + Value (`Q/K/V`) Perturbation Theorem (centered form)**:
When both scores change (`s₁ → s₂`) and values change (`v₁ → v₂` with `‖v₁ i - v₂ i‖ ≤ δ_V`),
if at least one key is unmasked and `‖v₁ i - v_bar‖ ≤ Δ_V`, the output change is at most
`Δ_V * weightL1Diff s₁ s₂ mask + δ_V`. -/
theorem rowOutput_joint_qkv_perturbation {n : ℕ}
    (s₁ s₂ : Tensor ℝ [n]) (mask : Tensor Bool [n])
    (v₁ v₂ : Fin n → E) (v_bar : E) (Δ_V δ_V : ℝ)
    (hmask : ∃ i, mask.getScalar i = true)
    (hΔ : ∀ i, mask.getScalar i = true → ‖v₁ i - v_bar‖ ≤ Δ_V)
    (hδ : ∀ i, ‖v₁ i - v₂ i‖ ≤ δ_V) :
    ‖Attention.rowOutput s₁ mask v₁ - Attention.rowOutput s₂ mask v₂‖ ≤
      Δ_V * weightL1Diff s₁ s₂ mask + δ_V := by
  have hdecomp :
      Attention.rowOutput s₁ mask v₁ - Attention.rowOutput s₂ mask v₂ =
        (Attention.rowOutput s₁ mask v₁ - Attention.rowOutput s₂ mask v₁) +
          (Attention.rowOutput s₂ mask v₁ - Attention.rowOutput s₂ mask v₂) := by
    abel
  rw [hdecomp]
  refine (norm_add_le _ _).trans (add_le_add ?_ ?_)
  · exact rowOutput_score_perturbation_centered s₁ s₂ mask v₁ v_bar Δ_V hmask hΔ
  · exact Attention.rowOutput_perturbation s₂ mask v₁ v₂ hmask δ_V hδ

/-- **Joint Score + Value Perturbation Theorem (uncentered / raw value norm form)**:
Special case of `rowOutput_joint_qkv_perturbation` with `v_bar = 0`. -/
theorem rowOutput_joint_score_value_perturbation {n : ℕ}
    (s₁ s₂ : Tensor ℝ [n]) (mask : Tensor Bool [n])
    (v₁ v₂ : Fin n → E) (B_V δ_V : ℝ)
    (hmask : ∃ i, mask.getScalar i = true)
    (hB : ∀ i, ‖v₁ i‖ ≤ B_V)
    (hδ : ∀ i, ‖v₁ i - v₂ i‖ ≤ δ_V) :
    ‖Attention.rowOutput s₁ mask v₁ - Attention.rowOutput s₂ mask v₂‖ ≤
      B_V * weightL1Diff s₁ s₂ mask + δ_V := by
  refine rowOutput_joint_qkv_perturbation s₁ s₂ mask v₁ v₂ 0 B_V δ_V hmask ?_ hδ
  intro i _
  simpa using hB i

/-- **Multi-Head Joint Score + Value (`Q/K/V`) Perturbation Theorem**:
Across `H` attention heads with output projections `proj h : E →L[ℝ] F`, if each head `h`
has at least one unmasked key, value dispersion `‖v₁ h i - v_bar h‖ ≤ Δ_V h` on unmasked keys,
and value perturbation `‖v₁ h i - v₂ h i‖ ≤ δ_V h`, then the multi-head output change under
simultaneous score and value perturbations is bounded by
`∑ h : Fin H, ‖proj h‖ * (Δ_V h * weightL1Diff (s₁ h) (s₂ h) (mask h) + δ_V h)`. -/
theorem multiHead_joint_qkv_perturbation {H n : ℕ}
    (s₁ s₂ : Fin H → Tensor ℝ [n])
    (mask : Fin H → Tensor Bool [n])
    (v₁ v₂ : Fin H → Fin n → E)
    (v_bar : Fin H → E)
    (proj : Fin H → (E →L[ℝ] F))
    (Δ_V δ_V : Fin H → ℝ)
    (hmask : ∀ h, ∃ i, (mask h).getScalar i = true)
    (hΔ : ∀ h i, (mask h).getScalar i = true → ‖v₁ h i - v_bar h‖ ≤ Δ_V h)
    (hδ : ∀ h i, ‖v₁ h i - v₂ h i‖ ≤ δ_V h) :
    ‖AttentionAblation.multiHeadOutput s₁ mask v₁ proj -
      AttentionAblation.multiHeadOutput s₂ mask v₂ proj‖ ≤
      ∑ h : Fin H, ‖proj h‖ * (Δ_V h * weightL1Diff (s₁ h) (s₂ h) (mask h) + δ_V h) := by
  unfold AttentionAblation.multiHeadOutput
  rw [← Finset.sum_sub_distrib]
  calc
    ‖∑ h : Fin H,
        (AttentionAblation.headContribution s₁ mask v₁ proj h -
          AttentionAblation.headContribution s₂ mask v₂ proj h)‖ ≤
        ∑ h : Fin H,
          ‖AttentionAblation.headContribution s₁ mask v₁ proj h -
            AttentionAblation.headContribution s₂ mask v₂ proj h‖ :=
      norm_sum_le _ _
    _ ≤ ∑ h : Fin H,
          ‖proj h‖ * (Δ_V h * weightL1Diff (s₁ h) (s₂ h) (mask h) + δ_V h) := by
      refine Finset.sum_le_sum fun h _ => ?_
      unfold AttentionAblation.headContribution
      rw [← map_sub]
      have hrow := rowOutput_joint_qkv_perturbation
        (s₁ h) (s₂ h) (mask h) (v₁ h) (v₂ h) (v_bar h) (Δ_V h) (δ_V h)
        (hmask h) (hΔ h) (hδ h)
      exact (proj h).le_opNorm _ |>.trans
        (mul_le_mul_of_nonneg_left hrow (norm_nonneg (proj h)))

/-! ## 2. Sharp / Focused Attention Concentration & Dominant-Token Stability -/

/-- Exact tail mass identity: when at least one key is unmasked, the sum of attention weights
over all keys other than `i_star` equals `1 - (rowWeights s mask).getScalar i_star`. -/
theorem tail_mass_eq {n : ℕ}
    (s : Tensor ℝ [n]) (mask : Tensor Bool [n]) (i_star : Fin n)
    (hmask : ∃ i, mask.getScalar i = true) :
    ∑ j ∈ Finset.univ.erase i_star, (Attention.rowWeights s mask).getScalar j =
      1 - (Attention.rowWeights s mask).getScalar i_star := by
  have hsplit := Finset.add_sum_erase Finset.univ
    (fun j => (Attention.rowWeights s mask).getScalar j) (Finset.mem_univ i_star)
  rw [Attention.weights_sum_one s mask hmask] at hsplit
  linarith

/-- If an attention row focuses at least `1 - ε` of its mass on `i_star` (with at least one
unmasked key), then the complementary tail mass over `Finset.univ.erase i_star` equals
`1 - (rowWeights s mask).getScalar i_star` and is at most `ε`. -/
theorem tail_mass_eq_of_focused {n : ℕ}
    (s : Tensor ℝ [n]) (mask : Tensor Bool [n]) (i_star : Fin n) (ε : ℝ)
    (hmask : ∃ i, mask.getScalar i = true)
    (hfocus : (Attention.rowWeights s mask).getScalar i_star ≥ 1 - ε) :
    ∑ j ∈ Finset.univ.erase i_star, (Attention.rowWeights s mask).getScalar j =
      1 - (Attention.rowWeights s mask).getScalar i_star ∧
    ∑ j ∈ Finset.univ.erase i_star, (Attention.rowWeights s mask).getScalar j ≤ ε := by
  have heq := tail_mass_eq s mask i_star hmask
  refine ⟨heq, ?_⟩
  linarith

/-- **Focused attention dominant-token approximation (dispersion form)**:
If key `i_star` receives weight at least `1 - ε` and all unmasked keys satisfy
`‖v j - v i_star‖ ≤ D_V`, then the attention row output is within `ε * D_V` of `v i_star`. -/
theorem focused_rowOutput_approx_dominant {n : ℕ}
    (s : Tensor ℝ [n]) (mask : Tensor Bool [n])
    (v : Fin n → E) (i_star : Fin n) (ε D_V : ℝ)
    (hmask : ∃ i, mask.getScalar i = true)
    (hfocus : (Attention.rowWeights s mask).getScalar i_star ≥ 1 - ε)
    (hD_nonneg : 0 ≤ D_V)
    (hD : ∀ j, mask.getScalar j = true → ‖v j - v i_star‖ ≤ D_V) :
    ‖Attention.rowOutput s mask v - v i_star‖ ≤ ε * D_V := by
  have hsum_one := Attention.weights_sum_one s mask hmask
  have hdecomp :
      Attention.rowOutput s mask v - v i_star =
        ∑ j ∈ Finset.univ.erase i_star,
          (Attention.rowWeights s mask).getScalar j • (v j - v i_star) := by
    unfold Attention.rowOutput
    calc
      (∑ j : Fin n, (Attention.rowWeights s mask).getScalar j • v j) - v i_star =
          (∑ j : Fin n, (Attention.rowWeights s mask).getScalar j • v j) -
            (∑ j : Fin n, (Attention.rowWeights s mask).getScalar j) • v i_star := by
        rw [hsum_one, one_smul]
      _ = ∑ j : Fin n,
            (Attention.rowWeights s mask).getScalar j • (v j - v i_star) := by
        rw [Finset.sum_smul, ← Finset.sum_sub_distrib]
        refine Finset.sum_congr rfl fun j _ => ?_
        rw [← smul_sub]
      _ = (Attention.rowWeights s mask).getScalar i_star • (v i_star - v i_star) +
            ∑ j ∈ Finset.univ.erase i_star,
              (Attention.rowWeights s mask).getScalar j • (v j - v i_star) := by
        exact (Finset.add_sum_erase Finset.univ
          (fun j => (Attention.rowWeights s mask).getScalar j • (v j - v i_star))
          (Finset.mem_univ i_star)).symm
      _ = ∑ j ∈ Finset.univ.erase i_star,
            (Attention.rowWeights s mask).getScalar j • (v j - v i_star) := by
        simp
  rw [hdecomp]
  calc
    ‖∑ j ∈ Finset.univ.erase i_star,
        (Attention.rowWeights s mask).getScalar j • (v j - v i_star)‖ ≤
        ∑ j ∈ Finset.univ.erase i_star,
          ‖(Attention.rowWeights s mask).getScalar j • (v j - v i_star)‖ :=
      norm_sum_le _ _
    _ = ∑ j ∈ Finset.univ.erase i_star,
          (Attention.rowWeights s mask).getScalar j * ‖v j - v i_star‖ := by
      refine Finset.sum_congr rfl fun j _ => ?_
      rw [norm_smul, Real.norm_eq_abs, abs_of_nonneg (Attention.weight_nonneg s mask j)]
    _ ≤ ∑ j ∈ Finset.univ.erase i_star,
          (Attention.rowWeights s mask).getScalar j * D_V := by
      refine Finset.sum_le_sum fun j _ => ?_
      cases hmj : mask.getScalar j
      · rw [Attention.weight_blocked s mask j hmj, zero_mul, zero_mul]
      · exact mul_le_mul_of_nonneg_left (hD j hmj) (Attention.weight_nonneg s mask j)
    _ = (∑ j ∈ Finset.univ.erase i_star, (Attention.rowWeights s mask).getScalar j) * D_V := by
      rw [← Finset.sum_mul]
    _ ≤ ε * D_V := by
      have htail := (tail_mass_eq_of_focused s mask i_star ε hmask hfocus).2
      exact mul_le_mul_of_nonneg_right htail hD_nonneg

/-- **Focused attention dominant-token approximation (raw value norm form)**:
If key `i_star` receives weight at least `1 - ε` and `‖v j‖ ≤ B_V` for all `j`, then
`‖Attention.rowOutput s mask v - v i_star‖ ≤ 2 * ε * B_V`. -/
theorem focused_rowOutput_approx_dominant_norm {n : ℕ}
    (s : Tensor ℝ [n]) (mask : Tensor Bool [n])
    (v : Fin n → E) (i_star : Fin n) (ε B_V : ℝ)
    (hmask : ∃ i, mask.getScalar i = true)
    (hfocus : (Attention.rowWeights s mask).getScalar i_star ≥ 1 - ε)
    (hB : ∀ j, ‖v j‖ ≤ B_V) :
    ‖Attention.rowOutput s mask v - v i_star‖ ≤ 2 * ε * B_V := by
  have hB_nonneg : 0 ≤ B_V := le_trans (norm_nonneg _) (hB i_star)
  have hD : ∀ j, mask.getScalar j = true → ‖v j - v i_star‖ ≤ 2 * B_V := by
    intro j _
    calc
      ‖v j - v i_star‖ ≤ ‖v j‖ + ‖v i_star‖ := norm_sub_le _ _
      _ ≤ B_V + B_V := add_le_add (hB j) (hB i_star)
      _ = 2 * B_V := by ring
  have h := focused_rowOutput_approx_dominant s mask v i_star ε (2 * B_V) hmask hfocus
    (by linarith) hD
  linarith

/-- **Focused attention `l₁` weight stability**:
If two score vectors `s₁, s₂` both concentrate on the same dominant key `i_star` with tail masses
`ε₁` and `ε₂` (`(rowWeights s₁ mask).getScalar i_star ≥ 1 - ε₁` and
`(rowWeights s₂ mask).getScalar i_star ≥ 1 - ε₂`), then their `l₁` weight distance satisfies
`weightL1Diff s₁ s₂ mask ≤ 2 * (ε₁ + ε₂)`. -/
theorem focused_weight_l1_bound {n : ℕ}
    (s₁ s₂ : Tensor ℝ [n]) (mask : Tensor Bool [n])
    (i_star : Fin n) (ε₁ ε₂ : ℝ)
    (hmask : ∃ i, mask.getScalar i = true)
    (hfocus₁ : (Attention.rowWeights s₁ mask).getScalar i_star ≥ 1 - ε₁)
    (hfocus₂ : (Attention.rowWeights s₂ mask).getScalar i_star ≥ 1 - ε₂) :
    weightL1Diff s₁ s₂ mask ≤ 2 * (ε₁ + ε₂) := by
  unfold weightL1Diff
  rw [← Finset.add_sum_erase Finset.univ
    (fun j =>
      |(Attention.rowWeights s₁ mask).getScalar j -
        (Attention.rowWeights s₂ mask).getScalar j|)
    (Finset.mem_univ i_star)]
  have htail₁ := (tail_mass_eq_of_focused s₁ mask i_star ε₁ hmask hfocus₁).2
  have htail₂ := (tail_mass_eq_of_focused s₂ mask i_star ε₂ hmask hfocus₂).2
  have heq₁ := (tail_mass_eq_of_focused s₁ mask i_star ε₁ hmask hfocus₁).1
  have heq₂ := (tail_mass_eq_of_focused s₂ mask i_star ε₂ hmask hfocus₂).1
  have h_istar_diff :
      (Attention.rowWeights s₁ mask).getScalar i_star -
        (Attention.rowWeights s₂ mask).getScalar i_star =
      (∑ j ∈ Finset.univ.erase i_star, (Attention.rowWeights s₂ mask).getScalar j) -
        (∑ j ∈ Finset.univ.erase i_star, (Attention.rowWeights s₁ mask).getScalar j) := by
    linarith
  have h_istar_abs :
      |(Attention.rowWeights s₁ mask).getScalar i_star -
        (Attention.rowWeights s₂ mask).getScalar i_star| ≤
      ∑ j ∈ Finset.univ.erase i_star,
        |(Attention.rowWeights s₁ mask).getScalar j -
          (Attention.rowWeights s₂ mask).getScalar j| := by
    rw [h_istar_diff, ← Finset.sum_sub_distrib]
    calc
      |∑ j ∈ Finset.univ.erase i_star,
          ((Attention.rowWeights s₂ mask).getScalar j -
            (Attention.rowWeights s₁ mask).getScalar j)| ≤
          ∑ j ∈ Finset.univ.erase i_star,
            |(Attention.rowWeights s₂ mask).getScalar j -
              (Attention.rowWeights s₁ mask).getScalar j| :=
        Finset.abs_sum_le_sum_abs _ _
      _ = ∑ j ∈ Finset.univ.erase i_star,
            |(Attention.rowWeights s₁ mask).getScalar j -
              (Attention.rowWeights s₂ mask).getScalar j| := by
        refine Finset.sum_congr rfl fun j _ => abs_sub_comm _ _
  have h_erase_le :
      ∑ j ∈ Finset.univ.erase i_star,
        |(Attention.rowWeights s₁ mask).getScalar j -
          (Attention.rowWeights s₂ mask).getScalar j| ≤ ε₁ + ε₂ := by
    calc
      ∑ j ∈ Finset.univ.erase i_star,
          |(Attention.rowWeights s₁ mask).getScalar j -
            (Attention.rowWeights s₂ mask).getScalar j| ≤
          ∑ j ∈ Finset.univ.erase i_star,
            ((Attention.rowWeights s₁ mask).getScalar j +
              (Attention.rowWeights s₂ mask).getScalar j) := by
        refine Finset.sum_le_sum fun j _ => ?_
        have h1 := Attention.weight_nonneg s₁ mask j
        have h2 := Attention.weight_nonneg s₂ mask j
        exact (abs_sub _ _).trans (by rw [abs_of_nonneg h1, abs_of_nonneg h2])
      _ = (∑ j ∈ Finset.univ.erase i_star, (Attention.rowWeights s₁ mask).getScalar j) +
            (∑ j ∈ Finset.univ.erase i_star, (Attention.rowWeights s₂ mask).getScalar j) :=
        Finset.sum_add_distrib
      _ ≤ ε₁ + ε₂ := add_le_add htail₁ htail₂
  linarith

/-- **Focused score-perturbation stability**:
If two score vectors `s₁, s₂` both concentrate on `i_star` with tail masses `ε₁` and `ε₂`,
and all unmasked keys have value distance at most `D_V` from `v i_star`, then:
1. `weightL1Diff s₁ s₂ mask ≤ 2 * (ε₁ + ε₂)`, and
2. `‖Attention.rowOutput s₁ mask v - Attention.rowOutput s₂ mask v‖ ≤ (ε₁ + ε₂) * D_V`. -/
theorem focused_score_perturbation_stability {n : ℕ}
    (s₁ s₂ : Tensor ℝ [n]) (mask : Tensor Bool [n])
    (v : Fin n → E) (i_star : Fin n) (ε₁ ε₂ D_V : ℝ)
    (hmask : ∃ i, mask.getScalar i = true)
    (hfocus₁ : (Attention.rowWeights s₁ mask).getScalar i_star ≥ 1 - ε₁)
    (hfocus₂ : (Attention.rowWeights s₂ mask).getScalar i_star ≥ 1 - ε₂)
    (hD_nonneg : 0 ≤ D_V)
    (hD : ∀ j, mask.getScalar j = true → ‖v j - v i_star‖ ≤ D_V) :
    weightL1Diff s₁ s₂ mask ≤ 2 * (ε₁ + ε₂) ∧
    ‖Attention.rowOutput s₁ mask v - Attention.rowOutput s₂ mask v‖ ≤ (ε₁ + ε₂) * D_V := by
  refine ⟨focused_weight_l1_bound s₁ s₂ mask i_star ε₁ ε₂ hmask hfocus₁ hfocus₂, ?_⟩
  have h1 := focused_rowOutput_approx_dominant s₁ mask v i_star ε₁ D_V hmask hfocus₁ hD_nonneg hD
  have h2 := focused_rowOutput_approx_dominant s₂ mask v i_star ε₂ D_V hmask hfocus₂ hD_nonneg hD
  have hdecomp :
      Attention.rowOutput s₁ mask v - Attention.rowOutput s₂ mask v =
        (Attention.rowOutput s₁ mask v - v i_star) -
          (Attention.rowOutput s₂ mask v - v i_star) := by
    abel
  rw [hdecomp]
  calc
    ‖(Attention.rowOutput s₁ mask v - v i_star) -
        (Attention.rowOutput s₂ mask v - v i_star)‖ ≤
        ‖Attention.rowOutput s₁ mask v - v i_star‖ +
          ‖Attention.rowOutput s₂ mask v - v i_star‖ :=
      norm_sub_le _ _
    _ ≤ ε₁ * D_V + ε₂ * D_V := add_le_add h1 h2
    _ = (ε₁ + ε₂) * D_V := by ring

/-! ## 3. Logit Margin → Exponential Concentration Certificate -/

/-- Pairwise exponential decay of attention weights under a logit margin:
if `mask.getScalar i_star = true` and every other unmasked key `j ≠ i_star` satisfies
`s.getScalar j ≤ s.getScalar i_star - γ`, then
`(rowWeights s mask).getScalar j ≤ Real.exp (-γ) * (rowWeights s mask).getScalar i_star`. -/
theorem weight_le_exp_neg_margin_mul_dominant {n : ℕ}
    (s : Tensor ℝ [n]) (mask : Tensor Bool [n])
    (i_star j : Fin n) (γ : ℝ)
    (hi_star : mask.getScalar i_star = true)
    (hj_ne : j ≠ i_star)
    (hmargin : ∀ k ≠ i_star, mask.getScalar k = true → s.getScalar k ≤ s.getScalar i_star - γ) :
    (Attention.rowWeights s mask).getScalar j ≤
      Real.exp (-γ) * (Attention.rowWeights s mask).getScalar i_star := by
  cases hmj : mask.getScalar j
  · rw [Attention.weight_blocked s mask j hmj]
    exact mul_nonneg (Real.exp_pos (-γ)).le (Attention.weight_nonneg s mask i_star)
  · rw [Attention.weight_formula s mask j, Attention.weight_formula s mask i_star,
      ← mul_div_assoc]
    have hdenom_nonneg :
        0 ≤ Proofs.Autograd.HardMaskedSoftmax.denominator mask
          (Proofs.Autograd.getScalarE s) :=
      (Proofs.Autograd.HardMaskedSoftmax.denominator_pos mask _ ⟨i_star, hi_star⟩).le
    refine div_le_div_of_nonneg_right ?_ hdenom_nonneg
    unfold Proofs.Autograd.HardMaskedSoftmax.numerator
    simp only [hmj, hi_star, ↓reduceIte, Proofs.Autograd.getScalarE_ofLp]
    rw [← Real.exp_add]
    apply Real.exp_le_exp.mpr
    have hm := hmargin j hj_ne hmj
    linarith

/-- If key `i_star` is unmasked, it receives at most `1` of the total attention weight, so
any upper bound on the tail sum immediately bounds `1 - (rowWeights s mask).getScalar i_star`. -/
theorem dominant_weight_le_one {n : ℕ}
    (s : Tensor ℝ [n]) (mask : Tensor Bool [n])
    (i_star : Fin n) (hi_star : mask.getScalar i_star = true) :
    (Attention.rowWeights s mask).getScalar i_star ≤ 1 := by
  have htail := tail_mass_eq s mask i_star ⟨i_star, hi_star⟩
  have htail_nonneg :
      0 ≤ ∑ j ∈ Finset.univ.erase i_star, (Attention.rowWeights s mask).getScalar j :=
    Finset.sum_nonneg fun j _ => Attention.weight_nonneg s mask j
  linarith

/-- **Logit Margin → Exponential Concentration Certificate**:
If key `i_star` is unmasked (`mask.getScalar i_star = true`) and has a logit margin `γ` over
every other unmasked key (`∀ j ≠ i_star, mask.getScalar j = true → s.getScalar j ≤ s.getScalar i_star - γ`),
then:
1. Every key `j ≠ i_star` satisfies `(rowWeights s mask).getScalar j ≤ Real.exp (-γ) * (rowWeights s mask).getScalar i_star`,
2. The tail mass `1 - (rowWeights s mask).getScalar i_star` is bounded by `((n - 1 : ℕ) : ℝ) * Real.exp (-γ)`, and
3. Consequently `1 - (rowWeights s mask).getScalar i_star ≤ (n : ℝ) * Real.exp (-γ)`. -/
theorem logit_margin_implies_focused {n : ℕ}
    (s : Tensor ℝ [n]) (mask : Tensor Bool [n])
    (i_star : Fin n) (γ : ℝ)
    (hi_star : mask.getScalar i_star = true)
    (hmargin : ∀ j ≠ i_star, mask.getScalar j = true → s.getScalar j ≤ s.getScalar i_star - γ) :
    (∀ j ≠ i_star,
      (Attention.rowWeights s mask).getScalar j ≤
        Real.exp (-γ) * (Attention.rowWeights s mask).getScalar i_star) ∧
    1 - (Attention.rowWeights s mask).getScalar i_star ≤
      ((n - 1 : ℕ) : ℝ) * Real.exp (-γ) ∧
    1 - (Attention.rowWeights s mask).getScalar i_star ≤
      (n : ℝ) * Real.exp (-γ) := by
  have hpair : ∀ j ≠ i_star,
      (Attention.rowWeights s mask).getScalar j ≤
        Real.exp (-γ) * (Attention.rowWeights s mask).getScalar i_star :=
    fun j hj => weight_le_exp_neg_margin_mul_dominant s mask i_star j γ hi_star hj hmargin
  have htail_eq := tail_mass_eq s mask i_star ⟨i_star, hi_star⟩
  have hdom_le_one := dominant_weight_le_one s mask i_star hi_star
  have htail_bound :
      1 - (Attention.rowWeights s mask).getScalar i_star ≤
        ((n - 1 : ℕ) : ℝ) * Real.exp (-γ) := by
    rw [← htail_eq]
    calc
      ∑ j ∈ Finset.univ.erase i_star, (Attention.rowWeights s mask).getScalar j ≤
          ∑ _j ∈ Finset.univ.erase i_star,
            (Real.exp (-γ) * (Attention.rowWeights s mask).getScalar i_star) := by
        refine Finset.sum_le_sum fun j hj => ?_
        exact hpair j (Finset.ne_of_mem_erase hj)
      _ = ((Finset.univ.erase i_star).card : ℝ) *
            (Real.exp (-γ) * (Attention.rowWeights s mask).getScalar i_star) := by
        simp [Finset.sum_const, nsmul_eq_mul]
      _ = ((n - 1 : ℕ) : ℝ) *
            (Real.exp (-γ) * (Attention.rowWeights s mask).getScalar i_star) := by
        rw [Finset.card_erase_of_mem (Finset.mem_univ i_star), Finset.card_univ, Fintype.card_fin]
      _ ≤ ((n - 1 : ℕ) : ℝ) * (Real.exp (-γ) * 1) := by
        refine mul_le_mul_of_nonneg_left ?_ (Nat.cast_nonneg _)
        exact mul_le_mul_of_nonneg_left hdom_le_one (Real.exp_pos (-γ)).le
      _ = ((n - 1 : ℕ) : ℝ) * Real.exp (-γ) := by ring
  have hn_bound :
      ((n - 1 : ℕ) : ℝ) * Real.exp (-γ) ≤ (n : ℝ) * Real.exp (-γ) := by
    refine mul_le_mul_of_nonneg_right ?_ (Real.exp_pos (-γ)).le
    exact_mod_cast Nat.sub_le n 1
  exact ⟨hpair, htail_bound, htail_bound.trans hn_bound⟩

/-- **Logit Margin Output Stability**:
If key `i_star` is unmasked and has a logit margin `γ` over all other unmasked keys, and
all unmasked keys satisfy `‖v j - v i_star‖ ≤ D_V`, then `Attention.rowOutput s mask v`
is within `((n - 1 : ℕ) : ℝ) * Real.exp (-γ) * D_V` (and within `(n : ℝ) * Real.exp (-γ) * D_V`)
of the dominant value `v i_star`. -/
theorem logit_margin_output_stability {n : ℕ}
    (s : Tensor ℝ [n]) (mask : Tensor Bool [n])
    (v : Fin n → E) (i_star : Fin n) (γ D_V : ℝ)
    (hi_star : mask.getScalar i_star = true)
    (hmargin : ∀ j ≠ i_star, mask.getScalar j = true → s.getScalar j ≤ s.getScalar i_star - γ)
    (hD_nonneg : 0 ≤ D_V)
    (hD : ∀ j, mask.getScalar j = true → ‖v j - v i_star‖ ≤ D_V) :
    ‖Attention.rowOutput s mask v - v i_star‖ ≤
      ((n - 1 : ℕ) : ℝ) * Real.exp (-γ) * D_V ∧
    ‖Attention.rowOutput s mask v - v i_star‖ ≤
      (n : ℝ) * Real.exp (-γ) * D_V := by
  have hcert := logit_margin_implies_focused s mask i_star γ hi_star hmargin
  have hfocus_n1 :
      (Attention.rowWeights s mask).getScalar i_star ≥
        1 - ((n - 1 : ℕ) : ℝ) * Real.exp (-γ) := by
    linarith [hcert.2.1]
  have hbound_n1 := focused_rowOutput_approx_dominant s mask v i_star
    (((n - 1 : ℕ) : ℝ) * Real.exp (-γ)) D_V ⟨i_star, hi_star⟩ hfocus_n1 hD_nonneg hD
  have hbound_n :
      ((n - 1 : ℕ) : ℝ) * Real.exp (-γ) * D_V ≤ (n : ℝ) * Real.exp (-γ) * D_V := by
    refine mul_le_mul_of_nonneg_right ?_ hD_nonneg
    refine mul_le_mul_of_nonneg_right ?_ (Real.exp_pos (-γ)).le
    exact_mod_cast Nat.sub_le n 1
  exact ⟨hbound_n1, hbound_n1.trans hbound_n⟩

end OpenTheory.AttentionLogitStability
