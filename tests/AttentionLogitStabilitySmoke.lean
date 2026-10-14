import OpenTheory.Architectures.AttentionLogitStability

noncomputable section

open TorchLean OpenTheory OpenTheory.Attention OpenTheory.AttentionAblation
open OpenTheory.AttentionLogitStability
open scoped BigOperators

variable {E F : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
  [NormedAddCommGroup F] [NormedSpace ℝ F]

-- 1. L1 weight distance properties (nonnegativity, reflexivity, symmetry, triangle inequality)
example {n : ℕ} (s₁ s₂ s₃ : Tensor ℝ [n]) (mask : Tensor Bool [n]) :
    0 ≤ weightL1Diff s₁ s₂ mask ∧
    weightL1Diff s₁ s₁ mask = 0 ∧
    weightL1Diff s₁ s₂ mask = weightL1Diff s₂ s₁ mask ∧
    weightL1Diff s₁ s₃ mask ≤ weightL1Diff s₁ s₂ mask + weightL1Diff s₂ s₃ mask :=
  ⟨weightL1Diff_nonneg s₁ s₂ mask,
   weightL1Diff_self s₁ mask,
   weightL1Diff_comm s₁ s₂ mask,
   weightL1Diff_triangle s₁ s₂ s₃ mask⟩

-- 2. Raw value norm L1 score perturbation bound
example {n : ℕ} (s₁ s₂ : Tensor ℝ [n]) (mask : Tensor Bool [n])
    (v : Fin n → E) (B_V : ℝ) (hv : ∀ i, ‖v i‖ ≤ B_V) :
    ‖rowOutput s₁ mask v - rowOutput s₂ mask v‖ ≤ B_V * weightL1Diff s₁ s₂ mask :=
  rowOutput_score_perturbation_l1 s₁ s₂ mask v B_V hv

-- 3. Centered value dispersion score perturbation bound
example {n : ℕ} (s₁ s₂ : Tensor ℝ [n]) (mask : Tensor Bool [n])
    (v : Fin n → E) (v_bar : E) (Δ_V : ℝ)
    (hmask : ∃ i, mask.getScalar i = true)
    (hΔ : ∀ i, mask.getScalar i = true → ‖v i - v_bar‖ ≤ Δ_V) :
    ‖rowOutput s₁ mask v - rowOutput s₂ mask v‖ ≤ Δ_V * weightL1Diff s₁ s₂ mask :=
  rowOutput_score_perturbation_centered s₁ s₂ mask v v_bar Δ_V hmask hΔ

-- 4. Concrete constant-value invariance check: when all values equal `v₀` (so `Δ_V = 0`),
-- any score perturbation produces zero output change!
example {n : ℕ} (s₁ s₂ : Tensor ℝ [n]) (mask : Tensor Bool [n])
    (v₀ : E) (hmask : ∃ i, mask.getScalar i = true) :
    ‖rowOutput s₁ mask (fun _ => v₀) - rowOutput s₂ mask (fun _ => v₀)‖ ≤ 0 := by
  simpa using rowOutput_score_perturbation_centered s₁ s₂ mask (fun _ => v₀) v₀ 0 hmask
    (fun _ _ => by simp)

-- 5. Joint Score + Value (Q/K/V) perturbation bound for a single attention row
example {n : ℕ} (s₁ s₂ : Tensor ℝ [n]) (mask : Tensor Bool [n])
    (v₁ v₂ : Fin n → E) (v_bar : E) (Δ_V δ_V : ℝ)
    (hmask : ∃ i, mask.getScalar i = true)
    (hΔ : ∀ i, mask.getScalar i = true → ‖v₁ i - v_bar‖ ≤ Δ_V)
    (hδ : ∀ i, ‖v₁ i - v₂ i‖ ≤ δ_V) :
    ‖rowOutput s₁ mask v₁ - rowOutput s₂ mask v₂‖ ≤
      Δ_V * weightL1Diff s₁ s₂ mask + δ_V :=
  rowOutput_joint_qkv_perturbation s₁ s₂ mask v₁ v₂ v_bar Δ_V δ_V hmask hΔ hδ

-- 6. Multi-Head Joint Score + Value (Q/K/V) perturbation bound across H heads
example {H n : ℕ}
    (s₁ s₂ : Fin H → Tensor ℝ [n])
    (mask : Fin H → Tensor Bool [n])
    (v₁ v₂ : Fin H → Fin n → E)
    (v_bar : Fin H → E)
    (proj : Fin H → (E →L[ℝ] F))
    (Δ_V δ_V : Fin H → ℝ)
    (hmask : ∀ h, ∃ i, (mask h).getScalar i = true)
    (hΔ : ∀ h i, (mask h).getScalar i = true → ‖v₁ h i - v_bar h‖ ≤ Δ_V h)
    (hδ : ∀ h i, ‖v₁ h i - v₂ h i‖ ≤ δ_V h) :
    ‖multiHeadOutput s₁ mask v₁ proj - multiHeadOutput s₂ mask v₂ proj‖ ≤
      ∑ h : Fin H, ‖proj h‖ * (Δ_V h * weightL1Diff (s₁ h) (s₂ h) (mask h) + δ_V h) :=
  multiHead_joint_qkv_perturbation s₁ s₂ mask v₁ v₂ v_bar proj Δ_V δ_V hmask hΔ hδ

-- 7. Focused attention tail mass & dominant-token approximation
example {n : ℕ} (s : Tensor ℝ [n]) (mask : Tensor Bool [n])
    (v : Fin n → E) (i_star : Fin n) (ε D_V B_V : ℝ)
    (hmask : ∃ i, mask.getScalar i = true)
    (hfocus : (rowWeights s mask).getScalar i_star ≥ 1 - ε)
    (hD_nonneg : 0 ≤ D_V)
    (hD : ∀ j, mask.getScalar j = true → ‖v j - v i_star‖ ≤ D_V)
    (hB : ∀ j, ‖v j‖ ≤ B_V) :
    (∑ j ∈ Finset.univ.erase i_star, (rowWeights s mask).getScalar j =
      1 - (rowWeights s mask).getScalar i_star ∧
     ∑ j ∈ Finset.univ.erase i_star, (rowWeights s mask).getScalar j ≤ ε) ∧
    ‖rowOutput s mask v - v i_star‖ ≤ ε * D_V ∧
    ‖rowOutput s mask v - v i_star‖ ≤ 2 * ε * B_V :=
  ⟨tail_mass_eq_of_focused s mask i_star ε hmask hfocus,
   focused_rowOutput_approx_dominant s mask v i_star ε D_V hmask hfocus hD_nonneg hD,
   focused_rowOutput_approx_dominant_norm s mask v i_star ε B_V hmask hfocus hB⟩

-- 8. Focused score perturbation stability (L1 weight distance & output stability)
example {n : ℕ} (s₁ s₂ : Tensor ℝ [n]) (mask : Tensor Bool [n])
    (v : Fin n → E) (i_star : Fin n) (ε₁ ε₂ D_V : ℝ)
    (hmask : ∃ i, mask.getScalar i = true)
    (hfocus₁ : (rowWeights s₁ mask).getScalar i_star ≥ 1 - ε₁)
    (hfocus₂ : (rowWeights s₂ mask).getScalar i_star ≥ 1 - ε₂)
    (hD_nonneg : 0 ≤ D_V)
    (hD : ∀ j, mask.getScalar j = true → ‖v j - v i_star‖ ≤ D_V) :
    weightL1Diff s₁ s₂ mask ≤ 2 * (ε₁ + ε₂) ∧
    ‖rowOutput s₁ mask v - rowOutput s₂ mask v‖ ≤ (ε₁ + ε₂) * D_V :=
  focused_score_perturbation_stability s₁ s₂ mask v i_star ε₁ ε₂ D_V
    hmask hfocus₁ hfocus₂ hD_nonneg hD

-- 9. Logit margin exponential concentration certificate & output stability
example {n : ℕ} (s : Tensor ℝ [n]) (mask : Tensor Bool [n])
    (v : Fin n → E) (i_star : Fin n) (γ D_V : ℝ)
    (hi_star : mask.getScalar i_star = true)
    (hmargin : ∀ j ≠ i_star, mask.getScalar j = true → s.getScalar j ≤ s.getScalar i_star - γ)
    (hD_nonneg : 0 ≤ D_V)
    (hD : ∀ j, mask.getScalar j = true → ‖v j - v i_star‖ ≤ D_V) :
    ((∀ j ≠ i_star,
        (rowWeights s mask).getScalar j ≤
          Real.exp (-γ) * (rowWeights s mask).getScalar i_star) ∧
      1 - (rowWeights s mask).getScalar i_star ≤ ((n - 1 : ℕ) : ℝ) * Real.exp (-γ) ∧
      1 - (rowWeights s mask).getScalar i_star ≤ (n : ℝ) * Real.exp (-γ)) ∧
    (‖rowOutput s mask v - v i_star‖ ≤ ((n - 1 : ℕ) : ℝ) * Real.exp (-γ) * D_V ∧
      ‖rowOutput s mask v - v i_star‖ ≤ (n : ℝ) * Real.exp (-γ) * D_V) :=
  ⟨logit_margin_implies_focused s mask i_star γ hi_star hmargin,
   logit_margin_output_stability s mask v i_star γ D_V hi_star hmargin hD_nonneg hD⟩
