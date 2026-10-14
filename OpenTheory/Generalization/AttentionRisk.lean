import OpenTheory.Architectures.Attention
import OpenTheory.Generalization.PopulationBounds

/-! Population bounds for an attention row, with arbitrary data-dependent scores and masks.
These bound risk itself, not the empirical-to-population generalization gap. -/
noncomputable section
open MeasureTheory TorchLean

namespace OpenTheory.Population

theorem attentionRisk_le {Z E : Type*} [MeasurableSpace Z]
    [NormedAddCommGroup E] [NormedSpace ℝ E] {n : ℕ} (P : Population Z)
    (scores : Z → Tensor ℝ [n]) (mask : Z → Tensor Bool [n])
    (values : Z → Fin n → E) (label : Z → E) (B T : ℝ)
    (hmeas : AEStronglyMeasurable
      (fun z => ‖Attention.rowOutput (scores z) (mask z) (values z) - label z‖ ^ 2) P.law)
    (h : ∀ᵐ z ∂P.law, (∃ i, (mask z).getScalar i = true) ∧
      (∀ i, ‖values z i‖ ≤ B) ∧ ‖label z‖ ≤ T) :
    P.IntegrableLoss (fun z => ‖Attention.rowOutput (scores z) (mask z) (values z) - label z‖ ^ 2) ∧
      P.risk (fun z => ‖Attention.rowOutput (scores z) (mask z) (values z) - label z‖ ^ 2) ≤
        (B + T) ^ 2 := by
  obtain ⟨hi, hb⟩ := P.risk_le_envelope _ (fun _ => (B + T) ^ 2) hmeas
    (integrable_const _) (h.mono fun z hz => by
      refine ⟨sq_nonneg _, ?_⟩
      have ho := Attention.rowOutput_norm_le (scores z) (mask z) (values z) hz.1 B hz.2.1
      have he : ‖Attention.rowOutput (scores z) (mask z) (values z) - label z‖ ≤ B + T :=
        (norm_sub_le _ _).trans (add_le_add ho hz.2.2)
      exact pow_le_pow_left₀ (norm_nonneg _) he 2)
  exact ⟨hi, by simpa [Population.risk] using hb⟩

end OpenTheory.Population
