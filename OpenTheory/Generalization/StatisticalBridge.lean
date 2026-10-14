import OpenTheory.Data.Foundation
import FoML.Learning.ERM

/-! Reuse of FoML's deterministic ERM oracle inequality. This is conditional on
an empirical optimization guarantee and a deviation bound, not a confidence theorem. -/
noncomputable section
open MeasureTheory

namespace OpenTheory

theorem Dataset.empiricalRisk_eq_foml {X H : Type*} {n : ℕ}
    (D : Dataset X ℝ n) (F : H → X → ℝ) (loss : ℝ → ℝ → ℝ) (h : H) :
    D.empiricalRisk (F h) loss =
      _root_.empiricalRisk n (supervisedLossClass F loss) D h := by
  simp [Dataset.empiricalRisk, _root_.empiricalRisk, supervisedLossClass,
    div_eq_mul_inv, mul_comm]

theorem Population.risk_eq_foml {Z H : Type*} [MeasurableSpace Z]
    (P : Population Z) (loss : H → Z → ℝ) (h : H) :
    P.risk (loss h) = populationRisk loss P.law id h := rfl

/-- Actual integrability, not just an equality between totalized integrals. -/
theorem Population.bounded_loss_integrable {Z H : Type*} [MeasurableSpace Z]
    (P : Population Z) (loss : H → Z → ℝ) (h : H) (b : ℝ)
    (hm : Measurable (loss h)) (hb : ∀ z, |loss h z| ≤ b) :
    P.IntegrableLoss (loss h) := by
  exact (integrable_const b).mono' hm.aestronglyMeasurable
    (Filter.Eventually.of_forall fun z => by simpa [Real.norm_eq_abs] using hb z)

/-- A sample-specific oracle inequality using OpenTheory risks. Optimization
and uniform deviation are explicit premises; no sampling conclusion is hidden. -/
theorem Population.supervised_excessRisk_le {X H : Type*}
    [MeasurableSpace (X × ℝ)] (P : Population (X × ℝ)) {n : ℕ}
    (D : Dataset X ℝ n) (F : H → X → ℝ) (loss : ℝ → ℝ → ℝ)
    (trained comparator : H) (η ε : ℝ)
    (hopt : ∀ h, D.empiricalRisk (F trained) loss ≤ D.empiricalRisk (F h) loss + η)
    (hdev : ∀ h, |D.empiricalRisk (F h) loss -
      P.risk (supervisedLossClass F loss h)| ≤ ε) :
    P.risk (supervisedLossClass F loss trained) -
      P.risk (supervisedLossClass F loss comparator) ≤ 2 * ε + η := by
  have ho : IsApproxERM η n (supervisedLossClass F loss) D trained := by
    intro h
    simpa only [Dataset.empiricalRisk_eq_foml] using hopt h
  have hd : ∀ h, riskDeviation n (supervisedLossClass F loss) P.law id D h ≤ ε := by
    intro h
    simpa only [Dataset.empiricalRisk_eq_foml, Population.risk_eq_foml,
      riskDeviation] using hdev h
  exact ho.excessRisk_le (hstar := comparator) hd

end OpenTheory
