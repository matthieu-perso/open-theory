import OpenTheory.Generalization.StatisticalBridge
import StatLean.StatisticalLearning.FiniteClass.AgnosticPAC

/-! A first high-probability learning bound, reusing StatLean's Hoeffding and
finite-union proof. The class is finite and fixed independently of the sample. -/
noncomputable section
open MeasureTheory

namespace OpenTheory.Population

/-- For a finite class of `[0,1]`-valued supervised losses, the probability
that any empirical risk differs from population risk by more than `ε` is at
most `2 |𝓗| exp (-2 n ε²)`. -/
theorem finiteClass_deviation_le {X H : Type*} [MeasurableSpace (X × ℝ)]
    (P : Population (X × ℝ)) {n : ℕ} (𝓗 : Finset H)
    (F : H → X → ℝ) (loss : ℝ → ℝ → ℝ) {ε : ℝ}
    (hrange : ∀ h ∈ 𝓗, ∀ z, supervisedLossClass F loss h z ∈ Set.Icc (0 : ℝ) 1)
    (hmeas : ∀ h ∈ 𝓗, Measurable (supervisedLossClass F loss h))
    (hn : 1 ≤ n) (hε : 0 < ε) :
    StatLean.StatisticalLearning.sampleLaw P.law n
      {D | ∃ h ∈ 𝓗, ε < |Dataset.empiricalRisk D (F h) loss -
        P.risk (supervisedLossClass F loss h)|} ≤
      ENNReal.ofReal (2 * 𝓗.card * Real.exp (-2 * n * ε ^ 2)) := by
  simpa [Dataset.empiricalRisk, StatLean.StatisticalLearning.empRisk,
    supervisedLossClass, Population.risk, StatLean.StatisticalLearning.risk, div_eq_mul_inv,
    mul_comm] using
    StatLean.StatisticalLearning.finiteClass_measure_uniformDeviation_le
      (D := P.law) (ℓ := supervisedLossClass F loss) 𝓗 hrange hmeas hn hε

end OpenTheory.Population
