import OpenTheory.Generalization.FiniteLearning

open MeasureTheory OpenTheory

-- Catches a bridge that swaps empirical/population risk or loses the sample law.
example {X H : Type*} [MeasurableSpace (X × ℝ)] (P : Population (X × ℝ))
    {n : ℕ} (𝓗 : Finset H) (F : H → X → ℝ) (loss : ℝ → ℝ → ℝ)
    {ε : ℝ}
    (hrange : ∀ h ∈ 𝓗, ∀ z, supervisedLossClass F loss h z ∈ Set.Icc (0 : ℝ) 1)
    (hmeas : ∀ h ∈ 𝓗, Measurable (supervisedLossClass F loss h))
    (hn : 1 ≤ n) (hε : 0 < ε) :
    StatLean.StatisticalLearning.sampleLaw P.law n
      {D | ∃ h ∈ 𝓗, ε < |Dataset.empiricalRisk D (F h) loss -
        P.risk (supervisedLossClass F loss h)|} ≤
      ENNReal.ofReal (2 * 𝓗.card * Real.exp (-2 * n * ε ^ 2)) :=
  P.finiteClass_deviation_le 𝓗 F loss hrange hmeas hn hε

-- On a singleton class, selection must return its member even for an empty sample.
example {Z H : Type*} (h : H) (loss : H → Z → ℝ) (s : Fin 0 → Z) :
    FiniteLearning.select {h} (by simp) loss s = h := by
  simpa using (FiniteLearning.select_isERM {h} (by simp) loss s).1
