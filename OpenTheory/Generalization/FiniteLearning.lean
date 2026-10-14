import OpenTheory.Generalization.FiniteClass
import Mathlib.Data.Set.Finite.Lemmas

/-! Statistical learning by exact selection from a fixed finite family.
Selection uses classical choice over real losses; this is mathematical training
semantics, not an executable floating-point search algorithm. -/
noncomputable section
open MeasureTheory

namespace OpenTheory.FiniteLearning

open StatLean.StatisticalLearning

/-- Select an empirical minimizer. Existence is proved from finiteness. -/
def select {Z H : Type*} {n : ℕ} (models : Finset H) (hne : models.Nonempty)
    (loss : H → Z → ℝ) (s : Fin n → Z) : H :=
  Classical.choose (models.exists_min_image (empRisk loss s) hne)

theorem select_isERM {Z H : Type*} {n : ℕ} (models : Finset H) (hne : models.Nonempty)
    (loss : H → Z → ℝ) (s : Fin n → Z) :
    StatLean.StatisticalLearning.IsERM (↑models : Set H) loss s (select models hne loss s) :=
  Classical.choose_spec (models.exists_min_image (empRisk loss s) hne)

/-- Any approximate selector in the fixed models has a quantitative excess-risk
tail. Outer probability of the bad event is bounded, so no measurability of the
selector is silently assumed. Loss measurability and boundedness are explicit. -/
theorem approx_select_tail {Z H : Type*} [MeasurableSpace Z] (P : Population Z)
    {n : ℕ} (models : Finset H) (loss : H → Z → ℝ)
    (hrange : ∀ h ∈ models, ∀ z, loss h z ∈ Set.Icc (0 : ℝ) 1)
    (hmeas : ∀ h ∈ models, Measurable (loss h))
    (learner : (Fin n → Z) → H) (η ε : ℝ)
    (htrain : ∀ s, StatLean.StatisticalLearning.IsApproxERM
      (↑models : Set H) loss s η (learner s))
    (hn : 1 ≤ n) (hε : 0 < ε) :
    sampleLaw P.law n {s | ∃ h ∈ models,
      P.risk (loss h) + 2 * ε + η < P.risk (loss (learner s))} ≤
      ENNReal.ofReal (2 * models.card * Real.exp (-2 * n * ε ^ 2)) := by
  apply le_trans (measure_mono ?_)
    (finiteClass_measure_uniformDeviation_le (D := P.law) models hrange hmeas hn hε)
  intro s hs
  obtain ⟨h, hh, hbad⟩ := hs
  by_contra hnone
  have hdev : ∀ g ∈ models, |empRisk loss s g - risk P.law loss g| ≤ ε := by
    simpa only [Set.mem_setOf_eq, not_exists, not_and, not_lt] using hnone
  have ha := abs_le.mp (hdev (learner s) (htrain s).1)
  have hb := abs_le.mp (hdev h hh)
  have ho := (htrain s).2 h hh
  change risk P.law loss h + 2 * ε + η < risk P.law loss (learner s) at hbad
  linarith

/-- An actual finite empirical minimizer satisfies the tail theorem with zero
optimization error. The event compares it simultaneously with every models member. -/
theorem select_tail {Z H : Type*} [MeasurableSpace Z] (P : Population Z)
    {n : ℕ} (models : Finset H) (hne : models.Nonempty) (loss : H → Z → ℝ)
    (hrange : ∀ h ∈ models, ∀ z, loss h z ∈ Set.Icc (0 : ℝ) 1)
    (hmeas : ∀ h ∈ models, Measurable (loss h))
    (ε : ℝ) (hn : 1 ≤ n) (hε : 0 < ε) :
    sampleLaw P.law n {s | ∃ h ∈ models,
      P.risk (loss h) + 2 * ε < P.risk (loss (select models hne loss s))} ≤
      ENNReal.ofReal (2 * models.card * Real.exp (-2 * n * ε ^ 2)) := by
  have htrain : ∀ s : Fin n → Z, StatLean.StatisticalLearning.IsApproxERM
      (↑models : Set H) loss s 0 (select models hne loss s) := by
    intro s
    simpa [StatLean.StatisticalLearning.IsApproxERM, StatLean.StatisticalLearning.IsERM]
      using select_isERM models hne loss s
  simpa using approx_select_tail P models loss hrange hmeas
    (select models hne loss) 0 ε htrain hn hε

end OpenTheory.FiniteLearning
