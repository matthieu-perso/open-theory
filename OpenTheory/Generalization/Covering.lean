import OpenTheory.Generalization.FiniteLearning
import Mathlib.Topology.MetricSpace.Pseudo.Basic

/-! Continuous classes via finite uniform loss covers. Concentration is reused
from StatLean; the only new statistical step is deterministic cover transfer.
All probability bounds are on outer measures of bad events. -/
noncomputable section
open MeasureTheory StatLean.StatisticalLearning
open scoped BigOperators

namespace OpenTheory.Covering

theorem risk_close {Z : Type*} [MeasurableSpace Z] (P : Population Z)
    {f g : Z → ℝ} {ρ : ℝ} (hf : Integrable f P.law) (hg : Integrable g P.law)
    (h : ∀ z, |f z - g z| ≤ ρ) : |P.risk f - P.risk g| ≤ ρ := by
  have hb := norm_integral_le_of_norm_le_const (μ := P.law)
    (f := fun z => f z - g z)
    (Filter.Eventually.of_forall fun z => by simpa [Real.norm_eq_abs] using h z)
  simpa [Population.risk, integral_sub hf hg, Real.norm_eq_abs] using hb

theorem empirical_close {Z H : Type*} {n : ℕ} (hn : 1 ≤ n)
    (loss : H → Z → ℝ) (s : Fin n → Z) {h g : H} {ρ : ℝ}
    (hc : ∀ z, |loss h z - loss g z| ≤ ρ) :
    |empRisk loss s h - empRisk loss s g| ≤ ρ := by
  have hn' : (n : ℝ) ≠ 0 := by exact_mod_cast (Nat.ne_of_gt hn)
  have hs : |∑ i, (loss h (s i) - loss g (s i))| ≤ (n : ℝ) * ρ := by
    calc
      _ ≤ ∑ i, |loss h (s i) - loss g (s i)| := Finset.abs_sum_le_sum_abs _ _
      _ ≤ ∑ _ : Fin n, ρ := Finset.sum_le_sum fun i _ => hc (s i)
      _ = _ := by simp
  unfold empRisk
  rw [← mul_sub, ← Finset.sum_sub_distrib, abs_mul,
    abs_of_nonneg (inv_nonneg.mpr (Nat.cast_nonneg n))]
  calc
    _ ≤ (n : ℝ)⁻¹ * ((n : ℝ) * ρ) :=
      mul_le_mul_of_nonneg_left hs (inv_nonneg.mpr (Nat.cast_nonneg n))
    _ = ρ := by field_simp

/-- A finite uniform loss cover controls the entire (possibly uncountable)
class. The cover and class are fixed independently of the IID sample. -/
theorem uniform_deviation_tail {Z H : Type*} [MeasurableSpace Z]
    (P : Population Z) {n : ℕ} (K : Set H) (models : Finset H)
    (loss : H → Z → ℝ) (hmodels : ∀ g ∈ models, g ∈ K)
    (hrange : ∀ h ∈ K, ∀ z, loss h z ∈ Set.Icc (0 : ℝ) 1)
    (hmeas : ∀ h ∈ K, Measurable (loss h)) (ρ ε : ℝ)
    (hcover : ∀ h ∈ K, ∃ g ∈ models, ∀ z, |loss h z - loss g z| ≤ ρ)
    (hn : 1 ≤ n) (hε : 0 < ε) :
    sampleLaw P.law n {s | ∃ h ∈ K,
      ε + 2 * ρ < |empRisk loss s h - P.risk (loss h)|} ≤
      ENNReal.ofReal (2 * models.card * Real.exp (-2 * n * ε ^ 2)) := by
  have hi : ∀ h ∈ K, Integrable (loss h) P.law := by
    intro h hh
    exact P.bounded_loss_integrable loss h 1 (hmeas h hh) fun z =>
      by rw [abs_of_nonneg (hrange h hh z).1]; exact (hrange h hh z).2
  apply le_trans (measure_mono ?_)
    (finiteClass_measure_uniformDeviation_le (D := P.law) models
      (fun g hg => hrange g (hmodels g hg))
      (fun g hg => hmeas g (hmodels g hg)) hn hε)
  intro s hs
  obtain ⟨h, hh, hbad⟩ := hs
  obtain ⟨g, hg, hc⟩ := hcover h hh
  refine ⟨g, hg, ?_⟩
  by_contra hnot
  have hd := abs_le.mp (le_of_not_gt hnot)
  have he := abs_le.mp (empirical_close hn loss s hc)
  have hp := abs_le.mp (risk_close P (hi h hh) (hi g (hmodels g hg)) hc)
  have hb : |empRisk loss s h - P.risk (loss h)| ≤ ε + 2 * ρ := by
    apply abs_le.mpr
    change -ε ≤ empRisk loss s g - P.risk (loss g) ∧
      empRisk loss s g - P.risk (loss g) ≤ ε at hd
    constructor <;> linarith
  exact (not_lt_of_ge hb) hbad

/-- Mathlib compactness supplies the finite parameter net; a uniform
parameter-to-loss Lipschitz bound turns it into a loss cover. No cardinality
estimate is asserted here. -/
theorem compact_loss_cover {H Z : Type*} [PseudoMetricSpace H]
    (K : Set H) (hK : IsCompact K) (loss : H → Z → ℝ)
    (L : ℝ) (hL : 0 ≤ L)
    (hlip : ∀ h ∈ K, ∀ g ∈ K, ∀ z, |loss h z - loss g z| ≤ L * dist h g)
    (δ : ℝ) (hδ : 0 < δ) :
    ∃ models : Finset H, (∀ g ∈ models, g ∈ K) ∧
      ∀ h ∈ K, ∃ g ∈ models, ∀ z, |loss h z - loss g z| ≤ L * δ := by
  classical
  obtain ⟨t, htK, ht, hcover⟩ := hK.finite_cover_balls hδ
  refine ⟨ht.toFinset, ?_, ?_⟩
  · intro g hg
    exact htK (by simpa using hg)
  · intro h hh
    have hm := hcover hh
    simp only [Set.mem_iUnion] at hm
    obtain ⟨g, hg, hdist⟩ := hm
    refine ⟨g, by simpa using hg, fun z => ?_⟩
    exact (hlip h hh g (htK hg) z).trans
      (mul_le_mul_of_nonneg_left (le_of_lt hdist) hL)

/-- Any learner returning a member of the covered class inherits an oracle
inequality when it supplies an approximate empirical minimization guarantee.
This does not claim that a nonconvex optimizer supplies that guarantee. -/
theorem approx_learner_tail {Z H : Type*} [MeasurableSpace Z]
    (P : Population Z) {n : ℕ} (K : Set H) (models : Finset H)
    (loss : H → Z → ℝ) (hmodels : ∀ g ∈ models, g ∈ K)
    (hrange : ∀ h ∈ K, ∀ z, loss h z ∈ Set.Icc (0 : ℝ) 1)
    (hmeas : ∀ h ∈ K, Measurable (loss h)) (ρ ε η : ℝ)
    (hcover : ∀ h ∈ K, ∃ g ∈ models, ∀ z, |loss h z - loss g z| ≤ ρ)
    (learner : (Fin n → Z) → H)
    (htrain : ∀ s, StatLean.StatisticalLearning.IsApproxERM K loss s η (learner s))
    (hn : 1 ≤ n) (hε : 0 < ε) :
    sampleLaw P.law n {s | ∃ h ∈ K,
      P.risk (loss h) + 2 * ε + 4 * ρ + η < P.risk (loss (learner s))} ≤
      ENNReal.ofReal (2 * models.card * Real.exp (-2 * n * ε ^ 2)) := by
  apply le_trans (measure_mono ?_)
    (uniform_deviation_tail P K models loss hmodels hrange hmeas ρ ε hcover hn hε)
  intro s hs
  obtain ⟨h, hh, hbad⟩ := hs
  by_contra hnone
  have hdev : ∀ g ∈ K, |empRisk loss s g - P.risk (loss g)| ≤ ε + 2 * ρ := by
    simpa only [Set.mem_ofPred_eq, not_exists, not_and, not_lt] using hnone
  have ha := abs_le.mp (hdev (learner s) (htrain s).1)
  have hb := abs_le.mp (hdev h hh)
  have ho := (htrain s).2 h hh
  linarith

end OpenTheory.Covering
