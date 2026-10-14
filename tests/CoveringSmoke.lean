import OpenTheory.Generalization.Covering
import OpenTheory.Optimization.FiniteRidge

open MeasureTheory StatLean.StatisticalLearning

-- Removing the cover transfer or its 2ρ tolerance must break these checks.
#check OpenTheory.Covering.uniform_deviation_tail
#check OpenTheory.Covering.compact_loss_cover
#check OpenTheory.Ridge.training_radius
#check OpenTheory.Dataset.ridge_training_radius

-- Fixing ρ=1/10 and ε=1/5 gives deviation threshold 2/5, not 3/10.
example {Z H : Type*} [MeasurableSpace Z] (P : OpenTheory.Population Z)
    {n : ℕ} (K : Set H) (models : Finset H) (loss : H → Z → ℝ)
    (hm : ∀ g ∈ models, g ∈ K)
    (hr : ∀ h ∈ K, ∀ z, loss h z ∈ Set.Icc (0 : ℝ) 1)
    (hmeas : ∀ h ∈ K, Measurable (loss h))
    (hc : ∀ h ∈ K, ∃ g ∈ models, ∀ z, |loss h z - loss g z| ≤ 1 / 10)
    (hn : 1 ≤ n) :
    sampleLaw P.law n {s | ∃ h ∈ K,
      (2 : ℝ) / 5 < |empRisk loss s h - P.risk (loss h)|} ≤
      ENNReal.ofReal (2 * models.card * Real.exp (-2 * n * ((1 : ℝ) / 5) ^ 2)) := by
  convert OpenTheory.Covering.uniform_deviation_tail P K models loss hm hr hmeas
    (1 / 10) (1 / 5) hc hn (by norm_num) using 1 <;> norm_num

example {Z : Type*} [MeasurableSpace Z] (P : OpenTheory.Population Z)
    (f g : Z → ℝ) (hf : Integrable f P.law) (hg : Integrable g P.law)
    (h : ∀ z, |f z - g z| ≤ 0) : |P.risk f - P.risk g| ≤ 0 :=
  OpenTheory.Covering.risk_close P hf hg h

-- A genuine continuous parameter interval, with a positive cover tolerance.
example : ∃ models : Finset ℝ, (∀ g ∈ models, g ∈ Set.Icc (0 : ℝ) 1) ∧
    ∀ h ∈ Set.Icc (0 : ℝ) 1, ∃ g ∈ models,
      ∀ _ : Unit, |h - g| ≤ (1 : ℝ) / 10 := by
  simpa using OpenTheory.Covering.compact_loss_cover (Set.Icc (0 : ℝ) 1)
    isCompact_Icc (fun h (_ : Unit) => h) 1 (by norm_num)
    (by intro h _ g _ z; simp [Real.dist_eq]) ((1 : ℝ) / 10) (by norm_num)

-- Bounded labels, λ=1 and R=1 yield the literal iterate radius 2.
example {d n : ℕ} (D : OpenTheory.Dataset (OpenTheory.Parameters d) ℝ n)
    (hn : 0 < n) (hy : ∀ i, |(D i).2| ≤ 1) (η : ℝ) (hη : 0 < η)
    (hstep : η * (OpenTheory.Ridge.smoothness D.observation (1 / n) 1 : ℝ) ^ 2 < 2) :
    ∀ k, ‖(Optim.GD.step η (gradient (D.ridgeLoss 1)))^[k] 0‖ ≤ 2 := by
  obtain ⟨_, _, _, h⟩ := D.ridge_training_radius hn 1 (by norm_num) hy
    1 1 (by norm_num) (by norm_num) (by norm_num) η hη (by simpa using hstep)
  simpa using h
