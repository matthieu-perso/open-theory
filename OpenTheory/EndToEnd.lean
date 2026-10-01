import Mathlib.Data.Real.Basic
import Mathlib.Tactic.Linarith

namespace OpenTheory

/-- The three error budgets of an end-to-end guarantee. -/
structure EndToEndError where
  ε_approx : ℝ
  ε_opt : ℝ
  ε_gen : ℝ
  approx_nonneg : 0 ≤ ε_approx
  opt_nonneg : 0 ≤ ε_opt
  gen_nonneg : 0 ≤ ε_gen

/-- Population-risk budget `ε_approx + ε_opt + ε_gen`. -/
def populationRiskBound (e : EndToEndError) : ℝ :=
  e.ε_approx + e.ε_opt + e.ε_gen

/-- If empirical risk is within `ε_opt` of the approximation error and the
    generalization gap is at most `ε_gen`, population risk is at most the sum. -/
theorem population_risk_le_sum (e : EndToEndError) (population empirical : ℝ)
    (h_fit : empirical ≤ e.ε_approx + e.ε_opt)
    (h_gen : population ≤ empirical + e.ε_gen) :
    population ≤ populationRiskBound e := by
  unfold populationRiskBound
  linarith

end OpenTheory
