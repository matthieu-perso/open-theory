import OpenTheory.Optimization.GradientFlow
import OpenTheory.Parameterization.MuP

namespace OpenTheory

/-- If the risk derivative is at most `-c` times the excess over `floor`,
    the excess decays as `e^{-c t}`. The rate hypothesis is the open input
    from a µP gradient-flow analysis; this statement does not prove that rate. -/
theorem risk_exponential_approach {d m n : ℕ} {sample : Sample d n}
    {path : ℝ → TwoLayerNet d m} (descent : DescentPath sample path)
    (_scale : MuPScaling d m) (floor c : ℝ) (hc : 0 < c)
    (hlower : ∀ t, floor ≤ EmpiricalRisk sample (path t))
    (hrate : ∀ t, descent.derivL t ≤ -c * (EmpiricalRisk sample (path t) - floor)) :
    ∀ t, 0 ≤ t →
      EmpiricalRisk sample (path t) ≤
        floor + Real.exp (-c * t) * (EmpiricalRisk sample (path 0) - floor) := by
  sorry

end OpenTheory
