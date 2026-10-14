import Mathlib.Analysis.Calculus.MeanValue
import Mathlib.Analysis.SpecialFunctions.ExpDeriv
import OpenTheory.Optimization.GradientFlow
import OpenTheory.Parameterization.MuP

namespace OpenTheory

/-- If the risk derivative is at most `-c` times the excess over `floor`,
    the excess decays as `e^{-c t}`. The rate hypothesis is the open input
    from a µP gradient-flow analysis; this statement does not prove that rate. -/
theorem risk_exponential_approach {d m n : ℕ} {sample : Sample d n}
    {path : ℝ → TwoLayerNet d m} (descent : DescentPath sample path)
    (_scale : MuPScaling d m) (floor c : ℝ) (_hc : 0 < c)
    (_hlower : ∀ t, floor ≤ EmpiricalRisk sample (path t))
    (hrate : ∀ t, descent.derivL t ≤ -c * (EmpiricalRisk sample (path t) - floor)) :
    ∀ t, 0 ≤ t →
      EmpiricalRisk sample (path t) ≤
        floor + Real.exp (-c * t) * (EmpiricalRisk sample (path 0) - floor) := by
  intro t ht
  let g : ℝ → ℝ := fun s => Real.exp (c * s) * (EmpiricalRisk sample (path s) - floor)
  let g' : ℝ → ℝ := fun s =>
    Real.exp (c * s) * c * (EmpiricalRisk sample (path s) - floor) +
      Real.exp (c * s) * descent.derivL s
  have hg_deriv : ∀ s, HasDerivAt g (g' s) s := by
    intro s
    have hexp : HasDerivAt (fun x => Real.exp (c * x)) (Real.exp (c * s) * c) s := by
      have hlin : HasDerivAt (fun x : ℝ => c * x) c s := by
        simpa using (hasDerivAt_id s).const_mul c
      exact hlin.exp
    have hrisk : HasDerivAt (fun x => EmpiricalRisk sample (path x) - floor)
        (descent.derivL s) s :=
      (descent.hasDeriv s).sub_const floor
    exact hexp.mul hrisk
  have hg_cont : ContinuousOn g (Set.Icc 0 t) :=
    fun s _ => (hg_deriv s).continuousAt.continuousWithinAt
  have hg_right : ∀ s ∈ Set.Ico 0 t, HasDerivWithinAt g (g' s) (Set.Ici s) s :=
    fun s _ => (hg_deriv s).hasDerivWithinAt
  have hB_cont : ContinuousOn (fun _ : ℝ => g 0) (Set.Icc 0 t) := continuousOn_const
  have hB_right : ∀ s ∈ Set.Ico 0 t,
      HasDerivWithinAt (fun _ : ℝ => g 0) ((fun _ : ℝ => (0 : ℝ)) s) (Set.Ici s) s :=
    fun s _ => hasDerivWithinAt_const s (Set.Ici s) (g 0)
  have hbound : ∀ s ∈ Set.Ico 0 t, g' s ≤ 0 := by
    intro s _
    have hr := hrate s
    have hexp_pos : 0 ≤ Real.exp (c * s) := (Real.exp_pos (c * s)).le
    have hsum : c * (EmpiricalRisk sample (path s) - floor) + descent.derivL s ≤ 0 := by
      linarith
    have hprod : Real.exp (c * s) *
        (c * (EmpiricalRisk sample (path s) - floor) + descent.derivL s) ≤ 0 :=
      mul_nonpos_of_nonneg_of_nonpos hexp_pos hsum
    dsimp [g']
    linarith
  have hle : g t ≤ g 0 :=
    image_le_of_deriv_right_le_deriv_boundary (B := fun _ => g 0) (B' := fun _ => 0)
      hg_cont hg_right le_rfl hB_cont hB_right hbound ⟨ht, le_rfl⟩
  have hg0 : g 0 = EmpiricalRisk sample (path 0) - floor := by
    simp [g]
  rw [hg0] at hle
  have hmul : Real.exp (-c * t) * g t ≤
      Real.exp (-c * t) * (EmpiricalRisk sample (path 0) - floor) :=
    mul_le_mul_of_nonneg_left hle (Real.exp_pos (-c * t)).le
  have hexp_cancel : Real.exp (-c * t) * g t = EmpiricalRisk sample (path t) - floor := by
    dsimp [g]
    rw [← mul_assoc, ← Real.exp_add]
    have hzero : -c * t + c * t = 0 := by ring
    rw [hzero, Real.exp_zero, one_mul]
  rw [hexp_cancel] at hmul
  linarith

end OpenTheory
