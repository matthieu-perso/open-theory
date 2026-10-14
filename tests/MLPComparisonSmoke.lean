import OpenTheory.Dynamics.MLPComparison

open OpenTheory

-- A zero horizon must agree with initialization, independently of the labels.
example {d m : ℕ} (net : TwoLayerNet d m) (x : Fin d → ℝ) (y η lam : ℝ) :
    MLPComparison.linearPrediction net x y η lam 0 = net.forward x := rfl

#check MLPComparison.linearized_trajectory_prediction
#check MLPComparison.nonlinear_linearized_gap

-- Two local errors bounded by 3 accumulate to at most 6, not 3.
example (F : ℝ → ℝ) (u : ℕ → ℝ)
    (hF : ∀ a b, |F a - F b| ≤ |a - b|)
    (hdef : ∀ t < 2, |u (t + 1) - F (u t)| ≤ 3) :
    |u 2 - F^[2] (u 0)| ≤ 6 := by
  convert MLPComparison.accumulated_defect F u 3 hF 2 hdef using 1 <;> norm_num

-- Zero derivative variation makes the separately trained predictions agree.
example {d m : ℕ} (net : TwoLayerNet d m) (x : Fin d → ℝ) (y η lam : ℝ)
    {A L R : ℝ} (hη : 0 ≤ η) (hA : 0 ≤ A) (hL : 0 ≤ L) (hR : 0 ≤ R)
    (hdecay : |1 - η * lam| ≤ 1)
    (hstep : |1 - η * (MLPComparison.kernel net x + lam)| ≤ 1)
    (hW : ∀ i, |net.W₂ 0 i| ≤ A) (hσ : ∀ u, |deriv net.act.σ u| ≤ L)
    (hconstant : ∀ a b, deriv net.act.σ a = deriv net.act.σ b) (T : ℕ)
    (hres : ∀ t < T, |FeatureStep.outputResidual
      (MLPTrajectory.atDisplacement net (MLPTrajectory.trajectory net x y η lam t)) x y| ≤ R) :
    (MLPTrajectory.atDisplacement net (MLPTrajectory.trajectory net x y η lam T)).forward x =
      MLPComparison.linearPrediction net x y η lam T := by
  have h := MLPComparison.explicit_nonlinear_linearized_gap net x y η lam
    (M := 0) hη hA hL hR le_rfl hdecay hstep hW hσ
    (by intro a b; rw [hconstant a b]; simp) T hres
  simpa [sub_eq_zero] using h
