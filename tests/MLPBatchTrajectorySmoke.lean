import OpenTheory.Dynamics.MLPBatchTrajectory

open OpenTheory

#check MLPBatchTrajectory.batch_norm_le
#check MLPBatchTrajectory.batch_shift_le
#check MLPBatchTrajectory.batch_residual_le
#check MLPBatchTrajectory.nonlinear_linear_param_gap
#check MLPBatchTrajectory.nonlinear_linear_output_gap
#check MLPBatchTrajectory.batch_trained_ridge_gap

-- Round-trip matrix-vector bijection.
example {d m : ℕ} (U : Matrix (Fin m) (Fin d) ℝ) :
    MLPBatchTrajectory.unflatten (MLPFixedDesign.flatten U) = U := by
  simp

-- With zero activation curvature (M = 0), the trajectory discrepancy vanishes
-- and only the linear optimization contraction term remains in batchGap.
example (T m n : ℕ) (η lam A L E Jnorm q S : ℝ) :
    MLPBatchTrajectory.batchGap T m n η lam A L 0 E Jnorm q S =
      Real.sqrt (q ^ T) * (S / Real.sqrt (n * lam) * Jnorm) := by
  simp [MLPBatchTrajectory.batchGap]

-- With zero activation curvature (M = 0), the nonlinear batch trajectory
-- exactly matches the linearized initial-Jacobian ridge trajectory at every step.
example {d m n : ℕ} (net : TwoLayerNet d m) (X : Fin n → Fin d → ℝ)
    (y : Parameters n) (η lam : ℝ) {A L E S : ℝ}
    (hlam : 0 < lam) (hη : 0 < η)
    (hA : 0 ≤ A) (hL : 0 ≤ L) (hE : 0 ≤ E) (hS : 0 ≤ S)
    (hdecay : |1 - η * lam| ≤ 1)
    (hstep : η * (Ridge.smoothness (MLPFixedDesign.design net X) (n : ℝ)⁻¹ lam : ℝ) ^ 2 < 2 * lam)
    (hW : ∀ i, |net.W₂ 0 i| ≤ A) (hσ : ∀ u, |deriv net.act.σ u| ≤ L)
    (hLip : ∀ a b, |deriv net.act.σ a - deriv net.act.σ b| ≤ 0 * |a - b|)
    (hX : ∀ r, FeatureStep.inputEnergy (X r) ≤ E)
    (hinit : ‖MLPFixedDesign.initialOutput net X - y‖ ≤ S) (T : ℕ) :
    MLPBatchTrajectory.batchOutput net X
      (MLPBatchTrajectory.nonlinearBatchTrajectory net X y η lam T) =
      MLPFixedDesign.predict net X
        (MLPBatchTrajectory.linearBatchTrajectory net X y η lam T) := by
  have h := MLPBatchTrajectory.nonlinear_linear_output_gap net X y η lam hlam hη
    hA hL (M := 0) (by norm_num) hE hS hdecay hstep hW hσ hLip hX hinit T
  have hzero : ‖MLPBatchTrajectory.batchOutput net X
      (MLPBatchTrajectory.nonlinearBatchTrajectory net X y η lam T) -
      MLPFixedDesign.predict net X
        (MLPBatchTrajectory.linearBatchTrajectory net X y η lam T)‖ ≤ 0 := by
    simpa using h
  exact sub_eq_zero.mp (norm_le_zero_iff.mp hzero)
