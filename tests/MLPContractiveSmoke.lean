import OpenTheory.Dynamics.MLPContractive

open OpenTheory

#check MLPContractive.nonlinearRate
#check MLPContractive.nonlinearStepConst
#check MLPContractive.uniformNormEnvelope
#check MLPContractive.uniformShiftEnvelope
#check MLPContractive.uniformResidualEnvelope
#check MLPContractive.nonlinearRate_eq_of_pos
#check MLPContractive.nonlinearRate_lt_one_of_lt_lam
#check MLPContractive.nonlinearBatchStep_norm_le
#check MLPContractive.contractive_norm_le
#check MLPContractive.contractive_shift_le
#check MLPContractive.contractive_residual_le
#check MLPContractive.accumulated_contracting_vector_defect
#check MLPContractive.linearBatchStep_contractive
#check MLPContractive.contractive_nonlinear_linear_param_gap
#check MLPContractive.contractive_nonlinear_linear_output_gap
#check MLPContractive.uniformBatchGap
#check MLPContractive.contractive_batch_trained_ridge_gap
#check MLPContractive.contractive_batch_pointwise_error_le
#check MLPContractive.contractive_batch_expected_mse
#check MLPContractive.contractive_batch_expected_mse_of_rank

-- With zero activation curvature (M = 0), the horizon-free trajectory discrepancy vanishes
-- and only the geometrically decaying optimization term remains in uniformBatchGap.
example (T m n : ℕ) (η lam A L E Jnorm q S : ℝ) :
    MLPContractive.uniformBatchGap T m n η lam A L 0 E Jnorm q S =
      Real.sqrt q ^ T * (S / Real.sqrt (n * lam) * Jnorm) := by
  simp [MLPContractive.uniformBatchGap]

-- With zero activation curvature (M = 0), the contractive nonlinear batch trajectory
-- exactly matches the linearized initial-Jacobian ridge trajectory at every step.
example {d m n : ℕ} (net : TwoLayerNet d m) (X : Fin n → Fin d → ℝ)
    (y : Parameters n) (η lam : ℝ) {A L E S : ℝ}
    (hlam : 0 < lam) (hη : 0 < η)
    (hA : 0 ≤ A) (hL : 0 ≤ L) (hE : 0 ≤ E) (hS : 0 ≤ S)
    (hq_nl : MLPContractive.nonlinearRate m n η lam A L E < 1)
    (hstep : η * (Ridge.smoothness (MLPFixedDesign.design net X) (n : ℝ)⁻¹ lam : ℝ) ^ 2 < 2 * lam)
    (hW : ∀ i, |net.W₂ 0 i| ≤ A) (hσ : ∀ u, |deriv net.act.σ u| ≤ L)
    (hLip : ∀ a b, |deriv net.act.σ a - deriv net.act.σ b| ≤ 0 * |a - b|)
    (hX : ∀ r, FeatureStep.inputEnergy (X r) ≤ E)
    (hinit : ‖MLPFixedDesign.initialOutput net X - y‖ ≤ S) (T : ℕ) :
    MLPBatchTrajectory.batchOutput net X
      (MLPBatchTrajectory.nonlinearBatchTrajectory net X y η lam T) =
      MLPFixedDesign.predict net X
        (MLPBatchTrajectory.linearBatchTrajectory net X y η lam T) := by
  have h := MLPContractive.contractive_nonlinear_linear_output_gap net X y η lam hlam hη
    hA hL (M := 0) (by norm_num) hE hS hq_nl hstep hW hσ hLip hX hinit T
  have hzero : ‖MLPBatchTrajectory.batchOutput net X
      (MLPBatchTrajectory.nonlinearBatchTrajectory net X y η lam T) -
      MLPFixedDesign.predict net X
        (MLPBatchTrajectory.linearBatchTrajectory net X y η lam T)‖ ≤ 0 := by
    simpa using h
  exact sub_eq_zero.mp (norm_le_zero_iff.mp hzero)
