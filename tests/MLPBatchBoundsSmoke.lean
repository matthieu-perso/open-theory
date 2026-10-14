import OpenTheory.Dynamics.MLPBatchBounds

open OpenTheory

#check MLPBatchBounds.output_remainder_le
#check MLPBatchBounds.design_drift_le
#check MLPBatchBounds.adjoint_residual_gap_le

-- At zero displacement every batch prediction retains its initial output.
example {d m n : ℕ} (net : TwoLayerNet d m) (X : Fin n → Fin d → ℝ) :
    MLPBatchBounds.output net X 0 = MLPFixedDesign.initialOutput net X := by
  simp [MLPBatchBounds.output, MLPTrajectory.atDisplacement]

-- Dataset-wide Taylor error must vanish for zero activation curvature.
example {d m n : ℕ} (net : TwoLayerNet d m) (X : Fin n → Fin d → ℝ)
    (U : Matrix (Fin m) (Fin d) ℝ) {A D : ℝ} (hA : 0 ≤ A) (hD : 0 ≤ D)
    (hW : ∀ i, |net.W₂ 0 i| ≤ A)
    (hLip : ∀ a b, |deriv net.act.σ a - deriv net.act.σ b| ≤ 0 * |a - b|)
    (hshift : ∀ r i, |MLPTrajectory.preShift U (X r) i| ≤ D) :
    MLPBatchBounds.output net X U = MLPFixedDesign.predict net X (MLPFixedDesign.flatten U) := by
  have h := MLPBatchBounds.output_remainder_le net X U hA (M := 0)
    (by norm_num) hD hW hLip hshift
  simpa [sub_eq_zero] using h

-- No hidden-unit displacement on the observed inputs means no design drift.
example {d m n : ℕ} (net : TwoLayerNet d m) (X : Fin n → Fin d → ℝ)
    (U : Matrix (Fin m) (Fin d) ℝ) {A M E : ℝ}
    (hA : 0 ≤ A) (hM : 0 ≤ M) (hE : 0 ≤ E)
    (hW : ∀ i, |net.W₂ 0 i| ≤ A)
    (hLip : ∀ a b, |deriv net.act.σ a - deriv net.act.σ b| ≤ M * |a - b|)
    (hshift : ∀ r i, |MLPTrajectory.preShift U (X r) i| ≤ 0)
    (hX : ∀ r, FeatureStep.inputEnergy (X r) ≤ E) :
    MLPFixedDesign.design (MLPTrajectory.atDisplacement net U) X =
      MLPFixedDesign.design net X := by
  have h := MLPBatchBounds.design_drift_le net X U hA hM (D := 0)
    (by norm_num) hE hW hLip hshift hX
  simpa [sub_eq_zero] using h
