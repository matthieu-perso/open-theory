import OpenTheory.Generalization.MLPFixedDesign

open OpenTheory

-- The multi-row affine predictor must retain all initial outputs at zero displacement.
example {d m n : ℕ} (net : TwoLayerNet d m) (X : Fin n → Fin d → ℝ) :
    MLPFixedDesign.predict net X 0 = MLPFixedDesign.initialOutput net X := by
  simp [MLPFixedDesign.predict]

-- Every row agrees with the original matrix-indexed Taylor model.
example {d m n : ℕ} (net : TwoLayerNet d m) (X : Fin n → Fin d → ℝ)
    (U : Matrix (Fin m) (Fin d) ℝ) :
    MLPFixedDesign.predict net X (MLPFixedDesign.flatten U) =
      WithLp.toLp 2 (fun r => MLPTrajectory.linearizedForward net U (X r)) := by
  ext r
  exact MLPFixedDesign.predict_flatten net X U r

#check MLPFixedDesign.predict_flatten
#check MLPFixedDesign.ridge_expected_mse
#check MLPFixedDesign.gd_to_ridge
