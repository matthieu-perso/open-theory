import OpenTheory.Generalization.MLPBatchRisk

open OpenTheory

#check MLPBatchRisk.integrable_ridge_error_sq
#check MLPBatchRisk.batch_trained_pointwise_error_le
#check MLPBatchRisk.batch_trained_expected_mse
#check MLPBatchRisk.batch_trained_expected_mse_of_rank

-- At T = 0, the trained batch output is the initial network output on the dataset.
example {d m n : ℕ} (net : TwoLayerNet d m) (X : Fin n → Fin d → ℝ)
    (y : Parameters n) (η lam : ℝ) :
    MLPBatchRisk.trainedBatchOutput net X y η lam 0 =
      MLPFixedDesign.initialOutput net X := by
  simp [MLPBatchRisk.trainedBatchOutput, MLPBatchTrajectory.batchOutput,
    MLPBatchTrajectory.nonlinearBatchTrajectory, MLPBatchBounds.output,
    MLPTrajectory.atDisplacement]
