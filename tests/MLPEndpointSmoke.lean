import OpenTheory.Dynamics.MLPEndpoint

open OpenTheory

-- A zero initial Jacobian must retain the initial output, not predict zero.
example {d m : ℕ} (net : TwoLayerNet d m) (x : Fin d → ℝ) (y lam : ℝ)
    (hK : MLPComparison.kernel net x = 0) :
    MLPEndpoint.endpoint net x y lam = net.forward x := by
  simp [MLPEndpoint.endpoint, hK]

#check MLPEndpoint.endpoint_is_ridge_prediction
#check MLPEndpoint.linearPrediction_error
#check MLPEndpoint.nonlinear_endpoint_gap

-- Nonzero initial output catches accidental uncentered ridge regression.
example {d m : ℕ} (net : TwoLayerNet d m) (x : Fin d → ℝ)
    (hK : MLPComparison.kernel net x = 3) (hf : net.forward x = 2) :
    MLPEndpoint.endpoint net x 10 1 = 8 := by
  norm_num [MLPEndpoint.endpoint, hK, hf]

-- q=1/2, T=2, initial error=6: the remaining prediction error is 3/2.
example {d m : ℕ} (net : TwoLayerNet d m) (x : Fin d → ℝ)
    (hK : MLPComparison.kernel net x = 3) (hf : net.forward x = 2) :
    |MLPComparison.linearPrediction net x 10 (1 / 8) 1 2 -
      MLPEndpoint.endpoint net x 10 1| = (3 : ℝ) / 2 := by
  rw [MLPEndpoint.linearPrediction_error net x 10 (1 / 8) 1 (by norm_num) 2]
  norm_num [MLPEndpoint.endpoint, hK, hf]
