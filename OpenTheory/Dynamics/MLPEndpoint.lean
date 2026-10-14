import OpenTheory.Dynamics.MLPComparison
import OpenTheory.Optimization.Ridge
import Mathlib.Analysis.SpecificLimits.Basic

/-! The centered ridge endpoint for the one-example initial-Jacobian model.
Its design uses the actual initial W₁ derivatives, indexed by neuron/input.
Ridge optimality is reused from `Ridge.stationary_minimizes`. The scalar
prediction recurrence admits an exact geometric error formula. Combining it
with the nonlinear comparison gives a deterministic network-to-ridge bound;
this is not yet the multi-observation statistical risk theorem. -/
noncomputable section
open scoped BigOperators RealInnerProductSpace
namespace OpenTheory.MLPEndpoint
open FeatureStep MLPTrajectory MLPComparison
variable {d m : ℕ}

abbrev TangentParameters (d m : ℕ) := EuclideanSpace ℝ (Fin m × Fin d)

def jacobian (net : TwoLayerNet d m) (x : Fin d → ℝ) : TangentParameters d m :=
  WithLp.toLp 2 fun p => net.W₂ 0 p.1 * deriv net.act.σ (preactivation net x p.1) * x p.2

def design (net : TwoLayerNet d m) (x : Fin d → ℝ) : TangentParameters d m →L[ℝ] ℝ :=
  innerSL ℝ (jacobian net x)

theorem jacobian_norm_sq (net : TwoLayerNet d m) (x : Fin d → ℝ) :
    ‖jacobian net x‖ ^ 2 = kernel net x := by
  rw [EuclideanSpace.real_norm_sq_eq]
  change (∑ p : Fin m × Fin d,
    (net.W₂ 0 p.1 * deriv net.act.σ (preactivation net x p.1) * x p.2) ^ 2) = _
  rw [Fintype.sum_prod_type]
  unfold kernel inputEnergy
  apply Finset.sum_congr rfl
  intro i _
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro j _
  ring

theorem kernel_nonneg (net : TwoLayerNet d m) (x : Fin d → ℝ) : 0 ≤ kernel net x := by
  rw [← jacobian_norm_sq]
  positivity

def endpoint (net : TwoLayerNet d m) (x : Fin d → ℝ) (y lam : ℝ) : ℝ :=
  net.forward x + (y - net.forward x) / (kernel net x + lam) * kernel net x

def ridgeWeights (net : TwoLayerNet d m) (x : Fin d → ℝ) (y lam : ℝ) :
    TangentParameters d m :=
  ((y - net.forward x) / (kernel net x + lam)) • jacobian net x

theorem design_ridgeWeights (net : TwoLayerNet d m) (x : Fin d → ℝ) (y lam : ℝ) :
    design net x (ridgeWeights net x y lam) =
      (y - net.forward x) / (kernel net x + lam) * kernel net x := by
  simp [design, ridgeWeights, jacobian_norm_sq]

/-- The endpoint is the prediction of a genuine ridge minimizer on the initial
Jacobian features, with labels centered by the initial network output. -/
theorem endpoint_is_ridge_prediction (net : TwoLayerNet d m) (x : Fin d → ℝ)
    (y lam : ℝ) (hlam : 0 < lam) :
    ∃ w : TangentParameters d m,
      (∀ v, Ridge.objective (design net x) (y - net.forward x) 1 lam w ≤
        Ridge.objective (design net x) (y - net.forward x) 1 lam v) ∧
      endpoint net x y lam = net.forward x + design net x w := by
  have hn : kernel net x + lam ≠ 0 := ne_of_gt (by linarith [kernel_nonneg net x])
  have hc : (y - net.forward x) / (kernel net x + lam) * kernel net x -
      (y - net.forward x) + lam * ((y - net.forward x) / (kernel net x + lam)) = 0 := by
    field_simp
    ring
  have ha (r : ℝ) : (design net x).adjoint r = r • jacobian net x := by
    simp [design, ContinuousLinearMap.adjoint_innerSL_apply]
  have hg : Ridge.grad (design net x) (y - net.forward x) 1 lam
      (ridgeWeights net x y lam) = 0 := by
    rw [Ridge.grad, one_smul, ha, design_ridgeWeights, ridgeWeights, smul_smul,
      ← add_smul, hc, zero_smul]
  refine ⟨ridgeWeights net x y lam,
    Ridge.stationary_minimizes _ _ _ _ (by norm_num) hlam.le hg, ?_⟩
  rw [design_ridgeWeights]
  rfl

theorem step_sub_endpoint (net : TwoLayerNet d m) (x : Fin d → ℝ)
    (y η lam z : ℝ) (hlam : 0 < lam) :
    predictionStep net x y η lam z - endpoint net x y lam =
      (1 - η * (kernel net x + lam)) * (z - endpoint net x y lam) := by
  have hn : kernel net x + lam ≠ 0 := ne_of_gt (by linarith [kernel_nonneg net x])
  unfold predictionStep endpoint
  field_simp
  ring

/-- Exact finite-time prediction error; strict contraction yields convergence. -/
theorem linearPrediction_error (net : TwoLayerNet d m) (x : Fin d → ℝ)
    (y η lam : ℝ) (hlam : 0 < lam) (T : ℕ) :
    |linearPrediction net x y η lam T - endpoint net x y lam| =
      |1 - η * (kernel net x + lam)| ^ T * |net.forward x - endpoint net x y lam| := by
  induction T with
  | zero => simp [linearPrediction]
  | succ T ih =>
    have ht : linearPrediction net x y η lam (T + 1) =
        predictionStep net x y η lam (linearPrediction net x y η lam T) := by
      simp only [linearPrediction, Function.iterate_succ_apply']
    rw [ht, step_sub_endpoint net x y η lam _ hlam, abs_mul, ih, pow_succ]
    ring

theorem linearPrediction_tendsto (net : TwoLayerNet d m) (x : Fin d → ℝ)
    (y η lam : ℝ) (hlam : 0 < lam)
    (hq : |1 - η * (kernel net x + lam)| < 1) :
    Filter.Tendsto (linearPrediction net x y η lam) Filter.atTop
      (nhds (endpoint net x y lam)) := by
  apply tendsto_iff_norm_sub_tendsto_zero.mpr
  have h := (tendsto_pow_atTop_nhds_zero_of_lt_one (abs_nonneg _) hq).mul_const
    |net.forward x - endpoint net x y lam|
  simpa only [Real.norm_eq_abs, linearPrediction_error net x y η lam hlam, zero_mul] using h

/-- Explicit deterministic distance of the trained network to its centered
Jacobian-ridge prediction: nonlinear error plus a geometric optimization term. -/
theorem nonlinear_endpoint_gap (net : TwoLayerNet d m) (x : Fin d → ℝ)
    (y η lam : ℝ) {A L R M : ℝ} (hlam : 0 < lam)
    (hη : 0 ≤ η) (hA : 0 ≤ A) (hL : 0 ≤ L) (hR : 0 ≤ R) (hM : 0 ≤ M)
    (hdecay : |1 - η * lam| ≤ 1)
    (hstep : |1 - η * (kernel net x + lam)| ≤ 1)
    (hW : ∀ i, |net.W₂ 0 i| ≤ A) (hσ : ∀ u, |deriv net.act.σ u| ≤ L)
    (hLip : ∀ a b, |deriv net.act.σ a - deriv net.act.σ b| ≤ M * |a - b|)
    (T : ℕ)
    (hres : ∀ t < T, |outputResidual (atDisplacement net (trajectory net x y η lam t)) x y| ≤ R) :
    let D : ℝ := T * (η * R * A * L * inputEnergy x)
    let B : ℝ := M * m * A * D ^ 2
    let G : ℝ := m * A ^ 2 * L * M * D * inputEnergy x
    |(atDisplacement net (trajectory net x y η lam T)).forward x - endpoint net x y lam| ≤
      B + T * η * (B * |kernel net x| + R * G) +
        |1 - η * (kernel net x + lam)| ^ T * |net.forward x - endpoint net x y lam| := by
  have hc := explicit_nonlinear_linearized_gap net x y η lam hη hA hL hR hM
    hdecay hstep hW hσ hLip T hres
  have ht := abs_sub_le ((atDisplacement net (trajectory net x y η lam T)).forward x)
    (linearPrediction net x y η lam T) (endpoint net x y lam)
  rw [linearPrediction_error net x y η lam hlam] at ht
  exact ht.trans (add_le_add hc le_rfl)

end OpenTheory.MLPEndpoint
