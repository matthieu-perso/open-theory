import OpenTheory.Dynamics.MLPEndpoint
import OpenTheory.Optimization.FiniteRidge
import OpenTheory.Generalization.RidgeRisk

/-! Actual MLP initial-Jacobian features for multiple fixed observations.
The affine prediction correspondence is proved by reindexing the weight matrix.
Optimization and expected risk then reuse the existing ridge theorems unchanged.
These results concern the trained LINEARIZATION, not nonlinear MLP training. -/
noncomputable section
open MeasureTheory
open scoped BigOperators RealInnerProductSpace
namespace OpenTheory.MLPFixedDesign
variable {d m n : ℕ}

def flatten (U : Matrix (Fin m) (Fin d) ℝ) : Parameters (m * d) :=
  WithLp.toLp 2 fun k => U (finProdFinEquiv.symm k).1 (finProdFinEquiv.symm k).2

def features (net : TwoLayerNet d m) (x : Fin d → ℝ) : Parameters (m * d) :=
  flatten fun i j => net.W₂ 0 i * deriv net.act.σ (FeatureStep.preactivation net x i) * x j

def design (net : TwoLayerNet d m) (X : Fin n → Fin d → ℝ) :
    Parameters (m * d) →L[ℝ] Parameters n :=
  Dataset.observation (fun i => (features net (X i), (0 : ℝ)))

def initialOutput (net : TwoLayerNet d m) (X : Fin n → Fin d → ℝ) : Parameters n :=
  WithLp.toLp 2 fun i => net.forward (X i)

def predict (net : TwoLayerNet d m) (X : Fin n → Fin d → ℝ) (u : Parameters (m * d)) :
    Parameters n := initialOutput net X + design net X u

/-- The design really is the initial MLP Taylor model, with every weight
coordinate included and with the nonzero initial output retained. -/
theorem predict_flatten (net : TwoLayerNet d m) (X : Fin n → Fin d → ℝ)
    (U : Matrix (Fin m) (Fin d) ℝ) (r : Fin n) :
    predict net X (flatten U) r = MLPTrajectory.linearizedForward net U (X r) := by
  change net.forward (X r) + ⟪features net (X r), flatten U⟫ = _
  unfold MLPTrajectory.linearizedForward
  congr 1
  rw [PiLp.inner_apply]
  simp only [Real.inner_apply]
  change (∑ k : Fin (m * d),
    (net.W₂ 0 (finProdFinEquiv.symm k).1 *
      deriv net.act.σ (FeatureStep.preactivation net (X r) (finProdFinEquiv.symm k).1) *
      X r (finProdFinEquiv.symm k).2) *
      U (finProdFinEquiv.symm k).1 (finProdFinEquiv.symm k).2) = _
  rw [(finProdFinEquiv.symm : Fin (m * d) ≃ Fin m × Fin d).sum_comp
    (fun p => (net.W₂ 0 p.1 * deriv net.act.σ (FeatureStep.preactivation net (X r) p.1) *
      X r p.2) * U p.1 p.2)]
  rw [Fintype.sum_prod_type]
  apply Finset.sum_congr rfl
  intro i _
  unfold MLPTrajectory.preShift
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro j _
  ring

def ridgePrediction (net : TwoLayerNet d m) (X : Fin n → Fin d → ℝ)
    (y : Parameters n) (lam : ℝ) (hn : 0 < n) (hlam : 0 < lam) : Parameters n :=
  predict net X (RidgeRisk.ridgeSolution (design net X) (y - initialOutput net X)
    (n : ℝ)⁻¹ lam (inv_pos.mpr (Nat.cast_pos.mpr hn)) hlam)

/-- Reuse of exact-GD convergence for the actual multi-row initial-Jacobian
design, with mean half-squared loss and correctly centered labels. -/
theorem gd_to_ridge (net : TwoLayerNet d m) (X : Fin n → Fin d → ℝ)
    (y : Parameters n) (lam : ℝ) (hn : 0 < n) (hlam : 0 < lam)
    (η : ℝ) (hη : 0 < η)
    (hstep : η * (Ridge.smoothness (design net X) (n : ℝ)⁻¹ lam : ℝ) ^ 2 < 2 * lam) :
    ∃ uStar, uStar = RidgeRisk.ridgeSolution (design net X) (y - initialOutput net X)
      (n : ℝ)⁻¹ lam (inv_pos.mpr (Nat.cast_pos.mpr hn)) hlam ∧
      ∀ u T, ‖(Optim.GD.step η (gradient (Ridge.objective (design net X)
        (y - initialOutput net X) (n : ℝ)⁻¹ lam)))^[T] u - uStar‖ ^ 2 ≤
        (Optim.GD.q η lam (Ridge.smoothness (design net X) (n : ℝ)⁻¹ lam)) ^ T *
          ‖u - uStar‖ ^ 2 :=
  RidgeRisk.gd_approaches_ridgeSolution (design net X) (y - initialOutput net X)
    (n : ℝ)⁻¹ lam (inv_pos.mpr (Nat.cast_pos.mpr hn)) hlam η hη hstep

/-- Expected in-sample error of ridge-trained INITIAL features. The net and
inputs are held fixed while labels vary as the affine signal plus white noise.
This is an architecture-specific adapter, not a new concentration theorem. -/
theorem ridge_expected_mse (net : TwoLayerNet d m) (X : Fin n → Fin d → ℝ)
    (uStar : Parameters (m * d)) (lam : ℝ) (hn : 0 < n) (hlam : 0 < lam)
    {μ : Measure (Parameters n)} {σ : ℝ} [IsProbabilityMeasure μ]
    (hnoise : RidgeRisk.LabelNoise μ σ) :
    (∫ z, ‖ridgePrediction net X (predict net X uStar + z) lam hn hlam -
      predict net X uStar‖ ^ 2 ∂μ) / n ≤
      (lam / 4) * ‖uStar‖ ^ 2 + σ ^ 2 * RidgeRisk.degreesOfFreedom (design net X)
        (n : ℝ)⁻¹ lam (inv_pos.mpr (Nat.cast_pos.mpr hn)) hlam / n := by
  simpa [ridgePrediction, predict, add_assoc] using
    RidgeRisk.fixed_design_mse (design net X) uStar lam hn hlam hnoise

end OpenTheory.MLPFixedDesign
