import OpenTheory.Dynamics.MLPBatchTrajectory
import OpenTheory.Generalization.PredictionGap

/-!
# Multi-Observation Statistical Risk of Nonlinearly Trained Two-Layer MLPs

This module combines the multi-input nonlinear trajectory distance theorem
`MLPBatchTrajectory.batch_trained_ridge_gap` with the fixed-design ridge risk
bound `MLPFixedDesign.ridge_expected_mse` and the prediction perturbation
inequality `PredictionGap.sq_norm_le_double`.

## What is proved

For an arbitrary dataset `X : Fin n → Fin d → ℝ` with `‖X r‖₂² ≤ E`, let the
ground-truth signal be the affine initial-feature predictor
`predict net X uStar = F₀ + J₀ uStar`, and let observed labels be `y(z) = predict net X uStar + z`
where `z` is white label noise of variance `σ²` (`RidgeRisk.LabelNoise μ σ`).

1. **Pointwise excess squared error (`batch_trained_pointwise_error_le`)**:
   Whenever the noise realization satisfies `‖z‖ ≤ Bz` (which gives
   `‖F₀ - y(z)‖ ≤ ‖J₀ uStar‖ + Bz`), the actual `T`-step nonlinear batch gradient descent
   predictor `trainedBatchOutput net X (y z) η lam T` satisfies
   ```
   ‖trainedBatchOutput ... - predict net X uStar‖²
     ≤ 2 ‖ridgePrediction ... - predict net X uStar‖² + 2 δ²
   ```
   where `δ = batchGap T m n η lam A L M E ‖J₀‖ q S` is the closed trajectory
   discrepancy + optimization bound from `MLPBatchTrajectory`.

2. **Expected in-sample MSE under bounded noise (`batch_trained_expected_mse`)**:
   Under almost-surely bounded noise `‖z‖ ≤ Bz` (so `‖F₀ - y(z)‖ ≤ ‖J₀ uStar‖ + Bz =: S`
   for a.e. `z`), the expected in-sample mean squared error over `n` observations is at most
   ```
   2 * ((λ / 4) ‖uStar‖² + σ² · df(J₀, λ) / n) + 2 δ² / n
   ```
   and therefore also at most
   ```
   2 * ((λ / 4) ‖uStar‖² + σ² · rank(J₀) / n) + 2 δ² / n.
   ```
   Notice that no unproved trajectory hypotheses remain: preactivation shifts,
   parameter norms, and output residuals along all `T` nonlinear steps are
   automatically bounded by `S = ‖J₀ uStar‖ + Bz`.
-/

noncomputable section

open MeasureTheory
open scoped BigOperators RealInnerProductSpace

namespace OpenTheory.MLPBatchRisk

open OpenTheory.FeatureStep
open OpenTheory.MLPFixedDesign
open OpenTheory.MLPBatchTrajectory

variable {d m n : ℕ}

/-- Output vector on `X` of the `T`-step nonlinear batch-trained two-layer network. -/
def trainedBatchOutput (net : TwoLayerNet d m) (X : Fin n → Fin d → ℝ)
    (y : Parameters n) (η lam : ℝ) (T : ℕ) : Parameters n :=
  batchOutput net X (nonlinearBatchTrajectory net X y η lam T)

/-- Ridge error integrability for white label noise: the squared error of the
ridge predictor around `predict net X uStar` is Bochner integrable. -/
theorem integrable_ridge_error_sq (net : TwoLayerNet d m) (X : Fin n → Fin d → ℝ)
    (uStar : Parameters (m * d)) (lam : ℝ) (hn : 0 < n) (hlam : 0 < lam)
    {μ : Measure (Parameters n)} {σ : ℝ} [IsProbabilityMeasure μ]
    (hnoise : RidgeRisk.LabelNoise μ σ) :
    Integrable (fun z =>
      ‖ridgePrediction net X (predict net X uStar + z) lam hn hlam -
        predict net X uStar‖ ^ 2) μ := by
  let A := design net X
  let hc : 0 < (n : ℝ)⁻¹ := inv_pos.mpr (Nat.cast_pos.mpr hn)
  let H := RidgeRisk.hat A (n : ℝ)⁻¹ lam hc hlam
  let bias : Parameters n := - (lam • A (RidgeRisk.hessianInv A (n : ℝ)⁻¹ lam hc.le hlam uStar))
  have herr : ∀ z, ridgePrediction net X (predict net X uStar + z) lam hn hlam -
      predict net X uStar = bias + H z := by
    intro z
    have hsub : predict net X uStar + z - initialOutput net X = A uStar + z := by
      simp only [predict, A]
      abel
    change (initialOutput net X + A (RidgeRisk.ridgeSolution A
        (predict net X uStar + z - initialOutput net X) (n : ℝ)⁻¹ lam hc hlam)) -
      (initialOutput net X + A uStar) = bias + H z
    rw [hsub, add_sub_add_left_eq_sub]
    simpa [bias, H, sub_eq_add_neg] using
      RidgeRisk.prediction_decomp A uStar z (n : ℝ)⁻¹ lam hc hlam
  have hconst : Integrable (fun _ : Parameters n => ‖bias‖ ^ 2) μ :=
    integrable_const _
  have hvar_int : Integrable (fun z => ‖H z‖ ^ 2) μ := by
    have hsq : (fun z => ‖H z‖ ^ 2) = fun z =>
        ∑ i, ⟪((RidgeRisk.hat_isSymmetric A (n : ℝ)⁻¹ lam hc hlam).eigenvectorBasis
          RidgeRisk.output_finrank) i, H z⟫ ^ 2 := by
      funext z
      exact ((RidgeRisk.hat_isSymmetric A (n : ℝ)⁻¹ lam hc hlam).eigenvectorBasis
        RidgeRisk.output_finrank).sum_sq_inner_right (H z) |>.symm
    rw [hsq]
    refine integrable_finsetSum _ ?_
    intro i _
    let b := (RidgeRisk.hat_isSymmetric A (n : ℝ)⁻¹ lam hc hlam).eigenvectorBasis
      RidgeRisk.output_finrank
    let eig := (RidgeRisk.hat_isSymmetric A (n : ℝ)⁻¹ lam hc hlam).eigenvalues
      RidgeRisk.output_finrank
    have hcoord : ∀ z, ⟪b i, H z⟫ = eig i * ⟪b i, z⟫ := by
      intro z
      have happ := (RidgeRisk.hat_isSymmetric A (n : ℝ)⁻¹ lam hc hlam).apply_eigenvectorBasis
        RidgeRisk.output_finrank i
      have hsym := ((RidgeRisk.hat_isSymmetric A (n : ℝ)⁻¹ lam hc hlam) (b i) z).symm
      change ⟪b i, (RidgeRisk.hat A (n : ℝ)⁻¹ lam hc hlam).toLinearMap z⟫ = eig i * ⟪b i, z⟫
      simp only [b] at hsym ⊢
      rw [hsym, happ, inner_smul_real_left, smul_eq_mul]
    have hrew : (fun z => ⟪b i, H z⟫ ^ 2) = fun z => (eig i) ^ 2 * (⟪b i, z⟫ * ⟪b i, z⟫) := by
      funext z
      rw [hcoord z]
      ring
    rw [hrew]
    exact (hnoise.second (b i) (b i)).const_mul _
  have hcross_int : Integrable (fun z => ⟪bias, H z⟫) μ :=
    ((innerSL ℝ bias).comp H).integrable_comp hnoise.integrable_id
  have hrew : (fun z => ‖ridgePrediction net X (predict net X uStar + z) lam hn hlam -
      predict net X uStar‖ ^ 2) =
      fun z => (‖bias‖ ^ 2 + ‖H z‖ ^ 2) + 2 * ⟪bias, H z⟫ := by
    funext z
    rw [herr z, norm_add_sq_real]
    ring
  rw [hrew]
  exact (hconst.add hvar_int).add (hcross_int.const_mul 2)

/-- Initial residual bound when labels are `predict net X uStar + z` and `‖z‖ ≤ Bz`. -/
theorem initial_residual_le_of_noise (net : TwoLayerNet d m) (X : Fin n → Fin d → ℝ)
    (uStar : Parameters (m * d)) {z : Parameters n} {Bz : ℝ} (hz : ‖z‖ ≤ Bz) :
    ‖initialOutput net X - (predict net X uStar + z)‖ ≤ ‖design net X uStar‖ + Bz := by
  have hsub : initialOutput net X - (predict net X uStar + z) = -(design net X uStar + z) := by
    simp only [predict]
    abel
  rw [hsub, norm_neg]
  exact (norm_add_le (design net X uStar) z).trans (by linarith)

/-- Pointwise squared prediction error of the `T`-step nonlinear batch GD network
relative to the target `predict net X uStar`, for any noise vector `z` with `‖z‖ ≤ Bz`. -/
theorem batch_trained_pointwise_error_le (net : TwoLayerNet d m) (X : Fin n → Fin d → ℝ)
    (uStar : Parameters (m * d)) (z : Parameters n) (η lam : ℝ) {A L M E Bz : ℝ}
    (hn : 0 < n) (hlam : 0 < lam) (hη : 0 < η)
    (hA : 0 ≤ A) (hL : 0 ≤ L) (hM : 0 ≤ M) (hE : 0 ≤ E) (hBz : 0 ≤ Bz)
    (hdecay : |1 - η * lam| ≤ 1)
    (hstep : η * (Ridge.smoothness (design net X) (n : ℝ)⁻¹ lam : ℝ) ^ 2 < 2 * lam)
    (hW : ∀ i, |net.W₂ 0 i| ≤ A) (hσ : ∀ u, |deriv net.act.σ u| ≤ L)
    (hLip : ∀ a b, |deriv net.act.σ a - deriv net.act.σ b| ≤ M * |a - b|)
    (hX : ∀ r, inputEnergy (X r) ≤ E)
    (hz : ‖z‖ ≤ Bz) (T : ℕ) :
    let S := ‖design net X uStar‖ + Bz
    let q := Optim.GD.q η lam (Ridge.smoothness (design net X) (n : ℝ)⁻¹ lam)
    let δ := batchGap T m n η lam A L M E ‖design net X‖ q S
    ‖trainedBatchOutput net X (predict net X uStar + z) η lam T - predict net X uStar‖ ^ 2 ≤
      2 * ‖ridgePrediction net X (predict net X uStar + z) lam hn hlam -
        predict net X uStar‖ ^ 2 + 2 * δ ^ 2 := by
  intro S q δ
  have hS : 0 ≤ S := add_nonneg (norm_nonneg _) hBz
  have hinit : ‖initialOutput net X - (predict net X uStar + z)‖ ≤ S :=
    initial_residual_le_of_noise net X uStar hz
  have hgap : ‖trainedBatchOutput net X (predict net X uStar + z) η lam T -
      ridgePrediction net X (predict net X uStar + z) lam hn hlam‖ ≤ δ := by
    unfold trainedBatchOutput
    exact batch_trained_ridge_gap net X (predict net X uStar + z) η lam hn hlam hη
      hA hL hM hE hS hdecay hstep hW hσ hLip hX hinit T
  have hδ : 0 ≤ δ := le_trans (norm_nonneg _) hgap
  exact PredictionGap.sq_norm_le_double
    (trainedBatchOutput net X (predict net X uStar + z) η lam T)
    (ridgePrediction net X (predict net X uStar + z) lam hn hlam)
    (predict net X uStar) hδ hgap

/-- **Main Multi-Observation Statistical Generalization Bound for Nonlinear MLP Training.**
For fixed inputs `X : Fin n → Fin d → ℝ` and labels `y = F₀ + J₀ uStar + z` with
mean-zero variance-`σ²` noise `z` satisfying `‖z‖ ≤ Bz` almost surely, the expected
in-sample MSE of the `T`-step nonlinear batch GD network is bounded by twice the
fixed-design initial-Jacobian ridge risk (bias `(λ/4)‖uStar‖²` + variance `σ² df(J₀, λ)/n`)
plus `2 δ² / n`, where `δ = batchGap ...` is the explicit deterministic trajectory
plus optimization discrepancy. -/
theorem batch_trained_expected_mse (net : TwoLayerNet d m) (X : Fin n → Fin d → ℝ)
    (uStar : Parameters (m * d)) (η lam : ℝ) {A L M E Bz : ℝ}
    (hn : 0 < n) (hlam : 0 < lam) (hη : 0 < η)
    (hA : 0 ≤ A) (hL : 0 ≤ L) (hM : 0 ≤ M) (hE : 0 ≤ E) (hBz : 0 ≤ Bz)
    (hdecay : |1 - η * lam| ≤ 1)
    (hstep : η * (Ridge.smoothness (design net X) (n : ℝ)⁻¹ lam : ℝ) ^ 2 < 2 * lam)
    (hW : ∀ i, |net.W₂ 0 i| ≤ A) (hσ : ∀ u, |deriv net.act.σ u| ≤ L)
    (hLip : ∀ a b, |deriv net.act.σ a - deriv net.act.σ b| ≤ M * |a - b|)
    (hX : ∀ r, inputEnergy (X r) ≤ E)
    {μ : Measure (Parameters n)} {σ : ℝ} [IsProbabilityMeasure μ]
    (hnoise : RidgeRisk.LabelNoise μ σ)
    (hz_ae : ∀ᵐ z ∂μ, ‖z‖ ≤ Bz)
    (T : ℕ)
    (hint_trained : Integrable (fun z =>
      ‖trainedBatchOutput net X (predict net X uStar + z) η lam T -
        predict net X uStar‖ ^ 2) μ) :
    let S := ‖design net X uStar‖ + Bz
    let q := Optim.GD.q η lam (Ridge.smoothness (design net X) (n : ℝ)⁻¹ lam)
    let δ := batchGap T m n η lam A L M E ‖design net X‖ q S
    (∫ z, ‖trainedBatchOutput net X (predict net X uStar + z) η lam T -
      predict net X uStar‖ ^ 2 ∂μ) / n ≤
      2 * ((lam / 4) * ‖uStar‖ ^ 2 +
        σ ^ 2 * RidgeRisk.degreesOfFreedom (design net X) (n : ℝ)⁻¹ lam
          (inv_pos.mpr (Nat.cast_pos.mpr hn)) hlam / n) +
        2 * δ ^ 2 / n := by
  intro S q δ
  have hpoint : ∀ᵐ z ∂μ,
      ‖trainedBatchOutput net X (predict net X uStar + z) η lam T - predict net X uStar‖ ^ 2 ≤
        2 * ‖ridgePrediction net X (predict net X uStar + z) lam hn hlam -
          predict net X uStar‖ ^ 2 + 2 * δ ^ 2 := by
    filter_upwards [hz_ae] with z hz
    exact batch_trained_pointwise_error_le net X uStar z η lam hn hlam hη
      hA hL hM hE hBz hdecay hstep hW hσ hLip hX hz T
  have hq_int := integrable_ridge_error_sq net X uStar lam hn hlam hnoise
  have hconst : Integrable (fun _ : Parameters n => 2 * δ ^ 2) μ := integrable_const _
  have hupper : Integrable (fun z =>
      2 * ‖ridgePrediction net X (predict net X uStar + z) lam hn hlam -
        predict net X uStar‖ ^ 2 + 2 * δ ^ 2) μ :=
    (hq_int.const_mul 2).add hconst
  have hint : ∫ z, ‖trainedBatchOutput net X (predict net X uStar + z) η lam T -
      predict net X uStar‖ ^ 2 ∂μ ≤
      ∫ z, 2 * ‖ridgePrediction net X (predict net X uStar + z) lam hn hlam -
        predict net X uStar‖ ^ 2 + 2 * δ ^ 2 ∂μ :=
    integral_mono_ae hint_trained hupper hpoint
  rw [integral_add (hq_int.const_mul 2) hconst, integral_const_mul, integral_const] at hint
  have hprob : μ.real Set.univ = 1 := by simp
  have hsmash : μ.real Set.univ • (2 * δ ^ 2) = 2 * δ ^ 2 := by rw [hprob, one_smul]
  have hmse := MLPFixedDesign.ridge_expected_mse net X uStar lam hn hlam hnoise
  have hn0 : (0 : ℝ) < n := Nat.cast_pos.mpr hn
  rw [div_le_iff₀ hn0]
  have hintq : ∫ z, ‖ridgePrediction net X (predict net X uStar + z) lam hn hlam -
      predict net X uStar‖ ^ 2 ∂μ ≤
      n * ((lam / 4) * ‖uStar‖ ^ 2 +
        σ ^ 2 * RidgeRisk.degreesOfFreedom (design net X) (n : ℝ)⁻¹ lam
          (inv_pos.mpr (Nat.cast_pos.mpr hn)) hlam / n) := by
    rw [div_le_iff₀ hn0, mul_comm] at hmse
    exact hmse
  calc
    ∫ z, ‖trainedBatchOutput net X (predict net X uStar + z) η lam T -
        predict net X uStar‖ ^ 2 ∂μ
        ≤ 2 * (∫ z, ‖ridgePrediction net X (predict net X uStar + z) lam hn hlam -
            predict net X uStar‖ ^ 2 ∂μ) + μ.real Set.univ • (2 * δ ^ 2) := hint
    _ = 2 * (∫ z, ‖ridgePrediction net X (predict net X uStar + z) lam hn hlam -
          predict net X uStar‖ ^ 2 ∂μ) + 2 * δ ^ 2 := by rw [hsmash]
    _ ≤ 2 * (n * ((lam / 4) * ‖uStar‖ ^ 2 +
          σ ^ 2 * RidgeRisk.degreesOfFreedom (design net X) (n : ℝ)⁻¹ lam
            (inv_pos.mpr (Nat.cast_pos.mpr hn)) hlam / n)) + 2 * δ ^ 2 := by linarith
    _ = (2 * ((lam / 4) * ‖uStar‖ ^ 2 +
          σ ^ 2 * RidgeRisk.degreesOfFreedom (design net X) (n : ℝ)⁻¹ lam
            (inv_pos.mpr (Nat.cast_pos.mpr hn)) hlam / n) +
        2 * δ ^ 2 / n) * n := by field_simp [ne_of_gt hn0]

/-- Corollary replacing the ridge degrees of freedom with any upper bound `r`
on the rank of the initial-Jacobian design operator `design net X`. -/
theorem batch_trained_expected_mse_of_rank (net : TwoLayerNet d m) (X : Fin n → Fin d → ℝ)
    (uStar : Parameters (m * d)) (η lam : ℝ) {A L M E Bz : ℝ}
    (hn : 0 < n) (hlam : 0 < lam) (hη : 0 < η)
    (hA : 0 ≤ A) (hL : 0 ≤ L) (hM : 0 ≤ M) (hE : 0 ≤ E) (hBz : 0 ≤ Bz)
    (hdecay : |1 - η * lam| ≤ 1)
    (hstep : η * (Ridge.smoothness (design net X) (n : ℝ)⁻¹ lam : ℝ) ^ 2 < 2 * lam)
    (hW : ∀ i, |net.W₂ 0 i| ≤ A) (hσ : ∀ u, |deriv net.act.σ u| ≤ L)
    (hLip : ∀ a b, |deriv net.act.σ a - deriv net.act.σ b| ≤ M * |a - b|)
    (hX : ∀ r, inputEnergy (X r) ≤ E)
    {μ : Measure (Parameters n)} {σ : ℝ} [IsProbabilityMeasure μ]
    (hnoise : RidgeRisk.LabelNoise μ σ)
    (hz_ae : ∀ᵐ z ∂μ, ‖z‖ ≤ Bz)
    (T : ℕ)
    (hint_trained : Integrable (fun z =>
      ‖trainedBatchOutput net X (predict net X uStar + z) η lam T -
        predict net X uStar‖ ^ 2) μ)
    {r : ℕ} (hr : Module.finrank ℝ (LinearMap.range (design net X).toLinearMap) ≤ r) :
    let S := ‖design net X uStar‖ + Bz
    let q := Optim.GD.q η lam (Ridge.smoothness (design net X) (n : ℝ)⁻¹ lam)
    let δ := batchGap T m n η lam A L M E ‖design net X‖ q S
    (∫ z, ‖trainedBatchOutput net X (predict net X uStar + z) η lam T -
      predict net X uStar‖ ^ 2 ∂μ) / n ≤
      2 * ((lam / 4) * ‖uStar‖ ^ 2 + σ ^ 2 * (r : ℝ) / n) + 2 * δ ^ 2 / n := by
  intro S q δ
  have hmse := batch_trained_expected_mse net X uStar η lam hn hlam hη
    hA hL hM hE hBz hdecay hstep hW hσ hLip hX hnoise hz_ae T hint_trained
  have hc : 0 < (n : ℝ)⁻¹ := inv_pos.mpr (Nat.cast_pos.mpr hn)
  have hdf := RidgeRisk.degreesOfFreedom_le_rank (design net X) (n : ℝ)⁻¹ lam hc hlam
  have hvar : σ ^ 2 * RidgeRisk.degreesOfFreedom (design net X) (n : ℝ)⁻¹ lam hc hlam / n ≤
      σ ^ 2 * (r : ℝ) / n := by
    have hσ : 0 ≤ σ ^ 2 := sq_nonneg σ
    have hn0 : (0 : ℝ) ≤ n := Nat.cast_nonneg _
    have hr' : RidgeRisk.degreesOfFreedom (design net X) (n : ℝ)⁻¹ lam hc hlam ≤ (r : ℝ) :=
      hdf.trans (Nat.cast_le.mpr hr)
    exact div_le_div_of_nonneg_right (mul_le_mul_of_nonneg_left hr' hσ) hn0
  linarith

end OpenTheory.MLPBatchRisk
