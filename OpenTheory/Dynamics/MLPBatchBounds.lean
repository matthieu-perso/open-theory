import OpenTheory.Generalization.MLPFixedDesign

/-! Dataset-wide Taylor and Jacobian-drift bounds for first-layer displacement.
The inputs may be arbitrary and correlated. These are deterministic estimates
at a given displacement, not yet bounds along multi-input nonlinear training. -/
noncomputable section
open scoped BigOperators RealInnerProductSpace
namespace OpenTheory.MLPBatchBounds
open FeatureStep MLPTrajectory MLPFixedDesign
variable {d m n : ℕ}

def output (net : TwoLayerNet d m) (X : Fin n → Fin d → ℝ)
    (U : Matrix (Fin m) (Fin d) ℝ) : Parameters n :=
  initialOutput (atDisplacement net U) X

private theorem norm_le_of_coordinates (v : Parameters n) {B : ℝ} (hB : 0 ≤ B)
    (hv : ∀ r, |v r| ≤ B) : ‖v‖ ≤ Real.sqrt n * B := by
  have hs : ‖v‖ ^ 2 ≤ n * B ^ 2 := by
    rw [parameters_norm_sq]
    calc
      _ ≤ ∑ _ : Fin n, B ^ 2 := by
        apply Finset.sum_le_sum
        intro r _
        exact sq_le_sq.mpr (by simpa [abs_of_nonneg hB] using hv r)
      _ = _ := by simp
  apply (sq_le_sq₀ (norm_nonneg _) (by positivity)).mp
  simpa [mul_pow, Real.sq_sqrt (Nat.cast_nonneg n)] using hs

/-- Vector Taylor error across every observation; no sample independence is needed. -/
theorem output_remainder_le (net : TwoLayerNet d m) (X : Fin n → Fin d → ℝ) (U)
    {A M D : ℝ} (hA : 0 ≤ A) (hM : 0 ≤ M) (_hD : 0 ≤ D)
    (hW : ∀ i, |net.W₂ 0 i| ≤ A)
    (hLip : ∀ a b, |deriv net.act.σ a - deriv net.act.σ b| ≤ M * |a - b|)
    (hshift : ∀ r i, |preShift U (X r) i| ≤ D) :
    ‖output net X U - predict net X (flatten U)‖ ≤
      Real.sqrt n * (M * m * A * D ^ 2) := by
  apply norm_le_of_coordinates _ (by positivity)
  intro r
  change |(atDisplacement net U).forward (X r) - predict net X (flatten U) r| ≤ _
  rw [predict_flatten]
  apply (forward_remainder_le net U (X r) hM hLip).trans
  have hs : (∑ i, |net.W₂ 0 i| * |preShift U (X r) i| ^ 2) ≤ m * (A * D ^ 2) := by
    calc
      _ ≤ ∑ _ : Fin m, A * D ^ 2 := by
        apply Finset.sum_le_sum
        intro i _
        exact mul_le_mul (hW i)
          (pow_le_pow_left₀ (abs_nonneg _) (hshift r i) 2) (sq_nonneg _) hA
      _ = _ := by simp
  calc
    _ ≤ M * (m * (A * D ^ 2)) := mul_le_mul_of_nonneg_left hs hM
    _ = _ := by ring

private theorem flatten_outer_norm_sq (a : Fin m → ℝ) (x : Fin d → ℝ) :
    ‖flatten (fun i j => a i * x j)‖ ^ 2 = ∑ i, a i ^ 2 * inputEnergy x := by
  rw [parameters_norm_sq]
  change (∑ k : Fin (m * d), (a (finProdFinEquiv.symm k).1 *
    x (finProdFinEquiv.symm k).2) ^ 2) = _
  rw [(finProdFinEquiv.symm : Fin (m * d) ≃ Fin m × Fin d).sum_comp
    (fun p => (a p.1 * x p.2) ^ 2), Fintype.sum_prod_type]
  simp only [mul_pow, inputEnergy, Finset.mul_sum]

/-- Each Jacobian row changes linearly with the hidden preactivation displacement. -/
theorem features_drift_le (net : TwoLayerNet d m) (x : Fin d → ℝ) (U)
    {A M D E : ℝ} (hA : 0 ≤ A) (hM : 0 ≤ M) (hD : 0 ≤ D) (hE : 0 ≤ E)
    (hW : ∀ i, |net.W₂ 0 i| ≤ A)
    (hLip : ∀ a b, |deriv net.act.σ a - deriv net.act.σ b| ≤ M * |a - b|)
    (hshift : ∀ i, |preShift U x i| ≤ D) (hx : inputEnergy x ≤ E) :
    ‖features (atDisplacement net U) x - features net x‖ ≤
      Real.sqrt m * A * M * D * Real.sqrt E := by
  let a : Fin m → ℝ := fun i => net.W₂ 0 i *
    (deriv net.act.σ (preactivation (atDisplacement net U) x i) -
      deriv net.act.σ (preactivation net x i))
  have he : features (atDisplacement net U) x - features net x =
      flatten (fun i j => a i * x j) := by
    ext k
    change (net.W₂ 0 (finProdFinEquiv.symm k).1 *
        deriv net.act.σ (preactivation (atDisplacement net U) x (finProdFinEquiv.symm k).1) *
          x (finProdFinEquiv.symm k).2) -
      (net.W₂ 0 (finProdFinEquiv.symm k).1 *
        deriv net.act.σ (preactivation net x (finProdFinEquiv.symm k).1) *
          x (finProdFinEquiv.symm k).2) = _
    dsimp [flatten, a]
    ring
  have ha (i : Fin m) : |a i| ≤ A * M * D := by
    have hd : |deriv net.act.σ (preactivation (atDisplacement net U) x i) -
        deriv net.act.σ (preactivation net x i)| ≤ M * D := by
      apply (hLip _ _).trans
      rw [preactivation_displacement, add_sub_cancel_left]
      exact mul_le_mul_of_nonneg_left (hshift i) hM
    dsimp [a]
    rw [abs_mul]
    exact (mul_le_mul (hW i) hd (abs_nonneg _) hA).trans_eq (by ring)
  have hs : ‖features (atDisplacement net U) x - features net x‖ ^ 2 ≤
      m * (A * M * D) ^ 2 * E := by
    rw [he, flatten_outer_norm_sq]
    calc
      _ ≤ ∑ _ : Fin m, (A * M * D) ^ 2 * E := by
        apply Finset.sum_le_sum
        intro i _
        have hi : a i ^ 2 ≤ (A * M * D) ^ 2 :=
          sq_le_sq.mpr (by simpa [abs_of_nonneg (show 0 ≤ A * M * D by positivity)] using ha i)
        exact mul_le_mul hi hx (inputEnergy_nonneg x) (sq_nonneg _)
      _ = _ := by simp [mul_assoc]
  apply (sq_le_sq₀ (norm_nonneg _) (by positivity)).mp
  simpa [mul_pow, Real.sq_sqrt (Nat.cast_nonneg m), Real.sq_sqrt hE, mul_assoc] using hs

/-- The whole design difference is itself an observation operator. -/
theorem design_sub_eq (net : TwoLayerNet d m) (X : Fin n → Fin d → ℝ) (U) :
    design (atDisplacement net U) X - design net X =
      Dataset.observation (fun r =>
        (features (atDisplacement net U) (X r) - features net (X r), (0 : ℝ))) := by
  ext u r
  change ⟪features (atDisplacement net U) (X r), u⟫ - ⟪features net (X r), u⟫ = _
  exact (inner_sub_left _ _ _).symm

/-- Operator-norm Jacobian drift, reusing the existing observation-operator bound. -/
theorem design_drift_le (net : TwoLayerNet d m) (X : Fin n → Fin d → ℝ) (U)
    {A M D E : ℝ} (hA : 0 ≤ A) (hM : 0 ≤ M) (hD : 0 ≤ D) (hE : 0 ≤ E)
    (hW : ∀ i, |net.W₂ 0 i| ≤ A)
    (hLip : ∀ a b, |deriv net.act.σ a - deriv net.act.σ b| ≤ M * |a - b|)
    (hshift : ∀ r i, |preShift U (X r) i| ≤ D) (hX : ∀ r, inputEnergy (X r) ≤ E) :
    ‖design (atDisplacement net U) X - design net X‖ ≤
      Real.sqrt n * (Real.sqrt m * A * M * D * Real.sqrt E) := by
  rw [design_sub_eq]
  exact Dataset.observation_opNorm_le _ _ (by positivity)
    (fun r => features_drift_le net (X r) U hA hM hD hE hW hLip (hshift r) (hX r))

private theorem adjoint_residual_bound {p : ℕ}
    (J K : Parameters p →L[ℝ] Parameters n) (f l y : Parameters n)
    {B G R : ℝ} (hG : 0 ≤ G)
    (hB : ‖f - l‖ ≤ B) (hK : ‖K - J‖ ≤ G) (hR : ‖f - y‖ ≤ R) :
    ‖K.adjoint (f - y) - J.adjoint (l - y)‖ ≤ ‖J‖ * B + G * R := by
  have he : K.adjoint (f - y) - J.adjoint (l - y) =
      J.adjoint (f - l) + (K - J).adjoint (f - y) := by
    simp only [map_sub, sub_apply]
    abel
  rw [he]
  apply (norm_add_le _ _).trans
  apply add_le_add
  · have h := J.adjoint.le_opNorm (f - l)
    simp only [LinearIsometryEquiv.norm_map] at h
    exact h.trans (mul_le_mul_of_nonneg_left hB (norm_nonneg _))
  · have h := (K - J).adjoint.le_opNorm (f - y)
    simp only [LinearIsometryEquiv.norm_map] at h
    exact h.trans (mul_le_mul hK hR (norm_nonneg _) hG)

/-- Difference of the batch Jacobian-transpose residual terms. Multiplying by
`1/n` gives the discrepancy for mean half-squared loss; the common displacement
regularizer cancels. The residual bound remains an explicit hypothesis here. -/
theorem adjoint_residual_gap_le (net : TwoLayerNet d m) (X : Fin n → Fin d → ℝ)
    (U) (y : Parameters n) {A M D E R : ℝ}
    (hA : 0 ≤ A) (hM : 0 ≤ M) (hD : 0 ≤ D) (hE : 0 ≤ E)
    (hW : ∀ i, |net.W₂ 0 i| ≤ A)
    (hLip : ∀ a b, |deriv net.act.σ a - deriv net.act.σ b| ≤ M * |a - b|)
    (hshift : ∀ r i, |preShift U (X r) i| ≤ D) (hX : ∀ r, inputEnergy (X r) ≤ E)
    (hres : ‖output net X U - y‖ ≤ R) :
    ‖(design (atDisplacement net U) X).adjoint (output net X U - y) -
      (design net X).adjoint (predict net X (flatten U) - y)‖ ≤
      ‖design net X‖ * (Real.sqrt n * (M * m * A * D ^ 2)) +
        (Real.sqrt n * (Real.sqrt m * A * M * D * Real.sqrt E)) * R := by
  exact adjoint_residual_bound _ _ _ _ _ (by positivity)
    (output_remainder_le net X U hA hM hD hW hLip hshift)
    (design_drift_le net X U hA hM hD hE hW hLip hshift hX) hres

end OpenTheory.MLPBatchBounds
