import OpenTheory.Dynamics.MLPBatchBounds
import OpenTheory.Dynamics.MLPResidual

/-! Multi-input nonlinear first-layer training trajectory versus independently
trained initial-Jacobian ridge GD.

For arbitrary fixed observations `X : Fin n → Fin d → ℝ` and labels `y : Parameters n`,
the nonlinear update trains the first-layer displacement `U : Matrix (Fin m) (Fin d) ℝ`
by exact batch gradient descent on the mean half-squared error plus `(lam / 2) ‖U‖_F²`.
In flattened Euclidean coordinates `u : Parameters (m * d)`, the step is

  `u ↦ (1 - η lam) u - (η / n) J(u)ᵀ (F(u) - y)`,

whereas exact gradient descent on the initial Taylor model (`MLPFixedDesign`) is

  `v ↦ (1 - η lam) v - (η / n) J₀ᵀ (F₀ + J₀ v - y)`,

which coincides with `Optim.GD.step η (gradient (Ridge.objective J₀ (y - F₀) n⁻¹ lam))`.

From an initial residual bound `‖F₀ - y‖ ≤ S` and bounded input energies `≤ E`,
we prove:
1. A derived finite-horizon parameter-norm, preactivation-shift, and residual envelope
   (`batch_norm_le`, `batch_shift_le`, `batch_residual_le`) requiring no trajectory-dependent
   hypothesis.
2. Accumulation of the local Taylor and Jacobian-drift discrepancy (`MLPBatchBounds`)
   along the contractively regularized initial-feature GD recurrence.
3. A fully discharged deterministic distance (`batch_trained_ridge_gap`) between the
   T-step nonlinear MLP output vector and the multi-input initial-Jacobian ridge
   prediction (`MLPFixedDesign.ridgePrediction`). -/

noncomputable section
open scoped BigOperators RealInnerProductSpace

namespace OpenTheory.MLPBatchTrajectory
open FeatureStep MLPTrajectory MLPFixedDesign MLPBatchBounds MLPResidual

variable {d m n : ℕ}

/-- Inverse of `flatten`, mapping a vector in `Parameters (m * d)` back to an `m × d` matrix. -/
def unflatten (u : Parameters (m * d)) : Matrix (Fin m) (Fin d) ℝ :=
  fun i j => u (finProdFinEquiv (i, j))

@[simp]
theorem unflatten_flatten (U : Matrix (Fin m) (Fin d) ℝ) :
    unflatten (flatten U) = U := by
  ext i j
  change (flatten U) (finProdFinEquiv (i, j)) = U i j
  unfold flatten
  simp only [Equiv.symm_apply_apply]

@[simp]
theorem flatten_unflatten (u : Parameters (m * d)) :
    flatten (unflatten u) = u := by
  ext k
  change u (finProdFinEquiv (finProdFinEquiv.symm k)) = u k
  rw [Equiv.apply_symm_apply]

@[simp]
theorem unflatten_zero : unflatten (0 : Parameters (m * d)) = 0 := by
  ext i j
  simp [unflatten]

@[simp]
theorem flatten_zero : flatten (0 : Matrix (Fin m) (Fin d) ℝ) = 0 := by
  ext k
  simp [flatten]

/-- `flatten` preserves the squared Euclidean / Frobenius norm. -/
theorem flatten_norm_sq (U : Matrix (Fin m) (Fin d) ℝ) :
    ‖flatten U‖ ^ 2 = ∑ i, ∑ j, (U i j) ^ 2 := by
  rw [parameters_norm_sq]
  change (∑ k : Fin (m * d), (U (finProdFinEquiv.symm k).1 (finProdFinEquiv.symm k).2) ^ 2) = _
  rw [(finProdFinEquiv.symm : Fin (m * d) ≃ Fin m × Fin d).sum_comp
    (fun p => (U p.1 p.2) ^ 2), Fintype.sum_prod_type]

theorem unflatten_norm_sq (u : Parameters (m * d)) :
    ∑ i, ∑ j, (unflatten u i j) ^ 2 = ‖u‖ ^ 2 := by
  rw [← flatten_norm_sq, flatten_unflatten]

/-- Cauchy–Schwarz on each row bounds `preShift` by `‖u‖ * √(inputEnergy x)`. -/
theorem preShift_unflatten_le_norm (u : Parameters (m * d)) (x : Fin d → ℝ) (i : Fin m) :
    |preShift (unflatten u) x i| ≤ ‖u‖ * Real.sqrt (inputEnergy x) := by
  let M : Matrix (Fin m) (Fin d) ℝ := fun a j => if a = i then x j else 0
  let v : Parameters (m * d) := flatten M
  have hinner : preShift (unflatten u) x i = ⟪v, u⟫ := by
    rw [PiLp.inner_apply]
    simp only [Real.inner_apply, v]
    change (∑ j, u (finProdFinEquiv (i, j)) * x j) =
      ∑ k : Fin (m * d), M (finProdFinEquiv.symm k).1 (finProdFinEquiv.symm k).2 * u k
    have hcomp := (finProdFinEquiv : Fin m × Fin d ≃ Fin (m * d)).sum_comp
      (fun k => M (finProdFinEquiv.symm k).1 (finProdFinEquiv.symm k).2 * u k)
    rw [← hcomp, Fintype.sum_prod_type]
    simp only [Equiv.symm_apply_apply, M]
    rw [Finset.sum_eq_single i]
    · apply Finset.sum_congr rfl
      intro j _
      simp [mul_comm]
    · intro a _ ha
      simp [ha]
    · intro h
      exact absurd (Finset.mem_univ i) h
  have hnorm_v : ‖v‖ = Real.sqrt (inputEnergy x) := by
    have hsq : ‖v‖ ^ 2 = inputEnergy x := by
      change ‖flatten M‖ ^ 2 = inputEnergy x
      rw [flatten_norm_sq]
      rw [Finset.sum_eq_single i]
      · simp [M, inputEnergy]
      · intro a _ ha
        simp [M, ha]
      · intro h
        exact absurd (Finset.mem_univ i) h
    rw [← Real.sqrt_sq (norm_nonneg v), hsq]
  rw [hinner]
  calc
    |⟪v, u⟫| ≤ ‖v‖ * ‖u‖ := abs_real_inner_le_norm v u
    _ = ‖u‖ * Real.sqrt (inputEnergy x) := by rw [hnorm_v, mul_comm]

/-- Output vector of the displaced network in flattened parameter coordinates. -/
def batchOutput (net : TwoLayerNet d m) (X : Fin n → Fin d → ℝ)
    (u : Parameters (m * d)) : Parameters n :=
  output net X (unflatten u)

/-- Jacobian design operator at flattened displacement `u`. -/
def batchDesign (net : TwoLayerNet d m) (X : Fin n → Fin d → ℝ)
    (u : Parameters (m * d)) : Parameters (m * d) →L[ℝ] Parameters n :=
  design (atDisplacement net (unflatten u)) X

@[simp]
theorem batchOutput_zero (net : TwoLayerNet d m) (X : Fin n → Fin d → ℝ) :
    batchOutput net X 0 = initialOutput net X := by
  simp [batchOutput, output, atDisplacement]

@[simp]
theorem batchDesign_zero (net : TwoLayerNet d m) (X : Fin n → Fin d → ℝ) :
    batchDesign net X 0 = design net X := by
  simp [batchDesign, atDisplacement]

/-- Norm of the Jacobian row `features net x` is bounded by `√m * A * L * √E`. -/
theorem features_norm_le (net : TwoLayerNet d m) (x : Fin d → ℝ)
    {A L E : ℝ} (hA : 0 ≤ A) (hL : 0 ≤ L) (hE : 0 ≤ E)
    (hW : ∀ i, |net.W₂ 0 i| ≤ A) (hσ : ∀ u, |deriv net.act.σ u| ≤ L)
    (hx : inputEnergy x ≤ E) :
    ‖features net x‖ ≤ Real.sqrt m * A * L * Real.sqrt E := by
  let a : Fin m → ℝ := fun i => net.W₂ 0 i * deriv net.act.σ (preactivation net x i)
  have he : features net x = flatten (fun i j => a i * x j) := rfl
  have hsq_eq : ‖features net x‖ ^ 2 = ∑ i, a i ^ 2 * inputEnergy x := by
    rw [he, parameters_norm_sq]
    change (∑ k : Fin (m * d), (a (finProdFinEquiv.symm k).1 *
      x (finProdFinEquiv.symm k).2) ^ 2) = _
    rw [(finProdFinEquiv.symm : Fin (m * d) ≃ Fin m × Fin d).sum_comp
      (fun p => (a p.1 * x p.2) ^ 2), Fintype.sum_prod_type]
    simp only [mul_pow, inputEnergy, Finset.mul_sum]
  have hs : ‖features net x‖ ^ 2 ≤ m * (A * L) ^ 2 * E := by
    rw [hsq_eq]
    calc
      _ ≤ ∑ _ : Fin m, (A * L) ^ 2 * E := by
        apply Finset.sum_le_sum
        intro i _
        have ha : |a i| ≤ A * L := by
          dsimp [a]
          rw [abs_mul]
          exact mul_le_mul (hW i) (hσ _) (abs_nonneg _) hA
        have ha2 : a i ^ 2 ≤ (A * L) ^ 2 :=
          sq_le_sq.mpr (by simpa [abs_of_nonneg (mul_nonneg hA hL)] using ha)
        exact mul_le_mul ha2 hx (inputEnergy_nonneg x) (sq_nonneg _)
      _ = _ := by simp [mul_assoc]
  apply (sq_le_sq₀ (norm_nonneg _) (by positivity)).mp
  simpa [mul_pow, Real.sq_sqrt (Nat.cast_nonneg m), Real.sq_sqrt hE, mul_assoc] using hs

/-- Operator norm of `design net X` is bounded by `√n * (√m * A * L * √E)`. -/
theorem design_opNorm_le (net : TwoLayerNet d m) (X : Fin n → Fin d → ℝ)
    {A L E : ℝ} (hA : 0 ≤ A) (hL : 0 ≤ L) (hE : 0 ≤ E)
    (hW : ∀ i, |net.W₂ 0 i| ≤ A) (hσ : ∀ u, |deriv net.act.σ u| ≤ L)
    (hX : ∀ r, inputEnergy (X r) ≤ E) :
    ‖design net X‖ ≤ Real.sqrt n * (Real.sqrt m * A * L * Real.sqrt E) := by
  exact Dataset.observation_opNorm_le _ _ (by positivity)
    (fun r => features_norm_le net (X r) hA hL hE hW hσ (hX r))

/-- Output vector difference from initialization bounded by `√n * (m * A * L * (‖u‖ * √E))`. -/
theorem batchOutput_change_le (net : TwoLayerNet d m) (X : Fin n → Fin d → ℝ)
    (u : Parameters (m * d)) {A L E : ℝ}
    (hA : 0 ≤ A) (hL : 0 ≤ L) (_hE : 0 ≤ E)
    (hW : ∀ i, |net.W₂ 0 i| ≤ A) (hσ : ∀ v, |deriv net.act.σ v| ≤ L)
    (hX : ∀ r, inputEnergy (X r) ≤ E) :
    ‖batchOutput net X u - initialOutput net X‖ ≤
      Real.sqrt n * (m * A * L * (‖u‖ * Real.sqrt E)) := by
  have hcoord : ∀ r, |(batchOutput net X u - initialOutput net X) r| ≤
      m * A * L * (‖u‖ * Real.sqrt E) := by
    intro r
    change |(atDisplacement net (unflatten u)).forward (X r) - net.forward (X r)| ≤ _
    apply forward_change_le net (X r) (unflatten u) hA hL hW hσ
    intro i
    apply (preShift_unflatten_le_norm u (X r) i).trans
    exact mul_le_mul_of_nonneg_left (Real.sqrt_le_sqrt (hX r)) (norm_nonneg u)
  have hB : 0 ≤ (m : ℝ) * A * L * (‖u‖ * Real.sqrt E) := by positivity
  have hs : ‖batchOutput net X u - initialOutput net X‖ ^ 2 ≤
      n * (m * A * L * (‖u‖ * Real.sqrt E)) ^ 2 := by
    rw [parameters_norm_sq]
    calc
      _ ≤ ∑ _ : Fin n, (m * A * L * (‖u‖ * Real.sqrt E)) ^ 2 := by
        apply Finset.sum_le_sum
        intro r _
        exact sq_le_sq.mpr (by simpa [abs_of_nonneg hB] using hcoord r)
      _ = _ := by simp
  apply (sq_le_sq₀ (norm_nonneg _) (by positivity)).mp
  simpa [mul_pow, Real.sq_sqrt (Nat.cast_nonneg n)] using hs

/-- Exact batch gradient step on the nonlinear first-layer displacement in `Parameters (m * d)`,
for mean half-squared loss `(1 / (2n)) ‖F(u) - y‖² + (lam / 2) ‖u‖²`. -/
def nonlinearBatchStep (net : TwoLayerNet d m) (X : Fin n → Fin d → ℝ)
    (y : Parameters n) (η lam : ℝ) (u : Parameters (m * d)) : Parameters (m * d) :=
  (1 - η * lam) • u -
    (η * (n : ℝ)⁻¹) • (batchDesign net X u).adjoint (batchOutput net X u - y)

/-- T-step nonlinear batch GD trajectory starting from zero displacement. -/
def nonlinearBatchTrajectory (net : TwoLayerNet d m) (X : Fin n → Fin d → ℝ)
    (y : Parameters n) (η lam : ℝ) (T : ℕ) : Parameters (m * d) :=
  (nonlinearBatchStep net X y η lam)^[T] 0

/-- Exact gradient step on the initial-Jacobian ridge objective with centered labels `y - F₀`. -/
def linearBatchStep (net : TwoLayerNet d m) (X : Fin n → Fin d → ℝ)
    (y : Parameters n) (η lam : ℝ) : Parameters (m * d) → Parameters (m * d) :=
  Optim.GD.step η
    (gradient (Ridge.objective (design net X) (y - initialOutput net X) (n : ℝ)⁻¹ lam))

/-- T-step linearized batch GD trajectory starting from zero displacement. -/
def linearBatchTrajectory (net : TwoLayerNet d m) (X : Fin n → Fin d → ℝ)
    (y : Parameters n) (η lam : ℝ) (T : ℕ) : Parameters (m * d) :=
  (linearBatchStep net X y η lam)^[T] 0

theorem linearBatchStep_eq (net : TwoLayerNet d m) (X : Fin n → Fin d → ℝ)
    (y : Parameters n) (η lam : ℝ) (u : Parameters (m * d)) :
    linearBatchStep net X y η lam u =
      (1 - η * lam) • u -
        (η * (n : ℝ)⁻¹) • (design net X).adjoint (predict net X u - y) := by
  unfold linearBatchStep Optim.GD.step
  rw [Ridge.gradient_eq]
  unfold Ridge.grad predict
  have hsub : initialOutput net X + design net X u - y =
      design net X u - (y - initialOutput net X) := by abel
  rw [hsub, smul_add, smul_smul, smul_smul, sub_smul, one_smul]
  abel

/-- One-step discrepancy between nonlinear batch GD and initial-feature ridge GD. -/
theorem batch_step_defect (net : TwoLayerNet d m) (X : Fin n → Fin d → ℝ)
    (y : Parameters n) (η lam : ℝ) (u : Parameters (m * d)) :
    nonlinearBatchStep net X y η lam u - linearBatchStep net X y η lam u =
      - ((η * (n : ℝ)⁻¹) •
        ((batchDesign net X u).adjoint (batchOutput net X u - y) -
          (design net X).adjoint (predict net X u - y))) := by
  rw [nonlinearBatchStep, linearBatchStep_eq]
  simp only [smul_sub]
  abel

/-- Norm of the one-step parameter discrepancy bounded via `MLPBatchBounds.adjoint_residual_gap_le`. -/
theorem batch_step_defect_norm_le (net : TwoLayerNet d m) (X : Fin n → Fin d → ℝ)
    (y : Parameters n) (η lam : ℝ) (u : Parameters (m * d))
    {A M D E R : ℝ} (hη : 0 ≤ η) (hA : 0 ≤ A) (hM : 0 ≤ M) (hD : 0 ≤ D) (hE : 0 ≤ E)
    (hW : ∀ i, |net.W₂ 0 i| ≤ A)
    (hLip : ∀ a b, |deriv net.act.σ a - deriv net.act.σ b| ≤ M * |a - b|)
    (hshift : ∀ r i, |preShift (unflatten u) (X r) i| ≤ D)
    (hX : ∀ r, inputEnergy (X r) ≤ E)
    (hres : ‖batchOutput net X u - y‖ ≤ R) :
    ‖nonlinearBatchStep net X y η lam u - linearBatchStep net X y η lam u‖ ≤
      (η * (n : ℝ)⁻¹) *
        (‖design net X‖ * (Real.sqrt n * (M * m * A * D ^ 2)) +
          (Real.sqrt n * (Real.sqrt m * A * M * D * Real.sqrt E)) * R) := by
  rw [batch_step_defect, norm_neg, norm_smul, Real.norm_eq_abs,
    abs_of_nonneg (mul_nonneg hη (inv_nonneg.mpr (Nat.cast_nonneg n)))]
  apply mul_le_mul_of_nonneg_left _ (mul_nonneg hη (inv_nonneg.mpr (Nat.cast_nonneg n)))
  have h := adjoint_residual_gap_le net X (unflatten u) y hA hM hD hE hW hLip hshift hX hres
  simpa [batchDesign, batchOutput, flatten_unflatten] using h

/-- Derived multi-input geometric growth factor per batch step. -/
def batchGrowth (m n : ℕ) (η A L E : ℝ) : ℝ :=
  1 + (η * (n : ℝ)⁻¹) * (Real.sqrt n * (Real.sqrt m * A * L * Real.sqrt E)) *
    (1 + Real.sqrt n * (m * A * L * Real.sqrt E))

/-- Derived parameter-norm envelope after `T` steps. -/
def batchNormEnvelope (m n : ℕ) (η A L E S : ℝ) (T : ℕ) : ℝ :=
  S * batchGrowth m n η A L E ^ T

/-- Derived preactivation-shift envelope after `T` steps. -/
def batchShiftEnvelope (m n : ℕ) (η A L E S : ℝ) (T : ℕ) : ℝ :=
  batchNormEnvelope m n η A L E S T * Real.sqrt E

/-- Derived batch residual-norm envelope after `T` steps. -/
def batchResidualEnvelope (m n : ℕ) (η A L E S : ℝ) (T : ℕ) : ℝ :=
  (1 + Real.sqrt n * (m * A * L * Real.sqrt E)) * batchNormEnvelope m n η A L E S T

theorem batchGrowth_one_le (m n : ℕ) {η A L E : ℝ}
    (hη : 0 ≤ η) (hA : 0 ≤ A) (hL : 0 ≤ L) (_hE : 0 ≤ E) :
    1 ≤ batchGrowth m n η A L E := by
  unfold batchGrowth
  exact le_add_of_nonneg_right (by positivity)

/-- Derived parameter-norm bound along the nonlinear batch trajectory, requiring only
an initial residual bound `‖F₀ - y‖ ≤ S`. -/
theorem batch_norm_le (net : TwoLayerNet d m) (X : Fin n → Fin d → ℝ)
    (y : Parameters n) (η lam : ℝ) {A L E S : ℝ}
    (hη : 0 ≤ η) (hA : 0 ≤ A) (hL : 0 ≤ L) (hE : 0 ≤ E) (hS : 0 ≤ S)
    (hdecay : |1 - η * lam| ≤ 1)
    (hW : ∀ i, |net.W₂ 0 i| ≤ A) (hσ : ∀ u, |deriv net.act.σ u| ≤ L)
    (hX : ∀ r, inputEnergy (X r) ≤ E)
    (hinit : ‖initialOutput net X - y‖ ≤ S) (T : ℕ) :
    ‖nonlinearBatchTrajectory net X y η lam T‖ ≤ batchNormEnvelope m n η A L E S T := by
  have hq := batchGrowth_one_le m n hη hA hL hE
  induction T with
  | zero =>
    simp [nonlinearBatchTrajectory, batchNormEnvelope, hS]
  | succ t ih =>
    let ut := nonlinearBatchTrajectory net X y η lam t
    have hpow : 1 ≤ batchGrowth m n η A L E ^ t := one_le_pow₀ hq
    have hS_le : S ≤ batchNormEnvelope m n η A L E S t := by
      unfold batchNormEnvelope
      nlinarith
    have hres : ‖batchOutput net X ut - y‖ ≤ batchResidualEnvelope m n η A L E S t := by
      have hf := batchOutput_change_le net X ut hA hL hE hW hσ hX
      have htri : ‖batchOutput net X ut - y‖ ≤
          ‖batchOutput net X ut - initialOutput net X‖ + ‖initialOutput net X - y‖ := by
        simpa using norm_add_le (batchOutput net X ut - initialOutput net X) (initialOutput net X - y)
      have huf : Real.sqrt n * (m * A * L * (‖ut‖ * Real.sqrt E)) ≤
          Real.sqrt n * (m * A * L * Real.sqrt E) * batchNormEnvelope m n η A L E S t := by
        have h1 : ‖ut‖ * Real.sqrt E ≤ batchNormEnvelope m n η A L E S t * Real.sqrt E :=
          mul_le_mul_of_nonneg_right ih (Real.sqrt_nonneg E)
        have h2 : (m : ℝ) * A * L * (‖ut‖ * Real.sqrt E) ≤
            (m : ℝ) * A * L * (batchNormEnvelope m n η A L E S t * Real.sqrt E) :=
          mul_le_mul_of_nonneg_left h1 (by positivity)
        calc
          Real.sqrt n * (m * A * L * (‖ut‖ * Real.sqrt E))
              ≤ Real.sqrt n * (m * A * L * (batchNormEnvelope m n η A L E S t * Real.sqrt E)) :=
            mul_le_mul_of_nonneg_left h2 (Real.sqrt_nonneg n)
          _ = Real.sqrt n * (m * A * L * Real.sqrt E) * batchNormEnvelope m n η A L E S t := by ring
      unfold batchResidualEnvelope
      linarith
    have hstep_eq : nonlinearBatchTrajectory net X y η lam (t + 1) =
        nonlinearBatchStep net X y η lam ut := by
      simp only [nonlinearBatchTrajectory, Function.iterate_succ_apply', ut]
    rw [hstep_eq, nonlinearBatchStep]
    apply (norm_sub_le _ _).trans
    have hd : ‖(1 - η * lam) • ut‖ ≤ batchNormEnvelope m n η A L E S t := by
      rw [norm_smul, Real.norm_eq_abs]
      exact (mul_le_mul_of_nonneg_right hdecay (norm_nonneg ut)).trans (by simpa using ih)
    have hJ : ‖batchDesign net X ut‖ ≤ Real.sqrt n * (Real.sqrt m * A * L * Real.sqrt E) := by
      have hWu : ∀ i, |(atDisplacement net (unflatten ut)).W₂ 0 i| ≤ A := hW
      have hσu : ∀ v, |deriv (atDisplacement net (unflatten ut)).act.σ v| ≤ L := hσ
      exact design_opNorm_le (atDisplacement net (unflatten ut)) X hA hL hE hWu hσu hX
    have hadj : ‖(batchDesign net X ut).adjoint (batchOutput net X ut - y)‖ ≤
        (Real.sqrt n * (Real.sqrt m * A * L * Real.sqrt E)) *
          batchResidualEnvelope m n η A L E S t := by
      have hop := (batchDesign net X ut).adjoint.le_opNorm (batchOutput net X ut - y)
      simp only [LinearIsometryEquiv.norm_map] at hop
      exact hop.trans (mul_le_mul hJ hres (norm_nonneg _) (by positivity))
    have hb : ‖(η * (n : ℝ)⁻¹) • (batchDesign net X ut).adjoint (batchOutput net X ut - y)‖ ≤
        (η * (n : ℝ)⁻¹) * (Real.sqrt n * (Real.sqrt m * A * L * Real.sqrt E)) *
          batchResidualEnvelope m n η A L E S t := by
      rw [norm_smul, Real.norm_eq_abs,
        abs_of_nonneg (mul_nonneg hη (inv_nonneg.mpr (Nat.cast_nonneg n)))]
      simpa [mul_assoc] using
        mul_le_mul_of_nonneg_left hadj (mul_nonneg hη (inv_nonneg.mpr (Nat.cast_nonneg n)))
    apply (add_le_add hd hb).trans
    simp only [batchResidualEnvelope, batchNormEnvelope, pow_succ, batchGrowth]
    ring_nf
    exact le_rfl

/-- Derived hidden preactivation shift bound across all observations and neurons up to horizon `T`. -/
theorem batch_shift_le (net : TwoLayerNet d m) (X : Fin n → Fin d → ℝ)
    (y : Parameters n) (η lam : ℝ) {A L E S : ℝ}
    (hη : 0 ≤ η) (hA : 0 ≤ A) (hL : 0 ≤ L) (hE : 0 ≤ E) (hS : 0 ≤ S)
    (hdecay : |1 - η * lam| ≤ 1)
    (hW : ∀ i, |net.W₂ 0 i| ≤ A) (hσ : ∀ u, |deriv net.act.σ u| ≤ L)
    (hX : ∀ r, inputEnergy (X r) ≤ E)
    (hinit : ‖initialOutput net X - y‖ ≤ S)
    (T t : ℕ) (ht : t ≤ T) (r : Fin n) (i : Fin m) :
    |preShift (unflatten (nonlinearBatchTrajectory net X y η lam t)) (X r) i| ≤
      batchShiftEnvelope m n η A L E S T := by
  have hq := batchGrowth_one_le m n hη hA hL hE
  have hnorm_t := batch_norm_le net X y η lam hη hA hL hE hS hdecay hW hσ hX hinit t
  have hnorm_T : batchNormEnvelope m n η A L E S t ≤ batchNormEnvelope m n η A L E S T := by
    unfold batchNormEnvelope
    exact mul_le_mul_of_nonneg_left (pow_le_pow_right₀ hq ht) hS
  apply (preShift_unflatten_le_norm _ (X r) i).trans
  unfold batchShiftEnvelope
  exact mul_le_mul (hnorm_t.trans hnorm_T) (Real.sqrt_le_sqrt (hX r))
    (Real.sqrt_nonneg _) (by unfold batchNormEnvelope; positivity)

/-- Derived batch residual bound up to horizon `T`, with no trajectory assumption. -/
theorem batch_residual_le (net : TwoLayerNet d m) (X : Fin n → Fin d → ℝ)
    (y : Parameters n) (η lam : ℝ) {A L E S : ℝ}
    (hη : 0 ≤ η) (hA : 0 ≤ A) (hL : 0 ≤ L) (hE : 0 ≤ E) (hS : 0 ≤ S)
    (hdecay : |1 - η * lam| ≤ 1)
    (hW : ∀ i, |net.W₂ 0 i| ≤ A) (hσ : ∀ u, |deriv net.act.σ u| ≤ L)
    (hX : ∀ r, inputEnergy (X r) ≤ E)
    (hinit : ‖initialOutput net X - y‖ ≤ S)
    (T t : ℕ) (ht : t ≤ T) :
    ‖batchOutput net X (nonlinearBatchTrajectory net X y η lam t) - y‖ ≤
      batchResidualEnvelope m n η A L E S T := by
  let ut := nonlinearBatchTrajectory net X y η lam t
  have hq := batchGrowth_one_le m n hη hA hL hE
  have hpow : 1 ≤ batchGrowth m n η A L E ^ t := one_le_pow₀ hq
  have hS_le : S ≤ batchNormEnvelope m n η A L E S t := by
    unfold batchNormEnvelope
    nlinarith
  have hnorm_t := batch_norm_le net X y η lam hη hA hL hE hS hdecay hW hσ hX hinit t
  have hf := batchOutput_change_le net X ut hA hL hE hW hσ hX
  have htri : ‖batchOutput net X ut - y‖ ≤
      ‖batchOutput net X ut - initialOutput net X‖ + ‖initialOutput net X - y‖ := by
    simpa using norm_add_le (batchOutput net X ut - initialOutput net X) (initialOutput net X - y)
  have huf : Real.sqrt n * (m * A * L * (‖ut‖ * Real.sqrt E)) ≤
      Real.sqrt n * (m * A * L * Real.sqrt E) * batchNormEnvelope m n η A L E S t := by
    have h1 : ‖ut‖ * Real.sqrt E ≤ batchNormEnvelope m n η A L E S t * Real.sqrt E :=
      mul_le_mul_of_nonneg_right hnorm_t (Real.sqrt_nonneg E)
    have h2 : (m : ℝ) * A * L * (‖ut‖ * Real.sqrt E) ≤
        (m : ℝ) * A * L * (batchNormEnvelope m n η A L E S t * Real.sqrt E) :=
      mul_le_mul_of_nonneg_left h1 (by positivity)
    calc
      Real.sqrt n * (m * A * L * (‖ut‖ * Real.sqrt E))
          ≤ Real.sqrt n * (m * A * L * (batchNormEnvelope m n η A L E S t * Real.sqrt E)) :=
        mul_le_mul_of_nonneg_left h2 (Real.sqrt_nonneg n)
      _ = Real.sqrt n * (m * A * L * Real.sqrt E) * batchNormEnvelope m n η A L E S t := by ring
  have hres_t : ‖batchOutput net X ut - y‖ ≤ batchResidualEnvelope m n η A L E S t := by
    unfold batchResidualEnvelope
    linarith
  apply hres_t.trans
  unfold batchResidualEnvelope batchNormEnvelope
  exact mul_le_mul_of_nonneg_left
    (mul_le_mul_of_nonneg_left (pow_le_pow_right₀ hq ht) hS) (by positivity)

/-- Accumulation of bounded local defects in a nonexpansive vector recurrence. -/
theorem accumulated_vector_defect {p : ℕ} (F : Parameters p → Parameters p)
    (u : ℕ → Parameters p) (C : ℝ)
    (hF : ∀ a b, ‖F a - F b‖ ≤ ‖a - b‖) (T : ℕ)
    (hdef : ∀ t < T, ‖u (t + 1) - F (u t)‖ ≤ C) :
    ‖u T - F^[T] (u 0)‖ ≤ T * C := by
  induction T with
  | zero => simp
  | succ T ih =>
    have hi := ih (fun t ht => hdef t (Nat.lt_succ_of_lt ht))
    have hd := hdef T (Nat.lt_succ_self T)
    rw [Function.iterate_succ_apply']
    have ht : ‖u (T + 1) - F (F^[T] (u 0))‖ ≤
        ‖u (T + 1) - F (u T)‖ + ‖F (u T) - F (F^[T] (u 0))‖ := by
      simpa using norm_add_le (u (T + 1) - F (u T)) (F (u T) - F (F^[T] (u 0)))
    have hl := hF (u T) (F^[T] (u 0))
    push_cast
    linarith

/-- Under the ridge step-size condition, `linearBatchStep` is nonexpansive on `Parameters (m * d)`. -/
theorem linearBatchStep_nonexpansive (net : TwoLayerNet d m) (X : Fin n → Fin d → ℝ)
    (y : Parameters n) (η lam : ℝ) (_hlam : 0 < lam) (hη : 0 < η)
    (hstep : η * (Ridge.smoothness (design net X) (n : ℝ)⁻¹ lam : ℝ) ^ 2 < 2 * lam)
    (a b : Parameters (m * d)) :
    ‖linearBatchStep net X y η lam a - linearBatchStep net X y η lam b‖ ≤ ‖a - b‖ := by
  unfold linearBatchStep
  have hg : gradient (Ridge.objective (design net X) (y - initialOutput net X) (n : ℝ)⁻¹ lam) =
      Ridge.grad (design net X) (y - initialOutput net X) (n : ℝ)⁻¹ lam :=
    funext (Ridge.gradient_eq (design net X) (y - initialOutput net X) (n : ℝ)⁻¹ lam)
  rw [hg]
  have hmono := Ridge.stronglyMonotone (design net X) (y - initialOutput net X)
    (n : ℝ)⁻¹ lam (inv_nonneg.mpr (Nat.cast_nonneg n))
  have hlip : LipschitzWith (Ridge.smoothness (design net X) (n : ℝ)⁻¹ lam)
      (Ridge.grad (design net X) (y - initialOutput net X) (n : ℝ)⁻¹ lam) :=
    (Ridge.grad_lipschitz (design net X) (y - initialOutput net X) (n : ℝ)⁻¹ lam).weaken
      (by simp [Ridge.smoothness])
  have hsq := Optim.GD.step_norm_sq_le η lam hη.le
    (Ridge.grad (design net X) (y - initialOutput net X) (n : ℝ)⁻¹ lam) hmono hlip a b
  have hq1 : Optim.GD.q η lam (Ridge.smoothness (design net X) (n : ℝ)⁻¹ lam) ≤ 1 :=
    (Optim.GD.q_lt_one_of_mul_sq_lt η lam _ hη hstep).le
  have hsq1 : ‖Optim.GD.step η (Ridge.grad (design net X) (y - initialOutput net X) (n : ℝ)⁻¹ lam) a -
      Optim.GD.step η (Ridge.grad (design net X) (y - initialOutput net X) (n : ℝ)⁻¹ lam) b‖ ^ 2 ≤
      ‖a - b‖ ^ 2 := by
    calc
      _ ≤ (1 - 2 * η * lam + η ^ 2 * (Ridge.smoothness (design net X) (n : ℝ)⁻¹ lam : ℝ) ^ 2) *
          ‖a - b‖ ^ 2 := hsq
      _ = Optim.GD.q η lam (Ridge.smoothness (design net X) (n : ℝ)⁻¹ lam) * ‖a - b‖ ^ 2 := rfl
      _ ≤ 1 * ‖a - b‖ ^ 2 := mul_le_mul_of_nonneg_right hq1 (sq_nonneg _)
      _ = ‖a - b‖ ^ 2 := one_mul _
  exact (sq_le_sq₀ (norm_nonneg _) (norm_nonneg _)).mp hsq1

/-- Accumulated parameter distance between the T-step nonlinear batch trajectory and
independently trained initial-feature ridge GD, with all shift and residual envelopes
discharged from `‖F₀ - y‖ ≤ S`. -/
theorem nonlinear_linear_param_gap (net : TwoLayerNet d m) (X : Fin n → Fin d → ℝ)
    (y : Parameters n) (η lam : ℝ) {A L M E S : ℝ}
    (hlam : 0 < lam) (hη : 0 < η)
    (hA : 0 ≤ A) (hL : 0 ≤ L) (hM : 0 ≤ M) (hE : 0 ≤ E) (hS : 0 ≤ S)
    (hdecay : |1 - η * lam| ≤ 1)
    (hstep : η * (Ridge.smoothness (design net X) (n : ℝ)⁻¹ lam : ℝ) ^ 2 < 2 * lam)
    (hW : ∀ i, |net.W₂ 0 i| ≤ A) (hσ : ∀ u, |deriv net.act.σ u| ≤ L)
    (hLip : ∀ a b, |deriv net.act.σ a - deriv net.act.σ b| ≤ M * |a - b|)
    (hX : ∀ r, inputEnergy (X r) ≤ E)
    (hinit : ‖initialOutput net X - y‖ ≤ S) (T : ℕ) :
    let D := batchShiftEnvelope m n η A L E S T
    let R := batchResidualEnvelope m n η A L E S T
    let B := Real.sqrt n * (M * m * A * D ^ 2)
    let G := Real.sqrt n * (Real.sqrt m * A * M * D * Real.sqrt E)
    ‖nonlinearBatchTrajectory net X y η lam T - linearBatchTrajectory net X y η lam T‖ ≤
      T * ((η * (n : ℝ)⁻¹) * (‖design net X‖ * B + G * R)) := by
  dsimp only
  let D := batchShiftEnvelope m n η A L E S T
  let R := batchResidualEnvelope m n η A L E S T
  let B := Real.sqrt n * (M * m * A * D ^ 2)
  let G := Real.sqrt n * (Real.sqrt m * A * M * D * Real.sqrt E)
  have hD : 0 ≤ D := by
    dsimp [D, batchShiftEnvelope, batchNormEnvelope, batchGrowth]
    positivity
  apply accumulated_vector_defect (linearBatchStep net X y η lam)
    (nonlinearBatchTrajectory net X y η lam)
    ((η * (n : ℝ)⁻¹) * (‖design net X‖ * B + G * R))
    (linearBatchStep_nonexpansive net X y η lam hlam hη hstep) T
  intro t ht
  have hstep_eq : nonlinearBatchTrajectory net X y η lam (t + 1) =
      nonlinearBatchStep net X y η lam (nonlinearBatchTrajectory net X y η lam t) := by
    simp only [nonlinearBatchTrajectory, Function.iterate_succ_apply']
  rw [hstep_eq]
  exact batch_step_defect_norm_le net X y η lam
    (nonlinearBatchTrajectory net X y η lam t)
    hη.le hA hM hD hE hW hLip
    (batch_shift_le net X y η lam hη.le hA hL hE hS hdecay hW hσ hX hinit T t ht.le)
    hX
    (batch_residual_le net X y η lam hη.le hA hL hE hS hdecay hW hσ hX hinit T t ht.le)

/-- Output-vector distance between the T-step nonlinear batch trajectory and
independently trained initial-feature ridge GD predictions. -/
theorem nonlinear_linear_output_gap (net : TwoLayerNet d m) (X : Fin n → Fin d → ℝ)
    (y : Parameters n) (η lam : ℝ) {A L M E S : ℝ}
    (hlam : 0 < lam) (hη : 0 < η)
    (hA : 0 ≤ A) (hL : 0 ≤ L) (hM : 0 ≤ M) (hE : 0 ≤ E) (hS : 0 ≤ S)
    (hdecay : |1 - η * lam| ≤ 1)
    (hstep : η * (Ridge.smoothness (design net X) (n : ℝ)⁻¹ lam : ℝ) ^ 2 < 2 * lam)
    (hW : ∀ i, |net.W₂ 0 i| ≤ A) (hσ : ∀ u, |deriv net.act.σ u| ≤ L)
    (hLip : ∀ a b, |deriv net.act.σ a - deriv net.act.σ b| ≤ M * |a - b|)
    (hX : ∀ r, inputEnergy (X r) ≤ E)
    (hinit : ‖initialOutput net X - y‖ ≤ S) (T : ℕ) :
    let D := batchShiftEnvelope m n η A L E S T
    let R := batchResidualEnvelope m n η A L E S T
    let B := Real.sqrt n * (M * m * A * D ^ 2)
    let G := Real.sqrt n * (Real.sqrt m * A * M * D * Real.sqrt E)
    ‖batchOutput net X (nonlinearBatchTrajectory net X y η lam T) -
      predict net X (linearBatchTrajectory net X y η lam T)‖ ≤
      B + ‖design net X‖ * (T * ((η * (n : ℝ)⁻¹) * (‖design net X‖ * B + G * R))) := by
  dsimp only
  let uT := nonlinearBatchTrajectory net X y η lam T
  let vT := linearBatchTrajectory net X y η lam T
  let D := batchShiftEnvelope m n η A L E S T
  let R := batchResidualEnvelope m n η A L E S T
  let B := Real.sqrt n * (M * m * A * D ^ 2)
  let G := Real.sqrt n * (Real.sqrt m * A * M * D * Real.sqrt E)
  have hD : 0 ≤ D := by
    dsimp [D, batchShiftEnvelope, batchNormEnvelope, batchGrowth]
    positivity
  have hrem : ‖batchOutput net X uT - predict net X uT‖ ≤ B := by
    have h := output_remainder_le net X (unflatten uT) hA hM hD hW hLip
      (batch_shift_le net X y η lam hη.le hA hL hE hS hdecay hW hσ hX hinit T T le_rfl)
    simpa [batchOutput, flatten_unflatten, B] using h
  have hpred : ‖predict net X uT - predict net X vT‖ ≤
      ‖design net X‖ * (T * ((η * (n : ℝ)⁻¹) * (‖design net X‖ * B + G * R))) := by
    have hsub : predict net X uT - predict net X vT = design net X (uT - vT) := by
      simp [predict, map_sub]
    rw [hsub]
    exact ((design net X).le_opNorm (uT - vT)).trans
      (mul_le_mul_of_nonneg_left
        (nonlinear_linear_param_gap net X y η lam hlam hη hA hL hM hE hS hdecay hstep hW hσ hLip hX hinit T)
        (norm_nonneg _))
  have htri : ‖batchOutput net X uT - predict net X vT‖ ≤
      ‖batchOutput net X uT - predict net X uT‖ + ‖predict net X uT - predict net X vT‖ := by
    simpa using norm_add_le (batchOutput net X uT - predict net X uT) (predict net X uT - predict net X vT)
  exact htri.trans (add_le_add hrem hpred)

/-- Multi-input prediction gap budget between T-step nonlinear batch GD and the
initial-Jacobian ridge prediction. -/
def batchGap (T m n : ℕ) (η lam A L M E J q S : ℝ) : ℝ :=
  let D := batchShiftEnvelope m n η A L E S T
  let R := batchResidualEnvelope m n η A L E S T
  let B := Real.sqrt n * (M * m * A * D ^ 2)
  let G := Real.sqrt n * (Real.sqrt m * A * M * D * Real.sqrt E)
  B + J * (T * ((η * (n : ℝ)⁻¹) * (J * B + G * R))) +
    Real.sqrt (q ^ T) * (S / Real.sqrt (n * lam) * J)

/-- Distance from the initial-Jacobian ridge solution to zero bounded via `Ridge.minimizer_norm_le`. -/
theorem ridgeSolution_norm_le_of_init (net : TwoLayerNet d m) (X : Fin n → Fin d → ℝ)
    (y : Parameters n) (lam : ℝ) (hn : 0 < n) (hlam : 0 < lam)
    {S : ℝ} (hS : 0 ≤ S) (hinit : ‖initialOutput net X - y‖ ≤ S) :
    ‖RidgeRisk.ridgeSolution (design net X) (y - initialOutput net X)
      (n : ℝ)⁻¹ lam (inv_pos.mpr (Nat.cast_pos.mpr hn)) hlam‖ ≤
      S / Real.sqrt (n * lam) := by
  have hn0 : (0 : ℝ) < n := Nat.cast_pos.mpr hn
  have hnlam : 0 < (n : ℝ) * lam := mul_pos hn0 hlam
  have hsqrt : 0 < Real.sqrt (n * lam) := Real.sqrt_pos.mpr hnlam
  have hR : 0 ≤ S / Real.sqrt (n * lam) := div_nonneg hS hsqrt.le
  apply Ridge.minimizer_norm_le (design net X) (y - initialOutput net X)
    (n : ℝ)⁻¹ lam (S / Real.sqrt (n * lam)) (inv_pos.mpr hn0).le hlam hR
  · have hy : ‖y - initialOutput net X‖ ^ 2 ≤ S ^ 2 := by
      rw [norm_sub_rev]
      exact (sq_le_sq₀ (norm_nonneg _) hS).mpr hinit
    calc
      (n : ℝ)⁻¹ * ‖y - initialOutput net X‖ ^ 2 ≤ (n : ℝ)⁻¹ * S ^ 2 :=
        mul_le_mul_of_nonneg_left hy (inv_nonneg.mpr hn0.le)
      _ = lam * (S / Real.sqrt (n * lam)) ^ 2 := by
        rw [div_pow, Real.sq_sqrt hnlam.le]
        field_simp
  · exact RidgeRisk.ridgeSolution_minimizes (design net X) (y - initialOutput net X)
      (n : ℝ)⁻¹ lam (inv_pos.mpr hn0) hlam

/-- Fully discharged deterministic distance between the T-step nonlinear batch MLP output vector
and the centered initial-Jacobian ridge prediction (`MLPFixedDesign.ridgePrediction`). -/
theorem batch_trained_ridge_gap (net : TwoLayerNet d m) (X : Fin n → Fin d → ℝ)
    (y : Parameters n) (η lam : ℝ) {A L M E S : ℝ}
    (hn : 0 < n) (hlam : 0 < lam) (hη : 0 < η)
    (hA : 0 ≤ A) (hL : 0 ≤ L) (hM : 0 ≤ M) (hE : 0 ≤ E) (hS : 0 ≤ S)
    (hdecay : |1 - η * lam| ≤ 1)
    (hstep : η * (Ridge.smoothness (design net X) (n : ℝ)⁻¹ lam : ℝ) ^ 2 < 2 * lam)
    (hW : ∀ i, |net.W₂ 0 i| ≤ A) (hσ : ∀ u, |deriv net.act.σ u| ≤ L)
    (hLip : ∀ a b, |deriv net.act.σ a - deriv net.act.σ b| ≤ M * |a - b|)
    (hX : ∀ r, inputEnergy (X r) ≤ E)
    (hinit : ‖initialOutput net X - y‖ ≤ S) (T : ℕ) :
    ‖batchOutput net X (nonlinearBatchTrajectory net X y η lam T) -
      ridgePrediction net X y lam hn hlam‖ ≤
      batchGap T m n η lam A L M E ‖design net X‖
        (Optim.GD.q η lam (Ridge.smoothness (design net X) (n : ℝ)⁻¹ lam)) S := by
  let uT := nonlinearBatchTrajectory net X y η lam T
  let vT := linearBatchTrajectory net X y η lam T
  let uStar := RidgeRisk.ridgeSolution (design net X) (y - initialOutput net X)
    (n : ℝ)⁻¹ lam (inv_pos.mpr (Nat.cast_pos.mpr hn)) hlam
  have hnl := nonlinear_linear_output_gap net X y η lam hlam hη hA hL hM hE hS
    hdecay hstep hW hσ hLip hX hinit T
  obtain ⟨uStar', huStar', hgd⟩ := gd_to_ridge net X y lam hn hlam η hη hstep
  have hgd0 : ‖vT - uStar‖ ^ 2 ≤
      (Optim.GD.q η lam (Ridge.smoothness (design net X) (n : ℝ)⁻¹ lam)) ^ T * ‖uStar‖ ^ 2 := by
    have h0 := hgd 0 T
    rw [huStar'] at h0
    simpa [vT, linearBatchTrajectory, linearBatchStep, zero_sub, norm_neg, uStar] using h0
  have hμL : lam ≤ (Ridge.smoothness (design net X) (n : ℝ)⁻¹ lam : ℝ) := by
    simp only [Ridge.smoothness, NNReal.coe_add, coe_nnnorm, Real.norm_eq_abs]
    linarith [norm_nonneg (Ridge.hessian (design net X) (n : ℝ)⁻¹ lam), le_abs_self lam]
  have hqnn := Optim.GD.q_nonneg_of_le η lam
    (Ridge.smoothness (design net X) (n : ℝ)⁻¹ lam) hlam.le hμL
  have hstar_norm := ridgeSolution_norm_le_of_init net X y lam hn hlam hS hinit
  have hRnn : 0 ≤ S / Real.sqrt (n * lam) :=
    div_nonneg hS (Real.sqrt_nonneg _)
  have hdist : ‖vT - uStar‖ ≤
      Real.sqrt ((Optim.GD.q η lam (Ridge.smoothness (design net X) (n : ℝ)⁻¹ lam)) ^ T) *
        (S / Real.sqrt (n * lam)) := by
    have h1 : ‖vT - uStar‖ ^ 2 ≤
        (Real.sqrt ((Optim.GD.q η lam (Ridge.smoothness (design net X) (n : ℝ)⁻¹ lam)) ^ T) *
          (S / Real.sqrt (n * lam))) ^ 2 := by
      rw [mul_pow, Real.sq_sqrt (pow_nonneg hqnn T)]
      exact hgd0.trans (mul_le_mul_of_nonneg_left
        ((sq_le_sq₀ (norm_nonneg _) hRnn).mpr hstar_norm) (pow_nonneg hqnn T))
    exact (sq_le_sq₀ (norm_nonneg _) (mul_nonneg (Real.sqrt_nonneg _) hRnn)).mp h1
  have hopt_pred : ‖predict net X vT - ridgePrediction net X y lam hn hlam‖ ≤
      Real.sqrt ((Optim.GD.q η lam (Ridge.smoothness (design net X) (n : ℝ)⁻¹ lam)) ^ T) *
        (S / Real.sqrt (n * lam) * ‖design net X‖) := by
    have hsub : predict net X vT - ridgePrediction net X y lam hn hlam =
        design net X (vT - uStar) := by
      simp [ridgePrediction, predict, uStar, map_sub]
    rw [hsub]
    calc
      ‖design net X (vT - uStar)‖ ≤ ‖design net X‖ * ‖vT - uStar‖ :=
        (design net X).le_opNorm (vT - uStar)
      _ ≤ ‖design net X‖ *
          (Real.sqrt ((Optim.GD.q η lam (Ridge.smoothness (design net X) (n : ℝ)⁻¹ lam)) ^ T) *
            (S / Real.sqrt (n * lam))) :=
        mul_le_mul_of_nonneg_left hdist (norm_nonneg _)
      _ = Real.sqrt ((Optim.GD.q η lam (Ridge.smoothness (design net X) (n : ℝ)⁻¹ lam)) ^ T) *
          (S / Real.sqrt (n * lam) * ‖design net X‖) := by ring
  have htri : ‖batchOutput net X uT - ridgePrediction net X y lam hn hlam‖ ≤
      ‖batchOutput net X uT - predict net X vT‖ +
        ‖predict net X vT - ridgePrediction net X y lam hn hlam‖ := by
    simpa using norm_add_le
      (batchOutput net X uT - predict net X vT)
      (predict net X vT - ridgePrediction net X y lam hn hlam)
  exact htri.trans (by
    unfold batchGap
    exact add_le_add hnl hopt_pred)

end OpenTheory.MLPBatchTrajectory
