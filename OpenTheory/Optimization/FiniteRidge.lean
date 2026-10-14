import OpenTheory.Data.Foundation
import OpenTheory.Optimization.Ridge
import Mathlib.Topology.Algebra.Module.FiniteDimension

noncomputable section
open scoped RealInnerProductSpace

namespace OpenTheory.Dataset

def observation {d n : ℕ} (D : Dataset (Parameters d) ℝ n) :
    Parameters d →L[ℝ] Parameters n :=
  LinearMap.toContinuousLinearMap {
    toFun := fun θ => WithLp.toLp 2 (fun i => ⟪(D i).1, θ⟫)
    map_add' := by
      intro x y
      ext i
      exact inner_add_right _ _ _
    map_smul' := by
      intro c x
      ext i
      exact real_inner_smul_right _ _ _ }

def labels {d n : ℕ} (D : Dataset (Parameters d) ℝ n) : Parameters n :=
  WithLp.toLp 2 (fun i => (D i).2)

def ridgeLoss {d n : ℕ} (D : Dataset (Parameters d) ℝ n) (lam : ℝ) : Parameters d → ℝ :=
  Ridge.objective D.observation D.labels (1 / n) lam

/-- The operator objective is exactly half mean squared prediction loss plus ridge regularization. -/
theorem ridgeLoss_eq {d n : ℕ} (D : Dataset (Parameters d) ℝ n) (lam : ℝ) (θ : Parameters d) :
    D.ridgeLoss lam θ =
      (∑ i, (⟪θ, (D i).1⟫ - (D i).2) ^ 2) / (2 * n) + lam / 2 * ‖θ‖ ^ 2 := by
  simp only [ridgeLoss, Ridge.objective, parameters_norm_sq]
  have he : ∀ i, (D.observation θ - D.labels) i = ⟪θ, (D i).1⟫ - (D i).2 := by
    intro i
    change ⟪(D i).1, θ⟫ - (D i).2 = _
    rw [real_inner_comm]
  simp only [he]
  ring

/-- Finite-data instantiation of the upstream GD bound with all analytic premises discharged. -/
theorem ridge_training {d n : ℕ} (D : Dataset (Parameters d) ℝ n) (_hn : 0 < n)
    (lam : ℝ) (hlam : 0 < lam) (η : ℝ) (hη : 0 < η)
    (hstep : η * (Ridge.smoothness D.observation (1 / n) lam : ℝ) ^ 2 < 2 * lam) :
    ∃ θStar, (∀ θ, D.ridgeLoss lam θStar ≤ D.ridgeLoss lam θ) ∧
      ∀ θ k, ‖(Optim.GD.step η (gradient (D.ridgeLoss lam)))^[k] θ - θStar‖ ^ 2 ≤
        (Optim.GD.q η lam (Ridge.smoothness D.observation (1 / n) lam)) ^ k * ‖θ - θStar‖ ^ 2 ∧
        Optim.GD.q η lam (Ridge.smoothness D.observation (1 / n) lam) < 1 := by
  exact Ridge.training D.observation D.labels (1 / n) lam
    (by positivity) hlam η hη hstep

/-- Bounded labels give a sample-independent mean response-energy bound. -/
theorem labels_energy_le {d n : ℕ} (D : Dataset (Parameters d) ℝ n)
    (hn : 0 < n) (T : ℝ) (hT : 0 ≤ T) (hy : ∀ i, |(D i).2| ≤ T) :
    (1 / (n : ℝ)) * ‖D.labels‖ ^ 2 ≤ T ^ 2 := by
  have hs : ‖D.labels‖ ^ 2 ≤ (n : ℝ) * T ^ 2 := by
    rw [parameters_norm_sq]
    calc
      _ ≤ ∑ _ : Fin n, T ^ 2 := by
        apply Finset.sum_le_sum
        intro i _
        change (D i).2 ^ 2 ≤ T ^ 2
        exact (sq_le_sq).mpr (by simpa [abs_of_nonneg hT] using hy i)
      _ = _ := by simp
  have hn' : (0 : ℝ) < n := by exact_mod_cast hn
  rw [one_div, mul_comm, ← div_eq_mul_inv]
  exact (div_le_iff₀ hn').mpr (by simpa [mul_comm] using hs)

/-- A fixed label bound supplies one ball containing every zero-initialized
training trajectory. The admissible step may still depend on the dataset. -/
theorem ridge_training_radius {d n : ℕ} (D : Dataset (Parameters d) ℝ n)
    (hn : 0 < n) (T : ℝ) (hT : 0 ≤ T) (hy : ∀ i, |(D i).2| ≤ T)
    (lam R : ℝ) (hlam : 0 < lam) (hR : 0 ≤ R) (hTR : T ^ 2 ≤ lam * R ^ 2)
    (η : ℝ) (hη : 0 < η)
    (hstep : η * (Ridge.smoothness D.observation (1 / n) lam : ℝ) ^ 2 < 2 * lam) :
    ∃ θStar, (∀ θ, D.ridgeLoss lam θStar ≤ D.ridgeLoss lam θ) ∧
      ‖θStar‖ ≤ R ∧ ∀ k,
        ‖(Optim.GD.step η (gradient (D.ridgeLoss lam)))^[k] 0‖ ≤ 2 * R :=
  Ridge.training_radius D.observation D.labels (1 / n) lam R (by positivity)
    hlam hR ((labels_energy_le D hn T hT hy).trans hTR) η hη hstep

/-- A smoothness majorant depending only on an input bound and the regularizer. -/
def uniformSmoothness (B lam : ℝ) : ℝ := B ^ 2 + 2 * lam

/-- One positive step legal for every design whose rows are bounded by `B`. -/
def uniformStep (B lam : ℝ) : ℝ := lam / uniformSmoothness B lam ^ 2

/-- Uniform geometric rate obtained by enlarging the dataset-dependent factor. -/
def uniformFactor (B lam : ℝ) : ℝ := 1 - (lam / uniformSmoothness B lam) ^ 2

theorem observation_apply {d n : ℕ} (D : Dataset (Parameters d) ℝ n)
    (θ : Parameters d) (i : Fin n) :
    D.observation θ i = ⟪(D i).1, θ⟫ := by
  rfl

theorem observation_norm_le {d n : ℕ} (D : Dataset (Parameters d) ℝ n)
    (B : ℝ) (hB : 0 ≤ B) (hBx : ∀ i, ‖(D i).1‖ ≤ B) (θ : Parameters d) :
    ‖D.observation θ‖ ≤ Real.sqrt (n : ℝ) * B * ‖θ‖ := by
  have hs : ‖D.observation θ‖ ^ 2 ≤ (n : ℝ) * B ^ 2 * ‖θ‖ ^ 2 := by
    rw [parameters_norm_sq]
    calc
      _ ≤ ∑ i : Fin n, (‖(D i).1‖ * ‖θ‖) ^ 2 := by
        apply Finset.sum_le_sum
        intro i _
        have hinner := abs_real_inner_le_norm (D i).1 θ
        rw [observation_apply]
        have hprod : 0 ≤ ‖(D i).1‖ * ‖θ‖ := mul_nonneg (norm_nonneg _) (norm_nonneg _)
        exact sq_le_sq.mpr (hinner.trans_eq (abs_of_nonneg hprod).symm)
      _ ≤ ∑ _ : Fin n, (B * ‖θ‖) ^ 2 := by
        apply Finset.sum_le_sum
        intro i _
        exact pow_le_pow_left₀ (mul_nonneg (norm_nonneg _) (norm_nonneg _))
          (mul_le_mul_of_nonneg_right (hBx i) (norm_nonneg _)) 2
      _ = _ := by simp [mul_pow, mul_assoc]
  have hnn : 0 ≤ Real.sqrt (n : ℝ) * B * ‖θ‖ :=
    mul_nonneg (mul_nonneg (Real.sqrt_nonneg _) hB) (norm_nonneg _)
  have hsq : ‖D.observation θ‖ ^ 2 ≤ (Real.sqrt (n : ℝ) * B * ‖θ‖) ^ 2 := by
    calc
      _ ≤ (n : ℝ) * B ^ 2 * ‖θ‖ ^ 2 := hs
      _ = (Real.sqrt (n : ℝ) ^ 2) * B ^ 2 * ‖θ‖ ^ 2 := by rw [Real.sq_sqrt (Nat.cast_nonneg n)]
      _ = _ := by ring
  exact (sq_le_sq₀ (norm_nonneg _) hnn).mp hsq

theorem observation_opNorm_le {d n : ℕ} (D : Dataset (Parameters d) ℝ n)
    (B : ℝ) (hB : 0 ≤ B) (hBx : ∀ i, ‖(D i).1‖ ≤ B) :
    ‖D.observation‖ ≤ Real.sqrt (n : ℝ) * B := by
  apply ContinuousLinearMap.opNorm_le_bound
  · exact mul_nonneg (Real.sqrt_nonneg _) hB
  · intro θ
    simpa [mul_assoc] using observation_norm_le D B hB hBx θ

theorem smoothness_le_uniform {d n : ℕ} (D : Dataset (Parameters d) ℝ n)
    (hn : 0 < n) (B lam : ℝ) (hB : 0 ≤ B) (hlam : 0 ≤ lam)
    (hBx : ∀ i, ‖(D i).1‖ ≤ B) :
    (Ridge.smoothness D.observation (1 / n) lam : ℝ) ≤ uniformSmoothness B lam := by
  have hA := observation_opNorm_le D B hB hBx
  have hAsq : ‖D.observation‖ ^ 2 ≤ (n : ℝ) * B ^ 2 := by
    have hnn : 0 ≤ Real.sqrt (n : ℝ) * B := mul_nonneg (Real.sqrt_nonneg _) hB
    have := (sq_le_sq₀ (norm_nonneg _) hnn).mpr hA
    simpa [mul_pow, Real.sq_sqrt (Nat.cast_nonneg n), mul_assoc] using this
  have hop : ‖((1 / (n : ℝ)) • (D.observation.adjoint ∘L D.observation))‖ ≤ B ^ 2 := by
    have hsmul := ContinuousLinearMap.opNorm_smul_le (1 / (n : ℝ))
      (D.observation.adjoint ∘L D.observation)
    have hadj := ContinuousLinearMap.norm_adjoint_comp_self D.observation
    calc
      ‖(1 / (n : ℝ)) • (D.observation.adjoint ∘L D.observation)‖
          ≤ ‖(1 / (n : ℝ) : ℝ)‖ * ‖D.observation.adjoint ∘L D.observation‖ := hsmul
      _ = (1 / (n : ℝ)) * ‖D.observation‖ ^ 2 := by
        rw [Real.norm_eq_abs, abs_of_nonneg (by positivity), hadj]
        ring
      _ ≤ (1 / (n : ℝ)) * ((n : ℝ) * B ^ 2) :=
        mul_le_mul_of_nonneg_left hAsq (by positivity)
      _ = B ^ 2 := by field_simp
  have hid : ‖lam • ContinuousLinearMap.id ℝ (Parameters d)‖ ≤ lam := by
    have hsmul := ContinuousLinearMap.opNorm_smul_le lam
      (ContinuousLinearMap.id ℝ (Parameters d))
    calc
      ‖lam • ContinuousLinearMap.id ℝ (Parameters d)‖
          ≤ ‖lam‖ * ‖ContinuousLinearMap.id ℝ (Parameters d)‖ := hsmul
      _ ≤ lam * 1 := by
        rw [Real.norm_eq_abs, abs_of_nonneg hlam]
        exact mul_le_mul_of_nonneg_left ContinuousLinearMap.norm_id_le hlam
      _ = lam := by ring
  have hhess : ‖Ridge.hessian D.observation (1 / n) lam‖ ≤ B ^ 2 + lam := by
    calc
      _ ≤ ‖((1 / n : ℝ) • (D.observation.adjoint ∘L D.observation))‖ +
          ‖lam • ContinuousLinearMap.id ℝ (Parameters d)‖ := norm_add_le _ _
      _ ≤ B ^ 2 + lam := add_le_add hop hid
  have hsmooth : (Ridge.smoothness D.observation (1 / n) lam : ℝ) =
      ‖Ridge.hessian D.observation (1 / n) lam‖ + |lam| := by
    simp [Ridge.smoothness, NNReal.coe_add, coe_nnnorm]
  rw [hsmooth, abs_of_nonneg hlam, uniformSmoothness]
  linarith

theorem uniformSmoothness_pos {B lam : ℝ} (_hB : 0 ≤ B) (hlam : 0 < lam) :
    0 < uniformSmoothness B lam := by
  unfold uniformSmoothness
  nlinarith

theorem uniformStep_pos {B lam : ℝ} (hB : 0 ≤ B) (hlam : 0 < lam) :
    0 < uniformStep B lam := by
  unfold uniformStep
  exact div_pos hlam (pow_pos (uniformSmoothness_pos hB hlam) 2)

theorem uniformStep_admissible {d n : ℕ} (D : Dataset (Parameters d) ℝ n)
    (hn : 0 < n) (B lam : ℝ) (hB : 0 ≤ B) (hlam : 0 < lam)
    (hBx : ∀ i, ‖(D i).1‖ ≤ B) :
    uniformStep B lam *
        (Ridge.smoothness D.observation (1 / n) lam : ℝ) ^ 2 < 2 * lam := by
  have hL := smoothness_le_uniform D hn B lam hB hlam.le hBx
  have hL0 := uniformSmoothness_pos hB hlam
  have hsq : (Ridge.smoothness D.observation (1 / n) lam : ℝ) ^ 2 ≤
      uniformSmoothness B lam ^ 2 :=
    pow_le_pow_left₀ (by positivity) hL 2
  have hstep : uniformStep B lam * (Ridge.smoothness D.observation (1 / n) lam : ℝ) ^ 2 ≤
      lam := by
    unfold uniformStep
    calc
      _ ≤ lam / uniformSmoothness B lam ^ 2 * uniformSmoothness B lam ^ 2 :=
        mul_le_mul_of_nonneg_left hsq (div_nonneg hlam.le (sq_nonneg _))
      _ = lam := by field_simp
  linarith

theorem uniformFactor_mem_Icc {B lam : ℝ} (hB : 0 ≤ B) (hlam : 0 < lam) :
    uniformFactor B lam ∈ Set.Icc (0 : ℝ) 1 := by
  have hL0 := uniformSmoothness_pos hB hlam
  have hratio : lam / uniformSmoothness B lam ≤ 1 / 2 := by
    rw [div_le_iff₀ hL0]
    unfold uniformSmoothness
    nlinarith
  have hsq : (lam / uniformSmoothness B lam) ^ 2 ≤ 1 := by
    have hr0 : 0 ≤ lam / uniformSmoothness B lam := div_nonneg hlam.le hL0.le
    nlinarith [hratio, hr0]
  unfold uniformFactor
  refine ⟨by nlinarith, by nlinarith⟩

theorem data_q_le_uniformFactor {d n : ℕ} (D : Dataset (Parameters d) ℝ n)
    (hn : 0 < n) (B lam : ℝ) (hB : 0 ≤ B) (hlam : 0 < lam)
    (hBx : ∀ i, ‖(D i).1‖ ≤ B) :
    Optim.GD.q (uniformStep B lam) lam (Ridge.smoothness D.observation (1 / n) lam) ≤
      uniformFactor B lam := by
  set L : ℝ := (Ridge.smoothness D.observation (1 / n) lam : ℝ)
  set L0 : ℝ := uniformSmoothness B lam
  have hL := smoothness_le_uniform D hn B lam hB hlam.le hBx
  have hL0 := uniformSmoothness_pos hB hlam
  have hsq : L ^ 2 ≤ L0 ^ 2 := pow_le_pow_left₀ (by positivity) hL 2
  have hstep : uniformStep B lam = lam / L0 ^ 2 := rfl
  rw [Optim.GD.q, hstep, uniformFactor]
  have hden : L0 ^ 2 ≠ 0 := ne_of_gt (pow_pos hL0 2)
  have hnum : (lam / L0 ^ 2) ^ 2 * L ^ 2 ≤ (lam / L0 ^ 2) ^ 2 * L0 ^ 2 :=
    mul_le_mul_of_nonneg_left hsq (sq_nonneg _)
  calc
    1 - 2 * (lam / L0 ^ 2) * lam + (lam / L0 ^ 2) ^ 2 * L ^ 2
      ≤ 1 - 2 * (lam / L0 ^ 2) * lam + (lam / L0 ^ 2) ^ 2 * L0 ^ 2 := by linarith
    _ = 1 - (lam / L0) ^ 2 := by field_simp; ring

end OpenTheory.Dataset
