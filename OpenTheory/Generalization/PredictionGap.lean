import OpenTheory.Generalization.RidgeRisk
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Linarith

/-! How close a predictor must be to the ridge predictor.

If `p` is within `δ` of the ridge prediction `q`, then
`‖p - target‖² ≤ 2 ‖q - target‖² + 2 δ²`. The factor two is
`(a - δ)² ≥ 0`, rearranged as `2 a δ ≤ a² + δ²`. Integrating and dividing by
the number of rows turns the fixed-design ridge bound into a bound for `p`,
with an explicit `2 δ² / m` term.

The hypothesis `‖p - q‖ ≤ δ` is not discharged for a trained network. One
gradient step controls the distance to the tangent update, in
`TangentStep.ntk_tangent_gap`, not the distance to the ridge solution.
-/

noncomputable section

open MeasureTheory

namespace OpenTheory.PredictionGap

variable {E : Type*} [SeminormedAddCommGroup E]

theorem sq_norm_le_double (p q t : E) {δ : ℝ} (hδ : 0 ≤ δ) (h : ‖p - q‖ ≤ δ) :
    ‖p - t‖ ^ 2 ≤ 2 * ‖q - t‖ ^ 2 + 2 * δ ^ 2 := by
  have hsum : (q - t) + (p - q) = p - t := by
    simp [sub_eq_add_neg, add_assoc, add_left_comm, add_comm]
  have htri : ‖p - t‖ ≤ ‖q - t‖ + ‖p - q‖ := by
    rw [← hsum]
    exact norm_add_le _ _
  have hle : ‖p - t‖ ≤ ‖q - t‖ + δ := htri.trans (by linarith)
  have hsq : ‖p - t‖ * ‖p - t‖ ≤ (‖q - t‖ + δ) * (‖q - t‖ + δ) :=
    mul_self_le_mul_self (norm_nonneg _) hle
  have hexp : (‖q - t‖ + δ) * (‖q - t‖ + δ) =
      ‖q - t‖ * ‖q - t‖ + 2 * ‖q - t‖ * δ + δ * δ := by ring
  have hcross : 2 * ‖q - t‖ * δ ≤ ‖q - t‖ * ‖q - t‖ + δ * δ := by
    have hnn := mul_nonneg hδ hδ
    have := sq_nonneg (‖q - t‖ - δ)
    rw [sub_sq] at this
    linarith [hnn]
  rw [sq, sq, sq]
  linarith

variable {d m : ℕ}

/-- In-sample squared error of a predictor within `δ` of the ridge predictor.
The integrability hypotheses are the ones used to integrate the pointwise
bound; the ridge side is integrable for white label noise, as in
`RidgeRisk.fixed_design_risk`. -/
theorem expected_mse_of_near_ridge (A : Parameters d →L[ℝ] Parameters m)
    (θStar : Parameters d) (lam : ℝ) (hm : 0 < m) (hlam : 0 < lam)
    {μ : Measure (Parameters m)} {σ δ : ℝ} [IsProbabilityMeasure μ]
    (hnoise : RidgeRisk.LabelNoise μ σ) (hδ : 0 ≤ δ)
    (p : Parameters m → Parameters m)
    (hclose : ∀ z, ‖p z - A (RidgeRisk.ridgeSolution A (A θStar + z) (m : ℝ)⁻¹ lam
      (inv_pos.mpr (Nat.cast_pos.mpr hm)) hlam)‖ ≤ δ)
    (hp : Integrable (fun z => ‖p z - A θStar‖ ^ 2) μ)
    (hq : Integrable (fun z => ‖A (RidgeRisk.ridgeSolution A (A θStar + z) (m : ℝ)⁻¹ lam
      (inv_pos.mpr (Nat.cast_pos.mpr hm)) hlam) - A θStar‖ ^ 2) μ) :
    (∫ z, ‖p z - A θStar‖ ^ 2 ∂μ) / m ≤
      2 * ((lam / 4) * ‖θStar‖ ^ 2 +
        σ ^ 2 * RidgeRisk.degreesOfFreedom A ((m : ℝ)⁻¹) lam
          (inv_pos.mpr (Nat.cast_pos.mpr hm)) hlam / m) +
        2 * δ ^ 2 / m := by
  let q : Parameters m → Parameters m := fun z =>
    A (RidgeRisk.ridgeSolution A (A θStar + z) (m : ℝ)⁻¹ lam
      (inv_pos.mpr (Nat.cast_pos.mpr hm)) hlam)
  have hpoint : ∀ z, ‖p z - A θStar‖ ^ 2 ≤ 2 * ‖q z - A θStar‖ ^ 2 + 2 * δ ^ 2 :=
    fun z => sq_norm_le_double (p z) (q z) (A θStar) hδ (hclose z)
  have hconst : Integrable (fun _ : Parameters m => (2 : ℝ) * δ ^ 2) μ := integrable_const _
  have hupper : Integrable (fun z => 2 * ‖q z - A θStar‖ ^ 2 + 2 * δ ^ 2) μ :=
    (hq.const_mul 2).add hconst
  have hint : ∫ z, ‖p z - A θStar‖ ^ 2 ∂μ ≤
      ∫ z, 2 * ‖q z - A θStar‖ ^ 2 + 2 * δ ^ 2 ∂μ :=
    integral_mono hp hupper hpoint
  rw [integral_add (hq.const_mul 2) hconst, integral_const_mul, integral_const] at hint
  have hprob : μ.real Set.univ = 1 := by simp
  have hsmash : μ.real Set.univ • (2 * δ ^ 2) = 2 * δ ^ 2 := by rw [hprob, one_smul]
  have hmse := RidgeRisk.fixed_design_mse A θStar lam hm hlam hnoise
  have hm0 : (0 : ℝ) < m := Nat.cast_pos.mpr hm
  rw [div_le_iff₀ hm0]
  have hintq : ∫ z, ‖q z - A θStar‖ ^ 2 ∂μ ≤
      m * ((lam / 4) * ‖θStar‖ ^ 2 +
        σ ^ 2 * RidgeRisk.degreesOfFreedom A ((m : ℝ)⁻¹) lam
          (inv_pos.mpr (Nat.cast_pos.mpr hm)) hlam / m) := by
    rw [div_le_iff₀ hm0, mul_comm] at hmse
    simpa [q] using hmse
  have htwo : 2 * (∫ z, ‖q z - A θStar‖ ^ 2 ∂μ) ≤
      2 * (m * ((lam / 4) * ‖θStar‖ ^ 2 +
        σ ^ 2 * RidgeRisk.degreesOfFreedom A ((m : ℝ)⁻¹) lam
          (inv_pos.mpr (Nat.cast_pos.mpr hm)) hlam / m)) :=
    mul_le_mul_of_nonneg_left hintq (by norm_num : (0 : ℝ) ≤ 2)
  calc
    ∫ z, ‖p z - A θStar‖ ^ 2 ∂μ ≤
        2 * (∫ z, ‖q z - A θStar‖ ^ 2 ∂μ) + μ.real Set.univ • (2 * δ ^ 2) := hint
    _ = 2 * (∫ z, ‖q z - A θStar‖ ^ 2 ∂μ) + 2 * δ ^ 2 := by rw [hsmash]
    _ ≤ 2 * (m * ((lam / 4) * ‖θStar‖ ^ 2 +
          σ ^ 2 * RidgeRisk.degreesOfFreedom A ((m : ℝ)⁻¹) lam
            (inv_pos.mpr (Nat.cast_pos.mpr hm)) hlam / m)) +
        2 * δ ^ 2 := by linarith
    _ = (2 * ((lam / 4) * ‖θStar‖ ^ 2 +
          σ ^ 2 * RidgeRisk.degreesOfFreedom A ((m : ℝ)⁻¹) lam
            (inv_pos.mpr (Nat.cast_pos.mpr hm)) hlam / m) +
        2 * δ ^ 2 / m) * m := by field_simp [ne_of_gt hm0]

end OpenTheory.PredictionGap
