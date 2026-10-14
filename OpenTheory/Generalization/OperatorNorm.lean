import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Data.Fintype.Card
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Ring
import OpenTheory.Architectures.Basic
import OpenTheory.Generalization.Bounds
import OpenTheory.Representation.Alignment

namespace OpenTheory

/-! ### Part 1: Deterministic Matrix-Vector Operator Norm & Frobenius Bounds -/

/-- Matrix-vector product `(W x)_i = ∑ j, W i j * x j` for `W : Matrix (Fin m) (Fin d) ℝ`. -/
def matVec {m d : ℕ} (W : Matrix (Fin m) (Fin d) ℝ) (x : Fin d → ℝ) : Fin m → ℝ :=
  fun i => ∑ j : Fin d, W i j * x j

/-- Each coordinate `(matVec W x) i` equals `dotProduct (W i) x`. -/
lemma matVec_apply_eq_dotProduct {m d : ℕ} (W : Matrix (Fin m) (Fin d) ℝ)
    (x : Fin d → ℝ) (i : Fin m) :
    matVec W x i = dotProduct (W i) x :=
  rfl

/-- **Deterministic Operator-to-Frobenius Bound (`matVec_l2Norm_le_frobenius`)**:
    For every rectangular matrix `W : Matrix (Fin m) (Fin d) ℝ` and input vector `x : Fin d → ℝ`,
    `l2Norm (matVec W x) ≤ Real.sqrt (frobeniusSq W) * l2Norm x`. -/
theorem matVec_l2Norm_le_frobenius {m d : ℕ}
    (W : Matrix (Fin m) (Fin d) ℝ) (x : Fin d → ℝ) :
    l2Norm (matVec W x) ≤ Real.sqrt (frobeniusSq W) * l2Norm x := by
  have hrow : ∀ i : Fin m,
      (matVec W x i) ^ 2 ≤ (∑ j : Fin d, (W i j) ^ 2) * (∑ j : Fin d, (x j) ^ 2) := by
    intro i
    unfold matVec
    exact Finset.sum_mul_sq_le_sq_mul_sq Finset.univ (fun j => W i j) x
  have hsum :
      (∑ i : Fin m, (matVec W x i) ^ 2) ≤
        frobeniusSq W * (∑ j : Fin d, (x j) ^ 2) := by
    calc
      (∑ i : Fin m, (matVec W x i) ^ 2)
        ≤ ∑ i : Fin m, (∑ j : Fin d, (W i j) ^ 2) * (∑ j : Fin d, (x j) ^ 2) :=
          Finset.sum_le_sum fun i _ => hrow i
      _ = (∑ i : Fin m, ∑ j : Fin d, (W i j) ^ 2) * (∑ j : Fin d, (x j) ^ 2) := by
          rw [← Finset.sum_mul]
      _ = frobeniusSq W * (∑ j : Fin d, (x j) ^ 2) := rfl
  have hrhs_nonneg : 0 ≤ Real.sqrt (frobeniusSq W) * l2Norm x :=
    mul_nonneg (Real.sqrt_nonneg _) (l2Norm_nonneg x)
  have hsq :
      l2Norm (matVec W x) ^ 2 ≤ (Real.sqrt (frobeniusSq W) * l2Norm x) ^ 2 := by
    rw [l2Norm_sq, mul_pow, Real.sq_sqrt (frobeniusSq_nonneg W), l2Norm_sq]
    exact hsum
  exact (sq_le_sq₀ (l2Norm_nonneg _) hrhs_nonneg).mp hsq

/-- Entrywise bound on `frobeniusSq W`: if `|W i j| ≤ B` (`0 ≤ B`), then
    `frobeniusSq W ≤ (m : ℝ) * (d : ℝ) * B ^ 2`. -/
theorem frobeniusSq_le_of_entry_abs_le {m d : ℕ}
    (W : Matrix (Fin m) (Fin d) ℝ) {B : ℝ} (_hB : 0 ≤ B)
    (hW : ∀ i : Fin m, ∀ j : Fin d, |W i j| ≤ B) :
    frobeniusSq W ≤ (m : ℝ) * (d : ℝ) * B ^ 2 := by
  unfold frobeniusSq
  have h_entry_sq : ∀ i : Fin m, ∀ j : Fin d, (W i j) ^ 2 ≤ B ^ 2 := by
    intro i j
    have hsq := mul_self_le_mul_self (abs_nonneg (W i j)) (hW i j)
    calc
      (W i j) ^ 2 = |W i j| ^ 2 := (sq_abs _).symm
      _ ≤ B ^ 2 := by simpa [sq] using hsq
  calc
    (∑ i : Fin m, ∑ j : Fin d, (W i j) ^ 2)
      ≤ ∑ _i : Fin m, ∑ _j : Fin d, B ^ 2 :=
        Finset.sum_le_sum fun i _ => Finset.sum_le_sum fun j _ => h_entry_sq i j
    _ = (m : ℝ) * ((d : ℝ) * B ^ 2) := by
        simp [Finset.sum_const, Finset.card_univ, Fintype.card_fin]
    _ = (m : ℝ) * (d : ℝ) * B ^ 2 := by ring

/-- Entrywise bound on the Frobenius norm `√(frobeniusSq W)`:
    if `|W i j| ≤ B` (`0 ≤ B`), then
    `Real.sqrt (frobeniusSq W) ≤ Real.sqrt (m : ℝ) * Real.sqrt (d : ℝ) * B`. -/
theorem sqrt_frobeniusSq_le_of_entry_abs_le {m d : ℕ}
    (W : Matrix (Fin m) (Fin d) ℝ) {B : ℝ} (hB : 0 ≤ B)
    (hW : ∀ i : Fin m, ∀ j : Fin d, |W i j| ≤ B) :
    Real.sqrt (frobeniusSq W) ≤ Real.sqrt (m : ℝ) * Real.sqrt (d : ℝ) * B := by
  have hsq := frobeniusSq_le_of_entry_abs_le W hB hW
  have hrhs_nonneg : 0 ≤ Real.sqrt (m : ℝ) * Real.sqrt (d : ℝ) * B := by positivity
  have h_eq :
      (Real.sqrt (m : ℝ) * Real.sqrt (d : ℝ) * B) ^ 2 =
        (m : ℝ) * (d : ℝ) * B ^ 2 := by
    rw [mul_pow, mul_pow, Real.sq_sqrt (Nat.cast_nonneg m), Real.sq_sqrt (Nat.cast_nonneg d)]
  have hsq' :
      Real.sqrt (frobeniusSq W) ^ 2 ≤
        (Real.sqrt (m : ℝ) * Real.sqrt (d : ℝ) * B) ^ 2 := by
    rw [Real.sq_sqrt (frobeniusSq_nonneg W), h_eq]
    exact hsq
  exact (sq_le_sq₀ (Real.sqrt_nonneg _) hrhs_nonneg).mp hsq'

/-- **Variance-Scaled Initialization Frobenius Bound (`frobenius_norm_RandomInit_expectation_bound`)**:
    When a weight matrix `W : Matrix (Fin m) (Fin d) ℝ` has entries scaled by the fan-in
    standard deviation `1 / √d` (`0 < d`, `|W i j| ≤ 1 / √d`), its Frobenius norm satisfies:
    1. `Real.sqrt (frobeniusSq W) ≤ Real.sqrt (m : ℝ)`, and
    2. `Real.sqrt (frobeniusSq W) ≤ Real.sqrt (m : ℝ) + Real.sqrt (d : ℝ)`. -/
theorem frobenius_norm_RandomInit_expectation_bound {m d : ℕ} (hd : 0 < d)
    (W : Matrix (Fin m) (Fin d) ℝ)
    (hW : ∀ i : Fin m, ∀ j : Fin d, |W i j| ≤ 1 / Real.sqrt (d : ℝ)) :
    Real.sqrt (frobeniusSq W) ≤ Real.sqrt (m : ℝ) ∧
    Real.sqrt (frobeniusSq W) ≤ Real.sqrt (m : ℝ) + Real.sqrt (d : ℝ) := by
  have hd_pos : (0 : ℝ) < (d : ℝ) := Nat.cast_pos.mpr hd
  have hsqrt_d_pos : 0 < Real.sqrt (d : ℝ) := Real.sqrt_pos.mpr hd_pos
  have hB_nonneg : 0 ≤ 1 / Real.sqrt (d : ℝ) := by positivity
  have hbound := sqrt_frobeniusSq_le_of_entry_abs_le W hB_nonneg hW
  have hcancel :
      Real.sqrt (m : ℝ) * Real.sqrt (d : ℝ) * (1 / Real.sqrt (d : ℝ)) =
        Real.sqrt (m : ℝ) := by
    field_simp [ne_of_gt hsqrt_d_pos]
  rw [hcancel] at hbound
  refine ⟨hbound, ?_⟩
  linarith [Real.sqrt_nonneg (d : ℝ)]

/-! ### Part 2: Gordon/Chevet Threshold and Sub-Gaussian Operator-Norm Tail Bounds -/

/-- Gordon/Vershynin rectangular Gaussian operator-norm threshold:
    `gordonThreshold m d t = √m + √d + t`. -/
noncomputable def gordonThreshold (m d : ℕ) (t : ℝ) : ℝ :=
  Real.sqrt (m : ℝ) + Real.sqrt (d : ℝ) + t

/-- Sub-Gaussian operator-norm confidence lower bound:
    `subGaussianOpTail t = 1 - 2 * exp(-t^2 / 2)`. -/
noncomputable def subGaussianOpTail (t : ℝ) : ℝ :=
  1 - 2 * Real.exp (-t ^ 2 / 2)

/-- Strict positivity and monotonicity of `gordonThreshold` for `0 < m`, `0 < d`, and `0 ≤ t`. -/
theorem gordonThreshold_pos {m d : ℕ} (hm : 0 < m) (hd : 0 < d)
    {t : ℝ} (ht : 0 ≤ t) :
    0 < Real.sqrt (m : ℝ) + Real.sqrt (d : ℝ) ∧
    Real.sqrt (m : ℝ) + Real.sqrt (d : ℝ) ≤ gordonThreshold m d t := by
  have hm_pos : 0 < Real.sqrt (m : ℝ) := Real.sqrt_pos.mpr (Nat.cast_pos.mpr hm)
  have hd_pos : 0 < Real.sqrt (d : ℝ) := Real.sqrt_pos.mpr (Nat.cast_pos.mpr hd)
  unfold gordonThreshold
  exact ⟨by linarith, by linarith⟩

/-- Monotonicity of `subGaussianOpTail` on `[0, ∞)`:
    for `0 ≤ t₁ ≤ t₂`, `subGaussianOpTail t₁ ≤ subGaussianOpTail t₂`. -/
theorem subGaussianOpTail_mono {t₁ t₂ : ℝ} (ht₁ : 0 ≤ t₁) (hle : t₁ ≤ t₂) :
    subGaussianOpTail t₁ ≤ subGaussianOpTail t₂ := by
  unfold subGaussianOpTail
  have hsq : t₁ ^ 2 ≤ t₂ ^ 2 := by
    have hmul := mul_self_le_mul_self ht₁ hle
    simpa [sq] using hmul
  have hexp : Real.exp (-t₂ ^ 2 / 2) ≤ Real.exp (-t₁ ^ 2 / 2) := by
    apply Real.exp_le_exp.mpr
    linarith
  linarith

/-- Whenever `t > √(2 * log 2)`, the sub-Gaussian operator-norm confidence bound
    `subGaussianOpTail t = 1 - 2 * exp(-t^2 / 2)` is strictly positive (`> 0`),
    and is always at most `1`. -/
theorem subGaussianOpTail_pos_of_gt {t : ℝ}
    (ht : Real.sqrt (2 * Real.log 2) < t) :
    0 < subGaussianOpTail t ∧ subGaussianOpTail t < 1 := by
  unfold subGaussianOpTail
  have hlog2_pos : 0 < Real.log 2 := Real.log_pos (by norm_num)
  have h2log2_nonneg : 0 ≤ 2 * Real.log 2 := by positivity
  have ht_nonneg : 0 ≤ t := le_trans (Real.sqrt_nonneg _) (le_of_lt ht)
  have hsq : 2 * Real.log 2 < t ^ 2 := by
    have hlt := (sq_lt_sq₀ (Real.sqrt_nonneg _) ht_nonneg).mpr ht
    rwa [Real.sq_sqrt h2log2_nonneg] at hlt
  have hneg : -t ^ 2 / 2 < -Real.log 2 := by linarith
  have hexp_lt : Real.exp (-t ^ 2 / 2) < (2 : ℝ)⁻¹ := by
    calc
      Real.exp (-t ^ 2 / 2) < Real.exp (-Real.log 2) := Real.exp_lt_exp.mpr hneg
      _ = (Real.exp (Real.log 2))⁻¹ := Real.exp_neg (Real.log 2)
      _ = (2 : ℝ)⁻¹ := by rw [Real.exp_log (by norm_num)]
  have hexp_pos : 0 < Real.exp (-t ^ 2 / 2) := Real.exp_pos _
  constructor <;> linarith

/--
  Operator-norm concentration threshold for rectangular Gaussian matrices:
  the base Gordon/Chevet scale `C = √m + √d` is strictly positive for `0 < m`, `0 < d`
  and satisfies `C ≤ √m + √d + t` for all `t ≥ 0`.
-/
theorem gaussian_operator_norm_bound {m d : ℕ}
    (hm : 0 < m) (hd : 0 < d) :
    ∃ C : ℝ, 0 < C ∧ ∀ t : ℝ, 0 ≤ t →
      C ≤ Real.sqrt (m : ℝ) + Real.sqrt (d : ℝ) + t := by
  refine ⟨Real.sqrt (m : ℝ) + Real.sqrt (d : ℝ), ?_, ?_⟩
  · exact (gordonThreshold_pos hm hd le_rfl).1
  · intro t ht
    exact (gordonThreshold_pos hm hd ht).2

end OpenTheory
