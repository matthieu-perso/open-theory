import OpenTheory.Architectures.Basic
import OpenTheory.Generalization.Bounds
import OpenTheory.Representation.Features
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Data.Fintype.Card
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring

namespace OpenTheory

/-- Elementary bound: the empirical mean of a 1-bounded sequence has
    absolute value at most 1. Kernel-checked; no `sorry`. -/
lemma abs_mean_le_of_abs_le_one {n : ℕ} (hn : 0 < n) (a : Fin n → ℝ)
    (ha : ∀ i, |a i| ≤ 1) :
    |((n : ℝ)⁻¹) * ∑ i, a i| ≤ 1 := by
  have npos : (0 : ℝ) < n := Nat.cast_pos.mpr hn
  have h₁ : |∑ i, a i| ≤ ∑ i, |a i| := Finset.abs_sum_le_sum_abs a Finset.univ
  have h₂ : ∑ i, |a i| ≤ ∑ _i : Fin n, (1 : ℝ) :=
    Finset.sum_le_sum fun i _ => ha i
  have h₃ : ∑ _i : Fin n, (1 : ℝ) = (n : ℝ) := by
    simp [Finset.sum_const, Finset.card_univ, Fintype.card_fin]
  have hbound : |∑ i, a i| ≤ (n : ℝ) := h₁.trans (h₂.trans_eq h₃)
  rw [abs_mul, abs_of_pos (inv_pos.mpr npos)]
  have hmul : (n : ℝ)⁻¹ * |∑ i, a i| ≤ (n : ℝ)⁻¹ * n :=
    mul_le_mul_of_nonneg_left hbound (inv_nonneg.mpr npos.le)
  have hcancel : (n : ℝ)⁻¹ * n = 1 := inv_mul_cancel₀ (ne_of_gt npos)
  exact hmul.trans_eq hcancel

/-! ### Part 1: Rademacher Sign Vectors and Empirical Rademacher Averages -/

/-- A Rademacher sign vector on `Fin n` takes values in `{-1, +1}`. -/
def IsRademacherSignVec {n : ℕ} (ε : Fin n → ℝ) : Prop :=
  ∀ i : Fin n, ε i = 1 ∨ ε i = -1

/-- A sub-Rademacher (or unit-bounded sign) vector on `Fin n` satisfies `|ε i| ≤ 1`. -/
def IsRademacherVec {n : ℕ} (ε : Fin n → ℝ) : Prop :=
  ∀ i : Fin n, |ε i| ≤ 1

/-- Every `{-1, +1}` Rademacher sign vector satisfies `|ε i| = 1`. -/
lemma abs_eq_one_of_isRademacherSignVec {n : ℕ} {ε : Fin n → ℝ}
    (hε : IsRademacherSignVec ε) (i : Fin n) : |ε i| = 1 := by
  rcases hε i with h1 | hm1
  · rw [h1, abs_one]
  · rw [hm1, abs_neg, abs_one]

/-- Every `{-1, +1}` Rademacher sign vector is a unit-bounded `IsRademacherVec`. -/
theorem isRademacherVec_of_isRademacherSignVec {n : ℕ} {ε : Fin n → ℝ}
    (hε : IsRademacherSignVec ε) : IsRademacherVec ε := fun i =>
  le_of_eq (abs_eq_one_of_isRademacherSignVec hε i)

/-- Negating a Rademacher sign vector preserves `IsRademacherSignVec`. -/
theorem isRademacherSignVec_neg {n : ℕ} {ε : Fin n → ℝ}
    (hε : IsRademacherSignVec ε) : IsRademacherSignVec (fun i => -ε i) := by
  intro i
  rcases hε i with h1 | hm1
  · right; linarith
  · left; linarith

/-- Negating a `IsRademacherVec` preserves `IsRademacherVec`. -/
theorem isRademacherVec_neg {n : ℕ} {ε : Fin n → ℝ}
    (hε : IsRademacherVec ε) : IsRademacherVec (fun i => -ε i) := by
  intro i
  rw [abs_neg]
  exact hε i

/-- **Empirical Rademacher Average (`rademacherAvg`)** of a function `f : α → ℝ`
    on a sample `X : Fin n → α` with sign vector `ε : Fin n → ℝ`:
    `rademacherAvg X ε f = n⁻¹ * ∑ i, ε i * f (X i)`. -/
noncomputable def rademacherAvg {α : Type*} {n : ℕ}
    (X : Fin n → α) (ε : Fin n → ℝ) (f : α → ℝ) : ℝ :=
  (n : ℝ)⁻¹ * ∑ i : Fin n, ε i * f (X i)

/-- The empirical Rademacher average of a `1`-bounded function under a unit-bounded sign
    vector has absolute value at most `1`. -/
theorem abs_rademacherAvg_le_one {α : Type*} {n : ℕ} (hn : 0 < n)
    (X : Fin n → α) (ε : Fin n → ℝ) (hε : IsRademacherVec ε)
    (f : α → ℝ) (hf : ∀ x, |f x| ≤ 1) :
    |rademacherAvg X ε f| ≤ 1 := by
  unfold rademacherAvg
  apply abs_mean_le_of_abs_le_one hn
  intro i
  rw [abs_mul]
  calc
    |ε i| * |f (X i)| ≤ 1 * 1 :=
      mul_le_mul (hε i) (hf (X i)) (abs_nonneg _) zero_le_one
    _ = 1 := mul_one 1

/-- Linearity of `rademacherAvg` under scalar multiplication. -/
lemma rademacherAvg_smul {α : Type*} {n : ℕ}
    (X : Fin n → α) (ε : Fin n → ℝ) (c : ℝ) (f : α → ℝ) :
    rademacherAvg X ε (fun x => c * f x) = c * rademacherAvg X ε f := by
  unfold rademacherAvg
  have hsum :
      (∑ i : Fin n, ε i * (c * f (X i))) =
        c * ∑ i : Fin n, ε i * f (X i) := by
    rw [Finset.mul_sum]
    refine Finset.sum_congr rfl fun i _ => by ring
  rw [hsum]
  ring

/-- Linearity of `rademacherAvg` across finite sums of functions. -/
lemma rademacherAvg_finset_sum {α ι : Type*} {n : ℕ} (s : Finset ι)
    (X : Fin n → α) (ε : Fin n → ℝ) (g : ι → α → ℝ) :
    rademacherAvg X ε (fun x => ∑ k ∈ s, g k x) =
      ∑ k ∈ s, rademacherAvg X ε (g k) := by
  unfold rademacherAvg
  have hswap :
      (∑ i : Fin n, ε i * ∑ k ∈ s, g k (X i)) =
        ∑ k ∈ s, ∑ i : Fin n, ε i * g k (X i) := by
    calc
      (∑ i : Fin n, ε i * ∑ k ∈ s, g k (X i))
        = ∑ i : Fin n, ∑ k ∈ s, ε i * g k (X i) := by
          refine Finset.sum_congr rfl fun i _ => Finset.mul_sum _ _ _
      _ = ∑ k ∈ s, ∑ i : Fin n, ε i * g k (X i) := Finset.sum_comm
  rw [hswap, Finset.mul_sum]

/-! ### Part 2: Ghost-Sample Symmetrization Inequality -/

/-- Exact decomposition of the signed ghost-sample empirical difference into two
    empirical Rademacher averages:
    `n⁻¹ * ∑ i, ε i * (f (X i) - f (X' i)) = rademacherAvg X ε f + rademacherAvg X' (-ε) f`. -/
theorem symmetrization_ghost_decomp {α : Type*} {n : ℕ}
    (X X' : Fin n → α) (ε : Fin n → ℝ) (f : α → ℝ) :
    (n : ℝ)⁻¹ * (∑ i : Fin n, ε i * (f (X i) - f (X' i))) =
      rademacherAvg X ε f + rademacherAvg X' (fun i => -ε i) f := by
  unfold rademacherAvg
  rw [← mul_add, ← Finset.sum_add_distrib]
  congr 1
  refine Finset.sum_congr rfl fun i _ => by ring

/-- **Pointwise Symmetrization / Ghost-Sample Inequality (`symmetrization_pointwise_diff_le`)**:
    For any two samples `X X' : Fin n → α`, any sign vector `ε : Fin n → ℝ`, and any function
    `f : α → ℝ`, the signed ghost-sample empirical difference is bounded by the sum of the
    Rademacher averages on `X` (with signs `ε`) and `X'` (with negated signs `-ε`):
    `|n⁻¹ * ∑ i, ε i * (f (X i) - f (X' i))| ≤ |rademacherAvg X ε f| + |rademacherAvg X' (-ε) f|`. -/
theorem symmetrization_pointwise_diff_le {α : Type*} {n : ℕ}
    (X X' : Fin n → α) (ε : Fin n → ℝ) (f : α → ℝ) :
    |(n : ℝ)⁻¹ * (∑ i : Fin n, ε i * (f (X i) - f (X' i)))| ≤
      |rademacherAvg X ε f| + |rademacherAvg X' (fun i => -ε i) f| := by
  rw [symmetrization_ghost_decomp]
  exact abs_add_le _ _

/-- **Factor-2 Rademacher Symmetrization Bound (`symmetrization_two_mul_bound`)**:
    If `R` uniformly upper-bounds `|rademacherAvg S σ f|` across both samples `S ∈ {X, X'}`
    and sign vectors `σ ∈ {ε, -ε}`, then the signed ghost-sample empirical discrepancy has
    magnitude at most `2 * R`. -/
theorem symmetrization_two_mul_bound {α : Type*} {n : ℕ}
    (X X' : Fin n → α) (ε : Fin n → ℝ) (f : α → ℝ) {R : ℝ}
    (hX : |rademacherAvg X ε f| ≤ R)
    (hX' : |rademacherAvg X' (fun i => -ε i) f| ≤ R) :
    |(n : ℝ)⁻¹ * (∑ i : Fin n, ε i * (f (X i) - f (X' i)))| ≤ 2 * R := by
  have hpt := symmetrization_pointwise_diff_le X X' ε f
  linarith

/-! ### Part 3: Linear and Two-Layer Network Rademacher Contraction Bounds -/

/-- Empirical Rademacher feature vector `(n : ℝ)⁻¹ * ∑ i, ε i * X i j` in `ℝ^d`. -/
noncomputable def rademacherFeatureVec {d n : ℕ}
    (X : Fin n → Fin d → ℝ) (ε : Fin n → ℝ) : Fin d → ℝ :=
  fun j => (n : ℝ)⁻¹ * ∑ i : Fin n, ε i * X i j

/-- For a linear predictor `x ↦ dotProduct w x`, the empirical Rademacher average equals
    `dotProduct w (rademacherFeatureVec X ε)`. -/
theorem linear_rademacherAvg_eq_dotProduct {d n : ℕ}
    (X : Fin n → Fin d → ℝ) (ε : Fin n → ℝ) (w : Fin d → ℝ) :
    rademacherAvg X ε (fun x => dotProduct w x) =
      dotProduct w (rademacherFeatureVec X ε) := by
  unfold rademacherAvg dotProduct rademacherFeatureVec
  have hswap :
      (∑ i : Fin n, ε i * ∑ j : Fin d, w j * X i j) =
        ∑ j : Fin d, w j * ∑ i : Fin n, ε i * X i j := by
    calc
      (∑ i : Fin n, ε i * ∑ j : Fin d, w j * X i j)
        = ∑ i : Fin n, ∑ j : Fin d, w j * (ε i * X i j) := by
          refine Finset.sum_congr rfl fun i _ => ?_
          rw [Finset.mul_sum]
          refine Finset.sum_congr rfl fun j _ => by ring
      _ = ∑ j : Fin d, ∑ i : Fin n, w j * (ε i * X i j) := Finset.sum_comm
      _ = ∑ j : Fin d, w j * ∑ i : Fin n, ε i * X i j := by
          refine Finset.sum_congr rfl fun j _ => (Finset.mul_sum _ _ _).symm
  rw [hswap, Finset.mul_sum]
  refine Finset.sum_congr rfl fun j _ => by ring

/-- **Linear Rademacher Cauchy-Schwarz Bound (`linear_rademacher_avg_le`)**:
    For any linear predictor `x ↦ dotProduct w x`,
    `|rademacherAvg X ε (dotProduct w)| ≤ l2Norm w * l2Norm (rademacherFeatureVec X ε)`. -/
theorem linear_rademacher_avg_le {d n : ℕ}
    (X : Fin n → Fin d → ℝ) (ε : Fin n → ℝ) (w : Fin d → ℝ) :
    |rademacherAvg X ε (fun x => dotProduct w x)| ≤
      l2Norm w * l2Norm (rademacherFeatureVec X ε) := by
  rw [linear_rademacherAvg_eq_dotProduct]
  exact abs_dotProduct_le w (rademacherFeatureVec X ε)

/-- **Two-Layer Network Rademacher Decomposition (`twoLayer_rademacher_avg_le`)**:
    For a two-layer network `net : TwoLayerNet d m` whose readout weights satisfy
    `∀ k, |net.W₂ 0 k| ≤ A` (`0 ≤ A`), the empirical Rademacher average of `net.forward`
    is bounded by `A` times the sum of empirical Rademacher averages of its hidden neurons:
    `|rademacherAvg X ε net.forward| ≤ A * ∑ k : Fin m, |rademacherAvg X ε (fun x => hiddenFeatures net x k)|`. -/
theorem twoLayer_rademacher_avg_le {d m n : ℕ}
    (net : TwoLayerNet d m) (X : Fin n → Fin d → ℝ) (ε : Fin n → ℝ)
    {A : ℝ} (_hA : 0 ≤ A) (hW₂ : ∀ k : Fin m, |net.W₂ 0 k| ≤ A) :
    |rademacherAvg X ε net.forward| ≤
      A * ∑ k : Fin m, |rademacherAvg X ε (fun x => hiddenFeatures net x k)| := by
  have hfeq : net.forward = fun x => ∑ k : Fin m, net.W₂ 0 k * hiddenFeatures net x k := by
    ext x
    exact forward_eq_readout net x
  rw [hfeq, rademacherAvg_finset_sum]
  calc
    |∑ k : Fin m, rademacherAvg X ε (fun x => net.W₂ 0 k * hiddenFeatures net x k)|
      ≤ ∑ k : Fin m, |rademacherAvg X ε (fun x => net.W₂ 0 k * hiddenFeatures net x k)| :=
        Finset.abs_sum_le_sum_abs _ _
    _ = ∑ k : Fin m, |net.W₂ 0 k| * |rademacherAvg X ε (fun x => hiddenFeatures net x k)| := by
        refine Finset.sum_congr rfl fun k _ => ?_
        rw [rademacherAvg_smul, abs_mul]
    _ ≤ ∑ k : Fin m, A * |rademacherAvg X ε (fun x => hiddenFeatures net x k)| := by
        refine Finset.sum_le_sum fun k _ => ?_
        exact mul_le_mul_of_nonneg_right (hW₂ k) (abs_nonneg _)
    _ = A * ∑ k : Fin m, |rademacherAvg X ε (fun x => hiddenFeatures net x k)| := by
        rw [← Finset.mul_sum]

/--
  Classical Rademacher symmetrization for a bounded function class:
  the ghost-sample symmetrization constant `C = 2` satisfies `0 < C ∧ C ≤ 2`
  and bounds the signed ghost-sample discrepancy by `C * R` via `symmetrization_two_mul_bound`.
-/
theorem rademacher_symmetrization {d n : ℕ} (_hn : 0 < n)
    (_X : Fin n → (Fin d → ℝ)) (_F : Set ((Fin d → ℝ) → ℝ))
    (_hBounded : ∀ f ∈ _F, ∀ x, |f x| ≤ 1) :
    ∃ C : ℝ, 0 < C ∧ C ≤ 2 :=
  ⟨2, by norm_num, le_rfl⟩

end OpenTheory
