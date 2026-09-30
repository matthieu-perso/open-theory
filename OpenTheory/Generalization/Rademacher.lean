import OpenTheory.Architectures.Basic
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Data.Fintype.Card
import Mathlib.Data.Real.Basic

namespace OpenTheory

/-- Elementary bound: the empirical mean of a 1-bounded sequence has
    absolute value at most 1. Kernel-checked; no `sorry`. -/
lemma abs_mean_le_of_abs_le_one {n : ℕ} (hn : 0 < n) (a : Fin n → ℝ)
    (ha : ∀ i, |a i| ≤ 1) :
    |((n : ℝ)⁻¹) * ∑ i, a i| ≤ 1 := by
  have npos : (0 : ℝ) < n := Nat.cast_pos.mpr hn
  have h₁ : |∑ i, a i| ≤ ∑ i, |a i| := abs_sum_le_sum_abs a Finset.univ
  have h₂ : ∑ i, |a i| ≤ ∑ _i : Fin n, (1 : ℝ) :=
    Finset.sum_le_sum fun i _ => ha i
  have h₃ : ∑ _i : Fin n, (1 : ℝ) = (n : ℝ) := by
    simp [Finset.sum_const, Finset.card_univ, Fintype.card_fin]
  have hbound : |∑ i, a i| ≤ (n : ℝ) := h₁.trans (h₂.trans_eq h₃)
  rw [abs_mul, abs_of_pos (inv_pos.mpr npos)]
  have hmul : (n : ℝ)⁻¹ * |∑ i, a i| ≤ (n : ℝ)⁻¹ * n :=
    mul_le_mul_of_nonneg_left hbound (inv_nonneg.mpr npos.le)
  have hcancel : (n : ℝ)⁻¹ * n = 1 := inv_mul_cancel (ne_of_gt npos)
  exact hmul.trans_eq hcancel

/--
  Classical Rademacher symmetrization for a bounded function class.
  The full empirical-process statement is an open target (do not tag
  `\leanok` on the proof until this `sorry` is gone).
-/
theorem rademacher_symmetrization {d n : ℕ} (_hn : 0 < n)
    (_X : Fin n → (Fin d → ℝ)) (_F : Set ((Fin d → ℝ) → ℝ))
    (_hBounded : ∀ f ∈ _F, ∀ x, |f x| ≤ 1) :
    ∃ C : ℝ, 0 < C ∧ C ≤ 2 := by
  sorry

end OpenTheory
