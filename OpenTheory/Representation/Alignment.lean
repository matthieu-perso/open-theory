import Mathlib.Algebra.BigOperators.Group.Finset
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Real.Basic

open Classical

namespace OpenTheory

/-- Squared Frobenius norm `‖A‖_F² = ∑_{ij} A_{ij}²`. -/
def frobeniusSq {r c : ℕ} (A : Matrix (Fin r) (Fin c) ℝ) : ℝ :=
  ∑ i : Fin r, ∑ j : Fin c, (A i j) ^ 2

lemma frobeniusSq_nonneg {r c : ℕ} (A : Matrix (Fin r) (Fin c) ℝ) :
    0 ≤ frobeniusSq A := by
  unfold frobeniusSq
  exact Finset.sum_nonneg fun _ _ => Finset.sum_nonneg fun _ _ => sq_nonneg _

/-- Orthogonal projector: `P² = P` and `P` symmetric. -/
def IsOrthoProjector {m : ℕ} (P : Matrix (Fin m) (Fin m) ℝ) : Prop :=
  P * P = P ∧ P.transpose = P

/-- Alignment of feature matrix `H` with projector `P`.
    Rows of `H` are samples, so `P` acts on the right: `‖HP‖_F² / ‖H‖_F²`.
    The value is `0` when `H = 0`. -/
noncomputable def alignment {n m : ℕ} (H : Matrix (Fin n) (Fin m) ℝ)
    (P : Matrix (Fin m) (Fin m) ℝ) : ℝ :=
  if frobeniusSq H = 0 then 0 else frobeniusSq (H * P) / frobeniusSq H

lemma alignment_nonneg {n m : ℕ} (H : Matrix (Fin n) (Fin m) ℝ)
    (P : Matrix (Fin m) (Fin m) ℝ) : 0 ≤ alignment H P := by
  unfold alignment
  split_ifs
  · exact le_rfl
  · exact div_nonneg (frobeniusSq_nonneg _) (frobeniusSq_nonneg _)

/-- If `P` is an orthogonal projector then alignment is at most 1. -/
theorem alignment_le_one {n m : ℕ} (H : Matrix (Fin n) (Fin m) ℝ)
    (P : Matrix (Fin m) (Fin m) ℝ) (hP : IsOrthoProjector P) :
    alignment H P ≤ 1 := by
  sorry

/-- Certificate: alignment at least `1 - ε` puts the features within relative
    distance `√ε` of the subspace. Equivalently `‖H - HP‖_F² ≤ ε ‖H‖_F²`. -/
theorem alignment_feature_distance {n m : ℕ} (H : Matrix (Fin n) (Fin m) ℝ)
    (P : Matrix (Fin m) (Fin m) ℝ) (hP : IsOrthoProjector P) (ε : ℝ)
    (hε : 0 ≤ ε) (hH : frobeniusSq H ≠ 0) (ha : alignment H P ≥ 1 - ε) :
    frobeniusSq (H - H * P) ≤ ε * frobeniusSq H := by
  sorry

end OpenTheory
