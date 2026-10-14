import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Data.Matrix.Basic
import Mathlib.Basic.Real.Basic
import Mathlib.Tactic.Ring
import Mathlib.Tactic.Linarith

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

/-- Row-wise inner product of `H * P` with `H` equals the squared row norm of `H * P`
    when `P` is an orthogonal projector. -/
lemma orthoProjector_row_sq_sum {n m : ℕ} (H : Matrix (Fin n) (Fin m) ℝ)
    (P : Matrix (Fin m) (Fin m) ℝ) (hP : IsOrthoProjector P) (i : Fin n) :
    ∑ j : Fin m, ((H * P) i j) ^ 2 = ∑ j : Fin m, (H * P) i j * H i j := by
  have hPP := hP.1
  have hPsym := hP.2
  have hP_entry : ∀ k l : Fin m, ∑ j : Fin m, P k j * P l j = P k l := by
    intro k l
    have hsym_entry : ∀ j : Fin m, P l j = P j l := fun j => by
      have := congr_fun (congr_fun hPsym j) l
      simpa [Matrix.transpose_apply] using this
    calc
      ∑ j : Fin m, P k j * P l j = ∑ j : Fin m, P k j * P j l := by
        refine Finset.sum_congr rfl fun j _ => ?_
        rw [hsym_entry j]
      _ = (P * P) k l := by rfl
      _ = P k l := by rw [hPP]
  calc
    ∑ j : Fin m, ((H * P) i j) ^ 2
        = ∑ j : Fin m, (∑ k : Fin m, H i k * P k j) * (∑ l : Fin m, H i l * P l j) := by
          refine Finset.sum_congr rfl fun j _ => ?_
          simp only [Matrix.mul_apply, sq]
    _ = ∑ j : Fin m, ∑ k : Fin m, ∑ l : Fin m, (H i k * H i l) * (P k j * P l j) := by
          refine Finset.sum_congr rfl fun j _ => ?_
          rw [Finset.sum_mul]
          refine Finset.sum_congr rfl fun k _ => ?_
          rw [Finset.mul_sum]
          refine Finset.sum_congr rfl fun l _ => by ring
    _ = ∑ k : Fin m, ∑ j : Fin m, ∑ l : Fin m, (H i k * H i l) * (P k j * P l j) :=
          Finset.sum_comm
    _ = ∑ k : Fin m, ∑ l : Fin m, ∑ j : Fin m, (H i k * H i l) * (P k j * P l j) := by
          refine Finset.sum_congr rfl fun k _ => Finset.sum_comm
    _ = ∑ k : Fin m, ∑ l : Fin m, (H i k * H i l) * (∑ j : Fin m, P k j * P l j) := by
          refine Finset.sum_congr rfl fun k _ => Finset.sum_congr rfl fun l _ => ?_
          rw [← Finset.mul_sum]
    _ = ∑ k : Fin m, ∑ l : Fin m, (H i k * P k l) * H i l := by
          refine Finset.sum_congr rfl fun k _ => Finset.sum_congr rfl fun l _ => ?_
          rw [hP_entry k l]
          ring
    _ = ∑ l : Fin m, ∑ k : Fin m, (H i k * P k l) * H i l := Finset.sum_comm
    _ = ∑ l : Fin m, (∑ k : Fin m, H i k * P k l) * H i l := by
          refine Finset.sum_congr rfl fun l _ => by rw [← Finset.sum_mul]
    _ = ∑ j : Fin m, (H * P) i j * H i j := rfl

/-- Exact Pythagorean identity for the Frobenius norm under an orthogonal projector:
    `‖H - H P‖_F² = ‖H‖_F² - ‖H P‖_F²`. -/
theorem frobeniusSq_sub_mul_orthoProjector {n m : ℕ} (H : Matrix (Fin n) (Fin m) ℝ)
    (P : Matrix (Fin m) (Fin m) ℝ) (hP : IsOrthoProjector P) :
    frobeniusSq (H - H * P) = frobeniusSq H - frobeniusSq (H * P) := by
  unfold frobeniusSq
  rw [← Finset.sum_sub_distrib]
  refine Finset.sum_congr rfl fun i _ => ?_
  have hrow := orthoProjector_row_sq_sum H P hP i
  have hexpand : ∑ j : Fin m, ((H - H * P) i j) ^ 2 =
      ∑ j : Fin m, (H i j) ^ 2 - 2 * ∑ j : Fin m, (H * P) i j * H i j +
        ∑ j : Fin m, ((H * P) i j) ^ 2 := by
    have hpt : ∀ j : Fin m, ((H - H * P) i j) ^ 2 =
        ((H i j) ^ 2 - 2 * ((H * P) i j * H i j)) + ((H * P) i j) ^ 2 := fun j => by
      simp only [Matrix.sub_apply]
      ring
    simp_rw [hpt, Finset.sum_add_distrib, Finset.sum_sub_distrib, ← Finset.mul_sum]
  linarith

/-- Orthogonal projection does not increase the squared Frobenius norm. -/
lemma frobeniusSq_mul_orthoProjector_le {n m : ℕ} (H : Matrix (Fin n) (Fin m) ℝ)
    (P : Matrix (Fin m) (Fin m) ℝ) (hP : IsOrthoProjector P) :
    frobeniusSq (H * P) ≤ frobeniusSq H := by
  have hpyth := frobeniusSq_sub_mul_orthoProjector H P hP
  have hnonneg := frobeniusSq_nonneg (H - H * P)
  linarith

/-- If `P` is an orthogonal projector then alignment is at most 1. -/
theorem alignment_le_one {n m : ℕ} (H : Matrix (Fin n) (Fin m) ℝ)
    (P : Matrix (Fin m) (Fin m) ℝ) (hP : IsOrthoProjector P) :
    alignment H P ≤ 1 := by
  unfold alignment
  split_ifs with h0
  · norm_num
  · have hpos : 0 < frobeniusSq H :=
      lt_of_le_of_ne (frobeniusSq_nonneg H) (Ne.symm h0)
    rw [div_le_one hpos]
    exact frobeniusSq_mul_orthoProjector_le H P hP

/-- Certificate: alignment at least `1 - ε` puts the features within relative
    distance `√ε` of the subspace. Equivalently `‖H - HP‖_F² ≤ ε ‖H‖_F²`. -/
theorem alignment_feature_distance {n m : ℕ} (H : Matrix (Fin n) (Fin m) ℝ)
    (P : Matrix (Fin m) (Fin m) ℝ) (hP : IsOrthoProjector P) (ε : ℝ)
    (_hε : 0 ≤ ε) (hH : frobeniusSq H ≠ 0) (ha : alignment H P ≥ 1 - ε) :
    frobeniusSq (H - H * P) ≤ ε * frobeniusSq H := by
  have hpos : 0 < frobeniusSq H :=
    lt_of_le_of_ne (frobeniusSq_nonneg H) (Ne.symm hH)
  have ha' : 1 - ε ≤ frobeniusSq (H * P) / frobeniusSq H := by
    simpa [alignment, hH] using ha
  have hmul : (1 - ε) * frobeniusSq H ≤ frobeniusSq (H * P) :=
    (le_div_iff₀ hpos).mp ha'
  have hpyth := frobeniusSq_sub_mul_orthoProjector H P hP
  linarith

end OpenTheory
