import Mathlib.Algebra.BigOperators.Group.Finset
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Algebra.Polynomial.Basic
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Data.Fintype.Card
import OpenTheory.Architectures.Basic
import OpenTheory.Data.Model
import OpenTheory.Data.Sample

namespace OpenTheory

/-- The first `k` coordinates of a vector in `ℝ^d`, using `k ≤ d`. -/
def coordinateLift {d k : ℕ} (hk : k ≤ d) (x : Fin d → ℝ) : Fin k → ℝ :=
  fun i => x ⟨i.val, lt_of_lt_of_le i.isLt hk⟩

/-- `σ` agrees with no polynomial. Universal approximation needs this. -/
def NonPolynomial (σ : ℝ → ℝ) : Prop :=
  ∀ p : Polynomial ℝ, ∃ x, σ x ≠ p.eval x

/-- Average absolute error of `net` against `target` on a sample. -/
noncomputable def empiricalApproxError {d m n : ℕ} (net : TwoLayerNet d m)
    (target : (Fin d → ℝ) → ℝ) (sample : Sample d n) : ℝ :=
  ((n : ℝ)⁻¹) * ∑ i : Fin n, |net.forward (sample.X i) - target (sample.X i)|

lemma empiricalApproxError_le_of_pointwise {d m n : ℕ} (net : TwoLayerNet d m)
    (target : (Fin d → ℝ) → ℝ) (sample : Sample d n) {ε : ℝ}
    (h : ∀ i, |net.forward (sample.X i) - target (sample.X i)| ≤ ε) :
    empiricalApproxError net target sample ≤ ε := by
  have npos : (0 : ℝ) < (n : ℝ) := Nat.cast_pos.mpr sample.n_pos
  unfold empiricalApproxError
  have hsum :
      ∑ i : Fin n, |net.forward (sample.X i) - target (sample.X i)| ≤
        ∑ _i : Fin n, ε :=
    Finset.sum_le_sum fun i _ => h i
  have hconst : ∑ _i : Fin n, ε = (n : ℝ) * ε := by
    simp [Finset.sum_const, Finset.card_univ, Fintype.card_fin]
  have hbound := hsum.trans_eq hconst
  have hmul := mul_le_mul_of_nonneg_left hbound (inv_nonneg.mpr npos.le)
  have hcancel : ((n : ℝ)⁻¹) * ((n : ℝ) * ε) = ε := by
    rw [← mul_assoc, inv_mul_cancel₀ (ne_of_gt npos), one_mul]
  exact hmul.trans_eq hcancel

/-- Open target. On a finite sample from a spiked model, a wide enough two-layer
    net with a non-polynomial activation approximates any target of the signal
    coordinates. -/
theorem exists_subspace_approximation {d n : ℕ} (model : SpikedModel d)
    (sample : Sample d n) (act : ActivationFunction) (hσ : NonPolynomial act.σ)
    (target : (Fin model.signalRank → ℝ) → ℝ) (ε : ℝ) (hε : 0 < ε) :
    ∃ m : ℕ, ∃ net : TwoLayerNet d m,
      net.act = act ∧
        empiricalApproxError net
          (fun x => target (coordinateLift model.rank_le x)) sample ≤ ε := by
  have _ := hσ
  have _ := hε
  sorry

end OpenTheory
