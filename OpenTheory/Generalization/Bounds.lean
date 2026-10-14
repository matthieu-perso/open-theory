import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Analysis.InnerProductSpace.PiL2
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import OpenTheory.Architectures.Basic

namespace OpenTheory

/-- Euclidean inner product on `ℝ^d` indexed by `Fin d`. -/
def dotProduct {d : ℕ} (u v : Fin d → ℝ) : ℝ := ∑ j : Fin d, u j * v j

/-- Euclidean norm. -/
noncomputable def l2Norm {d : ℕ} (u : Fin d → ℝ) : ℝ :=
  Real.sqrt (∑ j : Fin d, u j ^ 2)

theorem l2Norm_nonneg {d : ℕ} (u : Fin d → ℝ) : 0 ≤ l2Norm u :=
  Real.sqrt_nonneg _

theorem l2Norm_sq {d : ℕ} (u : Fin d → ℝ) :
    l2Norm u ^ 2 = ∑ j : Fin d, u j ^ 2 :=
  Real.sq_sqrt (Finset.sum_nonneg fun _ _ => sq_nonneg _)

theorem l2Norm_eq_euclideanNorm {d : ℕ} (u : Fin d → ℝ) :
    l2Norm u = ‖(WithLp.toLp 2 u : EuclideanSpace ℝ (Fin d))‖ := by
  have hsq : l2Norm u ^ 2 = ‖(WithLp.toLp 2 u : EuclideanSpace ℝ (Fin d))‖ ^ 2 := by
    rw [l2Norm_sq, EuclideanSpace.real_norm_sq_eq]
  exact (sq_eq_sq₀ (l2Norm_nonneg u) (norm_nonneg _)).mp hsq

theorem dotProduct_sub_right {d : ℕ} (w x y : Fin d → ℝ) :
    dotProduct w x - dotProduct w y = dotProduct w (fun j => x j - y j) := by
  unfold dotProduct
  rw [← Finset.sum_sub_distrib]
  refine Finset.sum_congr rfl fun j _ => ?_
  ring

/-- Cauchy-Schwarz inequality for `dotProduct` and `l2Norm`. -/
theorem abs_dotProduct_le {d : ℕ} (u v : Fin d → ℝ) :
    |dotProduct u v| ≤ l2Norm u * l2Norm v := by
  let uE : EuclideanSpace ℝ (Fin d) := WithLp.toLp 2 u
  let vE : EuclideanSpace ℝ (Fin d) := WithLp.toLp 2 v
  have hinner : dotProduct u v = inner ℝ uE vE := by
    rw [PiLp.inner_apply]
    simp only [Real.inner_apply, dotProduct, uE, vE]
  rw [hinner, l2Norm_eq_euclideanNorm u, l2Norm_eq_euclideanNorm v]
  exact abs_real_inner_le_norm uE vE

/-- Explicit Cauchy-Schwarz difference bound for `dotProduct` and `l2Norm`. -/
theorem abs_dotProduct_sub_le {d : ℕ} (w x y : Fin d → ℝ) :
    |dotProduct w x - dotProduct w y| ≤ l2Norm w * l2Norm (fun j => x j - y j) := by
  rw [dotProduct_sub_right]
  exact abs_dotProduct_le w (fun j => x j - y j)

/-- Any Lipschitz constant of a real function on `ℝ` is nonnegative. -/
theorem lipschitz_const_nonneg {σ : ℝ → ℝ} {L : ℝ}
    (hL : ∀ a b : ℝ, |σ a - σ b| ≤ L * |a - b|) : 0 ≤ L := by
  have h10 := hL 1 0
  have habs : (0 : ℝ) ≤ |σ 1 - σ 0| := abs_nonneg _
  have h1 : |(1 : ℝ) - 0| = 1 := by norm_num
  rw [h1, mul_one] at h10
  exact le_trans habs h10

/-- Explicit-constant input Lipschitz bound for a single hidden neuron. -/
theorem layer_lipschitz_of_lip (act : ActivationFunction) {d : ℕ} (w x y : Fin d → ℝ)
    {L : ℝ} (hL : ∀ a b : ℝ, |act.σ a - act.σ b| ≤ L * |a - b|) :
    |act.σ (dotProduct w x) - act.σ (dotProduct w y)| ≤
      L * l2Norm w * l2Norm (fun j => x j - y j) := by
  have hL0 : 0 ≤ L := lipschitz_const_nonneg hL
  calc
    |act.σ (dotProduct w x) - act.σ (dotProduct w y)|
      ≤ L * |dotProduct w x - dotProduct w y| := hL _ _
    _ ≤ L * (l2Norm w * l2Norm (fun j => x j - y j)) :=
        mul_le_mul_of_nonneg_left (abs_dotProduct_sub_le w x y) hL0
    _ = L * l2Norm w * l2Norm (fun j => x j - y j) := by ring

/-- One hidden unit is Lipschitz in the input, with constant `L ‖w‖₂`. -/
theorem layer_lipschitz (act : ActivationFunction) {d : ℕ} (w x y : Fin d → ℝ) :
    ∃ L : ℝ,
      |act.σ (dotProduct w x) - act.σ (dotProduct w y)| ≤
        L * l2Norm w * l2Norm (fun j => x j - y j) := by
  obtain ⟨L, hL⟩ := act.lipschitz_bound
  exact ⟨L, layer_lipschitz_of_lip act w x y hL⟩

/-- Explicit-constant input Lipschitz bound for `TwoLayerNet.forward`. -/
theorem forward_lipschitz_explicit {d m : ℕ} (net : TwoLayerNet d m) (x y : Fin d → ℝ)
    {L : ℝ} (hL : ∀ a b : ℝ, |net.act.σ a - net.act.σ b| ≤ L * |a - b|) :
    |net.forward x - net.forward y| ≤
      (∑ i : Fin m, |net.W₂ 0 i| * L * l2Norm (net.W₁ i)) *
        l2Norm (fun j => x j - y j) := by
  unfold TwoLayerNet.forward
  dsimp only
  rw [← Finset.sum_sub_distrib, Finset.sum_mul]
  refine (Finset.abs_sum_le_sum_abs _ _).trans ?_
  refine Finset.sum_le_sum fun i _ => ?_
  rw [← mul_sub, abs_mul]
  have hlayer := layer_lipschitz_of_lip net.act (net.W₁ i) x y hL
  calc
    |net.W₂ 0 i| * |net.act.σ (∑ j : Fin d, net.W₁ i j * x j) -
        net.act.σ (∑ j : Fin d, net.W₁ i j * y j)|
      = |net.W₂ 0 i| * |net.act.σ (dotProduct (net.W₁ i) x) -
          net.act.σ (dotProduct (net.W₁ i) y)| := rfl
    _ ≤ |net.W₂ 0 i| * (L * l2Norm (net.W₁ i) * l2Norm (fun j => x j - y j)) :=
        mul_le_mul_of_nonneg_left hlayer (abs_nonneg _)
    _ = |net.W₂ 0 i| * L * l2Norm (net.W₁ i) * l2Norm (fun j => x j - y j) := by ring

/-- The scalar network `forward` is Lipschitz in the input. -/
theorem forward_lipschitz {d m : ℕ} (net : TwoLayerNet d m) (x y : Fin d → ℝ) :
    ∃ C : ℝ, |net.forward x - net.forward y| ≤ C * l2Norm (fun j => x j - y j) := by
  obtain ⟨L, hL⟩ := net.act.lipschitz_bound
  exact ⟨∑ i : Fin m, |net.W₂ 0 i| * L * l2Norm (net.W₁ i),
    forward_lipschitz_explicit net x y hL⟩

end OpenTheory
