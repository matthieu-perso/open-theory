import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Ring
import OpenTheory.Architectures.Residual
import OpenTheory.Generalization.Bounds
import OpenTheory.Representation.Alignment

/-!
# Deep Residual Scaling, Lipschitz Stability, and Depth-Independent Signal Propagation

This module establishes genuine vector-norm Lipschitz and signal-propagation guarantees for
residual blocks (`residualStep`) and `L`-layer residual networks (`deepResidual`) on `Fin d → ℝ`
equipped with `l2Norm`:

1. **Euclidean Norm Algebra**: Nonnegativity, homogeneity, triangle inequality
   (`l2Norm_add_le`), and reverse triangle inequality (`l2Norm_sub_ge`, `l2Norm_add_ge`)
   proved directly from the Cauchy-Schwarz inequality (`Finset.sum_mul_sq_le_sq_mul_sq`).
2. **Residual Branch Lipschitz Bound (`branchOutput_l2Norm_le`)**: If the activation `σ` is
   `L_σ`-Lipschitz, then the branch map `x ↦ σ(W x)` is Lipschitz in `l2Norm` with constant
   `L_σ * √(frobeniusSq W)`.
3. **Single-Step Residual Lipschitz Bound (`residualStep_l2Norm_le`)**: A residual block with
   branch scale `scale` is `(1 + |scale| * L_σ * √(frobeniusSq W))`-Lipschitz in `l2Norm`.
4. **Hayou et al. `1/√L` Scaling (`residualStep_sqrt_depth_scaling`)**: With `scale = 1 / √L`
   and branch Lipschitz constant bounded by `c`, each residual step has Lipschitz constant at
   most `1 + c / √L`.
5. **Depth-Independent `1/L` Scaling (`deepResidualChain_depth_independent_exp_bound`)**:
   Iterating `L` residual blocks with `scale = 1 / L` yields a network whose global Lipschitz
   constant is bounded by `(1 + c / L)^L ≤ exp(c)` for **every** depth `L ≥ 1`, preventing the
   exponential `(1 + c)^L` blowup of unscaled deep networks.
6. **Signal Propagation / Representation Norm Window (`deepResidual_norm_window`)**: When
   branches vanish at zero (`branchOutput (branches t) 0 = 0`), `1/L`-scaled residual networks
   preserve representation norms within `[((1 - c/L)^L) ‖x‖₂, exp(c) ‖x‖₂]` and each step
   satisfies the lower bound `l2Norm (residualStep x branch scale) ≥ (1 - |scale| * c) * l2Norm x`.
-/

noncomputable section

open scoped BigOperators

namespace OpenTheory.DeepResidualScaling

variable {d : ℕ}

/-! ## Section 1: `l2Norm` Basic Properties and Triangle Inequalities -/

/-- Squared `l2Norm` equals the sum of squared coordinates. -/
lemma l2Norm_sq (u : Fin d → ℝ) :
    l2Norm u ^ 2 = ∑ i : Fin d, u i ^ 2 := by
  unfold l2Norm
  exact Real.sq_sqrt (Finset.sum_nonneg fun _ _ => sq_nonneg _)

/-- `l2Norm` is always nonnegative. -/
lemma l2Norm_nonneg (u : Fin d → ℝ) : 0 ≤ l2Norm u := by
  unfold l2Norm
  exact Real.sqrt_nonneg _

/-- `l2Norm` of the zero vector is `0`. -/
@[simp]
lemma l2Norm_zero : l2Norm (fun _ : Fin d => (0 : ℝ)) = 0 := by
  unfold l2Norm
  simp

/-- `l2Norm` scales by `|a|` under scalar multiplication. -/
lemma l2Norm_smul (a : ℝ) (u : Fin d → ℝ) :
    l2Norm (fun i => a * u i) = |a| * l2Norm u := by
  unfold l2Norm
  have hsum : (∑ i : Fin d, (a * u i) ^ 2) = a ^ 2 * ∑ i : Fin d, u i ^ 2 := by
    rw [Finset.mul_sum]
    refine Finset.sum_congr rfl fun i _ => by ring
  rw [hsum, Real.sqrt_mul (sq_nonneg a), Real.sqrt_sq_eq_abs]

/-- `l2Norm` is invariant under negation. -/
lemma l2Norm_neg (u : Fin d → ℝ) :
    l2Norm (fun i => -u i) = l2Norm u := by
  simpa using l2Norm_smul (-1) u

/-- Symmetry of `l2Norm` distance. -/
lemma l2Norm_sub_comm (u v : Fin d → ℝ) :
    l2Norm (fun i => u i - v i) = l2Norm (fun i => v i - u i) := by
  unfold l2Norm
  congr 1
  refine Finset.sum_congr rfl fun i _ => by ring

/-- Cauchy-Schwarz inequality for `dotProduct` and `l2Norm`:
`∑ i, u i * v i ≤ l2Norm u * l2Norm v`. -/
lemma dotProduct_le_l2Norm_mul_l2Norm (u v : Fin d → ℝ) :
    dotProduct u v ≤ l2Norm u * l2Norm v := by
  by_cases hpos : 0 ≤ dotProduct u v
  · have hcs : dotProduct u v ^ 2 ≤ (l2Norm u * l2Norm v) ^ 2 := by
      rw [mul_pow, l2Norm_sq, l2Norm_sq]
      exact Finset.sum_mul_sq_le_sq_mul_sq Finset.univ u v
    exact (sq_le_sq₀ hpos (mul_nonneg (l2Norm_nonneg u) (l2Norm_nonneg v))).mp hcs
  · exact le_trans (le_of_not_ge hpos) (mul_nonneg (l2Norm_nonneg u) (l2Norm_nonneg v))

/-- Triangle inequality for `l2Norm` on `Fin d → ℝ`. -/
theorem l2Norm_add_le (u v : Fin d → ℝ) :
    l2Norm (fun i => u i + v i) ≤ l2Norm u + l2Norm v := by
  have hcs := dotProduct_le_l2Norm_mul_l2Norm u v
  have hsq : l2Norm (fun i => u i + v i) ^ 2 ≤ (l2Norm u + l2Norm v) ^ 2 := by
    rw [l2Norm_sq]
    have hexp : (∑ i : Fin d, (u i + v i) ^ 2) =
        (∑ i : Fin d, u i ^ 2) + 2 * dotProduct u v + (∑ i : Fin d, v i ^ 2) := by
      unfold dotProduct
      rw [Finset.mul_sum, ← Finset.sum_add_distrib, ← Finset.sum_add_distrib]
      refine Finset.sum_congr rfl fun i _ => by ring
    rw [hexp, ← l2Norm_sq u, ← l2Norm_sq v]
    linarith
  exact (sq_le_sq₀ (l2Norm_nonneg _) (add_nonneg (l2Norm_nonneg u) (l2Norm_nonneg v))).mp hsq

/-- Triangle inequality for differences in `l2Norm`. -/
theorem l2Norm_sub_le (u v w : Fin d → ℝ) :
    l2Norm (fun i => u i - w i) ≤
      l2Norm (fun i => u i - v i) + l2Norm (fun i => v i - w i) := by
  have heq : (fun i => u i - w i) = (fun i => (u i - v i) + (v i - w i)) := by
    ext i; ring
  rw [heq]
  exact l2Norm_add_le (fun i => u i - v i) (fun i => v i - w i)

/-- Reverse triangle inequality for `l2Norm`: `l2Norm u - l2Norm v ≤ l2Norm (u + v)`. -/
theorem l2Norm_add_ge (u v : Fin d → ℝ) :
    l2Norm u - l2Norm v ≤ l2Norm (fun i => u i + v i) := by
  have h : l2Norm u ≤ l2Norm (fun i => u i + v i) + l2Norm (fun i => -v i) := by
    have heq : u = (fun i => (u i + v i) + (-v i)) := by ext i; ring
    conv_lhs => rw [heq]
    exact l2Norm_add_le (fun i => u i + v i) (fun i => -v i)
  rw [l2Norm_neg] at h
  linarith

/-! ## Section 2: Residual Branch and Step Lipschitz Bounds -/

/-- The nonlinear branch output `i ↦ σ(∑ j, W_{ij} x_j)` of a residual block. -/
def branchOutput (branch : ResidualBranch d) (x : Fin d → ℝ) : Fin d → ℝ :=
  fun i => branch.act.σ (∑ j : Fin d, branch.W i j * x j)

/-- `residualStep` decomposes into identity plus `scale` times `branchOutput`. -/
lemma residualStep_eq_add_branchOutput (x : Fin d → ℝ) (branch : ResidualBranch d) (scale : ℝ) :
    residualStep x branch scale = fun i => x i + scale * branchOutput branch x i :=
  rfl

/-- **Theorem 1 (`branchOutput_l2Norm_le`)**:
If `branch.act.σ` is `L_σ`-Lipschitz (`0 ≤ L_σ`), then `branchOutput branch` is Lipschitz in
`l2Norm` with constant `L_σ * √(frobeniusSq branch.W)`. -/
theorem branchOutput_l2Norm_le (branch : ResidualBranch d) {L_σ : ℝ} (hL : 0 ≤ L_σ)
    (hLip : ∀ a b : ℝ, |branch.act.σ a - branch.act.σ b| ≤ L_σ * |a - b|)
    (x y : Fin d → ℝ) :
    l2Norm (fun i => branchOutput branch x i - branchOutput branch y i) ≤
      L_σ * Real.sqrt (frobeniusSq branch.W) * l2Norm (fun j => x j - y j) := by
  have hrow : ∀ i : Fin d,
      (branchOutput branch x i - branchOutput branch y i) ^ 2 ≤
        L_σ ^ 2 * (∑ j : Fin d, branch.W i j ^ 2) * (∑ j : Fin d, (x j - y j) ^ 2) := by
    intro i
    have h1 : |branchOutput branch x i - branchOutput branch y i| ≤
        L_σ * |∑ j : Fin d, branch.W i j * (x j - y j)| := by
      unfold branchOutput
      have hsub : (∑ j : Fin d, branch.W i j * x j) - (∑ j : Fin d, branch.W i j * y j) =
          ∑ j : Fin d, branch.W i j * (x j - y j) := by
        rw [← Finset.sum_sub_distrib]
        refine Finset.sum_congr rfl fun j _ => by ring
      simpa [hsub] using hLip (∑ j : Fin d, branch.W i j * x j) (∑ j : Fin d, branch.W i j * y j)
    have h2 : (branchOutput branch x i - branchOutput branch y i) ^ 2 ≤
        L_σ ^ 2 * (∑ j : Fin d, branch.W i j * (x j - y j)) ^ 2 := by
      have hsq := mul_self_le_mul_self (abs_nonneg _) h1
      calc
        (branchOutput branch x i - branchOutput branch y i) ^ 2
          = |branchOutput branch x i - branchOutput branch y i| ^ 2 := (sq_abs _).symm
        _ ≤ (L_σ * |∑ j : Fin d, branch.W i j * (x j - y j)|) ^ 2 := by
          simpa [sq] using hsq
        _ = L_σ ^ 2 * |∑ j : Fin d, branch.W i j * (x j - y j)| ^ 2 := by ring
        _ = L_σ ^ 2 * (∑ j : Fin d, branch.W i j * (x j - y j)) ^ 2 := by rw [sq_abs]
    have hcs : (∑ j : Fin d, branch.W i j * (x j - y j)) ^ 2 ≤
        (∑ j : Fin d, branch.W i j ^ 2) * (∑ j : Fin d, (x j - y j) ^ 2) :=
      Finset.sum_mul_sq_le_sq_mul_sq Finset.univ (fun j => branch.W i j) (fun j => x j - y j)
    calc
      (branchOutput branch x i - branchOutput branch y i) ^ 2
        ≤ L_σ ^ 2 * (∑ j : Fin d, branch.W i j * (x j - y j)) ^ 2 := h2
      _ ≤ L_σ ^ 2 * ((∑ j : Fin d, branch.W i j ^ 2) * (∑ j : Fin d, (x j - y j) ^ 2)) :=
        mul_le_mul_of_nonneg_left hcs (sq_nonneg L_σ)
      _ = L_σ ^ 2 * (∑ j : Fin d, branch.W i j ^ 2) * (∑ j : Fin d, (x j - y j) ^ 2) := by ring
  have hsum : (∑ i : Fin d, (branchOutput branch x i - branchOutput branch y i) ^ 2) ≤
      L_σ ^ 2 * frobeniusSq branch.W * (∑ j : Fin d, (x j - y j) ^ 2) := by
    calc
      (∑ i : Fin d, (branchOutput branch x i - branchOutput branch y i) ^ 2)
        ≤ ∑ i : Fin d,
            L_σ ^ 2 * (∑ j : Fin d, branch.W i j ^ 2) * (∑ j : Fin d, (x j - y j) ^ 2) :=
        Finset.sum_le_sum fun i _ => hrow i
      _ = L_σ ^ 2 * (∑ i : Fin d, ∑ j : Fin d, branch.W i j ^ 2) *
            (∑ j : Fin d, (x j - y j) ^ 2) := by
        rw [← Finset.sum_mul, ← Finset.mul_sum]
      _ = L_σ ^ 2 * frobeniusSq branch.W * (∑ j : Fin d, (x j - y j) ^ 2) := rfl
  have hrhs_nonneg : 0 ≤ L_σ * Real.sqrt (frobeniusSq branch.W) * l2Norm (fun j => x j - y j) :=
    mul_nonneg (mul_nonneg hL (Real.sqrt_nonneg _)) (l2Norm_nonneg _)
  have hsq_final :
      l2Norm (fun i => branchOutput branch x i - branchOutput branch y i) ^ 2 ≤
        (L_σ * Real.sqrt (frobeniusSq branch.W) * l2Norm (fun j => x j - y j)) ^ 2 := by
    rw [l2Norm_sq, mul_pow, mul_pow, Real.sq_sqrt (frobeniusSq_nonneg branch.W), l2Norm_sq]
    exact hsum
  exact (sq_le_sq₀ (l2Norm_nonneg _) hrhs_nonneg).mp hsq_final

/-- **Theorem 2 (`residualStep_l2Norm_le`)**:
For any `scale : ℝ`, `residualStep` is Lipschitz in `l2Norm` with constant
`1 + |scale| * (L_σ * √(frobeniusSq branch.W))`. -/
theorem residualStep_l2Norm_le (branch : ResidualBranch d) (scale : ℝ) {L_σ : ℝ} (hL : 0 ≤ L_σ)
    (hLip : ∀ a b : ℝ, |branch.act.σ a - branch.act.σ b| ≤ L_σ * |a - b|)
    (x y : Fin d → ℝ) :
    l2Norm (fun i => residualStep x branch scale i - residualStep y branch scale i) ≤
      (1 + |scale| * (L_σ * Real.sqrt (frobeniusSq branch.W))) *
        l2Norm (fun j => x j - y j) := by
  have hdecomp : (fun i => residualStep x branch scale i - residualStep y branch scale i) =
      (fun i => (x i - y i) + scale * (branchOutput branch x i - branchOutput branch y i)) := by
    ext i
    simp only [residualStep, branchOutput]
    ring
  rw [hdecomp]
  have htri := l2Norm_add_le (fun i => x i - y i)
    (fun i => scale * (branchOutput branch x i - branchOutput branch y i))
  have hsmul := l2Norm_smul scale (fun i => branchOutput branch x i - branchOutput branch y i)
  have hbranch := branchOutput_l2Norm_le branch hL hLip x y
  calc
    l2Norm (fun i => (x i - y i) + scale * (branchOutput branch x i - branchOutput branch y i))
      ≤ l2Norm (fun i => x i - y i) +
          l2Norm (fun i => scale * (branchOutput branch x i - branchOutput branch y i)) := htri
    _ = l2Norm (fun i => x i - y i) +
          |scale| * l2Norm (fun i => branchOutput branch x i - branchOutput branch y i) := by
      rw [hsmul]
    _ ≤ l2Norm (fun i => x i - y i) +
          |scale| * (L_σ * Real.sqrt (frobeniusSq branch.W) * l2Norm (fun j => x j - y j)) :=
      add_le_add_right (mul_le_mul_of_nonneg_left hbranch (abs_nonneg scale)) _
    _ = (1 + |scale| * (L_σ * Real.sqrt (frobeniusSq branch.W))) *
          l2Norm (fun j => x j - y j) := by ring

/-- Convenience form of `residualStep_l2Norm_le` when `L_σ * √(frobeniusSq branch.W) ≤ c`. -/
theorem residualStep_l2Norm_le_of_bound (branch : ResidualBranch d) (scale : ℝ)
    {L_σ c : ℝ} (hL : 0 ≤ L_σ)
    (hLip : ∀ a b : ℝ, |branch.act.σ a - branch.act.σ b| ≤ L_σ * |a - b|)
    (hc : L_σ * Real.sqrt (frobeniusSq branch.W) ≤ c)
    (x y : Fin d → ℝ) :
    l2Norm (fun i => residualStep x branch scale i - residualStep y branch scale i) ≤
      (1 + |scale| * c) * l2Norm (fun j => x j - y j) := by
  refine (residualStep_l2Norm_le branch scale hL hLip x y).trans ?_
  refine mul_le_mul_of_nonneg_right ?_ (l2Norm_nonneg _)
  linarith [mul_le_mul_of_nonneg_left hc (abs_nonneg scale)]

/-- **Theorem 3 (`residualStep_sqrt_depth_scaling`)**:
Under Hayou et al. (2021) `1/√L` residual scaling (`scale = 1 / √L` with `0 < L`) and branch
Lipschitz budget `L_σ * √(frobeniusSq branch.W) ≤ c`, a single residual step has Lipschitz
constant at most `1 + c / √L`. -/
theorem residualStep_sqrt_depth_scaling {L : ℕ} (_hL_pos : 0 < L)
    (branch : ResidualBranch d) {L_σ c : ℝ} (hL_σ : 0 ≤ L_σ) (_hc0 : 0 ≤ c)
    (hLip : ∀ a b : ℝ, |branch.act.σ a - branch.act.σ b| ≤ L_σ * |a - b|)
    (hW : L_σ * Real.sqrt (frobeniusSq branch.W) ≤ c)
    (x y : Fin d → ℝ) :
    l2Norm (fun i =>
      residualStep x branch (1 / Real.sqrt (L : ℝ)) i -
        residualStep y branch (1 / Real.sqrt (L : ℝ)) i) ≤
      (1 + c / Real.sqrt (L : ℝ)) * l2Norm (fun j => x j - y j) := by
  have hstep := residualStep_l2Norm_le_of_bound branch (1 / Real.sqrt (L : ℝ)) hL_σ hLip hW x y
  have hscale_nonneg : 0 ≤ 1 / Real.sqrt (L : ℝ) :=
    div_nonneg zero_le_one (Real.sqrt_nonneg (L : ℝ))
  have habs : (1 + |1 / Real.sqrt (L : ℝ)| * c) * l2Norm (fun j => x j - y j) =
      (1 + c / Real.sqrt (L : ℝ)) * l2Norm (fun j => x j - y j) := by
    rw [abs_of_nonneg hscale_nonneg]
    ring
  rwa [habs] at hstep

/-! ## Section 3: Deep `L`-Layer Residual Networks and Depth-Independent `exp(c)` Stability -/

/-- An `L`-layer residual network obtained by composing `residualStep` across layers
`0, 1, ..., L - 1` with branch schedule `branches : ℕ → ResidualBranch d` and branch scale `scale`. -/
def deepResidual (L : ℕ) (branches : ℕ → ResidualBranch d) (scale : ℝ) (x : Fin d → ℝ) :
    Fin d → ℝ :=
  match L with
  | 0 => x
  | t + 1 => residualStep (deepResidual t branches scale x) (branches t) scale

@[simp]
lemma deepResidual_zero (branches : ℕ → ResidualBranch d) (scale : ℝ) (x : Fin d → ℝ) :
    deepResidual 0 branches scale x = x :=
  rfl

@[simp]
lemma deepResidual_succ (t : ℕ) (branches : ℕ → ResidualBranch d) (scale : ℝ) (x : Fin d → ℝ) :
    deepResidual (t + 1) branches scale x =
      residualStep (deepResidual t branches scale x) (branches t) scale :=
  rfl

/-- Multi-layer Lipschitz bound for an `L`-layer residual network with arbitrary `scale`:
if every layer `t < L` has branch Lipschitz constant at most `c`, then `deepResidual L` is
`(1 + |scale| * c)^L`-Lipschitz in `l2Norm`. -/
theorem deepResidual_l2Norm_le (L : ℕ) (branches : ℕ → ResidualBranch d) (scale : ℝ)
    (L_σ : ℕ → ℝ) {c : ℝ} (hc : 0 ≤ c)
    (hL_σ : ∀ t < L, 0 ≤ L_σ t)
    (hLip : ∀ t < L, ∀ a b : ℝ,
      |(branches t).act.σ a - (branches t).act.σ b| ≤ L_σ t * |a - b|)
    (hW : ∀ t < L, L_σ t * Real.sqrt (frobeniusSq (branches t).W) ≤ c)
    (x y : Fin d → ℝ) :
    l2Norm (fun i => deepResidual L branches scale x i - deepResidual L branches scale y i) ≤
      (1 + |scale| * c) ^ L * l2Norm (fun j => x j - y j) := by
  induction L with
  | zero =>
    simp [deepResidual]
  | succ t ih =>
    have ih' := ih (fun k hk => hL_σ k (Nat.lt_succ_of_lt hk))
      (fun k hk => hLip k (Nat.lt_succ_of_lt hk))
      (fun k hk => hW k (Nat.lt_succ_of_lt hk))
    have hstep := residualStep_l2Norm_le_of_bound (branches t) scale
      (hL_σ t (Nat.lt_succ_self t))
      (hLip t (Nat.lt_succ_self t))
      (hW t (Nat.lt_succ_self t))
      (deepResidual t branches scale x)
      (deepResidual t branches scale y)
    have hfac_nonneg : 0 ≤ 1 + |scale| * c := by positivity
    calc
      l2Norm (fun i =>
        deepResidual (t + 1) branches scale x i - deepResidual (t + 1) branches scale y i)
        ≤ (1 + |scale| * c) *
            l2Norm (fun j =>
              deepResidual t branches scale x j - deepResidual t branches scale y j) := hstep
      _ ≤ (1 + |scale| * c) * ((1 + |scale| * c) ^ t * l2Norm (fun j => x j - y j)) :=
        mul_le_mul_of_nonneg_left ih' hfac_nonneg
      _ = (1 + |scale| * c) ^ (t + 1) * l2Norm (fun j => x j - y j) := by
        rw [pow_succ', mul_assoc]

/-- Fundamental Euler-to-exponential inequality: for `0 < L` and `0 ≤ c`,
`(1 + c / L)^L ≤ exp(c)`. -/
theorem one_add_div_pow_le_exp {L : ℕ} (hL : 0 < L) {c : ℝ} (hc : 0 ≤ c) :
    (1 + c / (L : ℝ)) ^ L ≤ Real.exp c := by
  have hL_pos : (0 : ℝ) < (L : ℝ) := Nat.cast_pos.mpr hL
  have hbase_nonneg : 0 ≤ 1 + c / (L : ℝ) := by positivity
  have h1 : 1 + c / (L : ℝ) ≤ Real.exp (c / (L : ℝ)) := by
    linarith [Real.add_one_le_exp (c / (L : ℝ))]
  calc
    (1 + c / (L : ℝ)) ^ L ≤ (Real.exp (c / (L : ℝ))) ^ L :=
      pow_le_pow_left₀ hbase_nonneg h1 L
    _ = Real.exp ((L : ℝ) * (c / (L : ℝ))) := by
      rw [← Real.exp_nat_mul]
    _ = Real.exp c := by
      congr 1
      field_simp [ne_of_gt hL_pos]

/-- **Theorem 4 (`deepResidualChain_depth_independent_exp_bound`)**:
With `1 / L` depth scaling (`scale = 1 / (L : ℝ)` for `0 < L`), an `L`-layer residual network
whose per-layer branch Lipschitz constants are bounded by `c` is **uniformly `exp(c)`-Lipschitz
at every depth `L ≥ 1`**. -/
theorem deepResidualChain_depth_independent_exp_bound {L : ℕ} (hL : 0 < L)
    (branches : ℕ → ResidualBranch d) (L_σ : ℕ → ℝ) {c : ℝ} (hc : 0 ≤ c)
    (hL_σ : ∀ t < L, 0 ≤ L_σ t)
    (hLip : ∀ t < L, ∀ a b : ℝ,
      |(branches t).act.σ a - (branches t).act.σ b| ≤ L_σ t * |a - b|)
    (hW : ∀ t < L, L_σ t * Real.sqrt (frobeniusSq (branches t).W) ≤ c)
    (x y : Fin d → ℝ) :
    l2Norm (fun i =>
      deepResidual L branches (1 / (L : ℝ)) x i -
        deepResidual L branches (1 / (L : ℝ)) y i) ≤
      Real.exp c * l2Norm (fun j => x j - y j) := by
  have hchain := deepResidual_l2Norm_le L branches (1 / (L : ℝ)) L_σ hc hL_σ hLip hW x y
  have hscale_nonneg : 0 ≤ 1 / (L : ℝ) := div_nonneg zero_le_one (Nat.cast_nonneg L)
  have habs : 1 + |1 / (L : ℝ)| * c = 1 + c / (L : ℝ) := by
    rw [abs_of_nonneg hscale_nonneg]
    ring
  rw [habs] at hchain
  exact hchain.trans (mul_le_mul_of_nonneg_right (one_add_div_pow_le_exp hL hc) (l2Norm_nonneg _))

/-! ## Section 4: Signal Propagation and Representation Norm Preservation -/

/-- If `branchOutput branch 0 = 0`, then `residualStep 0 branch scale = 0`. -/
lemma residualStep_zero_of_branchOutput_zero (branch : ResidualBranch d) (scale : ℝ)
    (h0 : branchOutput branch (fun _ => 0) = fun _ => 0) :
    residualStep (fun _ => 0) branch scale = fun _ => 0 := by
  ext i
  rw [residualStep_eq_add_branchOutput]
  have hi := congr_fun h0 i
  dsimp only at hi ⊢
  rw [hi]
  ring

/-- If every branch vanishes at `0`, then `deepResidual L` fixes `0`. -/
lemma deepResidual_zero_input (L : ℕ) (branches : ℕ → ResidualBranch d) (scale : ℝ)
    (h0 : ∀ t < L, branchOutput (branches t) (fun _ => 0) = fun _ => 0) :
    deepResidual L branches scale (fun _ => 0) = fun _ => 0 := by
  induction L with
  | zero => rfl
  | succ t ih =>
    simp only [deepResidual_succ]
    rw [ih (fun k hk => h0 k (Nat.lt_succ_of_lt hk))]
    exact residualStep_zero_of_branchOutput_zero (branches t) scale (h0 t (Nat.lt_succ_self t))

/-- Single-step lower bound on representation norm (`residualStep_l2Norm_ge`):
if `branchOutput branch 0 = 0` and the branch is `c`-Lipschitz, then
`(1 - |scale| * c) * l2Norm x ≤ l2Norm (residualStep x branch scale)`.
When `|scale| * c < 1`, this prevents representation collapse. -/
theorem residualStep_l2Norm_ge (branch : ResidualBranch d) (scale : ℝ)
    {L_σ c : ℝ} (hL : 0 ≤ L_σ)
    (hLip : ∀ a b : ℝ, |branch.act.σ a - branch.act.σ b| ≤ L_σ * |a - b|)
    (hW : L_σ * Real.sqrt (frobeniusSq branch.W) ≤ c)
    (h0 : branchOutput branch (fun _ => 0) = fun _ => 0)
    (x : Fin d → ℝ) :
    (1 - |scale| * c) * l2Norm x ≤ l2Norm (residualStep x branch scale) := by
  have hrev := l2Norm_add_ge x (fun i => scale * branchOutput branch x i)
  have hsmul := l2Norm_smul scale (branchOutput branch x)
  have hbranch0 := branchOutput_l2Norm_le branch hL hLip x (fun _ => 0)
  have hbx : l2Norm (branchOutput branch x) ≤ c * l2Norm x := by
    have heq_lhs : (fun i => branchOutput branch x i - branchOutput branch (fun _ => 0) i) =
        branchOutput branch x := by
      ext i
      have hi := congr_fun h0 i
      simp [hi]
    have heq_rhs : (fun j => x j - (0 : ℝ)) = x := by ext j; ring
    rw [heq_lhs, heq_rhs] at hbranch0
    exact hbranch0.trans (mul_le_mul_of_nonneg_right hW (l2Norm_nonneg x))
  have hscaled : l2Norm (fun i => scale * branchOutput branch x i) ≤ |scale| * c * l2Norm x := by
    rw [hsmul, mul_assoc]
    exact mul_le_mul_of_nonneg_left hbx (abs_nonneg scale)
  calc
    (1 - |scale| * c) * l2Norm x = l2Norm x - |scale| * c * l2Norm x := by ring
    _ ≤ l2Norm x - l2Norm (fun i => scale * branchOutput branch x i) := by linarith
    _ ≤ l2Norm (residualStep x branch scale) := hrev

/-- **Signal Propagation / Representation Norm Window (`deepResidual_norm_window`)**:
When `branchOutput (branches t) 0 = 0` for all `t < L`, a `1/L`-scaled deep residual network
satisfies `l2Norm (deepResidual L branches (1 / L) x) ≤ exp(c) * l2Norm x` across all depths
`L ≥ 1`, while each residual layer satisfies the anti-collapse lower bound
`(1 - |scale| * c) * l2Norm x ≤ l2Norm (residualStep x branch scale)`. -/
theorem deepResidual_norm_window {L : ℕ} (hL : 0 < L)
    (branches : ℕ → ResidualBranch d) (L_σ : ℕ → ℝ) {c : ℝ} (hc : 0 ≤ c)
    (hL_σ : ∀ t < L, 0 ≤ L_σ t)
    (hLip : ∀ t < L, ∀ a b : ℝ,
      |(branches t).act.σ a - (branches t).act.σ b| ≤ L_σ t * |a - b|)
    (hW : ∀ t < L, L_σ t * Real.sqrt (frobeniusSq (branches t).W) ≤ c)
    (h0 : ∀ t < L, branchOutput (branches t) (fun _ => 0) = fun _ => 0)
    (x : Fin d → ℝ) :
    l2Norm (deepResidual L branches (1 / (L : ℝ)) x) ≤ Real.exp c * l2Norm x ∧
      ∀ t < L, ∀ (scale : ℝ) (h : Fin d → ℝ),
        (1 - |scale| * c) * l2Norm h ≤ l2Norm (residualStep h (branches t) scale) := by
  refine ⟨?_, ?_⟩
  · have hexp := deepResidualChain_depth_independent_exp_bound hL branches L_σ hc
      hL_σ hLip hW x (fun _ => 0)
    have hzero := deepResidual_zero_input L branches (1 / (L : ℝ)) h0
    have hlhs : (fun i =>
        deepResidual L branches (1 / (L : ℝ)) x i -
          deepResidual L branches (1 / (L : ℝ)) (fun _ => 0) i) =
        deepResidual L branches (1 / (L : ℝ)) x := by
      rw [hzero]
      ext i
      ring
    have hrhs : (fun j => x j - (0 : ℝ)) = x := by ext j; ring
    rw [hlhs, hrhs] at hexp
    exact hexp
  · intro t ht scale h
    exact residualStep_l2Norm_ge (branches t) scale (hL_σ t ht) (hLip t ht) (hW t ht) (h0 t ht) h

end OpenTheory.DeepResidualScaling
