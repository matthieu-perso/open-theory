import OpenTheory.Data.Foundation
import OpenTheory.Optimization.LocalSharpnessDescent

/-!
# Diagonal Preconditioned Descent, Anisotropic Sharpness Speedup, and SignGD / Adam-Style Optimization

Why do adaptive and diagonal-preconditioned optimizers (Adam, RMSProp, SignGD, diagonal Gauss-Newton,
and $\mu$P layerwise learning rates) train deep networks and Transformers substantially faster than
scalar-step SGD?

In deep architectures, the loss curvature is **highly anisotropic across coordinates, layers, and
attention heads**: a few sharp coordinates have large curvature `S_max`, while the vast majority of
coordinates have much smaller curvature `S_i ≪ S_max`. Scalar-step SGD is forced to choose a single
conservative learning rate `η ≤ 1 / S_max` dictated by the sharpest direction, whereas a diagonal
preconditioned optimizer steps with coordinate-wise learning rates `η_i = 1 / S_i` matched to each
coordinate's local sharpness.

This module proves four core theorem families with zero `sorry`:

1. **Coordinate-Wise Anisotropic Sharpness & Optimal Diagonal Preconditioning
   (`preconditionedStep`, `coordProgress`, `anisotropic_sharpness_descent`,
   `optimal_diagonal_progress_eq`, `coordProgress_sub_eq_sum_sq`,
   `optimal_diagonal_maximizes_progress`)**:
   Under a coordinate-wise quadratic upper bound with sharpness vector `S : Fin p → ℝ`, taking step
   `preconditionedStep η u g` decreases the objective by at least `coordProgress η S g`.
   The optimal diagonal preconditioner `optimalDiagStep S i = 1 / S i` achieves progress
   `∑ i, (g i)^2 / (2 * S i)` and uniquely maximizes `coordProgress η S g` over all step vectors
   `η : Fin p → ℝ` via an exact coordinate-wise completion-of-squares identity.

2. **Exact Dominance & Speedup Over Scalar SGD
   (`scalarMaxProgress`, `scalar_progress_eq`, `diagonal_vs_scalar_max_gap_eq`,
   `diagonal_vs_scalar_gap_eq`, `diagonal_dominates_scalar_max`, `diagonal_dominates_scalar_sgd`,
   `diagonal_strictly_beats_scalar_max`, `diagonal_strictly_beats_scalar_sgd`,
   `diagonal_dominates_any_scalar_step`)**:
   Against the conservative scalar step `1 / S_max` (which achieves `‖g‖^2 / (2 * S_max)`), optimal
   diagonal preconditioning gains the exact non-negative gap
   `∑ i, ((S_max - S i) / (2 * S i * S_max)) * (g i)^2`, strictly dominating scalar SGD whenever
   even a single active gradient coordinate has `S i₀ < S_max`, and dominating `coordProgress` for
   every constant scalar step `fun _ => η`.

3. **SignGD / Normalized Coordinate Descent (Memoryless Adam with $\beta_1 = \beta_2 = 0, \varepsilon \to 0$)
   (`realSign`, `realSign_mul_self`, `realSign_sq_le_one`, `l1Norm`, `sharpnessL1`, `signStep`,
   `signStep_eq_preconditionedStep`, `signGD_progress_eq`, `signGD_descent`,
   `signGD_optimal_lr_drop`, `signGD_strictly_beats_scalar_sgd`)**:
   SignGD steps by `-η * realSign (g i)` in each coordinate, which is an exact instance of
   `preconditionedStep` with effective step sizes `η / |g i|`. Its linear progress is governed by
   the $\ell_1$ gradient norm `l1Norm g = ∑ i, |g i|` while its curvature penalty is governed by
   the $\ell_1$ sharpness `sharpnessL1 S = ∑ i, S i`, yielding an optimal one-step loss drop of
   `(l1Norm g)^2 / (2 * sharpnessL1 S)` that strictly beats scalar SGD on dense gradients when
   `(l1Norm g)^2 / sharpnessL1 S > ‖g‖^2 / S_max`.

4. **Preconditioned Polyak-Łojasiewicz (PL) Geometric Convergence
   (`preconditioned_pl_one_step_contraction`, `preconditioned_pl_geometric_convergence`)**:
   When the preconditioned gradient norm satisfies the PL condition
   `∑ i, (g_t i)^2 / S i ≥ 2 * μ_P * (L (u t) - floor)` with `0 < μ_P ≤ 1`, optimal diagonal
   preconditioned descent converges geometrically as
   `L (u T) - floor ≤ (1 - μ_P)^T * (L (u 0) - floor)`.
-/

noncomputable section

open scoped BigOperators

namespace OpenTheory.PreconditionedDescent

variable {p : ℕ}

/-! ## 1. Coordinate-Wise Anisotropic Sharpness & Optimal Diagonal Preconditioning -/

/-- Diagonal preconditioned gradient step on `Parameters p` with coordinate-wise learning rates
`η : Fin p → ℝ` along gradient `g : Parameters p`:
`(preconditionedStep η u g) i = u i - η i * g i`. -/
def preconditionedStep (η : Fin p → ℝ) (u g : Parameters p) : Parameters p :=
  WithLp.toLp 2 (fun i => u i - η i * g i)

@[simp]
theorem preconditionedStep_apply (η : Fin p → ℝ) (u g : Parameters p) (i : Fin p) :
    preconditionedStep η u g i = u i - η i * g i :=
  rfl

/-- Guaranteed one-step objective progress under coordinate-wise learning rates `η : Fin p → ℝ`
and coordinate-wise sharpness `S : Fin p → ℝ`:
`∑ i : Fin p, η i * (1 - η i * S i / 2) * (g i) ^ 2`. -/
def coordProgress (η S : Fin p → ℝ) (g : Parameters p) : ℝ :=
  ∑ i : Fin p, η i * (1 - η i * S i / 2) * (g i) ^ 2

/-- Optimal diagonal preconditioner `η_i = 1 / S_i` matched to each coordinate's sharpness. -/
def optimalDiagStep (S : Fin p → ℝ) : Fin p → ℝ :=
  fun i => 1 / S i

@[simp]
theorem optimalDiagStep_apply (S : Fin p → ℝ) (i : Fin p) :
    optimalDiagStep S i = 1 / S i :=
  rfl

/-- **Anisotropic Sharpness Descent Lemma (`anisotropic_sharpness_descent`)**:
If `L : Parameters p → ℝ` satisfies the coordinate-wise quadratic upper bound with sharpness vector
`S : Fin p → ℝ` at `u` along `g`, then `preconditionedStep η u g` decreases `L` by at least
`coordProgress η S g`. -/
theorem anisotropic_sharpness_descent (L : Parameters p → ℝ) (u g : Parameters p)
    (η S : Fin p → ℝ)
    (hquad : L (preconditionedStep η u g) ≤
      L u - ∑ i : Fin p, η i * (g i) ^ 2 + (1 / 2) * ∑ i : Fin p, S i * (η i) ^ 2 * (g i) ^ 2) :
    L (preconditionedStep η u g) ≤ L u - coordProgress η S g := by
  have hcombine :
      ∑ i : Fin p, η i * (g i) ^ 2 =
        (1 / 2) * ∑ i : Fin p, S i * (η i) ^ 2 * (g i) ^ 2 + coordProgress η S g := by
    unfold coordProgress
    rw [Finset.mul_sum, ← Finset.sum_add_distrib]
    refine Finset.sum_congr rfl fun i _ => ?_
    ring
  linarith

/-- Under positive coordinate-wise sharpness `∀ i, 0 < S i`, the optimal diagonal preconditioner
`optimalDiagStep S` achieves guaranteed progress `∑ i : Fin p, (g i) ^ 2 / (2 * S i)`. -/
theorem optimal_diagonal_progress_eq (S : Fin p → ℝ) (g : Parameters p)
    (hS : ∀ i, 0 < S i) :
    coordProgress (optimalDiagStep S) S g = ∑ i : Fin p, (g i) ^ 2 / (2 * S i) := by
  unfold coordProgress optimalDiagStep
  refine Finset.sum_congr rfl fun i _ => ?_
  have hSi : S i ≠ 0 := ne_of_gt (hS i)
  field_simp [hSi]
  ring

/-- Combining `anisotropic_sharpness_descent` with `optimal_diagonal_progress_eq` gives the optimal
one-step diagonal preconditioned loss drop:
`L (preconditionedStep (optimalDiagStep S) u g) ≤ L u - ∑ i, (g i) ^ 2 / (2 * S i)`. -/
theorem optimal_diagonal_descent_drop (L : Parameters p → ℝ) (u g : Parameters p)
    (S : Fin p → ℝ) (hS : ∀ i, 0 < S i)
    (hquad : L (preconditionedStep (optimalDiagStep S) u g) ≤
      L u - ∑ i : Fin p, optimalDiagStep S i * (g i) ^ 2 +
        (1 / 2) * ∑ i : Fin p, S i * (optimalDiagStep S i) ^ 2 * (g i) ^ 2) :
    L (preconditionedStep (optimalDiagStep S) u g) ≤
      L u - ∑ i : Fin p, (g i) ^ 2 / (2 * S i) := by
  have hdesc := anisotropic_sharpness_descent L u g (optimalDiagStep S) S hquad
  rwa [optimal_diagonal_progress_eq S g hS] at hdesc

/-- **Coordinate-Wise Completion of Squares (`coordProgress_sub_eq_sum_sq`)**:
For ANY step vector `η : Fin p → ℝ` and positive sharpness vector `S : Fin p → ℝ`, the progress
suboptimality gap relative to `optimalDiagStep S` is an exact sum of squares:
`coordProgress (optimalDiagStep S) S g - coordProgress η S g = ∑ i, (S i / 2) * (η i - 1 / S i)^2 * (g i)^2`. -/
theorem coordProgress_sub_eq_sum_sq (η S : Fin p → ℝ) (g : Parameters p)
    (hS : ∀ i, 0 < S i) :
    coordProgress (optimalDiagStep S) S g - coordProgress η S g =
      ∑ i : Fin p, (S i / 2) * (η i - 1 / S i) ^ 2 * (g i) ^ 2 := by
  unfold coordProgress optimalDiagStep
  rw [← Finset.sum_sub_distrib]
  refine Finset.sum_congr rfl fun i _ => ?_
  have hSi : S i ≠ 0 := ne_of_gt (hS i)
  field_simp [hSi]
  ring

/-- **Global Optimality of the Diagonal Sharpness Preconditioner
(`optimal_diagonal_maximizes_progress`)**:
For any positive sharpness vector `S : Fin p → ℝ` and any coordinate step vector `η : Fin p → ℝ`,
`optimalDiagStep S` achieves at least as much guaranteed progress as `η`. -/
theorem optimal_diagonal_maximizes_progress (η S : Fin p → ℝ) (g : Parameters p)
    (hS : ∀ i, 0 < S i) :
    coordProgress η S g ≤ coordProgress (optimalDiagStep S) S g := by
  have hdiff := coordProgress_sub_eq_sum_sq η S g hS
  have hnonneg : 0 ≤ ∑ i : Fin p, (S i / 2) * (η i - 1 / S i) ^ 2 * (g i) ^ 2 := by
    refine Finset.sum_nonneg fun i _ => ?_
    have hSi : 0 < S i := hS i
    positivity
  linarith

/-! ## 2. Exact Dominance & Speedup Over Scalar SGD -/

/-- Guaranteed one-step progress of conservative scalar SGD with step size `1 / S_max` against
worst-case sharpness `S_max`:
`scalarMaxProgress S_max g = ∑ i : Fin p, (g i) ^ 2 / (2 * S_max)`. -/
def scalarMaxProgress (S_max : ℝ) (g : Parameters p) : ℝ :=
  ∑ i : Fin p, (g i) ^ 2 / (2 * S_max)

/-- Conservative scalar SGD progress equals `‖g‖ ^ 2 / (2 * S_max)`. -/
theorem scalar_progress_eq (S_max : ℝ) (g : Parameters p) :
    scalarMaxProgress S_max g = ‖g‖ ^ 2 / (2 * S_max) := by
  unfold scalarMaxProgress
  rw [parameters_norm_sq, Finset.sum_div]

/-- At constant sharpness `fun _ => S_max` (`S_max ≠ 0`), `coordProgress` with scalar step
`fun _ => 1 / S_max` equals `‖g‖ ^ 2 / (2 * S_max)`. -/
theorem coordProgress_scalar_max_eq (S_max : ℝ) (g : Parameters p) (hS_max : S_max ≠ 0) :
    coordProgress (fun _ => 1 / S_max) (fun _ => S_max) g = ‖g‖ ^ 2 / (2 * S_max) := by
  rw [← scalar_progress_eq]
  unfold coordProgress scalarMaxProgress
  refine Finset.sum_congr rfl fun i _ => ?_
  field_simp [hS_max]
  ring

/-- **Exact Algebraic Speedup Gap Over Conservative Scalar SGD (`diagonal_vs_scalar_max_gap_eq`)**:
The improvement of optimal diagonal preconditioning over conservative scalar SGD is the exact
coordinate sum `∑ i : Fin p, ((S_max - S i) / (2 * S i * S_max)) * (g i) ^ 2`. -/
theorem diagonal_vs_scalar_max_gap_eq (S : Fin p → ℝ) (S_max : ℝ) (g : Parameters p)
    (hS : ∀ i, 0 < S i) (hS_max : 0 < S_max) :
    coordProgress (optimalDiagStep S) S g - ∑ i : Fin p, (g i) ^ 2 / (2 * S_max) =
      ∑ i : Fin p, ((S_max - S i) / (2 * S i * S_max)) * (g i) ^ 2 := by
  rw [optimal_diagonal_progress_eq S g hS, ← Finset.sum_sub_distrib]
  refine Finset.sum_congr rfl fun i _ => ?_
  have hSi : S i ≠ 0 := ne_of_gt (hS i)
  have hSm : S_max ≠ 0 := ne_of_gt hS_max
  field_simp [hSi, hSm]

/-- Alias of `diagonal_vs_scalar_max_gap_eq` expressed with `scalarMaxProgress`. -/
theorem diagonal_vs_scalar_gap_eq (S : Fin p → ℝ) (S_max : ℝ) (g : Parameters p)
    (hS : ∀ i, 0 < S i) (hS_max : 0 < S_max) :
    coordProgress (optimalDiagStep S) S g - scalarMaxProgress S_max g =
      ∑ i : Fin p, ((S_max - S i) / (2 * S i * S_max)) * (g i) ^ 2 :=
  diagonal_vs_scalar_max_gap_eq S S_max g hS hS_max

/-- **Dominance of Diagonal Preconditioning Over Conservative Scalar SGD
(`diagonal_dominates_scalar_max`)**:
Whenever `0 < S i ≤ S_max` for all coordinates `i`, optimal diagonal preconditioning achieves
at least as much progress as conservative scalar SGD (`∑ i, (g i)^2 / (2 * S_max)`). -/
theorem diagonal_dominates_scalar_max (S : Fin p → ℝ) (S_max : ℝ) (g : Parameters p)
    (hS : ∀ i, 0 < S i) (hS_le : ∀ i, S i ≤ S_max) (hS_max : 0 < S_max) :
    ∑ i : Fin p, (g i) ^ 2 / (2 * S_max) ≤ coordProgress (optimalDiagStep S) S g := by
  have hgap := diagonal_vs_scalar_max_gap_eq S S_max g hS hS_max
  have hnonneg : 0 ≤ ∑ i : Fin p, ((S_max - S i) / (2 * S i * S_max)) * (g i) ^ 2 := by
    refine Finset.sum_nonneg fun i _ => ?_
    have hSi : 0 < S i := hS i
    have hsub : 0 ≤ S_max - S i := sub_nonneg.mpr (hS_le i)
    positivity
  linarith

/-- Dominance of diagonal preconditioning over conservative scalar SGD expressed via
`scalarMaxProgress`. -/
theorem diagonal_dominates_scalar_sgd (S : Fin p → ℝ) (S_max : ℝ) (g : Parameters p)
    (hS : ∀ i, 0 < S i) (hS_le : ∀ i, S i ≤ S_max) (hS_max : 0 < S_max) :
    scalarMaxProgress S_max g ≤ coordProgress (optimalDiagStep S) S g :=
  diagonal_dominates_scalar_max S S_max g hS hS_le hS_max

/-- **Strict Speedup of Diagonal Preconditioning Under Anisotropic Curvature
(`diagonal_strictly_beats_scalar_max`)**:
If `0 < S i ≤ S_max` for all `i` and there exists even a single coordinate `i₀ : Fin p` with
strictly smaller sharpness `S i₀ < S_max` and non-zero gradient `g i₀ ≠ 0`, then optimal diagonal
preconditioning strictly beats conservative scalar SGD. -/
theorem diagonal_strictly_beats_scalar_max (S : Fin p → ℝ) (S_max : ℝ) (g : Parameters p)
    (hS : ∀ i, 0 < S i) (hS_le : ∀ i, S i ≤ S_max)
    (i₀ : Fin p) (hi₀_lt : S i₀ < S_max) (hgi₀ : g i₀ ≠ 0) :
    ∑ i : Fin p, (g i) ^ 2 / (2 * S_max) < coordProgress (optimalDiagStep S) S g := by
  have hS_max : 0 < S_max := (hS i₀).trans hi₀_lt
  have hgap := diagonal_vs_scalar_max_gap_eq S S_max g hS hS_max
  have hterm_nonneg : ∀ i ∈ Finset.univ,
      0 ≤ ((S_max - S i) / (2 * S i * S_max)) * (g i) ^ 2 := by
    intro i _
    have hSi : 0 < S i := hS i
    have hsub : 0 ≤ S_max - S i := sub_nonneg.mpr (hS_le i)
    positivity
  have hterm_pos : 0 < ((S_max - S i₀) / (2 * S i₀ * S_max)) * (g i₀) ^ 2 := by
    have hSi₀ : 0 < S i₀ := hS i₀
    have hsub : 0 < S_max - S i₀ := sub_pos.mpr hi₀_lt
    have hg2 : 0 < (g i₀) ^ 2 := sq_pos_of_ne_zero hgi₀
    positivity
  have hsum_pos : 0 < ∑ i : Fin p, ((S_max - S i) / (2 * S i * S_max)) * (g i) ^ 2 :=
    Finset.sum_pos' hterm_nonneg ⟨i₀, Finset.mem_univ i₀, hterm_pos⟩
  linarith

/-- Strict speedup of diagonal preconditioning over conservative scalar SGD expressed via
`scalarMaxProgress`. -/
theorem diagonal_strictly_beats_scalar_sgd (S : Fin p → ℝ) (S_max : ℝ) (g : Parameters p)
    (hS : ∀ i, 0 < S i) (hS_le : ∀ i, S i ≤ S_max)
    (i₀ : Fin p) (hi₀_lt : S i₀ < S_max) (hgi₀ : g i₀ ≠ 0) :
    scalarMaxProgress S_max g < coordProgress (optimalDiagStep S) S g :=
  diagonal_strictly_beats_scalar_max S S_max g hS hS_le i₀ hi₀_lt hgi₀

/-- **Dominance Over Every Scalar Step Size (`diagonal_dominates_any_scalar_step`)**:
Under coordinate-wise sharpness `S`, the optimal diagonal preconditioner beats the coordinate
progress of ANY constant scalar step size `η : ℝ`, with exact gap
`∑ i : Fin p, (S i / 2) * (η - 1 / S i) ^ 2 * (g i) ^ 2`. -/
theorem diagonal_dominates_any_scalar_step (η : ℝ) (S : Fin p → ℝ) (g : Parameters p)
    (hS : ∀ i, 0 < S i) :
    coordProgress (fun _ => η) S g ≤ coordProgress (optimalDiagStep S) S g ∧
      coordProgress (optimalDiagStep S) S g - coordProgress (fun _ => η) S g =
        ∑ i : Fin p, (S i / 2) * (η - 1 / S i) ^ 2 * (g i) ^ 2 :=
  ⟨optimal_diagonal_maximizes_progress (fun _ => η) S g hS,
    coordProgress_sub_eq_sum_sq (fun _ => η) S g hS⟩

/-! ## 3. SignGD / Normalized Coordinate Descent (Adam with $\beta_1 = \beta_2 = 0, \varepsilon \to 0$) -/

/-- Exact sign function on `ℝ`: `1` for positive, `-1` for negative, and `0` at `0`. -/
def realSign (x : ℝ) : ℝ :=
  if 0 < x then 1 else if x < 0 then -1 else 0

/-- Multiplying `realSign x` by `x` recovers the absolute value `|x|`. -/
theorem realSign_mul_self (x : ℝ) : realSign x * x = |x| := by
  unfold realSign
  split_ifs with hpos hneg
  · rw [one_mul, abs_of_pos hpos]
  · rw [neg_one_mul, abs_of_neg hneg]
  · have hx : x = 0 := by linarith
    subst hx
    simp

/-- Multiplying `x` by `realSign x` recovers `|x|`. -/
theorem self_mul_realSign (x : ℝ) : x * realSign x = |x| := by
  rw [mul_comm, realSign_mul_self]

/-- The square of `realSign x` is always at most `1`. -/
theorem realSign_sq_le_one (x : ℝ) : (realSign x) ^ 2 ≤ 1 := by
  unfold realSign
  split_ifs <;> norm_num

/-- The square of `realSign x` is non-negative. -/
theorem realSign_sq_nonneg (x : ℝ) : 0 ≤ (realSign x) ^ 2 :=
  sq_nonneg _

/-- The $\ell_1$ norm of a gradient vector `g : Parameters p`: `∑ i : Fin p, |g i|`. -/
def l1Norm (g : Parameters p) : ℝ :=
  ∑ i : Fin p, |g i|

/-- Non-negativity of `l1Norm`. -/
theorem l1Norm_nonneg (g : Parameters p) : 0 ≤ l1Norm g :=
  Finset.sum_nonneg fun i _ => abs_nonneg (g i)

/-- The $\ell_1$ total sharpness across coordinates: `∑ i : Fin p, S i`. -/
def sharpnessL1 (S : Fin p → ℝ) : ℝ :=
  ∑ i : Fin p, S i

/-- SignGD update on `Parameters p` with scalar step size `η : ℝ`:
`(signStep η u g) i = u i - η * realSign (g i)`. -/
def signStep (η : ℝ) (u g : Parameters p) : Parameters p :=
  WithLp.toLp 2 (fun i => u i - η * realSign (g i))

@[simp]
theorem signStep_apply (η : ℝ) (u g : Parameters p) (i : Fin p) :
    signStep η u g i = u i - η * realSign (g i) :=
  rfl

/-- Effective coordinate-wise learning rates `η / |g i|` (with `0` when `g i = 0`) that realize
`signStep η u g` as a diagonal preconditioned step `preconditionedStep`. -/
def signStepVec (η : ℝ) (g : Parameters p) : Fin p → ℝ :=
  fun i => if g i = 0 then 0 else η / |g i|

/-- Key coordinate identity: `signStepVec η g i * g i = η * realSign (g i)` for all `i`. -/
theorem signStepVec_mul_grad (η : ℝ) (g : Parameters p) (i : Fin p) :
    signStepVec η g i * g i = η * realSign (g i) := by
  unfold signStepVec realSign
  split_ifs with hzero hpos hneg hpos2 hneg2
  · linarith
  · linarith
  · ring
  · rw [abs_of_pos hpos2]
    field_simp [hzero]
  · rw [abs_of_neg hneg2]
    field_simp [hzero]
  · exfalso
    exact hzero (by linarith)

/-- **SignGD is a Diagonal Preconditioned Step (`signStep_eq_preconditionedStep`)**:
SignGD with learning rate `η` is identical to `preconditionedStep` with coordinate learning rates
`signStepVec η g`. -/
theorem signStep_eq_preconditionedStep (η : ℝ) (u g : Parameters p) :
    signStep η u g = preconditionedStep (signStepVec η g) u g := by
  ext i
  simp [signStep, preconditionedStep, signStepVec_mul_grad]

/-- Linear progress of a SignGD step along `g` equals `η * l1Norm g`. -/
theorem signGD_progress_eq (η : ℝ) (g : Parameters p) :
    (∑ i : Fin p, η * realSign (g i) * g i) = η * l1Norm g := by
  unfold l1Norm
  rw [Finset.mul_sum]
  refine Finset.sum_congr rfl fun i _ => ?_
  rw [mul_assoc, realSign_mul_self]

/-- **SignGD Descent Bound (`signGD_descent`)**:
Under coordinate-wise non-negative sharpness `S i ≥ 0`, a SignGD step `signStep η u g` decreases
the objective according to the $\ell_1$ gradient norm and $\ell_1$ sharpness sum:
`L (signStep η u g) ≤ L u - η * l1Norm g + (η ^ 2 / 2) * sharpnessL1 S`. -/
theorem signGD_descent (L : Parameters p → ℝ) (u g : Parameters p)
    (η : ℝ) (S : Fin p → ℝ) (hS : ∀ i, 0 ≤ S i)
    (hquad : L (signStep η u g) ≤
      L u - ∑ i : Fin p, η * realSign (g i) * g i +
        (1 / 2) * ∑ i : Fin p, S i * (η * realSign (g i)) ^ 2) :
    L (signStep η u g) ≤ L u - η * l1Norm g + (η ^ 2 / 2) * sharpnessL1 S := by
  rw [signGD_progress_eq] at hquad
  have hcurv :
      (1 / 2) * ∑ i : Fin p, S i * (η * realSign (g i)) ^ 2 ≤
        (η ^ 2 / 2) * sharpnessL1 S := by
    unfold sharpnessL1
    have hfactor :
        (1 / 2) * ∑ i : Fin p, S i * (η * realSign (g i)) ^ 2 =
          (η ^ 2 / 2) * ∑ i : Fin p, S i * (realSign (g i)) ^ 2 := by
      rw [Finset.mul_sum, Finset.mul_sum]
      refine Finset.sum_congr rfl fun i _ => ?_
      ring
    rw [hfactor]
    refine mul_le_mul_of_nonneg_left ?_ (by positivity)
    refine Finset.sum_le_sum fun i _ => ?_
    have h1 := mul_le_mul_of_nonneg_left (realSign_sq_le_one (g i)) (hS i)
    linarith
  linarith

/-- SignGD descent derived directly from the general `preconditionedStep` anisotropic sharpness
assumption via `signStep_eq_preconditionedStep`. -/
theorem signGD_descent_from_anisotropic (L : Parameters p → ℝ) (u g : Parameters p)
    (η : ℝ) (S : Fin p → ℝ) (hS : ∀ i, 0 ≤ S i)
    (hquad : L (preconditionedStep (signStepVec η g) u g) ≤
      L u - ∑ i : Fin p, signStepVec η g i * (g i) ^ 2 +
        (1 / 2) * ∑ i : Fin p, S i * (signStepVec η g i) ^ 2 * (g i) ^ 2) :
    L (signStep η u g) ≤ L u - η * l1Norm g + (η ^ 2 / 2) * sharpnessL1 S := by
  rw [← signStep_eq_preconditionedStep] at hquad
  apply signGD_descent L u g η S hS
  have hlin :
      (∑ i : Fin p, signStepVec η g i * (g i) ^ 2) =
        ∑ i : Fin p, η * realSign (g i) * g i := by
    refine Finset.sum_congr rfl fun i _ => ?_
    calc
      signStepVec η g i * (g i) ^ 2
        = (signStepVec η g i * g i) * g i := by ring
      _ = η * realSign (g i) * g i := by rw [signStepVec_mul_grad]
  have hcurv :
      (∑ i : Fin p, S i * (signStepVec η g i) ^ 2 * (g i) ^ 2) =
        ∑ i : Fin p, S i * (η * realSign (g i)) ^ 2 := by
    refine Finset.sum_congr rfl fun i _ => ?_
    calc
      S i * (signStepVec η g i) ^ 2 * (g i) ^ 2
        = S i * (signStepVec η g i * g i) ^ 2 := by ring
      _ = S i * (η * realSign (g i)) ^ 2 := by rw [signStepVec_mul_grad]
  linarith

/-- Optimal SignGD learning rate `l1Norm g / sharpnessL1 S`. -/
def optimalSignLR (S : Fin p → ℝ) (g : Parameters p) : ℝ :=
  l1Norm g / sharpnessL1 S

/-- **Optimal SignGD One-Step Loss Drop (`signGD_optimal_lr_drop`)**:
When `0 < sharpnessL1 S` and `∀ i, 0 ≤ S i`, choosing `η = optimalSignLR S g` guarantees a
one-step loss reduction of at least `(l1Norm g) ^ 2 / (2 * sharpnessL1 S)`. -/
theorem signGD_optimal_lr_drop (L : Parameters p → ℝ) (u g : Parameters p)
    (S : Fin p → ℝ) (hS : ∀ i, 0 ≤ S i) (hS_sum : 0 < sharpnessL1 S)
    (hquad : L (signStep (optimalSignLR S g) u g) ≤
      L u - ∑ i : Fin p, optimalSignLR S g * realSign (g i) * g i +
        (1 / 2) * ∑ i : Fin p, S i * (optimalSignLR S g * realSign (g i)) ^ 2) :
    L (signStep (optimalSignLR S g) u g) ≤
      L u - (l1Norm g) ^ 2 / (2 * sharpnessL1 S) := by
  have hdesc := signGD_descent L u g (optimalSignLR S g) S hS hquad
  have hne : sharpnessL1 S ≠ 0 := ne_of_gt hS_sum
  have hid :
      L u - optimalSignLR S g * l1Norm g + (optimalSignLR S g ^ 2 / 2) * sharpnessL1 S =
        L u - (l1Norm g) ^ 2 / (2 * sharpnessL1 S) := by
    unfold optimalSignLR
    field_simp [hne]
    ring
  linarith

/-- **SignGD Strictly Beats Conservative Scalar SGD on Dense Gradients
(`signGD_strictly_beats_scalar_sgd`)**:
Whenever the $\ell_1$-to-$\ell_2$ gradient density ratio exceeds the $\ell_1$-to-$\ell_\infty$
sharpness ratio so that `‖g‖ ^ 2 / S_max < (l1Norm g) ^ 2 / sharpnessL1 S`, the guaranteed one-step
loss reduction of SignGD is strictly larger than that of conservative scalar SGD. -/
theorem signGD_strictly_beats_scalar_sgd (S : Fin p → ℝ) (S_max : ℝ) (g : Parameters p)
    (h_dense : ‖g‖ ^ 2 / S_max < (l1Norm g) ^ 2 / sharpnessL1 S) :
    scalarMaxProgress S_max g < (l1Norm g) ^ 2 / (2 * sharpnessL1 S) := by
  rw [scalar_progress_eq]
  have h1 : ‖g‖ ^ 2 / (2 * S_max) = (‖g‖ ^ 2 / S_max) / 2 := by ring
  have h2 : (l1Norm g) ^ 2 / (2 * sharpnessL1 S) = ((l1Norm g) ^ 2 / sharpnessL1 S) / 2 := by ring
  linarith

/-! ## 4. Preconditioned Polyak-Łojasiewicz (PL) Geometric Convergence -/

/-- **One-Step Preconditioned PL Contraction (`preconditioned_pl_one_step_contraction`)**:
If `L` satisfies the coordinate-wise sharpness bound at `u` with sharpness `S i > 0` and the
preconditioned gradient norm satisfies the PL inequality
`2 * μ_P * (L u - floor) ≤ ∑ i : Fin p, (g i) ^ 2 / S i`, then one step of optimal diagonal
preconditioned descent contracts the optimality gap by `1 - μ_P`. -/
theorem preconditioned_pl_one_step_contraction (L : Parameters p → ℝ) (u g : Parameters p)
    (S : Fin p → ℝ) (floor μ_P : ℝ)
    (hS : ∀ i, 0 < S i)
    (hquad : L (preconditionedStep (optimalDiagStep S) u g) ≤
      L u - ∑ i : Fin p, optimalDiagStep S i * (g i) ^ 2 +
        (1 / 2) * ∑ i : Fin p, S i * (optimalDiagStep S i) ^ 2 * (g i) ^ 2)
    (hPL : 2 * μ_P * (L u - floor) ≤ ∑ i : Fin p, (g i) ^ 2 / S i) :
    L (preconditionedStep (optimalDiagStep S) u g) - floor ≤ (1 - μ_P) * (L u - floor) := by
  have hdrop := optimal_diagonal_descent_drop L u g S hS hquad
  have hsum_half :
      (∑ i : Fin p, (g i) ^ 2 / (2 * S i)) =
        (1 / 2) * ∑ i : Fin p, (g i) ^ 2 / S i := by
    rw [Finset.mul_sum]
    refine Finset.sum_congr rfl fun i _ => ?_
    ring
  linarith

/-- **Preconditioned PL Geometric Convergence (`preconditioned_pl_geometric_convergence`)**:
Along any trajectory `u : ℕ → Parameters p` updated by optimal diagonal preconditioned steps
`u (t + 1) = preconditionedStep (optimalDiagStep S) (u t) (g t)` where each step satisfies the
coordinate-wise sharpness bound (`∀ i, 0 < S i`) and the preconditioned PL inequality
`2 * μ_P * (L (u t) - floor) ≤ ∑ i : Fin p, (g t i) ^ 2 / S i` with `0 < μ_P ≤ 1`, the optimality
gap converges geometrically:
`L (u T) - floor ≤ (1 - μ_P) ^ T * (L (u 0) - floor)`. -/
theorem preconditioned_pl_geometric_convergence (L : Parameters p → ℝ)
    (u g : ℕ → Parameters p) (S : Fin p → ℝ) (floor μ_P : ℝ)
    (hS : ∀ i, 0 < S i)
    (_hμ_pos : 0 < μ_P) (hμ_le : μ_P ≤ 1)
    (hstep : ∀ t, u (t + 1) = preconditionedStep (optimalDiagStep S) (u t) (g t))
    (hquad : ∀ t, L (preconditionedStep (optimalDiagStep S) (u t) (g t)) ≤
      L (u t) - ∑ i : Fin p, optimalDiagStep S i * (g t i) ^ 2 +
        (1 / 2) * ∑ i : Fin p, S i * (optimalDiagStep S i) ^ 2 * (g t i) ^ 2)
    (hPL : ∀ t, 2 * μ_P * (L (u t) - floor) ≤ ∑ i : Fin p, (g t i) ^ 2 / S i)
    (T : ℕ) :
    L (u T) - floor ≤ (1 - μ_P) ^ T * (L (u 0) - floor) := by
  have hrate_nonneg : 0 ≤ 1 - μ_P := sub_nonneg.mpr hμ_le
  induction T with
  | zero =>
    simp
  | succ t ih =>
    have hone := preconditioned_pl_one_step_contraction L (u t) (g t) S floor μ_P
      hS (hquad t) (hPL t)
    rw [← hstep t] at hone
    calc
      L (u (t + 1)) - floor
        ≤ (1 - μ_P) * (L (u t) - floor) := hone
      _ ≤ (1 - μ_P) * ((1 - μ_P) ^ t * (L (u 0) - floor)) :=
        mul_le_mul_of_nonneg_left ih hrate_nonneg
      _ = (1 - μ_P) ^ (t + 1) * (L (u 0) - floor) := by
        rw [pow_succ', mul_assoc]

end OpenTheory.PreconditionedDescent
