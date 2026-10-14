import OpenTheory.Dynamics.MLPBatchTrajectory
import OpenTheory.Optimization.Optimizers

/-!
# Local-Sharpness Adaptive Step Sizes, Objective Coercivity, and Discrete PL Convergence

Existing fixed-step trajectory bounds (`MLPBatchTrajectory`, `MLPContractive`) either accumulate
exponential growth `batchGrowth ^ T` or require a strong global weight-decay dominance condition
(`nonlinearRate < 1`, i.e., `m √m A^2 L^2 E < lam`) to keep the trajectory bounded.

This module establishes three foundational optimization upgrades that eliminate those bottlenecks:

1. **Local-Sharpness Descent & Optimal Adaptive Step Size (`local_sharpness_descent`,
   `adaptive_step_progress_eq`, `adaptive_step_maximizes_progress`,
   `adaptive_vs_global_sharpness_speedup`)**:
   Under a local quadratic upper bound with local sharpness `S_loc > 0` along the step `u - η • g`,
   the guaranteed one-step loss reduction is `stepProgress η S_loc (‖g‖^2) = η (1 - η S_loc / 2) ‖g‖^2`.
   The adaptive step size `adaptiveStep S_loc = 1 / S_loc` uniquely maximizes this quadratic model
   reduction across **all** step sizes `η : ℝ`, achieving `‖g‖^2 / (2 S_loc)` and outperforming any
   conservative global-sharpness step `1 / S_global` (`S_loc < S_global`) by the exact factor
   `S_global / S_loc > 1`.

2. **Objective-Descent Coercivity Invariant (`batchRegularizedObjective`,
   `regularized_loss_coercive_residual_le`, `regularized_loss_coercive_norm_le`,
   `regularized_loss_coercive_shift_le`, `descent_trajectory_horizon_free_bounds`)**:
   Whenever an algorithm (such as local-sharpness adaptive GD or line-search GD) monotonically
   decreases the regularized nonlinear batch objective
   `L(u) = (1 / (2n)) ‖F(u) - y‖^2 + (lam / 2) ‖u‖^2 ≤ L(0) ≤ S^2 / (2n)`,
   positivity of the two terms immediately forces:
   - `‖batchOutput net X u - y‖ ≤ S` (no `√n m A L` expansion factor!),
   - `‖u‖ ≤ S / √(n * lam)` (horizon-free, requiring **no** contraction condition `q_nl < 1`!),
   - `|preShift (unflatten u) (X r) i| ≤ (S / √(n * lam)) * √E`.

3. **Discrete Polyak-Łojasiewicz (PL) Geometric Convergence (`pl_adaptive_one_step_contraction`,
   `pl_trajectory_geometric_convergence`)**:
   Combining local-sharpness adaptive descent with a local PL inequality
   `‖g_t‖^2 ≥ 2 μ (L(u_t) - floor)` yields one-step contraction `(1 - μ / S_loc(t))` and
   uniform geometric convergence `(1 - μ / S_max)^T (L(u_0) - floor)`.
-/

noncomputable section

open scoped BigOperators RealInnerProductSpace

namespace OpenTheory.LocalSharpnessDescent

open OpenTheory.FeatureStep
open OpenTheory.MLPTrajectory
open OpenTheory.MLPFixedDesign
open OpenTheory.MLPBatchTrajectory

variable {d m n p : ℕ}

/-! ## 1. Local-Sharpness Descent Lemma and Optimal Adaptive Step Size -/

/-- Guaranteed one-step objective progress `η * (1 - η * S_loc / 2) * G` under local sharpness
`S_loc` and squared gradient norm `G = ‖g‖^2`. -/
def stepProgress (η S_loc G : ℝ) : ℝ :=
  η * (1 - η * S_loc / 2) * G

/-- Optimal adaptive step size `1 / S_loc` for local sharpness `S_loc`. -/
def adaptiveStep (S_loc : ℝ) : ℝ :=
  1 / S_loc

/-- **Local-Sharpness Descent Lemma (`local_sharpness_descent`)**:
If `L` satisfies the local quadratic upper bound with sharpness `S_loc` at `u` along `g`, then
taking step `η` reduces `L` by at least `stepProgress η S_loc (‖g‖ ^ 2)`. -/
theorem local_sharpness_descent {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    (L : E → ℝ) (u g : E) (η S_loc : ℝ)
    (hquad : L (u - η • g) ≤ L u - η * ‖g‖ ^ 2 + (S_loc / 2) * η ^ 2 * ‖g‖ ^ 2) :
    L (u - η • g) ≤ L u - stepProgress η S_loc (‖g‖ ^ 2) := by
  have hid : L u - η * ‖g‖ ^ 2 + (S_loc / 2) * η ^ 2 * ‖g‖ ^ 2 =
      L u - stepProgress η S_loc (‖g‖ ^ 2) := by
    unfold stepProgress
    ring
  linarith

/-- At the adaptive step `η = 1 / S_loc` (`S_loc ≠ 0`), the guaranteed one-step progress is
`G / (2 * S_loc)`. -/
theorem adaptive_step_progress_eq (S_loc G : ℝ) (hS : S_loc ≠ 0) :
    stepProgress (adaptiveStep S_loc) S_loc G = G / (2 * S_loc) := by
  unfold stepProgress adaptiveStep
  field_simp [hS]
  ring

/-- **Completion-of-Squares Identity for Step Progress (`stepProgress_sub_eq_sq`)**:
The gap between the progress at `adaptiveStep S_loc` and any step size `η : ℝ` is exactly
`(S_loc / 2) * (η - 1 / S_loc) ^ 2 * G`. -/
theorem stepProgress_sub_eq_sq (η S_loc G : ℝ) (hS : S_loc ≠ 0) :
    stepProgress (adaptiveStep S_loc) S_loc G - stepProgress η S_loc G =
      (S_loc / 2) * (η - adaptiveStep S_loc) ^ 2 * G := by
  unfold stepProgress adaptiveStep
  field_simp [hS]
  ring

/-- **Optimality of the Local-Sharpness Adaptive Step (`adaptive_step_maximizes_progress`)**:
For any `S_loc > 0`, `G ≥ 0`, and any step size `η : ℝ`, the adaptive step `1 / S_loc`
achieves at least as much guaranteed loss reduction as `η`. -/
theorem adaptive_step_maximizes_progress (η S_loc G : ℝ) (hS : 0 < S_loc) (hG : 0 ≤ G) :
    stepProgress η S_loc G ≤ stepProgress (adaptiveStep S_loc) S_loc G := by
  have hdiff := stepProgress_sub_eq_sq η S_loc G (ne_of_gt hS)
  have hnonneg : 0 ≤ (S_loc / 2) * (η - adaptiveStep S_loc) ^ 2 * G := by positivity
  linarith

/-- Combining `local_sharpness_descent` with `adaptive_step_progress_eq` gives the optimal
one-step adaptive loss drop `L (u - adaptiveStep S_loc • g) ≤ L u - ‖g‖ ^ 2 / (2 * S_loc)`. -/
theorem adaptive_step_optimal_drop {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    (L : E → ℝ) (u g : E) (S_loc : ℝ) (hS : S_loc ≠ 0)
    (hquad : L (u - adaptiveStep S_loc • g) ≤
      L u - adaptiveStep S_loc * ‖g‖ ^ 2 + (S_loc / 2) * adaptiveStep S_loc ^ 2 * ‖g‖ ^ 2) :
    L (u - adaptiveStep S_loc • g) ≤ L u - ‖g‖ ^ 2 / (2 * S_loc) := by
  have hdesc := local_sharpness_descent L u g (adaptiveStep S_loc) S_loc hquad
  rwa [adaptive_step_progress_eq S_loc (‖g‖ ^ 2) hS] at hdesc

/-- **Exact Speedup Ratio Over Global Conservative Step Size (`adaptive_vs_global_sharpness_speedup`)**:
Using local sharpness `S_loc` instead of a global worst-case sharpness bound `S_global`
scales the guaranteed one-step progress by the exact ratio `S_global / S_loc`. -/
theorem adaptive_vs_global_sharpness_speedup (S_loc S_global G : ℝ)
    (hS_loc : S_loc ≠ 0) (hS_glob : S_global ≠ 0) :
    stepProgress (adaptiveStep S_loc) S_loc G =
      (S_global / S_loc) * stepProgress (adaptiveStep S_global) S_global G := by
  rw [adaptive_step_progress_eq S_loc G hS_loc, adaptive_step_progress_eq S_global G hS_glob]
  field_simp [hS_loc, hS_glob]

/-- When `0 < S_loc < S_global` and `0 < G`, the local-sharpness adaptive step strictly beats
the conservative global step `1 / S_global`. -/
theorem adaptive_step_strictly_beats_conservative (S_loc S_global G : ℝ)
    (hS_loc : 0 < S_loc) (h_lt : S_loc < S_global) (hG : 0 < G) :
    stepProgress (adaptiveStep S_global) S_global G <
      stepProgress (adaptiveStep S_loc) S_loc G := by
  have hS_glob : 0 < S_global := hS_loc.trans h_lt
  rw [adaptive_step_progress_eq S_loc G (ne_of_gt hS_loc),
    adaptive_step_progress_eq S_global G (ne_of_gt hS_glob)]
  exact div_lt_div_of_pos_left hG (by positivity) (by linarith)

/-! ## 2. Objective-Descent Coercivity Invariant (No Weight-Decay Contraction Required) -/

/-- The regularized nonlinear batch training objective on flattened first-layer displacements
`u : Parameters (m * d)`:
`L(u) = (1 / (2n)) ‖batchOutput net X u - y‖^2 + (lam / 2) ‖u‖^2`. -/
def batchRegularizedObjective (net : TwoLayerNet d m) (X : Fin n → Fin d → ℝ)
    (y : Parameters n) (lam : ℝ) (u : Parameters (m * d)) : ℝ :=
  (1 / (2 * (n : ℝ))) * ‖batchOutput net X u - y‖ ^ 2 + (lam / 2) * ‖u‖ ^ 2

/-- At zero displacement `u = 0`, the regularized objective equals the initial empirical half-MSE
`(1 / (2n)) ‖initialOutput net X - y‖^2`. -/
@[simp]
theorem batchRegularizedObjective_zero (net : TwoLayerNet d m) (X : Fin n → Fin d → ℝ)
    (y : Parameters n) (lam : ℝ) :
    batchRegularizedObjective net X y lam 0 =
      (1 / (2 * (n : ℝ))) * ‖initialOutput net X - y‖ ^ 2 := by
  simp [batchRegularizedObjective]

/-- Upper bound on the initial regularized objective when `‖initialOutput net X - y‖ ≤ S`. -/
theorem batchRegularizedObjective_zero_le (net : TwoLayerNet d m) (X : Fin n → Fin d → ℝ)
    (y : Parameters n) (lam : ℝ) {S : ℝ} (hS : 0 ≤ S)
    (hinit : ‖initialOutput net X - y‖ ≤ S) :
    batchRegularizedObjective net X y lam 0 ≤ S ^ 2 / (2 * (n : ℝ)) := by
  rw [batchRegularizedObjective_zero, one_div_mul_eq_div]
  apply div_le_div_of_nonneg_right _ (by positivity)
  exact (sq_le_sq₀ (norm_nonneg _) hS).mpr hinit

/-- **Coercive Residual Bound (`regularized_loss_coercive_residual_le`)**:
For ANY parameter vector `u` whose regularized objective does not exceed the initial value
`batchRegularizedObjective net X y lam 0` (with `0 < n` and `0 ≤ lam`), the residual norm
satisfies `‖batchOutput net X u - y‖ ≤ S` directly—with NO `√n m A L` growth factor! -/
theorem regularized_loss_coercive_residual_le (net : TwoLayerNet d m) (X : Fin n → Fin d → ℝ)
    (y : Parameters n) (lam : ℝ) (u : Parameters (m * d)) {S : ℝ}
    (hn : 0 < n) (hlam : 0 ≤ lam) (hS : 0 ≤ S)
    (hinit : ‖initialOutput net X - y‖ ≤ S)
    (hdesc : batchRegularizedObjective net X y lam u ≤
      batchRegularizedObjective net X y lam 0) :
    ‖batchOutput net X u - y‖ ≤ S := by
  have hn0 : (0 : ℝ) < n := Nat.cast_pos.mpr hn
  have h2n : 0 < 2 * (n : ℝ) := by positivity
  have h0 := batchRegularizedObjective_zero_le net X y lam hS hinit
  have hu_le : batchRegularizedObjective net X y lam u ≤ S ^ 2 / (2 * (n : ℝ)) :=
    hdesc.trans h0
  have hreg_nonneg : 0 ≤ (lam / 2) * ‖u‖ ^ 2 := by positivity
  have hmse_le : (1 / (2 * (n : ℝ))) * ‖batchOutput net X u - y‖ ^ 2 ≤
      S ^ 2 / (2 * (n : ℝ)) := by
    unfold batchRegularizedObjective at hu_le
    linarith
  have hsq : ‖batchOutput net X u - y‖ ^ 2 ≤ S ^ 2 := by
    rw [one_div_mul_eq_div, div_le_div_iff_of_pos_right h2n] at hmse_le
    exact hmse_le
  exact (sq_le_sq₀ (norm_nonneg _) hS).mp hsq

/-- **Coercive Parameter-Norm Bound (`regularized_loss_coercive_norm_le`)**:
For ANY parameter vector `u` whose regularized objective does not exceed the initial value
`batchRegularizedObjective net X y lam 0` (with `0 < n` and `0 < lam`), the parameter norm
satisfies `‖u‖ ≤ S / √(n * lam)`—with NO contraction hypothesis `q_nl < 1` required! -/
theorem regularized_loss_coercive_norm_le (net : TwoLayerNet d m) (X : Fin n → Fin d → ℝ)
    (y : Parameters n) (lam : ℝ) (u : Parameters (m * d)) {S : ℝ}
    (hn : 0 < n) (hlam : 0 < lam) (hS : 0 ≤ S)
    (hinit : ‖initialOutput net X - y‖ ≤ S)
    (hdesc : batchRegularizedObjective net X y lam u ≤
      batchRegularizedObjective net X y lam 0) :
    ‖u‖ ≤ S / Real.sqrt (n * lam) := by
  have hn0 : (0 : ℝ) < n := Nat.cast_pos.mpr hn
  have hnlam : 0 < (n : ℝ) * lam := mul_pos hn0 hlam
  have h0 := batchRegularizedObjective_zero_le net X y lam hS hinit
  have hu_le : batchRegularizedObjective net X y lam u ≤ S ^ 2 / (2 * (n : ℝ)) :=
    hdesc.trans h0
  have hfit_nonneg : 0 ≤ (1 / (2 * (n : ℝ))) * ‖batchOutput net X u - y‖ ^ 2 := by positivity
  have hreg_le : (lam / 2) * ‖u‖ ^ 2 ≤ S ^ 2 / (2 * (n : ℝ)) := by
    unfold batchRegularizedObjective at hu_le
    linarith
  have hsq : ‖u‖ ^ 2 ≤ (S / Real.sqrt (n * lam)) ^ 2 := by
    rw [div_pow, Real.sq_sqrt hnlam.le]
    have hmul : (n : ℝ) * lam * ‖u‖ ^ 2 ≤ S ^ 2 := by
      have h1 : 2 * (n : ℝ) * ((lam / 2) * ‖u‖ ^ 2) ≤ 2 * (n : ℝ) * (S ^ 2 / (2 * (n : ℝ))) :=
        mul_le_mul_of_nonneg_left hreg_le (by positivity)
      have h2 : 2 * (n : ℝ) * (S ^ 2 / (2 * (n : ℝ))) = S ^ 2 :=
        mul_div_cancel₀ (S ^ 2) (by positivity)
      linarith
    rwa [le_div_iff₀ hnlam, mul_comm (‖u‖ ^ 2)]
  exact (sq_le_sq₀ (norm_nonneg _) (div_nonneg hS (Real.sqrt_nonneg _))).mp hsq

/-- **Coercive Preactivation-Shift Bound (`regularized_loss_coercive_shift_le`)**:
Under objective descent `L(u) ≤ L(0)`, every hidden preactivation shift is bounded by
`(S / √(n * lam)) * √E`. -/
theorem regularized_loss_coercive_shift_le (net : TwoLayerNet d m) (X : Fin n → Fin d → ℝ)
    (y : Parameters n) (lam : ℝ) (u : Parameters (m * d)) {E S : ℝ}
    (hn : 0 < n) (hlam : 0 < lam) (hS : 0 ≤ S)
    (hX : ∀ r, inputEnergy (X r) ≤ E)
    (hinit : ‖initialOutput net X - y‖ ≤ S)
    (hdesc : batchRegularizedObjective net X y lam u ≤
      batchRegularizedObjective net X y lam 0)
    (r : Fin n) (i : Fin m) :
    |preShift (unflatten u) (X r) i| ≤ (S / Real.sqrt (n * lam)) * Real.sqrt E := by
  have hnorm := regularized_loss_coercive_norm_le net X y lam u hn hlam hS hinit hdesc
  apply (preShift_unflatten_le_norm u (X r) i).trans
  exact mul_le_mul hnorm (Real.sqrt_le_sqrt (hX r)) (Real.sqrt_nonneg _)
    (div_nonneg hS (Real.sqrt_nonneg _))

/-- **Horizon-Free Parameter-Norm Bound Along Any Monotone Loss-Descent Trajectory
(`descent_trajectory_horizon_free_norm_le`)**:
Along any trajectory `u : ℕ → Parameters (m * d)` starting at `u 0 = 0` that monotonically
decreases `batchRegularizedObjective`, `‖u t‖ ≤ S / √(n * lam)` for all `t : ℕ` without
requiring `q_nl < 1`. -/
theorem descent_trajectory_horizon_free_norm_le (net : TwoLayerNet d m) (X : Fin n → Fin d → ℝ)
    (y : Parameters n) (lam : ℝ) (u : ℕ → Parameters (m * d)) {S : ℝ}
    (hn : 0 < n) (hlam : 0 < lam) (hS : 0 ≤ S)
    (hinit : ‖initialOutput net X - y‖ ≤ S)
    (hu0 : u 0 = 0)
    (hmono : ∀ t, batchRegularizedObjective net X y lam (u (t + 1)) ≤
      batchRegularizedObjective net X y lam (u t))
    (t : ℕ) :
    ‖u t‖ ≤ S / Real.sqrt (n * lam) := by
  have hdesc : batchRegularizedObjective net X y lam (u t) ≤
      batchRegularizedObjective net X y lam 0 := by
    rw [← hu0]
    induction t with
    | zero => exact le_rfl
    | succ k ih => exact (hmono k).trans ih
  exact regularized_loss_coercive_norm_le net X y lam (u t) hn hlam hS hinit hdesc

/-- **Horizon-Free Shift and Residual Bounds Along Any Monotone Loss-Descent Trajectory
(`descent_trajectory_shift_and_residual_le`)**:
Along any trajectory `u : ℕ → Parameters (m * d)` starting at `u 0 = 0` that monotonically
decreases `batchRegularizedObjective`, the residual is bounded by `S` and every preactivation
shift is bounded by `(S / √(n * lam)) * √E` for all `t : ℕ`. -/
theorem descent_trajectory_shift_and_residual_le (net : TwoLayerNet d m) (X : Fin n → Fin d → ℝ)
    (y : Parameters n) (lam : ℝ) (u : ℕ → Parameters (m * d)) {E S : ℝ}
    (hn : 0 < n) (hlam : 0 < lam) (hS : 0 ≤ S)
    (hX : ∀ r, inputEnergy (X r) ≤ E)
    (hinit : ‖initialOutput net X - y‖ ≤ S)
    (hu0 : u 0 = 0)
    (hmono : ∀ t, batchRegularizedObjective net X y lam (u (t + 1)) ≤
      batchRegularizedObjective net X y lam (u t))
    (t : ℕ) :
    ‖batchOutput net X (u t) - y‖ ≤ S ∧
      ∀ r i, |preShift (unflatten (u t)) (X r) i| ≤ (S / Real.sqrt (n * lam)) * Real.sqrt E := by
  have hdesc : batchRegularizedObjective net X y lam (u t) ≤
      batchRegularizedObjective net X y lam 0 := by
    rw [← hu0]
    induction t with
    | zero => exact le_rfl
    | succ k ih => exact (hmono k).trans ih
  exact ⟨regularized_loss_coercive_residual_le net X y lam (u t) hn hlam.le hS hinit hdesc,
    regularized_loss_coercive_shift_le net X y lam (u t) hn hlam hS hX hinit hdesc⟩

/-- **Horizon-Free Trajectory Bounds for Any Monotone Loss-Descent Trajectory
(`descent_trajectory_horizon_free_bounds`)**:
Let `u : ℕ → Parameters (m * d)` be ANY training trajectory initialized at `u 0 = 0` that is
monotone non-increasing on `batchRegularizedObjective` (`L(u (t + 1)) ≤ L(u t)`). Then for ALL
steps `t : ℕ`, with **no weight-decay contraction condition (`q_nl < 1`)**, the trajectory
simultaneously satisfies the horizon-free norm, residual, and shift bounds. -/
theorem descent_trajectory_horizon_free_bounds (net : TwoLayerNet d m) (X : Fin n → Fin d → ℝ)
    (y : Parameters n) (lam : ℝ) (u : ℕ → Parameters (m * d)) {E S : ℝ}
    (hn : 0 < n) (hlam : 0 < lam) (hS : 0 ≤ S)
    (hX : ∀ r, inputEnergy (X r) ≤ E)
    (hinit : ‖initialOutput net X - y‖ ≤ S)
    (hu0 : u 0 = 0)
    (hmono : ∀ t, batchRegularizedObjective net X y lam (u (t + 1)) ≤
      batchRegularizedObjective net X y lam (u t))
    (t : ℕ) :
    ‖u t‖ ≤ S / Real.sqrt (n * lam) ∧
      ‖batchOutput net X (u t) - y‖ ≤ S ∧
      ∀ r i, |preShift (unflatten (u t)) (X r) i| ≤ (S / Real.sqrt (n * lam)) * Real.sqrt E :=
  ⟨descent_trajectory_horizon_free_norm_le net X y lam u hn hlam hS hinit hu0 hmono t,
    descent_trajectory_shift_and_residual_le net X y lam u hn hlam hS hX hinit hu0 hmono t⟩

/-! ## 3. Discrete Polyak-Łojasiewicz (PL) Geometric Convergence with Local Adaptive Steps -/

/-- **One-Step PL Contraction with Local Adaptive Step Size (`pl_adaptive_one_step_contraction`)**:
If at step `u`, the objective satisfies the local quadratic upper bound with sharpness `S_loc > 0`
and the local PL inequality `‖g‖ ^ 2 ≥ 2 * μ * (L u - floor)`, then the adaptive step
`u_next = u - adaptiveStep S_loc • g` contracts the optimality gap by `1 - μ / S_loc`. -/
theorem pl_adaptive_one_step_contraction {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    (L : E → ℝ) (u g : E) (floor μ S_loc : ℝ)
    (hS : 0 < S_loc)
    (hquad : L (u - adaptiveStep S_loc • g) ≤
      L u - adaptiveStep S_loc * ‖g‖ ^ 2 + (S_loc / 2) * adaptiveStep S_loc ^ 2 * ‖g‖ ^ 2)
    (hPL : 2 * μ * (L u - floor) ≤ ‖g‖ ^ 2) :
    L (u - adaptiveStep S_loc • g) - floor ≤ (1 - μ / S_loc) * (L u - floor) := by
  have hdrop := adaptive_step_optimal_drop L u g S_loc (ne_of_gt hS) hquad
  have hdiv : (2 * μ * (L u - floor)) / (2 * S_loc) ≤ ‖g‖ ^ 2 / (2 * S_loc) :=
    div_le_div_of_nonneg_right hPL (by positivity)
  have hsimp : (2 * μ * (L u - floor)) / (2 * S_loc) = (μ / S_loc) * (L u - floor) := by
    field_simp [ne_of_gt hS]
  rw [hsimp] at hdiv
  linarith

/-- **Discrete PL Geometric Convergence Along an Adaptive Trajectory
(`pl_trajectory_geometric_convergence`)**:
Along any trajectory `u : ℕ → Parameters p` where each step satisfies local-sharpness adaptive
contraction with local sharpness `0 < S_loc t ≤ S_max` and PL constant `0 < μ ≤ S_max`, the
optimality gap `L (u T) - floor` decays geometrically at rate `(1 - μ / S_max) ^ T`. -/
theorem pl_trajectory_geometric_convergence {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    (L : E → ℝ) (u : ℕ → E) (S_loc : ℕ → ℝ) (floor μ S_max : ℝ)
    (hμ : 0 < μ) (hμ_le : μ ≤ S_max)
    (hfloor : ∀ t, floor ≤ L (u t))
    (hS_pos : ∀ t, 0 < S_loc t)
    (hS_le : ∀ t, S_loc t ≤ S_max)
    (hstep : ∀ t, L (u (t + 1)) - floor ≤ (1 - μ / S_loc t) * (L (u t) - floor))
    (T : ℕ) :
    L (u T) - floor ≤ (1 - μ / S_max) ^ T * (L (u 0) - floor) := by
  have hS_max_pos : 0 < S_max := hμ.trans_le hμ_le
  have hrate_nonneg : 0 ≤ 1 - μ / S_max := by
    have hdiv_le : μ / S_max ≤ 1 := (div_le_one hS_max_pos).mpr hμ_le
    linarith
  induction T with
  | zero =>
    simp
  | succ t ih =>
    have h_gap_nonneg : 0 ≤ L (u t) - floor := sub_nonneg.mpr (hfloor t)
    have h_ratio : μ / S_max ≤ μ / S_loc t :=
      div_le_div_of_nonneg_left hμ.le (hS_pos t) (hS_le t)
    have h_factor : 1 - μ / S_loc t ≤ 1 - μ / S_max := by linarith
    calc
      L (u (t + 1)) - floor
        ≤ (1 - μ / S_loc t) * (L (u t) - floor) := hstep t
      _ ≤ (1 - μ / S_max) * (L (u t) - floor) :=
        mul_le_mul_of_nonneg_right h_factor h_gap_nonneg
      _ ≤ (1 - μ / S_max) * ((1 - μ / S_max) ^ t * (L (u 0) - floor)) :=
        mul_le_mul_of_nonneg_left ih hrate_nonneg
      _ = (1 - μ / S_max) ^ (t + 1) * (L (u 0) - floor) := by
        rw [pow_succ', mul_assoc]

end OpenTheory.LocalSharpnessDescent
