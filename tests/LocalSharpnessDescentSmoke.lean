import OpenTheory.Optimization.Convergence
import OpenTheory.Optimization.LocalSharpnessDescent

open OpenTheory

#check risk_exponential_approach
#check LocalSharpnessDescent.stepProgress
#check LocalSharpnessDescent.adaptiveStep
#check LocalSharpnessDescent.local_sharpness_descent
#check LocalSharpnessDescent.adaptive_step_progress_eq
#check LocalSharpnessDescent.stepProgress_sub_eq_sq
#check LocalSharpnessDescent.adaptive_step_maximizes_progress
#check LocalSharpnessDescent.adaptive_step_optimal_drop
#check LocalSharpnessDescent.adaptive_vs_global_sharpness_speedup
#check LocalSharpnessDescent.adaptive_step_strictly_beats_conservative
#check LocalSharpnessDescent.batchRegularizedObjective
#check LocalSharpnessDescent.batchRegularizedObjective_zero
#check LocalSharpnessDescent.batchRegularizedObjective_zero_le
#check LocalSharpnessDescent.regularized_loss_coercive_residual_le
#check LocalSharpnessDescent.regularized_loss_coercive_norm_le
#check LocalSharpnessDescent.regularized_loss_coercive_shift_le
#check LocalSharpnessDescent.descent_trajectory_horizon_free_norm_le
#check LocalSharpnessDescent.descent_trajectory_shift_and_residual_le
#check LocalSharpnessDescent.descent_trajectory_horizon_free_bounds
#check LocalSharpnessDescent.pl_adaptive_one_step_contraction
#check LocalSharpnessDescent.pl_trajectory_geometric_convergence

-- Smoke test 1: When local sharpness is 10x smaller than global sharpness (S_loc = 2, S_global = 20),
-- the adaptive step achieves 10x larger guaranteed progress per step.
example (G : ℝ) :
    LocalSharpnessDescent.stepProgress (LocalSharpnessDescent.adaptiveStep 2) 2 G =
      10 * LocalSharpnessDescent.stepProgress (LocalSharpnessDescent.adaptiveStep 20) 20 G := by
  have h := LocalSharpnessDescent.adaptive_vs_global_sharpness_speedup 2 20 G
    (by norm_num) (by norm_num)
  linarith

-- Smoke test 2: When μ = S_max, a single adaptive PL step reaches the floor in 1 step.
example {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    (L : E → ℝ) (u : ℕ → E) (floor μ : ℝ) (hμ : 0 < μ)
    (hfloor : ∀ t, floor ≤ L (u t))
    (hstep : ∀ t, L (u (t + 1)) - floor ≤ (1 - μ / μ) * (L (u t) - floor)) :
    L (u 1) ≤ floor := by
  have h := LocalSharpnessDescent.pl_trajectory_geometric_convergence
    L u (fun _ => μ) floor μ μ hμ le_rfl hfloor (fun _ => hμ) (fun _ => le_rfl) hstep 1
  rw [div_self (ne_of_gt hμ), sub_self, pow_one, zero_mul] at h
  linarith
