import OpenTheory.Optimization.PreconditionedDescent

open OpenTheory
open OpenTheory.PreconditionedDescent

#check preconditionedStep
#check preconditionedStep_apply
#check coordProgress
#check optimalDiagStep
#check optimalDiagStep_apply
#check anisotropic_sharpness_descent
#check optimal_diagonal_progress_eq
#check optimal_diagonal_descent_drop
#check coordProgress_sub_eq_sum_sq
#check optimal_diagonal_maximizes_progress
#check scalarMaxProgress
#check scalar_progress_eq
#check coordProgress_scalar_max_eq
#check diagonal_vs_scalar_max_gap_eq
#check diagonal_vs_scalar_gap_eq
#check diagonal_dominates_scalar_max
#check diagonal_dominates_scalar_sgd
#check diagonal_strictly_beats_scalar_max
#check diagonal_strictly_beats_scalar_sgd
#check diagonal_dominates_any_scalar_step
#check realSign
#check realSign_mul_self
#check self_mul_realSign
#check realSign_sq_le_one
#check realSign_sq_nonneg
#check l1Norm
#check l1Norm_nonneg
#check sharpnessL1
#check signStep
#check signStep_apply
#check signStepVec
#check signStepVec_mul_grad
#check signStep_eq_preconditionedStep
#check signGD_progress_eq
#check signGD_descent
#check signGD_descent_from_anisotropic
#check optimalSignLR
#check signGD_optimal_lr_drop
#check signGD_strictly_beats_scalar_sgd
#check preconditioned_pl_one_step_contraction
#check preconditioned_pl_geometric_convergence

-- Smoke test 1: realSign basic identities
example : realSign 5 = 1 ∧ realSign (-3) = -1 ∧ realSign 0 = 0 := by
  unfold realSign
  norm_num

-- Smoke test 2: Under positive sharpness, optimalDiagStep maximizes progress over any step vector η
example {p : ℕ} (η S : Fin p → ℝ) (g : Parameters p) (hS : ∀ i, 0 < S i) :
    coordProgress η S g ≤ ∑ i : Fin p, (g i) ^ 2 / (2 * S i) := by
  rw [← optimal_diagonal_progress_eq S g hS]
  exact optimal_diagonal_maximizes_progress η S g hS

-- Smoke test 3: When μ_P = 1, optimal diagonal preconditioned PL descent reaches the floor in 1 step
example {p : ℕ} (L : Parameters p → ℝ) (u g : ℕ → Parameters p)
    (S : Fin p → ℝ) (floor : ℝ) (hS : ∀ i, 0 < S i)
    (hstep : ∀ t, u (t + 1) = preconditionedStep (optimalDiagStep S) (u t) (g t))
    (hquad : ∀ t, L (preconditionedStep (optimalDiagStep S) (u t) (g t)) ≤
      L (u t) - ∑ i : Fin p, optimalDiagStep S i * (g t i) ^ 2 +
        (1 / 2) * ∑ i : Fin p, S i * (optimalDiagStep S i) ^ 2 * (g t i) ^ 2)
    (hPL : ∀ t, 2 * 1 * (L (u t) - floor) ≤ ∑ i : Fin p, (g t i) ^ 2 / S i) :
    L (u 1) ≤ floor := by
  have h := preconditioned_pl_geometric_convergence L u g S floor 1
    hS (by norm_num) le_rfl hstep hquad hPL 1
  norm_num at h
  linarith
