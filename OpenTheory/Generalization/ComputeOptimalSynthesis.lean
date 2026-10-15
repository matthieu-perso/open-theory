import OpenTheory.EndToEnd
import OpenTheory.Optimization.StochasticSharpnessSGD
import OpenTheory.Optimization.MomentumAdamTrajectory
import Mathlib.Analysis.Real.Sqrt
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Ring

/-!
# Systematic (Data, Architecture, Optimizer, Compute) Co-Design & Compute-Optimal Synthesis Compiler

This module unifies architectural complexity (`A`), optimizer/sample complexity (`B_opt`),
mini-batch wall-clock vs. FLOP efficiency (`B_crit`), and compute budget (`C = N * D`) into a
complete, zero-`sorry` mathematical co-design and compute-optimal synthesis framework:

1. **Exact Variational Chinchilla / Compute-Optimal Allocation (`N*(C), D*(C)`) via Completion of Squares
   (`scalingRisk`, `optimalModelCapacity`, `optimalDataBudget`, `paretoComputeRisk`,
   `optimalCapacity_mul_optimalData_eq_compute`, `scalingRisk_at_optimal_eq_pareto`,
   `chinchilla_completion_of_squares`, `chinchilla_variational_global_optimality`)**:
   Under the parametric scaling law `scalingRisk R_inf A B_opt N D = R_inf + A / N + B_opt / D`
   subject to compute budget `N * D = C` (`C > 0`), completing the square proves the exact
   variational identity:
   `scalingRisk R_inf A B_opt N D - paretoComputeRisk R_inf A B_opt C = (√(A * D) - √(B_opt * N))^2 / C`,
   establishing that `(optimalModelCapacity A B_opt C, optimalDataBudget A B_opt C)` is the
   global minimizer achieving `paretoComputeRisk R_inf A B_opt C = R_inf + 2 * √(A * B_opt) / √C`.

2. **How Architecture and Optimizer Improvements Shift the Compute-Optimal Allocation & Risk Frontier
   (`paretoComputeRisk_mono_in_arch_and_opt`, `paretoComputeRisk_strict_mono_in_arch_and_opt`,
   `optimal_capacity_to_data_ratio_shift`, `optimalAllocation_directional_shifts`)**:
   Proves that reducing either the architectural complexity scale (`A₁ ≤ A₂`) or the optimizer
   sample complexity scale (`B₁ ≤ B₂`) with at least one strict inequality strictly lowers the
   Pareto compute risk frontier for every compute budget `C > 0`, while shifting the optimal
   parameter-to-token ratio `N*(C) / D*(C) = A / B_opt` toward more training tokens when the
   architecture is compressed (`A₁ < A₂`) and toward larger model capacity when the optimizer
   is improved (`B₁ < B₂`).

3. **Mini-Batch Compute-vs-Wall-Clock Pareto Frontier & Critical Batch Size Optimality
   (`stepProgress`, `flopEfficiency`, `stepProgress_eq_optimal_sgd`, `flopEfficiency_eq_div`,
   `stepProgress_strict_mono_in_batch`, `flopEfficiency_strict_anti_in_batch`,
   `balancedBatchObjective`, `balancedBatchObjective_at_critical`,
   `critical_batch_variational_optimality`, `critical_batch_global_maximizer`,
   `momentum_reduces_critical_batch_size`)**:
   Formalizes the fundamental wall-clock vs. FLOP trade-off of mini-batch training: increasing
   batch size `B` strictly increases per-step progress `stepProgress` but strictly decreases
   per-sample FLOP efficiency `flopEfficiency`. Completing the square on the balanced
   throughput-FLOP efficiency objective `balancedBatchObjective` proves that `B = B_crit` is
   its unique global maximizer, and proves that EMA momentum `0 < β < 1` strictly reduces the
   effective critical batch size by the factor `(1 - β) / (1 + β) < 1`.

4. **Master Co-Design Dominance Theorem (`master_codesign_compute_dominance`,
   `equivalentComputeBudget`, `codesign_compute_saving_factor`)**:
   Combines the 4-tier architectural hierarchy `0 < p_SESC < p_Subspace < p_CNN < p_MLP`
   (`A_i = c_arch * p_i`) with the 3-tier optimizer hierarchy `0 < B_RPM < B_Adam < B_SGD`
   to establish a strict 4-step Pareto risk dominance chain across all compute budgets `C > 0`,
   and proves that co-designing `(SESC, RPM)` achieves the same Pareto risk as `(MLP, SGD)` at a
   compute budget reduced by the exact factor `(p_SESC / p_MLP) * (B_RPM / B_SGD) < 1`.
-/

noncomputable section

namespace OpenTheory.ComputeOptimalSynthesis

open OpenTheory
open OpenTheory.StochasticSharpnessSGD
open OpenTheory.MomentumAdamTrajectory

/-! ## 1. Exact Variational Chinchilla / Compute-Optimal Allocation (`N*(C), D*(C)`) via Completion of Squares -/

/-- Parametric scaling risk `R(N, D) = R_inf + A / N + B_opt / D`, decomposing excess risk above
irreducible entropy `R_inf` into architectural approximation error `A / N` (for effective model
capacity `N`) and optimization / sample complexity error `B_opt / D` (for training data / step
budget `D`). -/
def scalingRisk (R_inf A B_opt N D : ℝ) : ℝ :=
  R_inf + A / N + B_opt / D

/-- Compute-optimal model capacity `N*(C) = √(A * C / B_opt)` minimizing `scalingRisk` under
the compute constraint `N * D = C`. -/
def optimalModelCapacity (A B_opt C : ℝ) : ℝ :=
  Real.sqrt (A * C / B_opt)

/-- Compute-optimal data / token budget `D*(C) = √(B_opt * C / A)` minimizing `scalingRisk` under
the compute constraint `N * D = C`. -/
def optimalDataBudget (A B_opt C : ℝ) : ℝ :=
  Real.sqrt (B_opt * C / A)

/-- Compute-optimal Pareto risk frontier `R*(C) = R_inf + 2 * √(A * B_opt) / √C`. -/
def paretoComputeRisk (R_inf A B_opt C : ℝ) : ℝ :=
  R_inf + 2 * Real.sqrt (A * B_opt) / Real.sqrt C

/-- Bridge to `EndToEndError`: when `0 ≤ R_inf`, `0 ≤ A`, `0 ≤ B_opt`, `0 < N`, and `0 < D`,
`scalingRisk R_inf A B_opt N D` is the `populationRiskBound` of the `EndToEndError` budget
with irreducible floor `R_inf`, approximation term `A / N`, and optimization/sample term `B_opt / D`. -/
def scalingEndToEndError (R_inf A B_opt N D : ℝ)
    (hR : 0 ≤ R_inf) (hA : 0 ≤ A) (hB : 0 ≤ B_opt) (hN : 0 < N) (hD : 0 < D) :
    EndToEndError where
  ε_approx := R_inf
  ε_opt := A / N
  ε_gen := B_opt / D
  approx_nonneg := hR
  opt_nonneg := div_nonneg hA hN.le
  gen_nonneg := div_nonneg hB hD.le

@[simp]
theorem populationRiskBound_scalingEndToEndError (R_inf A B_opt N D : ℝ)
    (hR : 0 ≤ R_inf) (hA : 0 ≤ A) (hB : 0 ≤ B_opt) (hN : 0 < N) (hD : 0 < D) :
    populationRiskBound (scalingEndToEndError R_inf A B_opt N D hR hA hB hN hD) =
      scalingRisk R_inf A B_opt N D :=
  rfl

/-- Strict positivity of `optimalModelCapacity` and `optimalDataBudget` when `0 < A`, `0 < B_opt`,
and `0 < C`. -/
theorem optimalAllocation_pos {A B_opt C : ℝ}
    (hA : 0 < A) (hB : 0 < B_opt) (hC : 0 < C) :
    0 < optimalModelCapacity A B_opt C ∧ 0 < optimalDataBudget A B_opt C := by
  unfold optimalModelCapacity optimalDataBudget
  refine ⟨Real.sqrt_pos.mpr (by positivity), Real.sqrt_pos.mpr (by positivity)⟩

/-- **Compute Constraint & Optimal Ratio Identity (`optimalCapacity_mul_optimalData_eq_compute`)**:
For `0 < A`, `0 < B_opt`, and `0 < C`, the compute-optimal allocation satisfies the exact budget
identity `optimalModelCapacity A B_opt C * optimalDataBudget A B_opt C = C`, and their optimal
capacity-to-data ratio is `optimalModelCapacity A B_opt C / optimalDataBudget A B_opt C = A / B_opt`. -/
theorem optimalCapacity_mul_optimalData_eq_compute {A B_opt C : ℝ}
    (hA : 0 < A) (hB : 0 < B_opt) (hC : 0 < C) :
    optimalModelCapacity A B_opt C * optimalDataBudget A B_opt C = C ∧
      optimalModelCapacity A B_opt C / optimalDataBudget A B_opt C = A / B_opt := by
  unfold optimalModelCapacity optimalDataBudget
  have hA_ne : A ≠ 0 := ne_of_gt hA
  have hB_ne : B_opt ≠ 0 := ne_of_gt hB
  have hC_ne : C ≠ 0 := ne_of_gt hC
  have harg1 : 0 ≤ A * C / B_opt := by positivity
  refine ⟨?_, ?_⟩
  · rw [← Real.sqrt_mul harg1]
    have hprod : (A * C / B_opt) * (B_opt * C / A) = C ^ 2 := by
      field_simp [hA_ne, hB_ne]
    rw [hprod, Real.sqrt_sq hC.le]
  · rw [← Real.sqrt_div harg1]
    have hquot : (A * C / B_opt) / (B_opt * C / A) = (A / B_opt) ^ 2 := by
      field_simp [hA_ne, hB_ne, hC_ne]
    have hAB_nonneg : 0 ≤ A / B_opt := by positivity
    rw [hquot, Real.sqrt_sq hAB_nonneg]

/-- Product identity for `0 ≤ C` (including `C = 0`):
`optimalModelCapacity A B_opt C * optimalDataBudget A B_opt C = C`. -/
theorem optimalCapacity_mul_optimalData_eq_compute_of_nonneg {A B_opt C : ℝ}
    (hA : 0 < A) (hB : 0 < B_opt) (hC : 0 ≤ C) :
    optimalModelCapacity A B_opt C * optimalDataBudget A B_opt C = C := by
  unfold optimalModelCapacity optimalDataBudget
  have hA_ne : A ≠ 0 := ne_of_gt hA
  have hB_ne : B_opt ≠ 0 := ne_of_gt hB
  have harg1 : 0 ≤ A * C / B_opt := by positivity
  rw [← Real.sqrt_mul harg1]
  have hprod : (A * C / B_opt) * (B_opt * C / A) = C ^ 2 := by
    field_simp [hA_ne, hB_ne]
  rw [hprod, Real.sqrt_sq hC]

/-- **Evaluation of Scaling Risk at the Compute-Optimal Allocation (`scalingRisk_at_optimal_eq_pareto`)**:
For `0 < A`, `0 < B_opt`, and `0 < C`, evaluating `scalingRisk` at `(N*(C), D*(C))` yields the
exact Pareto compute risk frontier `paretoComputeRisk R_inf A B_opt C`. -/
theorem scalingRisk_at_optimal_eq_pareto (R_inf : ℝ) {A B_opt C : ℝ}
    (hA : 0 < A) (hB : 0 < B_opt) (hC : 0 < C) :
    scalingRisk R_inf A B_opt (optimalModelCapacity A B_opt C) (optimalDataBudget A B_opt C) =
      paretoComputeRisk R_inf A B_opt C := by
  unfold scalingRisk optimalModelCapacity optimalDataBudget paretoComputeRisk
  have hsqA : Real.sqrt A ^ 2 = A := Real.sq_sqrt hA.le
  have hsqB : Real.sqrt B_opt ^ 2 = B_opt := Real.sq_sqrt hB.le
  have hsqA_ne : Real.sqrt A ≠ 0 := ne_of_gt (Real.sqrt_pos.mpr hA)
  have hsqB_ne : Real.sqrt B_opt ≠ 0 := ne_of_gt (Real.sqrt_pos.mpr hB)
  have hsqC_ne : Real.sqrt C ≠ 0 := ne_of_gt (Real.sqrt_pos.mpr hC)
  have h1 : Real.sqrt (A * C / B_opt) = Real.sqrt A * Real.sqrt C / Real.sqrt B_opt := by
    rw [Real.sqrt_div (by positivity), Real.sqrt_mul hA.le]
  have h2 : Real.sqrt (B_opt * C / A) = Real.sqrt B_opt * Real.sqrt C / Real.sqrt A := by
    rw [Real.sqrt_div (by positivity), Real.sqrt_mul hB.le]
  have h3 : Real.sqrt (A * B_opt) = Real.sqrt A * Real.sqrt B_opt :=
    Real.sqrt_mul hA.le B_opt
  rw [h1, h2, h3]
  have hterm1 : A / (Real.sqrt A * Real.sqrt C / Real.sqrt B_opt) =
      Real.sqrt A * Real.sqrt B_opt / Real.sqrt C := by
    calc
      A / (Real.sqrt A * Real.sqrt C / Real.sqrt B_opt)
        = (Real.sqrt A ^ 2) / (Real.sqrt A * Real.sqrt C / Real.sqrt B_opt) := by rw [hsqA]
      _ = Real.sqrt A * Real.sqrt B_opt / Real.sqrt C := by
        field_simp [hsqA_ne, hsqB_ne, hsqC_ne]
  have hterm2 : B_opt / (Real.sqrt B_opt * Real.sqrt C / Real.sqrt A) =
      Real.sqrt A * Real.sqrt B_opt / Real.sqrt C := by
    calc
      B_opt / (Real.sqrt B_opt * Real.sqrt C / Real.sqrt A)
        = (Real.sqrt B_opt ^ 2) / (Real.sqrt B_opt * Real.sqrt C / Real.sqrt A) := by rw [hsqB]
      _ = Real.sqrt A * Real.sqrt B_opt / Real.sqrt C := by
        field_simp [hsqA_ne, hsqB_ne, hsqC_ne]
  rw [hterm1, hterm2]
  ring

/-- **Exact Completion-of-Squares Variational Identity (`chinchilla_completion_of_squares`)**:
For any feasible allocation `(N, D)` with `0 < N`, `0 < D`, `0 < C`, `N * D = C`, and non-negative
complexity scales `0 ≤ A`, `0 ≤ B_opt`, the excess risk of `(N, D)` above the Pareto frontier
`paretoComputeRisk R_inf A B_opt C` is the exact square:
`scalingRisk R_inf A B_opt N D - paretoComputeRisk R_inf A B_opt C = (√(A * D) - √(B_opt * N))^2 / C`. -/
theorem chinchilla_completion_of_squares (R_inf : ℝ) {A B_opt N D C : ℝ}
    (hA : 0 ≤ A) (hB : 0 ≤ B_opt) (hN : 0 < N) (hD : 0 < D) (hC : 0 < C)
    (hbudget : N * D = C) :
    scalingRisk R_inf A B_opt N D - paretoComputeRisk R_inf A B_opt C =
      (Real.sqrt (A * D) - Real.sqrt (B_opt * N)) ^ 2 / C := by
  unfold scalingRisk paretoComputeRisk
  have hAD : 0 ≤ A * D := mul_nonneg hA hD.le
  have hBN : 0 ≤ B_opt * N := mul_nonneg hB hN.le
  have hsqAD : Real.sqrt (A * D) ^ 2 = A * D := Real.sq_sqrt hAD
  have hsqBN : Real.sqrt (B_opt * N) ^ 2 = B_opt * N := Real.sq_sqrt hBN
  have hprod_sqrt :
      Real.sqrt (A * D) * Real.sqrt (B_opt * N) =
        Real.sqrt (A * B_opt) * Real.sqrt C := by
    rw [← Real.sqrt_mul hAD, ← Real.sqrt_mul (mul_nonneg hA hB)]
    congr 1
    calc
      (A * D) * (B_opt * N) = (A * B_opt) * (N * D) := by ring
      _ = (A * B_opt) * C := by rw [hbudget]
  have h_expand :
      (Real.sqrt (A * D) - Real.sqrt (B_opt * N)) ^ 2 =
        A * D + B_opt * N - 2 * Real.sqrt (A * B_opt) * Real.sqrt C := by
    calc
      (Real.sqrt (A * D) - Real.sqrt (B_opt * N)) ^ 2
        = Real.sqrt (A * D) ^ 2 + Real.sqrt (B_opt * N) ^ 2 -
            2 * (Real.sqrt (A * D) * Real.sqrt (B_opt * N)) := by ring
      _ = A * D + B_opt * N - 2 * Real.sqrt (A * B_opt) * Real.sqrt C := by
          rw [hsqAD, hsqBN, hprod_sqrt]
          ring
  rw [h_expand]
  have hN_ne : N ≠ 0 := ne_of_gt hN
  have hD_ne : D ≠ 0 := ne_of_gt hD
  have hsqC_ne : Real.sqrt C ≠ 0 := ne_of_gt (Real.sqrt_pos.mpr hC)
  have hsqC : Real.sqrt C * Real.sqrt C = C := Real.mul_self_sqrt hC.le
  have h1 : A / N + B_opt / D = (A * D + B_opt * N) / C := by
    rw [← hbudget]
    field_simp [hN_ne, hD_ne]
  have h2 : 2 * Real.sqrt (A * B_opt) / Real.sqrt C =
      (2 * Real.sqrt (A * B_opt) * Real.sqrt C) / C := by
    calc
      2 * Real.sqrt (A * B_opt) / Real.sqrt C
        = (2 * Real.sqrt (A * B_opt) * Real.sqrt C) / (Real.sqrt C * Real.sqrt C) := by
            field_simp [hsqC_ne]
      _ = (2 * Real.sqrt (A * B_opt) * Real.sqrt C) / C := by rw [hsqC]
  calc
    R_inf + A / N + B_opt / D - (R_inf + 2 * Real.sqrt (A * B_opt) / Real.sqrt C)
      = (A / N + B_opt / D) - 2 * Real.sqrt (A * B_opt) / Real.sqrt C := by ring
    _ = (A * D + B_opt * N) / C - (2 * Real.sqrt (A * B_opt) * Real.sqrt C) / C := by
          rw [h1, h2]
    _ = (A * D + B_opt * N - 2 * Real.sqrt (A * B_opt) * Real.sqrt C) / C := by ring

/-- **Global Variational Optimality of the Chinchilla Pareto Frontier
(`chinchilla_variational_global_optimality`)**:
For every feasible allocation `(N, D)` with `0 < N`, `0 < D`, `0 < C`, `N * D = C`, and
`0 ≤ A`, `0 ≤ B_opt`, the scaling risk is lower-bounded by `paretoComputeRisk R_inf A B_opt C`. -/
theorem chinchilla_variational_global_optimality (R_inf : ℝ) {A B_opt N D C : ℝ}
    (hA : 0 ≤ A) (hB : 0 ≤ B_opt) (hN : 0 < N) (hD : 0 < D) (hC : 0 < C)
    (hbudget : N * D = C) :
    paretoComputeRisk R_inf A B_opt C ≤ scalingRisk R_inf A B_opt N D := by
  have hdiff := chinchilla_completion_of_squares R_inf hA hB hN hD hC hbudget
  have hnonneg : 0 ≤ (Real.sqrt (A * D) - Real.sqrt (B_opt * N)) ^ 2 / C := by positivity
  linarith

/-! ## 2. How Architecture and Optimizer Improvements Shift the Compute-Optimal Allocation & Risk Frontier -/

/-- Monotonicity of `paretoComputeRisk` under joint architecture (`A₁ ≤ A₂`) and optimizer
(`B₁ ≤ B₂`) improvements. -/
theorem paretoComputeRisk_mono_in_arch_and_opt (R_inf : ℝ) {A₁ A₂ B₁ B₂ C : ℝ}
    (hA₁ : 0 ≤ A₁) (hB₁ : 0 ≤ B₁) (hA : A₁ ≤ A₂) (hB : B₁ ≤ B₂) (hC : 0 < C) :
    paretoComputeRisk R_inf A₁ B₁ C ≤ paretoComputeRisk R_inf A₂ B₂ C := by
  unfold paretoComputeRisk
  have hmul : A₁ * B₁ ≤ A₂ * B₂ := mul_le_mul hA hB hB₁ (hA₁.trans hA)
  have hsqrt : Real.sqrt (A₁ * B₁) ≤ Real.sqrt (A₂ * B₂) := Real.sqrt_le_sqrt hmul
  have hsqC_pos : 0 < Real.sqrt C := Real.sqrt_pos.mpr hC
  have hdiv : 2 * Real.sqrt (A₁ * B₁) / Real.sqrt C ≤ 2 * Real.sqrt (A₂ * B₂) / Real.sqrt C := by
    apply div_le_div_of_nonneg_right _ hsqC_pos.le
    linarith
  linarith

/-- **Strict Pareto Risk Reduction Under Architecture or Optimizer Improvement
(`paretoComputeRisk_strict_mono_in_arch_and_opt`)**:
If an architectural improvement reduces the capacity coefficient from `A₂` to `A₁ ≤ A₂` (`0 < A₁`)
and an optimizer improvement reduces the optimization/sample coefficient from `B₂` to `B₁ ≤ B₂`
(`0 < B₁`), with at least one strict (`A₁ < A₂ ∨ B₁ < B₂`), then for every compute budget `C > 0`:
`paretoComputeRisk R_inf A₁ B₁ C < paretoComputeRisk R_inf A₂ B₂ C`. -/
theorem paretoComputeRisk_strict_mono_in_arch_and_opt (R_inf : ℝ) {A₁ A₂ B₁ B₂ C : ℝ}
    (hA₁ : 0 < A₁) (hB₁ : 0 < B₁) (hA : A₁ ≤ A₂) (hB : B₁ ≤ B₂)
    (hstrict : A₁ < A₂ ∨ B₁ < B₂) (hC : 0 < C) :
    paretoComputeRisk R_inf A₁ B₁ C < paretoComputeRisk R_inf A₂ B₂ C := by
  unfold paretoComputeRisk
  have hmul_lt : A₁ * B₁ < A₂ * B₂ := by
    rcases hstrict with hA_lt | hB_lt
    · calc
        A₁ * B₁ < A₂ * B₁ := mul_lt_mul_of_pos_right hA_lt hB₁
        _ ≤ A₂ * B₂ := mul_le_mul_of_nonneg_left hB (hA₁.trans hA_lt).le
    · calc
        A₁ * B₁ ≤ A₂ * B₁ := mul_le_mul_of_nonneg_right hA hB₁.le
        _ < A₂ * B₂ := mul_lt_mul_of_pos_left hB_lt (hA₁.trans_le hA)
  have hsqrt_lt : Real.sqrt (A₁ * B₁) < Real.sqrt (A₂ * B₂) :=
    Real.sqrt_lt_sqrt (by positivity) hmul_lt
  have hsqC_pos : 0 < Real.sqrt C := Real.sqrt_pos.mpr hC
  have hdiv_lt : 2 * Real.sqrt (A₁ * B₁) / Real.sqrt C < 2 * Real.sqrt (A₂ * B₂) / Real.sqrt C :=
    div_lt_div_of_pos_right (by linarith) hsqC_pos
  linarith

/-- **Co-Design Shift of the Optimal Capacity-to-Data Ratio (`optimal_capacity_to_data_ratio_shift`)**:
Shows how co-designing the architecture (`A`) and optimizer (`B_opt`) shifts the optimal
parameter-to-token ratio `optimalModelCapacity A B_opt C / optimalDataBudget A B_opt C = A / B_opt`:
1. A more compact/invariant architecture (`A₁ < A₂` at fixed `B_opt > 0`) strictly reduces the
   parameter-to-data ratio `N*(C) / D*(C)` (`A₁ / B_opt < A₂ / B_opt`), allocating a larger
   fraction of compute `C` to training tokens/steps `D*`.
2. A faster/lower-noise optimizer (`B₁ < B₂` at fixed `A > 0`) strictly increases the
   parameter-to-data ratio `N*(C) / D*(C)` (`A / B₂ < A / B₁`), enabling a larger model capacity
   `N*` at fixed compute `C`. -/
theorem optimal_capacity_to_data_ratio_shift {A A₁ A₂ B_opt B₁ B₂ C : ℝ}
    (hA : 0 < A) (hA₁ : 0 < A₁) (hA_lt : A₁ < A₂)
    (hB : 0 < B_opt) (hB₁ : 0 < B₁) (hB_lt : B₁ < B₂) (hC : 0 < C) :
    (optimalModelCapacity A₁ B_opt C / optimalDataBudget A₁ B_opt C = A₁ / B_opt ∧
      optimalModelCapacity A₂ B_opt C / optimalDataBudget A₂ B_opt C = A₂ / B_opt ∧
      optimalModelCapacity A₁ B_opt C / optimalDataBudget A₁ B_opt C <
        optimalModelCapacity A₂ B_opt C / optimalDataBudget A₂ B_opt C) ∧
    (optimalModelCapacity A B₁ C / optimalDataBudget A B₁ C = A / B₁ ∧
      optimalModelCapacity A B₂ C / optimalDataBudget A B₂ C = A / B₂ ∧
      optimalModelCapacity A B₂ C / optimalDataBudget A B₂ C <
        optimalModelCapacity A B₁ C / optimalDataBudget A B₁ C) := by
  have hA₂ : 0 < A₂ := hA₁.trans hA_lt
  have hB₂ : 0 < B₂ := hB₁.trans hB_lt
  have h1 := (optimalCapacity_mul_optimalData_eq_compute hA₁ hB hC).2
  have h2 := (optimalCapacity_mul_optimalData_eq_compute hA₂ hB hC).2
  have h3 := (optimalCapacity_mul_optimalData_eq_compute hA hB₁ hC).2
  have h4 := (optimalCapacity_mul_optimalData_eq_compute hA hB₂ hC).2
  refine ⟨⟨h1, h2, ?_⟩, ⟨h3, h4, ?_⟩⟩
  · rw [h1, h2]
    exact div_lt_div_of_pos_right hA_lt hB
  · rw [h3, h4]
    exact div_lt_div_of_pos_left hA hB₁ hB_lt

/-- Explicit directional shifts of `optimalModelCapacity` and `optimalDataBudget` at fixed compute `C > 0`:
compressing the architecture (`A₁ < A₂`) increases `optimalDataBudget`, while improving the
optimizer (`B₁ < B₂`) increases `optimalModelCapacity`. -/
theorem optimalAllocation_directional_shifts {A A₁ A₂ B_opt B₁ B₂ C : ℝ}
    (hA : 0 < A) (hA₁ : 0 < A₁) (hA_lt : A₁ < A₂)
    (hB : 0 < B_opt) (hB₁ : 0 < B₁) (hB_lt : B₁ < B₂) (hC : 0 < C) :
    optimalDataBudget A₂ B_opt C < optimalDataBudget A₁ B_opt C ∧
      optimalModelCapacity A B₂ C < optimalModelCapacity A B₁ C := by
  unfold optimalDataBudget optimalModelCapacity
  have hA₂ : 0 < A₂ := hA₁.trans hA_lt
  have hB₂ : 0 < B₂ := hB₁.trans hB_lt
  have hBC_pos : 0 < B_opt * C := mul_pos hB hC
  have hAC_pos : 0 < A * C := mul_pos hA hC
  have harg1 : 0 ≤ B_opt * C / A₂ := div_nonneg hBC_pos.le hA₂.le
  have harg2 : 0 ≤ A * C / B₂ := div_nonneg hAC_pos.le hB₂.le
  refine ⟨?_, ?_⟩
  · exact Real.sqrt_lt_sqrt harg1 (div_lt_div_of_pos_left hBC_pos hA₁ hA_lt)
  · exact Real.sqrt_lt_sqrt harg2 (div_lt_div_of_pos_left hAC_pos hB₁ hB_lt)

/-! ## 3. Mini-Batch Compute-vs-Wall-Clock Pareto Frontier & Critical Batch Size Optimality -/

/-- Per-step optimization progress as a function of batch size `B` and critical batch size `B_crit`:
`stepProgress G S_loc B_crit B = (G / (2 * S_loc)) * (B / (B + B_crit))`. -/
def stepProgress (G S_loc B_crit B : ℝ) : ℝ :=
  (G / (2 * S_loc)) * (B / (B + B_crit))

/-- Per-sample / per-FLOP optimization efficiency at batch size `B`:
`flopEfficiency G S_loc B_crit B = stepProgress G S_loc B_crit B / B`. -/
def flopEfficiency (G S_loc B_crit B : ℝ) : ℝ :=
  stepProgress G S_loc B_crit B / B

/-- Bridge to `StochasticSharpnessSGD`: when `B_crit = criticalBatchRatio G σ_sq = σ_sq / G`,
`stepProgress G S_loc (criticalBatchRatio G σ_sq) (B : ℝ)` equals the exact noise-aware optimal
mini-batch SGD progress `sgdNetProgress (noiseAwareStep S_loc G σ_sq B) S_loc G σ_sq B`. -/
theorem stepProgress_eq_optimal_sgd (S_loc G σ_sq : ℝ) (B : ℕ)
    (hB : 0 < B) (hG : 0 < G) (hS : 0 < S_loc) (hσ : 0 ≤ σ_sq) :
    stepProgress G S_loc (criticalBatchRatio G σ_sq) (B : ℝ) =
      sgdNetProgress (noiseAwareStep S_loc G σ_sq B) S_loc G σ_sq B := by
  unfold stepProgress criticalBatchRatio
  exact (optimal_sgd_progress_vs_full_batch S_loc G σ_sq B hB hG hS hσ).symm

/-- Closed-form expression for `flopEfficiency` when `B ≠ 0`:
`flopEfficiency G S_loc B_crit B = (G / (2 * S_loc)) / (B + B_crit)`. -/
theorem flopEfficiency_eq_div (G S_loc B_crit B : ℝ) (hB : B ≠ 0) :
    flopEfficiency G S_loc B_crit B = (G / (2 * S_loc)) / (B + B_crit) := by
  unfold flopEfficiency stepProgress
  by_cases hden : B + B_crit = 0
  · simp [hden]
  · field_simp [hB, hden]

/-- **Strict Wall-Clock Speedup from Larger Batch Size (`stepProgress_strict_mono_in_batch`)**:
When `0 < G`, `0 < S_loc`, `0 < B_crit`, and `0 < B₁ < B₂`, increasing the batch size from `B₁`
to `B₂` strictly increases per-step optimization progress `stepProgress`. -/
theorem stepProgress_strict_mono_in_batch {G S_loc B_crit B₁ B₂ : ℝ}
    (hG : 0 < G) (hS : 0 < S_loc) (hBcrit : 0 < B_crit)
    (hB₁ : 0 < B₁) (hB_lt : B₁ < B₂) :
    stepProgress G S_loc B_crit B₁ < stepProgress G S_loc B_crit B₂ := by
  unfold stepProgress
  have hB₂ : 0 < B₂ := hB₁.trans hB_lt
  have hden1 : 0 < B₁ + B_crit := by positivity
  have hden2 : 0 < B₂ + B_crit := by positivity
  have hfrac_lt : B₁ / (B₁ + B_crit) < B₂ / (B₂ + B_crit) := by
    rw [div_lt_div_iff₀ hden1 hden2]
    nlinarith
  exact mul_lt_mul_of_pos_left hfrac_lt (by positivity)

/-- **Strict Per-FLOP Efficiency Decay from Larger Batch Size (`flopEfficiency_strict_anti_in_batch`)**:
When `0 < G`, `0 < S_loc`, `0 < B_crit`, and `0 < B₁ < B₂`, increasing the batch size from `B₁`
to `B₂` strictly decreases per-sample FLOP efficiency `flopEfficiency`. -/
theorem flopEfficiency_strict_anti_in_batch {G S_loc B_crit B₁ B₂ : ℝ}
    (hG : 0 < G) (hS : 0 < S_loc) (hBcrit : 0 < B_crit)
    (hB₁ : 0 < B₁) (hB_lt : B₁ < B₂) :
    flopEfficiency G S_loc B_crit B₂ < flopEfficiency G S_loc B_crit B₁ := by
  have hB₂ : 0 < B₂ := hB₁.trans hB_lt
  rw [flopEfficiency_eq_div G S_loc B_crit B₁ (ne_of_gt hB₁),
    flopEfficiency_eq_div G S_loc B_crit B₂ (ne_of_gt hB₂)]
  have hcoef : 0 < G / (2 * S_loc) := by positivity
  have hden1 : 0 < B₁ + B_crit := by positivity
  have hden_lt : B₁ + B_crit < B₂ + B_crit := by linarith
  exact div_lt_div_of_pos_left hcoef hden1 hden_lt

/-- Balanced throughput-FLOP efficiency objective:
`balancedBatchObjective G S_loc B_crit B = stepProgress G S_loc B_crit B * (B_crit / (B + B_crit))`,
measuring the product of per-step progress and normalized per-sample FLOP efficiency
`B_crit / (B + B_crit)`. -/
def balancedBatchObjective (G S_loc B_crit B : ℝ) : ℝ :=
  stepProgress G S_loc B_crit B * (B_crit / (B + B_crit))

/-- Value of `balancedBatchObjective` at the critical batch size `B = B_crit` (`B_crit ≠ 0`):
`balancedBatchObjective G S_loc B_crit B_crit = (1 / 4) * (G / (2 * S_loc))`. -/
theorem balancedBatchObjective_at_critical (G S_loc B_crit : ℝ) (hBcrit : B_crit ≠ 0) :
    balancedBatchObjective G S_loc B_crit B_crit = (1 / 4) * (G / (2 * S_loc)) := by
  unfold balancedBatchObjective stepProgress
  have h2 : B_crit + B_crit ≠ 0 := by
    intro h
    apply hBcrit
    linarith
  field_simp [hBcrit, h2]
  ring

/-- **Exact Completion-of-Squares Theorem for Critical Batch Size (`critical_batch_variational_optimality`)**:
For any `B > 0` and `B_crit > 0` (with `0 < G`, `0 < S_loc`), the suboptimality gap of batch size
`B` relative to the critical batch size `B_crit` on `balancedBatchObjective` is the exact square:
`balancedBatchObjective G S_loc B_crit B_crit - balancedBatchObjective G S_loc B_crit B = (G / (2 * S_loc)) * ((B - B_crit)^2 / (4 * (B + B_crit)^2))`. -/
theorem critical_batch_variational_optimality {G S_loc B_crit B : ℝ}
    (_hG : 0 < G) (hS : 0 < S_loc) (hBcrit : 0 < B_crit) (hB : 0 < B) :
    balancedBatchObjective G S_loc B_crit B_crit -
        balancedBatchObjective G S_loc B_crit B =
      (G / (2 * S_loc)) * ((B - B_crit) ^ 2 / (4 * (B + B_crit) ^ 2)) := by
  rw [balancedBatchObjective_at_critical G S_loc B_crit (ne_of_gt hBcrit)]
  unfold balancedBatchObjective stepProgress
  have hS_ne : S_loc ≠ 0 := ne_of_gt hS
  have hden_ne : B + B_crit ≠ 0 := ne_of_gt (by positivity)
  field_simp [hS_ne, hden_ne]
  ring

/-- **Global Optimality and Uniqueness of the Critical Batch Size (`critical_batch_global_maximizer`)**:
For any `B > 0` and `B_crit > 0` (`0 < G`, `0 < S_loc`), `B = B_crit` globally maximizes
`balancedBatchObjective`, and strictly dominates every `B ≠ B_crit`. -/
theorem critical_batch_global_maximizer {G S_loc B_crit B : ℝ}
    (hG : 0 < G) (hS : 0 < S_loc) (hBcrit : 0 < B_crit) (hB : 0 < B) :
    balancedBatchObjective G S_loc B_crit B ≤
        balancedBatchObjective G S_loc B_crit B_crit ∧
      (B ≠ B_crit →
        balancedBatchObjective G S_loc B_crit B <
          balancedBatchObjective G S_loc B_crit B_crit) := by
  have hdiff := critical_batch_variational_optimality hG hS hBcrit hB
  have hnonneg : 0 ≤ (G / (2 * S_loc)) * ((B - B_crit) ^ 2 / (4 * (B + B_crit) ^ 2)) := by
    positivity
  refine ⟨by linarith, fun hne => ?_⟩
  have hsub_ne : B - B_crit ≠ 0 := sub_ne_zero.mpr hne
  have hsq_pos : 0 < (B - B_crit) ^ 2 := sq_pos_of_ne_zero hsub_ne
  have hpos : 0 < (G / (2 * S_loc)) * ((B - B_crit) ^ 2 / (4 * (B + B_crit) ^ 2)) := by
    positivity
  linarith

/-- Effective critical batch size under EMA momentum `β`:
`momentumCriticalBatchRatio β G σ_sq = ((1 - β) / (1 + β)) * criticalBatchRatio G σ_sq`. -/
def momentumCriticalBatchRatio (β G σ_sq : ℝ) : ℝ :=
  ((1 - β) / (1 + β)) * (σ_sq / G)

/-- **EMA Momentum Strictly Reduces the Critical Batch Size (`momentum_reduces_critical_batch_size`)**:
Under EMA momentum `0 < β < 1` with `0 < G` and `0 < σ_sq`, the effective critical batch size
`((1 - β) / (1 + β)) * (σ_sq / G)` is positive and **strictly smaller** than memoryless SGD's
critical batch size `criticalBatchRatio G σ_sq = σ_sq / G`. -/
theorem momentum_reduces_critical_batch_size {β G σ_sq : ℝ}
    (hβ0 : 0 < β) (hβ1 : β < 1) (hG : 0 < G) (hσ : 0 < σ_sq) :
    0 < ((1 - β) / (1 + β)) * (σ_sq / G) ∧
      ((1 - β) / (1 + β)) * (σ_sq / G) < σ_sq / G ∧
      momentumCriticalBatchRatio β G σ_sq < criticalBatchRatio G σ_sq := by
  unfold momentumCriticalBatchRatio criticalBatchRatio
  have hdamp := ema_damping_factor_lt_one hβ0 hβ1
  have hcrit_pos : 0 < σ_sq / G := div_pos hσ hG
  have hlt : ((1 - β) / (1 + β)) * (σ_sq / G) < σ_sq / G := by
    nlinarith [hdamp.2]
  exact ⟨mul_pos hdamp.1 hcrit_pos, hlt, hlt⟩

/-! ## 4. Master Co-Design Dominance Theorem -/

/-- Equivalent compute budget `C_co` required by a co-designed `(A_co, B_co)` system to match the
Pareto compute risk of a baseline `(A_base, B_base)` system at compute `C_base`:
`equivalentComputeBudget A_co A_base B_co B_base C_base = (A_co / A_base) * (B_co / B_base) * C_base`. -/
def equivalentComputeBudget (A_co A_base B_co B_base C_base : ℝ) : ℝ :=
  (A_co / A_base) * (B_co / B_base) * C_base

/-- **Exact Compute-Saving Factor Identity (`codesign_compute_saving_factor`)**:
When `0 < A_co`, `0 < A_base`, `0 < B_co`, `0 < B_base`, and `0 < C_base`, running `(A_co, B_co)`
at compute budget `C_co = (A_co / A_base) * (B_co / B_base) * C_base` achieves **the exact same**
Pareto compute risk as running `(A_base, B_base)` at `C_base`. -/
theorem codesign_compute_saving_factor (R_inf : ℝ) {A_co A_base B_co B_base C_base : ℝ}
    (hA_co : 0 < A_co) (hA_base : 0 < A_base)
    (hB_co : 0 < B_co) (hB_base : 0 < B_base) (hC_base : 0 < C_base) :
    paretoComputeRisk R_inf A_co B_co (equivalentComputeBudget A_co A_base B_co B_base C_base) =
      paretoComputeRisk R_inf A_base B_base C_base := by
  unfold paretoComputeRisk equivalentComputeBudget
  have hsqA_co_ne : Real.sqrt A_co ≠ 0 := ne_of_gt (Real.sqrt_pos.mpr hA_co)
  have hsqB_co_ne : Real.sqrt B_co ≠ 0 := ne_of_gt (Real.sqrt_pos.mpr hB_co)
  have hsqA_base_ne : Real.sqrt A_base ≠ 0 := ne_of_gt (Real.sqrt_pos.mpr hA_base)
  have hsqB_base_ne : Real.sqrt B_base ≠ 0 := ne_of_gt (Real.sqrt_pos.mpr hB_base)
  have hsqC_ne : Real.sqrt C_base ≠ 0 := ne_of_gt (Real.sqrt_pos.mpr hC_base)
  have hnum_co : Real.sqrt (A_co * B_co) = Real.sqrt A_co * Real.sqrt B_co :=
    Real.sqrt_mul hA_co.le B_co
  have hnum_base : Real.sqrt (A_base * B_base) = Real.sqrt A_base * Real.sqrt B_base :=
    Real.sqrt_mul hA_base.le B_base
  have hden_co :
      Real.sqrt (A_co / A_base * (B_co / B_base) * C_base) =
        (Real.sqrt A_co / Real.sqrt A_base) * (Real.sqrt B_co / Real.sqrt B_base) *
          Real.sqrt C_base := by
    rw [Real.sqrt_mul (by positivity), Real.sqrt_mul (by positivity),
      Real.sqrt_div hA_co.le, Real.sqrt_div hB_co.le]
  rw [hnum_co, hnum_base, hden_co]
  field_simp [hsqA_co_ne, hsqB_co_ne, hsqA_base_ne, hsqB_base_ne, hsqC_ne]

/-- **Master Co-Design Dominance Theorem (`master_codesign_compute_dominance`)**:
Combines the 4-tier architectural complexity compression `0 < p_SESC < p_Subspace < p_CNN < p_MLP`
(`A_i = c_arch * p_i` with `c_arch > 0`) and the 3-tier optimizer complexity reduction
`0 < B_RPM < B_Adam < B_SGD` to establish:
1. The strict 4-step Pareto compute risk chain for every compute budget `C > 0`:
   `paretoComputeRisk R_inf (c_arch * p_SESC) B_RPM C < paretoComputeRisk R_inf (c_arch * p_Subspace) B_Adam C < paretoComputeRisk R_inf (c_arch * p_CNN) B_SGD C < paretoComputeRisk R_inf (c_arch * p_MLP) B_SGD C`.
2. Exact compute equivalence: at compute budget `C_co = ((p_SESC / p_MLP) * (B_RPM / B_SGD)) * C_base`,
   the co-designed `(SESC, RPM)` system matches the Pareto risk of `(MLP, SGD)` at `C_base`, with
   strict compute reduction factor `0 < (p_SESC / p_MLP) * (B_RPM / B_SGD) < 1` and `C_co < C_base`. -/
theorem master_codesign_compute_dominance (R_inf : ℝ)
    {c_arch p_SESC p_Subspace p_CNN p_MLP B_RPM B_Adam B_SGD C C_base : ℝ}
    (hc : 0 < c_arch)
    (hp_SESC : 0 < p_SESC) (hp1 : p_SESC < p_Subspace)
    (hp2 : p_Subspace < p_CNN) (hp3 : p_CNN < p_MLP)
    (hB_RPM : 0 < B_RPM) (hB1 : B_RPM < B_Adam) (hB2 : B_Adam < B_SGD)
    (hC : 0 < C) (hC_base : 0 < C_base) :
    (paretoComputeRisk R_inf (c_arch * p_SESC) B_RPM C <
        paretoComputeRisk R_inf (c_arch * p_Subspace) B_Adam C ∧
      paretoComputeRisk R_inf (c_arch * p_Subspace) B_Adam C <
        paretoComputeRisk R_inf (c_arch * p_CNN) B_SGD C ∧
      paretoComputeRisk R_inf (c_arch * p_CNN) B_SGD C <
        paretoComputeRisk R_inf (c_arch * p_MLP) B_SGD C) ∧
    (0 < (p_SESC / p_MLP) * (B_RPM / B_SGD) ∧
      (p_SESC / p_MLP) * (B_RPM / B_SGD) < 1 ∧
      paretoComputeRisk R_inf (c_arch * p_SESC) B_RPM
        (((p_SESC / p_MLP) * (B_RPM / B_SGD)) * C_base) =
        paretoComputeRisk R_inf (c_arch * p_MLP) B_SGD C_base ∧
      ((p_SESC / p_MLP) * (B_RPM / B_SGD)) * C_base < C_base) := by
  have hp_Subspace : 0 < p_Subspace := hp_SESC.trans hp1
  have hp_CNN : 0 < p_CNN := hp_Subspace.trans hp2
  have hp_MLP : 0 < p_MLP := hp_CNN.trans hp3
  have hB_Adam : 0 < B_Adam := hB_RPM.trans hB1
  have hB_SGD : 0 < B_SGD := hB_Adam.trans hB2
  have hA_SESC : 0 < c_arch * p_SESC := mul_pos hc hp_SESC
  have hA_Subspace : 0 < c_arch * p_Subspace := mul_pos hc hp_Subspace
  have hA_CNN : 0 < c_arch * p_CNN := mul_pos hc hp_CNN
  have hA_MLP : 0 < c_arch * p_MLP := mul_pos hc hp_MLP
  have hA1 : c_arch * p_SESC < c_arch * p_Subspace := mul_lt_mul_of_pos_left hp1 hc
  have hA2 : c_arch * p_Subspace < c_arch * p_CNN := mul_lt_mul_of_pos_left hp2 hc
  have hA3 : c_arch * p_CNN < c_arch * p_MLP := mul_lt_mul_of_pos_left hp3 hc
  have hstep1 :
      paretoComputeRisk R_inf (c_arch * p_SESC) B_RPM C <
        paretoComputeRisk R_inf (c_arch * p_Subspace) B_Adam C :=
    paretoComputeRisk_strict_mono_in_arch_and_opt R_inf hA_SESC hB_RPM
      hA1.le hB1.le (Or.inl hA1) hC
  have hstep2 :
      paretoComputeRisk R_inf (c_arch * p_Subspace) B_Adam C <
        paretoComputeRisk R_inf (c_arch * p_CNN) B_SGD C :=
    paretoComputeRisk_strict_mono_in_arch_and_opt R_inf hA_Subspace hB_Adam
      hA2.le hB2.le (Or.inl hA2) hC
  have hstep3 :
      paretoComputeRisk R_inf (c_arch * p_CNN) B_SGD C <
        paretoComputeRisk R_inf (c_arch * p_MLP) B_SGD C :=
    paretoComputeRisk_strict_mono_in_arch_and_opt R_inf hA_CNN hB_SGD
      hA3.le le_rfl (Or.inl hA3) hC
  have hp_lt : p_SESC < p_MLP := (hp1.trans hp2).trans hp3
  have hB_lt : B_RPM < B_SGD := hB1.trans hB2
  have hfac_p_pos : 0 < p_SESC / p_MLP := div_pos hp_SESC hp_MLP
  have hfac_B_pos : 0 < B_RPM / B_SGD := div_pos hB_RPM hB_SGD
  have hfac_p_lt1 : p_SESC / p_MLP < 1 := (div_lt_one hp_MLP).mpr hp_lt
  have hfac_B_lt1 : B_RPM / B_SGD < 1 := (div_lt_one hB_SGD).mpr hB_lt
  have hfactor_pos : 0 < (p_SESC / p_MLP) * (B_RPM / B_SGD) := mul_pos hfac_p_pos hfac_B_pos
  have hfactor_lt1 : (p_SESC / p_MLP) * (B_RPM / B_SGD) < 1 := by
    calc
      (p_SESC / p_MLP) * (B_RPM / B_SGD) < 1 * (B_RPM / B_SGD) :=
        mul_lt_mul_of_pos_right hfac_p_lt1 hfac_B_pos
      _ = B_RPM / B_SGD := one_mul _
      _ < 1 := hfac_B_lt1
  have heq_budget :
      ((p_SESC / p_MLP) * (B_RPM / B_SGD)) * C_base =
        equivalentComputeBudget (c_arch * p_SESC) (c_arch * p_MLP) B_RPM B_SGD C_base := by
    unfold equivalentComputeBudget
    have hc_ne : c_arch ≠ 0 := ne_of_gt hc
    have hp_MLP_ne : p_MLP ≠ 0 := ne_of_gt hp_MLP
    field_simp [hc_ne, hp_MLP_ne]
  have heq_risk :
      paretoComputeRisk R_inf (c_arch * p_SESC) B_RPM
          (((p_SESC / p_MLP) * (B_RPM / B_SGD)) * C_base) =
        paretoComputeRisk R_inf (c_arch * p_MLP) B_SGD C_base := by
    rw [heq_budget]
    exact codesign_compute_saving_factor R_inf hA_SESC hA_MLP hB_RPM hB_SGD hC_base
  have hC_lt : ((p_SESC / p_MLP) * (B_RPM / B_SGD)) * C_base < C_base := by
    nlinarith
  exact ⟨⟨hstep1, hstep2, hstep3⟩, ⟨hfactor_pos, hfactor_lt1, heq_risk, hC_lt⟩⟩

end OpenTheory.ComputeOptimalSynthesis
