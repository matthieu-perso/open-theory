import OpenTheory.Optimization.CertiRPM
import OpenTheory.Generalization.ComputeOptimalSynthesis

/-!
# Verifiable Recursive AI Self-Improvement & Zero-GPU Symbolic Search Filtering

In empirical AI research, when an automated researcher or self-improving AI proposes a candidate
self-modification from generation $g$'s design $(A_{\mathrm{curr}}, B_{\mathrm{curr}}, \hat{S}_{\mathrm{curr}})$
to a candidate design $(A_{\mathrm{cand}}, B_{\mathrm{cand}}, \hat{S}_{\mathrm{cand}})$, testing
candidates directly on GPU clusters faces two fundamental failure modes:
1. **Catastrophic Training Divergence**: if the candidate optimizer's sharpness estimate
   underestimates true curvature ($\hat{S}_{\mathrm{cand}, i} < S_i / 2$), optimization overshoots
   the quadratic stability barrier and diverges.
2. **Silent Pareto Regression & Brute-Force GPU Waste**: if the candidate's combined complexity
   product $A_{\mathrm{cand}} B_{\mathrm{cand}}$ fails to contract below $A_{\mathrm{curr}} B_{\mathrm{curr}}$,
   or if $M_{\mathrm{cand}}$ candidates must all be trained empirically at cost $C_{\mathrm{train}}$
   each to identify a winner, massive GPU training FLOPs are wasted on divergent or suboptimal trials.

This module formalizes **Verifiable Recursive AI Self-Improvement** with zero `sorry`:

1. **Section 1: Symbolic Safety & Pareto-Dominance Filter for Candidate Self-Modifications
   (`IsVerifiablySafeImprovement`, `verifiably_safe_improvement_guarantees_descent_and_pareto_drop`)**:
   - Defines the Lean symbolic acceptance predicate `IsVerifiablySafeImprovement`, which checks
     that (i) $\forall i, S_i \le \hat{S}_{\mathrm{cand}, i}$ (certified upper-confidence sharpness)
     and (ii) $A_{\mathrm{cand}} B_{\mathrm{cand}} \le \gamma_{\mathrm{imp}} (A_{\mathrm{curr}} B_{\mathrm{curr}})$
     for contraction factor $\gamma_{\mathrm{imp}} \in (0, 1)$.
   - Proves that every candidate passing `IsVerifiablySafeImprovement` is simultaneously guaranteed
     **100% non-divergent positive descent** (`0 ≤ optimizerNetProgress`) and **certified Pareto
     risk contraction** (`2 * √(A_cand * B_cand) / √C ≤ √γ_imp * (2 * √(A_curr * B_curr) / √C)` and
     `paretoComputeRisk R_inf A_cand B_cand C < paretoComputeRisk R_inf A_curr B_curr C`).

2. **Section 2: Multi-Generation Recursive Self-Improvement Law
   (`generationComplexityProduct`, `generationParetoRisk`, `generationEquivalentCompute`,
   `generationParetoRisk_zero`, `generationParetoRisk_strict_anti_in_generation`,
   `recursive_self_improvement_exponential_compute_compression`)**:
   - Proves that across $G$ verified recursive self-improvement generations, the complexity product
     contracts exponentially as $\gamma_{\mathrm{imp}}^G \cdot AB_0$ and `generationParetoRisk`
     is strictly decreasing in the generation index $G$.
   - Proves **Exponential Compute Compression**: a generation-$G$ self-improved architecture and
     optimizer trained with only $\gamma_{\mathrm{imp}}^G C_0$ FLOPs achieves the **exact same**
     Pareto population risk as the generation-$0$ baseline trained with full compute $C_0$, with
     $\gamma_{\mathrm{imp}}^G C_0 < C_0$ for all $G \ge 1$.

3. **Section 3: Symbolic Verification vs. Empirical Grid-Search FLOP Savings
   (`empiricalSearchCompute`, `symbolicSearchCompute`,
   `symbolic_vs_empirical_search_flop_savings`)**:
   - Compares empirical brute-force search over $M_{\mathrm{cand}} \ge 2$ candidates
     (`M_cand * C_train` FLOPs) against Lean symbolic pre-verification (`M_cand * C_verify + C_train`
     FLOPs, filtering out $M_{\mathrm{cand}} - 1$ candidates at zero GPU training cost).
   - Proves the exact FLOP savings identity, the strict compute reduction threshold
     $C_{\mathrm{verify}} < \frac{M_{\mathrm{cand}} - 1}{M_{\mathrm{cand}}} C_{\mathrm{train}}$,
     and the $M_{\mathrm{cand}}\times$ per-generation ($M_{\mathrm{cand}}^G\times$ multi-generation)
     compute compression ratio in the zero-GPU-verification limit $C_{\mathrm{verify}} = 0$.
-/

noncomputable section

open scoped BigOperators RealInnerProductSpace

namespace OpenTheory.VerifiableSelfImprovement

open OpenTheory.LocalSharpnessDescent
open OpenTheory.PreconditionedDescent
open OpenTheory.StochasticSharpnessSGD
open OpenTheory.MomentumAdamTrajectory
open OpenTheory.ResidualPreconditionedMomentum
open OpenTheory.CertiRPM
open OpenTheory.ComputeOptimalSynthesis

variable {p : ℕ}

/-! ## Section 1: Symbolic Safety & Pareto-Dominance Filter for Candidate Self-Modifications -/

/-- **Lean Symbolic Acceptance Predicate (`IsVerifiablySafeImprovement`)**:
A candidate self-modification `(A_cand, B_cand, S_hat_cand)` is symbolically accepted over the
current generation's `(A_curr, B_curr)` with contraction factor `γ_imp` if:
1. Every coordinate sharpness estimate `S_hat_cand i` is a valid upper bound on true sharpness `S i`
   (`∀ i, S i ≤ S_hat_cand i`), preventing catastrophic step-size overshoot, and
2. The candidate complexity product `A_cand * B_cand` contracts by at least `γ_imp` relative to
   `A_curr * B_curr` (`A_cand * B_cand ≤ γ_imp * (A_curr * B_curr)`). -/
def IsVerifiablySafeImprovement (S : Fin p → ℝ) (S_hat_cand : Fin p → ℝ)
    (A_curr B_curr A_cand B_cand γ_imp : ℝ) : Prop :=
  (∀ i : Fin p, S i ≤ S_hat_cand i) ∧
    A_cand * B_cand ≤ γ_imp * (A_curr * B_curr)

/-- **Certified Non-Divergent Descent & Pareto Risk Contraction
(`verifiably_safe_improvement_guarantees_descent_and_pareto_drop`)**:
Whenever `0 < S i`, `0 < totalCoordMoment g β σ_coord B i`, `0 < A_cand`, `0 < B_cand`,
`0 < A_curr`, `0 < B_curr`, `0 < γ_imp < 1`, `0 < C`, and
`IsVerifiablySafeImprovement S S_hat_cand A_curr B_curr A_cand B_cand γ_imp` holds:
1. **100% Non-Divergent Step Guarantee**:
   `0 ≤ optimizerNetProgress (pluginSharpnessStep S_hat_cand g β σ_coord B) S g β σ_coord B`,
2. **Certified Reducible Pareto Error Contraction**:
   `2 * Real.sqrt (A_cand * B_cand) / Real.sqrt C ≤ Real.sqrt γ_imp * (2 * Real.sqrt (A_curr * B_curr) / Real.sqrt C)`, and
3. **Strict Pareto Compute Risk Drop**:
   `paretoComputeRisk R_inf A_cand B_cand C < paretoComputeRisk R_inf A_curr B_curr C`. -/
theorem verifiably_safe_improvement_guarantees_descent_and_pareto_drop
    (R_inf : ℝ) (S S_hat_cand : Fin p → ℝ) (g : Parameters p) (β : ℝ)
    (σ_coord : Fin p → ℝ) (B A_curr B_curr A_cand B_cand γ_imp C : ℝ)
    (hS : ∀ i, 0 < S i)
    (hM : ∀ i, 0 < totalCoordMoment g β σ_coord B i)
    (hA_cand : 0 < A_cand) (hB_cand : 0 < B_cand)
    (hA_curr : 0 < A_curr) (hB_curr : 0 < B_curr)
    (hγ_pos : 0 < γ_imp) (hγ_lt : γ_imp < 1) (hC : 0 < C)
    (hverif : IsVerifiablySafeImprovement S S_hat_cand A_curr B_curr A_cand B_cand γ_imp) :
    0 ≤ optimizerNetProgress (pluginSharpnessStep S_hat_cand g β σ_coord B) S g β σ_coord B ∧
      2 * Real.sqrt (A_cand * B_cand) / Real.sqrt C ≤
        Real.sqrt γ_imp * (2 * Real.sqrt (A_curr * B_curr) / Real.sqrt C) ∧
      paretoComputeRisk R_inf A_cand B_cand C <
        paretoComputeRisk R_inf A_curr B_curr C := by
  obtain ⟨hS_le, hAB_le⟩ := hverif
  have hsafe :=
    (upper_confidence_sharpness_guarantees_positive_progress
      S_hat_cand S g β σ_coord B hS hS_le hM).2.2
  have hsqC_pos : 0 < Real.sqrt C := Real.sqrt_pos.mpr hC
  have hAB_curr_pos : 0 < A_curr * B_curr := mul_pos hA_curr hB_curr
  have hAB_cand_pos : 0 < A_cand * B_cand := mul_pos hA_cand hB_cand
  have hsqrt_le :
      Real.sqrt (A_cand * B_cand) ≤ Real.sqrt γ_imp * Real.sqrt (A_curr * B_curr) := by
    rw [← Real.sqrt_mul hγ_pos.le]
    exact Real.sqrt_le_sqrt hAB_le
  have hreducible_le :
      2 * Real.sqrt (A_cand * B_cand) / Real.sqrt C ≤
        Real.sqrt γ_imp * (2 * Real.sqrt (A_curr * B_curr) / Real.sqrt C) := by
    calc
      2 * Real.sqrt (A_cand * B_cand) / Real.sqrt C
        ≤ 2 * (Real.sqrt γ_imp * Real.sqrt (A_curr * B_curr)) / Real.sqrt C := by
          apply div_le_div_of_nonneg_right _ hsqC_pos.le
          linarith
      _ = Real.sqrt γ_imp * (2 * Real.sqrt (A_curr * B_curr) / Real.sqrt C) := by ring
  have hsqrt_γ_lt_one : Real.sqrt γ_imp < 1 := by
    rw [← Real.sqrt_one]
    exact Real.sqrt_lt_sqrt hγ_pos.le hγ_lt
  have hcurr_reducible_pos : 0 < 2 * Real.sqrt (A_curr * B_curr) / Real.sqrt C := by
    have hsq_curr : 0 < Real.sqrt (A_curr * B_curr) := Real.sqrt_pos.mpr hAB_curr_pos
    positivity
  have hstrict_reducible :
      2 * Real.sqrt (A_cand * B_cand) / Real.sqrt C <
        2 * Real.sqrt (A_curr * B_curr) / Real.sqrt C := by
    calc
      2 * Real.sqrt (A_cand * B_cand) / Real.sqrt C
        ≤ Real.sqrt γ_imp * (2 * Real.sqrt (A_curr * B_curr) / Real.sqrt C) := hreducible_le
      _ < 1 * (2 * Real.sqrt (A_curr * B_curr) / Real.sqrt C) :=
          mul_lt_mul_of_pos_right hsqrt_γ_lt_one hcurr_reducible_pos
      _ = 2 * Real.sqrt (A_curr * B_curr) / Real.sqrt C := one_mul _
  have hpareto_lt :
      paretoComputeRisk R_inf A_cand B_cand C <
        paretoComputeRisk R_inf A_curr B_curr C := by
    unfold paretoComputeRisk
    linarith
  exact ⟨hsafe, hreducible_le, hpareto_lt⟩

/-! ## Section 2: Multi-Generation Recursive Self-Improvement Law -/

/-- **Generation-$G$ Complexity Product (`generationComplexityProduct`)**:
When each verified recursive self-improvement step compresses the joint architecture-optimizer
complexity product by factor `γ_imp ∈ (0, 1)`, the complexity product at generation `G : ℕ` is
`γ_imp ^ G * AB₀`. -/
def generationComplexityProduct (AB₀ γ_imp : ℝ) (G : ℕ) : ℝ :=
  γ_imp ^ G * AB₀

/-- **Generation-$G$ Pareto Compute Risk (`generationParetoRisk`)**:
The compute-optimal population risk at generation `G : ℕ` under compute budget `C`:
`R_inf + 2 * √(generationComplexityProduct AB₀ γ_imp G) / √C`. -/
def generationParetoRisk (R_inf AB₀ γ_imp C : ℝ) (G : ℕ) : ℝ :=
  R_inf + 2 * Real.sqrt (generationComplexityProduct AB₀ γ_imp G) / Real.sqrt C

/-- **Generation-$G$ Equivalent Training Compute (`generationEquivalentCompute`)**:
The training compute budget `γ_imp ^ G * C₀` required at generation `G : ℕ` to match the
generation-$0$ baseline Pareto risk at full compute `C₀`. -/
def generationEquivalentCompute (C₀ γ_imp : ℝ) (G : ℕ) : ℝ :=
  γ_imp ^ G * C₀

/-- At generation `G = 0`, `generationComplexityProduct AB₀ γ_imp 0 = AB₀`. -/
@[simp]
theorem generationComplexityProduct_zero (AB₀ γ_imp : ℝ) :
    generationComplexityProduct AB₀ γ_imp 0 = AB₀ := by
  simp [generationComplexityProduct]

/-- At generation `G = 0`, `generationParetoRisk R_inf (A₀ * B₀) γ_imp C 0` coincides with the
baseline `paretoComputeRisk R_inf A₀ B₀ C`. -/
@[simp]
theorem generationParetoRisk_zero (R_inf A₀ B₀ γ_imp C : ℝ) :
    generationParetoRisk R_inf (A₀ * B₀) γ_imp C 0 =
      paretoComputeRisk R_inf A₀ B₀ C := by
  simp [generationParetoRisk, generationComplexityProduct, paretoComputeRisk]

/-- **Strict Pareto Risk Reduction Across Recursive Generations
(`generationParetoRisk_strict_anti_in_generation`)**:
Whenever `0 < AB₀`, `0 < γ_imp < 1`, `0 < C`, and `G₁ < G₂`, recursive self-improvement strictly
reduces Pareto population risk:
`generationParetoRisk R_inf AB₀ γ_imp C G₂ < generationParetoRisk R_inf AB₀ γ_imp C G₁`. -/
theorem generationParetoRisk_strict_anti_in_generation
    (R_inf : ℝ) {AB₀ γ_imp C : ℝ} {G₁ G₂ : ℕ}
    (hAB₀ : 0 < AB₀) (hγ_pos : 0 < γ_imp) (hγ_lt : γ_imp < 1)
    (hC : 0 < C) (hG : G₁ < G₂) :
    generationParetoRisk R_inf AB₀ γ_imp C G₂ <
      generationParetoRisk R_inf AB₀ γ_imp C G₁ := by
  unfold generationParetoRisk generationComplexityProduct
  have hpow_lt : γ_imp ^ G₂ < γ_imp ^ G₁ :=
     pow_lt_pow_right_of_lt_one₀ hγ_pos hγ_lt hG
  have hprod_lt : γ_imp ^ G₂ * AB₀ < γ_imp ^ G₁ * AB₀ :=
    mul_lt_mul_of_pos_right hpow_lt hAB₀
  have hprod_nonneg : 0 ≤ γ_imp ^ G₂ * AB₀ := by positivity
  have hsqrt_lt : Real.sqrt (γ_imp ^ G₂ * AB₀) < Real.sqrt (γ_imp ^ G₁ * AB₀) :=
    Real.sqrt_lt_sqrt hprod_nonneg hprod_lt
  have hsqC_pos : 0 < Real.sqrt C := Real.sqrt_pos.mpr hC
  have hdiv_lt :
      2 * Real.sqrt (γ_imp ^ G₂ * AB₀) / Real.sqrt C <
        2 * Real.sqrt (γ_imp ^ G₁ * AB₀) / Real.sqrt C :=
    div_lt_div_of_pos_right (by linarith) hsqC_pos
  linarith

/-- **Exponential Compute Compression Across Recursive Generations
(`recursive_self_improvement_exponential_compute_compression`)**:
For any generation `G : ℕ` with `0 < A₀`, `0 < B₀`, `0 < γ_imp < 1`, and `0 < C₀`:
1. A self-improved model at generation `G` using only
   `generationEquivalentCompute C₀ γ_imp G = γ_imp ^ G * C₀` training FLOPs achieves the
   **exact same Pareto population risk** as the generation-`0` baseline trained with full
   compute `C₀`:
   `paretoComputeRisk R_inf (γ_imp ^ G * A₀) B₀ (generationEquivalentCompute C₀ γ_imp G) = paretoComputeRisk R_inf A₀ B₀ C₀`.
2. For every generation `G ≥ 1`, the required training compute is strictly compressed:
   `0 < generationEquivalentCompute C₀ γ_imp G` and
   `generationEquivalentCompute C₀ γ_imp G < C₀`. -/
theorem recursive_self_improvement_exponential_compute_compression
    (R_inf : ℝ) {A₀ B₀ γ_imp C₀ : ℝ} (G : ℕ)
    (_hA₀ : 0 < A₀) (_hB₀ : 0 < B₀) (hγ_pos : 0 < γ_imp) (hγ_lt : γ_imp < 1)
    (hC₀ : 0 < C₀) :
    paretoComputeRisk R_inf (γ_imp ^ G * A₀) B₀
        (generationEquivalentCompute C₀ γ_imp G) =
      paretoComputeRisk R_inf A₀ B₀ C₀ ∧
    0 < generationEquivalentCompute C₀ γ_imp G ∧
    (1 ≤ G → generationEquivalentCompute C₀ γ_imp G < C₀) := by
  unfold paretoComputeRisk generationEquivalentCompute
  have hpow_pos : 0 < γ_imp ^ G := pow_pos hγ_pos G
  have hsq_pow_pos : 0 < Real.sqrt (γ_imp ^ G) := Real.sqrt_pos.mpr hpow_pos
  have hsq_pow_ne : Real.sqrt (γ_imp ^ G) ≠ 0 := ne_of_gt hsq_pow_pos
  have hsqC₀_ne : Real.sqrt C₀ ≠ 0 := ne_of_gt (Real.sqrt_pos.mpr hC₀)
  have hnum :
      Real.sqrt (γ_imp ^ G * A₀ * B₀) =
        Real.sqrt (γ_imp ^ G) * Real.sqrt (A₀ * B₀) := by
    rw [mul_assoc, Real.sqrt_mul hpow_pos.le]
  have hden :
      Real.sqrt (γ_imp ^ G * C₀) =
        Real.sqrt (γ_imp ^ G) * Real.sqrt C₀ :=
    Real.sqrt_mul hpow_pos.le C₀
  refine ⟨?_, mul_pos hpow_pos hC₀, fun hG => ?_⟩
  · rw [hnum, hden]
    field_simp [hsq_pow_ne, hsqC₀_ne]
  · have hpow_lt_one : γ_imp ^ G < 1 := by
      have h0 : 0 < G := Nat.lt_of_lt_of_le Nat.zero_lt_one hG
      simpa using pow_lt_pow_right_of_lt_one₀ hγ_pos hγ_lt h0
    calc
      γ_imp ^ G * C₀ < 1 * C₀ := mul_lt_mul_of_pos_right hpow_lt_one hC₀
      _ = C₀ := one_mul C₀

/-! ## Section 3: Symbolic Verification vs. Empirical Grid-Search FLOP Savings -/

/-- **Empirical Brute-Force Search Compute (`empiricalSearchCompute`)**:
Total GPU compute spent when empirically training all `M_cand` candidate architectures/optimizers
at training cost `C_train` each: `(M_cand : ℝ) * C_train`. -/
def empiricalSearchCompute (M_cand : ℕ) (C_train : ℝ) : ℝ :=
  (M_cand : ℝ) * C_train

/-- **Lean Symbolic Pre-Verification Search Compute (`symbolicSearchCompute`)**:
Total compute spent when symbolically verifying all `M_cand` candidates in Lean at cost
`C_verify` each, filtering out the `M_cand - 1` suboptimal/divergent candidates at zero GPU
training cost, and training only the `1` verified winner on GPUs:
`(M_cand : ℝ) * C_verify + C_train`. -/
def symbolicSearchCompute (M_cand : ℕ) (C_verify C_train : ℝ) : ℝ :=
  (M_cand : ℝ) * C_verify + C_train

/-- **Symbolic Verification vs. Empirical Grid-Search FLOP Savings
(`symbolic_vs_empirical_search_flop_savings`)**:
For `M_cand ≥ 2` candidate architectures/optimizers per generation and `C_train > 0`:
1. **Exact FLOP Savings Identity**:
   `empiricalSearchCompute M_cand C_train - symbolicSearchCompute M_cand C_verify C_train = ((M_cand : ℝ) - 1) * C_train - (M_cand : ℝ) * C_verify`.
2. **Strict Compute Reduction Criterion**:
   Whenever `C_verify < (((M_cand : ℝ) - 1) / (M_cand : ℝ)) * C_train`, Lean symbolic
   pre-verification strictly reduces total search compute:
   `symbolicSearchCompute M_cand C_verify C_train < empiricalSearchCompute M_cand C_train`.
3. **Zero-GPU Symbolic Filtering Ratio (`C_verify = 0`)**:
   When symbolic verification cost is negligible (`C_verify = 0`), the single-generation compute
   ratio is `symbolicSearchCompute M_cand 0 C_train / empiricalSearchCompute M_cand C_train = 1 / (M_cand : ℝ)`,
   and the `G`-generation compound compute ratio is
   `(symbolicSearchCompute M_cand 0 C_train / empiricalSearchCompute M_cand C_train) ^ G = 1 / (M_cand : ℝ) ^ G`. -/
theorem symbolic_vs_empirical_search_flop_savings
    (M_cand G : ℕ) (C_verify C_train : ℝ)
    (hM : 2 ≤ M_cand) (hC_train : 0 < C_train)
    (hC_verify : C_verify < (((M_cand : ℝ) - 1) / (M_cand : ℝ)) * C_train) :
    empiricalSearchCompute M_cand C_train -
        symbolicSearchCompute M_cand C_verify C_train =
      ((M_cand : ℝ) - 1) * C_train - (M_cand : ℝ) * C_verify ∧
    symbolicSearchCompute M_cand C_verify C_train <
      empiricalSearchCompute M_cand C_train ∧
    symbolicSearchCompute M_cand 0 C_train /
        empiricalSearchCompute M_cand C_train =
      1 / (M_cand : ℝ) ∧
    (symbolicSearchCompute M_cand 0 C_train /
        empiricalSearchCompute M_cand C_train) ^ G =
      1 / (M_cand : ℝ) ^ G := by
  unfold empiricalSearchCompute symbolicSearchCompute
  have hM_pos_nat : 0 < M_cand := lt_of_lt_of_le (by decide : 0 < 2) hM
  have hM_pos : (0 : ℝ) < (M_cand : ℝ) := Nat.cast_pos.mpr hM_pos_nat
  have hM_ne : (M_cand : ℝ) ≠ 0 := ne_of_gt hM_pos
  have hC_ne : C_train ≠ 0 := ne_of_gt hC_train
  have hid :
      (M_cand : ℝ) * C_train - ((M_cand : ℝ) * C_verify + C_train) =
        ((M_cand : ℝ) - 1) * C_train - (M_cand : ℝ) * C_verify := by
    ring
  have hstrict :
      (M_cand : ℝ) * C_verify + C_train < (M_cand : ℝ) * C_train := by
    have hmul :
        (M_cand : ℝ) * C_verify <
          (M_cand : ℝ) * ((((M_cand : ℝ) - 1) / (M_cand : ℝ)) * C_train) :=
      mul_lt_mul_of_pos_left hC_verify hM_pos
    have hsimp :
        (M_cand : ℝ) * ((((M_cand : ℝ) - 1) / (M_cand : ℝ)) * C_train) =
          ((M_cand : ℝ) - 1) * C_train := by
      field_simp [hM_ne]
    rw [hsimp] at hmul
    linarith
  have hratio :
      ((M_cand : ℝ) * 0 + C_train) / ((M_cand : ℝ) * C_train) =
        1 / (M_cand : ℝ) := by
    field_simp [hM_ne, hC_ne]
    ring
  have hratio_G :
      (((M_cand : ℝ) * 0 + C_train) / ((M_cand : ℝ) * C_train)) ^ G =
        1 / (M_cand : ℝ) ^ G := by
    rw [hratio, one_div_pow]
  exact ⟨hid, hstrict, hratio, hratio_G⟩

end OpenTheory.VerifiableSelfImprovement
