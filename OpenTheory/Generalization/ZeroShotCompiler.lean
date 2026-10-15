import OpenTheory.Generalization.PredictiveModelPrescription

/-!
# Pre-Training Zero-Shot Architecture & Compute Compiler from Noisy Empirical Data Observables

Before spending $10^{24}$ FLOPs on full-scale pre-training, a practitioner runs a cheap probe on a
small data sample to estimate:
1. The empirical circuit-pruning / structural approximation residual
   $\hat{\varepsilon}_{\mathrm{prune}} \in [\varepsilon_{\mathrm{prune}} - \delta_\varepsilon, \varepsilon_{\mathrm{prune}} + \delta_\varepsilon]$,
2. The empirical architecture-to-optimizer scaling ratio
   $\hat{\rho} = \hat{A} / \hat{B}_{\mathrm{opt}} > 0$ (approximating the true ratio $\rho = A / B_{\mathrm{opt}}$),
and compiles a **certified architecture choice and compute-optimal `(N, D)` allocation**:

1. **Certified Zero-Shot Architecture Selection Under Empirical Estimation Noise
   (`certifiedScarceThreshold`, `certifiedAbundantThreshold`,
   `zero_shot_certified_threshold_brackets_oracle`,
   `zero_shot_architecture_compiler_soundness`,
   `zero_shot_plug_in_architecture_regret_le_two_delta`)**:
   - Brackets the unobserved oracle sample threshold `criticalSampleThreshold s m r c_gen ε_prune`
     between two observable confidence thresholds:
     `certifiedScarceThreshold ≤ criticalSampleThreshold ≤ certifiedAbundantThreshold`.
   - Proves that whenever `n < certifiedScarceThreshold`, pruned `SESC` (`s * r`) is **certifiably
     strictly optimal** over `SubspaceNet` (`m * r`) on the true unobserved population risk, and
     whenever `n > certifiedAbundantThreshold`, `SubspaceNet` is **certifiably strictly optimal**
     over `SESC`.
   - Proves that even inside the uncertain boundary zone
     `n ∈ [certifiedScarceThreshold, certifiedAbundantThreshold]`, selecting an architecture via the
     plug-in threshold `criticalSampleThreshold s m r c_gen ε_hat` incurs worst-case true risk
     regret at most `2 * horizonAmp * δ_ε`.

2. **Plug-In Chinchilla Compute Compiler & Quadratic Allocation Regret
   (`compiledCapacity`, `compiledData`, `compiledAllocation_pos`,
   `compiledCapacity_mul_compiledData_eq_compute`,
   `compiledAllocation_eq_optimal_of_exact_ratio`,
   `compiled_chinchilla_exact_regret`)**:
   - Compiles the model capacity `compiledCapacity ρ_hat C = √(ρ_hat * C)` and data budget
     `compiledData ρ_hat C = √(C / ρ_hat)` from any positive scaling ratio estimate `ρ_hat > 0`.
   - Proves exact budget feasibility `compiledCapacity ρ_hat C * compiledData ρ_hat C = C` and ratio
     consistency `compiledCapacity ρ_hat C / compiledData ρ_hat C = ρ_hat`.
   - Proves the exact completion-of-squares regret identity for the compiled Chinchilla allocation
     against the true unobserved Pareto frontier `paretoComputeRisk R_inf A B_opt C`, showing that
     regret is quadratic in the ratio estimation error and vanishes when `ρ_hat = A / B_opt`.

3. **End-to-End Zero-Shot Compiler Specification & Verification
   (`ZeroShotCompiledSpec`, `compileZeroShotSpec`,
   `zero_shot_compiler_end_to_end_guarantee`)**:
   - Bundles the compiled thresholds, plug-in architecture choice, budget-feasible `(N, D)`
     allocation, and end-to-end co-design FLOP saving factor
     `κ_save = ((s * r) / (m * d)) * (B_RPM / B_SGD) < 1` into `ZeroShotCompiledSpec` and proves the
     complete end-to-end verification theorem `zero_shot_compiler_end_to_end_guarantee`.
-/

noncomputable section

namespace OpenTheory.ZeroShotCompiler

open OpenTheory
open OpenTheory.ComputeOptimalSynthesis
open OpenTheory.SpectralSparseCircuit
open OpenTheory.ResidualPreconditionedMomentum
open OpenTheory.PredictiveModelPrescription

/-! ## 1. Certified Zero-Shot Architecture Selection Under Empirical Estimation Noise -/

/-- **Conservative Lower Certified Sample Threshold (`certifiedScarceThreshold`)**:
Given an empirical pruning residual estimate `ε_hat` with estimation noise bound `δ_ε ≥ 0`,
`certifiedScarceThreshold s m r c_gen ε_hat δ_ε = ((dimSubspace m r : ℝ) - (dimSESC s r : ℝ)) * c_gen / (ε_hat + δ_ε)`
is the worst-case lower bound on the oracle crossover sample size `criticalSampleThreshold`. -/
def certifiedScarceThreshold (s m r : ℕ) (c_gen ε_hat δ_ε : ℝ) : ℝ :=
  ((dimSubspace m r : ℝ) - (dimSESC s r : ℝ)) * c_gen / (ε_hat + δ_ε)

/-- **Conservative Upper Certified Sample Threshold (`certifiedAbundantThreshold`)**:
Given an empirical pruning residual estimate `ε_hat` with estimation noise bound `δ_ε < ε_hat`,
`certifiedAbundantThreshold s m r c_gen ε_hat δ_ε = ((dimSubspace m r : ℝ) - (dimSESC s r : ℝ)) * c_gen / (ε_hat - δ_ε)`
is the worst-case upper bound on the oracle crossover sample size `criticalSampleThreshold`. -/
def certifiedAbundantThreshold (s m r : ℕ) (c_gen ε_hat δ_ε : ℝ) : ℝ :=
  ((dimSubspace m r : ℝ) - (dimSESC s r : ℝ)) * c_gen / (ε_hat - δ_ε)

/-- **Certified Thresholds Bracket the Unobserved Oracle Threshold
(`zero_shot_certified_threshold_brackets_oracle`)**:
Whenever `s < m`, `0 < r`, `0 < c_gen`, `0 < ε_prune`, `0 ≤ δ_ε`, `δ_ε < ε_hat`, and the empirical
probe satisfies `|ε_hat - ε_prune| ≤ δ_ε`, the observable thresholds `certifiedScarceThreshold` and
`certifiedAbundantThreshold` are positive and bracket the true oracle sample threshold:
`0 < certifiedScarceThreshold ≤ criticalSampleThreshold ≤ certifiedAbundantThreshold`. -/
theorem zero_shot_certified_threshold_brackets_oracle
    {s m r : ℕ} {c_gen ε_prune ε_hat δ_ε : ℝ}
    (hsm : s < m) (hr : 0 < r) (hc : 0 < c_gen) (hε : 0 < ε_prune)
    (hδ_nonneg : 0 ≤ δ_ε) (hδ_lt : δ_ε < ε_hat)
    (h_est : |ε_hat - ε_prune| ≤ δ_ε) :
    0 < certifiedScarceThreshold s m r c_gen ε_hat δ_ε ∧
    certifiedScarceThreshold s m r c_gen ε_hat δ_ε ≤
      criticalSampleThreshold s m r c_gen ε_prune ∧
    criticalSampleThreshold s m r c_gen ε_prune ≤
      certifiedAbundantThreshold s m r c_gen ε_hat δ_ε := by
  unfold certifiedScarceThreshold certifiedAbundantThreshold criticalSampleThreshold
  have hdim_lt : dimSESC s r < dimSubspace m r := Nat.mul_lt_mul_of_pos_right hsm hr
  have hp_diff_pos : 0 < (dimSubspace m r : ℝ) - (dimSESC s r : ℝ) :=
    sub_pos.mpr (Nat.cast_lt.mpr hdim_lt)
  have hnum_pos : 0 < ((dimSubspace m r : ℝ) - (dimSESC s r : ℝ)) * c_gen :=
    mul_pos hp_diff_pos hc
  have h_abs := abs_le.mp h_est
  have h_lower_eps : ε_hat - δ_ε ≤ ε_prune := by linarith [h_abs.2]
  have h_upper_eps : ε_prune ≤ ε_hat + δ_ε := by linarith [h_abs.1]
  have h_sub_pos : 0 < ε_hat - δ_ε := sub_pos.mpr hδ_lt
  have h_add_pos : 0 < ε_hat + δ_ε := by linarith
  refine ⟨div_pos hnum_pos h_add_pos, ?_, ?_⟩
  · exact div_le_div_of_nonneg_left hnum_pos.le hε h_upper_eps
  · exact div_le_div_of_nonneg_left hnum_pos.le h_sub_pos h_lower_eps

/-- **Zero-Shot Certified Architecture Compiler Soundness
(`zero_shot_architecture_compiler_soundness`)**:
Under empirical estimation noise `|ε_hat - ε_prune| ≤ δ_ε` (`δ_ε < ε_hat`), the zero-shot compiler
provides **provably sound architecture selection** on the true unobserved population risk:
1. **Certified Data-Scarce Regime (`n < certifiedScarceThreshold`)**: Pruned `SESC` (`s * r`)
   certifiably achieves strictly smaller true `predictiveTaskRisk` than `SubspaceNet` (`m * r`).
2. **Certified Data-Abundant Regime (`n > certifiedAbundantThreshold`)**: Unpruned `SubspaceNet`
   (`m * r`) certifiably achieves strictly smaller true `predictiveTaskRisk` than `SESC` (`s * r`). -/
theorem zero_shot_architecture_compiler_soundness
    {s m r : ℕ} {c_gen ε_prune ε_hat δ_ε n R_opt horizonAmp Δ_OOD_sq : ℝ}
    (hsm : s < m) (hr : 0 < r) (hc : 0 < c_gen) (hε : 0 < ε_prune)
    (hδ_nonneg : 0 ≤ δ_ε) (hδ_lt : δ_ε < ε_hat)
    (h_est : |ε_hat - ε_prune| ≤ δ_ε)
    (hn : 0 < n) (hamp : 0 < horizonAmp) :
    (n < certifiedScarceThreshold s m r c_gen ε_hat δ_ε →
      predictiveTaskRisk ε_prune (dimSESC s r : ℝ) c_gen n R_opt horizonAmp 0 Δ_OOD_sq <
        predictiveTaskRisk 0 (dimSubspace m r : ℝ) c_gen n R_opt horizonAmp 0 Δ_OOD_sq) ∧
    (certifiedAbundantThreshold s m r c_gen ε_hat δ_ε < n →
      predictiveTaskRisk 0 (dimSubspace m r : ℝ) c_gen n R_opt horizonAmp 0 Δ_OOD_sq <
        predictiveTaskRisk ε_prune (dimSESC s r : ℝ) c_gen n R_opt horizonAmp 0 Δ_OOD_sq) := by
  obtain ⟨_, h_scarce_le, h_abundant_ge⟩ :=
    zero_shot_certified_threshold_brackets_oracle hsm hr hc hε hδ_nonneg hδ_lt h_est
  have hadapt := adaptive_architecture_selection_by_sample_size hsm hr hc hε hn hamp
    (R_opt := R_opt) (Δ_OOD_sq := Δ_OOD_sq)
  refine ⟨fun hn_lt => ?_, fun hn_gt => ?_⟩
  · exact hadapt.2.1.mp (lt_of_lt_of_le hn_lt h_scarce_le)
  · exact hadapt.2.2.2.mp (lt_of_le_of_lt h_abundant_ge hn_gt)

/-- **Plug-In Architecture Selection Worst-Case Regret Bound
(`zero_shot_plug_in_architecture_regret_le_two_delta`)**:
Even inside the uncertain boundary zone `n ∈ [certifiedScarceThreshold, certifiedAbundantThreshold]`,
if the compiler selects between `SESC` and `SubspaceNet` using the empirical plug-in threshold
`n_hat = criticalSampleThreshold s m r c_gen ε_hat`, the true population risk regret of the selected
architecture relative to the better of the two architectures is bounded by `2 * horizonAmp * δ_ε`
(and in fact by `horizonAmp * δ_ε` against the alternative):
1. When `n ≤ n_hat` (plug-in compiler selects `SESC`), the true excess risk of `SESC` over
   `SubspaceNet` is at most `horizonAmp * δ_ε ≤ 2 * horizonAmp * δ_ε`.
2. When `n_hat ≤ n` (plug-in compiler selects `SubspaceNet`), the true excess risk of `SubspaceNet`
   over `SESC` is at most `horizonAmp * δ_ε ≤ 2 * horizonAmp * δ_ε`. -/
theorem zero_shot_plug_in_architecture_regret_le_two_delta
    {s m r : ℕ} {c_gen ε_prune ε_hat δ_ε n R_opt horizonAmp Δ_OOD_sq : ℝ}
    (hsm : s < m) (hr : 0 < r) (hc : 0 < c_gen)
    (hε_hat : 0 < ε_hat) (hδ_nonneg : 0 ≤ δ_ε)
    (h_est : |ε_hat - ε_prune| ≤ δ_ε)
    (hn : 0 < n) (hamp : 0 < horizonAmp) :
    let n_hat := criticalSampleThreshold s m r c_gen ε_hat
    let risk_SESC := predictiveTaskRisk ε_prune (dimSESC s r : ℝ) c_gen n R_opt horizonAmp 0 Δ_OOD_sq
    let risk_Sub := predictiveTaskRisk 0 (dimSubspace m r : ℝ) c_gen n R_opt horizonAmp 0 Δ_OOD_sq
    (n ≤ n_hat → risk_SESC - risk_Sub ≤ 2 * horizonAmp * δ_ε) ∧
    (n_hat ≤ n → risk_Sub - risk_SESC ≤ 2 * horizonAmp * δ_ε) := by
  intro n_hat risk_SESC risk_Sub
  have h_abs := abs_le.mp h_est
  have hdim_lt : dimSESC s r < dimSubspace m r := Nat.mul_lt_mul_of_pos_right hsm hr
  have hp_diff_pos : 0 < (dimSubspace m r : ℝ) - (dimSESC s r : ℝ) :=
    sub_pos.mpr (Nat.cast_lt.mpr hdim_lt)
  have hnum_pos : 0 < ((dimSubspace m r : ℝ) - (dimSESC s r : ℝ)) * c_gen :=
    mul_pos hp_diff_pos hc
  have hn_ne : n ≠ 0 := ne_of_gt hn
  have hε_hat_ne : ε_hat ≠ 0 := ne_of_gt hε_hat
  have h_plug_id :
      predictiveTaskRisk 0 (dimSubspace m r : ℝ) c_gen n R_opt horizonAmp 0 Δ_OOD_sq -
        predictiveTaskRisk ε_hat (dimSESC s r : ℝ) c_gen n R_opt horizonAmp 0 Δ_OOD_sq =
      (horizonAmp * ε_hat / n) * (n_hat - n) := by
    unfold predictiveTaskRisk n_hat criticalSampleThreshold
    field_simp [hn_ne, hε_hat_ne]
    ring
  have h_true_vs_hat :
      risk_SESC - predictiveTaskRisk ε_hat (dimSESC s r : ℝ) c_gen n R_opt horizonAmp 0 Δ_OOD_sq =
        horizonAmp * (ε_prune - ε_hat) := by
    unfold risk_SESC predictiveTaskRisk
    ring
  have hcoef_pos : 0 < horizonAmp * ε_hat / n := div_pos (mul_pos hamp hε_hat) hn
  refine ⟨fun h_le => ?_, fun h_ge => ?_⟩
  · have h_plug_nonneg : 0 ≤ (horizonAmp * ε_hat / n) * (n_hat - n) :=
      mul_nonneg hcoef_pos.le (sub_nonneg.mpr h_le)
    have h_hat_le_sub :
        predictiveTaskRisk ε_hat (dimSESC s r : ℝ) c_gen n R_opt horizonAmp 0 Δ_OOD_sq ≤ risk_Sub := by
      unfold risk_Sub
      linarith
    have h_err_le : ε_prune - ε_hat ≤ δ_ε := by linarith [h_abs.1]
    have h_amp_err : horizonAmp * (ε_prune - ε_hat) ≤ horizonAmp * δ_ε :=
      mul_le_mul_of_nonneg_left h_err_le hamp.le
    have h_two : horizonAmp * δ_ε ≤ 2 * horizonAmp * δ_ε := by nlinarith
    linarith
  · have h_plug_nonpos : (horizonAmp * ε_hat / n) * (n_hat - n) ≤ 0 :=
      mul_nonpos_of_nonneg_of_nonpos hcoef_pos.le (sub_nonpos.mpr h_ge)
    have h_sub_le_hat :
        risk_Sub ≤ predictiveTaskRisk ε_hat (dimSESC s r : ℝ) c_gen n R_opt horizonAmp 0 Δ_OOD_sq := by
      unfold risk_Sub
      linarith
    have h_err_le : ε_hat - ε_prune ≤ δ_ε := h_abs.2
    have h_amp_err : horizonAmp * (ε_hat - ε_prune) ≤ horizonAmp * δ_ε :=
      mul_le_mul_of_nonneg_left h_err_le hamp.le
    have h_two : horizonAmp * δ_ε ≤ 2 * horizonAmp * δ_ε := by nlinarith
    linarith

/-! ## 2. Plug-In Chinchilla Compute Compiler & Quadratic Allocation Regret -/

/-- **Compiled Model Capacity (`compiledCapacity`)**:
Given an empirical scaling-ratio estimate `ρ_hat = A_hat / B_hat > 0` and total compute budget `C`,
the zero-shot compiler allocates model capacity `compiledCapacity ρ_hat C = √(ρ_hat * C)`. -/
def compiledCapacity (ρ_hat C : ℝ) : ℝ :=
  Real.sqrt (ρ_hat * C)

/-- **Compiled Data / Token Budget (`compiledData`)**:
Given an empirical scaling-ratio estimate `ρ_hat = A_hat / B_hat > 0` and total compute budget `C`,
the zero-shot compiler allocates data budget `compiledData ρ_hat C = √(C / ρ_hat)`. -/
def compiledData (ρ_hat C : ℝ) : ℝ :=
  Real.sqrt (C / ρ_hat)

/-- Strict positivity of `compiledCapacity` and `compiledData` when `0 < ρ_hat` and `0 < C`. -/
theorem compiledAllocation_pos {ρ_hat C : ℝ} (hρ : 0 < ρ_hat) (hC : 0 < C) :
    0 < compiledCapacity ρ_hat C ∧ 0 < compiledData ρ_hat C := by
  unfold compiledCapacity compiledData
  exact ⟨Real.sqrt_pos.mpr (mul_pos hρ hC), Real.sqrt_pos.mpr (div_pos hC hρ)⟩

/-- **Exact Budget Feasibility and Ratio Consistency of the Compiled Allocation
(`compiledCapacity_mul_compiledData_eq_compute`)**:
For any `0 < ρ_hat` and `0 < C`, the compiled allocation satisfies the exact FLOP budget constraint
`compiledCapacity ρ_hat C * compiledData ρ_hat C = C` and achieves the exact target capacity-to-data
ratio `compiledCapacity ρ_hat C / compiledData ρ_hat C = ρ_hat`. -/
theorem compiledCapacity_mul_compiledData_eq_compute {ρ_hat C : ℝ}
    (hρ : 0 < ρ_hat) (hC : 0 < C) :
    compiledCapacity ρ_hat C * compiledData ρ_hat C = C ∧
      compiledCapacity ρ_hat C / compiledData ρ_hat C = ρ_hat := by
  unfold compiledCapacity compiledData
  have hρ_ne : ρ_hat ≠ 0 := ne_of_gt hρ
  have hC_ne : C ≠ 0 := ne_of_gt hC
  have harg1 : 0 ≤ ρ_hat * C := by positivity
  refine ⟨?_, ?_⟩
  · rw [← Real.sqrt_mul harg1]
    have hprod : (ρ_hat * C) * (C / ρ_hat) = C ^ 2 := by
      field_simp [hρ_ne]
    rw [hprod, Real.sqrt_sq hC.le]
  · rw [← Real.sqrt_div harg1]
    have hquot : (ρ_hat * C) / (C / ρ_hat) = ρ_hat ^ 2 := by
      field_simp [hρ_ne, hC_ne]
    rw [hquot, Real.sqrt_sq hρ.le]

/-- When the empirical ratio equals the true scaling ratio `ρ_hat = A / B_opt`, the compiled
allocation `(compiledCapacity (A / B_opt) C, compiledData (A / B_opt) C)` coincides identically
with the oracle Chinchilla allocation `(optimalModelCapacity A B_opt C, optimalDataBudget A B_opt C)`. -/
theorem compiledAllocation_eq_optimal_of_exact_ratio {A B_opt C : ℝ}
    (hA : 0 < A) (hB : 0 < B_opt) :
    compiledCapacity (A / B_opt) C = optimalModelCapacity A B_opt C ∧
      compiledData (A / B_opt) C = optimalDataBudget A B_opt C := by
  unfold compiledCapacity compiledData optimalModelCapacity optimalDataBudget
  have hA_ne : A ≠ 0 := ne_of_gt hA
  have hB_ne : B_opt ≠ 0 := ne_of_gt hB
  refine ⟨congrArg Real.sqrt ?_, congrArg Real.sqrt ?_⟩
  · ring
  · field_simp [hA_ne, hB_ne]

/-- **Exact Plug-In Chinchilla Regret Identity (`compiled_chinchilla_exact_regret`)**:
For any empirical ratio `0 < ρ_hat`, true scaling constants `0 < A`, `0 < B_opt`, and compute budget
`0 < C`, the excess risk of the compiled allocation `(compiledCapacity ρ_hat C, compiledData ρ_hat C)`
above the true Pareto frontier `paretoComputeRisk R_inf A B_opt C` is the exact non-negative square:
1. `scalingRisk R_inf A B_opt (compiledCapacity ρ_hat C) (compiledData ρ_hat C) - paretoComputeRisk R_inf A B_opt C = (√(A * compiledData ρ_hat C) - √(B_opt * compiledCapacity ρ_hat C))^2 / C`,
2. `0 ≤ scalingRisk R_inf A B_opt (compiledCapacity ρ_hat C) (compiledData ρ_hat C) - paretoComputeRisk R_inf A B_opt C`,
3. When `ρ_hat = A / B_opt`, the compiled allocation achieves zero regret (`= paretoComputeRisk R_inf A B_opt C`). -/
theorem compiled_chinchilla_exact_regret (R_inf : ℝ) {ρ_hat A B_opt C : ℝ}
    (hρ : 0 < ρ_hat) (hA : 0 < A) (hB : 0 < B_opt) (hC : 0 < C) :
    let N_comp := compiledCapacity ρ_hat C
    let D_comp := compiledData ρ_hat C
    scalingRisk R_inf A B_opt N_comp D_comp - paretoComputeRisk R_inf A B_opt C =
      (Real.sqrt (A * D_comp) - Real.sqrt (B_opt * N_comp)) ^ 2 / C ∧
    0 ≤ scalingRisk R_inf A B_opt N_comp D_comp - paretoComputeRisk R_inf A B_opt C ∧
    scalingRisk R_inf A B_opt
        (compiledCapacity (A / B_opt) C) (compiledData (A / B_opt) C) =
      paretoComputeRisk R_inf A B_opt C := by
  intro N_comp D_comp
  obtain ⟨hN_pos, hD_pos⟩ := compiledAllocation_pos hρ hC
  have hbudget : N_comp * D_comp = C :=
    (compiledCapacity_mul_compiledData_eq_compute hρ hC).1
  have h_sq := chinchilla_completion_of_squares R_inf hA.le hB.le hN_pos hD_pos hC hbudget
  have h_nonneg : 0 ≤ (Real.sqrt (A * D_comp) - Real.sqrt (B_opt * N_comp)) ^ 2 / C := by
    positivity
  obtain ⟨hcap_eq, hdata_eq⟩ := compiledAllocation_eq_optimal_of_exact_ratio hA hB (C := C)
  have h_exact :
      scalingRisk R_inf A B_opt
          (compiledCapacity (A / B_opt) C) (compiledData (A / B_opt) C) =
        paretoComputeRisk R_inf A B_opt C := by
    rw [hcap_eq, hdata_eq]
    exact scalingRisk_at_optimal_eq_pareto R_inf hA hB hC
  exact ⟨h_sq, by linarith, h_exact⟩

/-! ## 3. End-to-End Zero-Shot Compiler Specification & Verification (`ZeroShotCompiledSpec`) -/

/-- **End-to-End Zero-Shot Compiled Specification (`ZeroShotCompiledSpec`)**:
Bundles the complete output of the pre-training zero-shot architecture & compute compiler from
empirical probe observables `(ε_hat, δ_ε, ρ_hat)` at sample size `n` and compute budget `C`:
- `n_scarce_cert`: certified data-scarce sample threshold `certifiedScarceThreshold`
- `n_abundant_cert`: certified data-abundant sample threshold `certifiedAbundantThreshold`
- `n_plug_in`: empirical plug-in sample crossover threshold `criticalSampleThreshold s m r c_gen ε_hat`
- `useSparseCircuit`: boolean decision (`n ≤ n_plug_in`) selecting pruned `SESC` vs. `SubspaceNet`
- `p_selected`: effective parameter dimension of the compiled architecture (`s * r` or `m * r`)
- `N_compiled`: compiled Chinchilla model capacity `compiledCapacity ρ_hat C`
- `D_compiled`: compiled Chinchilla data budget `compiledData ρ_hat C`
- `κ_flop_save`: end-to-end co-design FLOP saving factor `((s * r) / (m * d)) * (B_RPM / B_SGD)` -/
structure ZeroShotCompiledSpec where
  n_scarce_cert : ℝ
  n_abundant_cert : ℝ
  n_plug_in : ℝ
  useSparseCircuit : Bool
  p_selected : ℕ
  N_compiled : ℝ
  D_compiled : ℝ
  κ_flop_save : ℝ

/-- **Zero-Shot Compiler Constructor (`compileZeroShotSpec`)**:
Compiles task symmetry dimensions `(s, m, r, d)`, generalization constant `c_gen`, empirical probe
observables `(ε_hat, δ_ε, ρ_hat)`, optimizer ratio `(B_RPM, B_SGD)`, and budgets `(n, C)` into a
certified `ZeroShotCompiledSpec`. -/
def compileZeroShotSpec (s m r d : ℕ) (c_gen ε_hat δ_ε ρ_hat B_RPM B_SGD n C : ℝ) :
    ZeroShotCompiledSpec :=
  let n_plug := criticalSampleThreshold s m r c_gen ε_hat
  let useSESC := decide (n ≤ n_plug)
  { n_scarce_cert := certifiedScarceThreshold s m r c_gen ε_hat δ_ε
    n_abundant_cert := certifiedAbundantThreshold s m r c_gen ε_hat δ_ε
    n_plug_in := n_plug
    useSparseCircuit := useSESC
    p_selected := if useSESC then dimSESC s r else dimSubspace m r
    N_compiled := compiledCapacity ρ_hat C
    D_compiled := compiledData ρ_hat C
    κ_flop_save := ((dimSESC s r : ℝ) / (dimMLP m d : ℝ)) * (B_RPM / B_SGD) }

/-- **Master End-to-End Zero-Shot Compiler Guarantee (`zero_shot_compiler_end_to_end_guarantee`)**:
Proves that for any task with `0 < s < m`, `0 < r < k < d`, empirical pruning probe
`|ε_hat - ε_prune| ≤ δ_ε` (`0 ≤ δ_ε < ε_hat`), empirical scaling ratio `0 < ρ_hat`, and two-phase
optimizer improvement `0 < B_RPM < B_SGD`, the compiled specification
`spec = compileZeroShotSpec s m r d c_gen ε_hat δ_ε ρ_hat B_RPM B_SGD n C` simultaneously guarantees:
1. **Oracle Threshold Bracketing**: `spec.n_scarce_cert ≤ criticalSampleThreshold s m r c_gen ε_prune ≤ spec.n_abundant_cert`.
2. **Certified Architecture Optimality Outside Boundary Zone**:
   - If `n < spec.n_scarce_cert`, pruned `SESC` strictly beats `SubspaceNet` on the true risk.
   - If `spec.n_abundant_cert < n`, `SubspaceNet` strictly beats `SESC` on the true risk.
3. **Exact Compute Budget Feasibility & Quadratic Chinchilla Regret**:
   `spec.N_compiled * spec.D_compiled = C`, `spec.N_compiled / spec.D_compiled = ρ_hat`, and
   `scalingRisk R_inf A B_opt spec.N_compiled spec.D_compiled - paretoComputeRisk R_inf A B_opt C = (√(A * spec.D_compiled) - √(B_opt * spec.N_compiled))^2 / C ≥ 0`.
4. **Strict End-to-End Co-Design FLOP Compression**:
   `0 < spec.κ_flop_save ∧ spec.κ_flop_save < 1`, and at compute budget `spec.κ_flop_save * C`,
   the `(SESC, RPM)` Pareto frontier matches the `(MLP, SGD)` Pareto frontier at full budget `C`. -/
theorem zero_shot_compiler_end_to_end_guarantee (R_inf : ℝ)
    {s m r k d : ℕ}
    {c_gen c_arch ε_prune ε_hat δ_ε ρ_hat A B_opt B_RPM B_SGD n C R_opt horizonAmp Δ_OOD_sq : ℝ}
    (hs : 0 < s) (hsm : s < m) (hr : 0 < r) (hrk : r < k) (hkd : k < d)
    (hc_gen : 0 < c_gen) (hc_arch : 0 < c_arch) (hε : 0 < ε_prune)
    (hδ_nonneg : 0 ≤ δ_ε) (hδ_lt : δ_ε < ε_hat) (h_est : |ε_hat - ε_prune| ≤ δ_ε)
    (hρ : 0 < ρ_hat) (hA : 0 < A) (hB : 0 < B_opt)
    (hB_RPM : 0 < B_RPM) (hB_lt : B_RPM < B_SGD)
    (hn : 0 < n) (hC : 0 < C) (hamp : 0 < horizonAmp) :
    let spec := compileZeroShotSpec s m r d c_gen ε_hat δ_ε ρ_hat B_RPM B_SGD n C
    let n_crit := criticalSampleThreshold s m r c_gen ε_prune
    (spec.n_scarce_cert ≤ n_crit ∧ n_crit ≤ spec.n_abundant_cert) ∧
    ((n < spec.n_scarce_cert →
        predictiveTaskRisk ε_prune (dimSESC s r : ℝ) c_gen n R_opt horizonAmp 0 Δ_OOD_sq <
          predictiveTaskRisk 0 (dimSubspace m r : ℝ) c_gen n R_opt horizonAmp 0 Δ_OOD_sq) ∧
      (spec.n_abundant_cert < n →
        predictiveTaskRisk 0 (dimSubspace m r : ℝ) c_gen n R_opt horizonAmp 0 Δ_OOD_sq <
          predictiveTaskRisk ε_prune (dimSESC s r : ℝ) c_gen n R_opt horizonAmp 0 Δ_OOD_sq)) ∧
    (spec.N_compiled * spec.D_compiled = C ∧
      spec.N_compiled / spec.D_compiled = ρ_hat ∧
      scalingRisk R_inf A B_opt spec.N_compiled spec.D_compiled -
          paretoComputeRisk R_inf A B_opt C =
        (Real.sqrt (A * spec.D_compiled) - Real.sqrt (B_opt * spec.N_compiled)) ^ 2 / C) ∧
    (0 < spec.κ_flop_save ∧ spec.κ_flop_save < 1 ∧
      paretoComputeRisk R_inf (c_arch * (dimSESC s r : ℝ)) B_RPM (spec.κ_flop_save * C) =
        paretoComputeRisk R_inf (c_arch * (dimMLP m d : ℝ)) B_SGD C) := by
  intro spec n_crit
  have hbracket :=
    zero_shot_certified_threshold_brackets_oracle hsm hr hc_gen hε hδ_nonneg hδ_lt h_est
  have hsound :=
    zero_shot_architecture_compiler_soundness hsm hr hc_gen hε hδ_nonneg hδ_lt h_est hn hamp
      (R_opt := R_opt) (Δ_OOD_sq := Δ_OOD_sq)
  have hbudget := compiledCapacity_mul_compiledData_eq_compute hρ hC
  have hregret := compiled_chinchilla_exact_regret R_inf hρ hA hB hC
  obtain ⟨hdim1, hdim2, hdim3⟩ := four_tier_dimension_strict_hierarchy hs hsm hr hrk hkd
  have hdim_lt : dimSESC s r < dimMLP m d := (hdim1.trans hdim2).trans hdim3
  have hp_SESC_pos : (0 : ℝ) < (dimSESC s r : ℝ) := Nat.cast_pos.mpr (Nat.mul_pos hs hr)
  have hp_lt : (dimSESC s r : ℝ) < (dimMLP m d : ℝ) := Nat.cast_lt.mpr hdim_lt
  have hp_MLP_pos : (0 : ℝ) < (dimMLP m d : ℝ) := hp_SESC_pos.trans hp_lt
  have hB_SGD_pos : 0 < B_SGD := hB_RPM.trans hB_lt
  have hfac_p_pos : 0 < (dimSESC s r : ℝ) / (dimMLP m d : ℝ) := div_pos hp_SESC_pos hp_MLP_pos
  have hfac_B_pos : 0 < B_RPM / B_SGD := div_pos hB_RPM hB_SGD_pos
  have hfac_p_lt1 : (dimSESC s r : ℝ) / (dimMLP m d : ℝ) < 1 := (div_lt_one hp_MLP_pos).mpr hp_lt
  have hfac_B_lt1 : B_RPM / B_SGD < 1 := (div_lt_one hB_SGD_pos).mpr hB_lt
  have hκ_pos : 0 < spec.κ_flop_save := mul_pos hfac_p_pos hfac_B_pos
  have hκ_lt1 : spec.κ_flop_save < 1 := by
    change ((dimSESC s r : ℝ) / (dimMLP m d : ℝ)) * (B_RPM / B_SGD) < 1
    calc
      ((dimSESC s r : ℝ) / (dimMLP m d : ℝ)) * (B_RPM / B_SGD)
        < 1 * (B_RPM / B_SGD) := mul_lt_mul_of_pos_right hfac_p_lt1 hfac_B_pos
      _ = B_RPM / B_SGD := one_mul _
      _ < 1 := hfac_B_lt1
  have hA_SESC_pos : 0 < c_arch * (dimSESC s r : ℝ) := mul_pos hc_arch hp_SESC_pos
  have hA_MLP_pos : 0 < c_arch * (dimMLP m d : ℝ) := mul_pos hc_arch hp_MLP_pos
  have heq_budget :
      spec.κ_flop_save * C =
        equivalentComputeBudget (c_arch * (dimSESC s r : ℝ)) (c_arch * (dimMLP m d : ℝ))
          B_RPM B_SGD C := by
    change (((dimSESC s r : ℝ) / (dimMLP m d : ℝ)) * (B_RPM / B_SGD)) * C =
      equivalentComputeBudget (c_arch * (dimSESC s r : ℝ)) (c_arch * (dimMLP m d : ℝ))
        B_RPM B_SGD C
    unfold equivalentComputeBudget
    have hc_ne : c_arch ≠ 0 := ne_of_gt hc_arch
    have hp_MLP_ne : (dimMLP m d : ℝ) ≠ 0 := ne_of_gt hp_MLP_pos
    field_simp [hc_ne, hp_MLP_ne]
  have h_pareto_eq :
      paretoComputeRisk R_inf (c_arch * (dimSESC s r : ℝ)) B_RPM (spec.κ_flop_save * C) =
        paretoComputeRisk R_inf (c_arch * (dimMLP m d : ℝ)) B_SGD C := by
    rw [heq_budget]
    exact codesign_compute_saving_factor R_inf hA_SESC_pos hA_MLP_pos hB_RPM hB_SGD_pos hC
  exact ⟨⟨hbracket.2.1, hbracket.2.2⟩, hsound, ⟨hbudget.1, hbudget.2, hregret.1⟩,
    ⟨hκ_pos, hκ_lt1, h_pareto_eq⟩⟩

end OpenTheory.ZeroShotCompiler
