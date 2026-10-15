import OpenTheory.Generalization.ComputeOptimalSynthesis
import OpenTheory.Architectures.SpectralSparseCircuit
import OpenTheory.Optimization.ResidualPreconditionedMomentum
import Mathlib.Analysis.SpecialFunctions.Exp

/-!
# Master Predictive Task-to-Model Synthesis Compiler

Given any learning task specification—ambient input dimension `d`, local receptive patch width `k`,
spectral signal rank `r`, hidden width `m`, active sparse circuit size `s`, nonlinear information
exponent `s_info`, autoregressive reasoning horizon `H`, out-of-distribution spurious shift energy
`Δ_OOD_sq`, and sample/compute budget `(n, C)`—this module proves how to prescribe the
**optimal architecture, depth/Chain-of-Thought decomposition, two-phase optimizer schedule, and
compute allocation** that strictly dominates all mismatched designs:

1. **Formal Task Specification & Model Design Space (`TaskSignature`, `ModelPrescription`,
   `predictiveTaskRisk`, `predictiveTaskRisk_nonneg`, `predictiveTaskRisk_mono`,
   `predictiveTaskRisk_strict_mono_in_each_factor`)**:
   Defines the task invariant bundle `TaskSignature`, the model prescription bundle
   `ModelPrescription`, and the unified **Total Predictive Task Risk Functional**
   `predictiveTaskRisk ε_struct p_eff c_gen n R_opt horizonAmp w_spurious Δ_OOD_sq = horizonAmp * (ε_struct + p_eff * c_gen / n + R_opt) + w_spurious * Δ_OOD_sq`,
   establishing non-negativity, joint monotonicity, and strict monotonicity in each factor.

2. **Predictive Law 1 — Optimal Architecture Selection from Task Symmetries & Sample Budget
   (`prescribed_architecture_dominance_hierarchy`, `criticalSampleThreshold`,
   `adaptive_architecture_selection_by_sample_size`)**:
   - Under structural well-specification (`ε_struct = 0`), the prescribed `SESC` architecture
     (`p_SESC = s * r`, `w_spurious = 0`) strictly dominates the `SubspaceNet` (`m * r`),
     `CNN` (`m * k`), and `Dense MLP` (`m * d`) on both in-distribution finite-sample risk and
     out-of-distribution shift robustness.
   - When the target has a weak non-sparse residual `ε_prune > 0` outside the `s * r` circuit,
     proves the exact sample-size decision boundary
     `n_crit = ((m * r - s * r) * c_gen) / ε_prune`: in the **data-scarce regime** (`n < n_crit`),
     pruned `SESC` strictly beats `SubspaceNet`; in the **data-abundant regime** (`n > n_crit`),
     expanding to `SubspaceNet` strictly beats `SESC`.

3. **Predictive Law 2 — Optimal Depth, Softmax vs. Linear Attention, and CoT Horizon Prescription
   (`compositionalReasoningRisk`, `softmaxDistractorLeakage`, `linearDistractorLeakage`,
   `hierarchical_cot_softmax_strictly_dominates_shallow_linear`)**:
   Defines `compositionalReasoningRisk H d s_deg n distractorLeakage = H * (d ^ s_deg / n + distractorLeakage)`
   and proves that decomposing a compositional degree `s_total = H * s_step` task into an `H`-step
   Chain-of-Thought with hierarchical features (`d ^ s_step`) and Softmax induction heads
   (`m_dist * exp(-Δ)`) strictly dominates a 1-step shallow architecture (`d ^ s_total`) with
   linear attention (`m_dist / (Δ + m_dist)`).

4. **The Master End-to-End Task-to-Model Synthesis Theorem (`endToEndComputeRisk`,
   `prescribedParetoEndToEndRisk`, `master_predictive_model_prescription`)**:
   Unifies all 4 axes of deep learning design—Architecture (`SESC` vs. `MLP`), Optimizer (`RPM` vs.
   `SGD`), Compute Allocation (`(N*(C), D*(C))` vs. off-frontier `(N, D)`), and Reasoning Rollout
   (contractive CoT `1 / (1 - Λ_CoT)` vs. unanchored rollout)—proving strict end-to-end dominance
   at fixed compute `C > 0` and exact compute-saving factor
   `(s * r / (m * d)) * (B_RPM / B_SGD) < 1` at matched in-distribution Pareto risk.
-/

noncomputable section

namespace OpenTheory.PredictiveModelPrescription

open OpenTheory
open OpenTheory.ComputeOptimalSynthesis
open OpenTheory.SpectralSparseCircuit
open OpenTheory.ResidualPreconditionedMomentum

/-! ## 1. Formal Task Specification & Model Design Space (`TaskSignature` & `ModelPrescription`) -/

/-- **Formal Task Signature (`TaskSignature`)**:
Captures the fundamental mathematical invariants of a learning and reasoning task:
- `d_ambient`: ambient input dimension $d$
- `k_local`: local receptive patch width $k$
- `r_spike`: spectral signal rank $r$
- `m_width`: candidate hidden channel width $m$
- `s_circuit`: active sparse circuit width $s$
- `s_info_exp`: nonlinear information exponent $s_{\mathrm{info}}$
- `H_horizon`: autoregressive reasoning horizon $H$
- `Δ_OOD_sq`: out-of-distribution spurious shift energy $\Delta_{\mathrm{OOD}}^2$ -/
structure TaskSignature where
  d_ambient : ℕ
  k_local : ℕ
  r_spike : ℕ
  m_width : ℕ
  s_circuit : ℕ
  s_info_exp : ℕ
  H_horizon : ℕ
  Δ_OOD_sq : ℝ

/-- **Model & Training Prescription (`ModelPrescription`)**:
Encapsulates the synthesized model and training hyperparameters across the four design axes:
- `p_eff`: effective invariant parameter dimension $p_{\mathrm{eff}}$
- `A_arch`: architectural scaling coefficient $A$
- `B_opt`: optimizer sample/step complexity coefficient $B_{\mathrm{opt}}$
- `N_cap`: allocated model capacity $N$
- `D_data`: allocated data/token budget $D$
- `horizonAmp`: autoregressive rollout error amplification factor
- `w_spurious`: sensitivity to out-of-distribution orthogonal bulk shifts -/
structure ModelPrescription where
  p_eff : ℝ
  A_arch : ℝ
  B_opt : ℝ
  N_cap : ℝ
  D_data : ℝ
  horizonAmp : ℝ
  w_spurious : ℝ

/-- **Total Predictive Task Risk Functional (`predictiveTaskRisk`)**:
Unifies structural approximation bias `ε_struct`, finite-sample generalization risk
`p_eff * c_gen / n`, optimization / saddle residual `R_opt`, autoregressive horizon amplification
`horizonAmp`, and out-of-distribution shift penalty `w_spurious * Δ_OOD_sq`:
`horizonAmp * (ε_struct + p_eff * c_gen / n + R_opt) + w_spurious * Δ_OOD_sq`. -/
def predictiveTaskRisk (ε_struct p_eff c_gen n R_opt horizonAmp w_spurious Δ_OOD_sq : ℝ) : ℝ :=
  horizonAmp * (ε_struct + p_eff * c_gen / n + R_opt) + w_spurious * Δ_OOD_sq

/-- Non-negativity of `predictiveTaskRisk` when all constituent risk factors and multipliers are
non-negative (`0 < n`). -/
theorem predictiveTaskRisk_nonneg
    {ε_struct p_eff c_gen n R_opt horizonAmp w_spurious Δ_OOD_sq : ℝ}
    (hε : 0 ≤ ε_struct) (hp : 0 ≤ p_eff) (hc : 0 ≤ c_gen) (hn : 0 < n)
    (hR : 0 ≤ R_opt) (hamp : 0 ≤ horizonAmp) (hw : 0 ≤ w_spurious) (hΔ : 0 ≤ Δ_OOD_sq) :
    0 ≤ predictiveTaskRisk ε_struct p_eff c_gen n R_opt horizonAmp w_spurious Δ_OOD_sq := by
  unfold predictiveTaskRisk
  have hgen : 0 ≤ p_eff * c_gen / n := div_nonneg (mul_nonneg hp hc) hn.le
  positivity

/-- Joint monotonicity of `predictiveTaskRisk` across structural bias, effective dimension,
optimization residual, horizon amplification, and OOD shift sensitivity. -/
theorem predictiveTaskRisk_mono
    {ε₁ ε₂ p₁ p₂ c_gen n R₁ R₂ amp₁ amp₂ w₁ w₂ Δ_OOD_sq : ℝ}
    (hε₁ : 0 ≤ ε₁) (hp₁ : 0 ≤ p₁) (hc : 0 ≤ c_gen) (hn : 0 < n)
    (hR₁ : 0 ≤ R₁) (hamp₁ : 0 ≤ amp₁) (hΔ : 0 ≤ Δ_OOD_sq)
    (hε : ε₁ ≤ ε₂) (hp : p₁ ≤ p₂) (hR : R₁ ≤ R₂) (hamp : amp₁ ≤ amp₂) (hw : w₁ ≤ w₂) :
    predictiveTaskRisk ε₁ p₁ c_gen n R₁ amp₁ w₁ Δ_OOD_sq ≤
      predictiveTaskRisk ε₂ p₂ c_gen n R₂ amp₂ w₂ Δ_OOD_sq := by
  unfold predictiveTaskRisk
  have hgen_le : p₁ * c_gen / n ≤ p₂ * c_gen / n := by
    apply div_le_div_of_nonneg_right (mul_le_mul_of_nonneg_right hp hc) hn.le
  have hinner₁_nonneg : 0 ≤ ε₁ + p₁ * c_gen / n + R₁ := by
    have : 0 ≤ p₁ * c_gen / n := div_nonneg (mul_nonneg hp₁ hc) hn.le
    linarith
  have hinner_le : ε₁ + p₁ * c_gen / n + R₁ ≤ ε₂ + p₂ * c_gen / n + R₂ := by linarith
  have hterm1 : amp₁ * (ε₁ + p₁ * c_gen / n + R₁) ≤ amp₂ * (ε₂ + p₂ * c_gen / n + R₂) :=
    mul_le_mul hamp hinner_le hinner₁_nonneg (hamp₁.trans hamp)
  have hterm2 : w₁ * Δ_OOD_sq ≤ w₂ * Δ_OOD_sq := mul_le_mul_of_nonneg_right hw hΔ
  linarith

/-- **Strict Monotonicity of `predictiveTaskRisk` in Each Design Factor
(`predictiveTaskRisk_strict_mono_in_each_factor`)**:
When `0 < horizonAmp`, `0 < c_gen`, `0 < n`, and `0 < Δ_OOD_sq`:
1. Strictly smaller effective parameter dimension (`p₁ < p₂`) strictly reduces `predictiveTaskRisk`.
2. Strictly smaller optimization residual (`R₁ < R₂`) strictly reduces `predictiveTaskRisk`.
3. Strictly smaller OOD spurious sensitivity (`w₁ < w₂`) strictly reduces `predictiveTaskRisk`. -/
theorem predictiveTaskRisk_strict_mono_in_each_factor
    {ε_struct p₁ p₂ c_gen n R_opt R₁ R₂ horizonAmp w_spurious w₁ w₂ Δ_OOD_sq : ℝ}
    (hc : 0 < c_gen) (hn : 0 < n) (hamp : 0 < horizonAmp) (hΔ : 0 < Δ_OOD_sq)
    (hp : p₁ < p₂) (hR : R₁ < R₂) (hw : w₁ < w₂) :
    predictiveTaskRisk ε_struct p₁ c_gen n R_opt horizonAmp w_spurious Δ_OOD_sq <
      predictiveTaskRisk ε_struct p₂ c_gen n R_opt horizonAmp w_spurious Δ_OOD_sq ∧
    predictiveTaskRisk ε_struct p₁ c_gen n R₁ horizonAmp w_spurious Δ_OOD_sq <
      predictiveTaskRisk ε_struct p₁ c_gen n R₂ horizonAmp w_spurious Δ_OOD_sq ∧
    predictiveTaskRisk ε_struct p₁ c_gen n R_opt horizonAmp w₁ Δ_OOD_sq <
      predictiveTaskRisk ε_struct p₁ c_gen n R_opt horizonAmp w₂ Δ_OOD_sq := by
  unfold predictiveTaskRisk
  have hgen_lt : p₁ * c_gen / n < p₂ * c_gen / n :=
    div_lt_div_of_pos_right (mul_lt_mul_of_pos_right hp hc) hn
  refine ⟨?_, ?_, ?_⟩
  · have hinner_lt : ε_struct + p₁ * c_gen / n + R_opt < ε_struct + p₂ * c_gen / n + R_opt := by
      linarith
    have hmul_lt := mul_lt_mul_of_pos_left hinner_lt hamp
    linarith
  · have hinner_lt : ε_struct + p₁ * c_gen / n + R₁ < ε_struct + p₁ * c_gen / n + R₂ := by
      linarith
    have hmul_lt := mul_lt_mul_of_pos_left hinner_lt hamp
    linarith
  · have hmul_lt := mul_lt_mul_of_pos_right hw hΔ
    linarith

/-! ## 2. Predictive Law 1 — Optimal Architecture Selection from Task Symmetries & Sample Budget -/

/-- **Predictive Law 1A: 4-Tier Prescribed Architecture Dominance Hierarchy
(`prescribed_architecture_dominance_hierarchy`)**:
Given a task with spatial locality `k < d`, spectral rank `r < k`, sparse circuit width `s < m`,
and OOD shift energy `Δ_OOD_sq > 0`:
Under structural well-specification (`ε_struct = 0`), the prescribed `SESC` architecture
(`dimSESC s r = s * r`, `w_spurious = 0`) strictly dominates:
1. `SubspaceNet` (`dimSubspace m r = m * r`, `w_spurious = 0`),
2. `CNN` (`dimCNN m k = m * k`, `w_CNN > 0`),
3. `Dense MLP` (`dimMLP m d = m * d`, `w_MLP ≥ w_CNN > 0`),
simultaneously across both in-distribution finite-sample risk (`Δ_OOD_sq = 0`) and out-of-distribution
total predictive task risk (`Δ_OOD_sq > 0`). -/
theorem prescribed_architecture_dominance_hierarchy
    {s m r k d : ℕ} {c_gen n R_opt horizonAmp w_CNN w_MLP Δ_OOD_sq : ℝ}
    (hs : 0 < s) (hsm : s < m) (hr : 0 < r) (hrk : r < k) (hkd : k < d)
    (hc : 0 < c_gen) (hn : 0 < n) (hamp : 0 < horizonAmp)
    (hw_CNN : 0 < w_CNN) (hw_MLP : w_CNN ≤ w_MLP) (hΔ : 0 < Δ_OOD_sq) :
    (predictiveTaskRisk 0 (dimSESC s r : ℝ) c_gen n R_opt horizonAmp 0 0 <
       predictiveTaskRisk 0 (dimSubspace m r : ℝ) c_gen n R_opt horizonAmp 0 0 ∧
     predictiveTaskRisk 0 (dimSubspace m r : ℝ) c_gen n R_opt horizonAmp 0 0 <
       predictiveTaskRisk 0 (dimCNN m k : ℝ) c_gen n R_opt horizonAmp 0 0 ∧
     predictiveTaskRisk 0 (dimCNN m k : ℝ) c_gen n R_opt horizonAmp 0 0 <
       predictiveTaskRisk 0 (dimMLP m d : ℝ) c_gen n R_opt horizonAmp 0 0) ∧
    (predictiveTaskRisk 0 (dimSESC s r : ℝ) c_gen n R_opt horizonAmp 0 Δ_OOD_sq <
       predictiveTaskRisk 0 (dimSubspace m r : ℝ) c_gen n R_opt horizonAmp 0 Δ_OOD_sq ∧
     predictiveTaskRisk 0 (dimSubspace m r : ℝ) c_gen n R_opt horizonAmp 0 Δ_OOD_sq <
       predictiveTaskRisk 0 (dimCNN m k : ℝ) c_gen n R_opt horizonAmp w_CNN Δ_OOD_sq ∧
     predictiveTaskRisk 0 (dimCNN m k : ℝ) c_gen n R_opt horizonAmp w_CNN Δ_OOD_sq <
       predictiveTaskRisk 0 (dimMLP m d : ℝ) c_gen n R_opt horizonAmp w_MLP Δ_OOD_sq) := by
  obtain ⟨hdim1, hdim2, hdim3⟩ := four_tier_dimension_strict_hierarchy hs hsm hr hrk hkd
  have hp1 : (dimSESC s r : ℝ) < (dimSubspace m r : ℝ) := Nat.cast_lt.mpr hdim1
  have hp2 : (dimSubspace m r : ℝ) < (dimCNN m k : ℝ) := Nat.cast_lt.mpr hdim2
  have hp3 : (dimCNN m k : ℝ) < (dimMLP m d : ℝ) := Nat.cast_lt.mpr hdim3
  have hgen1 : (dimSESC s r : ℝ) * c_gen / n < (dimSubspace m r : ℝ) * c_gen / n :=
    div_lt_div_of_pos_right (mul_lt_mul_of_pos_right hp1 hc) hn
  have hgen2 : (dimSubspace m r : ℝ) * c_gen / n < (dimCNN m k : ℝ) * c_gen / n :=
    div_lt_div_of_pos_right (mul_lt_mul_of_pos_right hp2 hc) hn
  have hgen3 : (dimCNN m k : ℝ) * c_gen / n < (dimMLP m d : ℝ) * c_gen / n :=
    div_lt_div_of_pos_right (mul_lt_mul_of_pos_right hp3 hc) hn
  unfold predictiveTaskRisk
  have h_amp1 : horizonAmp * (0 + (dimSESC s r : ℝ) * c_gen / n + R_opt) <
      horizonAmp * (0 + (dimSubspace m r : ℝ) * c_gen / n + R_opt) :=
    mul_lt_mul_of_pos_left (by linarith) hamp
  have h_amp2 : horizonAmp * (0 + (dimSubspace m r : ℝ) * c_gen / n + R_opt) <
      horizonAmp * (0 + (dimCNN m k : ℝ) * c_gen / n + R_opt) :=
    mul_lt_mul_of_pos_left (by linarith) hamp
  have h_amp3 : horizonAmp * (0 + (dimCNN m k : ℝ) * c_gen / n + R_opt) <
      horizonAmp * (0 + (dimMLP m d : ℝ) * c_gen / n + R_opt) :=
    mul_lt_mul_of_pos_left (by linarith) hamp
  have hood_CNN : 0 < w_CNN * Δ_OOD_sq := mul_pos hw_CNN hΔ
  have hood_MLP : w_CNN * Δ_OOD_sq ≤ w_MLP * Δ_OOD_sq := mul_le_mul_of_nonneg_right hw_MLP hΔ.le
  refine ⟨⟨by linarith, by linarith, by linarith⟩, ⟨by linarith, by linarith, by linarith⟩⟩

/-- **Exact Sample-Size Decision Threshold (`criticalSampleThreshold`)**:
The exact sample size $n_{\mathrm{crit}} = \frac{(m \cdot r - s \cdot r) c_{\mathrm{gen}}}{\varepsilon_{\mathrm{prune}}}$
at which the finite-sample variance savings of pruning to `SESC` (`s * r`) balances the structural
pruning residual `ε_prune > 0` of `SubspaceNet` (`m * r`). -/
def criticalSampleThreshold (s m r : ℕ) (c_gen ε_prune : ℝ) : ℝ :=
  ((dimSubspace m r : ℝ) - (dimSESC s r : ℝ)) * c_gen / ε_prune

/-- **Predictive Law 1B: Sample-Adaptive Architecture Selection Rule
(`adaptive_architecture_selection_by_sample_size`)**:
When the true target has a weak non-sparse residual `ε_prune > 0` outside the `s * r` circuit
(so pruned `SESC` incurs structural bias `ε_prune` with dimension `s * r`, while unpruned
`SubspaceNet` has zero structural bias `0` with dimension `m * r > s * r`), the critical sample
threshold `n_crit = criticalSampleThreshold s m r c_gen ε_prune` satisfies:
1. `0 < n_crit`,
2. **Data-Scarce Regime (`n < n_crit`)**: Pruned `SESC` strictly dominates `SubspaceNet`,
3. **Exact Crossover (`n = n_crit`)**: `SESC` and `SubspaceNet` achieve identical risk,
4. **Data-Abundant Regime (`n > n_crit`)**: Expanding capacity to `SubspaceNet` strictly dominates
   pruned `SESC`. -/
theorem adaptive_architecture_selection_by_sample_size
    {s m r : ℕ} {c_gen ε_prune n R_opt horizonAmp Δ_OOD_sq : ℝ}
    (hsm : s < m) (hr : 0 < r) (hc : 0 < c_gen) (hε : 0 < ε_prune)
    (hn : 0 < n) (hamp : 0 < horizonAmp) :
    let n_crit := criticalSampleThreshold s m r c_gen ε_prune
    0 < n_crit ∧
    (n < n_crit ↔
      predictiveTaskRisk ε_prune (dimSESC s r : ℝ) c_gen n R_opt horizonAmp 0 Δ_OOD_sq <
        predictiveTaskRisk 0 (dimSubspace m r : ℝ) c_gen n R_opt horizonAmp 0 Δ_OOD_sq) ∧
    (n = n_crit →
      predictiveTaskRisk ε_prune (dimSESC s r : ℝ) c_gen n R_opt horizonAmp 0 Δ_OOD_sq =
        predictiveTaskRisk 0 (dimSubspace m r : ℝ) c_gen n R_opt horizonAmp 0 Δ_OOD_sq) ∧
    (n_crit < n ↔
      predictiveTaskRisk 0 (dimSubspace m r : ℝ) c_gen n R_opt horizonAmp 0 Δ_OOD_sq <
        predictiveTaskRisk ε_prune (dimSESC s r : ℝ) c_gen n R_opt horizonAmp 0 Δ_OOD_sq) := by
  intro n_crit
  have hdim_lt : dimSESC s r < dimSubspace m r := Nat.mul_lt_mul_of_pos_right hsm hr
  have hp_diff_pos : 0 < (dimSubspace m r : ℝ) - (dimSESC s r : ℝ) :=
    sub_pos.mpr (Nat.cast_lt.mpr hdim_lt)
  have hncrit_pos : 0 < n_crit := div_pos (mul_pos hp_diff_pos hc) hε
  have hdiff_id :
      predictiveTaskRisk 0 (dimSubspace m r : ℝ) c_gen n R_opt horizonAmp 0 Δ_OOD_sq -
        predictiveTaskRisk ε_prune (dimSESC s r : ℝ) c_gen n R_opt horizonAmp 0 Δ_OOD_sq =
      (horizonAmp * ε_prune / n) * (n_crit - n) := by
    unfold predictiveTaskRisk n_crit criticalSampleThreshold
    have hn_ne : n ≠ 0 := ne_of_gt hn
    have hε_ne : ε_prune ≠ 0 := ne_of_gt hε
    field_simp [hn_ne, hε_ne]
    ring
  have hcoef_pos : 0 < horizonAmp * ε_prune / n := div_pos (mul_pos hamp hε) hn
  refine ⟨hncrit_pos, ?_, ?_, ?_⟩
  · constructor
    · intro hlt
      have hsub_pos : 0 < n_crit - n := sub_pos.mpr hlt
      have hprod_pos : 0 < (horizonAmp * ε_prune / n) * (n_crit - n) := mul_pos hcoef_pos hsub_pos
      linarith
    · intro hlt
      have hprod_pos : 0 < (horizonAmp * ε_prune / n) * (n_crit - n) := by linarith
      have hsub_pos : 0 < n_crit - n := (mul_pos_iff_of_pos_left hcoef_pos).mp hprod_pos
      linarith
  · intro heq
    have hzero : (horizonAmp * ε_prune / n) * (n_crit - n) = 0 := by
      rw [heq, sub_self, mul_zero]
    linarith
  · constructor
    · intro hgt
      have hsub_neg : n_crit - n < 0 := sub_neg.mpr hgt
      have hprod_neg : (horizonAmp * ε_prune / n) * (n_crit - n) < 0 :=
        mul_neg_of_pos_of_neg hcoef_pos hsub_neg
      linarith
    · intro hgt
      have hprod_pos : 0 < (horizonAmp * ε_prune / n) * (n - n_crit) := by linarith
      have hsub_pos : 0 < n - n_crit := (mul_pos_iff_of_pos_left hcoef_pos).mp hprod_pos
      linarith

/-! ## 3. Predictive Law 2 — Optimal Depth, Softmax vs. Linear Attention, and CoT Horizon Prescription -/

/-- **Compositional Reasoning Risk (`compositionalReasoningRisk`)**:
For an `H`-step reasoning decomposition where each step has polynomial complexity degree `s_deg`
over ambient dimension `d` at sample size `n` and per-step attention distractor leakage
`distractorLeakage`:
`compositionalReasoningRisk H d s_deg n distractorLeakage = (H : ℝ) * (d ^ s_deg / n + distractorLeakage)`. -/
def compositionalReasoningRisk (H : ℕ) (d : ℝ) (s_deg : ℕ) (n : ℝ) (distractorLeakage : ℝ) : ℝ :=
  (H : ℝ) * (d ^ s_deg / n + distractorLeakage)

/-- Softmax induction head distractor leakage across `m_dist` context distractors at logit margin `Δ`:
`softmaxDistractorLeakage m_dist Δ = m_dist * Real.exp (-Δ)`. -/
def softmaxDistractorLeakage (m_dist Δ : ℝ) : ℝ :=
  m_dist * Real.exp (-Δ)

/-- Linear / unnormalized attention distractor leakage across `m_dist` context distractors at signal
strength `Δ`: `linearDistractorLeakage m_dist Δ = m_dist / (Δ + m_dist)`. -/
def linearDistractorLeakage (m_dist Δ : ℝ) : ℝ :=
  m_dist / (Δ + m_dist)

/-- **Predictive Law 2: Hierarchical CoT + Softmax Induction Strictly Dominates Shallow + Linear
(`hierarchical_cot_softmax_strictly_dominates_shallow_linear`)**:
Comparing the **Prescribed Compositional Design** (`H`-step CoT + deep hierarchical per-step degree
`s_step` + Softmax attention leakage `softmaxDistractorLeakage m_dist Δ`) against the **Baseline
Direct Design** (`1`-step shallow degree `s_total` + Linear attention leakage
`linearDistractorLeakage m_dist Δ`):
Whenever `(H : ℝ) * d ^ s_step < d ^ s_total` (exponential sample-complexity reduction from
hierarchical CoT decomposition) and `(H : ℝ) * softmaxDistractorLeakage m_dist Δ < linearDistractorLeakage m_dist Δ`
(exponential attention sharpening beats polynomial linear-attention dilution), the prescribed
design achieves strictly smaller `compositionalReasoningRisk` at every sample size `n > 0`, and
furthermore satisfies the exact risk gap identity:
`compositionalReasoningRisk 1 d s_total n (linearDistractorLeakage m_dist Δ) - compositionalReasoningRisk H d s_step n (softmaxDistractorLeakage m_dist Δ) = (d ^ s_total - (H : ℝ) * d ^ s_step) / n + (linearDistractorLeakage m_dist Δ - (H : ℝ) * softmaxDistractorLeakage m_dist Δ)`. -/
theorem hierarchical_cot_softmax_strictly_dominates_shallow_linear
    {H s_step s_total : ℕ} {d n m_dist Δ : ℝ}
    (hn : 0 < n)
    (h_sample_comp : (H : ℝ) * d ^ s_step < d ^ s_total)
    (h_attn_leak : (H : ℝ) * softmaxDistractorLeakage m_dist Δ < linearDistractorLeakage m_dist Δ) :
    compositionalReasoningRisk H d s_step n (softmaxDistractorLeakage m_dist Δ) <
      compositionalReasoningRisk 1 d s_total n (linearDistractorLeakage m_dist Δ) ∧
    compositionalReasoningRisk 1 d s_total n (linearDistractorLeakage m_dist Δ) -
        compositionalReasoningRisk H d s_step n (softmaxDistractorLeakage m_dist Δ) =
      (d ^ s_total - (H : ℝ) * d ^ s_step) / n +
        (linearDistractorLeakage m_dist Δ - (H : ℝ) * softmaxDistractorLeakage m_dist Δ) := by
  unfold compositionalReasoningRisk
  have hgap :
      ((1 : ℕ) : ℝ) * (d ^ s_total / n + linearDistractorLeakage m_dist Δ) -
        (H : ℝ) * (d ^ s_step / n + softmaxDistractorLeakage m_dist Δ) =
      (d ^ s_total - (H : ℝ) * d ^ s_step) / n +
        (linearDistractorLeakage m_dist Δ - (H : ℝ) * softmaxDistractorLeakage m_dist Δ) := by
    push_cast
    ring
  have h1 : 0 < (d ^ s_total - (H : ℝ) * d ^ s_step) / n :=
    div_pos (sub_pos.mpr h_sample_comp) hn
  have h2 : 0 < linearDistractorLeakage m_dist Δ - (H : ℝ) * softmaxDistractorLeakage m_dist Δ :=
    sub_pos.mpr h_attn_leak
  refine ⟨by linarith, hgap⟩

/-! ## 4. The Master End-to-End Task-to-Model Synthesis Theorem (`master_predictive_model_prescription`) -/

/-- **End-to-End Compute, Reasoning, and OOD Risk Functional (`endToEndComputeRisk`)**:
Evaluates the total end-to-end risk of a design `(A, B_opt, N, D, horizonAmp, w_spurious)` with
irreducible floor `R_inf` and OOD shift energy `Δ_OOD_sq`:
`horizonAmp * scalingRisk R_inf A B_opt N D + w_spurious * Δ_OOD_sq`. -/
def endToEndComputeRisk (R_inf A B_opt N D horizonAmp w_spurious Δ_OOD_sq : ℝ) : ℝ :=
  horizonAmp * scalingRisk R_inf A B_opt N D + w_spurious * Δ_OOD_sq

/-- **Prescribed Pareto End-to-End Risk (`prescribedParetoEndToEndRisk`)**:
The end-to-end risk achieved along the Chinchilla compute-optimal Pareto frontier
`paretoComputeRisk R_inf A B_opt C` under contractive CoT amplification `1 / (1 - Λ_CoT)` and
OOD sensitivity `w_spurious`:
`(1 / (1 - Λ_CoT)) * paretoComputeRisk R_inf A B_opt C + w_spurious * Δ_OOD_sq`. -/
def prescribedParetoEndToEndRisk (R_inf A B_opt C Λ_CoT w_spurious Δ_OOD_sq : ℝ) : ℝ :=
  (1 / (1 - Λ_CoT)) * paretoComputeRisk R_inf A B_opt C + w_spurious * Δ_OOD_sq

/-- Evaluating `endToEndComputeRisk` at `(optimalModelCapacity A B_opt C, optimalDataBudget A B_opt C)`
and `horizonAmp = 1 / (1 - Λ_CoT)` recovers `prescribedParetoEndToEndRisk`. -/
theorem endToEndComputeRisk_at_optimal_eq_prescribed (R_inf : ℝ)
    {A B_opt C Λ_CoT w_spurious Δ_OOD_sq : ℝ}
    (hA : 0 < A) (hB : 0 < B_opt) (hC : 0 < C) :
    endToEndComputeRisk R_inf A B_opt
        (optimalModelCapacity A B_opt C) (optimalDataBudget A B_opt C)
        (1 / (1 - Λ_CoT)) w_spurious Δ_OOD_sq =
      prescribedParetoEndToEndRisk R_inf A B_opt C Λ_CoT w_spurious Δ_OOD_sq := by
  unfold endToEndComputeRisk prescribedParetoEndToEndRisk
  rw [scalingRisk_at_optimal_eq_pareto R_inf hA hB hC]

/-- **Master Predictive Task-to-Model Synthesis Theorem (`master_predictive_model_prescription`)**:
Unifies all 4 axes of deep learning design at fixed compute budget `C > 0` and OOD shift
`Δ_OOD_sq > 0`:
- **Axis 1 (Architecture)**: `SESC` (`A_SESC = c_arch * dimSESC s r`, `w_spurious = 0`) vs.
  `Dense MLP` (`A_MLP = c_arch * dimMLP m d`, `w_MLP > 0`), with `0 < s < m` and `0 < r < k < d`.
- **Axis 2 (Optimizer)**: Two-Phase `RPM` (`B_RPM`) vs. `Scalar SGD` (`B_SGD > B_RPM > 0`).
- **Axis 3 (Compute Allocation)**: Variational Chinchilla `(N*(C), D*(C))` vs. any feasible
  allocation `(N, D)` satisfying `N * D = C` (`0 < N`, `0 < D`).
- **Axis 4 (Reasoning Rollout)**: Contractive CoT (`0 ≤ Λ_CoT < 1`, amplification `1 / (1 - Λ_CoT)`)
  vs. unanchored/direct rollout with amplification `amp_base ≥ 1 / (1 - Λ_CoT)`.

Proves:
1. **Strict End-to-End Dominance**: The prescribed tuple
   `(SESC, RPM, N*(C), D*(C), 1 / (1 - Λ_CoT), 0)` achieves strictly smaller `endToEndComputeRisk`
   than any baseline tuple `(MLP, SGD, N, D, amp_base, w_MLP)` at the same compute budget `C`.
2. **Exact Compute-Saving Factor & OOD Floor Elimination**: Even against an optimally-allocated
   Chinchilla `(MLP, SGD)` baseline on in-distribution data (`Δ_OOD_sq = 0`), the prescribed
   `(SESC, RPM)` design achieves **identical** risk using the exact fraction
   `C_prescribed / C_base = ((s * r : ℝ) / (m * d : ℝ)) * (B_RPM / B_SGD) < 1`
   of total training FLOPs, while simultaneously eliminating the positive OOD error floor
   `w_MLP * Δ_OOD_sq > 0`! -/
theorem master_predictive_model_prescription (R_inf : ℝ)
    {s m r k d : ℕ} {c_arch B_RPM B_SGD C N D Λ_CoT amp_base w_MLP Δ_OOD_sq : ℝ}
    (hR : 0 ≤ R_inf) (hc : 0 < c_arch)
    (hs : 0 < s) (hsm : s < m) (hr : 0 < r) (hrk : r < k) (hkd : k < d)
    (hB_RPM : 0 < B_RPM) (hB_lt : B_RPM < B_SGD)
    (hC : 0 < C) (hN : 0 < N) (hD : 0 < D) (hbudget : N * D = C)
    (hΛ0 : 0 ≤ Λ_CoT) (hΛ1 : Λ_CoT < 1)
    (hamp_le : 1 / (1 - Λ_CoT) ≤ amp_base)
    (hw_MLP : 0 < w_MLP) (hΔ : 0 < Δ_OOD_sq) :
    let p_SESC : ℝ := (dimSESC s r : ℝ)
    let p_MLP : ℝ := (dimMLP m d : ℝ)
    let A_SESC : ℝ := c_arch * p_SESC
    let A_MLP : ℝ := c_arch * p_MLP
    let amp_CoT : ℝ := 1 / (1 - Λ_CoT)
    let κ_save : ℝ := (p_SESC / p_MLP) * (B_RPM / B_SGD)
    (endToEndComputeRisk R_inf A_SESC B_RPM
        (optimalModelCapacity A_SESC B_RPM C) (optimalDataBudget A_SESC B_RPM C)
        amp_CoT 0 Δ_OOD_sq <
      endToEndComputeRisk R_inf A_MLP B_SGD N D amp_base w_MLP Δ_OOD_sq) ∧
    (0 < κ_save ∧ κ_save < 1 ∧
      prescribedParetoEndToEndRisk R_inf A_SESC B_RPM (κ_save * C) Λ_CoT 0 0 =
        prescribedParetoEndToEndRisk R_inf A_MLP B_SGD C Λ_CoT w_MLP 0 ∧
      prescribedParetoEndToEndRisk R_inf A_SESC B_RPM (κ_save * C) Λ_CoT 0 Δ_OOD_sq +
          w_MLP * Δ_OOD_sq =
        prescribedParetoEndToEndRisk R_inf A_MLP B_SGD C Λ_CoT w_MLP Δ_OOD_sq) := by
  intro p_SESC p_MLP A_SESC A_MLP amp_CoT κ_save
  obtain ⟨hdim1, hdim2, hdim3⟩ := four_tier_dimension_strict_hierarchy hs hsm hr hrk hkd
  have hdim_lt : dimSESC s r < dimMLP m d := (hdim1.trans hdim2).trans hdim3
  have hp_SESC_pos : 0 < p_SESC := Nat.cast_pos.mpr (Nat.mul_pos hs hr)
  have hp_lt : p_SESC < p_MLP := Nat.cast_lt.mpr hdim_lt
  have hp_MLP_pos : 0 < p_MLP := hp_SESC_pos.trans hp_lt
  have hA_SESC_pos : 0 < A_SESC := mul_pos hc hp_SESC_pos
  have hA_MLP_pos : 0 < A_MLP := mul_pos hc hp_MLP_pos
  have hA_lt : A_SESC < A_MLP := mul_lt_mul_of_pos_left hp_lt hc
  have hB_SGD_pos : 0 < B_SGD := hB_RPM.trans hB_lt
  have hamp_CoT_pos : 0 < amp_CoT := one_div_pos.mpr (sub_pos.mpr hΛ1)
  have _ := hΛ0
  -- 1. Strict End-to-End Dominance
  have h_opt_eval :=
    endToEndComputeRisk_at_optimal_eq_prescribed R_inf hA_SESC_pos hB_RPM hC
      (Λ_CoT := Λ_CoT) (w_spurious := 0) (Δ_OOD_sq := Δ_OOD_sq)
  have h_pareto_lt :
      paretoComputeRisk R_inf A_SESC B_RPM C < paretoComputeRisk R_inf A_MLP B_SGD C :=
    paretoComputeRisk_strict_mono_in_arch_and_opt R_inf hA_SESC_pos hB_RPM
      hA_lt.le hB_lt.le (Or.inl hA_lt) hC
  have h_chinchilla_le :
      paretoComputeRisk R_inf A_MLP B_SGD C ≤ scalingRisk R_inf A_MLP B_SGD N D :=
    chinchilla_variational_global_optimality R_inf hA_MLP_pos.le hB_SGD_pos.le hN hD hC hbudget
  have h_pareto_nonneg : 0 ≤ paretoComputeRisk R_inf A_MLP B_SGD C := by
    unfold paretoComputeRisk
    positivity
  have h_scaled_lt :
      amp_CoT * paretoComputeRisk R_inf A_SESC B_RPM C <
        amp_base * scalingRisk R_inf A_MLP B_SGD N D := by
    calc
      amp_CoT * paretoComputeRisk R_inf A_SESC B_RPM C
        < amp_CoT * paretoComputeRisk R_inf A_MLP B_SGD C :=
          mul_lt_mul_of_pos_left h_pareto_lt hamp_CoT_pos
      _ ≤ amp_base * scalingRisk R_inf A_MLP B_SGD N D :=
          mul_le_mul hamp_le h_chinchilla_le h_pareto_nonneg (hamp_CoT_pos.le.trans hamp_le)
  have hood_pos : 0 < w_MLP * Δ_OOD_sq := mul_pos hw_MLP hΔ
  have h_dom :
      endToEndComputeRisk R_inf A_SESC B_RPM
          (optimalModelCapacity A_SESC B_RPM C) (optimalDataBudget A_SESC B_RPM C)
          amp_CoT 0 Δ_OOD_sq <
        endToEndComputeRisk R_inf A_MLP B_SGD N D amp_base w_MLP Δ_OOD_sq := by
    rw [h_opt_eval]
    unfold prescribedParetoEndToEndRisk endToEndComputeRisk
    change amp_CoT * paretoComputeRisk R_inf A_SESC B_RPM C + 0 * Δ_OOD_sq <
      amp_base * scalingRisk R_inf A_MLP B_SGD N D + w_MLP * Δ_OOD_sq
    linarith
  -- 2. Exact Compute-Saving Factor & OOD Floor Elimination
  have hfac_p_pos : 0 < p_SESC / p_MLP := div_pos hp_SESC_pos hp_MLP_pos
  have hfac_B_pos : 0 < B_RPM / B_SGD := div_pos hB_RPM hB_SGD_pos
  have hfac_p_lt1 : p_SESC / p_MLP < 1 := (div_lt_one hp_MLP_pos).mpr hp_lt
  have hfac_B_lt1 : B_RPM / B_SGD < 1 := (div_lt_one hB_SGD_pos).mpr hB_lt
  have hκ_pos : 0 < κ_save := mul_pos hfac_p_pos hfac_B_pos
  have hκ_lt1 : κ_save < 1 := by
    calc
      κ_save < 1 * (B_RPM / B_SGD) := mul_lt_mul_of_pos_right hfac_p_lt1 hfac_B_pos
      _ = B_RPM / B_SGD := one_mul _
      _ < 1 := hfac_B_lt1
  have heq_budget :
      κ_save * C = equivalentComputeBudget A_SESC A_MLP B_RPM B_SGD C := by
    unfold κ_save equivalentComputeBudget A_SESC A_MLP
    have hc_ne : c_arch ≠ 0 := ne_of_gt hc
    have hp_MLP_ne : p_MLP ≠ 0 := ne_of_gt hp_MLP_pos
    field_simp [hc_ne, hp_MLP_ne]
  have h_pareto_eq :
      paretoComputeRisk R_inf A_SESC B_RPM (κ_save * C) =
        paretoComputeRisk R_inf A_MLP B_SGD C := by
    rw [heq_budget]
    exact codesign_compute_saving_factor R_inf hA_SESC_pos hA_MLP_pos hB_RPM hB_SGD_pos hC
  refine ⟨h_dom, hκ_pos, hκ_lt1, ?_, ?_⟩
  · unfold prescribedParetoEndToEndRisk
    rw [h_pareto_eq]
    ring
  · unfold prescribedParetoEndToEndRisk
    rw [h_pareto_eq]
    ring

end OpenTheory.PredictiveModelPrescription
