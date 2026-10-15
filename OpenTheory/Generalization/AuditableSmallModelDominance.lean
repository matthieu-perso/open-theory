import OpenTheory.Generalization.TestTimeScaling
import OpenTheory.Generalization.PredictiveModelPrescription

/-!
# 100× Smaller Auditable Circuit Models Beating Monolithic Black-Box Giants

This module formalizes the exact conditions under which a compact, mechanistically auditable
**Spectral-Equivariant Sparse Circuit (`SESC`)** combined with contractive Chain-of-Thought (CoT)
and Best-of-`K` test-time reasoning strictly outperforms an `M×` (e.g., `100×`) larger monolithic
1-step dense model across **all four** axes simultaneously:
1. **Human / Mechanistic Auditability Complexity**: `s * r < m * d` with compression ratio
   `auditCompressionRatio s r m d = (s * r) / (m * d) < 1`,
2. **Pre-Training Compute FLOPs**: exact compute-saving factor
   `κ_save = ((s * r) / (m * d)) * (B_RPM / B_SGD) < 1`,
3. **Per-Query Inference FLOPs**: `smallAuditableInferenceFLOPs N_small T_reason ≤ giantOneStepInferenceFLOPs M N_small`
   for any reasoning budget `T_reason ≤ M` (strict `<` when `T_reason < M`),
4. **In-Distribution + Out-of-Distribution Deployment Risk**: strict dominance in `fullDeploymentRisk`,
   including **unconditional dominance** for any `β_reason > 0` and any `1 < T_reason ≤ M` under
   architecture-matched compression `M = (m * d) / (s * r)`.
-/

noncomputable section

namespace OpenTheory.AuditableSmallModelDominance

open OpenTheory
open OpenTheory.ComputeOptimalSynthesis
open OpenTheory.SpectralSparseCircuit
open OpenTheory.TestTimeScaling
open OpenTheory.PredictiveModelPrescription

/-! ## 1. Mechanistic Auditability Complexity & Compression Ratio -/

/-- Mechanistic auditability complexity of a **Spectral-Equivariant Sparse Circuit (`SESC`)** with
`s` active circuit units and local spectral rank `r`, measured as the number of active scalar
parameters that must be inspected to verify its input-output circuit:
`auditComplexitySESC s r = dimSESC s r = s * r`. -/
def auditComplexitySESC (s r : ℕ) : ℕ :=
  dimSESC s r

/-- Mechanistic auditability complexity of a monolithic **Dense MLP (`Giant`)** with hidden width
`m` and ambient input dimension `d`:
`auditComplexityGiant m d = dimMLP m d = m * d`. -/
def auditComplexityGiant (m d : ℕ) : ℕ :=
  dimMLP m d

/-- Mechanistic **Audit Compression Ratio (`auditCompressionRatio`)** comparing the active scalar
circuit size of `SESC` against a monolithic dense `MLP`:
`auditCompressionRatio s r m d = (dimSESC s r : ℝ) / (dimMLP m d : ℝ)`. -/
def auditCompressionRatio (s r m d : ℕ) : ℝ :=
  (dimSESC s r : ℝ) / (dimMLP m d : ℝ)

@[simp]
theorem auditComplexitySESC_eq (s r : ℕ) :
    auditComplexitySESC s r = s * r :=
  rfl

@[simp]
theorem auditComplexityGiant_eq (m d : ℕ) :
    auditComplexityGiant m d = m * d :=
  rfl

/-- **Strict Auditability Compression of SESC vs. Monolithic Giant (`auditCompressionRatio_pos_and_lt_one`)**:
Whenever active circuit width satisfies `0 < s < m` and spectral rank satisfies `0 < r < d`:
1. `0 < auditCompressionRatio s r m d`,
2. `auditCompressionRatio s r m d < 1`,
3. `auditComplexitySESC s r < auditComplexityGiant m d`. -/
theorem auditCompressionRatio_pos_and_lt_one {s r m d : ℕ}
    (hs : 0 < s) (hsm : s < m) (hr : 0 < r) (hrd : r < d) :
    0 < auditCompressionRatio s r m d ∧
      auditCompressionRatio s r m d < 1 ∧
      auditComplexitySESC s r < auditComplexityGiant m d := by
  have hm_pos : 0 < m := lt_trans hs hsm
  have h_sr_lt_mr : s * r < m * r := Nat.mul_lt_mul_of_pos_right hsm hr
  have h_mr_lt_md : m * r < m * d := Nat.mul_lt_mul_of_pos_left hrd hm_pos
  have h_sr_lt_md : dimSESC s r < dimMLP m d := lt_trans h_sr_lt_mr h_mr_lt_md
  have h_sesc_pos : 0 < (dimSESC s r : ℝ) := Nat.cast_pos.mpr (Nat.mul_pos hs hr)
  have h_lt_real : (dimSESC s r : ℝ) < (dimMLP m d : ℝ) := Nat.cast_lt.mpr h_sr_lt_md
  have h_mlp_pos : 0 < (dimMLP m d : ℝ) := lt_trans h_sesc_pos h_lt_real
  unfold auditCompressionRatio auditComplexitySESC auditComplexityGiant
  refine ⟨div_pos h_sesc_pos h_mlp_pos, (div_lt_one h_mlp_pos).mpr h_lt_real, h_sr_lt_md⟩

/-! ## 2. Small Auditable Reasoning Model vs. `M×` Monolithic Giant at Equal or Lower Inference FLOPs -/

/-- Per-query inference FLOP cost of an `M×` larger monolithic 1-step giant model with parameter
count `M * N_small` and direct 1-step generation (`T_inf = 1`):
`giantOneStepInferenceFLOPs M N_small = (M * N_small) * 1`. -/
def giantOneStepInferenceFLOPs (M N_small : ℝ) : ℝ :=
  (M * N_small) * 1

/-- Per-query inference FLOP cost of a compact auditable reasoning model of size `N_small` using
test-time reasoning multiplier `T_reason`:
`smallAuditableInferenceFLOPs N_small T_reason = N_small * T_reason`. -/
def smallAuditableInferenceFLOPs (N_small T_reason : ℝ) : ℝ :=
  N_small * T_reason

/-- **Inference FLOP Dominance of the Small Auditable Reasoning Model
(`small_auditable_uses_le_inference_flops`)**:
For any base parameter count `0 < N_small`:
1. Whenever `T_reason ≤ M`, `smallAuditableInferenceFLOPs N_small T_reason ≤ giantOneStepInferenceFLOPs M N_small`.
2. Whenever `T_reason < M`, `smallAuditableInferenceFLOPs N_small T_reason < giantOneStepInferenceFLOPs M N_small`.
3. When `T_reason = M`, `smallAuditableInferenceFLOPs N_small M = giantOneStepInferenceFLOPs M N_small`. -/
theorem small_auditable_uses_le_inference_flops {M N_small T_reason : ℝ}
    (hN : 0 < N_small) (hTM : T_reason ≤ M) :
    smallAuditableInferenceFLOPs N_small T_reason ≤ giantOneStepInferenceFLOPs M N_small ∧
      (T_reason < M →
        smallAuditableInferenceFLOPs N_small T_reason < giantOneStepInferenceFLOPs M N_small) ∧
      smallAuditableInferenceFLOPs N_small M = giantOneStepInferenceFLOPs M N_small := by
  unfold smallAuditableInferenceFLOPs giantOneStepInferenceFLOPs
  have hle : N_small * T_reason ≤ N_small * M := mul_le_mul_of_nonneg_left hTM hN.le
  refine ⟨by linarith, ?_, by ring⟩
  intro hlt
  have hstrict : N_small * T_reason < N_small * M := mul_lt_mul_of_pos_left hlt hN
  linarith

/-- Architectural capacity constant `c_arch * p` grows strictly with effective parameter dimension:
for `0 < c_arch`, `0 < s < m`, and `0 < r < d`, the `SESC` capacity constant
`α_small = c_arch * dimSESC s r` is strictly smaller than the monolithic giant's capacity constant
`α_giant = c_arch * dimMLP m d`. -/
theorem sesc_capacity_constant_lt_giant {s r m d : ℕ} {c_arch : ℝ}
    (hc : 0 < c_arch) (hs : 0 < s) (hsm : s < m) (hr : 0 < r) (hrd : r < d) :
    c_arch * (dimSESC s r : ℝ) < c_arch * (dimMLP m d : ℝ) := by
  have hdim_lt := (auditCompressionRatio_pos_and_lt_one hs hsm hr hrd).2.2
  exact mul_lt_mul_of_pos_left (Nat.cast_lt.mpr hdim_lt) hc

/-- **Full In-Distribution + OOD Deployment Risk (`fullDeploymentRisk`)**:
Combines per-query test-time scaling risk `inferenceScalingRisk R_inf α_cap β_reason N T_inf`
with out-of-distribution spurious shift risk `w_spurious * Δ_OOD_sq`:
`fullDeploymentRisk R_inf α_cap β_reason N T_inf w_spurious Δ_OOD_sq = inferenceScalingRisk R_inf α_cap β_reason N T_inf + w_spurious * Δ_OOD_sq`. -/
def fullDeploymentRisk (R_inf α_cap β_reason N T_inf w_spurious Δ_OOD_sq : ℝ) : ℝ :=
  inferenceScalingRisk R_inf α_cap β_reason N T_inf + w_spurious * Δ_OOD_sq

/-- Exact risk gap identity between an `M×` larger monolithic 1-step giant
`(M * N_small, T_inf = 1, w_giant)` and a compact auditable reasoning model
`(N_small, T_reason, w_spurious = 0)`:
`fullDeploymentRisk R_inf α_giant β_reason (M * N_small) 1 w_giant Δ_OOD_sq - fullDeploymentRisk R_inf α_small β_reason N_small T_reason 0 Δ_OOD_sq = (β_reason * (1 - 1 / T_reason) - (α_small / N_small - α_giant / (M * N_small))) + w_giant * Δ_OOD_sq`. -/
theorem giant_minus_small_auditable_deployment_risk_gap
    (R_inf α_small α_giant β_reason N_small M T_reason w_giant Δ_OOD_sq : ℝ)
    (hT_ne : T_reason ≠ 0) :
    fullDeploymentRisk R_inf α_giant β_reason (M * N_small) 1 w_giant Δ_OOD_sq -
        fullDeploymentRisk R_inf α_small β_reason N_small T_reason 0 Δ_OOD_sq =
      (β_reason * (1 - 1 / T_reason) - (α_small / N_small - α_giant / (M * N_small))) +
        w_giant * Δ_OOD_sq := by
  unfold fullDeploymentRisk inferenceScalingRisk
  field_simp [hT_ne]
  ring

/-- **Auditable Small Reasoning Model Strictly Beats Monolithic 1-Step Giant
(`auditable_small_model_strictly_beats_monolithic_giant`)**:
Whenever:
- `0 < N_small` and `1 < T_reason ≤ M` (so the small model uses at most the inference FLOPs of the
  `M×` giant),
- `α_small / N_small - α_giant / (M * N_small) < β_reason * (1 - 1 / T_reason)` (the test-time
  reasoning reduction exceeds any net capacity-per-parameter difference), and
- `0 ≤ w_giant * Δ_OOD_sq` (nonnegative OOD spurious shift penalty on the unstructured giant),
the compact auditable `SESC` model (`w_spurious = 0`) achieves **strictly lower deployment risk**
while using **equal or fewer inference FLOPs**:
`smallAuditableInferenceFLOPs N_small T_reason ≤ giantOneStepInferenceFLOPs M N_small` and
`fullDeploymentRisk R_inf α_small β_reason N_small T_reason 0 Δ_OOD_sq < fullDeploymentRisk R_inf α_giant β_reason (M * N_small) 1 w_giant Δ_OOD_sq`. -/
theorem auditable_small_model_strictly_beats_monolithic_giant
    (R_inf α_small α_giant β_reason N_small M T_reason w_giant Δ_OOD_sq : ℝ)
    (hN : 0 < N_small) (hT1 : 1 < T_reason) (hTM : T_reason ≤ M)
    (h_reason_dom : α_small / N_small - α_giant / (M * N_small) < β_reason * (1 - 1 / T_reason))
    (h_ood_nonneg : 0 ≤ w_giant * Δ_OOD_sq) :
    smallAuditableInferenceFLOPs N_small T_reason ≤ giantOneStepInferenceFLOPs M N_small ∧
      fullDeploymentRisk R_inf α_small β_reason N_small T_reason 0 Δ_OOD_sq <
        fullDeploymentRisk R_inf α_giant β_reason (M * N_small) 1 w_giant Δ_OOD_sq := by
  have hT_pos : 0 < T_reason := zero_lt_one.trans hT1
  have hT_ne : T_reason ≠ 0 := ne_of_gt hT_pos
  have hgap := giant_minus_small_auditable_deployment_risk_gap
    R_inf α_small α_giant β_reason N_small M T_reason w_giant Δ_OOD_sq hT_ne
  have hflops := (small_auditable_uses_le_inference_flops hN hTM).1
  refine ⟨hflops, ?_⟩
  linarith

/-! ## 3. The Special Case of Architecture-Matched Compression (`M = (m * d) / (s * r)`) -/

/-- **Unconditional Dominance Under Architecture-Matched Compression
(`symmetry_matched_small_model_unconditional_dominance`)**:
Suppose the monolithic giant has `M > 1` times more parameters than the compact `SESC` model
(`N_giant = M * N_small`), AND its capacity constant is `M` times larger (`α_giant = M * α_small`,
e.g. `M = (m * d) / (s * r)` where the `(M - 1) * N_small` extra parameters of the dense giant are
spent fitting non-invariant ambient directions). Then:
1. **Identical Capacity-to-Parameter Error**:
   `α_small / N_small = (M * α_small) / (M * N_small)` (so the capacity gap vanishes identically!),
2. **Exact Positive Risk Margin**:
   `fullDeploymentRisk R_inf (M * α_small) β_reason (M * N_small) 1 w_giant Δ_OOD_sq - fullDeploymentRisk R_inf α_small β_reason N_small T_reason 0 Δ_OOD_sq = β_reason * (1 - 1 / T_reason) + w_giant * Δ_OOD_sq`,
3. **Unconditional Strict Risk Dominance**: For **every** `β_reason > 0`, **every** `1 < T_reason ≤ M`,
   and any `0 ≤ w_giant * Δ_OOD_sq`, the `M×` smaller `SESC` reasoning model strictly beats the
   `M×` larger 1-step giant while using equal or fewer per-query inference FLOPs! -/
theorem symmetry_matched_small_model_unconditional_dominance
    (R_inf α_small β_reason N_small M T_reason w_giant Δ_OOD_sq : ℝ)
    (hN : 0 < N_small) (hT1 : 1 < T_reason) (hTM : T_reason ≤ M)
    (hβ : 0 < β_reason) (h_ood_nonneg : 0 ≤ w_giant * Δ_OOD_sq) :
    α_small / N_small = (M * α_small) / (M * N_small) ∧
      smallAuditableInferenceFLOPs N_small T_reason ≤ giantOneStepInferenceFLOPs M N_small ∧
      fullDeploymentRisk R_inf (M * α_small) β_reason (M * N_small) 1 w_giant Δ_OOD_sq -
          fullDeploymentRisk R_inf α_small β_reason N_small T_reason 0 Δ_OOD_sq =
        β_reason * (1 - 1 / T_reason) + w_giant * Δ_OOD_sq ∧
      0 < β_reason * (1 - 1 / T_reason) + w_giant * Δ_OOD_sq ∧
      fullDeploymentRisk R_inf α_small β_reason N_small T_reason 0 Δ_OOD_sq <
        fullDeploymentRisk R_inf (M * α_small) β_reason (M * N_small) 1 w_giant Δ_OOD_sq := by
  have hM1 : 1 < M := lt_of_lt_of_le hT1 hTM
  have hM_pos : 0 < M := zero_lt_one.trans hM1
  have hM_ne : M ≠ 0 := ne_of_gt hM_pos
  have hT_pos : 0 < T_reason := zero_lt_one.trans hT1
  have hT_ne : T_reason ≠ 0 := ne_of_gt hT_pos
  have h_cap_eq : α_small / N_small = (M * α_small) / (M * N_small) := by
    rw [mul_div_mul_left α_small N_small hM_ne]
  have hgap := giant_minus_small_auditable_deployment_risk_gap
    R_inf α_small (M * α_small) β_reason N_small M T_reason w_giant Δ_OOD_sq hT_ne
  have hgap_simp :
      fullDeploymentRisk R_inf (M * α_small) β_reason (M * N_small) 1 w_giant Δ_OOD_sq -
          fullDeploymentRisk R_inf α_small β_reason N_small T_reason 0 Δ_OOD_sq =
        β_reason * (1 - 1 / T_reason) + w_giant * Δ_OOD_sq := by
    rw [hgap, ← h_cap_eq, sub_self, sub_zero]
  have h1_div_lt : 1 / T_reason < 1 := (div_lt_one hT_pos).mpr hT1
  have hfac_pos : 0 < 1 - 1 / T_reason := sub_pos.mpr h1_div_lt
  have h_reason_pos : 0 < β_reason * (1 - 1 / T_reason) := mul_pos hβ hfac_pos
  have h_margin_pos : 0 < β_reason * (1 - 1 / T_reason) + w_giant * Δ_OOD_sq := by linarith
  have hflops := (small_auditable_uses_le_inference_flops hN hTM).1
  refine ⟨h_cap_eq, hflops, hgap_simp, h_margin_pos, by linarith⟩

/-! ## 4. Master 4-Way Dominance Theorem (`master_auditable_small_vs_monolithic_giant_theorem`) -/

/-- **Master 4-Way Dominance Theorem: Auditable Small SESC + CoT Model vs. `M×` Monolithic Giant
(`master_auditable_small_vs_monolithic_giant_theorem`)**:
Given a task with sparse circuit size `0 < s < m` and spectral rank `0 < r < d`, define the
architectural compression factor `M = (dimMLP m d : ℝ) / (dimSESC s r : ℝ) = (m * d) / (s * r) > 1`,
capacity constants `α_small = c_arch * dimSESC s r` and `α_giant = c_arch * dimMLP m d`, and
pre-training compute-saving factor `κ_save = auditCompressionRatio s r m d * (B_RPM / B_SGD)`.

Then for any base parameter count `0 < N_small`, any reasoning multiplier `1 < T_reason ≤ M`,
any positive reasoning sensitivity `0 < β_reason`, any optimizer improvement `0 < B_RPM < B_SGD`,
and any nonnegative OOD shift penalty `0 ≤ w_giant * Δ_OOD_sq`, the `M×` smaller auditable `SESC`
reasoning model simultaneously achieves all **four** dominance properties over the `M×` larger
monolithic 1-step dense `MLP` giant:
1. **Auditability Compression**: `auditComplexitySESC s r < auditComplexityGiant m d` and
   `0 < auditCompressionRatio s r m d < 1` (with `auditCompressionRatio s r m d = 1 / M`),
2. **Pre-Training Compute Compression**: `0 < κ_save < 1`, achieving identical Chinchilla Pareto
   pre-training risk `paretoComputeRisk R_inf α_small B_RPM (κ_save * C) = paretoComputeRisk R_inf α_giant B_SGD C`
   at fraction `κ_save < 1` of the training FLOPs,
3. **Inference FLOP Savings**: `smallAuditableInferenceFLOPs N_small T_reason ≤ giantOneStepInferenceFLOPs M N_small`
   (strict `<` whenever `T_reason < M`),
4. **Strict In-Distribution + OOD Deployment Risk Dominance**:
   `fullDeploymentRisk R_inf α_small β_reason N_small T_reason 0 Δ_OOD_sq < fullDeploymentRisk R_inf α_giant β_reason (M * N_small) 1 w_giant Δ_OOD_sq`
   with exact margin `β_reason * (1 - 1 / T_reason) + w_giant * Δ_OOD_sq > 0`. -/
theorem master_auditable_small_vs_monolithic_giant_theorem
    (R_inf : ℝ) {s r m d : ℕ}
    {c_arch B_RPM B_SGD C N_small T_reason β_reason w_giant Δ_OOD_sq : ℝ}
    (hc : 0 < c_arch)
    (hs : 0 < s) (hsm : s < m) (hr : 0 < r) (hrd : r < d)
    (hB_RPM : 0 < B_RPM) (hB_lt : B_RPM < B_SGD) (hC : 0 < C)
    (hN : 0 < N_small)
    (hT1 : 1 < T_reason)
    (hTM : T_reason ≤ (dimMLP m d : ℝ) / (dimSESC s r : ℝ))
    (hβ : 0 < β_reason)
    (h_ood_nonneg : 0 ≤ w_giant * Δ_OOD_sq) :
    let p_SESC : ℝ := (dimSESC s r : ℝ)
    let p_MLP : ℝ := (dimMLP m d : ℝ)
    let M : ℝ := p_MLP / p_SESC
    let α_small : ℝ := c_arch * p_SESC
    let α_giant : ℝ := c_arch * p_MLP
    let κ_save : ℝ := auditCompressionRatio s r m d * (B_RPM / B_SGD)
    -- (1) Mechanistic Auditability Compression
    (auditComplexitySESC s r < auditComplexityGiant m d ∧
      0 < auditCompressionRatio s r m d ∧
      auditCompressionRatio s r m d < 1 ∧
      1 < M) ∧
    -- (2) Pre-Training Compute FLOP Compression
    (0 < κ_save ∧ κ_save < 1 ∧
      paretoComputeRisk R_inf α_small B_RPM (κ_save * C) =
        paretoComputeRisk R_inf α_giant B_SGD C) ∧
    -- (3) Per-Query Inference FLOP Savings
    (smallAuditableInferenceFLOPs N_small T_reason ≤ giantOneStepInferenceFLOPs M N_small ∧
      (T_reason < M →
        smallAuditableInferenceFLOPs N_small T_reason < giantOneStepInferenceFLOPs M N_small)) ∧
    -- (4) Strict In-Distribution + OOD Deployment Risk Dominance
    (fullDeploymentRisk R_inf α_giant β_reason (M * N_small) 1 w_giant Δ_OOD_sq -
        fullDeploymentRisk R_inf α_small β_reason N_small T_reason 0 Δ_OOD_sq =
      β_reason * (1 - 1 / T_reason) + w_giant * Δ_OOD_sq ∧
      fullDeploymentRisk R_inf α_small β_reason N_small T_reason 0 Δ_OOD_sq <
        fullDeploymentRisk R_inf α_giant β_reason (M * N_small) 1 w_giant Δ_OOD_sq) := by
  intro p_SESC p_MLP M α_small α_giant κ_save
  obtain ⟨hratio_pos, hratio_lt1, haudit_lt⟩ :=
    auditCompressionRatio_pos_and_lt_one hs hsm hr hrd
  have hp_SESC_pos : 0 < p_SESC := Nat.cast_pos.mpr (Nat.mul_pos hs hr)
  have hp_lt : p_SESC < p_MLP := Nat.cast_lt.mpr haudit_lt
  have hp_MLP_pos : 0 < p_MLP := lt_trans hp_SESC_pos hp_lt
  have hM_gt1 : 1 < M := (one_lt_div hp_SESC_pos).mpr hp_lt
  have hα_small_pos : 0 < α_small := mul_pos hc hp_SESC_pos
  have hα_giant_pos : 0 < α_giant := mul_pos hc hp_MLP_pos
  have hB_SGD_pos : 0 < B_SGD := lt_trans hB_RPM hB_lt
  -- (2) Pre-Training Compute Compression
  have hB_ratio_pos : 0 < B_RPM / B_SGD := div_pos hB_RPM hB_SGD_pos
  have hB_ratio_lt1 : B_RPM / B_SGD < 1 := (div_lt_one hB_SGD_pos).mpr hB_lt
  have hκ_pos : 0 < κ_save := mul_pos hratio_pos hB_ratio_pos
  have hκ_lt1 : κ_save < 1 := by
    calc
      κ_save < 1 * (B_RPM / B_SGD) := mul_lt_mul_of_pos_right hratio_lt1 hB_ratio_pos
      _ = B_RPM / B_SGD := one_mul _
      _ < 1 := hB_ratio_lt1
  have heq_budget :
      κ_save * C = equivalentComputeBudget α_small α_giant B_RPM B_SGD C := by
    unfold κ_save auditCompressionRatio equivalentComputeBudget α_small α_giant p_SESC p_MLP
    have hc_ne : c_arch ≠ 0 := ne_of_gt hc
    have hp_MLP_ne : (dimMLP m d : ℝ) ≠ 0 := ne_of_gt hp_MLP_pos
    field_simp [hc_ne, hp_MLP_ne]
  have h_pareto_eq :
      paretoComputeRisk R_inf α_small B_RPM (κ_save * C) =
        paretoComputeRisk R_inf α_giant B_SGD C := by
    rw [heq_budget]
    exact codesign_compute_saving_factor R_inf hα_small_pos hα_giant_pos hB_RPM hB_SGD_pos hC
  -- (3) & (4) Inference FLOPs and Deployment Risk Dominance
  have hflops := small_auditable_uses_le_inference_flops hN hTM
  have hα_giant_eq : α_giant = M * α_small := by
    unfold α_giant α_small M p_SESC p_MLP
    have hp_SESC_ne : (dimSESC s r : ℝ) ≠ 0 := ne_of_gt hp_SESC_pos
    field_simp [hp_SESC_ne]
  have hdom := symmetry_matched_small_model_unconditional_dominance
    R_inf α_small β_reason N_small M T_reason w_giant Δ_OOD_sq
    hN hT1 hTM hβ h_ood_nonneg
  rw [← hα_giant_eq] at hdom
  exact ⟨⟨haudit_lt, hratio_pos, hratio_lt1, hM_gt1⟩,
    ⟨hκ_pos, hκ_lt1, h_pareto_eq⟩,
    ⟨hflops.1, hflops.2.1⟩,
    ⟨hdom.2.2.1, hdom.2.2.2.2⟩⟩

end OpenTheory.AuditableSmallModelDominance
