import OpenTheory.Optimization.CertiRPM
import OpenTheory.Generalization.ZeroShotCompiler
import OpenTheory.Generalization.TestTimeScaling
import OpenTheory.Architectures.RuntimeReasoningGuardrail

/-!
# Executable Telemetry-to-Certificate Synthesis Engine & Grand Unified Theorem of OpenTheory

This module formalizes the **Executable Telemetry-to-Certificate Synthesis Engine
(`synthesizeSystem`)** and the **Grand Unified Soundness Theorem of OpenTheory
(`grand_unified_open_theory_soundness`)**:

A single verifiable synthesis pipeline that ingests observable telemetry (`ObservableTelemetry`)
from a cheap task/model probe and outputs a complete, machine-checked system certificate
(`SynthesizedCertificate`) across all four practical pillars of `OpenTheory`:

1. **Pillar 1 — Zero-Shot Pre-Training Architecture & Compute Compiler (`ZeroShotCompiler`)**:
   - Computes certified sample-size brackets `[scarceThreshold, abundantThreshold]` bracketing the
     unobserved oracle threshold `criticalSampleThreshold`, decides whether to deploy pruned `SESC`
     (`s * r`) via `n < scarceThreshold`, compiles the budget-feasible Chinchilla allocation
     `(N_compiled, D_compiled)` with exact completion-of-squares regret, and certifies the
     architecture-optimizer co-design FLOP compression ratio `κ_flop_save = (sr / md) * (B_RPM / B_SGD) < 1`.

2. **Pillar 2 — Self-Tuning `CertiRPM` Optimizer (`CertiRPM`)**:
   - Automatically selects between the small, high-variance saddle-escape batch size
     `B_escape_batch` (`κ_hat < 0`) and the large, low-variance PL-basin batch size
     `B_basin_batch` (`κ_hat ≥ 0`), while guaranteeing 100% safe non-divergent descent
     (`0 ≤ optimizerNetProgress`) and pure quadratic sharpness-estimation regret under
     upper-confidence sharpness `0 < S_i ≤ S_hat_i`.

3. **Pillar 3 — Test-Time Reasoning Allocation & Small-Model Dominance (`TestTimeScaling`)**:
   - Certifies that whenever the single-step capacity gap `α_model / N_small` is smaller than the
     reasoning bottleneck `β_reason` (`α_model / N_small < β_reason`) and `M_giant > 1`, a small
     reasoning model `(N_small, T_inf = M_giant)` strictly outperforms an `M_giant×` larger 1-step
     model `(M_giant * N_small, 1)` at identical per-query inference FLOPs.

4. **Pillar 4 — Runtime Triple-Shield Horizon-Free & OOD Safety (`RuntimeReasoningGuardrail`)**:
   - Synthesizes the minimum certified context-anchoring strength `requiredAnchor` and horizon-free
     rollout error ceiling `rolloutErrorCeiling = ε_token / (1 - Λ_max)` bounding `H`-step
     autoregressive Chain-of-Thought drift across **all** horizons `H ∈ ℕ`, while simultaneously
     guaranteeing identically zero OOD shift penalty under orthogonal spectral-bulk perturbations on
     `SESC`.
-/

noncomputable section

open scoped BigOperators

namespace OpenTheory.ExecutableSynthesisEngine

open OpenTheory
open OpenTheory.LocalSharpnessDescent
open OpenTheory.PreconditionedDescent
open OpenTheory.StochasticSharpnessSGD
open OpenTheory.MomentumAdamTrajectory
open OpenTheory.MLPLocalSharpness
open OpenTheory.ResidualPreconditionedMomentum
open OpenTheory.GlobalSaddlePL
open OpenTheory.CertiRPM
open OpenTheory.ConvolutionalGeneralization
open OpenTheory.SpectralSparseCircuit
open OpenTheory.AutoregressiveOOD
open OpenTheory.ComputeOptimalSynthesis
open OpenTheory.PredictiveModelPrescription
open OpenTheory.ZeroShotCompiler
open OpenTheory.TestTimeScaling
open OpenTheory.RuntimeReasoningGuardrail

/-! ## Section 1: Observable Telemetry Input & Synthesized System Certificate Bundles -/

/-- **Observable Probe Telemetry (`ObservableTelemetry`)**:
Bundles all empirical measurements collected from a small-scale pre-training, optimization,
inference, and runtime safety probe:
- Architecture dimensions: `s m r k d : ℕ`
- Dataset & pruning probe: `n c_gen ε_hat δ_ε : ℝ`
- Scaling probe: `ρ_hat C_train : ℝ`
- Optimizer probe: `κ_hat η_escape B_RPM B_SGD : ℝ` and `B_escape_batch B_basin_batch : ℕ`
- Inference & reasoning probe: `α_model β_reason N_small M_giant : ℝ`
- Runtime safety probe: `Λ_raw Λ_max ε_token : ℝ` -/
structure ObservableTelemetry where
  s : ℕ
  m : ℕ
  r : ℕ
  k : ℕ
  d : ℕ
  n : ℝ
  c_gen : ℝ
  ε_hat : ℝ
  δ_ε : ℝ
  ρ_hat : ℝ
  C_train : ℝ
  κ_hat : ℝ
  η_escape : ℝ
  B_RPM : ℝ
  B_SGD : ℝ
  B_escape_batch : ℕ
  B_basin_batch : ℕ
  α_model : ℝ
  β_reason : ℝ
  N_small : ℝ
  M_giant : ℝ
  Λ_raw : ℝ
  Λ_max : ℝ
  ε_token : ℝ

/-- **Synthesized System Certificate (`SynthesizedCertificate`)**:
Bundles the complete machine-certified system configuration synthesized from `ObservableTelemetry`:
- `scarceThreshold abundantThreshold : ℝ`: certified sample-size brackets around `n_crit`
- `usePrunedSESC : Bool`: certified data-scarce architecture decision (`n < scarceThreshold`)
- `N_compiled D_compiled : ℝ`: compiled Chinchilla model capacity and token budget
- `κ_flop_save : ℝ`: pre-training co-design FLOP reduction ratio `(sr / md) * (B_RPM / B_SGD)`
- `activeBatchSize : ℕ`: `CertiRPM` self-tuning mini-batch size (`certiRPMBatchSize`)
- `requiredAnchor : ℝ`: runtime CoT context-anchoring strength (`requiredAnchorStrength Λ_raw Λ_max`)
- `rolloutErrorCeiling : ℝ`: horizon-free rollout error bound `ε_token / (1 - Λ_max)` -/
structure SynthesizedCertificate where
  scarceThreshold : ℝ
  abundantThreshold : ℝ
  usePrunedSESC : Bool
  N_compiled : ℝ
  D_compiled : ℝ
  κ_flop_save : ℝ
  activeBatchSize : ℕ
  requiredAnchor : ℝ
  rolloutErrorCeiling : ℝ

/-- **Executable Telemetry-to-Certificate Synthesis Pipeline (`synthesizeSystem`)**:
Ingests an `ObservableTelemetry` record `t` and synthesizes the complete `SynthesizedCertificate`
across pre-training compilation, `CertiRPM` batch selection, and runtime rollout stabilization. -/
def synthesizeSystem (t : ObservableTelemetry) : SynthesizedCertificate :=
  let scarce := certifiedScarceThreshold t.s t.m t.r t.c_gen t.ε_hat t.δ_ε
  let abundant := certifiedAbundantThreshold t.s t.m t.r t.c_gen t.ε_hat t.δ_ε
  { scarceThreshold := scarce
    abundantThreshold := abundant
    usePrunedSESC := decide (t.n < scarce)
    N_compiled := compiledCapacity t.ρ_hat t.C_train
    D_compiled := compiledData t.ρ_hat t.C_train
    κ_flop_save := ((dimSESC t.s t.r : ℝ) / (dimMLP t.m t.d : ℝ)) * (t.B_RPM / t.B_SGD)
    activeBatchSize := certiRPMBatchSize t.κ_hat t.B_escape_batch t.B_basin_batch
    requiredAnchor := requiredAnchorStrength t.Λ_raw t.Λ_max
    rolloutErrorCeiling := t.ε_token / (1 - t.Λ_max) }

/-! ## Section 2: Component-Level Soundness of `synthesizeSystem` -/

@[simp]
theorem synthesizeSystem_scarceThreshold (t : ObservableTelemetry) :
    (synthesizeSystem t).scarceThreshold =
      certifiedScarceThreshold t.s t.m t.r t.c_gen t.ε_hat t.δ_ε :=
  rfl

@[simp]
theorem synthesizeSystem_abundantThreshold (t : ObservableTelemetry) :
    (synthesizeSystem t).abundantThreshold =
      certifiedAbundantThreshold t.s t.m t.r t.c_gen t.ε_hat t.δ_ε :=
  rfl

@[simp]
theorem synthesizeSystem_usePrunedSESC (t : ObservableTelemetry) :
    (synthesizeSystem t).usePrunedSESC =
      decide (t.n < certifiedScarceThreshold t.s t.m t.r t.c_gen t.ε_hat t.δ_ε) :=
  rfl

@[simp]
theorem synthesizeSystem_N_compiled (t : ObservableTelemetry) :
    (synthesizeSystem t).N_compiled = compiledCapacity t.ρ_hat t.C_train :=
  rfl

@[simp]
theorem synthesizeSystem_D_compiled (t : ObservableTelemetry) :
    (synthesizeSystem t).D_compiled = compiledData t.ρ_hat t.C_train :=
  rfl

@[simp]
theorem synthesizeSystem_κ_flop_save (t : ObservableTelemetry) :
    (synthesizeSystem t).κ_flop_save =
      ((dimSESC t.s t.r : ℝ) / (dimMLP t.m t.d : ℝ)) * (t.B_RPM / t.B_SGD) :=
  rfl

@[simp]
theorem synthesizeSystem_activeBatchSize (t : ObservableTelemetry) :
    (synthesizeSystem t).activeBatchSize =
      certiRPMBatchSize t.κ_hat t.B_escape_batch t.B_basin_batch :=
  rfl

@[simp]
theorem synthesizeSystem_requiredAnchor (t : ObservableTelemetry) :
    (synthesizeSystem t).requiredAnchor = requiredAnchorStrength t.Λ_raw t.Λ_max :=
  rfl

@[simp]
theorem synthesizeSystem_rolloutErrorCeiling (t : ObservableTelemetry) :
    (synthesizeSystem t).rolloutErrorCeiling = t.ε_token / (1 - t.Λ_max) :=
  rfl

/-- **Budget Feasibility & Co-Design FLOP Compression Soundness
(`synthesizeSystem_budget_and_compression_sound`)**:
For any valid positive telemetry `t` satisfying `0 < t.s < t.m`, `0 < t.r < t.k < t.d`,
`0 < t.ρ_hat`, `0 < t.C_train`, and `0 < t.B_RPM < t.B_SGD`:
1. `(synthesizeSystem t).N_compiled * (synthesizeSystem t).D_compiled = t.C_train`,
2. `(synthesizeSystem t).N_compiled / (synthesizeSystem t).D_compiled = t.ρ_hat`,
3. `0 < (synthesizeSystem t).κ_flop_save ∧ (synthesizeSystem t).κ_flop_save < 1`. -/
theorem synthesizeSystem_budget_and_compression_sound (t : ObservableTelemetry)
    (hs : 0 < t.s) (hsm : t.s < t.m) (hr : 0 < t.r) (hrk : t.r < t.k) (hkd : t.k < t.d)
    (hρ : 0 < t.ρ_hat) (hC : 0 < t.C_train)
    (hB_RPM : 0 < t.B_RPM) (hB_lt : t.B_RPM < t.B_SGD) :
    (synthesizeSystem t).N_compiled * (synthesizeSystem t).D_compiled = t.C_train ∧
      (synthesizeSystem t).N_compiled / (synthesizeSystem t).D_compiled = t.ρ_hat ∧
      (0 < (synthesizeSystem t).κ_flop_save ∧ (synthesizeSystem t).κ_flop_save < 1) := by
  simp only [synthesizeSystem_N_compiled, synthesizeSystem_D_compiled,
    synthesizeSystem_κ_flop_save]
  have hbudget := compiledCapacity_mul_compiledData_eq_compute hρ hC
  obtain ⟨hdim1, hdim2, hdim3⟩ := four_tier_dimension_strict_hierarchy hs hsm hr hrk hkd
  have hdim_lt : dimSESC t.s t.r < dimMLP t.m t.d := (hdim1.trans hdim2).trans hdim3
  have hp_SESC_pos : (0 : ℝ) < (dimSESC t.s t.r : ℝ) := Nat.cast_pos.mpr (Nat.mul_pos hs hr)
  have hp_lt : (dimSESC t.s t.r : ℝ) < (dimMLP t.m t.d : ℝ) := Nat.cast_lt.mpr hdim_lt
  have hp_MLP_pos : (0 : ℝ) < (dimMLP t.m t.d : ℝ) := hp_SESC_pos.trans hp_lt
  have hB_SGD_pos : 0 < t.B_SGD := hB_RPM.trans hB_lt
  have hfac_p_pos : 0 < (dimSESC t.s t.r : ℝ) / (dimMLP t.m t.d : ℝ) :=
    div_pos hp_SESC_pos hp_MLP_pos
  have hfac_B_pos : 0 < t.B_RPM / t.B_SGD := div_pos hB_RPM hB_SGD_pos
  have hfac_p_lt1 : (dimSESC t.s t.r : ℝ) / (dimMLP t.m t.d : ℝ) < 1 :=
    (div_lt_one hp_MLP_pos).mpr hp_lt
  have hfac_B_lt1 : t.B_RPM / t.B_SGD < 1 := (div_lt_one hB_SGD_pos).mpr hB_lt
  have hκ_pos : 0 < ((dimSESC t.s t.r : ℝ) / (dimMLP t.m t.d : ℝ)) * (t.B_RPM / t.B_SGD) :=
    mul_pos hfac_p_pos hfac_B_pos
  have hκ_lt1 : ((dimSESC t.s t.r : ℝ) / (dimMLP t.m t.d : ℝ)) * (t.B_RPM / t.B_SGD) < 1 := by
    calc
      ((dimSESC t.s t.r : ℝ) / (dimMLP t.m t.d : ℝ)) * (t.B_RPM / t.B_SGD)
        < 1 * (t.B_RPM / t.B_SGD) := mul_lt_mul_of_pos_right hfac_p_lt1 hfac_B_pos
      _ = t.B_RPM / t.B_SGD := one_mul _
      _ < 1 := hfac_B_lt1
  exact ⟨hbudget.1, hbudget.2, hκ_pos, hκ_lt1⟩

/-- **Runtime Stabilization Soundness of `synthesizeSystem`
(`synthesizeSystem_runtime_stabilization_sound`)**:
When `0 < t.Λ_raw`, `0 ≤ t.Λ_max < 1`, `t.Λ_max ≤ t.Λ_raw`, and `0 ≤ t.ε_token`, deploying the
synthesized anchor strength `α_anchor = (synthesizeSystem t).requiredAnchor` guarantees that
`0 ≤ (synthesizeSystem t).requiredAnchor ≤ 1`, the damped step sensitivity equals `t.Λ_max`, and
for **all** reasoning horizons `H : ℕ`:
`autoregressiveRolloutError (dampedSensitivity (synthesizeSystem t).requiredAnchor t.Λ_raw) t.ε_token H ≤ (synthesizeSystem t).rolloutErrorCeiling`. -/
theorem synthesizeSystem_runtime_stabilization_sound (t : ObservableTelemetry)
    (hΛ_pos : 0 < t.Λ_raw)
    (hΛ_max_nonneg : 0 ≤ t.Λ_max)
    (hΛ_max_lt : t.Λ_max < 1)
    (hΛ_le : t.Λ_max ≤ t.Λ_raw)
    (hε : 0 ≤ t.ε_token) :
    0 ≤ (synthesizeSystem t).requiredAnchor ∧
      (synthesizeSystem t).requiredAnchor ≤ 1 ∧
      dampedSensitivity (synthesizeSystem t).requiredAnchor t.Λ_raw = t.Λ_max ∧
      ∀ H : ℕ,
        autoregressiveRolloutError
            (dampedSensitivity (synthesizeSystem t).requiredAnchor t.Λ_raw) t.ε_token H ≤
          (synthesizeSystem t).rolloutErrorCeiling := by
  simp only [synthesizeSystem_requiredAnchor, synthesizeSystem_rolloutErrorCeiling]
  have hunit := requiredAnchorStrength_mem_unit_interval hΛ_pos hΛ_max_nonneg hΛ_le
  have hcal := dampedSensitivity_at_requiredAnchorStrength t.Λ_max (ne_of_gt hΛ_pos)
  refine ⟨hunit.1, hunit.2, hcal, fun H => ?_⟩
  exact (runtime_damped_rollout_horizon_free_certificate
    hΛ_pos hΛ_max_nonneg hΛ_max_lt le_rfl hunit.2 hε H).2.2.2.2

/-! ## Section 3: The Grand Unified Soundness Theorem of OpenTheory -/

/-- **The Grand Unified Soundness Theorem of OpenTheory (`grand_unified_open_theory_soundness`)**:
Unites all four foundational pillars of `OpenTheory` into a single machine-checked guarantee for
the executable synthesis pipeline `cert = synthesizeSystem t`:

1. **Pillar 1 — Zero-Shot Pre-Training Compiler Soundness**:
   - The synthesized thresholds bracket the true unobserved oracle threshold:
     `cert.scarceThreshold ≤ criticalSampleThreshold t.s t.m t.r t.c_gen ε_prune ≤ cert.abundantThreshold`.
   - When `t.n < cert.scarceThreshold`, pruned `SESC` (`t.s * t.r`) strictly beats `SubspaceNet`
     (`t.m * t.r`) on the true population risk.
   - The compiled Chinchilla allocation satisfies `cert.N_compiled * cert.D_compiled = t.C_train`,
     `cert.N_compiled / cert.D_compiled = t.ρ_hat`, and exact quadratic completion-of-squares
     regret `(√(A * cert.D_compiled) - √(B_opt * cert.N_compiled))^2 / t.C_train`.
   - The `(SESC, RPM)` co-design compresses required pre-training compute by
     `0 < cert.κ_flop_save < 1`, matching the `(MLP, SGD)` Pareto frontier at full budget `t.C_train`.

2. **Pillar 2 — Self-Tuning `CertiRPM` Non-Divergence & Pure Quadratic Regret**:
   - Under upper-confidence sharpness `0 < S i ≤ S_hat i`, `pluginSharpnessStep` is guaranteed
     100% safe and non-divergent (`0 ≤ optimizerNetProgress`) with pure quadratic estimation regret
     `∑ i, (((g i)^2)^2 / (2 * S i * M_i)) * ((S_hat i - S i) / S_hat i)^2`.

3. **Pillar 3 — Test-Time Reasoning Dominance over `M_giant×` Giant Models**:
   - Whenever `t.α_model / t.N_small < t.β_reason` and `1 < t.M_giant`, the small reasoning model
     `(t.N_small, T_inf = t.M_giant)` strictly beats the `t.M_giant×` larger 1-step model
     `(t.M_giant * t.N_small, 1)` at identical per-query inference FLOPs
     `t.N_small * t.M_giant = (t.M_giant * t.N_small) * 1`.

4. **Pillar 4 — Runtime Triple-Shield Horizon-Free & OOD Guarantee**:
   - Deploying `cert.requiredAnchor` bounds the `H`-step autoregressive rollout error by
     `cert.rolloutErrorCeiling` for **all** reasoning horizons `H : ℕ`, while `SESC` incurs
     identically zero OOD shift penalty under any orthogonal spectral-bulk perturbation `η_bulk`. -/
theorem grand_unified_open_theory_soundness
    {p : ℕ}
    (t : ObservableTelemetry) [NeZero t.d]
    (R_inf c_arch ε_prune A B_opt R_opt horizonAmp Δ_OOD_sq : ℝ)
    -- Pillar 1 hypotheses (Pre-Training Zero-Shot Compiler)
    (hs : 0 < t.s) (hsm : t.s < t.m) (hr : 0 < t.r) (hrk : t.r < t.k) (hkd : t.k < t.d)
    (hc_gen : 0 < t.c_gen) (hc_arch : 0 < c_arch) (hε : 0 < ε_prune)
    (hδ_nonneg : 0 ≤ t.δ_ε) (hδ_lt : t.δ_ε < t.ε_hat) (h_est : |t.ε_hat - ε_prune| ≤ t.δ_ε)
    (hρ : 0 < t.ρ_hat) (hA : 0 < A) (hB : 0 < B_opt)
    (hB_RPM : 0 < t.B_RPM) (hB_lt : t.B_RPM < t.B_SGD)
    (hn : 0 < t.n) (hC : 0 < t.C_train) (hamp : 0 < horizonAmp)
    -- Pillar 2 hypotheses (Self-Tuning CertiRPM Optimizer)
    (S S_hat : Fin p → ℝ) (g : Parameters p) (β : ℝ) (σ_coord : Fin p → ℝ)
    (hS : ∀ i, 0 < S i) (hle : ∀ i, S i ≤ S_hat i)
    (hM : ∀ i, 0 < totalCoordMoment g β σ_coord (t.B_basin_batch : ℝ) i)
    -- Pillar 3 hypotheses (Test-Time Reasoning Dominance)
    (hN_small : 0 < t.N_small) (hM_giant : 1 < t.M_giant)
    (h_reason_dom : t.α_model / t.N_small < t.β_reason)
    -- Pillar 4 hypotheses (Runtime Horizon-Free Rollout & OOD Bulk Immunity)
    (hΛ_pos : 0 < t.Λ_raw) (hΛ_max_nonneg : 0 ≤ t.Λ_max) (hΛ_max_lt : t.Λ_max < 1)
    (hΛ_le : t.Λ_max ≤ t.Λ_raw) (hε_tok : 0 ≤ t.ε_token)
    (net : SpectralSparseCircuit t.r t.m)
    (embed : Fin t.k → Fin t.d)
    (B_spec : Matrix (Fin t.r) (Fin t.k) ℝ)
    (x η_bulk : Fin t.d → ℝ)
    (h_ortho : ∀ q : Fin t.d, ∀ l : Fin t.r,
      patchProject B_spec (cyclicPatch embed η_bulk q) l = 0) :
    let cert := synthesizeSystem t
    let n_crit := criticalSampleThreshold t.s t.m t.r t.c_gen ε_prune
    -- (1) Pillar 1: Zero-Shot Pre-Training Compiler Soundness
    ((cert.scarceThreshold ≤ n_crit ∧ n_crit ≤ cert.abundantThreshold) ∧
      (t.n < cert.scarceThreshold →
        predictiveTaskRisk ε_prune (dimSESC t.s t.r : ℝ) t.c_gen t.n R_opt horizonAmp 0 Δ_OOD_sq <
          predictiveTaskRisk 0 (dimSubspace t.m t.r : ℝ) t.c_gen t.n R_opt horizonAmp 0 Δ_OOD_sq) ∧
      cert.N_compiled * cert.D_compiled = t.C_train ∧
      cert.N_compiled / cert.D_compiled = t.ρ_hat ∧
      scalingRisk R_inf A B_opt cert.N_compiled cert.D_compiled -
          paretoComputeRisk R_inf A B_opt t.C_train =
        (Real.sqrt (A * cert.D_compiled) - Real.sqrt (B_opt * cert.N_compiled)) ^ 2 / t.C_train ∧
      0 < cert.κ_flop_save ∧ cert.κ_flop_save < 1 ∧
      paretoComputeRisk R_inf (c_arch * (dimSESC t.s t.r : ℝ)) t.B_RPM
          (cert.κ_flop_save * t.C_train) =
        paretoComputeRisk R_inf (c_arch * (dimMLP t.m t.d : ℝ)) t.B_SGD t.C_train) ∧
    -- (2) Pillar 2: Self-Tuning CertiRPM Non-Divergence & Pure Quadratic Regret
    (0 ≤ optimizerNetProgress
        (pluginSharpnessStep S_hat g β σ_coord (t.B_basin_batch : ℝ))
        S g β σ_coord (t.B_basin_batch : ℝ) ∧
      optimizerNetProgress (rpmOptimalStep S g β σ_coord (t.B_basin_batch : ℝ))
          S g β σ_coord (t.B_basin_batch : ℝ) -
        optimizerNetProgress (pluginSharpnessStep S_hat g β σ_coord (t.B_basin_batch : ℝ))
          S g β σ_coord (t.B_basin_batch : ℝ) =
        ∑ i : Fin p,
          (((g i) ^ 2) ^ 2 /
              (2 * S i * totalCoordMoment g β σ_coord (t.B_basin_batch : ℝ) i)) *
            ((S_hat i - S i) / S_hat i) ^ 2) ∧
    -- (3) Pillar 3: Test-Time Reasoning Dominance over M_giant× Giant 1-Step Models
    (t.N_small * t.M_giant = (t.M_giant * t.N_small) * 1 ∧
      inferenceScalingRisk R_inf t.α_model t.β_reason t.N_small t.M_giant <
        inferenceScalingRisk R_inf t.α_model t.β_reason (t.M_giant * t.N_small) 1) ∧
    -- (4) Pillar 4: Runtime Horizon-Free Rollout Stability & Zero OOD Bulk Shift Penalty
    ((∀ H : ℕ,
        autoregressiveRolloutError
            (dampedSensitivity cert.requiredAnchor t.Λ_raw) t.ε_token H ≤
          cert.rolloutErrorCeiling) ∧
      sescCircuitForward net embed B_spec (x + η_bulk) =
        sescCircuitForward net embed B_spec x ∧
      (sescCircuitForward net embed B_spec (x + η_bulk) -
        sescCircuitForward net embed B_spec x) ^ 2 = 0) := by
  intro cert n_crit
  -- Pillar 1 proof via zero_shot_compiler_end_to_end_guarantee
  have hp1 := zero_shot_compiler_end_to_end_guarantee R_inf
    hs hsm hr hrk hkd hc_gen hc_arch hε hδ_nonneg hδ_lt h_est
    hρ hA hB hB_RPM hB_lt hn hC hamp
    (R_opt := R_opt) (Δ_OOD_sq := Δ_OOD_sq)
  -- Pillar 2 proof via CertiRPM upper-confidence sharpness guarantees
  have hS_hat : ∀ i, 0 < S_hat i := fun i => (hS i).trans_le (hle i)
  have hp2_safe :=
    (upper_confidence_sharpness_guarantees_positive_progress
      S_hat S g β σ_coord (t.B_basin_batch : ℝ) hS hle hM).2.2
  have hp2_regret :=
    (pluginSharpness_exact_progress_ratio
      S_hat S g β σ_coord (t.B_basin_batch : ℝ) hS_hat hS hM).1
  -- Pillar 3 proof via small_reasoning_model_beats_giant_one_step_at_equal_inference_flops
  have hp3_equiv := small_reasoning_model_beats_giant_one_step_at_equal_inference_flops
    R_inf t.α_model t.β_reason hN_small hM_giant
  have hp3_lt := hp3_equiv.2.mpr h_reason_dom
  -- Pillar 4 proof via synthesizeSystem_runtime_stabilization_sound and SESC OOD bulk invariance
  have hp4_rollout :=
    (synthesizeSystem_runtime_stabilization_sound t hΛ_pos hΛ_max_nonneg hΛ_max_lt hΛ_le hε_tok).2.2.2
  have hp4_ood_eq := (sesc_orthogonal_bulk_invariance net embed B_spec x η_bulk h_ortho).1
  have hp4_ood_sq := (sesc_zero_ood_shift_penalty net embed B_spec x η_bulk h_ortho).1
  refine ⟨?_, ⟨hp2_safe, hp2_regret⟩, ⟨hp3_equiv.1, hp3_lt⟩,
    ⟨hp4_rollout, hp4_ood_eq, hp4_ood_sq⟩⟩
  exact ⟨hp1.1, hp1.2.1.1, hp1.2.2.1.1, hp1.2.2.1.2.1, hp1.2.2.1.2.2,
    hp1.2.2.2.1, hp1.2.2.2.2.1, hp1.2.2.2.2.2⟩

end OpenTheory.ExecutableSynthesisEngine
