import OpenTheory.Certification.ExecutableSynthesisEngine

noncomputable section

open OpenTheory
open OpenTheory.CertiRPM
open OpenTheory.ComputeOptimalSynthesis
open OpenTheory.SpectralSparseCircuit
open OpenTheory.ResidualPreconditionedMomentum
open OpenTheory.PredictiveModelPrescription
open OpenTheory.ZeroShotCompiler
open OpenTheory.TestTimeScaling
open OpenTheory.AutoregressiveOOD
open OpenTheory.RuntimeReasoningGuardrail
open OpenTheory.ExecutableSynthesisEngine

-- Section 1: Observable Telemetry Input & Synthesized System Certificate Bundles
#check @ObservableTelemetry
#check @SynthesizedCertificate
#check @synthesizeSystem

-- Section 2: Component-Level Soundness of `synthesizeSystem`
#check @synthesizeSystem_scarceThreshold
#check @synthesizeSystem_abundantThreshold
#check @synthesizeSystem_usePrunedSESC
#check @synthesizeSystem_N_compiled
#check @synthesizeSystem_D_compiled
#check @synthesizeSystem_κ_flop_save
#check @synthesizeSystem_activeBatchSize
#check @synthesizeSystem_requiredAnchor
#check @synthesizeSystem_rolloutErrorCeiling
#check @synthesizeSystem_budget_and_compression_sound
#check @synthesizeSystem_runtime_stabilization_sound

-- Section 3: The Grand Unified Soundness Theorem of OpenTheory
#check @grand_unified_open_theory_soundness

/-- Concrete sample telemetry probe in strict-saddle mode (`κ_hat = -2 < 0`). -/
def sampleTelemetrySaddle : ObservableTelemetry where
  s := 2
  m := 8
  r := 3
  k := 6
  d := 16
  n := 120
  c_gen := 1
  ε_hat := 1 / 10
  δ_ε := 1 / 50
  ρ_hat := 4 / 9
  C_train := 36
  κ_hat := -2
  η_escape := 1 / 10
  B_RPM := 1
  B_SGD := 4
  B_escape_batch := 8
  B_basin_batch := 64
  α_model := 2
  β_reason := 5
  N_small := 4
  M_giant := 10
  Λ_raw := 2
  Λ_max := 1 / 2
  ε_token := 1 / 10

/-- Concrete sample telemetry probe in PL-basin mode (`κ_hat = 3 ≥ 0`). -/
def sampleTelemetryBasin : ObservableTelemetry :=
  { sampleTelemetrySaddle with κ_hat := 3 }

-- Smoke test 1: End-to-end numerical evaluation of `synthesizeSystem` on `sampleTelemetrySaddle`
-- and `sampleTelemetryBasin`, verifying:
-- - `scarceThreshold = 150` and `abundantThreshold = 225`
-- - `usePrunedSESC = true` (since `n = 120 < 150`)
-- - `κ_flop_save = 3 / 256`
-- - `requiredAnchor = 3 / 4` and `rolloutErrorCeiling = 1 / 5`
-- - `activeBatchSize = 8` in saddle mode (`κ_hat = -2`) and `64` in basin mode (`κ_hat = 3`).
example :
    (synthesizeSystem sampleTelemetrySaddle).scarceThreshold = 150 ∧
      (synthesizeSystem sampleTelemetrySaddle).abundantThreshold = 225 ∧
      (synthesizeSystem sampleTelemetrySaddle).usePrunedSESC = true ∧
      (synthesizeSystem sampleTelemetrySaddle).κ_flop_save = 3 / 256 ∧
      (synthesizeSystem sampleTelemetrySaddle).requiredAnchor = 3 / 4 ∧
      (synthesizeSystem sampleTelemetrySaddle).rolloutErrorCeiling = 1 / 5 ∧
      (synthesizeSystem sampleTelemetrySaddle).activeBatchSize = 8 ∧
      (synthesizeSystem sampleTelemetryBasin).activeBatchSize = 64 := by
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · norm_num [synthesizeSystem, sampleTelemetrySaddle, certifiedScarceThreshold,
      dimSubspace, dimSESC]
  · norm_num [synthesizeSystem, sampleTelemetrySaddle, certifiedAbundantThreshold,
      dimSubspace, dimSESC]
  · norm_num [synthesizeSystem, sampleTelemetrySaddle, certifiedScarceThreshold,
      dimSubspace, dimSESC]
  · norm_num [synthesizeSystem, sampleTelemetrySaddle, dimSESC, dimMLP]
  · norm_num [synthesizeSystem, sampleTelemetrySaddle, requiredAnchorStrength]
  · norm_num [synthesizeSystem, sampleTelemetrySaddle]
  · norm_num [synthesizeSystem, sampleTelemetrySaddle, certiRPMBatchSize]
  · norm_num [synthesizeSystem, sampleTelemetryBasin, sampleTelemetrySaddle, certiRPMBatchSize]

-- Smoke test 2: Component-level soundness of `synthesizeSystem` on `sampleTelemetrySaddle`
-- (budget feasibility `N_compiled * D_compiled = 36`, ratio `N_compiled / D_compiled = 4 / 9`,
-- compression `0 < κ_flop_save < 1`, and horizon-free rollout bound `≤ 1 / 5` for all `H : ℕ`).
example (H : ℕ) :
    (synthesizeSystem sampleTelemetrySaddle).N_compiled *
        (synthesizeSystem sampleTelemetrySaddle).D_compiled = 36 ∧
      (synthesizeSystem sampleTelemetrySaddle).N_compiled /
        (synthesizeSystem sampleTelemetrySaddle).D_compiled = 4 / 9 ∧
      0 < (synthesizeSystem sampleTelemetrySaddle).κ_flop_save ∧
      (synthesizeSystem sampleTelemetrySaddle).κ_flop_save < 1 ∧
      autoregressiveRolloutError
          (dampedSensitivity (synthesizeSystem sampleTelemetrySaddle).requiredAnchor
            sampleTelemetrySaddle.Λ_raw)
          sampleTelemetrySaddle.ε_token H ≤ 1 / 5 := by
  have hbc := synthesizeSystem_budget_and_compression_sound sampleTelemetrySaddle
    (by decide) (by decide) (by decide) (by decide) (by decide)
    (by norm_num [sampleTelemetrySaddle]) (by norm_num [sampleTelemetrySaddle])
    (by norm_num [sampleTelemetrySaddle]) (by norm_num [sampleTelemetrySaddle])
  have hrt := (synthesizeSystem_runtime_stabilization_sound sampleTelemetrySaddle
    (by norm_num [sampleTelemetrySaddle]) (by norm_num [sampleTelemetrySaddle])
    (by norm_num [sampleTelemetrySaddle]) (by norm_num [sampleTelemetrySaddle])
    (by norm_num [sampleTelemetrySaddle])).2.2.2 H
  have hceil : (synthesizeSystem sampleTelemetrySaddle).rolloutErrorCeiling = 1 / 5 := by
    norm_num [synthesizeSystem, sampleTelemetrySaddle]
  rw [hceil] at hrt
  exact ⟨hbc.1, hbc.2.1, hbc.2.2.1, hbc.2.2.2, hrt⟩

-- Axiom audit of the Grand Unified Soundness Theorem
#print axioms OpenTheory.ExecutableSynthesisEngine.grand_unified_open_theory_soundness
