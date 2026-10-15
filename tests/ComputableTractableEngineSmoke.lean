import OpenTheory.Certification.ComputableTractableEngine

open OpenTheory
open OpenTheory.CertiRPM
open OpenTheory.SpectralSparseCircuit
open OpenTheory.AutoregressiveOOD
open OpenTheory.ZeroShotCompiler
open OpenTheory.RuntimeReasoningGuardrail
open OpenTheory.ComputableTractableEngine

attribute [local semireducible] Rat.add Rat.sub Rat.mul Rat.inv

-- Section 1: Formal Algorithmic Tractability & O(p) vs. O(p^3) Complexity Calculus
#check @certiRPMProbeFLOPs
#check @certiRPMProbeMemory
#check @fullHessianPrecondFLOPs
#check @fullHessianPrecondMemory
#check @tractabilitySpeedupRatio
#check @tractabilitySpeedupRatio_eq
#check @certiRPM_linear_vs_cubic_tractability

-- Section 2: 100% Computable Rational (ℚ) Synthesis, Optimizer & Safety Engine
#check @computableRPMStepQ
#check @computableScarceThresholdQ
#check @computableAbundantThresholdQ
#check @computableFlopSavingsRatioQ
#check @computableAnchorStrengthQ
#check @computableDampedSensitivityQ
#check @computableBarrierMarginQ
#check @computableRolloutCeilingQ
#check @ComputableTelemetryQ
#check @ComputablePrescriptionQ
#check @runTractableSynthesisQ

-- Section 3: The Exact ℚ → ℝ Computable-to-Real Soundness Bridge Theorems
#check @flopSavingsRatio
#check @computableAnchorStrengthQ_cast_eq
#check @computableDampedSensitivityQ_cast_eq
#check @computableFlopSavingsRatioQ_cast_eq
#check @computableRolloutCeilingQ_cast_eq
#check @computableScarceThresholdQ_cast_eq
#check @computable_engine_certifies_real_deployment_guarantees

/-- Concrete computable rational telemetry profile for kernel-reduction (`rfl`) smoke tests. -/
def sampleTelemetry : ComputableTelemetryQ where
  s := 2
  m := 8
  r := 3
  d := 16
  k_probes := 2
  p_params := 1000000
  n := 120
  c_gen := 1
  ε_hat := 1 / 10
  δ_ε := 1 / 50
  S_hat := 5
  g := 1
  β := 1 / 2
  σ_coord := 4
  B_batch := 12
  B_RPM := 1
  B_SGD := 4
  Λ_raw := 2
  Λ_max := 1 / 2
  δ_Λ := 1 / 20
  ε_token := 1 / 10
  B_safe := 5

-- Smoke test 1: Pure kernel reduction (`by rfl`) of `runTractableSynthesisQ` on `sampleTelemetry`
-- (proving that the entire rational synthesis engine is 100% computable with zero `noncomputable`!).
example : (runTractableSynthesisQ sampleTelemetry).anchorStrength = 3 / 4 := by rfl
example : (runTractableSynthesisQ sampleTelemetry).dampedSensitivity = 1 / 2 := by rfl
example : (runTractableSynthesisQ sampleTelemetry).rolloutCeiling = 1 / 5 := by rfl
example : (runTractableSynthesisQ sampleTelemetry).scarceThreshold = 150 := by rfl
example : (runTractableSynthesisQ sampleTelemetry).abundantThreshold = 225 := by rfl
example : (runTractableSynthesisQ sampleTelemetry).flopSavingsRatio = 3 / 256 := by rfl
example : (runTractableSynthesisQ sampleTelemetry).rpmStepSize = 1 / 10 := by rfl
example : (runTractableSynthesisQ sampleTelemetry).selectedDim = 6 := by rfl
example : (runTractableSynthesisQ sampleTelemetry).isScarceRegime = true := by rfl
example : (runTractableSynthesisQ sampleTelemetry).isAbundantRegime = false := by rfl
example : (runTractableSynthesisQ sampleTelemetry).isFlopSaving = true := by rfl

-- Smoke test 2: For a p = 10^6 parameter model with k_probes = 2, CertiRPM achieves an exact
-- 250,000,000,000× (2.5 × 10^11×) per-step FLOP speedup over O(p^3) Hessian preconditioning,
-- verified by pure kernel reduction `rfl`!
example : tractabilitySpeedupRatio 2 1000000 = 250000000000 := by rfl
example : (runTractableSynthesisQ sampleTelemetry).speedupVsCubic = 250000000000 := by rfl

-- Smoke test 3: Formal O(p) vs. O(p^3) tractability theorem instantiated at p = 10^6, k_probes = 2.
example :
    certiRPMProbeMemory 1000000 < fullHessianPrecondMemory 1000000 ∧
      certiRPMProbeFLOPs 2 1000000 < fullHessianPrecondFLOPs 1000000 ∧
      tractabilitySpeedupRatio 2 1000000 = 250000000000 := by
  have h := certiRPM_linear_vs_cubic_tractability (k_probes := 2) (p := 1000000)
    (by decide) (by decide) (by decide)
  exact ⟨h.1, h.2.2.2.1, by rfl⟩

-- Smoke test 4: End-to-end ℚ → ℝ soundness bridge on `sampleTelemetry`:
-- Because `(runTractableSynthesisQ sampleTelemetry).isFlopSaving = true` holds by `rfl`,
-- the real-valued FLOP savings bound `< 1` and horizon-free rollout bound `≤ 1 / 5` hold for all H : ℕ!
example (H : ℕ) :
    flopSavingsRatio sampleTelemetry.s sampleTelemetry.r sampleTelemetry.m sampleTelemetry.d
        (sampleTelemetry.B_RPM : ℝ) (sampleTelemetry.B_SGD : ℝ) < 1 ∧
      autoregressiveRolloutError
          (dampedSensitivity
            (((runTractableSynthesisQ sampleTelemetry).anchorStrength : ℚ) : ℝ)
            (sampleTelemetry.Λ_raw : ℝ))
          (sampleTelemetry.ε_token : ℝ) H ≤ (1 / 5 : ℝ) := by
  have hbridge := computable_engine_certifies_real_deployment_guarantees sampleTelemetry
    (by decide) (by decide) (by decide) (by decide) (by decide) rfl
  have hceil : (((runTractableSynthesisQ sampleTelemetry).rolloutCeiling : ℚ) : ℝ) = (1 / 5 : ℝ) := by
    norm_num [runTractableSynthesisQ, sampleTelemetry, computableRolloutCeilingQ]
  refine ⟨hbridge.1, ?_⟩
  simpa [hceil] using hbridge.2.2 H

-- Axiom audit
#print axioms OpenTheory.ComputableTractableEngine.certiRPM_linear_vs_cubic_tractability
#print axioms OpenTheory.ComputableTractableEngine.runTractableSynthesisQ
#print axioms OpenTheory.ComputableTractableEngine.computable_engine_certifies_real_deployment_guarantees
