import OpenTheory.Generalization.TransformerGeneralization

noncomputable section

open TorchLean MeasureTheory StatLean.StatisticalLearning
open OpenTheory OpenTheory.Attention OpenTheory.AttentionAblation
open OpenTheory.AttentionLogitStability OpenTheory.InContextRetrieval
open OpenTheory.TrajectoryGeneralization OpenTheory.BallCover OpenTheory.Covering
open OpenTheory.TransformerGeneralization
open scoped BigOperators

variable {E F : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
  [NormedAddCommGroup F] [NormedSpace ℝ F]

-- Section 1: Multi-Layer Transformer Parameter-to-Output Stability & Depth Propagation
#check OpenTheory.TransformerGeneralization.layerParameterPerturbationEnvelope
#check OpenTheory.TransformerGeneralization.transformerStackLip
#check OpenTheory.TransformerGeneralization.transformerPerturbationEnvelope_smul
#check OpenTheory.TransformerGeneralization.layerParameterPerturbationEnvelope_eq_mul
#check OpenTheory.TransformerGeneralization.transformerStackLip_nonneg
#check OpenTheory.TransformerGeneralization.transformer_stack_param_perturbation_le
#check OpenTheory.TransformerGeneralization.residualPerturbationEnvelope_mono
#check OpenTheory.TransformerGeneralization.residualPerturbationEnvelope_const_le
#check OpenTheory.TransformerGeneralization.transformer_depth_scaled_lip_bound

-- Section 2: Transformer Hypothesis Class Covering Numbers & Uniform Generalization Bounds
#check OpenTheory.TransformerGeneralization.transformerParamLoss
#check OpenTheory.TransformerGeneralization.transformerLossLipConst
#check OpenTheory.TransformerGeneralization.transformerLossLipConst_nonneg
#check OpenTheory.TransformerGeneralization.transformerParamLoss_lipschitz
#check OpenTheory.TransformerGeneralization.transformerStack_paramLoss_lipschitz
#check OpenTheory.TransformerGeneralization.transformerBall_loss_cover
#check OpenTheory.TransformerGeneralization.transformer_uniform_deviation_tail
#check OpenTheory.TransformerGeneralization.transformer_approx_erm_excess_risk_tail

-- Section 3: End-to-End In-Context Learning (ICL) Excess Population Risk Decomposition
#check OpenTheory.TransformerGeneralization.iclTotalRiskBound
#check OpenTheory.TransformerGeneralization.icl_end_to_end_excess_risk_bound
#check OpenTheory.TransformerGeneralization.icl_total_risk_monotone_in_steps
#check OpenTheory.TransformerGeneralization.icl_end_to_end_target_accuracy

-- Section 4: Mechanistic Interpretability & Safety Alignment Control
#check OpenTheory.TransformerGeneralization.safety_intervention_tradeoff_bound
#check OpenTheory.TransformerGeneralization.ablated_multiHead_unsafe_proj_zero
#check OpenTheory.TransformerGeneralization.orthogonal_circuit_ablation_zero_unsafe_and_risk_bound

-- Smoke test 1: Exact numerical verification of 2-layer transformerPerturbationEnvelope_smul
example :
    let c : Fin 2 → ℝ := ![1, 2]
    let L_param : Fin 2 → ℝ := ![3, 4]
    residualPerturbationEnvelope 2 c (fun t => L_param t * 5) =
      transformerStackLip 2 c L_param * 5 ∧
    transformerStackLip 2 c L_param = 13 := by
  refine ⟨transformerPerturbationEnvelope_smul 2 ![1, 2] ![3, 4] 5, ?_⟩
  norm_num [transformerStackLip, residualPerturbationEnvelope, Fin.last]

-- Smoke test 2: Depth-scaled 1/L residual Transformer stack Lipschitz bound at arbitrary depth L ≥ 1
example {L : ℕ} (hL : 0 < L) (c L_param : Fin L → ℝ) {c_0 L_0 : ℝ}
    (hc0 : 0 ≤ c_0) (hL0 : 0 ≤ L_0)
    (hc_nonneg : ∀ t, 0 ≤ c t) (hL_nonneg : ∀ t, 0 ≤ L_param t)
    (hc_bound : ∀ t, c t ≤ c_0 / (L : ℝ)) (hL_bound : ∀ t, L_param t ≤ L_0 / (L : ℝ)) :
    transformerStackLip L c L_param ≤ L_0 * Real.exp c_0 :=
  (transformer_depth_scaled_lip_bound hL c L_param hc0 hL0 hc_nonneg hL_nonneg hc_bound hL_bound).2

-- Smoke test 3: Monotonicity of ICL end-to-end risk bound in alignment steps T₁ ≤ T₂
example (R_oracle ε_gen : ℝ) {n : ℕ} {D_V γ₀ η Δ_g : ℝ} {T₁ T₂ : ℕ}
    (hn : 1 ≤ n) (hD_V : 0 ≤ D_V) (hη : 0 ≤ η) (hΔ_g : 0 ≤ Δ_g) (hT : T₁ ≤ T₂) :
    iclTotalRiskBound R_oracle n D_V γ₀ η Δ_g T₂ ε_gen ≤
      iclTotalRiskBound R_oracle n D_V γ₀ η Δ_g T₁ ε_gen :=
  icl_total_risk_monotone_in_steps R_oracle ε_gen hn hD_V hη hΔ_g hT

-- Smoke test 4: Exact safety eradication when retained attention heads are orthogonal to P_unsafe
example {H n : ℕ} (P_unsafe : F →L[ℝ] F)
    (scores : Fin H → Tensor ℝ [n]) (mask : Fin H → Tensor Bool [n])
    (values : Fin H → Fin n → E) (proj : Fin H → (E →L[ℝ] F))
    (keep : Finset (Fin H))
    (horth : ∀ h ∈ keep, P_unsafe (headContribution scores mask values proj h) = 0) :
    P_unsafe (ablatedMultiHeadOutput scores mask values proj keep) = 0 :=
  (ablated_multiHead_unsafe_proj_zero P_unsafe scores mask values proj keep horth).1

-- Axiom audits (ensuring no sorryAx)
#print axioms OpenTheory.TransformerGeneralization.transformer_stack_param_perturbation_le
#print axioms OpenTheory.TransformerGeneralization.transformerPerturbationEnvelope_smul
#print axioms OpenTheory.TransformerGeneralization.transformer_depth_scaled_lip_bound
#print axioms OpenTheory.TransformerGeneralization.transformerParamLoss_lipschitz
#print axioms OpenTheory.TransformerGeneralization.transformerBall_loss_cover
#print axioms OpenTheory.TransformerGeneralization.transformer_uniform_deviation_tail
#print axioms OpenTheory.TransformerGeneralization.transformer_approx_erm_excess_risk_tail
#print axioms OpenTheory.TransformerGeneralization.icl_end_to_end_excess_risk_bound
#print axioms OpenTheory.TransformerGeneralization.icl_total_risk_monotone_in_steps
#print axioms OpenTheory.TransformerGeneralization.icl_end_to_end_target_accuracy
#print axioms OpenTheory.TransformerGeneralization.safety_intervention_tradeoff_bound
#print axioms OpenTheory.TransformerGeneralization.orthogonal_circuit_ablation_zero_unsafe_and_risk_bound
