import OpenTheory.Architectures.Attention
import OpenTheory.Architectures.AttentionAblation
import OpenTheory.Architectures.AttentionLogitStability
import OpenTheory.Architectures.InContextRetrieval
import OpenTheory.Dynamics.DeepResidualScaling
import OpenTheory.Generalization.TrajectoryGeneralization
import OpenTheory.Generalization.Covering
import OpenTheory.Generalization.BallCover
import OpenTheory.Generalization.PredictionGap
import OpenTheory.Generalization.PopulationBounds

/-!
# End-to-End Transformer Parameter Stability, Covering Generalization, In-Context Risk, and Mechanistic Safety Alignment

This module unifies four core pillars of Transformer statistical learning and mechanistic safety
theory in `OpenTheory.TransformerGeneralization`:

1. **Multi-Layer Transformer Parameter-to-Output Stability & Depth Propagation**:
   - Defines `layerParameterPerturbationEnvelope` and `transformerStackLip L c L_param` via
     `AttentionAblation.residualPerturbationEnvelope`.
   - Proves `transformerPerturbationEnvelope_smul` (linear homogeneity of the residual perturbation
     envelope in the per-layer perturbation budget) and `transformerStackLip_nonneg`.
   - Proves `transformer_stack_param_perturbation_le`: for two parameter vectors `u, v : Parameters p`
     generating `L`-layer residual branch families `branch u, branch v : Fin L → F → F` where
     `branch u t` is `c t`-Lipschitz in its state input and `‖branch u t y - branch v t y‖ ≤ L_param t * ‖u - v‖`,
     the `L`-layer outputs differ in norm by at most `transformerStackLip L c L_param * ‖u - v‖`.
   - Proves `transformer_depth_scaled_lip_bound`: under `1 / L` residual depth scaling
     (`c t ≤ c_0 / L` and `L_param t ≤ L_0 / L`), the cumulative parameter-to-output Lipschitz
     constant `transformerStackLip L c L_param` is bounded by `L_0 * (1 + c_0 / L) ^ L` and
     uniformly in depth by `L_0 * Real.exp c_0`.

2. **Transformer Hypothesis Class Covering Numbers & Uniform Generalization Bounds**:
   - Defines `transformerParamLoss` and `transformerLossLipConst` for an `L`-layer parameterized
     Transformer `model : Parameters p → Z → F` and an `L_ℓ`-Lipschitz loss `ℓ : F → F → ℝ`.
   - Proves `transformerParamLoss_lipschitz`: `transformerParamLoss` is
     `(L_ℓ * transformerStackLip L c L_param)`-Lipschitz in the parameter vector `u : Parameters p`.
   - Proves `transformerBall_loss_cover`: using `BallCover.euclideanBall_net`, constructs an
     explicit finite cover `models : Finset (Parameters p)` of `trajectoryBall p rad` of size at
     most `(coverAxis rad δ p + 1) ^ p` with uniform loss error `(L_ℓ * transformerStackLip L c L_param) * δ`.
   - Proves `transformer_uniform_deviation_tail` and `transformer_approx_erm_excess_risk_tail`:
     exact high-probability uniform generalization and oracle excess population risk tail bounds
     `2 * (N : ℝ) * exp(-2 * n * ε^2)` for `L`-layer Transformers.

3. **End-to-End In-Context Learning (ICL) Excess Population Risk Decomposition**:
   - Defines `iclTotalRiskBound R_oracle n D_V γ₀ η Δ_g T ε_gen`.
   - Proves `icl_end_to_end_excess_risk_bound`: combines `InContextRetrieval.icl_population_risk_bound`
     with an empirical-to-population generalization certificate `ε_gen`.
   - Proves `icl_total_risk_monotone_in_steps`: increasing in-context alignment steps `T₁ ≤ T₂`
     monotonically reduces the end-to-end risk bound when `1 ≤ n`, `0 ≤ η`, `0 ≤ Δ_g`, `0 ≤ D_V`.
   - Proves `icl_end_to_end_target_accuracy`: explicit step threshold `T` guaranteeing that the
     in-context retrieval error term is at most `2 * ε_ret ^ 2` and the total population risk is at
     most `2 * R_oracle + 2 * ε_ret ^ 2 + ε_gen`.

4. **Mechanistic Interpretability & Safety Alignment Control via Circuit / Subspace Intervention**:
   - Proves `safety_intervention_tradeoff_bound`: if a circuit intervention replaces `branch₁` by
     `branch₂` with layerwise edit budget `‖branch₁ z t y - branch₂ z t y‖ ≤ ε_edit t` so that an
     unsafe linear probe `P_unsafe : F →L[ℝ] F` satisfies `‖P_unsafe (residualTrajectory L (branch₂ z) (xOf z))‖ ≤ τ_safe`,
     then the benign task population MSE is at most
     `2 * P.risk (clean_model) + 2 * (residualPerturbationEnvelope L c ε_edit) ^ 2` while the unsafe
     projection is certified at most `τ_safe`.
   - Proves `orthogonal_circuit_ablation_zero_unsafe_and_risk_bound`: if head ablation `keep`
     retains only safe heads (`∀ h ∈ keep, P_unsafe (headContribution ... h) = 0`), then
     `P_unsafe (ablatedMultiHeadOutput ... keep) = 0` (exact safety eradication) while benign
     population MSE is bounded by `2 * P.risk (full) + 2 * (∑ h ∈ keepᶜ, ‖proj h‖ * B_V h) ^ 2`.
-/

noncomputable section

open TorchLean MeasureTheory StatLean.StatisticalLearning
open scoped BigOperators

namespace OpenTheory.TransformerGeneralization

open OpenTheory.Attention
open OpenTheory.AttentionAblation
open OpenTheory.AttentionLogitStability
open OpenTheory.InContextRetrieval
open OpenTheory.DeepResidualScaling
open OpenTheory.TrajectoryGeneralization
open OpenTheory.BallCover
open OpenTheory.Covering
open OpenTheory.PredictionGap

variable {E F : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
  [NormedAddCommGroup F] [NormedSpace ℝ F]

/-! ## 1. Multi-Layer Transformer Parameter-to-Output Stability & Depth Propagation -/

/-- Cumulative perturbation envelope across an `L`-layer Transformer residual stack where
layer `t : Fin L` has branch Lipschitz constant `c t` and parameter sensitivity `L_param t`
under parameter displacement `δ_param`. -/
def layerParameterPerturbationEnvelope (L : ℕ) (c : Fin L → ℝ) (L_param : Fin L → ℝ)
    (δ_param : ℝ) : ℝ :=
  residualPerturbationEnvelope L c (fun t => L_param t * δ_param)

/-- Cumulative parameter-to-output Lipschitz constant of an `L`-layer Transformer residual stack
with per-layer branch Lipschitz constants `c : Fin L → ℝ` and parameter sensitivities
`L_param : Fin L → ℝ`. -/
def transformerStackLip (L : ℕ) (c : Fin L → ℝ) (L_param : Fin L → ℝ) : ℝ :=
  residualPerturbationEnvelope L c L_param

/-- Linear homogeneity of `residualPerturbationEnvelope` in the additive perturbation schedule:
scaling every layer's perturbation budget by `d` scales the cumulative envelope by `d`. -/
theorem transformerPerturbationEnvelope_smul :
    ∀ (L : ℕ) (c L_param : Fin L → ℝ) (d : ℝ),
      residualPerturbationEnvelope L c (fun t => L_param t * d) =
        residualPerturbationEnvelope L c L_param * d
  | 0, _, _, _ => by
      simp [residualPerturbationEnvelope]
  | L + 1, c, L_param, d => by
      simp only [residualPerturbationEnvelope]
      have ih := transformerPerturbationEnvelope_smul L
        (fun t => c t.castSucc) (fun t => L_param t.castSucc) d
      rw [ih]
      ring

/-- `layerParameterPerturbationEnvelope` factors as `transformerStackLip L c L_param * δ_param`. -/
theorem layerParameterPerturbationEnvelope_eq_mul (L : ℕ) (c L_param : Fin L → ℝ) (δ_param : ℝ) :
    layerParameterPerturbationEnvelope L c L_param δ_param =
      transformerStackLip L c L_param * δ_param :=
  transformerPerturbationEnvelope_smul L c L_param δ_param

/-- Nonnegativity of `transformerStackLip` when branch Lipschitz constants and parameter
sensitivities are nonnegative. -/
theorem transformerStackLip_nonneg (L : ℕ) (c L_param : Fin L → ℝ)
    (hc : ∀ t, 0 ≤ c t) (hL_param : ∀ t, 0 ≤ L_param t) :
    0 ≤ transformerStackLip L c L_param :=
  residualPerturbationEnvelope_nonneg L c L_param hc hL_param

omit [NormedSpace ℝ F] in
/-- **Multi-Layer Transformer Parameter-to-Output Perturbation Bound
(`transformer_stack_param_perturbation_le`)**:
For two parameter vectors `u, v : Parameters p` generating `L`-layer residual branch families
`branch u, branch v : Fin L → F → F` where `branch u t` is `c t`-Lipschitz in its state input and
`‖branch u t y - branch v t y‖ ≤ L_param t * ‖u - v‖`, the `L`-layer residual outputs starting
from `x : F` differ in norm by at most `transformerStackLip L c L_param * ‖u - v‖`. -/
theorem transformer_stack_param_perturbation_le {p : ℕ} (L : ℕ)
    (branch : Parameters p → Fin L → F → F)
    (c : Fin L → NNReal) (L_param : Fin L → ℝ)
    (u v : Parameters p)
    (hLip : ∀ t, LipschitzWith (c t) (branch u t))
    (hParam : ∀ t y, ‖branch u t y - branch v t y‖ ≤ L_param t * ‖u - v‖)
    (x : F) :
    ‖residualTrajectory L (branch u) x - residualTrajectory L (branch v) x‖ ≤
      transformerStackLip L (fun t => (c t : ℝ)) L_param * ‖u - v‖ := by
  have hpert := multilayer_residual_perturbation L (branch u) (branch v) c
    (fun t => L_param t * ‖u - v‖) hLip hParam x
  rwa [transformerPerturbationEnvelope_smul] at hpert

/-- Monotonicity of `residualPerturbationEnvelope` with respect to per-layer Lipschitz constants
and perturbation budgets. -/
theorem residualPerturbationEnvelope_mono :
    ∀ (L : ℕ) (c₁ c₂ ε₁ ε₂ : Fin L → ℝ),
      (∀ t, 0 ≤ c₁ t) →
      (∀ t, 0 ≤ ε₁ t) →
      (∀ t, c₁ t ≤ c₂ t) →
      (∀ t, ε₁ t ≤ ε₂ t) →
      residualPerturbationEnvelope L c₁ ε₁ ≤ residualPerturbationEnvelope L c₂ ε₂
  | 0, _, _, _, _, _, _, _, _ => le_rfl
  | L + 1, c₁, c₂, ε₁, ε₂, hc₁, hε₁, hc, hε => by
      simp only [residualPerturbationEnvelope]
      have ih := residualPerturbationEnvelope_mono L
        (fun t => c₁ t.castSucc) (fun t => c₂ t.castSucc)
        (fun t => ε₁ t.castSucc) (fun t => ε₂ t.castSucc)
        (fun t => hc₁ t.castSucc) (fun t => hε₁ t.castSucc)
        (fun t => hc t.castSucc) (fun t => hε t.castSucc)
      have henv_nonneg := residualPerturbationEnvelope_nonneg L
        (fun t => c₁ t.castSucc) (fun t => ε₁ t.castSucc)
        (fun t => hc₁ t.castSucc) (fun t => hε₁ t.castSucc)
      have h1 : 0 ≤ 1 + c₂ (Fin.last L) := by linarith [hc₁ (Fin.last L), hc (Fin.last L)]
      have hmul :
          (1 + c₁ (Fin.last L)) *
            residualPerturbationEnvelope L (fun t => c₁ t.castSucc) (fun t => ε₁ t.castSucc) ≤
          (1 + c₂ (Fin.last L)) *
            residualPerturbationEnvelope L (fun t => c₂ t.castSucc) (fun t => ε₂ t.castSucc) := by
        calc
          (1 + c₁ (Fin.last L)) *
              residualPerturbationEnvelope L (fun t => c₁ t.castSucc) (fun t => ε₁ t.castSucc) ≤
              (1 + c₂ (Fin.last L)) *
                residualPerturbationEnvelope L (fun t => c₁ t.castSucc) (fun t => ε₁ t.castSucc) :=
            mul_le_mul_of_nonneg_right (by linarith [hc (Fin.last L)]) henv_nonneg
          _ ≤ (1 + c₂ (Fin.last L)) *
                residualPerturbationEnvelope L (fun t => c₂ t.castSucc) (fun t => ε₂ t.castSucc) :=
            mul_le_mul_of_nonneg_left ih h1
      exact add_le_add hmul (hε (Fin.last L))

/-- Upper bound for `residualPerturbationEnvelope` with constant per-layer branch Lipschitz
constant `c_step ≥ 0` and constant per-layer perturbation `ε_step ≥ 0`:
by induction on `k`, `residualPerturbationEnvelope k (fun _ => c_step) (fun _ => ε_step) ≤ k * ε_step * (1 + c_step) ^ k`. -/
theorem residualPerturbationEnvelope_const_le :
    ∀ (k : ℕ) (c_step ε_step : ℝ),
      0 ≤ c_step → 0 ≤ ε_step →
      residualPerturbationEnvelope k (fun _ => c_step) (fun _ => ε_step) ≤
        (k : ℝ) * ε_step * (1 + c_step) ^ k
  | 0, _, _, _, _ => by
      simp [residualPerturbationEnvelope]
  | k + 1, c_step, ε_step, hc, hε => by
      simp only [residualPerturbationEnvelope]
      have ih := residualPerturbationEnvelope_const_le k c_step ε_step hc hε
      have h1_nonneg : 0 ≤ 1 + c_step := by linarith
      have hpow_ge_one : 1 ≤ (1 + c_step) ^ (k + 1) :=
        one_le_pow₀ (by linarith : 1 ≤ 1 + c_step)
      have hε_pow : ε_step ≤ ε_step * (1 + c_step) ^ (k + 1) := by
        calc
          ε_step = ε_step * 1 := (mul_one ε_step).symm
          _ ≤ ε_step * (1 + c_step) ^ (k + 1) := mul_le_mul_of_nonneg_left hpow_ge_one hε
      calc
        (1 + c_step) * residualPerturbationEnvelope k (fun _ => c_step) (fun _ => ε_step) + ε_step ≤
            (1 + c_step) * ((k : ℝ) * ε_step * (1 + c_step) ^ k) +
              ε_step * (1 + c_step) ^ (k + 1) :=
          add_le_add (mul_le_mul_of_nonneg_left ih h1_nonneg) hε_pow
        _ = ((k + 1 : ℕ) : ℝ) * ε_step * (1 + c_step) ^ (k + 1) := by
          push_cast
          rw [pow_succ]
          ring

/-- **Depth-Independent `1 / L` Residual Scaling Bound for Transformer Stack Lipschitz Constant
(`transformer_depth_scaled_lip_bound`)**:
When an `L`-layer Transformer (`0 < L`) uses `1 / L` residual scaling so that each layer `t : Fin L`
satisfies `0 ≤ c t ≤ c_0 / L` and `0 ≤ L_param t ≤ L_0 / L` (with `0 ≤ c_0` and `0 ≤ L_0`), the
cumulative parameter-to-output Lipschitz constant `transformerStackLip L c L_param` satisfies:
1. `transformerStackLip L c L_param ≤ L_0 * (1 + c_0 / L) ^ L`, and
2. `transformerStackLip L c L_param ≤ L_0 * Real.exp c_0` (uniformly bounded in depth `L`). -/
theorem transformer_depth_scaled_lip_bound {L : ℕ} (hL : 0 < L)
    (c L_param : Fin L → ℝ) {c_0 L_0 : ℝ}
    (hc0 : 0 ≤ c_0) (hL0 : 0 ≤ L_0)
    (hc_nonneg : ∀ t, 0 ≤ c t)
    (hL_nonneg : ∀ t, 0 ≤ L_param t)
    (hc_bound : ∀ t, c t ≤ c_0 / (L : ℝ))
    (hL_bound : ∀ t, L_param t ≤ L_0 / (L : ℝ)) :
    transformerStackLip L c L_param ≤ L_0 * (1 + c_0 / (L : ℝ)) ^ L ∧
    transformerStackLip L c L_param ≤ L_0 * Real.exp c_0 := by
  have hL_pos : (0 : ℝ) < (L : ℝ) := Nat.cast_pos.mpr hL
  have hc_step_nonneg : 0 ≤ c_0 / (L : ℝ) := div_nonneg hc0 hL_pos.le
  have hL_step_nonneg : 0 ≤ L_0 / (L : ℝ) := div_nonneg hL0 hL_pos.le
  have hmono := residualPerturbationEnvelope_mono L
    c (fun _ => c_0 / (L : ℝ)) L_param (fun _ => L_0 / (L : ℝ))
    hc_nonneg hL_nonneg hc_bound hL_bound
  have hconst := residualPerturbationEnvelope_const_le L
    (c_0 / (L : ℝ)) (L_0 / (L : ℝ)) hc_step_nonneg hL_step_nonneg
  have hcancel : (L : ℝ) * (L_0 / (L : ℝ)) = L_0 :=
    mul_div_cancel₀ L_0 (ne_of_gt hL_pos)
  have hpow_bound : transformerStackLip L c L_param ≤ L_0 * (1 + c_0 / (L : ℝ)) ^ L := by
    unfold transformerStackLip
    calc
      residualPerturbationEnvelope L c L_param ≤
          residualPerturbationEnvelope L (fun _ => c_0 / (L : ℝ)) (fun _ => L_0 / (L : ℝ)) := hmono
      _ ≤ (L : ℝ) * (L_0 / (L : ℝ)) * (1 + c_0 / (L : ℝ)) ^ L := hconst
      _ = L_0 * (1 + c_0 / (L : ℝ)) ^ L := by rw [hcancel]
  have hexp_bound : transformerStackLip L c L_param ≤ L_0 * Real.exp c_0 :=
    hpow_bound.trans (mul_le_mul_of_nonneg_left (one_add_div_pow_le_exp hL hc0) hL0)
  exact ⟨hpow_bound, hexp_bound⟩

/-! ## 2. Transformer Hypothesis Class Covering Numbers & Uniform Generalization Bounds -/

/-- Parameter-indexed observation loss for a parameterized model `model : Parameters p → Z → F`
(such as an `L`-layer Transformer `fun u z => residualTrajectory L (branch u) (xOf z)`),
label map `yOf : Z → F`, and loss function `ℓ : F → F → ℝ`. -/
def transformerParamLoss {Z : Type*} {p : ℕ}
    (model : Parameters p → Z → F) (yOf : Z → F) (ℓ : F → F → ℝ)
    (u : Parameters p) (z : Z) : ℝ :=
  ℓ (model u z) (yOf z)

/-- Parameter-to-loss Lipschitz constant `L_ℓ * transformerStackLip L c L_param` for an
`L`-layer Transformer. -/
def transformerLossLipConst (L : ℕ) (L_ℓ : ℝ) (c L_param : Fin L → ℝ) : ℝ :=
  L_ℓ * transformerStackLip L c L_param

/-- Nonnegativity of `transformerLossLipConst` when `L_ℓ`, `c`, and `L_param` are nonnegative. -/
theorem transformerLossLipConst_nonneg (L : ℕ) {L_ℓ : ℝ} (c L_param : Fin L → ℝ)
    (hLℓ : 0 ≤ L_ℓ) (hc : ∀ t, 0 ≤ c t) (hL_param : ∀ t, 0 ≤ L_param t) :
    0 ≤ transformerLossLipConst L L_ℓ c L_param :=
  mul_nonneg hLℓ (transformerStackLip_nonneg L c L_param hc hL_param)

omit [NormedSpace ℝ F] in
/-- **Parameter-to-Loss Lipschitz Bound for `L`-Layer Transformers (`transformerParamLoss_lipschitz`)**:
If `model : Parameters p → Z → F` is an `L`-layer Transformer satisfying
`‖model u z - model v z‖ ≤ transformerStackLip L c L_param * ‖u - v‖` and `ℓ : F → F → ℝ` is
`L_ℓ`-Lipschitz in its prediction argument, then `transformerParamLoss model yOf ℓ` is
`(L_ℓ * transformerStackLip L c L_param)`-Lipschitz in the parameter vector `u : Parameters p`. -/
theorem transformerParamLoss_lipschitz {Z : Type*} {p : ℕ} (L : ℕ)
    (model : Parameters p → Z → F) (yOf : Z → F) (ℓ : F → F → ℝ)
    {L_ℓ : ℝ} (c L_param : Fin L → ℝ)
    (hLℓ : 0 ≤ L_ℓ)
    (hℓ : ∀ p₁ p₂ y : F, |ℓ p₁ y - ℓ p₂ y| ≤ L_ℓ * ‖p₁ - p₂‖)
    (hmodel : ∀ (u v : Parameters p) (z : Z),
      ‖model u z - model v z‖ ≤ transformerStackLip L c L_param * ‖u - v‖)
    (u v : Parameters p) (z : Z) :
    |transformerParamLoss model yOf ℓ u z - transformerParamLoss model yOf ℓ v z| ≤
      (L_ℓ * transformerStackLip L c L_param) * ‖u - v‖ := by
  unfold transformerParamLoss
  calc
    |ℓ (model u z) (yOf z) - ℓ (model v z) (yOf z)| ≤
        L_ℓ * ‖model u z - model v z‖ := hℓ (model u z) (model v z) (yOf z)
    _ ≤ L_ℓ * (transformerStackLip L c L_param * ‖u - v‖) :=
        mul_le_mul_of_nonneg_left (hmodel u v z) hLℓ
    _ = (L_ℓ * transformerStackLip L c L_param) * ‖u - v‖ := by ring

omit [NormedSpace ℝ F] in
/-- Direct specialization of `transformerParamLoss_lipschitz` to an explicit `L`-layer residual
stack `fun u z => residualTrajectory L (branch u) (xOf z)`. -/
theorem transformerStack_paramLoss_lipschitz {Z : Type*} {p : ℕ} (L : ℕ)
    (branch : Parameters p → Fin L → F → F) (xOf : Z → F) (yOf : Z → F) (ℓ : F → F → ℝ)
    {L_ℓ : ℝ} (c : Fin L → NNReal) (L_param : Fin L → ℝ)
    (hLℓ : 0 ≤ L_ℓ)
    (hℓ : ∀ p₁ p₂ y : F, |ℓ p₁ y - ℓ p₂ y| ≤ L_ℓ * ‖p₁ - p₂‖)
    (hLip : ∀ (u : Parameters p) (t : Fin L), LipschitzWith (c t) (branch u t))
    (hParam : ∀ (u v : Parameters p) (t : Fin L) (y : F),
      ‖branch u t y - branch v t y‖ ≤ L_param t * ‖u - v‖)
    (u v : Parameters p) (z : Z) :
    |transformerParamLoss (fun w z' => residualTrajectory L (branch w) (xOf z')) yOf ℓ u z -
      transformerParamLoss (fun w z' => residualTrajectory L (branch w) (xOf z')) yOf ℓ v z| ≤
      (L_ℓ * transformerStackLip L (fun t => (c t : ℝ)) L_param) * ‖u - v‖ :=
  transformerParamLoss_lipschitz L
    (fun w z' => residualTrajectory L (branch w) (xOf z')) yOf ℓ
    (fun t => (c t : ℝ)) L_param hLℓ hℓ
    (fun u' v' z' =>
      transformer_stack_param_perturbation_le L branch c L_param u' v'
        (hLip u') (hParam u' v') (xOf z'))
    u v z

omit [NormedSpace ℝ F] in
/-- **Explicit Finite Loss Cover for the Transformer Parameter Ball (`transformerBall_loss_cover`)**:
Using `BallCover.euclideanBall_net`, for any radius `rad ≥ 0` and grid resolution `δ > 0`, there is
an explicit finite cover `models : Finset (Parameters p)` of `trajectoryBall p rad` of cardinality
at most `(coverAxis rad δ p + 1) ^ p` with uniform loss error
`(L_ℓ * transformerStackLip L c L_param) * δ`. -/
theorem transformerBall_loss_cover {Z : Type*} {p : ℕ} (L : ℕ)
    (model : Parameters p → Z → F) (yOf : Z → F) (ℓ : F → F → ℝ)
    {L_ℓ rad δ : ℝ} (c L_param : Fin L → ℝ)
    (hLℓ : 0 ≤ L_ℓ) (hc : ∀ t, 0 ≤ c t) (hL_param : ∀ t, 0 ≤ L_param t)
    (hrad : 0 ≤ rad) (hδ : 0 < δ)
    (hℓ : ∀ p₁ p₂ y : F, |ℓ p₁ y - ℓ p₂ y| ≤ L_ℓ * ‖p₁ - p₂‖)
    (hmodel : ∀ (u v : Parameters p) (z : Z),
      ‖model u z - model v z‖ ≤ transformerStackLip L c L_param * ‖u - v‖) :
    ∃ models : Finset (Parameters p),
      (∀ g ∈ models, g ∈ trajectoryBall p rad) ∧
      models.card ≤ (coverAxis rad δ p + 1) ^ p ∧
      ∀ u ∈ trajectoryBall p rad, ∃ g ∈ models, ∀ z,
        |transformerParamLoss model yOf ℓ u z - transformerParamLoss model yOf ℓ g z| ≤
          (L_ℓ * transformerStackLip L c L_param) * δ := by
  obtain ⟨models, hmodels, hcard, hnet⟩ := euclideanBall_net (d := p) hrad hδ
  refine ⟨models, hmodels, hcard, ?_⟩
  intro u hu
  obtain ⟨g, hg, hdist⟩ := hnet u hu
  refine ⟨g, hg, fun z => ?_⟩
  have hlip := transformerParamLoss_lipschitz L model yOf ℓ c L_param hLℓ hℓ hmodel u g z
  exact hlip.trans
    (mul_le_mul_of_nonneg_left hdist (transformerLossLipConst_nonneg L c L_param hLℓ hc hL_param))

omit [NormedSpace ℝ F] in
/-- **Uniform Generalization Tail Bound for `L`-Layer Transformers
(`transformer_uniform_deviation_tail`)**:
Over an IID sample `s ~ sampleLaw P.law n`, every parameter vector `u ∈ trajectoryBall p rad`
simultaneously has empirical risk within `ε + 2 * ((L_ℓ * transformerStackLip L c L_param) * δ)`
of its population risk `P.risk (transformerParamLoss model yOf ℓ u)` outside an event of
probability at most `2 * (N : ℝ) * Real.exp (-2 * n * ε ^ 2)` where
`N = (coverAxis rad δ p + 1) ^ p`. -/
theorem transformer_uniform_deviation_tail {Z : Type*} [MeasurableSpace Z]
    (P : Population Z) {p n : ℕ} (L : ℕ)
    (model : Parameters p → Z → F) (yOf : Z → F) (ℓ : F → F → ℝ)
    {L_ℓ rad δ ε : ℝ} (c L_param : Fin L → ℝ)
    (hLℓ : 0 ≤ L_ℓ) (hc : ∀ t, 0 ≤ c t) (hL_param : ∀ t, 0 ≤ L_param t)
    (hrad : 0 ≤ rad) (hδ : 0 < δ) (hn : 1 ≤ n) (hε : 0 < ε)
    (hℓ : ∀ p₁ p₂ y : F, |ℓ p₁ y - ℓ p₂ y| ≤ L_ℓ * ‖p₁ - p₂‖)
    (hmodel : ∀ (u v : Parameters p) (z : Z),
      ‖model u z - model v z‖ ≤ transformerStackLip L c L_param * ‖u - v‖)
    (hrange : ∀ u ∈ trajectoryBall p rad, ∀ z,
      transformerParamLoss model yOf ℓ u z ∈ Set.Icc (0 : ℝ) 1)
    (hmeas : ∀ u ∈ trajectoryBall p rad, Measurable (transformerParamLoss model yOf ℓ u)) :
    let ρ := (L_ℓ * transformerStackLip L c L_param) * δ
    let N := (coverAxis rad δ p + 1) ^ p
    sampleLaw P.law n {s | ∃ u ∈ trajectoryBall p rad,
      ε + 2 * ρ < |empRisk (transformerParamLoss model yOf ℓ) s u -
        P.risk (transformerParamLoss model yOf ℓ u)|} ≤
      ENNReal.ofReal (2 * (N : ℝ) * Real.exp (-2 * n * ε ^ 2)) := by
  intro ρ N
  obtain ⟨models, hmodels, hcard, hcover⟩ :=
    transformerBall_loss_cover L model yOf ℓ c L_param hLℓ hc hL_param hrad hδ hℓ hmodel
  have htail := uniform_deviation_tail P (trajectoryBall p rad) models
    (transformerParamLoss model yOf ℓ) hmodels hrange hmeas ρ ε hcover hn hε
  apply htail.trans
  apply ENNReal.ofReal_le_ofReal
  have hcard_real : (models.card : ℝ) ≤ (N : ℝ) := by exact_mod_cast hcard
  have hexp : 0 ≤ Real.exp (-2 * n * ε ^ 2) := Real.exp_nonneg _
  nlinarith

omit [NormedSpace ℝ F] in
/-- **Excess Population Risk Tail Bound for Approximate ERM Transformer Learners
(`transformer_approx_erm_excess_risk_tail`)**:
Any training procedure `learner : (Fin n → Z) → Parameters p` returning parameters in
`trajectoryBall p rad` with `η_opt`-approximate empirical risk minimization over `trajectoryBall p rad`
satisfies an oracle excess population risk bound with tail probability at most
`2 * (N : ℝ) * Real.exp (-2 * n * ε ^ 2)` where `N = (coverAxis rad δ p + 1) ^ p`. -/
theorem transformer_approx_erm_excess_risk_tail {Z : Type*} [MeasurableSpace Z]
    (P : Population Z) {p n : ℕ} (L : ℕ)
    (model : Parameters p → Z → F) (yOf : Z → F) (ℓ : F → F → ℝ)
    {L_ℓ rad δ ε η_opt : ℝ} (c L_param : Fin L → ℝ)
    (hLℓ : 0 ≤ L_ℓ) (hc : ∀ t, 0 ≤ c t) (hL_param : ∀ t, 0 ≤ L_param t)
    (hrad : 0 ≤ rad) (hδ : 0 < δ) (hn : 1 ≤ n) (hε : 0 < ε)
    (hℓ : ∀ p₁ p₂ y : F, |ℓ p₁ y - ℓ p₂ y| ≤ L_ℓ * ‖p₁ - p₂‖)
    (hmodel : ∀ (u v : Parameters p) (z : Z),
      ‖model u z - model v z‖ ≤ transformerStackLip L c L_param * ‖u - v‖)
    (hrange : ∀ u ∈ trajectoryBall p rad, ∀ z,
      transformerParamLoss model yOf ℓ u z ∈ Set.Icc (0 : ℝ) 1)
    (hmeas : ∀ u ∈ trajectoryBall p rad, Measurable (transformerParamLoss model yOf ℓ u))
    (learner : (Fin n → Z) → Parameters p)
    (htrain : ∀ s, StatLean.StatisticalLearning.IsApproxERM
      (trajectoryBall p rad) (transformerParamLoss model yOf ℓ) s η_opt (learner s)) :
    let ρ := (L_ℓ * transformerStackLip L c L_param) * δ
    let N := (coverAxis rad δ p + 1) ^ p
    sampleLaw P.law n {s | ∃ u ∈ trajectoryBall p rad,
      P.risk (transformerParamLoss model yOf ℓ u) + 2 * ε + 4 * ρ + η_opt <
        P.risk (transformerParamLoss model yOf ℓ (learner s))} ≤
      ENNReal.ofReal (2 * (N : ℝ) * Real.exp (-2 * n * ε ^ 2)) := by
  intro ρ N
  obtain ⟨models, hmodels, hcard, hcover⟩ :=
    transformerBall_loss_cover L model yOf ℓ c L_param hLℓ hc hL_param hrad hδ hℓ hmodel
  have htail := approx_learner_tail P (trajectoryBall p rad) models
    (transformerParamLoss model yOf ℓ) hmodels hrange hmeas ρ ε η_opt hcover learner htrain hn hε
  apply htail.trans
  apply ENNReal.ofReal_le_ofReal
  have hcard_real : (models.card : ℝ) ≤ (N : ℝ) := by exact_mod_cast hcard
  have hexp : 0 ≤ Real.exp (-2 * n * ε ^ 2) := Real.exp_nonneg _
  nlinarith

/-! ## 3. End-to-End In-Context Learning (ICL) Excess Population Risk Decomposition -/

/-- End-to-end In-Context Learning (ICL) population risk bound combining:
- `2 * R_oracle`: twice the ideal context-oracle population risk,
- `2 * (((n : ℝ) - 1) * D_V * Real.exp (-γ₀ - (T : ℝ) * η * Δ_g)) ^ 2`: exponentially decaying
  in-context retrieval error after `T` score/query alignment steps, and
- `ε_gen`: empirical-to-population generalization / estimation penalty. -/
def iclTotalRiskBound (R_oracle : ℝ) (n : ℕ) (D_V γ₀ η Δ_g : ℝ) (T : ℕ) (ε_gen : ℝ) : ℝ :=
  2 * R_oracle + 2 * (((n : ℝ) - 1) * D_V * Real.exp (-γ₀ - (T : ℝ) * η * Δ_g)) ^ 2 + ε_gen

/-- **End-to-End In-Context Learning Excess Population Risk Bound
(`icl_end_to_end_excess_risk_bound`)**:
If a deployedpredictor `loss_deployed : Z → ℝ` has population risk within `ε_gen` of the `T`-step
in-context attention retriever (`P.risk loss_deployed ≤ P.risk loss_icl + ε_gen`), and the oracle
retriever has population MSE at most `R_oracle`, then under the standard ICL margin-growth
conditions of `InContextRetrieval.icl_population_risk_bound`, `P.risk loss_deployed` is bounded by
`iclTotalRiskBound R_oracle n D_V γ₀ η Δ_g T ε_gen`. -/
theorem icl_end_to_end_excess_risk_bound {Z : Type*} [MeasurableSpace Z] {n : ℕ}
    (P : Population Z)
    (R_oracle η γ₀ Δ_g D_V ε_gen : ℝ)
    (scores₀ : Z → Fin n → ℝ)
    (g_seq : Z → ℕ → Fin n → ℝ)
    (mask : Z → Tensor Bool [n])
    (values : Z → Fin n → E)
    (i_star : Z → Fin n)
    (label : Z → E)
    (loss_deployed : Z → ℝ)
    (T : ℕ)
    (hη : 0 ≤ η)
    (hD_nonneg : 0 ≤ D_V)
    (hmeas_icl : AEStronglyMeasurable
      (fun z =>
        ‖Attention.rowOutput (scoreTensor (scoreTrajectory η (scores₀ z) (g_seq z) T))
          (mask z) (values z) - label z‖ ^ 2)
      P.law)
    (hint_oracle : P.IntegrableLoss
      (fun z => ‖values z (i_star z) - label z‖ ^ 2))
    (hR_oracle : P.risk (fun z => ‖values z (i_star z) - label z‖ ^ 2) ≤ R_oracle)
    (hae : ∀ᵐ z ∂P.law,
      (mask z).getScalar (i_star z) = true ∧
      (∀ j : Fin n, (mask z).getScalar j = true → j ≠ i_star z →
        scores₀ z j ≤ scores₀ z (i_star z) - γ₀) ∧
      (∀ (t : ℕ) (j : Fin n), (mask z).getScalar j = true → j ≠ i_star z →
        g_seq z t j ≤ g_seq z t (i_star z) - Δ_g) ∧
      (∀ j : Fin n, (mask z).getScalar j = true →
        ‖values z j - values z (i_star z)‖ ≤ D_V))
    (hgen : P.risk loss_deployed ≤
      P.risk (fun z =>
        ‖Attention.rowOutput (scoreTensor (scoreTrajectory η (scores₀ z) (g_seq z) T))
          (mask z) (values z) - label z‖ ^ 2) + ε_gen) :
    P.IntegrableLoss
      (fun z =>
        ‖Attention.rowOutput (scoreTensor (scoreTrajectory η (scores₀ z) (g_seq z) T))
          (mask z) (values z) - label z‖ ^ 2) ∧
    P.risk loss_deployed ≤ iclTotalRiskBound R_oracle n D_V γ₀ η Δ_g T ε_gen := by
  obtain ⟨hint_icl, hrisk_icl⟩ :=
    icl_population_risk_bound P η γ₀ Δ_g D_V scores₀ g_seq mask values i_star label T
      hη hD_nonneg hmeas_icl hint_oracle hae
  refine ⟨hint_icl, ?_⟩
  unfold iclTotalRiskBound
  linarith

/-- **Monotonicity of `iclTotalRiskBound` in In-Context Alignment Steps
(`icl_total_risk_monotone_in_steps`)**:
When `1 ≤ n`, `0 ≤ D_V`, `0 ≤ η`, and `0 ≤ Δ_g`, increasing the number of in-context alignment
steps `T₁ ≤ T₂` monotonically decreases `iclTotalRiskBound`. -/
theorem icl_total_risk_monotone_in_steps
    (R_oracle : ℝ) {n : ℕ} {D_V γ₀ η Δ_g : ℝ} {T₁ T₂ : ℕ} (ε_gen : ℝ)
    (hn : 1 ≤ n) (hD_V : 0 ≤ D_V) (hη : 0 ≤ η) (hΔ_g : 0 ≤ Δ_g)
    (hT : T₁ ≤ T₂) :
    iclTotalRiskBound R_oracle n D_V γ₀ η Δ_g T₂ ε_gen ≤
      iclTotalRiskBound R_oracle n D_V γ₀ η Δ_g T₁ ε_gen := by
  unfold iclTotalRiskBound
  have hn_nonneg : 0 ≤ (n : ℝ) - 1 := by
    have : (1 : ℝ) ≤ (n : ℝ) := by exact_mod_cast hn
    linarith
  have hC_nonneg : 0 ≤ ((n : ℝ) - 1) * D_V := mul_nonneg hn_nonneg hD_V
  have hT_real : (T₁ : ℝ) ≤ (T₂ : ℝ) := by exact_mod_cast hT
  have hstep_le : (T₁ : ℝ) * (η * Δ_g) ≤ (T₂ : ℝ) * (η * Δ_g) :=
    mul_le_mul_of_nonneg_right hT_real (mul_nonneg hη hΔ_g)
  have hexp_arg : -γ₀ - (T₂ : ℝ) * η * Δ_g ≤ -γ₀ - (T₁ : ℝ) * η * Δ_g := by
    linarith
  have hexp_le :
      Real.exp (-γ₀ - (T₂ : ℝ) * η * Δ_g) ≤ Real.exp (-γ₀ - (T₁ : ℝ) * η * Δ_g) :=
    Real.exp_le_exp.mpr hexp_arg
  have hterm_le :
      ((n : ℝ) - 1) * D_V * Real.exp (-γ₀ - (T₂ : ℝ) * η * Δ_g) ≤
        ((n : ℝ) - 1) * D_V * Real.exp (-γ₀ - (T₁ : ℝ) * η * Δ_g) :=
    mul_le_mul_of_nonneg_left hexp_le hC_nonneg
  have hterm2_nonneg : 0 ≤ ((n : ℝ) - 1) * D_V * Real.exp (-γ₀ - (T₂ : ℝ) * η * Δ_g) :=
    mul_nonneg hC_nonneg (Real.exp_nonneg _)
  have hsq_le :
      (((n : ℝ) - 1) * D_V * Real.exp (-γ₀ - (T₂ : ℝ) * η * Δ_g)) ^ 2 ≤
        (((n : ℝ) - 1) * D_V * Real.exp (-γ₀ - (T₁ : ℝ) * η * Δ_g)) ^ 2 := by
    nlinarith [mul_self_le_mul_self hterm2_nonneg hterm_le]
  linarith

/-- **Explicit Target-Accuracy Step Threshold for End-to-End ICL Risk
(`icl_end_to_end_target_accuracy`)**:
For any target retrieval error `ε_ret > 0` (with `0 < (n : ℝ) - 1` and `0 < D_V`), once the
in-context alignment horizon `T : ℕ` satisfies
`(T : ℝ) * η * Δ_g ≥ Real.log (((n : ℝ) - 1) * D_V / ε_ret) - γ₀`,
the retrieval penalty `2 * (((n : ℝ) - 1) * D_V * Real.exp (-γ₀ - (T : ℝ) * η * Δ_g)) ^ 2` is at
most `2 * ε_ret ^ 2`, and `iclTotalRiskBound R_oracle n D_V γ₀ η Δ_g T ε_gen` is at most
`2 * R_oracle + 2 * ε_ret ^ 2 + ε_gen`. -/
theorem icl_end_to_end_target_accuracy
    (R_oracle : ℝ) {n : ℕ} {D_V γ₀ η Δ_g ε_ret : ℝ} (T : ℕ) (ε_gen : ℝ)
    (hn : 0 < (n : ℝ) - 1) (hD_pos : 0 < D_V) (hε_ret : 0 < ε_ret)
    (hT : (T : ℝ) * η * Δ_g ≥ Real.log (((n : ℝ) - 1) * D_V / ε_ret) - γ₀) :
    2 * (((n : ℝ) - 1) * D_V * Real.exp (-γ₀ - (T : ℝ) * η * Δ_g)) ^ 2 ≤ 2 * ε_ret ^ 2 ∧
    iclTotalRiskBound R_oracle n D_V γ₀ η Δ_g T ε_gen ≤
      2 * R_oracle + 2 * ε_ret ^ 2 + ε_gen := by
  have hC_pos : 0 < ((n : ℝ) - 1) * D_V := mul_pos hn hD_pos
  have hratio_pos : 0 < ((n : ℝ) - 1) * D_V / ε_ret := div_pos hC_pos hε_ret
  have hlog_le : -γ₀ - (T : ℝ) * η * Δ_g ≤ -Real.log (((n : ℝ) - 1) * D_V / ε_ret) := by
    linarith
  have hexp_le :
      Real.exp (-γ₀ - (T : ℝ) * η * Δ_g) ≤
        Real.exp (-Real.log (((n : ℝ) - 1) * D_V / ε_ret)) :=
    Real.exp_le_exp.mpr hlog_le
  have hexp_inv :
      Real.exp (-Real.log (((n : ℝ) - 1) * D_V / ε_ret)) =
        ε_ret / (((n : ℝ) - 1) * D_V) := by
    rw [Real.exp_neg, Real.exp_log hratio_pos, inv_div]
  rw [hexp_inv] at hexp_le
  have hterm_le :
      ((n : ℝ) - 1) * D_V * Real.exp (-γ₀ - (T : ℝ) * η * Δ_g) ≤ ε_ret := by
    calc
      ((n : ℝ) - 1) * D_V * Real.exp (-γ₀ - (T : ℝ) * η * Δ_g) ≤
          ((n : ℝ) - 1) * D_V * (ε_ret / (((n : ℝ) - 1) * D_V)) :=
        mul_le_mul_of_nonneg_left hexp_le hC_pos.le
      _ = ε_ret := mul_div_cancel₀ ε_ret (ne_of_gt hC_pos)
  have hterm_nonneg : 0 ≤ ((n : ℝ) - 1) * D_V * Real.exp (-γ₀ - (T : ℝ) * η * Δ_g) :=
    mul_nonneg hC_pos.le (Real.exp_nonneg _)
  have hsq_le :
      (((n : ℝ) - 1) * D_V * Real.exp (-γ₀ - (T : ℝ) * η * Δ_g)) ^ 2 ≤ ε_ret ^ 2 := by
    nlinarith [mul_self_le_mul_self hterm_nonneg hterm_le]
  refine ⟨by linarith, ?_⟩
  unfold iclTotalRiskBound
  linarith

/-! ## 4. Mechanistic Interpretability & Safety Alignment Control via Circuit / Subspace Intervention -/

/-- **Safety-Alignment Circuit Intervention Tradeoff Bound (`safety_intervention_tradeoff_bound`)**:
Let `P_unsafe : F →L[ℝ] F` be a continuous linear probe/projection onto an unsafe behavior subspace.
Suppose a circuit/steering intervention across an `L`-layer Transformer replaces `branch₁ z` with
`branch₂ z` under layerwise edit budget `‖branch₁ z t y - branch₂ z t y‖ ≤ ε_edit t` (`ε_edit t ≥ 0`)
such that almost surely under `P.law`, the intervened representation satisfies the safety certificate
`‖P_unsafe (residualTrajectory L (branch₂ z) (xOf z))‖ ≤ τ_safe`.
Then:
1. The safety-intervened model has integrable squared-error loss on the benign task `label : Z → F`,
2. Its benign task population MSE is at most
   `2 * P.risk (clean_model) + 2 * (residualPerturbationEnvelope L c ε_edit) ^ 2`, and
3. Almost surely under `P.law`, the unsafe subspace projection of the intervened Transformer output
   is certified at most `τ_safe`. -/
theorem safety_intervention_tradeoff_bound {Z : Type*} [MeasurableSpace Z] (L : ℕ)
    (P : Population Z)
    (P_unsafe : F →L[ℝ] F)
    (branch₁ branch₂ : Z → Fin L → F → F)
    (xOf label : Z → F)
    (c : Fin L → NNReal)
    (ε_edit : Fin L → ℝ)
    (τ_safe : ℝ)
    (hε_nonneg : ∀ t, 0 ≤ ε_edit t)
    (hLip : ∀ (z : Z) (t : Fin L), LipschitzWith (c t) (branch₁ z t))
    (hEdit : ∀ (z : Z) (t : Fin L) (y : F),
      ‖branch₁ z t y - branch₂ z t y‖ ≤ ε_edit t)
    (hmeas_intervened : AEStronglyMeasurable
      (fun z => ‖residualTrajectory L (branch₂ z) (xOf z) - label z‖ ^ 2)
      P.law)
    (hint_clean : P.IntegrableLoss
      (fun z => ‖residualTrajectory L (branch₁ z) (xOf z) - label z‖ ^ 2))
    (hsafe : ∀ᵐ z ∂P.law,
      ‖P_unsafe (residualTrajectory L (branch₂ z) (xOf z))‖ ≤ τ_safe) :
    P.IntegrableLoss
      (fun z => ‖residualTrajectory L (branch₂ z) (xOf z) - label z‖ ^ 2) ∧
    P.risk (fun z => ‖residualTrajectory L (branch₂ z) (xOf z) - label z‖ ^ 2) ≤
      2 * P.risk (fun z => ‖residualTrajectory L (branch₁ z) (xOf z) - label z‖ ^ 2) +
        2 * (residualPerturbationEnvelope L (fun t => (c t : ℝ)) ε_edit) ^ 2 ∧
    (∀ᵐ z ∂P.law, ‖P_unsafe (residualTrajectory L (branch₂ z) (xOf z))‖ ≤ τ_safe) := by
  let δ := residualPerturbationEnvelope L (fun t => (c t : ℝ)) ε_edit
  have hδ_nonneg : 0 ≤ δ :=
    residualPerturbationEnvelope_nonneg L (fun t => (c t : ℝ)) ε_edit
      (fun t => (c t).coe_nonneg) hε_nonneg
  let lossClean : Z → ℝ :=
    fun z => ‖residualTrajectory L (branch₁ z) (xOf z) - label z‖ ^ 2
  let lossIntervened : Z → ℝ :=
    fun z => ‖residualTrajectory L (branch₂ z) (xOf z) - label z‖ ^ 2
  let envelope : Z → ℝ := fun z => 2 * lossClean z + 2 * δ ^ 2
  have hconst : Integrable (fun _ : Z => 2 * δ ^ 2) P.law := integrable_const _
  have henv_int : Integrable envelope P.law := (hint_clean.const_mul 2).add hconst
  have hae_bound : ∀ᵐ z ∂P.law, 0 ≤ lossIntervened z ∧ lossIntervened z ≤ envelope z :=
    Filter.Eventually.of_forall fun z => by
      refine ⟨sq_nonneg _, ?_⟩
      have hdiff :
          ‖residualTrajectory L (branch₂ z) (xOf z) -
              residualTrajectory L (branch₁ z) (xOf z)‖ ≤ δ := by
        rw [norm_sub_rev]
        exact multilayer_residual_perturbation L (branch₁ z) (branch₂ z) c ε_edit
          (hLip z) (hEdit z) (xOf z)
      exact PredictionGap.sq_norm_le_double
        (residualTrajectory L (branch₂ z) (xOf z))
        (residualTrajectory L (branch₁ z) (xOf z))
        (label z)
        hδ_nonneg
        hdiff
  obtain ⟨hi_int, hle⟩ :=
    P.risk_le_envelope lossIntervened envelope hmeas_intervened henv_int hae_bound
  refine ⟨hi_int, ?_, hsafe⟩
  unfold Population.risk at hle ⊢
  rw [integral_add (hint_clean.const_mul 2) hconst, integral_const_mul, integral_const] at hle
  have hprob : P.law.real Set.univ = 1 := by simp
  rw [hprob, one_smul] at hle
  exact hle

/-- Pointwise exact safety eradication via head ablation:
if every retained head `h ∈ keep` is orthogonal to the unsafe linear projection `P_unsafe : F →L[ℝ] F`
(`P_unsafe (headContribution ... h) = 0`), then `P_unsafe` vanishes identically on
`ablatedMultiHeadOutput scores mask values proj keep`, and the full model's unsafe projection
comes exclusively from the ablated heads `keepᶜ`. -/
theorem ablated_multiHead_unsafe_proj_zero {H n : ℕ}
    (P_unsafe : F →L[ℝ] F)
    (scores : Fin H → Tensor ℝ [n])
    (mask : Fin H → Tensor Bool [n])
    (values : Fin H → Fin n → E)
    (proj : Fin H → (E →L[ℝ] F))
    (keep : Finset (Fin H))
    (horth : ∀ h ∈ keep, P_unsafe (headContribution scores mask values proj h) = 0) :
    P_unsafe (ablatedMultiHeadOutput scores mask values proj keep) = 0 ∧
    P_unsafe (multiHeadOutput scores mask values proj) =
      ∑ h ∈ keepᶜ, P_unsafe (headContribution scores mask values proj h) := by
  have habl : P_unsafe (ablatedMultiHeadOutput scores mask values proj keep) = 0 := by
    unfold ablatedMultiHeadOutput
    rw [map_sum]
    exact Finset.sum_eq_zero horth
  refine ⟨habl, ?_⟩
  have hdiff := congr_arg P_unsafe (ablation_diff_eq scores mask values proj keep)
  rw [map_sub, habl, sub_zero, map_sum] at hdiff
  exact hdiff

/-- **Orthogonal Circuit Ablation: Exact Safety Eradication & Benign Risk Transfer
(`orthogonal_circuit_ablation_zero_unsafe_and_risk_bound`)**:
If head ablation `keep : Finset (Fin H)` removes the unsafe heads `keepᶜ` while every retained head
`h ∈ keep` has zero unsafe projection (`∀ z, ∀ h ∈ keep, P_unsafe (headContribution (scores z) (mask z) (values z) proj h) = 0`),
then:
1. For every observation `z : Z`, `P_unsafe (ablatedMultiHeadOutput (scores z) (mask z) (values z) proj keep) = 0`
   (exact safety eradication),
2. The ablated multi-head model has integrable squared-error loss on the benign task, and
3. Benign task population MSE satisfies
   `P.risk (ablated) ≤ 2 * P.risk (full) + 2 * (∑ h ∈ keepᶜ, ‖proj h‖ * B_V h) ^ 2`. -/
theorem orthogonal_circuit_ablation_zero_unsafe_and_risk_bound
    {Z : Type*} [MeasurableSpace Z] {H n : ℕ}
    (P : Population Z)
    (P_unsafe : F →L[ℝ] F)
    (scores : Z → Fin H → Tensor ℝ [n])
    (mask : Z → Fin H → Tensor Bool [n])
    (values : Z → Fin H → Fin n → E)
    (proj : Fin H → (E →L[ℝ] F))
    (keep : Finset (Fin H))
    (B_V : Fin H → ℝ)
    (label : Z → F)
    (horth : ∀ (z : Z), ∀ h ∈ keep,
      P_unsafe (headContribution (scores z) (mask z) (values z) proj h) = 0)
    (hB : ∀ h ∈ keepᶜ, 0 ≤ B_V h)
    (hmeas_ablated : AEStronglyMeasurable
      (fun z => ‖ablatedMultiHeadOutput (scores z) (mask z) (values z) proj keep - label z‖ ^ 2)
      P.law)
    (hint_full : P.IntegrableLoss
      (fun z => ‖multiHeadOutput (scores z) (mask z) (values z) proj - label z‖ ^ 2))
    (hae : ∀ᵐ z ∂P.law,
      (∀ h ∈ keepᶜ, ∃ i, (mask z h).getScalar i = true) ∧
      (∀ h ∈ keepᶜ, ∀ i, ‖values z h i‖ ≤ B_V h)) :
    (∀ z : Z, P_unsafe (ablatedMultiHeadOutput (scores z) (mask z) (values z) proj keep) = 0) ∧
    P.IntegrableLoss
      (fun z => ‖ablatedMultiHeadOutput (scores z) (mask z) (values z) proj keep - label z‖ ^ 2) ∧
    P.risk
      (fun z => ‖ablatedMultiHeadOutput (scores z) (mask z) (values z) proj keep - label z‖ ^ 2) ≤
      2 * P.risk
        (fun z => ‖multiHeadOutput (scores z) (mask z) (values z) proj - label z‖ ^ 2) +
        2 * (∑ h ∈ keepᶜ, ‖proj h‖ * B_V h) ^ 2 := by
  have hzero : ∀ z : Z,
      P_unsafe (ablatedMultiHeadOutput (scores z) (mask z) (values z) proj keep) = 0 :=
    fun z =>
      (ablated_multiHead_unsafe_proj_zero P_unsafe
        (scores z) (mask z) (values z) proj keep (horth z)).1
  obtain ⟨hint_abl, hrisk_abl⟩ :=
    ablation_population_mse_le P scores mask values proj keep B_V label
      hB hmeas_ablated hint_full hae
  exact ⟨hzero, hint_abl, hrisk_abl⟩

end OpenTheory.TransformerGeneralization
