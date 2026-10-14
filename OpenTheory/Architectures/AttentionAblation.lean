import OpenTheory.Architectures.Attention
import OpenTheory.Generalization.PredictionGap
import OpenTheory.Generalization.PopulationBounds
import Mathlib.Analysis.Normed.Operator.ContinuousLinearMap

/-!
# Multi-Head Attention Head-Ablation Fidelity and Statistical Risk Transfer

This module formalizes:
1. **Multi-head attention decomposition and head-ablation interventions**:
   representing an `H`-head attention layer via per-head masked softmax row outputs
   composed with continuous linear output projections `proj h : E →L[ℝ] F`, together
   with subset head-ablation `ablatedMultiHeadOutput` and single-head ablation
   `ablateSingleHead`.
2. **Exact ablation difference and norm bounds**:
   proving that the difference between the full multi-head output and the ablated output
   equals the sum over the ablated heads `keepᶜ`, and bounding this difference in terms
   of the projection operator norms `‖proj h‖` and value bounds `B_V h`, as well as
   proving value-perturbation stability across all heads.
3. **Downstream residual-block and multilayer Lipschitz propagation**:
   bounding the perturbation induced by head ablation after a post-attention residual
   connection and a Lipschitz downstream feedforward/normalization block, and proving
   an `L`-layer iterated residual perturbation bound.
4. **Statistical risk transfer under head ablation**:
   using `PredictionGap.sq_norm_le_double` and `Population.risk` to certify that pruning
   a subset of heads `keepᶜ` with small projection-value product `∑_{h ∉ keep} ‖proj h‖ B_V h`
   controls the population mean-squared error of the ablated model up to a factor of 2
   plus an explicit squared ablation penalty.
-/

noncomputable section

open TorchLean MeasureTheory
open scoped BigOperators

namespace OpenTheory.AttentionAblation

variable {E F : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
  [NormedAddCommGroup F] [NormedSpace ℝ F]

/-! ## 1. Multi-Head Attention Decomposition & Head-Ablation Intervention -/

/-- The projected output contribution of a single attention head `h : Fin H`. -/
def headContribution {H n : ℕ}
    (scores : Fin H → Tensor ℝ [n])
    (mask : Fin H → Tensor Bool [n])
    (values : Fin H → Fin n → E)
    (proj : Fin H → (E →L[ℝ] F))
    (h : Fin H) : F :=
  proj h (Attention.rowOutput (scores h) (mask h) (values h))

/-- The full multi-head attention output, summing projected contributions across all `H` heads. -/
def multiHeadOutput {H n : ℕ}
    (scores : Fin H → Tensor ℝ [n])
    (mask : Fin H → Tensor Bool [n])
    (values : Fin H → Fin n → E)
    (proj : Fin H → (E →L[ℝ] F)) : F :=
  ∑ h : Fin H, headContribution scores mask values proj h

/-- The ablated multi-head attention output retaining only the subset of heads `keep`. -/
def ablatedMultiHeadOutput {H n : ℕ}
    (scores : Fin H → Tensor ℝ [n])
    (mask : Fin H → Tensor Bool [n])
    (values : Fin H → Fin n → E)
    (proj : Fin H → (E →L[ℝ] F))
    (keep : Finset (Fin H)) : F :=
  ∑ h ∈ keep, headContribution scores mask values proj h

/-- Special case of head ablation that zeros out a single head `h₀ : Fin H` while
leaving all other heads unchanged. -/
def ablateSingleHead {H n : ℕ}
    (scores : Fin H → Tensor ℝ [n])
    (mask : Fin H → Tensor Bool [n])
    (values : Fin H → Fin n → E)
    (proj : Fin H → (E →L[ℝ] F))
    (h₀ : Fin H) : F :=
  ablatedMultiHeadOutput scores mask values proj (Finset.univ.erase h₀)

/-! ## 2. Exact Ablation Difference & Norm Bounds -/

/-- Norm bound for a single head's projected contribution when at least one key is unmasked
and its values are bounded in norm by `B`. -/
theorem headContribution_norm_le {H n : ℕ}
    (scores : Fin H → Tensor ℝ [n])
    (mask : Fin H → Tensor Bool [n])
    (values : Fin H → Fin n → E)
    (proj : Fin H → (E →L[ℝ] F))
    (h : Fin H)
    (hmask : ∃ i, (mask h).getScalar i = true)
    (B : ℝ) (hval : ∀ i, ‖values h i‖ ≤ B) :
    ‖headContribution scores mask values proj h‖ ≤ ‖proj h‖ * B := by
  unfold headContribution
  have hrow := Attention.rowOutput_norm_le (scores h) (mask h) (values h) hmask B hval
  exact (proj h).le_opNorm _ |>.trans
    (mul_le_mul_of_nonneg_left hrow ( norm_nonneg (proj h) ))

/-- If all keys in head `h` are masked out, its projected contribution is identically zero. -/
theorem headContribution_blocked {H n : ℕ}
    (scores : Fin H → Tensor ℝ [n])
    (mask : Fin H → Tensor Bool [n])
    (values : Fin H → Fin n → E)
    (proj : Fin H → (E →L[ℝ] F))
    (h : Fin H) (hmask : ∀ i, (mask h).getScalar i = false) :
    headContribution scores mask values proj h = 0 := by
  unfold headContribution
  rw [Attention.rowOutput_blocked (scores h) (mask h) (values h) hmask, map_zero]

/-- Exact algebraic decomposition: the difference between the full multi-head output
and the `keep`-ablated output is the sum of head contributions over `keepᶜ`. -/
theorem ablation_diff_eq {H n : ℕ}
    (scores : Fin H → Tensor ℝ [n])
    (mask : Fin H → Tensor Bool [n])
    (values : Fin H → Fin n → E)
    (proj : Fin H → (E →L[ℝ] F))
    (keep : Finset (Fin H)) :
    multiHeadOutput scores mask values proj -
      ablatedMultiHeadOutput scores mask values proj keep =
      ∑ h ∈ keepᶜ, headContribution scores mask values proj h := by
  unfold multiHeadOutput ablatedMultiHeadOutput
  have hsplit := (Finset.sum_add_sum_compl keep
    (fun h => headContribution scores mask values proj h)).symm
  exact sub_eq_of_eq_add' hsplit

/-- Exact single-head ablation identity: zeroing head `h₀` changes the multi-head output
by exactly `headContribution ... h₀`. -/
theorem single_head_ablation_diff_eq {H n : ℕ}
    (scores : Fin H → Tensor ℝ [n])
    (mask : Fin H → Tensor Bool [n])
    (values : Fin H → Fin n → E)
    (proj : Fin H → (E →L[ℝ] F))
    (h₀ : Fin H) :
    multiHeadOutput scores mask values proj -
      ablateSingleHead scores mask values proj h₀ =
      headContribution scores mask values proj h₀ := by
  unfold multiHeadOutput ablateSingleHead ablatedMultiHeadOutput
  have hsplit := (Finset.add_sum_erase Finset.univ
    (fun h => headContribution scores mask values proj h) (Finset.mem_univ h₀)).symm
  exact sub_eq_of_eq_add hsplit

/-- Subset head-ablation fidelity: if every ablated head `h ∈ keepᶜ` has at least one
unmasked key and values bounded by `B_V h`, then the output change from ablating `keepᶜ`
is at most `∑ h ∈ keepᶜ, ‖proj h‖ * B_V h`. -/
theorem subset_ablation_norm_le {H n : ℕ}
    (scores : Fin H → Tensor ℝ [n])
    (mask : Fin H → Tensor Bool [n])
    (values : Fin H → Fin n → E)
    (proj : Fin H → (E →L[ℝ] F))
    (keep : Finset (Fin H))
    (B_V : Fin H → ℝ)
    (hmask : ∀ h ∈ keepᶜ, ∃ i, (mask h).getScalar i = true)
    (hval : ∀ h ∈ keepᶜ, ∀ i, ‖values h i‖ ≤ B_V h) :
    ‖multiHeadOutput scores mask values proj -
      ablatedMultiHeadOutput scores mask values proj keep‖ ≤
      ∑ h ∈ keepᶜ, ‖proj h‖ * B_V h := by
  rw [ablation_diff_eq]
  calc
    ‖∑ h ∈ keepᶜ, headContribution scores mask values proj h‖ ≤
        ∑ h ∈ keepᶜ, ‖headContribution scores mask values proj h‖ :=
      norm_sum_le _ _
    _ ≤ ∑ h ∈ keepᶜ, ‖proj h‖ * B_V h :=
      Finset.sum_le_sum fun h hh =>
        headContribution_norm_le scores mask values proj h (hmask h hh) (B_V h) (hval h hh)

/-- Single-head ablation fidelity: if head `h₀` has at least one unmasked key and values
bounded by `B_V h₀`, then zeroing out head `h₀` changes the multi-head output by at most
`‖proj h₀‖ * B_V h₀`. -/
theorem single_head_ablation_norm_le {H n : ℕ}
    (scores : Fin H → Tensor ℝ [n])
    (mask : Fin H → Tensor Bool [n])
    (values : Fin H → Fin n → E)
    (proj : Fin H → (E →L[ℝ] F))
    (h₀ : Fin H)
    (B_V : Fin H → ℝ)
    (hmask : ∃ i, (mask h₀).getScalar i = true)
    (hval : ∀ i, ‖values h₀ i‖ ≤ B_V h₀) :
    ‖multiHeadOutput scores mask values proj -
      ablateSingleHead scores mask values proj h₀‖ ≤
      ‖proj h₀‖ * B_V h₀ := by
  rw [single_head_ablation_diff_eq]
  exact headContribution_norm_le scores mask values proj h₀ hmask (B_V h₀) hval

/-- Value-perturbation stability across all heads: with fixed scores and masks, if the
values on each head `h` are perturbed by at most `δ h`, then the multi-head output
changes by at most `∑ h : Fin H, ‖proj h‖ * δ h`. -/
theorem multiHead_value_perturbation {H n : ℕ}
    (scores : Fin H → Tensor ℝ [n])
    (mask : Fin H → Tensor Bool [n])
    (v w : Fin H → Fin n → E)
    (proj : Fin H → (E →L[ℝ] F))
    (δ : Fin H → ℝ)
    (hmask : ∀ h, ∃ i, (mask h).getScalar i = true)
    (hδ : ∀ h i, ‖v h i - w h i‖ ≤ δ h) :
    ‖multiHeadOutput scores mask v proj -
      multiHeadOutput scores mask w proj‖ ≤
      ∑ h : Fin H, ‖proj h‖ * δ h := by
  unfold multiHeadOutput
  rw [← Finset.sum_sub_distrib]
  calc
    ‖∑ h : Fin H, (headContribution scores mask v proj h -
        headContribution scores mask w proj h)‖ ≤
        ∑ h : Fin H, ‖headContribution scores mask v proj h -
          headContribution scores mask w proj h‖ :=
      norm_sum_le _ _
    _ ≤ ∑ h : Fin H, ‖proj h‖ * δ h := by
      refine Finset.sum_le_sum fun h _ => ?_
      unfold headContribution
      rw [← map_sub]
      have hrow := Attention.rowOutput_perturbation (scores h) (mask h) (v h) (w h)
        (hmask h) (δ h) (hδ h)
      exact (proj h).le_opNorm _ |>.trans
        (mul_le_mul_of_nonneg_left hrow (norm_nonneg (proj h)))

/-! ## 3. Propagation Through Downstream Residual & Lipschitz Layers -/

/-- A post-attention residual block feeding `x + mha` into a downstream map `G : F → F`
(representing layer normalization + feedforward sublayer + residual connection, or a
stack of downstream transformer blocks). -/
def postAttentionBlock (G : F → F) (x mha : F) : F :=
  G (x + mha)

/-- Downstream residual-block ablation bound: if the post-attention block `G : F → F`
is `L_ff`-Lipschitz, then ablating the attention heads outside `keep` changes the
output `G (x + mha)` by at most `(L_ff : ℝ) * ∑ h ∈ keepᶜ, ‖proj h‖ * B_V h`. -/
theorem residual_block_ablation_bound {H n : ℕ}
    (G : F → F) {L_ff : NNReal} (hG : LipschitzWith L_ff G)
    (x : F)
    (scores : Fin H → Tensor ℝ [n])
    (mask : Fin H → Tensor Bool [n])
    (values : Fin H → Fin n → E)
    (proj : Fin H → (E →L[ℝ] F))
    (keep : Finset (Fin H))
    (B_V : Fin H → ℝ)
    (hmask : ∀ h ∈ keepᶜ, ∃ i, (mask h).getScalar i = true)
    (hval : ∀ h ∈ keepᶜ, ∀ i, ‖values h i‖ ≤ B_V h) :
    ‖postAttentionBlock G x (multiHeadOutput scores mask values proj) -
      postAttentionBlock G x (ablatedMultiHeadOutput scores mask values proj keep)‖ ≤
      (L_ff : ℝ) * ∑ h ∈ keepᶜ, ‖proj h‖ * B_V h := by
  unfold postAttentionBlock
  have hlip : ‖G (x + multiHeadOutput scores mask values proj) -
      G (x + ablatedMultiHeadOutput scores mask values proj keep)‖ ≤
      (L_ff : ℝ) * ‖(x + multiHeadOutput scores mask values proj) -
        (x + ablatedMultiHeadOutput scores mask values proj keep)‖ := by
    simpa [dist_eq_norm] using
      hG.dist_le_mul (x + multiHeadOutput scores mask values proj)
        (x + ablatedMultiHeadOutput scores mask values proj keep)
  rw [add_sub_add_left_eq_sub] at hlip
  exact hlip.trans (mul_le_mul_of_nonneg_left
    (subset_ablation_norm_le scores mask values proj keep B_V hmask hval) L_ff.coe_nonneg)

/-- Single-head downstream residual-block ablation bound. -/
theorem residual_block_single_head_ablation_bound {H n : ℕ}
    (G : F → F) {L_ff : NNReal} (hG : LipschitzWith L_ff G)
    (x : F)
    (scores : Fin H → Tensor ℝ [n])
    (mask : Fin H → Tensor Bool [n])
    (values : Fin H → Fin n → E)
    (proj : Fin H → (E →L[ℝ] F))
    (h₀ : Fin H)
    (B_V : Fin H → ℝ)
    (hmask : ∃ i, (mask h₀).getScalar i = true)
    (hval : ∀ i, ‖values h₀ i‖ ≤ B_V h₀) :
    ‖postAttentionBlock G x (multiHeadOutput scores mask values proj) -
      postAttentionBlock G x (ablateSingleHead scores mask values proj h₀)‖ ≤
      (L_ff : ℝ) * (‖proj h₀‖ * B_V h₀) := by
  unfold postAttentionBlock
  have hlip : ‖G (x + multiHeadOutput scores mask values proj) -
      G (x + ablateSingleHead scores mask values proj h₀)‖ ≤
      (L_ff : ℝ) * ‖(x + multiHeadOutput scores mask values proj) -
        (x + ablateSingleHead scores mask values proj h₀)‖ := by
    simpa [dist_eq_norm] using
      hG.dist_le_mul (x + multiHeadOutput scores mask values proj)
        (x + ablateSingleHead scores mask values proj h₀)
  rw [add_sub_add_left_eq_sub] at hlip
  exact hlip.trans (mul_le_mul_of_nonneg_left
    (single_head_ablation_norm_le scores mask values proj h₀ B_V hmask hval) L_ff.coe_nonneg)

/-- Recurrence for the cumulative perturbation bound through `L` residual stages,
where stage `t` has branch Lipschitz constant `c t ≥ 0` (so the residual map `id + f_t`
is `(1 + c t)`-Lipschitz) and experiences an additive intervention error `ε t`. -/
def residualPerturbationEnvelope : (L : ℕ) → (Fin L → ℝ) → (Fin L → ℝ) → ℝ
  | 0, _, _ => 0
  | L + 1, c, ε =>
      (1 + c (Fin.last L)) *
        residualPerturbationEnvelope L (fun t => c t.castSucc) (fun t => ε t.castSucc) +
        ε (Fin.last L)

/-- State trajectory of an `L`-layer residual network `x_{t+1} = x_t + branch_t(x_t)`. -/
def residualTrajectory : (L : ℕ) → (Fin L → F → F) → F → F
  | 0, _, x => x
  | L + 1, branch, x =>
      let prev := residualTrajectory L (fun t => branch t.castSucc) x
      prev + branch (Fin.last L) prev

/-- Nonnegativity of `residualPerturbationEnvelope` when branch Lipschitz constants
and stage perturbations are nonnegative. -/
theorem residualPerturbationEnvelope_nonneg :
    ∀ (L : ℕ) (c ε : Fin L → ℝ),
      (∀ t, 0 ≤ c t) → (∀ t, 0 ≤ ε t) → 0 ≤ residualPerturbationEnvelope L c ε
  | 0, _, _, _, _ => le_refl 0
  | L + 1, c, ε, hc, hε => by
      unfold residualPerturbationEnvelope
      have ih := residualPerturbationEnvelope_nonneg L
        (fun t => c t.castSucc) (fun t => ε t.castSucc)
        (fun t => hc t.castSucc) (fun t => hε t.castSucc)
      have h1 : 0 ≤ 1 + c (Fin.last L) := by linarith [hc (Fin.last L)]
      exact add_nonneg (mul_nonneg h1 ih) (hε (Fin.last L))

omit [NormedSpace ℝ F] in
/-- Multilayer residual perturbation stability: if each residual stage `t : Fin L` has a
branch `branch₁ t` that is `c t`-Lipschitz and differs pointwise from an ablated/perturbed
branch `branch₂ t` by at most `ε t`, then the end-to-end `L`-layer residual trajectories
starting from the same input `x` differ by at most `residualPerturbationEnvelope L c ε`. -/
theorem multilayer_residual_perturbation :
    ∀ (L : ℕ) (branch₁ branch₂ : Fin L → F → F) (c : Fin L → NNReal) (ε : Fin L → ℝ)
      (_hLip : ∀ t, LipschitzWith (c t) (branch₁ t))
      (_hPert : ∀ t y, ‖branch₁ t y - branch₂ t y‖ ≤ ε t)
      (x : F),
      ‖residualTrajectory L branch₁ x - residualTrajectory L branch₂ x‖ ≤
        residualPerturbationEnvelope L (fun t => (c t : ℝ)) ε
  | 0, _, _, _, _, _, _, x => by
      simp [residualTrajectory, residualPerturbationEnvelope]
  | L + 1, branch₁, branch₂, c, ε, hLip, hPert, x => by
      simp only [residualTrajectory, residualPerturbationEnvelope]
      set y₁ := residualTrajectory L (fun t => branch₁ t.castSucc) x
      set y₂ := residualTrajectory L (fun t => branch₂ t.castSucc) x
      have ih := multilayer_residual_perturbation L
        (fun t => branch₁ t.castSucc) (fun t => branch₂ t.castSucc)
        (fun t => c t.castSucc) (fun t => ε t.castSucc)
        (fun t => hLip t.castSucc) (fun t y => hPert t.castSucc y) x
      have hdecomp : (y₁ + branch₁ (Fin.last L) y₁) - (y₂ + branch₂ (Fin.last L) y₂) =
          (y₁ - y₂) + (branch₁ (Fin.last L) y₁ - branch₁ (Fin.last L) y₂) +
            (branch₁ (Fin.last L) y₂ - branch₂ (Fin.last L) y₂) := by abel
      rw [hdecomp]
      have htri1 := norm_add_le
        ((y₁ - y₂) + (branch₁ (Fin.last L) y₁ - branch₁ (Fin.last L) y₂))
        (branch₁ (Fin.last L) y₂ - branch₂ (Fin.last L) y₂)
      have htri2 := norm_add_le (y₁ - y₂) (branch₁ (Fin.last L) y₁ - branch₁ (Fin.last L) y₂)
      have hlip_last : ‖branch₁ (Fin.last L) y₁ - branch₁ (Fin.last L) y₂‖ ≤
          (c (Fin.last L) : ℝ) * ‖y₁ - y₂‖ := by
        simpa [dist_eq_norm] using (hLip (Fin.last L)).dist_le_mul y₁ y₂
      have hpert_last := hPert (Fin.last L) y₂
      have hcomb : ‖y₁ - y₂‖ + (c (Fin.last L) : ℝ) * ‖y₁ - y₂‖ =
          (1 + (c (Fin.last L) : ℝ)) * ‖y₁ - y₂‖ := by ring
      have h1_nonneg : 0 ≤ 1 + (c (Fin.last L) : ℝ) := by
        linarith [(c (Fin.last L)).coe_nonneg]
      have hmul := mul_le_mul_of_nonneg_left ih h1_nonneg
      linarith

/-! ## 4. Statistical Risk Transfer Under Head Ablation -/

/-- Nonnegativity of the head-ablation bound `∑ h ∈ keepᶜ, ‖proj h‖ * B_V h` when
the value bounds `B_V h` are nonnegative on `keepᶜ`. -/
theorem ablationBound_nonneg {H : ℕ}
    (proj : Fin H → (E →L[ℝ] F)) (keep : Finset (Fin H)) (B_V : Fin H → ℝ)
    (hB : ∀ h ∈ keepᶜ, 0 ≤ B_V h) :
    0 ≤ ∑ h ∈ keepᶜ, ‖proj h‖ * B_V h :=
  Finset.sum_nonneg fun h hh => mul_nonneg (norm_nonneg (proj h)) (hB h hh)

/-- Pointwise squared-error risk comparison between the head-ablated multi-head model
and the full multi-head model: for any observation `z` where ablated heads have valid
masks and values bounded by `B_V h`, the squared prediction error of the ablated model
is at most twice the squared error of the full model plus twice the squared ablation bound. -/
theorem ablation_pointwise_sq_error_le {Z : Type*} {H n : ℕ}
    (scores : Z → Fin H → Tensor ℝ [n])
    (mask : Z → Fin H → Tensor Bool [n])
    (values : Z → Fin H → Fin n → E)
    (proj : Fin H → (E →L[ℝ] F))
    (keep : Finset (Fin H))
    (B_V : Fin H → ℝ)
    (label : Z → F)
    (z : Z)
    (hB : ∀ h ∈ keepᶜ, 0 ≤ B_V h)
    (hmask : ∀ h ∈ keepᶜ, ∃ i, (mask z h).getScalar i = true)
    (hval : ∀ h ∈ keepᶜ, ∀ i, ‖values z h i‖ ≤ B_V h) :
    ‖ablatedMultiHeadOutput (scores z) (mask z) (values z) proj keep - label z‖ ^ 2 ≤
      2 * ‖multiHeadOutput (scores z) (mask z) (values z) proj - label z‖ ^ 2 +
        2 * (∑ h ∈ keepᶜ, ‖proj h‖ * B_V h) ^ 2 := by
  have hdiff : ‖ablatedMultiHeadOutput (scores z) (mask z) (values z) proj keep -
      multiHeadOutput (scores z) (mask z) (values z) proj‖ ≤
      ∑ h ∈ keepᶜ, ‖proj h‖ * B_V h := by
    rw [norm_sub_rev]
    exact subset_ablation_norm_le (scores z) (mask z) (values z) proj keep B_V hmask hval
  exact PredictionGap.sq_norm_le_double
    (ablatedMultiHeadOutput (scores z) (mask z) (values z) proj keep)
    (multiHeadOutput (scores z) (mask z) (values z) proj)
    (label z)
    (ablationBound_nonneg proj keep B_V hB)
    hdiff

/-- Statistical Risk Transfer under Head Ablation:
Given a population `P : Population Z`, if the full multi-head attention predictor has
integrable squared-error loss and the ablated heads `keepᶜ` have valid masks and
values bounded by `B_V h` almost everywhere under `P.law`, then the `keep`-ablated
predictor also has integrable squared-error loss and its population MSE satisfies:
`P.risk (ablated) ≤ 2 * P.risk (full) + 2 * (∑ h ∈ keepᶜ, ‖proj h‖ * B_V h) ^ 2`. -/
theorem ablation_population_mse_le {Z : Type*} [MeasurableSpace Z] {H n : ℕ}
    (P : Population Z)
    (scores : Z → Fin H → Tensor ℝ [n])
    (mask : Z → Fin H → Tensor Bool [n])
    (values : Z → Fin H → Fin n → E)
    (proj : Fin H → (E →L[ℝ] F))
    (keep : Finset (Fin H))
    (B_V : Fin H → ℝ)
    (label : Z → F)
    (hB : ∀ h ∈ keepᶜ, 0 ≤ B_V h)
    (hmeas_ablated : AEStronglyMeasurable
      (fun z => ‖ablatedMultiHeadOutput (scores z) (mask z) (values z) proj keep - label z‖ ^ 2)
      P.law)
    (hint_full : P.IntegrableLoss
      (fun z => ‖multiHeadOutput (scores z) (mask z) (values z) proj - label z‖ ^ 2))
    (hae : ∀ᵐ z ∂P.law,
      (∀ h ∈ keepᶜ, ∃ i, (mask z h).getScalar i = true) ∧
      (∀ h ∈ keepᶜ, ∀ i, ‖values z h i‖ ≤ B_V h)) :
    P.IntegrableLoss
      (fun z => ‖ablatedMultiHeadOutput (scores z) (mask z) (values z) proj keep - label z‖ ^ 2) ∧
    P.risk
      (fun z => ‖ablatedMultiHeadOutput (scores z) (mask z) (values z) proj keep - label z‖ ^ 2) ≤
      2 * P.risk
        (fun z => ‖multiHeadOutput (scores z) (mask z) (values z) proj - label z‖ ^ 2) +
        2 * (∑ h ∈ keepᶜ, ‖proj h‖ * B_V h) ^ 2 := by
  let δ := ∑ h ∈ keepᶜ, ‖proj h‖ * B_V h
  let lossFull : Z → ℝ :=
    fun z => ‖multiHeadOutput (scores z) (mask z) (values z) proj - label z‖ ^ 2
  let lossAblated : Z → ℝ :=
    fun z => ‖ablatedMultiHeadOutput (scores z) (mask z) (values z) proj keep - label z‖ ^ 2
  let envelope : Z → ℝ := fun z => 2 * lossFull z + 2 * δ ^ 2
  have hconst : Integrable (fun _ : Z => 2 * δ ^ 2) P.law := integrable_const _
  have henv_int : Integrable envelope P.law := (hint_full.const_mul 2).add hconst
  have hae_bound : ∀ᵐ z ∂P.law, 0 ≤ lossAblated z ∧ lossAblated z ≤ envelope z :=
    hae.mono fun z hz =>
      ⟨sq_nonneg _,
        ablation_pointwise_sq_error_le scores mask values proj keep B_V label z hB hz.1 hz.2⟩
  obtain ⟨hi_abl, hle⟩ := P.risk_le_envelope lossAblated envelope hmeas_ablated henv_int hae_bound
  refine ⟨hi_abl, ?_⟩
  unfold Population.risk at hle ⊢
  rw [integral_add (hint_full.const_mul 2) hconst, integral_const_mul, integral_const] at hle
  have hprob : P.law.real Set.univ = 1 := by simp
  rw [hprob, one_smul] at hle
  exact hle

/-- Single-head ablation population MSE transfer: pruning a single head `h₀` with
projection operator `proj h₀` and value bound `B_V h₀ ≥ 0` bounds the population MSE
by `2 * P.risk (full) + 2 * (‖proj h₀‖ * B_V h₀) ^ 2`. -/
theorem single_head_ablation_population_mse_le {Z : Type*} [MeasurableSpace Z] {H n : ℕ}
    (P : Population Z)
    (scores : Z → Fin H → Tensor ℝ [n])
    (mask : Z → Fin H → Tensor Bool [n])
    (values : Z → Fin H → Fin n → E)
    (proj : Fin H → (E →L[ℝ] F))
    (h₀ : Fin H)
    (B_V : Fin H → ℝ)
    (label : Z → F)
    (hB : 0 ≤ B_V h₀)
    (hmeas_ablated : AEStronglyMeasurable
      (fun z => ‖ablateSingleHead (scores z) (mask z) (values z) proj h₀ - label z‖ ^ 2)
      P.law)
    (hint_full : P.IntegrableLoss
      (fun z => ‖multiHeadOutput (scores z) (mask z) (values z) proj - label z‖ ^ 2))
    (hae : ∀ᵐ z ∂P.law,
      (∃ i, (mask z h₀).getScalar i = true) ∧
      (∀ i, ‖values z h₀ i‖ ≤ B_V h₀)) :
    P.IntegrableLoss
      (fun z => ‖ablateSingleHead (scores z) (mask z) (values z) proj h₀ - label z‖ ^ 2) ∧
    P.risk
      (fun z => ‖ablateSingleHead (scores z) (mask z) (values z) proj h₀ - label z‖ ^ 2) ≤
      2 * P.risk
        (fun z => ‖multiHeadOutput (scores z) (mask z) (values z) proj - label z‖ ^ 2) +
        2 * (‖proj h₀‖ * B_V h₀) ^ 2 := by
  let δ := ‖proj h₀‖ * B_V h₀
  let lossFull : Z → ℝ :=
    fun z => ‖multiHeadOutput (scores z) (mask z) (values z) proj - label z‖ ^ 2
  let lossAblated : Z → ℝ :=
    fun z => ‖ablateSingleHead (scores z) (mask z) (values z) proj h₀ - label z‖ ^ 2
  let envelope : Z → ℝ := fun z => 2 * lossFull z + 2 * δ ^ 2
  have hδ : 0 ≤ δ := mul_nonneg (norm_nonneg (proj h₀)) hB
  have hconst : Integrable (fun _ : Z => 2 * δ ^ 2) P.law := integrable_const _
  have henv_int : Integrable envelope P.law := (hint_full.const_mul 2).add hconst
  have hae_bound : ∀ᵐ z ∂P.law, 0 ≤ lossAblated z ∧ lossAblated z ≤ envelope z :=
    hae.mono fun z hz => by
      refine ⟨sq_nonneg _, ?_⟩
      have hdiff : ‖ablateSingleHead (scores z) (mask z) (values z) proj h₀ -
          multiHeadOutput (scores z) (mask z) (values z) proj‖ ≤ δ := by
        rw [norm_sub_rev]
        exact single_head_ablation_norm_le (scores z) (mask z) (values z) proj h₀ B_V hz.1 hz.2
      exact PredictionGap.sq_norm_le_double
        (ablateSingleHead (scores z) (mask z) (values z) proj h₀)
        (multiHeadOutput (scores z) (mask z) (values z) proj)
        (label z) hδ hdiff
  obtain ⟨hi_abl, hle⟩ := P.risk_le_envelope lossAblated envelope hmeas_ablated henv_int hae_bound
  refine ⟨hi_abl, ?_⟩
  unfold Population.risk at hle ⊢
  rw [integral_add (hint_full.const_mul 2) hconst, integral_const_mul, integral_const] at hle
  have hprob : P.law.real Set.univ = 1 := by simp
  rw [hprob, one_smul] at hle
  exact hle

end OpenTheory.AttentionAblation
