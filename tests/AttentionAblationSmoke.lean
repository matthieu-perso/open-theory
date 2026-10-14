import OpenTheory.Architectures.AttentionAblation

noncomputable section

open TorchLean MeasureTheory OpenTheory OpenTheory.AttentionAblation
open scoped BigOperators

-- 1. Single-head ablation on a 2-head scalar attention layer:
-- Head 0 has equal scores and values [2, 4] (rowOutput = 3) with projection scale 1.
-- Head 1 has equal scores and values [10, 20] (rowOutput = 15) with projection scale 1.
-- Zeroing Head 1 leaves exactly Head 0's contribution (= 3).
example :
    let scores : Fin 2 → Tensor ℝ [2] := fun _ => Tensor.full [2] (0 : ℝ)
    let mask : Fin 2 → Tensor Bool [2] := fun _ => Tensor.full [2] true
    let values : Fin 2 → Fin 2 → ℝ := fun h i =>
      if h = 0 then (if i = 0 then 2 else 4) else (if i = 0 then 10 else 20)
    let proj : Fin 2 → (ℝ →L[ℝ] ℝ) := fun _ => ContinuousLinearMap.id ℝ ℝ
    multiHeadOutput scores mask values proj -
      ablateSingleHead scores mask values proj 1 =
      headContribution scores mask values proj 1 :=
  single_head_ablation_diff_eq _ _ _ _ 1

-- 2. Subset head-ablation norm bound check:
example {E F : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    [NormedAddCommGroup F] [NormedSpace ℝ F] {H n : ℕ}
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
      ∑ h ∈ keepᶜ, ‖proj h‖ * B_V h :=
  subset_ablation_norm_le scores mask values proj keep B_V hmask hval

-- 3. Value perturbation stability across heads:
example {E F : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    [NormedAddCommGroup F] [NormedSpace ℝ F] {H n : ℕ}
    (scores : Fin H → Tensor ℝ [n])
    (mask : Fin H → Tensor Bool [n])
    (v w : Fin H → Fin n → E)
    (proj : Fin H → (E →L[ℝ] F))
    (δ : Fin H → ℝ)
    (hmask : ∀ h, ∃ i, (mask h).getScalar i = true)
    (hδ : ∀ h i, ‖v h i - w h i‖ ≤ δ h) :
    ‖multiHeadOutput scores mask v proj -
      multiHeadOutput scores mask w proj‖ ≤
      ∑ h : Fin H, ‖proj h‖ * δ h :=
  multiHead_value_perturbation scores mask v w proj δ hmask hδ

-- 4. Downstream Lipschitz residual block ablation bound:
example {E F : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    [NormedAddCommGroup F] [NormedSpace ℝ F] {H n : ℕ}
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
      (L_ff : ℝ) * ∑ h ∈ keepᶜ, ‖proj h‖ * B_V h :=
  residual_block_ablation_bound G hG x scores mask values proj keep B_V hmask hval

-- 5. Multilayer residual perturbation propagation:
example {F : Type*} [NormedAddCommGroup F]
    (L : ℕ) (branch₁ branch₂ : Fin L → F → F) (c : Fin L → NNReal) (ε : Fin L → ℝ)
    (hLip : ∀ t, LipschitzWith (c t) (branch₁ t))
    (hPert : ∀ t y, ‖branch₁ t y - branch₂ t y‖ ≤ ε t)
    (x : F) :
    ‖residualTrajectory L branch₁ x - residualTrajectory L branch₂ x‖ ≤
      residualPerturbationEnvelope L (fun t => (c t : ℝ)) ε :=
  multilayer_residual_perturbation L branch₁ branch₂ c ε hLip hPert x

-- 6. Statistical population MSE risk transfer under head ablation:
example {Z E F : Type*} [MeasurableSpace Z]
    [NormedAddCommGroup E] [NormedSpace ℝ E]
    [NormedAddCommGroup F] [NormedSpace ℝ F] {H n : ℕ}
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
        2 * (∑ h ∈ keepᶜ, ‖proj h‖ * B_V h) ^ 2 :=
  ablation_population_mse_le P scores mask values proj keep B_V label
    hB hmeas_ablated hint_full hae
