import OpenTheory.Architectures.InContextRetrieval

noncomputable section

open TorchLean MeasureTheory OpenTheory OpenTheory.Attention
open OpenTheory.AttentionLogitStability OpenTheory.AttentionAblation
open OpenTheory.InContextRetrieval
open scoped BigOperators

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

-- 1. Score step and T-step trajectory linear logit-margin growth
example {n : ℕ} (η γ₀ Δ_g : ℝ)
    (scores₀ : Fin n → ℝ) (g_seq : ℕ → Fin n → ℝ) (mask : Fin n → Bool)
    (i_star : Fin n) (hη : 0 ≤ η)
    (hmargin₀ : ∀ j : Fin n, mask j = true → j ≠ i_star → scores₀ j ≤ scores₀ i_star - γ₀)
    (hsep : ∀ (t : ℕ) (j : Fin n), mask j = true → j ≠ i_star →
      g_seq t j ≤ g_seq t i_star - Δ_g)
    (T : ℕ) :
    ∀ j : Fin n, mask j = true → j ≠ i_star →
      scoreTrajectory η scores₀ g_seq T j ≤
        scoreTrajectory η scores₀ g_seq T i_star - (γ₀ + (T : ℝ) * η * Δ_g) :=
  logitMargin_trajectory_ge η γ₀ Δ_g scores₀ g_seq mask i_star hη hmargin₀ hsep T

-- 2. Concrete 2-token numerical check of scoreTrajectory:
-- Starting from equal scores [0, 0] (γ₀ = 0) with g_seq t = [1, 0] (Δ_g = 1 at i_star = 0)
-- and learning rate η = 2, after T = 3 steps token 0 has score 6 and token 1 has score 0!
example :
    let scores₀ : Fin 2 → ℝ := fun _ => 0
    let g_seq : ℕ → Fin 2 → ℝ := fun _ j => if j = 0 then 1 else 0
    scoreTrajectory 2 scores₀ g_seq 3 0 = 6 ∧
    scoreTrajectory 2 scores₀ g_seq 3 1 = 0 := by
  norm_num [scoreTrajectory, scoreStep]

-- 3. Bilinear query-key score update decomposition and correlation-margin separation
example {n d_k : ℕ} (scale η Δ_K : ℝ) (q d_q : Fin d_k → ℝ)
    (K : Fin n → Fin d_k → ℝ) (mask : Fin n → Bool) (i_star : Fin n)
    (hscale : 0 ≤ scale)
    (hcorr : ∀ j : Fin n, mask j = true → j ≠ i_star →
      ∑ l : Fin d_k, K i_star l * (K i_star l - K j l) ≥ Δ_K) :
    bilinearScore scale (fun l => q l + η * d_q l) K =
      scoreStep η (bilinearScore scale q K) (bilinearScore scale d_q K) ∧
    (∀ j : Fin n, mask j = true → j ≠ i_star →
      bilinearScore scale (K i_star) K j ≤
        bilinearScore scale (K i_star) K i_star - scale * Δ_K) :=
  ⟨bilinearScore_step_eq scale η q d_q K,
   queryStep_score_separation scale Δ_K K mask i_star hscale hcorr⟩

-- 4. Exponential convergence of target attention weight along training
example {n : ℕ} (η γ₀ Δ_g : ℝ)
    (scores₀ : Fin n → ℝ) (g_seq : ℕ → Fin n → ℝ) (mask : Tensor Bool [n])
    (i_star : Fin n) (hη : 0 ≤ η) (hi_star : mask.getScalar i_star = true)
    (hmargin₀ : ∀ j : Fin n, mask.getScalar j = true → j ≠ i_star → scores₀ j ≤ scores₀ i_star - γ₀)
    (hsep : ∀ (t : ℕ) (j : Fin n), mask.getScalar j = true → j ≠ i_star →
      g_seq t j ≤ g_seq t i_star - Δ_g)
    (T : ℕ) :
    1 - (rowWeights (scoreTensor (scoreTrajectory η scores₀ g_seq T)) mask).getScalar i_star ≤
      ((n : ℝ) - 1) * Real.exp (-γ₀) * Real.exp (-(T : ℝ) * η * Δ_g) :=
  icl_attention_weight_convergence_real_sub η γ₀ Δ_g scores₀ g_seq mask i_star
    hη hi_star hmargin₀ hsep T

-- 5. Exponential decay of in-context retrieval error in any normed space E
example {n : ℕ} (η γ₀ Δ_g D_V : ℝ)
    (scores₀ : Fin n → ℝ) (g_seq : ℕ → Fin n → ℝ) (mask : Tensor Bool [n])
    (values : Fin n → E) (i_star : Fin n) (hη : 0 ≤ η) (hi_star : mask.getScalar i_star = true)
    (hmargin₀ : ∀ j : Fin n, mask.getScalar j = true → j ≠ i_star → scores₀ j ≤ scores₀ i_star - γ₀)
    (hsep : ∀ (t : ℕ) (j : Fin n), mask.getScalar j = true → j ≠ i_star →
      g_seq t j ≤ g_seq t i_star - Δ_g)
    (hD_nonneg : 0 ≤ D_V)
    (hD : ∀ j : Fin n, mask.getScalar j = true → ‖values j - values i_star‖ ≤ D_V)
    (T : ℕ) :
    ‖rowOutput (scoreTensor (scoreTrajectory η scores₀ g_seq T)) mask values - values i_star‖ ≤
      ((n : ℝ) - 1) * D_V * Real.exp (-γ₀) * Real.exp (-(T : ℝ) * η * Δ_g) :=
  icl_retrieval_error_exponential_decay η γ₀ Δ_g D_V scores₀ g_seq mask values i_star
    hη hi_star hmargin₀ hsep hD_nonneg hD T

-- 6. Explicit step complexity certificate for ε-accurate in-context retrieval
example {n : ℕ} (η γ₀ Δ_g D_V ε : ℝ)
    (scores₀ : Fin n → ℝ) (g_seq : ℕ → Fin n → ℝ) (mask : Tensor Bool [n])
    (values : Fin n → E) (i_star : Fin n) (hη : 0 ≤ η) (hi_star : mask.getScalar i_star = true)
    (hmargin₀ : ∀ j : Fin n, mask.getScalar j = true → j ≠ i_star → scores₀ j ≤ scores₀ i_star - γ₀)
    (hsep : ∀ (t : ℕ) (j : Fin n), mask.getScalar j = true → j ≠ i_star →
      g_seq t j ≤ g_seq t i_star - Δ_g)
    (hn : 0 < (n : ℝ) - 1) (hD_pos : 0 < D_V) (hε_pos : 0 < ε)
    (hD : ∀ j : Fin n, mask.getScalar j = true → ‖values j - values i_star‖ ≤ D_V)
    (T : ℕ) (hT : (T : ℝ) * η * Δ_g ≥ Real.log (((n : ℝ) - 1) * D_V / ε) - γ₀) :
    ‖rowOutput (scoreTensor (scoreTrajectory η scores₀ g_seq T)) mask values - values i_star‖ ≤ ε :=
  icl_step_complexity_certificate η γ₀ Δ_g D_V ε scores₀ g_seq mask values i_star
    hη hi_star hmargin₀ hsep hn hD_pos hε_pos hD T hT

-- 7. Population MSE risk bound for trained in-context retrieval vs. oracle retriever
example {Z : Type*} [MeasurableSpace Z] {n : ℕ}
    (P : Population Z) (η γ₀ Δ_g D_V : ℝ)
    (scores₀ : Z → Fin n → ℝ) (g_seq : Z → ℕ → Fin n → ℝ) (mask : Z → Tensor Bool [n])
    (values : Z → Fin n → E) (i_star : Z → Fin n) (label : Z → E) (T : ℕ)
    (hη : 0 ≤ η) (hD_nonneg : 0 ≤ D_V)
    (hmeas_icl : AEStronglyMeasurable
      (fun z =>
        ‖rowOutput (scoreTensor (scoreTrajectory η (scores₀ z) (g_seq z) T))
          (mask z) (values z) - label z‖ ^ 2)
      P.law)
    (hint_oracle : P.IntegrableLoss (fun z => ‖values z (i_star z) - label z‖ ^ 2))
    (hae : ∀ᵐ z ∂P.law,
      (mask z).getScalar (i_star z) = true ∧
      (∀ j : Fin n, (mask z).getScalar j = true → j ≠ i_star z →
        scores₀ z j ≤ scores₀ z (i_star z) - γ₀) ∧
      (∀ (t : ℕ) (j : Fin n), (mask z).getScalar j = true → j ≠ i_star z →
        g_seq z t j ≤ g_seq z t (i_star z) - Δ_g) ∧
      (∀ j : Fin n, (mask z).getScalar j = true →
        ‖values z j - values z (i_star z)‖ ≤ D_V)) :
    P.IntegrableLoss
      (fun z =>
        ‖rowOutput (scoreTensor (scoreTrajectory η (scores₀ z) (g_seq z) T))
          (mask z) (values z) - label z‖ ^ 2) ∧
    P.risk
      (fun z =>
        ‖rowOutput (scoreTensor (scoreTrajectory η (scores₀ z) (g_seq z) T))
          (mask z) (values z) - label z‖ ^ 2) ≤
      2 * P.risk (fun z => ‖values z (i_star z) - label z‖ ^ 2) +
        2 * (((n : ℝ) - 1) * D_V * Real.exp (-γ₀ - (T : ℝ) * η * Δ_g)) ^ 2 :=
  icl_population_risk_bound P η γ₀ Δ_g D_V scores₀ g_seq mask values i_star label T
    hη hD_nonneg hmeas_icl hint_oracle hae
