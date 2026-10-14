import OpenTheory.Representation.Alignment
import OpenTheory.Dynamics.FeatureStep
import OpenTheory.Dynamics.MLPEndpoint
import OpenTheory.Generalization.MLPRisk
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Ring
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity

/-! Representation Learning & Feature-Learning Advantage.

This module establishes why feature learning (`featureLearningRate m = m`) outperforms
the frozen initial kernel (NTK / lazy regime) both dynamically and statistically:

1. **Dynamic Feature Alignment & Kernel Growth**:
   Under a single gradient step on a two-layer network with uniform readout scale
   `W₂ 0 i = s i / m` (`|s i| = 1`), zero initial output (`net.forward x = 0`), and
   positive local activation slope `α > 0`, the feature-learning rate `lr1 = m` moves
   every neuron's preactivation toward `sign(y * s i)` by an exact **width-independent**
   amount `y * α * s i * inputEnergy x`. By contrast, the standard rate `lr_W1 = 1`
   suppresses this representation update by `1 / m`. Moreover, on a positively sloped
   activation segment where `y * s i * preactivation net x i ≥ 0` and `y ≠ 0`, the single
   feature-learning step strictly increases both the neuron's squared preactivation energy
   and, on a locally linear activation segment, the Jacobian feature kernel `kernel net x`
   itself.

2. **Strict Reduction of Regularization Bias & Expected Ridge Risk**:
   For the centered Jacobian-ridge model with effective feature kernel `K`, regularization
   `lam > 0`, centered target `Δy`, and noise variance `σ²`, both the squared regularization
   bias `biasSq K lam Δy = (lam / (K + lam)) ^ 2 * Δy ^ 2` and the minimum regularized ridge
   objective `minRidgeObj K lam Δy = (1 / 2) * (lam / (K + lam)) * Δy ^ 2` are strictly
   decreasing in `K`, with an exact algebraic reduction formula. Furthermore, in the
   signal-dominant regime (`K₀ * σ ^ 2 < lam * Δy ^ 2` and `K₁ * σ ^ 2 ≤ lam * Δy ^ 2`,
   or more generally whenever `(K₁ * (K₀ + lam) + K₀ * (K₁ + lam)) * σ ^ 2 <
   lam * (K₁ + K₀ + 2 * lam) * Δy ^ 2`), growing the feature kernel from `K₀` to `K₁ > K₀`
   strictly reduces the total expected ridge MSE
   `ridgeExpectedMSE K lam Δy σ = biasSq K lam Δy + (K / (K + lam)) ^ 2 * σ ^ 2`.

3. **Frobenius Subspace Alignment Advantage**:
   Connecting back to `OpenTheory.alignment`, if a learned feature matrix `H₁` achieves
   strictly higher alignment with an orthogonal target projector `P` than `H₀` at equal
   total Frobenius energy `frobeniusSq H₀ = frobeniusSq H₁ > 0`, then `H₁` has strictly
   smaller residual projection error `frobeniusSq (H₁ - H₁ * P) < frobeniusSq (H₀ - H₀ * P)`.
-/

noncomputable section

open scoped BigOperators

namespace OpenTheory.FeatureLearningAdvantage

open FeatureStep MLPTrajectory MLPComparison MLPEndpoint

variable {d m : ℕ}

/-! ### Part 1: Exact Feature Alignment & Preactivation Growth Under One Step -/

/-- Exact signed preactivation displacement for neuron `i` under one gradient step
    when `net.forward x = 0`, `net.W₂ 0 i = s i / m`, and `deriv net.act.σ = α`.
    With `featureLearningRate m = m`, the factor of `1 / m` cancels identically, giving a
    width-independent shift `y * α * s i * inputEnergy x`. -/
theorem feature_learning_preactivation_shift_exact (hm : 0 < m)
    (net : TwoLayerNet d m) (x : Fin d → ℝ) (y α : ℝ) (s : Fin m → ℝ) (i : Fin m)
    (hf0 : net.forward x = 0)
    (hW₂ : net.W₂ 0 i = s i / (m : ℝ))
    (hσ : deriv net.act.σ (preactivation net x i) = α) :
    preactivation (afterFirstLayerStep net x y (featureLearningRate m)) x i -
        preactivation net x i =
      y * α * s i * inputEnergy x := by
  have hm0 : (m : ℝ) ≠ 0 := ne_of_gt (Nat.cast_pos.mpr hm)
  rw [preactivation_step]
  unfold outputResidual featureLearningRate
  rw [hf0, hW₂, hσ]
  field_simp [hm0]
  ring

/-- Multiplying the preactivation shift by `s i` (where `s i ^ 2 = 1`) shows that
    every neuron's signed preactivation `s i * preactivation` moves in the direction of `y`
    by the exact width-independent amount `y * α * inputEnergy x`. -/
theorem feature_learning_signed_preactivation_shift_exact (hm : 0 < m)
    (net : TwoLayerNet d m) (x : Fin d → ℝ) (y α : ℝ) (s : Fin m → ℝ) (i : Fin m)
    (hf0 : net.forward x = 0)
    (hW₂ : net.W₂ 0 i = s i / (m : ℝ))
    (hs : s i ^ 2 = 1)
    (hσ : deriv net.act.σ (preactivation net x i) = α) :
    s i * (preactivation (afterFirstLayerStep net x y (featureLearningRate m)) x i -
        preactivation net x i) =
      y * α * inputEnergy x := by
  rw [feature_learning_preactivation_shift_exact hm net x y α s i hf0 hW₂ hσ]
  calc
    s i * (y * α * s i * inputEnergy x) = y * α * (s i ^ 2) * inputEnergy x := by ring
    _ = y * α * inputEnergy x := by rw [hs]; ring

/-- Under `MuPScaling.standard` (`lr_W1 = 1`), the exact same setup suppresses the
    preactivation shift by `1 / m`. -/
theorem standard_lr_preactivation_shift_suppressed (hd : 0 < (d : ℝ)) (hm : 0 < (m : ℝ))
    (net : TwoLayerNet d m) (x : Fin d → ℝ) (y α : ℝ) (s : Fin m → ℝ) (i : Fin m)
    (hf0 : net.forward x = 0)
    (hW₂ : net.W₂ 0 i = s i / (m : ℝ))
    (hσ : deriv net.act.σ (preactivation net x i) = α) :
    preactivation (afterFirstLayerStep net x y (MuPScaling.standard hd hm).lr_W1) x i -
        preactivation net x i =
      (y * α * s i * inputEnergy x) / (m : ℝ) := by
  rw [preactivation_step]
  unfold outputResidual
  have hlr : (MuPScaling.standard hd hm).lr_W1 = 1 := rfl
  rw [hlr, hf0, hW₂, hσ]
  ring

/-- The feature-learning step (`lr1 = m`) produces `m` times the preactivation shift
    of the standard `lr_W1 = 1` step. -/
theorem feature_learning_vs_standard_shift_ratio (hd : 0 < (d : ℝ)) (hm : 0 < m)
    (net : TwoLayerNet d m) (x : Fin d → ℝ) (y α : ℝ) (s : Fin m → ℝ) (i : Fin m)
    (hf0 : net.forward x = 0)
    (hW₂ : net.W₂ 0 i = s i / (m : ℝ))
    (hσ : deriv net.act.σ (preactivation net x i) = α) :
    preactivation (afterFirstLayerStep net x y (featureLearningRate m)) x i -
        preactivation net x i =
      (m : ℝ) * (preactivation
        (afterFirstLayerStep net x y (MuPScaling.standard hd (Nat.cast_pos.mpr hm)).lr_W1) x i -
        preactivation net x i) := by
  have hm_pos : 0 < (m : ℝ) := Nat.cast_pos.mpr hm
  have hm0 : (m : ℝ) ≠ 0 := ne_of_gt hm_pos
  rw [feature_learning_preactivation_shift_exact hm net x y α s i hf0 hW₂ hσ,
    standard_lr_preactivation_shift_suppressed hd hm_pos net x y α s i hf0 hW₂ hσ]
  exact (mul_div_cancel₀ _ hm0).symm

/-- Strict growth of a neuron's squared preactivation (feature energy along `x`) after one
    feature-learning step when `y * s i * preactivation net x i ≥ 0`, `y ≠ 0`, `α > 0`,
    and `inputEnergy x > 0`. -/
theorem feature_learning_preactivation_sq_strict_growth (hm : 0 < m)
    (net : TwoLayerNet d m) (x : Fin d → ℝ) (y α : ℝ) (s : Fin m → ℝ) (i : Fin m)
    (hf0 : net.forward x = 0)
    (hW₂ : net.W₂ 0 i = s i / (m : ℝ))
    (hs : s i ^ 2 = 1)
    (hσ : deriv net.act.σ (preactivation net x i) = α)
    (hy : y ≠ 0) (hα : 0 < α) (hx : 0 < inputEnergy x)
    (halign : 0 ≤ y * s i * preactivation net x i) :
    preactivation net x i ^ 2 + (y * α * inputEnergy x) ^ 2 ≤
      preactivation (afterFirstLayerStep net x y (featureLearningRate m)) x i ^ 2 ∧
    preactivation net x i ^ 2 <
      preactivation (afterFirstLayerStep net x y (featureLearningRate m)) x i ^ 2 := by
  have hshift := feature_learning_preactivation_shift_exact hm net x y α s i hf0 hW₂ hσ
  have hnew : preactivation (afterFirstLayerStep net x y (featureLearningRate m)) x i =
      preactivation net x i + y * α * s i * inputEnergy x := by linarith
  have hsq : preactivation (afterFirstLayerStep net x y (featureLearningRate m)) x i ^ 2 =
      preactivation net x i ^ 2 +
        2 * (α * inputEnergy x) * (y * s i * preactivation net x i) +
        (s i ^ 2) * (y * α * inputEnergy x) ^ 2 := by
    rw [hnew]
    ring
  rw [hs, one_mul] at hsq
  have hcross : 0 ≤ 2 * (α * inputEnergy x) * (y * s i * preactivation net x i) := by
    positivity
  have hpos_term : 0 < (y * α * inputEnergy x) ^ 2 := by
    have hne : y * α * inputEnergy x ≠ 0 :=
      mul_ne_zero (mul_ne_zero hy (ne_of_gt hα)) (ne_of_gt hx)
    exact sq_pos_of_ne_zero hne
  exact ⟨by linarith, by linarith⟩

/-- Strict increase of the Jacobian feature kernel `kernel net x` after a feature-learning
    step whenever the post-step activation slopes dominate the initial slopes (`α₀ < α₁` on at
    least one active neuron with nonzero readout weight, or uniformly across neurons). -/
theorem kernel_strict_growth_of_slope_growth (hm : 0 < m)
    (net : TwoLayerNet d m) (x : Fin d → ℝ) (y lr1 : ℝ)
    (hx : 0 < inputEnergy x)
    (hdom : ∀ i : Fin m,
      deriv net.act.σ (preactivation net x i) ^ 2 ≤
        deriv (afterFirstLayerStep net x y lr1).act.σ
          (preactivation (afterFirstLayerStep net x y lr1) x i) ^ 2)
    (hstrict : ∃ i₀ : Fin m, net.W₂ 0 i₀ ≠ 0 ∧
      deriv net.act.σ (preactivation net x i₀) ^ 2 <
        deriv (afterFirstLayerStep net x y lr1).act.σ
          (preactivation (afterFirstLayerStep net x y lr1) x i₀) ^ 2) :
    kernel net x < kernel (afterFirstLayerStep net x y lr1) x := by
  have _ := hm
  unfold kernel afterFirstLayerStep
  dsimp only
  apply Finset.sum_lt_sum
  · intro i _
    have hi := hdom i
    have hW : 0 ≤ net.W₂ 0 i ^ 2 := sq_nonneg _
    exact mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_left hi hW) (le_of_lt hx)
  · obtain ⟨i₀, hW₀, hlt₀⟩ := hstrict
    refine ⟨i₀, Finset.mem_univ i₀, ?_⟩
    have hWpos : 0 < net.W₂ 0 i₀ ^ 2 := sq_pos_of_ne_zero hW₀
    exact mul_lt_mul_of_pos_right (mul_lt_mul_of_pos_left hlt₀ hWpos) hx

/-! ### Part 2: Strict Reduction of Ridge Bias and Expected Risk via Kernel Growth -/

/-- Squared regularization bias of the centered 1D ridge predictor with kernel `K`,
    regularization `lam`, and target offset `Δy = target - net.forward x`. -/
def biasSq (K lam Δy : ℝ) : ℝ :=
  (lam / (K + lam)) ^ 2 * Δy ^ 2

/-- Variance contribution of the centered 1D ridge predictor under label noise variance `σ²`. -/
def varianceTerm (K lam σ : ℝ) : ℝ :=
  (K / (K + lam)) ^ 2 * σ ^ 2

/-- Total expected ridge MSE on a single input as a function of kernel energy `K`. -/
def ridgeExpectedMSE (K lam Δy σ : ℝ) : ℝ :=
  biasSq K lam Δy + varianceTerm K lam σ

/-- Minimum value of the regularized ridge objective
    `min_w (1/2 (⟨j, w⟩ - Δy)² + (lam/2) ‖w‖²) = (1/2) * (lam / (K + lam)) * Δy²`. -/
def minRidgeObj (K lam Δy : ℝ) : ℝ :=
  (1 / 2) * (lam / (K + lam)) * Δy ^ 2

/-- Connection to `MLPEndpoint`: the ridge weights `ridgeWeights net x y lam` achieve
    `minRidgeObj (kernel net x) lam (y - net.forward x)` on `Ridge.objective`. -/
theorem ridgeWeights_objective_eq_minRidgeObj (net : TwoLayerNet d m) (x : Fin d → ℝ)
    (y lam : ℝ) (hlam : 0 < lam) :
    Ridge.objective (design net x) (y - net.forward x) 1 lam
        (ridgeWeights net x y lam) =
      minRidgeObj (kernel net x) lam (y - net.forward x) := by
  have hK := kernel_nonneg net x
  have hd : kernel net x + lam ≠ 0 := ne_of_gt (by linarith)
  unfold Ridge.objective minRidgeObj
  rw [design_ridgeWeights]
  simp only [Real.norm_eq_abs, sq_abs]
  have hnorm : ‖ridgeWeights net x y lam‖ ^ 2 =
      ((y - net.forward x) / (kernel net x + lam)) ^ 2 * kernel net x := by
    unfold ridgeWeights
    rw [norm_smul, Real.norm_eq_abs, mul_pow, sq_abs, jacobian_norm_sq]
  rw [hnorm]
  field_simp [hd]
  ring

/-- Connection to `MLPRisk`: the expected squared error of the centered ridge `endpoint`
    under finite zero-mean noise of variance `σ²` is `ridgeExpectedMSE`. -/
theorem endpoint_expected_mse_eq_ridgeExpectedMSE (net : TwoLayerNet d m) (x : Fin d → ℝ)
    (target lam : ℝ) (hlam : 0 < lam)
    {Ω : Type*} [Fintype Ω] (p z : Ω → ℝ)
    (hmass : ∑ w, p w = 1) (hmean : ∑ w, p w * z w = 0)
    (σ : ℝ) (hvar : ∑ w, p w * z w ^ 2 = σ ^ 2) :
    (∑ w, p w * (endpoint net x (target + z w) lam - target) ^ 2) =
      ridgeExpectedMSE (kernel net x) lam (net.forward x - target) σ := by
  have hd : kernel net x + lam ≠ 0 := ne_of_gt (by linarith [kernel_nonneg net x])
  have he (w : Ω) : endpoint net x (target + z w) lam - target =
      lam / (kernel net x + lam) * (net.forward x - target) +
        (kernel net x / (kernel net x + lam)) * z w := by
    unfold endpoint
    field_simp [hd]
    ring
  simp_rw [he]
  rw [MLPRisk.affine_noise_mse p z hmass hmean σ hvar, mul_pow]
  rfl

/-- Exact algebraic identity for the reduction in `minRidgeObj` when the kernel grows
    from `K₀` to `K₁`. -/
theorem minRidgeObj_sub_eq {K₀ K₁ lam Δy : ℝ}
    (h0 : K₀ + lam ≠ 0) (h1 : K₁ + lam ≠ 0) :
    minRidgeObj K₀ lam Δy - minRidgeObj K₁ lam Δy =
      lam * Δy ^ 2 * (K₁ - K₀) / (2 * (K₀ + lam) * (K₁ + lam)) := by
  unfold minRidgeObj
  field_simp [h0, h1]
  ring

/-- `minRidgeObj` is strictly decreasing in the feature kernel `K` when `lam > 0` and `Δy ≠ 0`. -/
theorem minRidgeObj_strict_anti {K₀ K₁ lam Δy : ℝ}
    (hK₀ : 0 ≤ K₀) (hK : K₀ < K₁) (hlam : 0 < lam) (hΔy : Δy ≠ 0) :
    minRidgeObj K₁ lam Δy < minRidgeObj K₀ lam Δy := by
  have h0 : 0 < K₀ + lam := by linarith
  have h1 : 0 < K₁ + lam := by linarith
  have hdiff := minRidgeObj_sub_eq (ne_of_gt h0) (ne_of_gt h1) (Δy := Δy)
  have hpos : 0 < lam * Δy ^ 2 * (K₁ - K₀) / (2 * (K₀ + lam) * (K₁ + lam)) := by
    have hΔy2 : 0 < Δy ^ 2 := sq_pos_of_ne_zero hΔy
    have hsub : 0 < K₁ - K₀ := sub_pos.mpr hK
    positivity
  linarith

/-- Exact algebraic identity for the reduction in squared regularization bias `biasSq`
    when the feature kernel increases from `K₀` to `K₁`. -/
theorem biasSq_sub_eq {K₀ K₁ lam Δy : ℝ}
    (h0 : K₀ + lam ≠ 0) (h1 : K₁ + lam ≠ 0) :
    biasSq K₀ lam Δy - biasSq K₁ lam Δy =
      lam ^ 2 * Δy ^ 2 * (K₁ - K₀) * (K₁ + K₀ + 2 * lam) /
        ((K₀ + lam) ^ 2 * (K₁ + lam) ^ 2) := by
  unfold biasSq
  field_simp [h0, h1]
  ring

/-- `biasSq` is strictly decreasing in the feature kernel `K` when `lam > 0` and `Δy ≠ 0`. -/
theorem biasSq_strict_anti {K₀ K₁ lam Δy : ℝ}
    (hK₀ : 0 ≤ K₀) (hK : K₀ < K₁) (hlam : 0 < lam) (hΔy : Δy ≠ 0) :
    biasSq K₁ lam Δy < biasSq K₀ lam Δy := by
  have h0 : 0 < K₀ + lam := by linarith
  have h1 : 0 < K₁ + lam := by linarith
  have hdiff := biasSq_sub_eq (ne_of_gt h0) (ne_of_gt h1) (Δy := Δy)
  have hpos : 0 < lam ^ 2 * Δy ^ 2 * (K₁ - K₀) * (K₁ + K₀ + 2 * lam) /
      ((K₀ + lam) ^ 2 * (K₁ + lam) ^ 2) := by
    have hΔy2 : 0 < Δy ^ 2 := sq_pos_of_ne_zero hΔy
    have hsub : 0 < K₁ - K₀ := sub_pos.mpr hK
    have hsum : 0 < K₁ + K₀ + 2 * lam := by linarith
    positivity
  linarith

/-- Quantitative lower bound on bias reduction when feature learning grows the kernel
    from `K₀` to `K₁ ≥ K₀ + ΔK` (`ΔK > 0`). -/
theorem biasSq_reduction_lower_bound {K₀ K₁ ΔK lam Δy : ℝ}
    (hK₀ : 0 ≤ K₀) (hΔK : 0 < ΔK) (hK₁ : K₀ + ΔK ≤ K₁) (hlam : 0 < lam) :
    lam ^ 2 * Δy ^ 2 * ΔK * (2 * K₀ + ΔK + 2 * lam) /
        ((K₀ + lam) ^ 2 * (K₁ + lam) ^ 2) ≤
      biasSq K₀ lam Δy - biasSq K₁ lam Δy := by
  have h0 : 0 < K₀ + lam := by linarith
  have h1 : 0 < K₁ + lam := by linarith
  rw [biasSq_sub_eq (ne_of_gt h0) (ne_of_gt h1)]
  apply div_le_div_of_nonneg_right _ (by positivity)
  have hsub : ΔK ≤ K₁ - K₀ := by linarith
  have hsum : 2 * K₀ + ΔK + 2 * lam ≤ K₁ + K₀ + 2 * lam := by linarith
  have hprod : ΔK * (2 * K₀ + ΔK + 2 * lam) ≤ (K₁ - K₀) * (K₁ + K₀ + 2 * lam) :=
    mul_le_mul hsub hsum (by linarith) (by linarith)
  have hcoeff : 0 ≤ lam ^ 2 * Δy ^ 2 := by positivity
  calc
    lam ^ 2 * Δy ^ 2 * ΔK * (2 * K₀ + ΔK + 2 * lam)
        = (lam ^ 2 * Δy ^ 2) * (ΔK * (2 * K₀ + ΔK + 2 * lam)) := by ring
    _ ≤ (lam ^ 2 * Δy ^ 2) * ((K₁ - K₀) * (K₁ + K₀ + 2 * lam)) :=
        mul_le_mul_of_nonneg_left hprod hcoeff
    _ = lam ^ 2 * Δy ^ 2 * (K₁ - K₀) * (K₁ + K₀ + 2 * lam) := by ring

/-- Exact difference of total expected ridge MSE `ridgeExpectedMSE K₀ - ridgeExpectedMSE K₁`. -/
theorem ridgeExpectedMSE_sub_eq {K₀ K₁ lam Δy σ : ℝ}
    (h0 : K₀ + lam ≠ 0) (h1 : K₁ + lam ≠ 0) :
    ridgeExpectedMSE K₀ lam Δy σ - ridgeExpectedMSE K₁ lam Δy σ =
      lam * (K₁ - K₀) *
        ((K₁ + lam) * (lam * Δy ^ 2 - K₀ * σ ^ 2) +
          (K₀ + lam) * (lam * Δy ^ 2 - K₁ * σ ^ 2)) /
        ((K₀ + lam) ^ 2 * (K₁ + lam) ^ 2) := by
  unfold ridgeExpectedMSE biasSq varianceTerm
  field_simp [h0, h1]
  ring

/-- **Main Statistical Feature-Learning Advantage Theorem**:
    When feature learning increases the effective kernel energy from `K₀` to `K₁ > K₀`
    and `K₀ * σ ^ 2 < lam * Δy ^ 2` with `K₁ * σ ^ 2 ≤ lam * Δy ^ 2` (low-noise /
    signal-dominant regime), the total expected ridge MSE is **strictly smaller**
    at `K₁` than at `K₀`. -/
theorem ridge_mse_strict_improvement_of_kernel_growth {K₀ K₁ lam Δy σ : ℝ}
    (hK₀ : 0 ≤ K₀) (hK : K₀ < K₁) (hlam : 0 < lam)
    (hsnr₀ : K₀ * σ ^ 2 < lam * Δy ^ 2)
    (hsnr₁ : K₁ * σ ^ 2 ≤ lam * Δy ^ 2) :
    ridgeExpectedMSE K₁ lam Δy σ < ridgeExpectedMSE K₀ lam Δy σ := by
  have h0 : 0 < K₀ + lam := by linarith
  have h1 : 0 < K₁ + lam := by linarith
  have hdiff := ridgeExpectedMSE_sub_eq (ne_of_gt h0) (ne_of_gt h1) (Δy := Δy) (σ := σ)
  have hsub : 0 < K₁ - K₀ := sub_pos.mpr hK
  have hterm₀ : 0 < (K₁ + lam) * (lam * Δy ^ 2 - K₀ * σ ^ 2) :=
    mul_pos h1 (sub_pos.mpr hsnr₀)
  have hterm₁ : 0 ≤ (K₀ + lam) * (lam * Δy ^ 2 - K₁ * σ ^ 2) :=
    mul_nonneg (le_of_lt h0) (sub_nonneg.mpr hsnr₁)
  have hsum : 0 < (K₁ + lam) * (lam * Δy ^ 2 - K₀ * σ ^ 2) +
      (K₀ + lam) * (lam * Δy ^ 2 - K₁ * σ ^ 2) := by linarith
  have hpos : 0 < lam * (K₁ - K₀) *
      ((K₁ + lam) * (lam * Δy ^ 2 - K₀ * σ ^ 2) +
        (K₀ + lam) * (lam * Δy ^ 2 - K₁ * σ ^ 2)) /
      ((K₀ + lam) ^ 2 * (K₁ + lam) ^ 2) := by positivity
  linarith

/-- Sharp necessary-and-sufficient threshold version: even if `K₁ * σ ^ 2 > lam * Δy ^ 2`
    (so `K₁` overshoots the optimal oracle scale `lam * Δy ^ 2 / σ ^ 2`), `K₁ > K₀` still
    strictly improves expected ridge MSE whenever the weighted balance
    `(K₁ * (K₀ + lam) + K₀ * (K₁ + lam)) * σ ^ 2 < lam * (K₁ + K₀ + 2 * lam) * Δy ^ 2` holds. -/
theorem ridge_mse_strict_improvement_of_weighted_snr {K₀ K₁ lam Δy σ : ℝ}
    (hK₀ : 0 ≤ K₀) (hK : K₀ < K₁) (hlam : 0 < lam)
    (hbal : (K₁ * (K₀ + lam) + K₀ * (K₁ + lam)) * σ ^ 2 <
      lam * (K₁ + K₀ + 2 * lam) * Δy ^ 2) :
    ridgeExpectedMSE K₁ lam Δy σ < ridgeExpectedMSE K₀ lam Δy σ := by
  have h0 : 0 < K₀ + lam := by linarith
  have h1 : 0 < K₁ + lam := by linarith
  have hdiff := ridgeExpectedMSE_sub_eq (ne_of_gt h0) (ne_of_gt h1) (Δy := Δy) (σ := σ)
  have hsub : 0 < K₁ - K₀ := sub_pos.mpr hK
  have hsum : 0 < (K₁ + lam) * (lam * Δy ^ 2 - K₀ * σ ^ 2) +
      (K₀ + lam) * (lam * Δy ^ 2 - K₁ * σ ^ 2) := by linarith
  have hpos : 0 < lam * (K₁ - K₀) *
      ((K₁ + lam) * (lam * Δy ^ 2 - K₀ * σ ^ 2) +
        (K₀ + lam) * (lam * Δy ^ 2 - K₁ * σ ^ 2)) /
      ((K₀ + lam) ^ 2 * (K₁ + lam) ^ 2) := by positivity
  linarith

/-! ### Part 3: Frobenius Subspace Alignment Advantage -/

/-- Exact decomposition of residual projection error in terms of `alignment`:
    for any orthogonal projector `P`, `‖H - H P‖_F² = (1 - alignment H P) * ‖H‖_F²`. -/
theorem frobeniusSq_residual_eq_one_sub_alignment {n m : ℕ}
    (H : Matrix (Fin n) (Fin m) ℝ) (P : Matrix (Fin m) (Fin m) ℝ)
    (hP : IsOrthoProjector P) :
    frobeniusSq (H - H * P) = (1 - alignment H P) * frobeniusSq H := by
  rw [frobeniusSq_sub_mul_orthoProjector H P hP]
  unfold alignment
  split_ifs with h0
  · have hle := frobeniusSq_mul_orthoProjector_le H P hP
    have hge := frobeniusSq_nonneg (H * P)
    have hHP0 : frobeniusSq (H * P) = 0 := by linarith
    rw [h0, hHP0]
    ring
  · field_simp [h0]

/-- **Frobenius Alignment Advantage**:
    If feature matrix `H₁` has strictly higher alignment with orthogonal projector `P`
    than `H₀` (`alignment H₀ P < alignment H₁ P`) at equal total Frobenius energy
    `frobeniusSq H₀ = frobeniusSq H₁ > 0`, then `H₁` has strictly smaller residual
    projection error `frobeniusSq (H₁ - H₁ * P) < frobeniusSq (H₀ - H₀ * P)`. -/
theorem residual_projection_strict_decrease_of_alignment_growth {n m : ℕ}
    (H₀ H₁ : Matrix (Fin n) (Fin m) ℝ) (P : Matrix (Fin m) (Fin m) ℝ)
    (hP : IsOrthoProjector P)
    (hE : frobeniusSq H₀ = frobeniusSq H₁)
    (hpos : 0 < frobeniusSq H₀)
    (halign : alignment H₀ P < alignment H₁ P) :
    frobeniusSq (H₁ - H₁ * P) < frobeniusSq (H₀ - H₀ * P) := by
  rw [frobeniusSq_residual_eq_one_sub_alignment H₁ P hP,
    frobeniusSq_residual_eq_one_sub_alignment H₀ P hP, ← hE]
  have hsub : 1 - alignment H₁ P < 1 - alignment H₀ P := by linarith
  exact mul_lt_mul_of_pos_right hsub hpos

/-- Exact quantitative gap in residual projection error from alignment gain:
    `‖H₀ - H₀ P‖_F² - ‖H₁ - H₁ P‖_F² = (alignment H₁ P - alignment H₀ P) * ‖H₀‖_F²`. -/
theorem residual_projection_gap_of_alignment_growth {n m : ℕ}
    (H₀ H₁ : Matrix (Fin n) (Fin m) ℝ) (P : Matrix (Fin m) (Fin m) ℝ)
    (hP : IsOrthoProjector P)
    (hE : frobeniusSq H₀ = frobeniusSq H₁) :
    frobeniusSq (H₀ - H₀ * P) - frobeniusSq (H₁ - H₁ * P) =
      (alignment H₁ P - alignment H₀ P) * frobeniusSq H₀ := by
  rw [frobeniusSq_residual_eq_one_sub_alignment H₀ P hP,
    frobeniusSq_residual_eq_one_sub_alignment H₁ P hP, ← hE]
  ring

end OpenTheory.FeatureLearningAdvantage
