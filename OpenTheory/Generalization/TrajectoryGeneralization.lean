import OpenTheory.Dynamics.MLPContractive
import OpenTheory.Generalization.BallCover
import OpenTheory.Generalization.Covering

/-!
# Trajectory-Localized Out-of-Sample Generalization Bounds for Nonlinear MLPs

This module bridges parameter-space training norm envelopes (`‖u‖ ≤ rad`) along
nonlinear batch gradient descent (`MLPBatchTrajectory`, `MLPContractive`) to
out-of-sample population test risk (`Population.risk`) on unseen IID observations
`z ~ P.law`.

1. **Parameter-Space Perturbation Lipschitz Bound (`forward_param_diff_le`, `paramLoss_lipschitz`)**:
   For any two flattened first-layer displacements `u v : Parameters (m * d)` and
   any input `x : Fin d → ℝ` with `inputEnergy x ≤ E`, the scalar network outputs satisfy
   `|(atDisplacement net (unflatten u)).forward x - (atDisplacement net (unflatten v)).forward x|
     ≤ m * A * L * √E * ‖u - v‖`.
   For any `L_ℓ`-Lipschitz loss `ℓ : ℝ → ℝ → ℝ`, the induced observation loss `paramLoss`
   is `(L_ℓ * (m * A * L * √E))`-Lipschitz in the parameter vector `u`.

2. **Explicit Finite Loss Cover of the Trajectory Ball (`trajectoryBall_loss_cover`)**:
   Combining `BallCover.euclideanBall_net` with `paramLoss_lipschitz` yields an explicit
   finite net `models : Finset (Parameters (m * d))` inside `trajectoryBall (m * d) rad`
   of cardinality at most `(coverAxis rad δ (m * d) + 1) ^ (m * d)` that uniformly
   approximates `paramLoss` to within `(L_ℓ * (m * A * L * √E)) * δ`.

3. **Out-of-Sample Population Risk Generalization Bounds
   (`trajectory_uniform_deviation_tail`, `contractive_trajectory_generalization_tail`,
   `approx_erm_trajectory_excess_risk_tail`)**:
   Using `Covering.uniform_deviation_tail` and `Covering.approx_learner_tail`, we bound
   the probability over an IID test/validation sample `s ~ sampleLaw P.law n` that any
   parameter in `trajectoryBall` (and in particular `nonlinearBatchTrajectory` at any
   horizon `T`) deviates from its population risk or violates an oracle excess-risk bound.
-/

noncomputable section

open MeasureTheory StatLean.StatisticalLearning
open scoped BigOperators RealInnerProductSpace

namespace OpenTheory.TrajectoryGeneralization

open OpenTheory.FeatureStep
open OpenTheory.MLPTrajectory
open OpenTheory.MLPFixedDesign
open OpenTheory.MLPBatchBounds
open OpenTheory.MLPResidual
open OpenTheory.MLPBatchTrajectory
open OpenTheory.MLPContractive
open OpenTheory.BallCover
open OpenTheory.Covering

variable {d m : ℕ}

/-- `unflatten` is linear with respect to subtraction. -/
theorem unflatten_sub (u v : Parameters (m * d)) :
    unflatten (u - v) = unflatten u - unflatten v := by
  ext i j
  rfl

/-- Composing two first-layer displacements adds the displacement matrices. -/
theorem atDisplacement_add (net : TwoLayerNet d m) (U V : Matrix (Fin m) (Fin d) ℝ) :
    atDisplacement (atDisplacement net V) U = atDisplacement net (V + U) := by
  cases net
  simp [atDisplacement, add_assoc]

/-- Expressing `atDisplacement net U` as a displacement from `atDisplacement net V`. -/
theorem atDisplacement_sub_cancel (net : TwoLayerNet d m) (U V : Matrix (Fin m) (Fin d) ℝ) :
    atDisplacement (atDisplacement net V) (U - V) = atDisplacement net U := by
  rw [atDisplacement_add, add_sub_cancel]

/-- **Parameter-space perturbation Lipschitz bound (`forward_param_diff_le`)**:
For any two flattened first-layer displacements `u, v : Parameters (m * d)` and any input
`x : Fin d → ℝ` with `inputEnergy x ≤ E`, the network output difference is bounded by
`m * A * L * √E * ‖u - v‖`. -/
theorem forward_param_diff_le (net : TwoLayerNet d m) (x : Fin d → ℝ)
    (u v : Parameters (m * d)) {A L E : ℝ}
    (hA : 0 ≤ A) (hL : 0 ≤ L) (_hE : 0 ≤ E)
    (hW : ∀ i, |net.W₂ 0 i| ≤ A) (hσ : ∀ s, |deriv net.act.σ s| ≤ L)
    (hx : inputEnergy x ≤ E) :
    |(atDisplacement net (unflatten u)).forward x -
      (atDisplacement net (unflatten v)).forward x| ≤
      m * A * L * Real.sqrt E * ‖u - v‖ := by
  have h1 : atDisplacement net (unflatten u) =
      atDisplacement (atDisplacement net (unflatten v)) (unflatten (u - v)) := by
    rw [unflatten_sub, atDisplacement_sub_cancel]
  rw [h1]
  have hshift : ∀ i : Fin m,
      |preShift (unflatten (u - v)) x i| ≤ ‖u - v‖ * Real.sqrt E := by
    intro i
    exact (preShift_unflatten_le_norm (u - v) x i).trans
      (mul_le_mul_of_nonneg_left (Real.sqrt_le_sqrt hx) (norm_nonneg (u - v)))
  have hf := forward_change_le (atDisplacement net (unflatten v)) x (unflatten (u - v))
    hA hL hW hσ hshift
  calc
    |(atDisplacement (atDisplacement net (unflatten v)) (unflatten (u - v))).forward x -
        (atDisplacement net (unflatten v)).forward x|
      ≤ m * A * L * (‖u - v‖ * Real.sqrt E) := hf
    _ = m * A * L * Real.sqrt E * ‖u - v‖ := by ring

/-- Parameter-indexed loss on an observation `z : Z` induced by loss function `ℓ : ℝ → ℝ → ℝ`,
input extractor `xOf : Z → Fin d → ℝ`, and label extractor `yOf : Z → ℝ`. -/
def paramLoss {Z : Type*} (net : TwoLayerNet d m) (xOf : Z → Fin d → ℝ) (yOf : Z → ℝ)
    (ℓ : ℝ → ℝ → ℝ) (u : Parameters (m * d)) (z : Z) : ℝ :=
  ℓ ((atDisplacement net (unflatten u)).forward (xOf z)) (yOf z)

/-- Parameter-to-loss Lipschitz constant `L_ℓ * (m * A * L * √E)`. -/
def paramLossLipConst (m : ℕ) (L_ℓ A L E : ℝ) : ℝ :=
  L_ℓ * (m * A * L * Real.sqrt E)

theorem paramLossLipConst_nonneg (m : ℕ) {L_ℓ A L E : ℝ}
    (hLℓ : 0 ≤ L_ℓ) (hA : 0 ≤ A) (hL : 0 ≤ L) :
    0 ≤ paramLossLipConst m L_ℓ A L E := by
  unfold paramLossLipConst
  positivity

/-- **Parameter-to-loss Lipschitz bound (`paramLoss_lipschitz`)**:
When `ℓ` is `L_ℓ`-Lipschitz in its prediction argument and `inputEnergy (xOf z) ≤ E`,
`paramLoss` is `paramLossLipConst m L_ℓ A L E`-Lipschitz in `u : Parameters (m * d)`. -/
theorem paramLoss_lipschitz {Z : Type*} (net : TwoLayerNet d m)
    (xOf : Z → Fin d → ℝ) (yOf : Z → ℝ) (ℓ : ℝ → ℝ → ℝ)
    {L_ℓ A L E : ℝ} (hLℓ : 0 ≤ L_ℓ) (hA : 0 ≤ A) (hL : 0 ≤ L) (hE : 0 ≤ E)
    (hℓ : ∀ p₁ p₂ y, |ℓ p₁ y - ℓ p₂ y| ≤ L_ℓ * |p₁ - p₂|)
    (hW : ∀ i, |net.W₂ 0 i| ≤ A) (hσ : ∀ s, |deriv net.act.σ s| ≤ L)
    (hx : ∀ z, inputEnergy (xOf z) ≤ E)
    (u v : Parameters (m * d)) (z : Z) :
    |paramLoss net xOf yOf ℓ u z - paramLoss net xOf yOf ℓ v z| ≤
      paramLossLipConst m L_ℓ A L E * ‖u - v‖ := by
  unfold paramLoss paramLossLipConst
  calc
    |ℓ ((atDisplacement net (unflatten u)).forward (xOf z)) (yOf z) -
        ℓ ((atDisplacement net (unflatten v)).forward (xOf z)) (yOf z)|
      ≤ L_ℓ * |(atDisplacement net (unflatten u)).forward (xOf z) -
          (atDisplacement net (unflatten v)).forward (xOf z)| := hℓ _ _ _
    _ ≤ L_ℓ * (m * A * L * Real.sqrt E * ‖u - v‖) :=
        mul_le_mul_of_nonneg_left
          (forward_param_diff_le net (xOf z) u v hA hL hE hW hσ (hx z)) hLℓ
    _ = L_ℓ * (m * A * L * Real.sqrt E) * ‖u - v‖ := by ring

/-- Canonical `[0, 1]`-valued clipped squared loss `min 1 ((pred - y)^2 / B^2)`. -/
def clippedSqLoss (B : ℝ) (pred y : ℝ) : ℝ :=
  min 1 (((pred - y) / B) ^ 2)

/-- `clippedSqLoss B` takes values in `[0, 1]` for all `pred, y : ℝ`. -/
theorem clippedSqLoss_mem_Icc (B : ℝ) (pred y : ℝ) :
    clippedSqLoss B pred y ∈ Set.Icc (0 : ℝ) 1 := by
  unfold clippedSqLoss
  exact ⟨le_min zero_le_one (sq_nonneg _), min_le_left 1 _⟩

/-- Auxiliary algebraic lemma: `min 1 (a^2)` is `2`-Lipschitz on `ℝ`. -/
theorem abs_min_one_sq_sub_le (a b : ℝ) :
    |min (1 : ℝ) (a ^ 2) - min (1 : ℝ) (b ^ 2)| ≤ 2 * |a - b| := by
  have h1 : ∀ u v : ℝ, min (1 : ℝ) (u ^ 2) ≤ min (1 : ℝ) (v ^ 2) + 2 * |u - v| := by
    intro u v
    by_cases hv : |v| ≤ 1
    · by_cases hu : |u| ≤ 1
      · have hmin_u : min (1 : ℝ) (u ^ 2) = u ^ 2 :=
          min_eq_right (by nlinarith [abs_le.mp hu, sq_abs u])
        have hmin_v : min (1 : ℝ) (v ^ 2) = v ^ 2 :=
          min_eq_right (by nlinarith [abs_le.mp hv, sq_abs v])
        rw [hmin_u, hmin_v]
        have hdiff : u ^ 2 - v ^ 2 ≤ |u ^ 2 - v ^ 2| := le_abs_self _
        have hfac : |u ^ 2 - v ^ 2| = |u + v| * |u - v| := by
          rw [← abs_mul]
          ring_nf
        have hsum : |u + v| ≤ 2 := by
          calc
            |u + v| ≤ |u| + |v| := abs_add_le u v
            _ ≤ 1 + 1 := add_le_add hu hv
            _ = 2 := by norm_num
        have hprod : |u + v| * |u - v| ≤ 2 * |u - v| :=
          mul_le_mul_of_nonneg_right hsum (abs_nonneg _)
        linarith
      · have hu' : 1 < |u| := not_le.mp hu
        have hmin_u : min (1 : ℝ) (u ^ 2) = 1 :=
          min_eq_left (by nlinarith [sq_abs u, le_of_lt hu'])
        have hmin_v : min (1 : ℝ) (v ^ 2) = v ^ 2 :=
          min_eq_right (by nlinarith [abs_le.mp hv, sq_abs v])
        rw [hmin_u, hmin_v]
        have huv : |u| - |v| ≤ |u - v| := abs_sub_abs_le_abs_sub u v
        have hv_sq : 1 - v ^ 2 = (1 + |v|) * (1 - |v|) := by
          rw [← sq_abs v]
          ring
        have h1v : 0 ≤ 1 - |v| := by linarith
        have h1v2 : 1 + |v| ≤ 2 := by linarith [abs_nonneg v]
        have hstep1 : (1 + |v|) * (1 - |v|) ≤ 2 * (1 - |v|) :=
          mul_le_mul_of_nonneg_right h1v2 h1v
        have hstep2 : 1 - |v| ≤ |u| - |v| := by linarith
        linarith
    · have hv' : 1 < |v| := not_le.mp hv
      have hmin_v : min (1 : ℝ) (v ^ 2) = 1 :=
        min_eq_left (by nlinarith [sq_abs v, le_of_lt hv'])
      rw [hmin_v]
      have hle1 : min (1 : ℝ) (u ^ 2) ≤ 1 := min_le_left 1 (u ^ 2)
      have hpos : 0 ≤ 2 * |u - v| := mul_nonneg (by norm_num) (abs_nonneg _)
      linarith
  rw [abs_sub_le_iff]
  refine ⟨by linarith [h1 a b], ?_⟩
  have hba := h1 b a
  rw [abs_sub_comm] at hba
  linarith

/-- `clippedSqLoss B` is `(2 / B)`-Lipschitz in the prediction for `B > 0`. -/
theorem clippedSqLoss_lipschitz {B : ℝ} (hB : 0 < B) (p₁ p₂ y : ℝ) :
    |clippedSqLoss B p₁ y - clippedSqLoss B p₂ y| ≤ (2 / B) * |p₁ - p₂| := by
  unfold clippedSqLoss
  have h := abs_min_one_sq_sub_le ((p₁ - y) / B) ((p₂ - y) / B)
  have hdiff : (p₁ - y) / B - (p₂ - y) / B = (p₁ - p₂) / B := by ring
  rw [hdiff, abs_div, abs_of_pos hB] at h
  calc
    |min 1 (((p₁ - y) / B) ^ 2) - min 1 (((p₂ - y) / B) ^ 2)|
      ≤ 2 * (|p₁ - p₂| / B) := h
    _ = (2 / B) * |p₁ - p₂| := by ring

/-- Parameter-Lipschitz corollary for `clippedSqLoss B`. -/
theorem clippedSqLoss_param_lipschitz {Z : Type*} (net : TwoLayerNet d m)
    (xOf : Z → Fin d → ℝ) (yOf : Z → ℝ) {B A L E : ℝ}
    (hB : 0 < B) (hA : 0 ≤ A) (hL : 0 ≤ L) (hE : 0 ≤ E)
    (hW : ∀ i, |net.W₂ 0 i| ≤ A) (hσ : ∀ s, |deriv net.act.σ s| ≤ L)
    (hx : ∀ z, inputEnergy (xOf z) ≤ E)
    (u v : Parameters (m * d)) (z : Z) :
    |paramLoss net xOf yOf (clippedSqLoss B) u z -
      paramLoss net xOf yOf (clippedSqLoss B) v z| ≤
      paramLossLipConst m (2 / B) A L E * ‖u - v‖ :=
  paramLoss_lipschitz net xOf yOf (clippedSqLoss B) (div_nonneg (by norm_num) hB.le)
    hA hL hE (clippedSqLoss_lipschitz hB) hW hσ hx u v z

/-- Closed Euclidean ball of radius `rad` in parameter space `Parameters p`. -/
def trajectoryBall (p : ℕ) (rad : ℝ) : Set (Parameters p) :=
  {u | ‖u‖ ≤ rad}

@[simp]
theorem mem_trajectoryBall {p : ℕ} {rad : ℝ} {u : Parameters p} :
    u ∈ trajectoryBall p rad ↔ ‖u‖ ≤ rad := Iff.rfl

/-- **Explicit finite loss cover for the trajectory ball (`trajectoryBall_loss_cover`)**:
For any radius `rad ≥ 0` and grid resolution `δ > 0`, there is an explicit finite net
`models : Finset (Parameters (m * d))` inside `trajectoryBall (m * d) rad` of size at most
`(coverAxis rad δ (m * d) + 1) ^ (m * d)` that uniformly covers `paramLoss` within
`paramLossLipConst m L_ℓ A L E * δ`. -/
theorem trajectoryBall_loss_cover {Z : Type*} (net : TwoLayerNet d m)
    (xOf : Z → Fin d → ℝ) (yOf : Z → ℝ) (ℓ : ℝ → ℝ → ℝ)
    {L_ℓ A L E rad δ : ℝ}
    (hLℓ : 0 ≤ L_ℓ) (hA : 0 ≤ A) (hL : 0 ≤ L) (hE : 0 ≤ E)
    (hrad : 0 ≤ rad) (hδ : 0 < δ)
    (hℓ : ∀ p₁ p₂ y, |ℓ p₁ y - ℓ p₂ y| ≤ L_ℓ * |p₁ - p₂|)
    (hW : ∀ i, |net.W₂ 0 i| ≤ A) (hσ : ∀ s, |deriv net.act.σ s| ≤ L)
    (hx : ∀ z, inputEnergy (xOf z) ≤ E) :
    ∃ models : Finset (Parameters (m * d)),
      (∀ g ∈ models, g ∈ trajectoryBall (m * d) rad) ∧
      models.card ≤ (coverAxis rad δ (m * d) + 1) ^ (m * d) ∧
      ∀ u ∈ trajectoryBall (m * d) rad, ∃ g ∈ models, ∀ z,
        |paramLoss net xOf yOf ℓ u z - paramLoss net xOf yOf ℓ g z| ≤
          paramLossLipConst m L_ℓ A L E * δ := by
  obtain ⟨models, hmodels, hcard, hnet⟩ := euclideanBall_net (d := m * d) hrad hδ
  refine ⟨models, hmodels, hcard, ?_⟩
  intro u hu
  obtain ⟨g, hg, hdist⟩ := hnet u hu
  refine ⟨g, hg, fun z => ?_⟩
  have hlip := paramLoss_lipschitz net xOf yOf ℓ hLℓ hA hL hE hℓ hW hσ hx u g z
  exact hlip.trans
    (mul_le_mul_of_nonneg_left hdist (paramLossLipConst_nonneg m hLℓ hA hL))

/-- **Out-of-sample uniform risk deviation tail (`trajectory_uniform_deviation_tail`)**:
Over an IID sample `s ~ sampleLaw P.law n`, every parameter vector `u ∈ trajectoryBall (m * d) rad`
simultaneously has empirical risk within `ε + 2 * paramLossLipConst m L_ℓ A L E * δ` of its
population risk `P.risk (paramLoss ... u)` outside an event of probability at most
`2 * (coverAxis rad δ (m * d) + 1) ^ (m * d) * exp (-2 * n * ε ^ 2)`. -/
theorem trajectory_uniform_deviation_tail {Z : Type*} [MeasurableSpace Z]
    (P : Population Z) {n : ℕ} (net : TwoLayerNet d m)
    (xOf : Z → Fin d → ℝ) (yOf : Z → ℝ) (ℓ : ℝ → ℝ → ℝ)
    {L_ℓ A L E rad δ ε : ℝ}
    (hLℓ : 0 ≤ L_ℓ) (hA : 0 ≤ A) (hL : 0 ≤ L) (hE : 0 ≤ E)
    (hrad : 0 ≤ rad) (hδ : 0 < δ) (hn : 1 ≤ n) (hε : 0 < ε)
    (hℓ : ∀ p₁ p₂ y, |ℓ p₁ y - ℓ p₂ y| ≤ L_ℓ * |p₁ - p₂|)
    (hrange : ∀ u ∈ trajectoryBall (m * d) rad, ∀ z,
      paramLoss net xOf yOf ℓ u z ∈ Set.Icc (0 : ℝ) 1)
    (hmeas : ∀ u ∈ trajectoryBall (m * d) rad, Measurable (paramLoss net xOf yOf ℓ u))
    (hW : ∀ i, |net.W₂ 0 i| ≤ A) (hσ : ∀ s, |deriv net.act.σ s| ≤ L)
    (hx : ∀ z, inputEnergy (xOf z) ≤ E) :
    let ρ := paramLossLipConst m L_ℓ A L E * δ
    let N := (coverAxis rad δ (m * d) + 1) ^ (m * d)
    sampleLaw P.law n {s | ∃ u ∈ trajectoryBall (m * d) rad,
      ε + 2 * ρ < |empRisk (paramLoss net xOf yOf ℓ) s u -
        P.risk (paramLoss net xOf yOf ℓ u)|} ≤
      ENNReal.ofReal (2 * (N : ℝ) * Real.exp (-2 * n * ε ^ 2)) := by
  intro ρ N
  obtain ⟨models, hmodels, hcard, hcover⟩ :=
    trajectoryBall_loss_cover net xOf yOf ℓ hLℓ hA hL hE hrad hδ hℓ hW hσ hx
  have htail := uniform_deviation_tail P (trajectoryBall (m * d) rad) models
    (paramLoss net xOf yOf ℓ) hmodels hrange hmeas ρ ε hcover hn hε
  apply htail.trans
  apply ENNReal.ofReal_le_ofReal
  have hcard_real : (models.card : ℝ) ≤ (N : ℝ) := by exact_mod_cast hcard
  have hexp : 0 ≤ Real.exp (-2 * n * ε ^ 2) := Real.exp_nonneg _
  nlinarith

/-- **Contractive trajectory containment (`contractive_trajectory_in_ball`)**:
Under `nonlinearRate m n_train η lam A L E < 1`, for every training horizon `T : ℕ`,
the trained displacement `nonlinearBatchTrajectory net X y η lam T` lies in
`trajectoryBall (m * d) (uniformNormEnvelope m n_train η lam A L E S)`. -/
theorem contractive_trajectory_in_ball {n_train : ℕ}
    (net : TwoLayerNet d m) (X : Fin n_train → Fin d → ℝ)
    (y : Parameters n_train) (η lam : ℝ) {A L E S : ℝ}
    (hη : 0 ≤ η) (hA : 0 ≤ A) (hL : 0 ≤ L) (hE : 0 ≤ E) (hS : 0 ≤ S)
    (hq : nonlinearRate m n_train η lam A L E < 1)
    (hW : ∀ i, |net.W₂ 0 i| ≤ A) (hσ : ∀ v, |deriv net.act.σ v| ≤ L)
    (hX : ∀ r, inputEnergy (X r) ≤ E)
    (hinit : ‖initialOutput net X - y‖ ≤ S) (T : ℕ) :
    nonlinearBatchTrajectory net X y η lam T ∈
      trajectoryBall (m * d) (uniformNormEnvelope m n_train η lam A L E S) :=
  contractive_norm_le net X y η lam hη hA hL hE hS hq hW hσ hX hinit T

/-- **Horizon-free out-of-sample generalization tail along the contractive trajectory
(`contractive_trajectory_generalization_tail`)**:
Under `nonlinearRate m n_train η lam A L E < 1`, the entire infinite training trajectory
`{nonlinearBatchTrajectory net X y η lam T | T : ℕ}` simultaneously enjoys a horizon-free
out-of-sample generalization bound on an IID test sample `s ~ sampleLaw P.law n`. -/
theorem contractive_trajectory_generalization_tail {Z : Type*} [MeasurableSpace Z]
    (P : Population Z) {n n_train : ℕ} (net : TwoLayerNet d m)
    (X : Fin n_train → Fin d → ℝ) (y : Parameters n_train) (η lam : ℝ)
    (xOf : Z → Fin d → ℝ) (yOf : Z → ℝ) (ℓ : ℝ → ℝ → ℝ)
    {L_ℓ A L E S δ ε : ℝ}
    (hη : 0 ≤ η) (hLℓ : 0 ≤ L_ℓ) (hA : 0 ≤ A) (hL : 0 ≤ L) (hE : 0 ≤ E) (hS : 0 ≤ S)
    (hq : nonlinearRate m n_train η lam A L E < 1)
    (hδ : 0 < δ) (hn : 1 ≤ n) (hε : 0 < ε)
    (hℓ : ∀ p₁ p₂ y', |ℓ p₁ y' - ℓ p₂ y'| ≤ L_ℓ * |p₁ - p₂|)
    (hrange : ∀ u ∈ trajectoryBall (m * d) (uniformNormEnvelope m n_train η lam A L E S), ∀ z,
      paramLoss net xOf yOf ℓ u z ∈ Set.Icc (0 : ℝ) 1)
    (hmeas : ∀ u ∈ trajectoryBall (m * d) (uniformNormEnvelope m n_train η lam A L E S),
      Measurable (paramLoss net xOf yOf ℓ u))
    (hW : ∀ i, |net.W₂ 0 i| ≤ A) (hσ : ∀ s, |deriv net.act.σ s| ≤ L)
    (hX : ∀ r, inputEnergy (X r) ≤ E)
    (hx : ∀ z, inputEnergy (xOf z) ≤ E)
    (hinit : ‖initialOutput net X - y‖ ≤ S) :
    let rad := uniformNormEnvelope m n_train η lam A L E S
    let ρ := paramLossLipConst m L_ℓ A L E * δ
    let N := (coverAxis rad δ (m * d) + 1) ^ (m * d)
    sampleLaw P.law n {s | ∃ T : ℕ,
      ε + 2 * ρ < |empRisk (paramLoss net xOf yOf ℓ) s
        (nonlinearBatchTrajectory net X y η lam T) -
        P.risk (paramLoss net xOf yOf ℓ (nonlinearBatchTrajectory net X y η lam T))|} ≤
      ENNReal.ofReal (2 * (N : ℝ) * Real.exp (-2 * n * ε ^ 2)) := by
  intro rad ρ N
  have hrad : 0 ≤ rad := uniformNormEnvelope_nonneg m n_train hη hA hL hE hS hq
  apply (measure_mono ?_).trans
    (trajectory_uniform_deviation_tail P net xOf yOf ℓ hLℓ hA hL hE hrad hδ hn hε
      hℓ hrange hmeas hW hσ hx)
  intro s hs
  obtain ⟨T, hbad⟩ := hs
  exact ⟨nonlinearBatchTrajectory net X y η lam T,
    contractive_trajectory_in_ball net X y η lam hη hA hL hE hS hq hW hσ hX hinit T,
    hbad⟩

/-- **Out-of-sample excess population risk tail for approximate ERM learners in the trajectory ball
(`approx_erm_trajectory_excess_risk_tail`)**:
Any training procedure `learner : (Fin n → Z) → Parameters (m * d)` whose output lies in
`trajectoryBall (m * d) rad` and achieves `η_opt`-approximate empirical risk minimization over
`trajectoryBall (m * d) rad` satisfies an excess population risk tail bound with explicit
covering cardinality `(coverAxis rad δ (m * d) + 1) ^ (m * d)`. -/
theorem approx_erm_trajectory_excess_risk_tail {Z : Type*} [MeasurableSpace Z]
    (P : Population Z) {n : ℕ} (net : TwoLayerNet d m)
    (xOf : Z → Fin d → ℝ) (yOf : Z → ℝ) (ℓ : ℝ → ℝ → ℝ)
    {L_ℓ A L E rad δ ε η_opt : ℝ}
    (hLℓ : 0 ≤ L_ℓ) (hA : 0 ≤ A) (hL : 0 ≤ L) (hE : 0 ≤ E)
    (hrad : 0 ≤ rad) (hδ : 0 < δ) (hn : 1 ≤ n) (hε : 0 < ε)
    (hℓ : ∀ p₁ p₂ y, |ℓ p₁ y - ℓ p₂ y| ≤ L_ℓ * |p₁ - p₂|)
    (hrange : ∀ u ∈ trajectoryBall (m * d) rad, ∀ z,
      paramLoss net xOf yOf ℓ u z ∈ Set.Icc (0 : ℝ) 1)
    (hmeas : ∀ u ∈ trajectoryBall (m * d) rad, Measurable (paramLoss net xOf yOf ℓ u))
    (hW : ∀ i, |net.W₂ 0 i| ≤ A) (hσ : ∀ s, |deriv net.act.σ s| ≤ L)
    (hx : ∀ z, inputEnergy (xOf z) ≤ E)
    (learner : (Fin n → Z) → Parameters (m * d))
    (htrain : ∀ s, StatLean.StatisticalLearning.IsApproxERM
      (trajectoryBall (m * d) rad) (paramLoss net xOf yOf ℓ) s η_opt (learner s)) :
    let ρ := paramLossLipConst m L_ℓ A L E * δ
    let N := (coverAxis rad δ (m * d) + 1) ^ (m * d)
    sampleLaw P.law n {s | ∃ u ∈ trajectoryBall (m * d) rad,
      P.risk (paramLoss net xOf yOf ℓ u) + 2 * ε + 4 * ρ + η_opt <
        P.risk (paramLoss net xOf yOf ℓ (learner s))} ≤
      ENNReal.ofReal (2 * (N : ℝ) * Real.exp (-2 * n * ε ^ 2)) := by
  intro ρ N
  obtain ⟨models, hmodels, hcard, hcover⟩ :=
    trajectoryBall_loss_cover net xOf yOf ℓ hLℓ hA hL hE hrad hδ hℓ hW hσ hx
  have htail := approx_learner_tail P (trajectoryBall (m * d) rad) models
    (paramLoss net xOf yOf ℓ) hmodels hrange hmeas ρ ε η_opt hcover learner htrain hn hε
  apply htail.trans
  apply ENNReal.ofReal_le_ofReal
  have hcard_real : (models.card : ℝ) ≤ (N : ℝ) := by exact_mod_cast hcard
  have hexp : 0 ≤ Real.exp (-2 * n * ε ^ 2) := Real.exp_nonneg _
  nlinarith

end OpenTheory.TrajectoryGeneralization
