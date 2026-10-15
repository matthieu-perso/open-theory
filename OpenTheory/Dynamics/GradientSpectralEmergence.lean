import OpenTheory.Data.Spectrum
import OpenTheory.Representation.Alignment
import OpenTheory.Dynamics.MultiStepSpectralBias
import OpenTheory.Generalization.SpikedSampleComplexity
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Data.Matrix.Basic
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Ring

/-! # First-Principles Spectral Feature Emergence from Raw Weight Gradients

This module derives the discrete-time power-iteration law, modal subspace energy bounds,
signal-to-bulk amplification, subspace alignment convergence, and spiked effective dimension
collapse directly from the raw first-layer weight gradient update driven by the empirical
data Gram matrix, discharging the energy-growth hypotheses of `MultiStepSpectralBias` and
`SpikedSampleComplexity` from first principles:

1. **Raw First-Layer Weight Gradient Update Driven by the Empirical Data Gram Matrix**:
   Defines the Euclidean inner product `vecDot`, the Gram matrix-vector action `gramAct`,
   the one-step raw weight gradient update `rawWeightGradStep η c_σ G w`, and the `T`-step
   raw weight trajectory `rawWeightTrajectory η c_σ G w₀ T`.

2. **Exact Eigenmode Power-Iteration Law from Raw Weight Gradients**:
   Proves self-adjointness of `gramAct` under `IsSymmetricGram G` (`vecDot_gramAct_symm`),
   eigenmode projection scaling (`vecDot_gramAct_eigenpair`), the exact one-step recurrence
   (`rawWeightGradStep_eigenmode_step`), and the `T`-step closed-form power-iteration law
   (`rawWeightTrajectory_eigenmode_eq`), matching `MultiStepSpectralBias.modeAmpDisc`.

3. **Multi-Mode Subspace Energy Evolution Derived from Raw Weight Updates**:
   Defines `modalSubspaceEnergy` across a family of eigenmodes and proves its exact `T`-step
   expansion (`modalSubspaceEnergy_trajectory_eq`), lower bounds on the signal subspace energy
   (`signal_subspace_energy_lower_bound_from_gradients`), and upper bounds on the bulk subspace
   energy (`bulk_subspace_energy_upper_bound_from_gradients`).

4. **First-Principles Discharge of `MultiStepSpectralBias` & `SpikedSampleComplexity`**:
   Combines the derived modal energy bounds with `MultiStepSpectralBias` and
   `SpikedSampleComplexity` to establish `first_principles_gradient_signal_to_bulk_ratio`,
   `first_principles_gradient_alignment_convergence`, and
   `first_principles_gradient_effective_dim_bound` without external energy growth assumptions.
-/

noncomputable section

namespace OpenTheory.GradientSpectralEmergence

open OpenTheory
open OpenTheory.MultiStepSpectralBias
open OpenTheory.SpikedSampleComplexity

/-! ### Part 1: Raw First-Layer Weight Gradient Update Driven by the Empirical Data Gram Matrix -/

/-- Euclidean inner product of two vectors in `ℝ^d`:
    `vecDot u v = ∑ j : Fin d, u j * v j`. -/
def vecDot {d : ℕ} (u v : Fin d → ℝ) : ℝ :=
  ∑ j : Fin d, u j * v j

/-- Commutativity of the Euclidean inner product `vecDot`. -/
lemma vecDot_comm {d : ℕ} (u v : Fin d → ℝ) : vecDot u v = vecDot v u := by
  unfold vecDot
  refine Finset.sum_congr rfl fun j _ => ?_
  ring

/-- Linearity of `vecDot` in the first argument under vector addition. -/
lemma vecDot_add_left {d : ℕ} (u₁ u₂ v : Fin d → ℝ) :
    vecDot (fun i => u₁ i + u₂ i) v = vecDot u₁ v + vecDot u₂ v := by
  unfold vecDot
  rw [← Finset.sum_add_distrib]
  refine Finset.sum_congr rfl fun j _ => ?_
  ring

/-- Linearity of `vecDot` in the first argument under scalar multiplication. -/
lemma vecDot_smul_left {d : ℕ} (c : ℝ) (u v : Fin d → ℝ) :
    vecDot (fun i => c * u i) v = c * vecDot u v := by
  unfold vecDot
  rw [Finset.mul_sum]
  refine Finset.sum_congr rfl fun j _ => ?_
  ring

/-- Nonnegativity of self inner product `vecDot u u`. -/
lemma vecDot_self_nonneg {d : ℕ} (u : Fin d → ℝ) : 0 ≤ vecDot u u := by
  unfold vecDot
  refine Finset.sum_nonneg fun j _ => ?_
  have hsq : u j * u j = (u j) ^ 2 := (sq (u j)).symm
  rw [hsq]
  exact sq_nonneg (u j)

/-- Action of the empirical data Gram / covariance matrix `G : Matrix (Fin d) (Fin d) ℝ`
    on a weight vector `w : Fin d → ℝ`:
    `(gramAct G w)_i = ∑ j : Fin d, G i j * w j`. -/
def gramAct {d : ℕ} (G : Matrix (Fin d) (Fin d) ℝ) (w : Fin d → ℝ) : Fin d → ℝ :=
  fun i => ∑ j : Fin d, G i j * w j

/-- One-step raw first-layer weight gradient update with learning rate `η` and activation
    slope `c_σ` at the origin, driven by the empirical Gram matrix `G`:
    `w_{t+1} = w_t + η * c_σ * G w_t`. -/
def rawWeightGradStep {d : ℕ} (η c_σ : ℝ) (G : Matrix (Fin d) (Fin d) ℝ)
    (w : Fin d → ℝ) : Fin d → ℝ :=
  fun i => w i + η * c_σ * gramAct G w i

/-- `T`-step raw weight trajectory starting from initialization `w₀ : Fin d → ℝ` under
    repeated application of `rawWeightGradStep η c_σ G`. -/
def rawWeightTrajectory {d : ℕ} (η c_σ : ℝ) (G : Matrix (Fin d) (Fin d) ℝ)
    (w₀ : Fin d → ℝ) : ℕ → (Fin d → ℝ)
  | 0 => w₀
  | t + 1 => rawWeightGradStep η c_σ G (rawWeightTrajectory η c_σ G w₀ t)

@[simp]
lemma rawWeightTrajectory_zero {d : ℕ} (η c_σ : ℝ) (G : Matrix (Fin d) (Fin d) ℝ)
    (w₀ : Fin d → ℝ) :
    rawWeightTrajectory η c_σ G w₀ 0 = w₀ :=
  rfl

@[simp]
lemma rawWeightTrajectory_succ {d : ℕ} (η c_σ : ℝ) (G : Matrix (Fin d) (Fin d) ℝ)
    (w₀ : Fin d → ℝ) (t : ℕ) :
    rawWeightTrajectory η c_σ G w₀ (t + 1) =
      rawWeightGradStep η c_σ G (rawWeightTrajectory η c_σ G w₀ t) :=
  rfl

/-! ### Part 2: Exact Eigenmode Power-Iteration Law from Raw Weight Gradients -/

/-- Symmetry predicate for an empirical data Gram / covariance matrix `G`. -/
def IsSymmetricGram {d : ℕ} (G : Matrix (Fin d) (Fin d) ℝ) : Prop :=
  ∀ i j : Fin d, G i j = G j i

/-- Eigenpair predicate for the Gram matrix action: `G v = γ v` entrywise. -/
def IsGramEigenpair {d : ℕ} (G : Matrix (Fin d) (Fin d) ℝ)
    (v : Fin d → ℝ) (γ : ℝ) : Prop :=
  ∀ i : Fin d, gramAct G v i = γ * v i

/-- **Self-Adjointness of `gramAct` (`vecDot_gramAct_symm`)**:
    If `G` is a symmetric Gram matrix (`IsSymmetricGram G`), then `gramAct G` is self-adjoint
    with respect to `vecDot`: `⟨G w, v⟩ = ⟨w, G v⟩`. -/
theorem vecDot_gramAct_symm {d : ℕ} {G : Matrix (Fin d) (Fin d) ℝ}
    (hG : IsSymmetricGram G) (w v : Fin d → ℝ) :
    vecDot (gramAct G w) v = vecDot w (gramAct G v) := by
  unfold vecDot gramAct
  calc
    ∑ i : Fin d, (∑ j : Fin d, G i j * w j) * v i
        = ∑ i : Fin d, ∑ j : Fin d, G i j * w j * v i := by
          refine Finset.sum_congr rfl fun i _ => ?_
          rw [Finset.sum_mul]
    _ = ∑ j : Fin d, ∑ i : Fin d, G i j * w j * v i := Finset.sum_comm
    _ = ∑ j : Fin d, ∑ i : Fin d, w j * (G j i * v i) := by
          refine Finset.sum_congr rfl fun j _ => Finset.sum_congr rfl fun i _ => ?_
          rw [hG i j]
          ring
    _ = ∑ j : Fin d, w j * (∑ i : Fin d, G j i * v i) := by
          refine Finset.sum_congr rfl fun j _ => ?_
          rw [← Finset.mul_sum]

/-- **Eigenpair Projection Law for `gramAct` (`vecDot_gramAct_eigenpair`)**:
    If `IsSymmetricGram G` and `IsGramEigenpair G v γ`, then for any weight vector `w`,
    `⟨G w, v⟩ = γ ⟨w, v⟩`. -/
theorem vecDot_gramAct_eigenpair {d : ℕ} {G : Matrix (Fin d) (Fin d) ℝ}
    {v : Fin d → ℝ} {γ : ℝ}
    (hG : IsSymmetricGram G) (hev : IsGramEigenpair G v γ) (w : Fin d → ℝ) :
    vecDot (gramAct G w) v = γ * vecDot w v := by
  rw [vecDot_gramAct_symm hG w v]
  unfold vecDot
  rw [Finset.mul_sum]
  refine Finset.sum_congr rfl fun j _ => ?_
  rw [hev j]
  ring

/-- **One-Step Eigenmode Amplification from Raw Weight Gradients (`rawWeightGradStep_eigenmode_step`)**:
    If `IsSymmetricGram G` and `IsGramEigenpair G v γ`, then one step of raw first-layer
    weight gradient update scales the projection of `w` onto eigenmode `v` by `(1 + η * c_σ * γ)`. -/
theorem rawWeightGradStep_eigenmode_step {d : ℕ} (η c_σ : ℝ)
    {G : Matrix (Fin d) (Fin d) ℝ} {v : Fin d → ℝ} {γ : ℝ}
    (hG : IsSymmetricGram G) (hev : IsGramEigenpair G v γ) (w : Fin d → ℝ) :
    vecDot (rawWeightGradStep η c_σ G w) v = (1 + η * c_σ * γ) * vecDot w v := by
  unfold rawWeightGradStep
  rw [vecDot_add_left, vecDot_smul_left, vecDot_gramAct_eigenpair hG hev w]
  ring

/-- **`T`-Step Eigenmode Power-Iteration Law (`rawWeightTrajectory_eigenmode_eq`)**:
    For all `T : ℕ`, if `IsSymmetricGram G` and `IsGramEigenpair G v γ`, then along the raw
    weight trajectory `rawWeightTrajectory η c_σ G w₀`:
    1. `vecDot (rawWeightTrajectory η c_σ G w₀ T) v = modeAmpDisc η (c_σ * γ) T (vecDot w₀ v)`,
    2. `vecDot (rawWeightTrajectory η c_σ G w₀ T) v = (1 + η * c_σ * γ) ^ T * vecDot w₀ v`,
    3. `(vecDot (rawWeightTrajectory η c_σ G w₀ T) v) ^ 2 = (1 + η * c_σ * γ) ^ (2 * T) * (vecDot w₀ v) ^ 2`. -/
theorem rawWeightTrajectory_eigenmode_eq {d : ℕ} (η c_σ : ℝ)
    {G : Matrix (Fin d) (Fin d) ℝ} {v : Fin d → ℝ} {γ : ℝ}
    (hG : IsSymmetricGram G) (hev : IsGramEigenpair G v γ)
    (w₀ : Fin d → ℝ) (T : ℕ) :
    vecDot (rawWeightTrajectory η c_σ G w₀ T) v = modeAmpDisc η (c_σ * γ) T (vecDot w₀ v) ∧
    vecDot (rawWeightTrajectory η c_σ G w₀ T) v = (1 + η * c_σ * γ) ^ T * vecDot w₀ v ∧
    (vecDot (rawWeightTrajectory η c_σ G w₀ T) v) ^ 2 =
      (1 + η * c_σ * γ) ^ (2 * T) * (vecDot w₀ v) ^ 2 := by
  have hpow : vecDot (rawWeightTrajectory η c_σ G w₀ T) v =
      (1 + η * c_σ * γ) ^ T * vecDot w₀ v := by
    induction T with
    | zero =>
      simp [rawWeightTrajectory]
    | succ t ih =>
      rw [rawWeightTrajectory_succ, rawWeightGradStep_eigenmode_step η c_σ hG hev, ih, pow_succ']
      ring
  have hmode : vecDot (rawWeightTrajectory η c_σ G w₀ T) v =
      modeAmpDisc η (c_σ * γ) T (vecDot w₀ v) := by
    unfold modeAmpDisc
    rw [hpow]
    ring
  have hsq : (vecDot (rawWeightTrajectory η c_σ G w₀ T) v) ^ 2 =
      (1 + η * c_σ * γ) ^ (2 * T) * (vecDot w₀ v) ^ 2 := by
    rw [hpow, mul_pow, ← pow_mul, Nat.mul_comm T 2]
  exact ⟨hmode, hpow, hsq⟩

/-! ### Part 3: Multi-Mode Subspace Energy Evolution Derived from Raw Weight Updates -/

/-- Modal subspace energy of a weight vector `w : Fin d → ℝ` across a family of modes
    `v_fam : Fin k → (Fin d → ℝ)`:
    `modalSubspaceEnergy v_fam w = ∑ l : Fin k, (vecDot w (v_fam l)) ^ 2`. -/
def modalSubspaceEnergy {d k : ℕ} (v_fam : Fin k → (Fin d → ℝ)) (w : Fin d → ℝ) : ℝ :=
  ∑ l : Fin k, (vecDot w (v_fam l)) ^ 2

/-- Nonnegativity of `modalSubspaceEnergy`. -/
theorem modalSubspaceEnergy_nonneg {d k : ℕ} (v_fam : Fin k → (Fin d → ℝ)) (w : Fin d → ℝ) :
    0 ≤ modalSubspaceEnergy v_fam w := by
  unfold modalSubspaceEnergy
  exact Finset.sum_nonneg fun l _ => sq_nonneg _

/-- **Exact `T`-Step Modal Subspace Energy Evolution (`modalSubspaceEnergy_trajectory_eq`)**:
    If `IsSymmetricGram G` and every mode `v_fam l` is an eigenvector of `G` with eigenvalue
    `γ_fam l` (`∀ l, IsGramEigenpair G (v_fam l) (γ_fam l)`), then after `T` steps of raw weight
    gradient descent:
    `modalSubspaceEnergy v_fam (rawWeightTrajectory η c_σ G w₀ T) = ∑ l : Fin k, (1 + η * c_σ * γ_fam l) ^ (2 * T) * (vecDot w₀ (v_fam l)) ^ 2`. -/
theorem modalSubspaceEnergy_trajectory_eq {d k : ℕ} (η c_σ : ℝ)
    {G : Matrix (Fin d) (Fin d) ℝ}
    {v_fam : Fin k → (Fin d → ℝ)} {γ_fam : Fin k → ℝ}
    (hG : IsSymmetricGram G) (hev : ∀ l : Fin k, IsGramEigenpair G (v_fam l) (γ_fam l))
    (w₀ : Fin d → ℝ) (T : ℕ) :
    modalSubspaceEnergy v_fam (rawWeightTrajectory η c_σ G w₀ T) =
      ∑ l : Fin k, (1 + η * c_σ * γ_fam l) ^ (2 * T) * (vecDot w₀ (v_fam l)) ^ 2 := by
  unfold modalSubspaceEnergy
  refine Finset.sum_congr rfl fun l _ => ?_
  exact (rawWeightTrajectory_eigenmode_eq η c_σ hG (hev l) w₀ T).2.2

/-- **Signal Subspace Energy Lower Bound from Raw Weight Gradients
    (`signal_subspace_energy_lower_bound_from_gradients`)**:
    If `IsSymmetricGram G`, every signal mode `l : Fin r` is an eigenpair with eigenvalue
    `γ_sig l ≥ sg.leading`, and `0 ≤ η`, `0 ≤ c_σ`, then the `T`-step signal subspace energy
    satisfies:
    `(1 + (η * c_σ) * sg.leading) ^ (2 * T) * modalSubspaceEnergy v_sig w₀ ≤ modalSubspaceEnergy v_sig (rawWeightTrajectory η c_σ G w₀ T)`. -/
theorem signal_subspace_energy_lower_bound_from_gradients {d r : ℕ}
    (sg : SpectralGap) {η c_σ : ℝ} (hη : 0 ≤ η) (hc_σ : 0 ≤ c_σ)
    {G : Matrix (Fin d) (Fin d) ℝ} (hG : IsSymmetricGram G)
    {v_sig : Fin r → (Fin d → ℝ)} {γ_sig : Fin r → ℝ}
    (hev_sig : ∀ l : Fin r, IsGramEigenpair G (v_sig l) (γ_sig l))
    (hγ_sig : ∀ l : Fin r, sg.leading ≤ γ_sig l)
    (w₀ : Fin d → ℝ) (T : ℕ) :
    (1 + (η * c_σ) * sg.leading) ^ (2 * T) * modalSubspaceEnergy v_sig w₀ ≤
      modalSubspaceEnergy v_sig (rawWeightTrajectory η c_σ G w₀ T) := by
  rw [modalSubspaceEnergy_trajectory_eq η c_σ hG hev_sig w₀ T]
  unfold modalSubspaceEnergy
  rw [Finset.mul_sum]
  refine Finset.sum_le_sum fun l _ => ?_
  have hlead_pos : 0 < sg.leading := leading_pos_of_spectralGap sg
  have hηc : 0 ≤ η * c_σ := mul_nonneg hη hc_σ
  have hbase_nonneg : 0 ≤ 1 + (η * c_σ) * sg.leading := by positivity
  have hbase_le : 1 + (η * c_σ) * sg.leading ≤ 1 + η * c_σ * γ_sig l := by
    have := mul_le_mul_of_nonneg_left (hγ_sig l) hηc
    linarith
  have hpow_le : (1 + (η * c_σ) * sg.leading) ^ (2 * T) ≤
      (1 + η * c_σ * γ_sig l) ^ (2 * T) :=
    pow_le_pow_left₀ hbase_nonneg hbase_le (2 * T)
  exact mul_le_mul_of_nonneg_right hpow_le (sq_nonneg _)

/-- **Bulk Subspace Energy Upper Bound from Raw Weight Gradients
    (`bulk_subspace_energy_upper_bound_from_gradients`)**:
    If `IsSymmetricGram G`, every bulk mode `l : Fin d_bulk` is an eigenpair with eigenvalue
    `0 ≤ γ_bulk l ≤ sg.bulk`, and `0 ≤ η`, `0 ≤ c_σ`, then the `T`-step bulk subspace energy
    satisfies:
    `modalSubspaceEnergy v_bulk (rawWeightTrajectory η c_σ G w₀ T) ≤ (1 + (η * c_σ) * sg.bulk) ^ (2 * T) * modalSubspaceEnergy v_bulk w₀`. -/
theorem bulk_subspace_energy_upper_bound_from_gradients {d d_bulk : ℕ}
    (sg : SpectralGap) {η c_σ : ℝ} (hη : 0 ≤ η) (hc_σ : 0 ≤ c_σ)
    {G : Matrix (Fin d) (Fin d) ℝ} (hG : IsSymmetricGram G)
    {v_bulk : Fin d_bulk → (Fin d → ℝ)} {γ_bulk : Fin d_bulk → ℝ}
    (hev_bulk : ∀ l : Fin d_bulk, IsGramEigenpair G (v_bulk l) (γ_bulk l))
    (hγ_bulk_nonneg : ∀ l : Fin d_bulk, 0 ≤ γ_bulk l)
    (hγ_bulk_le : ∀ l : Fin d_bulk, γ_bulk l ≤ sg.bulk)
    (w₀ : Fin d → ℝ) (T : ℕ) :
    modalSubspaceEnergy v_bulk (rawWeightTrajectory η c_σ G w₀ T) ≤
      (1 + (η * c_σ) * sg.bulk) ^ (2 * T) * modalSubspaceEnergy v_bulk w₀ := by
  rw [modalSubspaceEnergy_trajectory_eq η c_σ hG hev_bulk w₀ T]
  unfold modalSubspaceEnergy
  rw [Finset.mul_sum]
  refine Finset.sum_le_sum fun l _ => ?_
  have hηc : 0 ≤ η * c_σ := mul_nonneg hη hc_σ
  have hγ_l : 0 ≤ γ_bulk l := hγ_bulk_nonneg l
  have hbase_nonneg : 0 ≤ 1 + η * c_σ * γ_bulk l := by positivity
  have hbase_le : 1 + η * c_σ * γ_bulk l ≤ 1 + (η * c_σ) * sg.bulk := by
    have := mul_le_mul_of_nonneg_left (hγ_bulk_le l) hηc
    linarith
  have hpow_le : (1 + η * c_σ * γ_bulk l) ^ (2 * T) ≤
      (1 + (η * c_σ) * sg.bulk) ^ (2 * T) :=
    pow_le_pow_left₀ hbase_nonneg hbase_le (2 * T)
  exact mul_le_mul_of_nonneg_right hpow_le (sq_nonneg _)

/-! ### Part 4: First-Principles Discharge of `MultiStepSpectralBias` & `SpikedSampleComplexity` -/

/-- **First-Principles Signal-to-Bulk Ratio Decay from Raw Weight Gradients
    (`first_principles_gradient_signal_to_bulk_ratio`)**:
    Along the actual raw weight gradient trajectory `w_T = rawWeightTrajectory η c_σ G w₀ T`
    with `0 < η`, `0 < c_σ`, and `0 < modalSubspaceEnergy v_sig w₀`, the bulk-to-signal
    subspace energy ratio decays geometrically at rate `((1 + (η * c_σ) * sg.bulk) / (1 + (η * c_σ) * sg.leading)) ^ (2 * T)`
    without any external energy-growth assumptions. -/
theorem first_principles_gradient_signal_to_bulk_ratio {d r d_bulk : ℕ}
    (sg : SpectralGap) {η c_σ : ℝ} (hη : 0 < η) (hc_σ : 0 < c_σ)
    {G : Matrix (Fin d) (Fin d) ℝ} (hG : IsSymmetricGram G)
    {v_sig : Fin r → (Fin d → ℝ)} {γ_sig : Fin r → ℝ}
    (hev_sig : ∀ l : Fin r, IsGramEigenpair G (v_sig l) (γ_sig l))
    (hγ_sig : ∀ l : Fin r, sg.leading ≤ γ_sig l)
    {v_bulk : Fin d_bulk → (Fin d → ℝ)} {γ_bulk : Fin d_bulk → ℝ}
    (hev_bulk : ∀ l : Fin d_bulk, IsGramEigenpair G (v_bulk l) (γ_bulk l))
    (hγ_bulk_nonneg : ∀ l : Fin d_bulk, 0 ≤ γ_bulk l)
    (hγ_bulk_le : ∀ l : Fin d_bulk, γ_bulk l ≤ sg.bulk)
    (w₀ : Fin d → ℝ) (hS₀ : 0 < modalSubspaceEnergy v_sig w₀) (T : ℕ) :
    modalSubspaceEnergy v_bulk (rawWeightTrajectory η c_σ G w₀ T) /
        modalSubspaceEnergy v_sig (rawWeightTrajectory η c_σ G w₀ T) ≤
      (modalSubspaceEnergy v_bulk w₀ / modalSubspaceEnergy v_sig w₀) *
        ((1 + (η * c_σ) * sg.bulk) / (1 + (η * c_σ) * sg.leading)) ^ (2 * T) := by
  have hS := signal_subspace_energy_lower_bound_from_gradients
    sg (le_of_lt hη) (le_of_lt hc_σ) hG hev_sig hγ_sig w₀ T
  have hB := bulk_subspace_energy_upper_bound_from_gradients
    sg (le_of_lt hη) (le_of_lt hc_σ) hG hev_bulk hγ_bulk_nonneg hγ_bulk_le w₀ T
  have hB₀ : 0 ≤ modalSubspaceEnergy v_bulk w₀ := modalSubspaceEnergy_nonneg v_bulk w₀
  exact bulk_div_signal_disc_le sg T (mul_pos hη hc_σ) hS₀ hB₀ hS hB

/-- **First-Principles Subspace Alignment Convergence from Raw Weight Gradients
    (`first_principles_gradient_alignment_convergence`)**:
    If at step `T` the feature matrix `H_T` has signal projection energy
    `frobeniusSq (H_T * P) = modalSubspaceEnergy v_sig (rawWeightTrajectory η c_σ G w₀ T)`
    and residual bulk energy
    `frobeniusSq (H_T - H_T * P) = modalSubspaceEnergy v_bulk (rawWeightTrajectory η c_σ G w₀ T)`,
    then directly from the raw weight gradient updates:
    1. `alignment H_T P ≥ 1 - (B₀ / S₀) * ((1 + (η * c_σ) * sg.bulk) / (1 + (η * c_σ) * sg.leading)) ^ (2 * T)`,
    2. `frobeniusSq (H_T - H_T * P) ≤ ((B₀ / S₀) * ((1 + (η * c_σ) * sg.bulk) / (1 + (η * c_σ) * sg.leading)) ^ (2 * T)) * frobeniusSq H_T`,
    3. For any ridge parameter `lam > 0`, the spiked effective dimension satisfies
       `spikedEffectiveDim (frobeniusSq (H_T * P)) (frobeniusSq (H_T - H_T * P)) r d_bulk lam ≤ (r : ℝ) + (d_bulk : ℝ) * ((B₀ / S₀) * ((1 + (η * c_σ) * sg.bulk) / (1 + (η * c_σ) * sg.leading)) ^ (2 * T)) * (frobeniusSq H_T / lam)`. -/
theorem first_principles_gradient_alignment_convergence {d n m r d_bulk : ℕ}
    (sg : SpectralGap) {η c_σ : ℝ} (hη : 0 < η) (hc_σ : 0 < c_σ)
    {G : Matrix (Fin d) (Fin d) ℝ} (hG : IsSymmetricGram G)
    {v_sig : Fin r → (Fin d → ℝ)} {γ_sig : Fin r → ℝ}
    (hev_sig : ∀ l : Fin r, IsGramEigenpair G (v_sig l) (γ_sig l))
    (hγ_sig : ∀ l : Fin r, sg.leading ≤ γ_sig l)
    {v_bulk : Fin d_bulk → (Fin d → ℝ)} {γ_bulk : Fin d_bulk → ℝ}
    (hev_bulk : ∀ l : Fin d_bulk, IsGramEigenpair G (v_bulk l) (γ_bulk l))
    (hγ_bulk_nonneg : ∀ l : Fin d_bulk, 0 ≤ γ_bulk l)
    (hγ_bulk_le : ∀ l : Fin d_bulk, γ_bulk l ≤ sg.bulk)
    (w₀ : Fin d → ℝ) (hS₀ : 0 < modalSubspaceEnergy v_sig w₀) (T : ℕ)
    (H_T : Matrix (Fin n) (Fin m) ℝ) (P : Matrix (Fin m) (Fin m) ℝ)
    (hP : IsOrthoProjector P)
    (hH_sig : frobeniusSq (H_T * P) =
      modalSubspaceEnergy v_sig (rawWeightTrajectory η c_σ G w₀ T))
    (hH_bulk : frobeniusSq (H_T - H_T * P) =
      modalSubspaceEnergy v_bulk (rawWeightTrajectory η c_σ G w₀ T)) :
    1 - (modalSubspaceEnergy v_bulk w₀ / modalSubspaceEnergy v_sig w₀) *
        ((1 + (η * c_σ) * sg.bulk) / (1 + (η * c_σ) * sg.leading)) ^ (2 * T) ≤
      alignment H_T P ∧
    frobeniusSq (H_T - H_T * P) ≤
      ((modalSubspaceEnergy v_bulk w₀ / modalSubspaceEnergy v_sig w₀) *
        ((1 + (η * c_σ) * sg.bulk) / (1 + (η * c_σ) * sg.leading)) ^ (2 * T)) *
        frobeniusSq H_T ∧
    ∀ {lam : ℝ}, 0 < lam →
      spikedEffectiveDim (frobeniusSq (H_T * P)) (frobeniusSq (H_T - H_T * P)) r d_bulk lam ≤
        (r : ℝ) + (d_bulk : ℝ) *
          ((modalSubspaceEnergy v_bulk w₀ / modalSubspaceEnergy v_sig w₀) *
            ((1 + (η * c_σ) * sg.bulk) / (1 + (η * c_σ) * sg.leading)) ^ (2 * T)) *
          (frobeniusSq H_T / lam) := by
  have hS_grad := signal_subspace_energy_lower_bound_from_gradients
    sg (le_of_lt hη) (le_of_lt hc_σ) hG hev_sig hγ_sig w₀ T
  have hB_grad := bulk_subspace_energy_upper_bound_from_gradients
    sg (le_of_lt hη) (le_of_lt hc_σ) hG hev_bulk hγ_bulk_nonneg hγ_bulk_le w₀ T
  have hS : (1 + (η * c_σ) * sg.leading) ^ (2 * T) * modalSubspaceEnergy v_sig w₀ ≤
      frobeniusSq (H_T * P) := by
    linarith [hS_grad, hH_sig]
  have hB : frobeniusSq (H_T - H_T * P) ≤
      (1 + (η * c_σ) * sg.bulk) ^ (2 * T) * modalSubspaceEnergy v_bulk w₀ := by
    linarith [hB_grad, hH_bulk]
  have hB₀ : 0 ≤ modalSubspaceEnergy v_bulk w₀ := modalSubspaceEnergy_nonneg v_bulk w₀
  obtain ⟨halign, hdist⟩ := spectral_bias_alignment_disc_ge
    sg H_T P hP T (mul_pos hη hc_σ) hS₀ hB₀ hS hB
  refine ⟨halign, hdist, ?_⟩
  intro lam hlam
  exact (mup_aligned_effectiveDim_le_rank_add_eps
    r d_bulk (frobeniusSq_nonneg _) (frobeniusSq_nonneg _) hlam hdist).1

end OpenTheory.GradientSpectralEmergence
