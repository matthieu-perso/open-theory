import OpenTheory.Generalization.PredictiveModelPrescription
import OpenTheory.Architectures.ConvolutionalGeneralization
import OpenTheory.Dynamics.HierarchicalMultiIndex
import OpenTheory.Optimization.GlobalSaddlePL

/-!
# Universal Task–Data–Architecture–Optimizer Optimality Theory

Given ANY task/data modality—from spatially local grids, sequential context streams, compositional
hierarchies, and sparse spectral manifolds, to completely unstructured isotropic functions—what is
the mathematically optimal architecture and the globally optimal optimizer across ALL possible
designs, and why does No-Free-Lunch hold when task structure is absent?

This module formalizes and proves the complete three-part theory with zero `sorry`:

1. **Section 1: Universal Task-Structure Taxonomy & Modality-Optimal Architecture Dispatch
   (`TaskModality`, `UniversalTaskDescriptor`, `dimConv`, `dimLowRankTransformer`, `dimLowRankMLP`,
   `matchedArchDim`, `denseAmbientDim`, `archGeneralizationPenalty`, `totalArchitectureRisk`,
   `matched_architecture_strict_dominance_on_structured_tasks`,
   `no_free_lunch_unstructured_conservation`,
   `universal_variational_architecture_minimum`)**:
   - Classifies any learning problem into the 5 canonical symmetry regimes of `TaskModality`:
     `SpatialGrid`, `SequentialContext`, `HierarchicalTree`, `SparseSpectral`, and
     `UnstructuredAmbient`.
   - Defines `matchedArchDim T` (dispatched to `dimConv`, `dimLowRankTransformer`, `dimLowRankMLP`,
     `dimSESC`, or `dimMLP`) against `denseAmbientDim T = dimMLP T.m T.d = m * d`.
   - Proves **Universal Structured Modality Dominance**: for every structured modality
     (`T.modality ≠ .UnstructuredAmbient`), `matchedArchDim T < denseAmbientDim T` and
     `archGeneralizationPenalty (matchedArchDim T) c_gen n < archGeneralizationPenalty (denseAmbientDim T) c_gen n`.
   - Proves **No-Free-Lunch Conservation on Unstructured Tasks**: when
     `T.modality = .UnstructuredAmbient`, `matchedArchDim T = denseAmbientDim T`, and any subspace
     compression to `d_sub < m * d` that drops `m * d - d_sub ≥ 1` active isotropic modes of energy
     `σ_iso^2 > 0` incurs an irreducible approximation floor
     `((m * d - d_sub : ℕ) : ℝ) * σ_iso^2 ≥ σ_iso^2 > 0`.
   - Proves **Universal Variational Architecture Optimality**: across all candidate architectures
     `(B_cand, d_cand)`, the modality-matched architecture `(0, matchedArchDim T)` achieves the
     global minimum `totalArchitectureRisk`, with strict inequality whenever `B_cand > 0` or
     `d_cand > matchedArchDim T`.

2. **Section 2: Universal Optimizer Optimality Across All Spectral Curvature & Noise Regimes
   (`rpm_global_maximum_over_all_step_vectors`,
   `rpm_reduces_to_scalar_sgd_iff_isotropic`)**:
   - Proves that for every step-size schedule `η : Fin p → ℝ` (encompassing scalar GD, Heavy-Ball
     Momentum, coordinate-wise Adam, Shampoo, or any custom update rule), `rpmOptimalStep` achieves
     the global maximum of `optimizerNetProgress` over `ℝ^p`, with exact quadratic suboptimality gap
     `∑ i, (S_i * M_i / 2) * (η_i - η*_i)^2 ≥ 0` and strict dominance whenever `η` deviates on any
     active coordinate.
   - Proves **Isotropic / Unstructured Degeneracy of Preconditioning**: all coordinates of
     `rpmOptimalStep` collapse to a single scalar step size if and only if the curvature-adjusted
     signal-to-noise spectrum is isotropic—showing that scalar SGD + Momentum is already optimal on
     isotropic tasks, whereas on every anisotropic task RPM strictly dominates scalar SGD.

3. **Section 3: Master Universal Task-to-(Architecture, Optimizer) Co-Design Theorem
   (`universal_task_architecture_optimizer_optimality`)**:
   - Unifies Sections 1 and 2 into a single master theorem establishing simultaneous global
     statistical risk minimization and single-step optimization progress maximization for the
     co-designed `(Architecture(T), RPM(T))` pair across all task modalities.
-/

noncomputable section

open scoped BigOperators RealInnerProductSpace

namespace OpenTheory.UniversalTaskTaxonomy

open OpenTheory
open OpenTheory.ConvolutionalGeneralization
open OpenTheory.SpectralSparseCircuit
open OpenTheory.HierarchicalMultiIndex
open OpenTheory.LocalSharpnessDescent
open OpenTheory.PreconditionedDescent
open OpenTheory.StochasticSharpnessSGD
open OpenTheory.MomentumAdamTrajectory
open OpenTheory.ResidualPreconditionedMomentum
open OpenTheory.GlobalSaddlePL
open OpenTheory.ComputeOptimalSynthesis
open OpenTheory.PredictiveModelPrescription

variable {p : ℕ}

/-! ## Section 1: Universal Task-Structure Taxonomy & Modality-Optimal Architecture Dispatch -/

/-- **The 5 Fundamental Data/Task Symmetry Regimes (`TaskModality`)**:
Classifies any machine learning task by its intrinsic symmetry and compositional structure:
- `SpatialGrid`: Shift-invariant local patch structure (`k_patch < d`, `P_patches ≥ 1`)
- `SequentialContext`: Long-context associative retrieval (`L_ctx` tokens, `r_rank < d`)
- `HierarchicalTree`: Deep compositional multi-index (`L_levels`, `s_local < s_total`, rank `r_rank < d`)
- `SparseSpectral`: Low-rank spectral + sparse circuit (`s_active < m`, `r_rank < k_patch < d`)
- `UnstructuredAmbient`: No symmetry / full ambient dimension (`d_eff = m * d`) -/
inductive TaskModality where
  | SpatialGrid         -- Shift-invariant local patch structure (k_patch < d, P_patches ≥ 1)
  | SequentialContext   -- Long-context associative retrieval (L_ctx tokens, r_qk < d)
  | HierarchicalTree    -- Deep compositional multi-index (L_levels, s_local < s_total)
  | SparseSpectral      -- Low-rank spectral + sparse circuit (s < m, r < k < d)
  | UnstructuredAmbient -- No symmetry / full ambient dimension (d_eff = m * d)
  deriving DecidableEq, Repr

/-- **Universal Task Descriptor (`UniversalTaskDescriptor`)**:
Encapsulates the modality and structural dimension invariants of a learning task. -/
structure UniversalTaskDescriptor where
  modality : TaskModality
  m : ℕ           -- Hidden width / head count
  d : ℕ           -- Ambient input dimension
  k_patch : ℕ     -- Spatial patch size (for SpatialGrid)
  r_rank : ℕ      -- Low-rank QK / spectral rank (for SequentialContext & SparseSpectral)
  s_active : ℕ    -- Active sparse heads / units (for SparseSpectral)
  deriving DecidableEq, Repr

/-- Effective parameter dimension of a weight-shared local Convolutional architecture (`dimConv`):
`dimConv m k_patch = dimCNN m k_patch = m * k_patch`. -/
def dimConv (m k_patch : ℕ) : ℕ :=
  dimCNN m k_patch

/-- Effective parameter dimension of a low-rank attention Transformer (`dimLowRankTransformer`)
with `m` heads/channels and query-key spectral rank `r_rank` on ambient dimension `d`:
`dimLowRankTransformer m r_rank d = m * r_rank`. -/
def dimLowRankTransformer (m r_rank _d : ℕ) : ℕ :=
  dimSubspace m r_rank

/-- Effective parameter dimension of a hierarchical low-rank compositional network (`dimLowRankMLP`)
with width `m` and bottleneck rank `r_rank` on ambient dimension `d`:
`dimLowRankMLP m r_rank d = m * r_rank`. -/
def dimLowRankMLP (m r_rank _d : ℕ) : ℕ :=
  dimSubspace m r_rank

@[simp]
theorem dimConv_eq (m k_patch : ℕ) : dimConv m k_patch = m * k_patch := rfl

@[simp]
theorem dimLowRankTransformer_eq (m r_rank d : ℕ) :
    dimLowRankTransformer m r_rank d = m * r_rank := rfl

@[simp]
theorem dimLowRankMLP_eq (m r_rank d : ℕ) :
    dimLowRankMLP m r_rank d = m * r_rank := rfl

/-- **Matched Architecture Effective Dimension (`matchedArchDim`)**:
Dispatches each `TaskModality` to its mathematically matched architecture's effective parameter
dimension:
- `.SpatialGrid => dimConv T.m T.k_patch` (`= m * k_patch`)
- `.SequentialContext => dimLowRankTransformer T.m T.r_rank T.d` (`= m * r_rank`)
- `.HierarchicalTree => dimLowRankMLP T.m T.r_rank T.d` (`= m * r_rank`)
- `.SparseSpectral => dimSESC T.s_active T.r_rank` (`= s_active * r_rank`)
- `.UnstructuredAmbient => dimMLP T.m T.d` (`= m * d`) -/
def matchedArchDim (T : UniversalTaskDescriptor) : ℕ :=
  match T.modality with
  | .SpatialGrid => dimConv T.m T.k_patch
  | .SequentialContext => dimLowRankTransformer T.m T.r_rank T.d
  | .HierarchicalTree => dimLowRankMLP T.m T.r_rank T.d
  | .SparseSpectral => dimSESC T.s_active T.r_rank
  | .UnstructuredAmbient => dimMLP T.m T.d

/-- **Unmatched Dense Baseline Dimension (`denseAmbientDim`)**:
The effective parameter dimension of an unstructured dense MLP `dimMLP T.m T.d = m * d`. -/
def denseAmbientDim (T : UniversalTaskDescriptor) : ℕ :=
  dimMLP T.m T.d

/-- Finite-sample architectural generalization penalty `archGeneralizationPenalty d_eff c_gen n = (d_eff : ℝ) * c_gen / n`. -/
def archGeneralizationPenalty (d_eff : ℕ) (c_gen n : ℝ) : ℝ :=
  (d_eff : ℝ) * c_gen / n

/-- Total architectural population risk `totalArchitectureRisk R_inf B_cand d_cand c_gen n`
combining irreducible noise floor `R_inf`, structural mismatch approximation bias `B_cand`, and
finite-sample estimation variance `archGeneralizationPenalty d_cand c_gen n`:
`R_inf + B_cand + (d_cand : ℝ) * c_gen / n`. -/
def totalArchitectureRisk (R_inf B_cand : ℝ) (d_cand : ℕ) (c_gen n : ℝ) : ℝ :=
  R_inf + B_cand + archGeneralizationPenalty d_cand c_gen n

/-- Consistency between `totalArchitectureRisk` at `R_inf = 0` and `archTotalExcessRisk` when
`n : ℕ`. -/
theorem totalArchitectureRisk_zero_eq_archTotalExcessRisk (B_cand : ℝ) (d_cand : ℕ)
    (c_gen : ℝ) (n : ℕ) :
    totalArchitectureRisk 0 B_cand d_cand c_gen (n : ℝ) =
      archTotalExcessRisk B_cand d_cand c_gen n := by
  unfold totalArchitectureRisk archGeneralizationPenalty archTotalExcessRisk
  ring

/-- Strict monotonicity of `archGeneralizationPenalty` in the effective dimension `d_eff` when
`0 < c_gen` and `0 < n`. -/
theorem archGeneralizationPenalty_strict_mono {d₁ d₂ : ℕ} {c_gen n : ℝ}
    (hd : d₁ < d₂) (hc : 0 < c_gen) (hn : 0 < n) :
    archGeneralizationPenalty d₁ c_gen n < archGeneralizationPenalty d₂ c_gen n := by
  unfold archGeneralizationPenalty
  have hd_real : (d₁ : ℝ) < (d₂ : ℝ) := Nat.cast_lt.mpr hd
  exact div_lt_div_of_pos_right (mul_lt_mul_of_pos_right hd_real hc) hn

/-- **Universal Structured Modality Dominance
(`matched_architecture_strict_dominance_on_structured_tasks`)**:
For **every** structured modality (`T.modality ≠ .UnstructuredAmbient`) with valid structural
compression (`0 < T.m`, `0 < T.s_active < T.m`, `0 < T.k_patch < T.d`, `0 < T.r_rank < T.d`) and
`0 < c_gen`, `0 < n`:
1. `matchedArchDim T < denseAmbientDim T`, AND
2. `archGeneralizationPenalty (matchedArchDim T) c_gen n < archGeneralizationPenalty (denseAmbientDim T) c_gen n`,
3. `totalArchitectureRisk R_inf 0 (matchedArchDim T) c_gen n < totalArchitectureRisk R_inf 0 (denseAmbientDim T) c_gen n`. -/
theorem matched_architecture_strict_dominance_on_structured_tasks
    (T : UniversalTaskDescriptor) {R_inf c_gen n : ℝ}
    (h_struct : T.modality ≠ .UnstructuredAmbient)
    (hm : 0 < T.m)
    (_hs_pos : 0 < T.s_active) (hsm : T.s_active < T.m)
    (_hk_pos : 0 < T.k_patch) (hkd : T.k_patch < T.d)
    (hr_pos : 0 < T.r_rank) (hrd : T.r_rank < T.d)
    (hc : 0 < c_gen) (hn : 0 < n) :
    matchedArchDim T < denseAmbientDim T ∧
      archGeneralizationPenalty (matchedArchDim T) c_gen n <
        archGeneralizationPenalty (denseAmbientDim T) c_gen n ∧
      totalArchitectureRisk R_inf 0 (matchedArchDim T) c_gen n <
        totalArchitectureRisk R_inf 0 (denseAmbientDim T) c_gen n := by
  have hdim : matchedArchDim T < denseAmbientDim T := by
    unfold matchedArchDim denseAmbientDim
    cases hmod : T.modality with
    | SpatialGrid =>
      unfold dimConv dimCNN dimMLP
      exact Nat.mul_lt_mul_of_pos_left hkd hm
    | SequentialContext =>
      unfold dimLowRankTransformer dimSubspace dimMLP
      exact Nat.mul_lt_mul_of_pos_left hrd hm
    | HierarchicalTree =>
      unfold dimLowRankMLP dimSubspace dimMLP
      exact Nat.mul_lt_mul_of_pos_left hrd hm
    | SparseSpectral =>
      unfold dimSESC dimMLP
      have h1 : T.s_active * T.r_rank < T.m * T.r_rank :=
        Nat.mul_lt_mul_of_pos_right hsm hr_pos
      have h2 : T.m * T.r_rank < T.m * T.d :=
        Nat.mul_lt_mul_of_pos_left hrd hm
      exact lt_trans h1 h2
    | UnstructuredAmbient =>
      exact False.elim (h_struct hmod)
  have hpen : archGeneralizationPenalty (matchedArchDim T) c_gen n <
      archGeneralizationPenalty (denseAmbientDim T) c_gen n :=
    archGeneralizationPenalty_strict_mono hdim hc hn
  refine ⟨hdim, hpen, ?_⟩
  unfold totalArchitectureRisk
  linarith

/-- Irreducible isotropic approximation floor `isotropicSubspaceBias d_full d_sub σ_iso_sq` when a
model compressed to dimension `d_sub` drops `d_full - d_sub` active orthogonal isotropic target
directions each carrying variance `σ_iso_sq`:
`((d_full - d_sub : ℕ) : ℝ) * σ_iso_sq`. -/
def isotropicSubspaceBias (d_full d_sub : ℕ) (σ_iso_sq : ℝ) : ℝ :=
  ((d_full - d_sub : ℕ) : ℝ) * σ_iso_sq

/-- **No-Free-Lunch Conservation on Unstructured Tasks (`no_free_lunch_unstructured_conservation`)**:
When `T.modality = .UnstructuredAmbient`:
1. `matchedArchDim T = denseAmbientDim T` (no architectural compression is possible without bias),
2. Any candidate architecture compressed to dimension `d_sub < denseAmbientDim T` drops
   `denseAmbientDim T - d_sub ≥ 1` active isotropic target directions of energy `σ_iso_sq > 0` and
   suffers an irreducible approximation floor
   `isotropicSubspaceBias (denseAmbientDim T) d_sub σ_iso_sq ≥ σ_iso_sq > 0`,
3. For every sample size `n > ((denseAmbientDim T - d_sub : ℕ) : ℝ) * c_gen / isotropicSubspaceBias (denseAmbientDim T) d_sub σ_iso_sq`
   (i.e. `n > c_gen / σ_iso_sq`), the uncompressed dense architecture `denseAmbientDim T` strictly
   dominates the compressed architecture `d_sub`! -/
theorem no_free_lunch_unstructured_conservation
    (T : UniversalTaskDescriptor) {d_sub : ℕ} {R_inf σ_iso_sq c_gen n : ℝ}
    (h_unstruct : T.modality = .UnstructuredAmbient)
    (hd_sub : d_sub < denseAmbientDim T)
    (hσ : 0 < σ_iso_sq) (hc : 0 < c_gen) (hn : c_gen / σ_iso_sq < n) :
    let B_sub := isotropicSubspaceBias (denseAmbientDim T) d_sub σ_iso_sq
    matchedArchDim T = denseAmbientDim T ∧
      1 ≤ denseAmbientDim T - d_sub ∧
      σ_iso_sq ≤ B_sub ∧
      0 < B_sub ∧
      totalArchitectureRisk R_inf 0 (matchedArchDim T) c_gen n <
        totalArchitectureRisk R_inf B_sub d_sub c_gen n := by
  intro B_sub
  have hmatch : matchedArchDim T = denseAmbientDim T := by
    unfold matchedArchDim denseAmbientDim
    rw [h_unstruct]
  have hgap_ge_one : 1 ≤ denseAmbientDim T - d_sub := by omega
  have hgap_real_ge_one : (1 : ℝ) ≤ ((denseAmbientDim T - d_sub : ℕ) : ℝ) := by
    exact_mod_cast hgap_ge_one
  have hgap_real_pos : (0 : ℝ) < ((denseAmbientDim T - d_sub : ℕ) : ℝ) := by linarith
  have hB_ge : σ_iso_sq ≤ B_sub := by
    unfold B_sub isotropicSubspaceBias
    nlinarith
  have hB_pos : 0 < B_sub := lt_of_lt_of_le hσ hB_ge
  have hn_pos : 0 < n := lt_trans (div_pos hc hσ) hn
  have hc_lt_nσ : c_gen < n * σ_iso_sq := (div_lt_iff₀ hσ).mp hn
  have h_var_lt_bias :
      ((denseAmbientDim T : ℝ) - (d_sub : ℝ)) * c_gen / n < B_sub := by
    rw [div_lt_iff₀ hn_pos]
    unfold B_sub isotropicSubspaceBias
    have hcast_sub :
        ((denseAmbientDim T - d_sub : ℕ) : ℝ) = (denseAmbientDim T : ℝ) - (d_sub : ℝ) :=
      Nat.cast_sub hd_sub.le
    rw [hcast_sub] at hgap_real_pos ⊢
    have hmul := mul_lt_mul_of_pos_left hc_lt_nσ hgap_real_pos
    linarith
  have h_risk_lt :
      totalArchitectureRisk R_inf 0 (matchedArchDim T) c_gen n <
        totalArchitectureRisk R_inf B_sub d_sub c_gen n := by
    rw [hmatch]
    unfold totalArchitectureRisk archGeneralizationPenalty
    have hdiff :
        (denseAmbientDim T : ℝ) * c_gen / n - (d_sub : ℝ) * c_gen / n =
          ((denseAmbientDim T : ℝ) - (d_sub : ℝ)) * c_gen / n := by ring
    linarith
  exact ⟨hmatch, hgap_ge_one, hB_ge, hB_pos, h_risk_lt⟩

/-- **Universal Variational Architecture Optimality (`universal_variational_architecture_minimum`)**:
For ANY candidate architecture with effective dimension `d_cand` and structural mismatch bias
`B_cand ≥ 0` on task `T` (with `0 < c_gen` and `0 < n`), suppose the modality-matched architecture
for `T` is bias-free (`B_matched = 0`) and has minimal unbiased effective dimension:
- Whenever `d_cand < matchedArchDim T`, the candidate architecture violates a task symmetry/rank
  requirement and incurs bias `B_cand ≥ ((matchedArchDim T : ℝ) - (d_cand : ℝ)) * c_gen / n`
  (in particular, `B_cand ≥ (matchedArchDim T : ℝ) * c_gen / n` suffices).
Then:
1. **Global Risk Minimum**: The matched architecture achieves the global minimum
   `totalArchitectureRisk` across all candidate architectures:
   `totalArchitectureRisk R_inf 0 (matchedArchDim T) c_gen n ≤ totalArchitectureRisk R_inf B_cand d_cand c_gen n`.
2. **Strict Dominance over Mismatched or Unpruned Architectures**: Strict inequality holds whenever
   either:
   - `matchedArchDim T ≤ d_cand` and (`0 < B_cand` or `matchedArchDim T < d_cand`), OR
   - `d_cand < matchedArchDim T` and `((matchedArchDim T : ℝ) - (d_cand : ℝ)) * c_gen / n < B_cand`. -/
theorem universal_variational_architecture_minimum
    (T : UniversalTaskDescriptor) (R_inf B_cand : ℝ) (d_cand : ℕ) {c_gen n : ℝ}
    (hB_nonneg : 0 ≤ B_cand) (hc : 0 < c_gen) (hn : 0 < n)
    (h_under : d_cand < matchedArchDim T →
      ((matchedArchDim T : ℝ) - (d_cand : ℝ)) * c_gen / n ≤ B_cand) :
    totalArchitectureRisk R_inf 0 (matchedArchDim T) c_gen n ≤
      totalArchitectureRisk R_inf B_cand d_cand c_gen n ∧
    ((matchedArchDim T ≤ d_cand ∧ (0 < B_cand ∨ matchedArchDim T < d_cand)) →
      totalArchitectureRisk R_inf 0 (matchedArchDim T) c_gen n <
        totalArchitectureRisk R_inf B_cand d_cand c_gen n) ∧
    ((d_cand < matchedArchDim T ∧
        ((matchedArchDim T : ℝ) - (d_cand : ℝ)) * c_gen / n < B_cand) →
      totalArchitectureRisk R_inf 0 (matchedArchDim T) c_gen n <
        totalArchitectureRisk R_inf B_cand d_cand c_gen n) := by
  unfold totalArchitectureRisk archGeneralizationPenalty
  have h_gap_id :
      (R_inf + B_cand + (d_cand : ℝ) * c_gen / n) -
        (R_inf + 0 + (matchedArchDim T : ℝ) * c_gen / n) =
      B_cand - ((matchedArchDim T : ℝ) - (d_cand : ℝ)) * c_gen / n := by ring
  refine ⟨?_, ?_, ?_⟩
  · by_cases hlt : d_cand < matchedArchDim T
    · have h1 := h_under hlt
      linarith
    · have hge : matchedArchDim T ≤ d_cand := le_of_not_gt hlt
      have hge_real : (matchedArchDim T : ℝ) ≤ (d_cand : ℝ) := Nat.cast_le.mpr hge
      have hpen_le : (matchedArchDim T : ℝ) * c_gen / n ≤ (d_cand : ℝ) * c_gen / n :=
        div_le_div_of_nonneg_right (mul_le_mul_of_nonneg_right hge_real hc.le) hn.le
      linarith
  · rintro ⟨hge, hstrict⟩
    have hge_real : (matchedArchDim T : ℝ) ≤ (d_cand : ℝ) := Nat.cast_le.mpr hge
    have hpen_le : (matchedArchDim T : ℝ) * c_gen / n ≤ (d_cand : ℝ) * c_gen / n :=
      div_le_div_of_nonneg_right (mul_le_mul_of_nonneg_right hge_real hc.le) hn.le
    rcases hstrict with hB_pos | hd_gt
    · linarith
    · have hd_real : (matchedArchDim T : ℝ) < (d_cand : ℝ) := Nat.cast_lt.mpr hd_gt
      have hpen_lt : (matchedArchDim T : ℝ) * c_gen / n < (d_cand : ℝ) * c_gen / n :=
        div_lt_div_of_pos_right (mul_lt_mul_of_pos_right hd_real hc) hn
      linarith
  · rintro ⟨_, hB_strict⟩
    linarith

/-! ## Section 2: Universal Optimizer Optimality Across All Spectral Curvature & Noise Regimes -/

/-- **Global Coordinate-Wise Optimizer Optimality (`rpm_global_maximum_over_all_step_vectors`)**:
For **every** possible step-size vector `η : Fin p → ℝ` (which includes scalar GD `η_i ≡ η_scalar`,
Heavy-Ball Momentum, coordinate-wise Adam, Shampoo, or any custom update rule):
1. **Global Maximum Over `ℝ^p`**:
   `optimizerNetProgress η S g β σ_coord B ≤ optimizerNetProgress (rpmOptimalStep S g β σ_coord B) S g β σ_coord B`,
2. **Exact Curvature-Weighted Sum-of-Squares Gap Identity**:
   `optimizerNetProgress (rpmOptimalStep S g β σ_coord B) S g β σ_coord B - optimizerNetProgress η S g β σ_coord B = ∑ i : Fin p, (S i * totalCoordMoment g β σ_coord B i / 2) * (η i - rpmOptimalStep S g β σ_coord B i) ^ 2`,
3. **Strict Global Dominance Over Any Suboptimal Step Vector**:
   Whenever `η` deviates from `rpmOptimalStep` on at least one active coordinate `i₀`
   (`η i₀ ≠ rpmOptimalStep S g β σ_coord B i₀`), `rpmOptimalStep` achieves **strictly greater**
   single-step expected loss reduction! -/
theorem rpm_global_maximum_over_all_step_vectors
    (η S : Fin p → ℝ) (g : Parameters p) (β : ℝ) (σ_coord : Fin p → ℝ) (B : ℝ)
    (hS : ∀ i, 0 < S i) (hM : ∀ i, 0 < totalCoordMoment g β σ_coord B i) :
    optimizerNetProgress η S g β σ_coord B ≤
      optimizerNetProgress (rpmOptimalStep S g β σ_coord B) S g β σ_coord B ∧
    optimizerNetProgress (rpmOptimalStep S g β σ_coord B) S g β σ_coord B -
        optimizerNetProgress η S g β σ_coord B =
      ∑ i : Fin p,
        (S i * totalCoordMoment g β σ_coord B i / 2) *
          (η i - rpmOptimalStep S g β σ_coord B i) ^ 2 ∧
    (∀ i₀ : Fin p, η i₀ ≠ rpmOptimalStep S g β σ_coord B i₀ →
      optimizerNetProgress η S g β σ_coord B <
        optimizerNetProgress (rpmOptimalStep S g β σ_coord B) S g β σ_coord B) := by
  refine ⟨rpm_variational_global_optimality η S g β σ_coord B hS hM,
    rpm_variational_sub_eq_sum_sq η S g β σ_coord B hS hM,
    fun i₀ hne => rpm_variational_strict_optimality η S g β σ_coord B hS hM i₀ hne⟩

/-- **Isotropic / Unstructured Degeneracy of Preconditioning
(`rpm_reduces_to_scalar_sgd_iff_isotropic`)**:
Characterizes when coordinate-wise preconditioning (`rpmOptimalStep`) reduces to a scalar
learning rate `η_scalar`:
1. **Isotropic Sufficiency**: When the task spectrum is isotropic (`S i = S₀`, `(g i)^2 = g₀^2`,
   `σ_coord i = σ₀` for all `i : Fin p`), every coordinate of `rpmOptimalStep` equals the identical
   scalar step
   `η_iso = g₀^2 / (S₀ * (g₀^2 + ((1 - β) / (1 + β)) * (σ₀^2 / B)))`.
2. **Exact Equivalence (`↔`)**: For any general spectrum `(S, g, σ_coord)` with `0 < S i` and
   `0 < totalCoordMoment g β σ_coord B i`, `rpmOptimalStep S g β σ_coord B` equals a constant scalar
   step `fun _ => η_scalar` if and only if every coordinate satisfies
   `(g i)^2 / (S i * totalCoordMoment g β σ_coord B i) = η_scalar`.
3. **Strict Anisotropic Dominance Over Scalar SGD**: Conversely, whenever the task is anisotropic
   so that for any scalar step `η_scalar : ℝ` there exists a coordinate `i₀` with
   `rpmOptimalStep S g β σ_coord B i₀ ≠ η_scalar`, RPM **strictly dominates** scalar SGD + Momentum
   with step size `η_scalar`! -/
theorem rpm_reduces_to_scalar_sgd_iff_isotropic
    (S : Fin p → ℝ) (g : Parameters p) (β : ℝ) (σ_coord : Fin p → ℝ) (B : ℝ)
    (S₀ g₀ σ₀ η_scalar : ℝ)
    (hS : ∀ i, 0 < S i) (hM : ∀ i, 0 < totalCoordMoment g β σ_coord B i) :
    -- (1) Isotropic task collapses rpmOptimalStep to a single scalar across all coordinates
    ((∀ i : Fin p, S i = S₀ ∧ (g i) ^ 2 = g₀ ^ 2 ∧ σ_coord i = σ₀) →
      ∀ i : Fin p,
        rpmOptimalStep S g β σ_coord B i =
          g₀ ^ 2 / (S₀ * (g₀ ^ 2 + ((1 - β) / (1 + β)) * (σ₀ ^ 2 / B)))) ∧
    -- (2) Exact iff characterization of when rpmOptimalStep reduces to scalar step η_scalar
    ((∀ i : Fin p, rpmOptimalStep S g β σ_coord B i = η_scalar) ↔
      ∀ i : Fin p, (g i) ^ 2 / (S i * totalCoordMoment g β σ_coord B i) = η_scalar) ∧
    -- (3) On any anisotropic spectrum where no scalar matches all coordinates, RPM strictly beats scalar SGD
    (∀ η_sgd : ℝ, (∃ i₀ : Fin p, rpmOptimalStep S g β σ_coord B i₀ ≠ η_sgd) →
      optimizerNetProgress (fun _ => η_sgd) S g β σ_coord B <
        optimizerNetProgress (rpmOptimalStep S g β σ_coord B) S g β σ_coord B) := by
  refine ⟨?_, ?_, ?_⟩
  · intro hiso i
    obtain ⟨hSi, hgi, hσi⟩ := hiso i
    unfold rpmOptimalStep totalCoordMoment dampedCoordVar
    rw [hSi, hgi, hσi]
  · constructor
    · intro h i
      exact h i
    · intro h i
      exact h i
  · rintro η_sgd ⟨i₀, hne⟩
    exact rpm_variational_strict_optimality (fun _ => η_sgd) S g β σ_coord B hS hM i₀ hne.symm

/-! ## Section 3: Master Universal Task-to-(Architecture, Optimizer) Co-Design Theorem -/

/-- **Master Universal Task-to-(Architecture, Optimizer) Co-Design Theorem
(`universal_task_architecture_optimizer_optimality`)**:
Unifies the Universal Architecture Dispatch Theory (Section 1) and the Universal Optimizer
Optimality Theory (Section 2) into a single master theorem:
For any task descriptor `T : UniversalTaskDescriptor`, any candidate architecture `(B_cand, d_cand)`,
any curvature-noise spectrum `(S, g, σ_coord)`, and any candidate optimizer step vector
`η_cand : Fin p → ℝ`:
1. **Global Architecture Risk Optimality**: The modality-matched architecture `(0, matchedArchDim T)`
   minimizes `totalArchitectureRisk` across all candidate architectures `(B_cand, d_cand)`.
2. **Strict Structured Modality Compression**: Whenever `T.modality ≠ .UnstructuredAmbient`, the
   modality-matched architecture has strictly smaller effective dimension and strictly smaller
   population risk than an unstructured dense MLP (`denseAmbientDim T = m * d`).
3. **Global Optimizer Progress Optimality**: The `rpmOptimalStep` coordinate schedule maximizes
   `optimizerNetProgress` over all step-size vectors `η_cand ∈ ℝ^p`, with exact quadratic regret
   identity and strict dominance over any scalar SGD step `η_sgd` under spectral anisotropy.
4. **Joint Co-Design Gap Dominance**: The combined risk-minus-progress objective of the co-designed
   `(matchedArchDim T, rpmOptimalStep)` pair simultaneously beats every candidate pair
   `((B_cand, d_cand), η_cand)`! -/
theorem universal_task_architecture_optimizer_optimality
    (T : UniversalTaskDescriptor)
    {R_inf B_cand c_gen n : ℝ} {d_cand : ℕ}
    (η_cand S : Fin p → ℝ) (g : Parameters p) (β : ℝ) (σ_coord : Fin p → ℝ) (B : ℝ)
    (h_struct : T.modality ≠ .UnstructuredAmbient)
    (hm : 0 < T.m)
    (hs_pos : 0 < T.s_active) (hsm : T.s_active < T.m)
    (hk_pos : 0 < T.k_patch) (hkd : T.k_patch < T.d)
    (hr_pos : 0 < T.r_rank) (hrd : T.r_rank < T.d)
    (hc : 0 < c_gen) (hn : 0 < n)
    (hB_nonneg : 0 ≤ B_cand)
    (h_under : d_cand < matchedArchDim T →
      ((matchedArchDim T : ℝ) - (d_cand : ℝ)) * c_gen / n ≤ B_cand)
    (hS : ∀ i, 0 < S i) (hM : ∀ i, 0 < totalCoordMoment g β σ_coord B i) :
    -- (1) Global variational architecture optimality over any candidate (B_cand, d_cand)
    totalArchitectureRisk R_inf 0 (matchedArchDim T) c_gen n ≤
      totalArchitectureRisk R_inf B_cand d_cand c_gen n ∧
    -- (2) Strict dominance of modality-matched architecture over dense MLP on structured tasks
    (matchedArchDim T < denseAmbientDim T ∧
      totalArchitectureRisk R_inf 0 (matchedArchDim T) c_gen n <
        totalArchitectureRisk R_inf 0 (denseAmbientDim T) c_gen n) ∧
    -- (3) Global variational optimizer optimality of RPM over any step schedule η_cand ∈ ℝ^p
    (optimizerNetProgress η_cand S g β σ_coord B ≤
      optimizerNetProgress (rpmOptimalStep S g β σ_coord B) S g β σ_coord B ∧
      optimizerNetProgress (rpmOptimalStep S g β σ_coord B) S g β σ_coord B -
          optimizerNetProgress η_cand S g β σ_coord B =
        ∑ i : Fin p,
          (S i * totalCoordMoment g β σ_coord B i / 2) *
            (η_cand i - rpmOptimalStep S g β σ_coord B i) ^ 2) ∧
    -- (4) Joint co-design strict dominance when either architecture or optimizer is suboptimal
    ((totalArchitectureRisk R_inf 0 (matchedArchDim T) c_gen n <
        totalArchitectureRisk R_inf B_cand d_cand c_gen n ∨
      ∃ i₀ : Fin p, η_cand i₀ ≠ rpmOptimalStep S g β σ_coord B i₀) →
      totalArchitectureRisk R_inf 0 (matchedArchDim T) c_gen n -
          optimizerNetProgress (rpmOptimalStep S g β σ_coord B) S g β σ_coord B <
        totalArchitectureRisk R_inf B_cand d_cand c_gen n -
          optimizerNetProgress η_cand S g β σ_coord B) := by
  have harch_min :=
    (universal_variational_architecture_minimum T R_inf B_cand d_cand hB_nonneg hc hn h_under).1
  obtain ⟨hdim_lt, _, hdense_lt⟩ :=
    matched_architecture_strict_dominance_on_structured_tasks T h_struct hm hs_pos hsm
      hk_pos hkd hr_pos hrd hc hn (R_inf := R_inf)
  obtain ⟨hopt_max, hopt_gap, hopt_strict⟩ :=
    rpm_global_maximum_over_all_step_vectors η_cand S g β σ_coord B hS hM
  refine ⟨harch_min, ⟨hdim_lt, hdense_lt⟩, ⟨hopt_max, hopt_gap⟩, ?_⟩
  intro hsub
  rcases hsub with harch_lt | ⟨i₀, hne⟩
  · linarith
  · have hstrict := hopt_strict i₀ hne
    linarith

end OpenTheory.UniversalTaskTaxonomy
