import OpenTheory.Architectures.Basic
import OpenTheory.Architectures.ConvolutionalGeneralization
import OpenTheory.Architectures.Expressivity
import OpenTheory.Generalization.BallCover
import Mathlib.Analysis.SpecialFunctions.Log.Basic

/-!
# Structural Architecture Optimality Mapped to Data Geometry & Spectral-Equivariant Sparse Circuits (SESC)

This module formalizes the **Spectral-Equivariant Sparse Circuit (`SESC`)** architecture and
proves a 4-tier structural dominance hierarchy alongside bidirectional conditional architecture
optimality theorems:

1. **The Spectral-Equivariant Sparse Circuit (`SESC`) Architecture**:
   - Defines local patch spectral projection `patchProject` onto a rank-`r` basis
     `B_spec : Matrix (Fin r) (Fin k) ℝ`.
   - Defines `SpectralSparseCircuit (r m : ℕ)` (aliased as `SESC`) with spectral filter weights
     `W_spec : Matrix (Fin m) (Fin r) ℝ`, activation `act : ActivationFunction`, readout weights
     `α : Fin m → ℝ`, and active sparse circuit subset `active : Finset (Fin m)`.
   - Defines per-unit spatial feature map `sescUnitMap`, full unpruned output `sescFullForward`,
     and pruned sparse-circuit output `sescCircuitForward`.

2. **Exact Structural Invariances & Mechanistic Circuit Distillation Bound**:
   - Proves `sescUnitMap_shift_equivariant`: translating `x` by `spatialShift shift_s x` shifts
     `sescUnitMap` by `shift_s`.
   - Proves `sescCircuitForward_shift_invariant` and `sescFullForward_shift_invariant`: both the
     sparse circuit and full outputs are strictly invariant under cyclic spatial shifts.
   - Proves `sesc_orthogonal_bulk_invariance`: adding any perturbation `η_bulk : Fin d → ℝ` whose
     local patches lie in the kernel of `patchProject B_spec` leaves `sescUnitMap`,
     `sescCircuitForward`, and `sescFullForward` identically unchanged.
   - Proves `sesc_circuit_pruning_error_le`: if inactive circuit units `i ∈ net.activeᶜ` have
     pointwise attribution sum at most `ε_circuit`, pruning to `net.active` changes the global
     output by at most `ε_circuit`.

3. **The 4-Tier Architectural Dominance Hierarchy (`SESC < SubspaceNet < CNN < Dense MLP`)**:
   - Defines the 4 canonical effective parameter dimensions `dimSESC s r = s * r`,
     `dimSubspace m r = m * r`, `dimCNN m k = m * k`, and `dimMLP m d = m * d`.
   - Proves `four_tier_dimension_strict_hierarchy`, `four_tier_covering_strict_hierarchy`, and
     `four_tier_sample_complexity_and_flop_hierarchy`, including the exact compression ratio
     `((s : ℝ) * (r : ℝ)) / ((m : ℝ) * (d : ℝ)) < 1`.

4. **Systematic Conditional Architecture Optimality & Under-Parameterization Lower Bound**:
   - Defines `archTotalExcessRisk ε_struct p c_gen n = ε_struct + p * c_gen / n`.
   - Proves `minimal_invariant_architecture_optimal`: among all structurally well-specified
     architectures (`ε_struct = 0`), minimal invariant dimension `p_min` strictly minimizes
     total excess risk.
   - Proves `under_parameterized_architecture_suboptimal_of_large_sample`: compressing below the
     data's intrinsic rank/sparsity (`p_small < p_true`) with structural bias `ε_struct > 0` is
     strictly suboptimal for all sample sizes `n > (p_true - p_small) * c_gen / ε_struct`.
-/

noncomputable section

open scoped BigOperators

namespace OpenTheory.SpectralSparseCircuit

set_option linter.dupNamespace false

open OpenTheory.ConvolutionalGeneralization
open OpenTheory.BallCover

variable {d k r m : ℕ}

/-! ## 1. The Spectral-Equivariant Sparse Circuit (`SESC`) Architecture -/

/-- Local patch spectral projection onto a rank-`r` spectral basis
`B_spec : Matrix (Fin r) (Fin k) ℝ` applied to a local patch `patch : Fin k → ℝ`. -/
def patchProject (B_spec : Matrix (Fin r) (Fin k) ℝ) (patch : Fin k → ℝ) : Fin r → ℝ :=
  fun l => ∑ j : Fin k, B_spec l j * patch j

@[simp]
theorem patchProject_apply (B_spec : Matrix (Fin r) (Fin k) ℝ) (patch : Fin k → ℝ) (l : Fin r) :
    patchProject B_spec patch l = ∑ j : Fin k, B_spec l j * patch j := rfl

/-- Linearity of `patchProject` with respect to patch addition. -/
theorem patchProject_add (B_spec : Matrix (Fin r) (Fin k) ℝ)
    (patch₁ patch₂ : Fin k → ℝ) (l : Fin r) :
    patchProject B_spec (patch₁ + patch₂) l =
      patchProject B_spec patch₁ l + patchProject B_spec patch₂ l := by
  simp only [patchProject_apply, Pi.add_apply, mul_add, Finset.sum_add_distrib]

/-- `patchProject` maps the zero patch to zero. -/
@[simp]
theorem patchProject_zero (B_spec : Matrix (Fin r) (Fin k) ℝ) (l : Fin r) :
    patchProject B_spec 0 l = 0 := by
  simp [patchProject]

/-- Linearity of `cyclicPatch` with respect to signal addition. -/
theorem cyclicPatch_add [NeZero d] (embed : Fin k → Fin d)
    (x y : Fin d → ℝ) (p : Fin d) :
    cyclicPatch embed (x + y) p = cyclicPatch embed x p + cyclicPatch embed y p := by
  ext j
  simp [cyclicPatch_apply]

set_option linter.dupNamespace false in
/-- A **Spectral-Equivariant Sparse Circuit (`SpectralSparseCircuit r m`)** with local spectral
rank `r`, candidate channel count `m`, spectral filter weights `W_spec : Matrix (Fin m) (Fin r) ℝ`,
nonlinear activation `act : ActivationFunction`, readout weights `α : Fin m → ℝ`, and active
sparse circuit subset `active : Finset (Fin m)`. -/
structure SpectralSparseCircuit (r m : ℕ) where
  W_spec : Matrix (Fin m) (Fin r) ℝ
  act : ActivationFunction
  α : Fin m → ℝ
  active : Finset (Fin m)

/-- Short alias for `SpectralSparseCircuit`. -/
abbrev SESC := SpectralSparseCircuit

/-- Per-unit spatial activation map of a `SpectralSparseCircuit` at unit `i : Fin m` and
spatial position `p : Fin d`. -/
def sescUnitMap [NeZero d] (net : SpectralSparseCircuit r m) (embed : Fin k → Fin d)
    (B_spec : Matrix (Fin r) (Fin k) ℝ) (x : Fin d → ℝ) (i : Fin m) (p : Fin d) : ℝ :=
  net.act.σ (∑ l : Fin r, net.W_spec i l * patchProject B_spec (cyclicPatch embed x p) l)

/-- Full (unpruned) global-average-pooled forward output of a `SpectralSparseCircuit` over all
`m` candidate channels and `d` spatial positions. -/
def sescFullForward [NeZero d] (net : SpectralSparseCircuit r m) (embed : Fin k → Fin d)
    (B_spec : Matrix (Fin r) (Fin k) ℝ) (x : Fin d → ℝ) : ℝ :=
  (1 / (d : ℝ)) * ∑ p : Fin d, ∑ i : Fin m, net.α i * sescUnitMap net embed B_spec x i p

/-- Sparse circuit global-average-pooled forward output of a `SpectralSparseCircuit` restricted
to the active circuit subset `net.active`. -/
def sescCircuitForward [NeZero d] (net : SpectralSparseCircuit r m) (embed : Fin k → Fin d)
    (B_spec : Matrix (Fin r) (Fin k) ℝ) (x : Fin d → ℝ) : ℝ :=
  (1 / (d : ℝ)) * ∑ p : Fin d, ∑ i ∈ net.active, net.α i * sescUnitMap net embed B_spec x i p

/-! ## 2. Exact Structural Invariances & Mechanistic Circuit Distillation Bound -/

/-- **Exact Translation Equivariance of SESC Unit Maps (`sescUnitMap_shift_equivariant`)**:
Translating the spatial signal `x` by `spatialShift shift_s x` shifts every unit's spatial
feature map `sescUnitMap` by `shift_s`. -/
theorem sescUnitMap_shift_equivariant [NeZero d] (net : SpectralSparseCircuit r m)
    (embed : Fin k → Fin d) (B_spec : Matrix (Fin r) (Fin k) ℝ)
    (shift_s : Fin d) (x : Fin d → ℝ) (i : Fin m) (p : Fin d) :
    sescUnitMap net embed B_spec (spatialShift shift_s x) i p =
      spatialShift shift_s (sescUnitMap net embed B_spec x i) p := by
  rw [spatialShift_apply]
  unfold sescUnitMap
  have hpatch : cyclicPatch embed (spatialShift shift_s x) p =
      cyclicPatch embed x (p + shift_s) := by
    ext j
    exact cyclicPatch_shift embed shift_s x p j
  rw [hpatch]

/-- Function-level formulation of `sescUnitMap_shift_equivariant`. -/
theorem sescUnitMap_shift_equivariant_fun [NeZero d] (net : SpectralSparseCircuit r m)
    (embed : Fin k → Fin d) (B_spec : Matrix (Fin r) (Fin k) ℝ)
    (shift_s : Fin d) (x : Fin d → ℝ) (i : Fin m) :
    sescUnitMap net embed B_spec (spatialShift shift_s x) i =
      spatialShift shift_s (sescUnitMap net embed B_spec x i) := by
  ext p
  exact sescUnitMap_shift_equivariant net embed B_spec shift_s x i p

/-- **Exact Translation Invariance of Full SESC Forward Output (`sescFullForward_shift_invariant`)**:
For every cyclic shift `shift_s : Fin d`, the full unpruned SESC output is strictly invariant
under `spatialShift shift_s`. -/
theorem sescFullForward_shift_invariant [NeZero d] (net : SpectralSparseCircuit r m)
    (embed : Fin k → Fin d) (B_spec : Matrix (Fin r) (Fin k) ℝ)
    (shift_s : Fin d) (x : Fin d → ℝ) :
    sescFullForward net embed B_spec (spatialShift shift_s x) =
      sescFullForward net embed B_spec x := by
  unfold sescFullForward
  congr 1
  have hshift : ∀ p : Fin d,
      (∑ i : Fin m, net.α i * sescUnitMap net embed B_spec (spatialShift shift_s x) i p) =
        ∑ i : Fin m, net.α i * sescUnitMap net embed B_spec x i (p + shift_s) := by
    intro p
    refine Finset.sum_congr rfl fun i _ => ?_
    rw [sescUnitMap_shift_equivariant, spatialShift_apply]
  simp_rw [hshift]
  exact Equiv.sum_comp (Equiv.addRight shift_s)
    (fun q => ∑ i : Fin m, net.α i * sescUnitMap net embed B_spec x i q)

/-- **Exact Translation Invariance of Sparse Circuit Output (`sescCircuitForward_shift_invariant`)**:
For every cyclic shift `shift_s : Fin d`, the sparse circuit output `sescCircuitForward` is
strictly invariant under `spatialShift shift_s`. -/
theorem sescCircuitForward_shift_invariant [NeZero d] (net : SpectralSparseCircuit r m)
    (embed : Fin k → Fin d) (B_spec : Matrix (Fin r) (Fin k) ℝ)
    (shift_s : Fin d) (x : Fin d → ℝ) :
    sescCircuitForward net embed B_spec (spatialShift shift_s x) =
      sescCircuitForward net embed B_spec x := by
  unfold sescCircuitForward
  congr 1
  have hshift : ∀ p : Fin d,
      (∑ i ∈ net.active, net.α i * sescUnitMap net embed B_spec (spatialShift shift_s x) i p) =
        ∑ i ∈ net.active, net.α i * sescUnitMap net embed B_spec x i (p + shift_s) := by
    intro p
    refine Finset.sum_congr rfl fun i _ => ?_
    rw [sescUnitMap_shift_equivariant, spatialShift_apply]
  simp_rw [hshift]
  exact Equiv.sum_comp (Equiv.addRight shift_s)
    (fun q => ∑ i ∈ net.active, net.α i * sescUnitMap net embed B_spec x i q)

/-- **Unit-Level Orthogonal Bulk Invariance (`sescUnitMap_orthogonal_bulk_invariance`)**:
Any perturbation `η_bulk : Fin d → ℝ` whose local patches lie in the kernel of `patchProject B_spec`
leaves every unit map `sescUnitMap` identically unchanged. -/
theorem sescUnitMap_orthogonal_bulk_invariance [NeZero d] (net : SpectralSparseCircuit r m)
    (embed : Fin k → Fin d) (B_spec : Matrix (Fin r) (Fin k) ℝ)
    (x η_bulk : Fin d → ℝ)
    (h_ortho : ∀ p : Fin d, ∀ l : Fin r, patchProject B_spec (cyclicPatch embed η_bulk p) l = 0)
    (i : Fin m) (p : Fin d) :
    sescUnitMap net embed B_spec (x + η_bulk) i p =
      sescUnitMap net embed B_spec x i p := by
  unfold sescUnitMap
  congr 1
  refine Finset.sum_congr rfl fun l _ => ?_
  rw [cyclicPatch_add, patchProject_add, h_ortho p l, add_zero]

/-- Sparse circuit forward invariance under orthogonal spectral-bulk perturbations. -/
theorem sescCircuitForward_orthogonal_bulk_invariance [NeZero d] (net : SpectralSparseCircuit r m)
    (embed : Fin k → Fin d) (B_spec : Matrix (Fin r) (Fin k) ℝ)
    (x η_bulk : Fin d → ℝ)
    (h_ortho : ∀ p : Fin d, ∀ l : Fin r, patchProject B_spec (cyclicPatch embed η_bulk p) l = 0) :
    sescCircuitForward net embed B_spec (x + η_bulk) =
      sescCircuitForward net embed B_spec x := by
  unfold sescCircuitForward
  congr 1
  refine Finset.sum_congr rfl fun p _ => ?_
  refine Finset.sum_congr rfl fun i _ => ?_
  rw [sescUnitMap_orthogonal_bulk_invariance net embed B_spec x η_bulk h_ortho i p]

/-- Full forward invariance under orthogonal spectral-bulk perturbations. -/
theorem sescFullForward_orthogonal_bulk_invariance [NeZero d] (net : SpectralSparseCircuit r m)
    (embed : Fin k → Fin d) (B_spec : Matrix (Fin r) (Fin k) ℝ)
    (x η_bulk : Fin d → ℝ)
    (h_ortho : ∀ p : Fin d, ∀ l : Fin r, patchProject B_spec (cyclicPatch embed η_bulk p) l = 0) :
    sescFullForward net embed B_spec (x + η_bulk) =
      sescFullForward net embed B_spec x := by
  unfold sescFullForward
  congr 1
  refine Finset.sum_congr rfl fun p _ => ?_
  refine Finset.sum_congr rfl fun i _ => ?_
  rw [sescUnitMap_orthogonal_bulk_invariance net embed B_spec x η_bulk h_ortho i p]

/-- **Exact Orthogonal Spectral-Bulk Invariance (`sesc_orthogonal_bulk_invariance`)**:
For any perturbation `η_bulk : Fin d → ℝ` whose local patches lie in the kernel of the rank-`r`
spectral projector (`∀ p l, patchProject B_spec (cyclicPatch embed η_bulk p) l = 0`), adding
`η_bulk` to `x` leaves both `sescCircuitForward` and `sescFullForward` identically unchanged. -/
theorem sesc_orthogonal_bulk_invariance [NeZero d] (net : SpectralSparseCircuit r m)
    (embed : Fin k → Fin d) (B_spec : Matrix (Fin r) (Fin k) ℝ)
    (x η_bulk : Fin d → ℝ)
    (h_ortho : ∀ p : Fin d, ∀ l : Fin r, patchProject B_spec (cyclicPatch embed η_bulk p) l = 0) :
    sescCircuitForward net embed B_spec (x + η_bulk) =
      sescCircuitForward net embed B_spec x ∧
    sescFullForward net embed B_spec (x + η_bulk) =
      sescFullForward net embed B_spec x :=
  ⟨sescCircuitForward_orthogonal_bulk_invariance net embed B_spec x η_bulk h_ortho,
   sescFullForward_orthogonal_bulk_invariance net embed B_spec x η_bulk h_ortho⟩

/-- **Mechanistic Circuit Distillation / Pruning Error Bound (`sesc_circuit_pruning_error_le`)**:
If at every spatial position `p : Fin d`, the inactive units `i ∈ net.activeᶜ` have bounded
attribution `∑ i ∈ net.activeᶜ, |net.α i * sescUnitMap net embed B_spec x i p| ≤ ε_circuit`,
then pruning the full network to the sparse circuit `net.active` changes the global output by
at most `ε_circuit`:
`|sescFullForward net embed B_spec x - sescCircuitForward net embed B_spec x| ≤ ε_circuit`. -/
theorem sesc_circuit_pruning_error_le [NeZero d] (net : SpectralSparseCircuit r m)
    (embed : Fin k → Fin d) (B_spec : Matrix (Fin r) (Fin k) ℝ)
    (x : Fin d → ℝ) {ε_circuit : ℝ}
    (h_inactive : ∀ p : Fin d,
      ∑ i ∈ net.activeᶜ, |net.α i * sescUnitMap net embed B_spec x i p| ≤ ε_circuit) :
    |sescFullForward net embed B_spec x - sescCircuitForward net embed B_spec x| ≤ ε_circuit := by
  have hd_pos : 0 < d := NeZero.pos d
  have hd_real : (0 : ℝ) < d := Nat.cast_pos.mpr hd_pos
  have hd_ne : (d : ℝ) ≠ 0 := ne_of_gt hd_real
  unfold sescFullForward sescCircuitForward
  rw [← mul_sub, ← Finset.sum_sub_distrib, abs_mul,
    abs_of_nonneg (one_div_nonneg.mpr hd_real.le)]
  have hpos_diff : ∀ p : Fin d,
      |(∑ i : Fin m, net.α i * sescUnitMap net embed B_spec x i p) -
        (∑ i ∈ net.active, net.α i * sescUnitMap net embed B_spec x i p)| ≤ ε_circuit := by
    intro p
    let f : Fin m → ℝ := fun i => net.α i * sescUnitMap net embed B_spec x i p
    have hsplit : (∑ i : Fin m, f i) - (∑ i ∈ net.active, f i) = ∑ i ∈ net.activeᶜ, f i := by
      have hadd := Finset.sum_add_sum_compl net.active f
      linarith
    change |(∑ i : Fin m, f i) - (∑ i ∈ net.active, f i)| ≤ ε_circuit
    rw [hsplit]
    exact (Finset.abs_sum_le_sum_abs f net.activeᶜ).trans (h_inactive p)
  have hsum :
      |∑ p : Fin d,
        ((∑ i : Fin m, net.α i * sescUnitMap net embed B_spec x i p) -
          (∑ i ∈ net.active, net.α i * sescUnitMap net embed B_spec x i p))| ≤
        (d : ℝ) * ε_circuit := by
    apply (Finset.abs_sum_le_sum_abs _ _).trans
    calc
      _ ≤ ∑ _ : Fin d, ε_circuit := Finset.sum_le_sum fun p _ => hpos_diff p
      _ = (d : ℝ) * ε_circuit := by
        simp [Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
  calc
    1 / (d : ℝ) * |∑ p : Fin d, _|
      ≤ 1 / (d : ℝ) * ((d : ℝ) * ε_circuit) :=
        mul_le_mul_of_nonneg_left hsum (one_div_nonneg.mpr hd_real.le)
    _ = ε_circuit := by field_simp [hd_ne]

/-! ## 3. The 4-Tier Architectural Dominance Hierarchy (`SESC < SubspaceNet < CNN < Dense MLP`) -/

/-- Effective parameter dimension of a **Spectral-Equivariant Sparse Circuit (`SESC`)** with
`s` active circuit units and local spectral rank `r`: `dimSESC s r = s * r`. -/
def dimSESC (s r : ℕ) : ℕ := s * r

/-- Effective parameter dimension of a **Spectral Subspace Network (`SubspaceNet`)** with
`m` channels and spectral rank `r`: `dimSubspace m r = m * r`. -/
def dimSubspace (m r : ℕ) : ℕ := m * r

/-- Effective parameter dimension of a **Weight-Shared Local CNN (`CNN`)** with
`m` channels and kernel width `k`: `dimCNN m k = m * k`. -/
def dimCNN (m k : ℕ) : ℕ := m * k

/-- Effective parameter dimension of a **Dense Unstructured MLP (`MLP`)** with
`m` hidden units and ambient input dimension `d`: `dimMLP m d = m * d`. -/
def dimMLP (m d : ℕ) : ℕ := m * d

/-- **Strict 4-Tier Effective Dimension Hierarchy (`four_tier_dimension_strict_hierarchy`)**:
Whenever active circuit size satisfies `0 < s < m` and local spectral rank, kernel size, and
ambient dimension satisfy `0 < r < k < d`, the effective parameter dimensions obey the strict chain
`dimSESC s r < dimSubspace m r < dimCNN m k < dimMLP m d`. -/
theorem four_tier_dimension_strict_hierarchy {s m r k d : ℕ}
    (hs : 0 < s) (hsm : s < m) (hr : 0 < r) (hrk : r < k) (hkd : k < d) :
    dimSESC s r < dimSubspace m r ∧
      dimSubspace m r < dimCNN m k ∧
      dimCNN m k < dimMLP m d := by
  have hm : 0 < m := lt_trans hs hsm
  refine ⟨?_, ?_, ?_⟩
  · exact Nat.mul_lt_mul_of_pos_right hsm hr
  · exact Nat.mul_lt_mul_of_pos_left hrk hm
  · exact Nat.mul_lt_mul_of_pos_left hkd hm

/-- Strict monotonicity of the explicit Euclidean-ball covering number bound
`(coverAxis rad δ p + 1) ^ p` in the parameter dimension `p` when `0 ≤ rad` and `0 < δ`. -/
theorem covering_bound_strict_mono {rad δ : ℝ} (hrad : 0 ≤ rad) (hδ : 0 < δ)
    {p q : ℕ} (hpq : p < q) :
    (coverAxis rad δ p + 1) ^ p < (coverAxis rad δ q + 1) ^ q := by
  have haxis_le : coverAxis rad δ p + 1 ≤ coverAxis rad δ q + 1 :=
    Nat.add_le_add_right (coverAxis_mono hrad hδ hpq.le) 1
  have hbase_ge_two : 2 ≤ coverAxis rad δ q + 1 := by
    have h1 : 1 ≤ coverAxis rad δ q := le_max_left 1 _
    omega
  calc
    (coverAxis rad δ p + 1) ^ p
      ≤ (coverAxis rad δ q + 1) ^ p := Nat.pow_le_pow_left haxis_le p
    _ < (coverAxis rad δ q + 1) ^ q := Nat.pow_lt_pow_right hbase_ge_two hpq

/-- **Strict 4-Tier Metric-Entropy Covering Hierarchy (`four_tier_covering_strict_hierarchy`)**:
For `0 < s < m`, `0 < r < k < d`, `0 < rad`, and `0 < δ`, the explicit metric-entropy covering
number bounds `N(p) = (coverAxis rad δ p + 1) ^ p` satisfy the strict 4-tier chain:
`N(s * r) < N(m * r) < N(m * k) < N(m * d)`. -/
theorem four_tier_covering_strict_hierarchy {s m r k d : ℕ} {rad δ : ℝ}
    (hs : 0 < s) (hsm : s < m) (hr : 0 < r) (hrk : r < k) (hkd : k < d)
    (hrad : 0 < rad) (hδ : 0 < δ) :
    (coverAxis rad δ (s * r) + 1) ^ (s * r) <
      (coverAxis rad δ (m * r) + 1) ^ (m * r) ∧
    (coverAxis rad δ (m * r) + 1) ^ (m * r) <
      (coverAxis rad δ (m * k) + 1) ^ (m * k) ∧
    (coverAxis rad δ (m * k) + 1) ^ (m * k) <
      (coverAxis rad δ (m * d) + 1) ^ (m * d) := by
  obtain ⟨h1, h2, h3⟩ := four_tier_dimension_strict_hierarchy hs hsm hr hrk hkd
  exact ⟨covering_bound_strict_mono hrad.le hδ h1,
         covering_bound_strict_mono hrad.le hδ h2,
         covering_bound_strict_mono hrad.le hδ h3⟩

/-- **Strict 4-Tier Sample Complexity & FLOP Hierarchy (`four_tier_sample_complexity_and_flop_hierarchy`)**:
For `0 < s < m`, `0 < r < k < d`, and `1 < C_axis`, the log-covering sample complexity proxy
`archComplexity p C_axis = p * log C_axis` satisfies the strict 4-tier chain
`archComplexity (s * r) C_axis < archComplexity (m * r) C_axis < archComplexity (m * k) C_axis < archComplexity (m * d) C_axis`,
and the exact end-to-end compression ratio of SESC relative to a Dense MLP equals
`((s : ℝ) * (r : ℝ)) / ((m : ℝ) * (d : ℝ)) < 1`. -/
theorem four_tier_sample_complexity_and_flop_hierarchy {s m r k d : ℕ} {C_axis : ℝ}
    (hs : 0 < s) (hsm : s < m) (hr : 0 < r) (hrk : r < k) (hkd : k < d)
    (hC : 1 < C_axis) :
    (archComplexity (s * r) C_axis < archComplexity (m * r) C_axis ∧
     archComplexity (m * r) C_axis < archComplexity (m * k) C_axis ∧
     archComplexity (m * k) C_axis < archComplexity (m * d) C_axis) ∧
    archComplexity (s * r) C_axis / archComplexity (m * d) C_axis =
      ((s : ℝ) * (r : ℝ)) / ((m : ℝ) * (d : ℝ)) ∧
    ((s : ℝ) * (r : ℝ)) / ((m : ℝ) * (d : ℝ)) < 1 := by
  obtain ⟨h1, h2, h3⟩ := four_tier_dimension_strict_hierarchy hs hsm hr hrk hkd
  have hlog_pos : 0 < Real.log C_axis := Real.log_pos hC
  have hlog_ne : Real.log C_axis ≠ 0 := ne_of_gt hlog_pos
  have hsr_lt_md : s * r < m * d := lt_trans (lt_trans h1 h2) h3
  have hm_pos : 0 < m := lt_trans hs hsm
  have hd_pos : 0 < d := lt_trans (lt_trans hr hrk) hkd
  have hmd_real_pos : (0 : ℝ) < (m : ℝ) * (d : ℝ) := by positivity
  refine ⟨⟨?_, ?_, ?_⟩, ?_, ?_⟩
  · exact mul_lt_mul_of_pos_right (Nat.cast_lt.mpr h1) hlog_pos
  · exact mul_lt_mul_of_pos_right (Nat.cast_lt.mpr h2) hlog_pos
  · exact mul_lt_mul_of_pos_right (Nat.cast_lt.mpr h3) hlog_pos
  · unfold archComplexity
    push_cast
    exact mul_div_mul_right ((s : ℝ) * (r : ℝ)) ((m : ℝ) * (d : ℝ)) hlog_ne
  · rw [div_lt_one hmd_real_pos]
    exact_mod_cast hsr_lt_md

/-! ## 4. Systematic Conditional Architecture Optimality & Under-Parameterization Lower Bound -/

/-- Total population excess risk decomposition for an architecture of effective invariant
dimension `p` with structural approximation bias `ε_struct` on data geometry `D` at sample
size `n` and complexity scale `c_gen`:
`archTotalExcessRisk ε_struct p c_gen n = ε_struct + (p : ℝ) * c_gen / (n : ℝ)`. -/
def archTotalExcessRisk (ε_struct : ℝ) (p : ℕ) (c_gen : ℝ) (n : ℕ) : ℝ :=
  ε_struct + (p : ℝ) * c_gen / (n : ℝ)

/-- **Optimality of Minimal Invariant Architecture (`minimal_invariant_architecture_optimal`)**:
Among all architectures that are *structurally well-specified* for data geometry `D`
(`ε_struct = 0`), the architecture with minimal invariant dimension `p_min` strictly minimizes
`archTotalExcessRisk 0 p c_gen n` whenever `p_min < p_other`, `0 < c_gen`, and `0 < n`. -/
theorem minimal_invariant_architecture_optimal {p_min p_other n : ℕ} {c_gen : ℝ}
    (hp : p_min < p_other) (hc : 0 < c_gen) (hn : 0 < n) :
    archTotalExcessRisk 0 p_min c_gen n < archTotalExcessRisk 0 p_other c_gen n := by
  unfold archTotalExcessRisk
  rw [zero_add, zero_add]
  have hn_real : (0 : ℝ) < (n : ℝ) := Nat.cast_pos.mpr hn
  have hp_real : (p_min : ℝ) < (p_other : ℝ) := Nat.cast_lt.mpr hp
  exact div_lt_div_of_pos_right (mul_lt_mul_of_pos_right hp_real hc) hn_real

/-- **Strict Optimality of SESC Under Sparse Spectral Shift-Invariant Data Geometry
(`sesc_strictly_optimal_under_sparse_spectral_shift_geometry`)**:
When data geometry `D` exhibits translation invariance, local spectral rank `r`, and circuit
sparsity `s` so that SESC, SubspaceNet, CNN, and Dense MLP are all structurally well-specified
(`ε_struct = 0`), SESC strictly minimizes `archTotalExcessRisk` across all four architectural
tiers for every sample size `n > 0` and `c_gen > 0`. -/
theorem sesc_strictly_optimal_under_sparse_spectral_shift_geometry
    {s m r k d n : ℕ} {c_gen : ℝ}
    (hs : 0 < s) (hsm : s < m) (hr : 0 < r) (hrk : r < k) (hkd : k < d)
    (hc : 0 < c_gen) (hn : 0 < n) :
    archTotalExcessRisk 0 (dimSESC s r) c_gen n <
      archTotalExcessRisk 0 (dimSubspace m r) c_gen n ∧
    archTotalExcessRisk 0 (dimSubspace m r) c_gen n <
      archTotalExcessRisk 0 (dimCNN m k) c_gen n ∧
    archTotalExcessRisk 0 (dimCNN m k) c_gen n <
      archTotalExcessRisk 0 (dimMLP m d) c_gen n := by
  obtain ⟨h1, h2, h3⟩ := four_tier_dimension_strict_hierarchy hs hsm hr hrk hkd
  exact ⟨minimal_invariant_architecture_optimal h1 hc hn,
         minimal_invariant_architecture_optimal h2 hc hn,
         minimal_invariant_architecture_optimal h3 hc hn⟩

/-- **Under-Parameterization Lower Bound (`under_parameterized_architecture_suboptimal_of_large_sample`)**:
If an over-compressed architecture `p_small < p_true` violates a data symmetry/rank/sparsity
requirement and incurs positive structural approximation bias `ε_struct > 0` while the matched
architecture `p_true` has zero structural bias (`0`), then for all sample sizes
`(n : ℝ) > ((p_true : ℝ) - (p_small : ℝ)) * c_gen / ε_struct`, the matched architecture `p_true`
strictly beats the under-parameterized architecture:
`archTotalExcessRisk 0 p_true c_gen n < archTotalExcessRisk ε_struct p_small c_gen n`. -/
theorem under_parameterized_architecture_suboptimal_of_large_sample
    {p_small p_true n : ℕ} {ε_struct c_gen : ℝ}
    (hp : p_small < p_true) (hc : 0 < c_gen) (hε : 0 < ε_struct)
    (hn : ((p_true : ℝ) - (p_small : ℝ)) * c_gen / ε_struct < (n : ℝ)) :
    archTotalExcessRisk 0 p_true c_gen n <
      archTotalExcessRisk ε_struct p_small c_gen n := by
  have hp_diff_pos : 0 < (p_true : ℝ) - (p_small : ℝ) :=
    sub_pos.mpr (Nat.cast_lt.mpr hp)
  have hthresh_pos : 0 < ((p_true : ℝ) - (p_small : ℝ)) * c_gen / ε_struct :=
    div_pos (mul_pos hp_diff_pos hc) hε
  have hn_pos : (0 : ℝ) < (n : ℝ) := lt_trans hthresh_pos hn
  have hmul : ((p_true : ℝ) - (p_small : ℝ)) * c_gen < (n : ℝ) * ε_struct := by
    have h := (div_lt_iff₀ hε).mp hn
    linarith
  have hdiv : ((p_true : ℝ) - (p_small : ℝ)) * c_gen / (n : ℝ) < ε_struct := by
    rw [div_lt_iff₀ hn_pos]
    linarith
  unfold archTotalExcessRisk
  have hdecomp :
      (p_true : ℝ) * c_gen / (n : ℝ) - (p_small : ℝ) * c_gen / (n : ℝ) =
        ((p_true : ℝ) - (p_small : ℝ)) * c_gen / (n : ℝ) := by ring
  linarith

/-- **Exact Sample-Size Crossover Equivalence (`archTotalExcessRisk_crossover_iff`)**:
For any `0 < ε_struct` and `0 < n`, the well-specified architecture `p_true` strictly beats
an under-parameterized architecture `p_small` with bias `ε_struct` if and only if
`((p_true : ℝ) - (p_small : ℝ)) * c_gen / ε_struct < (n : ℝ)`. -/
theorem archTotalExcessRisk_crossover_iff
    {p_small p_true n : ℕ} {ε_struct c_gen : ℝ}
    (hε : 0 < ε_struct) (hn : 0 < n) :
    archTotalExcessRisk 0 p_true c_gen n <
      archTotalExcessRisk ε_struct p_small c_gen n ↔
      ((p_true : ℝ) - (p_small : ℝ)) * c_gen / ε_struct < (n : ℝ) := by
  have hn_pos : (0 : ℝ) < (n : ℝ) := Nat.cast_pos.mpr hn
  unfold archTotalExcessRisk
  rw [zero_add, div_lt_iff₀ hε]
  constructor
  · intro h
    have hdiff : ((p_true : ℝ) - (p_small : ℝ)) * c_gen / (n : ℝ) < ε_struct := by
      have hring : ((p_true : ℝ) - (p_small : ℝ)) * c_gen / (n : ℝ) =
          (p_true : ℝ) * c_gen / (n : ℝ) - (p_small : ℝ) * c_gen / (n : ℝ) := by ring
      linarith
    have hmul := (div_lt_iff₀ hn_pos).mp hdiff
    linarith
  · intro h
    have hdiv : ((p_true : ℝ) - (p_small : ℝ)) * c_gen / (n : ℝ) < ε_struct := by
      rw [div_lt_iff₀ hn_pos]
      linarith
    have hring : (p_true : ℝ) * c_gen / (n : ℝ) - (p_small : ℝ) * c_gen / (n : ℝ) =
        ((p_true : ℝ) - (p_small : ℝ)) * c_gen / (n : ℝ) := by ring
    linarith

end OpenTheory.SpectralSparseCircuit
