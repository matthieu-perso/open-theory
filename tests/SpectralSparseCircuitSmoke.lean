import OpenTheory.Architectures.SpectralSparseCircuit

noncomputable section

open OpenTheory OpenTheory.ConvolutionalGeneralization OpenTheory.BallCover
open OpenTheory.SpectralSparseCircuit
open scoped BigOperators

-- 1. The Spectral-Equivariant Sparse Circuit (SESC) Architecture
#check @patchProject
#check @patchProject_apply
#check @patchProject_add
#check @patchProject_zero
#check @cyclicPatch_add
#check @SpectralSparseCircuit
#check @SESC
#check @sescUnitMap
#check @sescFullForward
#check @sescCircuitForward

-- 2. Exact Structural Invariances & Mechanistic Circuit Distillation Bound
#check @sescUnitMap_shift_equivariant
#check @sescUnitMap_shift_equivariant_fun
#check @sescFullForward_shift_invariant
#check @sescCircuitForward_shift_invariant
#check @sescUnitMap_orthogonal_bulk_invariance
#check @sescCircuitForward_orthogonal_bulk_invariance
#check @sescFullForward_orthogonal_bulk_invariance
#check @sesc_orthogonal_bulk_invariance
#check @sesc_circuit_pruning_error_le

-- 3. The 4-Tier Architectural Dominance Hierarchy (SESC < SubspaceNet < CNN < Dense MLP)
#check @dimSESC
#check @dimSubspace
#check @dimCNN
#check @dimMLP
#check @four_tier_dimension_strict_hierarchy
#check @covering_bound_strict_mono
#check @four_tier_covering_strict_hierarchy
#check @four_tier_sample_complexity_and_flop_hierarchy

-- 4. Systematic Conditional Architecture Optimality & Under-Parameterization Lower Bound
#check @archTotalExcessRisk
#check @minimal_invariant_architecture_optimal
#check @sesc_strictly_optimal_under_sparse_spectral_shift_geometry
#check @under_parameterized_architecture_suboptimal_of_large_sample
#check @archTotalExcessRisk_crossover_iff

-- Smoke test 1: Exact translation invariance of sparse circuit output on Fin 16 with kernel k = 5, rank r = 2.
example (net : SpectralSparseCircuit 2 6) (embed : Fin 5 → Fin 16)
    (B_spec : Matrix (Fin 2) (Fin 5) ℝ) (shift_s : Fin 16) (x : Fin 16 → ℝ) :
    sescCircuitForward net embed B_spec (spatialShift shift_s x) =
      sescCircuitForward net embed B_spec x :=
  sescCircuitForward_shift_invariant net embed B_spec shift_s x

-- Smoke test 2: Orthogonal spectral-bulk perturbation leaves both sparse and full outputs unchanged.
example (net : SpectralSparseCircuit 2 6) (embed : Fin 5 → Fin 16)
    (B_spec : Matrix (Fin 2) (Fin 5) ℝ) (x η_bulk : Fin 16 → ℝ)
    (h_ortho : ∀ p : Fin 16, ∀ l : Fin 2, patchProject B_spec (cyclicPatch embed η_bulk p) l = 0) :
    sescCircuitForward net embed B_spec (x + η_bulk) = sescCircuitForward net embed B_spec x ∧
    sescFullForward net embed B_spec (x + η_bulk) = sescFullForward net embed B_spec x :=
  sesc_orthogonal_bulk_invariance net embed B_spec x η_bulk h_ortho

-- Smoke test 3: Concrete 4-tier covering hierarchy for s = 2, m = 6, r = 3, k = 5, d = 16.
example :
    (coverAxis 1 1 (2 * 3) + 1) ^ (2 * 3) < (coverAxis 1 1 (6 * 3) + 1) ^ (6 * 3) ∧
    (coverAxis 1 1 (6 * 3) + 1) ^ (6 * 3) < (coverAxis 1 1 (6 * 5) + 1) ^ (6 * 5) ∧
    (coverAxis 1 1 (6 * 5) + 1) ^ (6 * 5) < (coverAxis 1 1 (6 * 16) + 1) ^ (6 * 16) :=
  four_tier_covering_strict_hierarchy (s := 2) (m := 6) (r := 3) (k := 5) (d := 16)
    (by decide) (by decide) (by decide) (by decide) (by decide) (by norm_num) (by norm_num)

-- Smoke test 4: Concrete bidirectional optimality (minimal invariant vs. under-parameterized).
example :
    archTotalExcessRisk 0 (dimSESC 2 3) 1 100 < archTotalExcessRisk 0 (dimMLP 6 16) 1 100 ∧
    archTotalExcessRisk 0 (dimSESC 2 3) 1 100 < archTotalExcessRisk (1 / 2) 2 1 100 := by
  refine ⟨?_, ?_⟩
  · exact minimal_invariant_architecture_optimal (by decide) (by norm_num) (by decide)
  · apply under_parameterized_architecture_suboptimal_of_large_sample
      (by decide) (by norm_num) (by norm_num)
    norm_num [dimSESC]
