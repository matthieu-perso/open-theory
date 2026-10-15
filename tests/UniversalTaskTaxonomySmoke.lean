import OpenTheory.Generalization.UniversalTaskTaxonomy

noncomputable section

open OpenTheory
open OpenTheory.SpectralSparseCircuit
open OpenTheory.ResidualPreconditionedMomentum
open OpenTheory.UniversalTaskTaxonomy

-- Section 1: Universal Task-Structure Taxonomy & Modality-Optimal Architecture Dispatch
#check @TaskModality
#check @UniversalTaskDescriptor
#check @dimConv
#check @dimLowRankTransformer
#check @dimLowRankMLP
#check @matchedArchDim
#check @denseAmbientDim
#check @archGeneralizationPenalty
#check @totalArchitectureRisk
#check @isotropicSubspaceBias
#check @matched_architecture_strict_dominance_on_structured_tasks
#check @no_free_lunch_unstructured_conservation
#check @universal_variational_architecture_minimum

-- Section 2: Universal Optimizer Optimality Across All Spectral Curvature & Noise Regimes
#check @rpm_global_maximum_over_all_step_vectors
#check @rpm_reduces_to_scalar_sgd_iff_isotropic

-- Section 3: Master Universal Task-to-(Architecture, Optimizer) Co-Design Theorem
#check @universal_task_architecture_optimizer_optimality

-- Concrete numerical example 1: Evaluating `matchedArchDim` vs `denseAmbientDim` across all 5
-- `TaskModality` constructors (`SpatialGrid`, `SequentialContext`, `HierarchicalTree`,
-- `SparseSpectral`, `UnstructuredAmbient`) with m = 8, d = 16, k_patch = 5, r_rank = 3, s_active = 2:
--   SpatialGrid         => matchedArchDim = 8 * 5  = 40  < denseAmbientDim = 8 * 16 = 128
--   SequentialContext   => matchedArchDim = 8 * 3  = 24  < denseAmbientDim = 128
--   HierarchicalTree    => matchedArchDim = 8 * 3  = 24  < denseAmbientDim = 128
--   SparseSpectral      => matchedArchDim = 2 * 3  = 6   < denseAmbientDim = 128
--   UnstructuredAmbient => matchedArchDim = 8 * 16 = 128 = denseAmbientDim = 128
example :
    let T_spatial : UniversalTaskDescriptor := ⟨.SpatialGrid, 8, 16, 5, 3, 2⟩
    let T_seq     : UniversalTaskDescriptor := ⟨.SequentialContext, 8, 16, 5, 3, 2⟩
    let T_hier    : UniversalTaskDescriptor := ⟨.HierarchicalTree, 8, 16, 5, 3, 2⟩
    let T_sparse  : UniversalTaskDescriptor := ⟨.SparseSpectral, 8, 16, 5, 3, 2⟩
    let T_unstr   : UniversalTaskDescriptor := ⟨.UnstructuredAmbient, 8, 16, 5, 3, 2⟩
    matchedArchDim T_spatial = 40 ∧ denseAmbientDim T_spatial = 128 ∧
    matchedArchDim T_spatial < denseAmbientDim T_spatial ∧
    matchedArchDim T_seq = 24 ∧ denseAmbientDim T_seq = 128 ∧
    matchedArchDim T_seq < denseAmbientDim T_seq ∧
    matchedArchDim T_hier = 24 ∧ denseAmbientDim T_hier = 128 ∧
    matchedArchDim T_hier < denseAmbientDim T_hier ∧
    matchedArchDim T_sparse = 6 ∧ denseAmbientDim T_sparse = 128 ∧
    matchedArchDim T_sparse < denseAmbientDim T_sparse ∧
    matchedArchDim T_unstr = 128 ∧ denseAmbientDim T_unstr = 128 ∧
    matchedArchDim T_unstr = denseAmbientDim T_unstr := by
  decide

-- Concrete numerical example 2: Finite-sample generalization penalty and total architecture risk
-- comparison across structured modalities and No-Free-Lunch unstructured conservation.
example :
    let T_sparse : UniversalTaskDescriptor := ⟨.SparseSpectral, 8, 16, 5, 3, 2⟩
    let T_unstr  : UniversalTaskDescriptor := ⟨.UnstructuredAmbient, 8, 16, 5, 3, 2⟩
    archGeneralizationPenalty (matchedArchDim T_sparse) 1 100 = 3 / 50 ∧
    archGeneralizationPenalty (denseAmbientDim T_sparse) 1 100 = 32 / 25 ∧
    archGeneralizationPenalty (matchedArchDim T_sparse) 1 100 <
      archGeneralizationPenalty (denseAmbientDim T_sparse) 1 100 ∧
    isotropicSubspaceBias (denseAmbientDim T_unstr) 64 (1 / 10) = 32 / 5 ∧
    totalArchitectureRisk 0 0 (matchedArchDim T_unstr) 1 100 <
      totalArchitectureRisk 0 (isotropicSubspaceBias (denseAmbientDim T_unstr) 64 (1 / 10)) 64 1 100 := by
  norm_num [matchedArchDim, denseAmbientDim, dimSESC, dimMLP,
    archGeneralizationPenalty, isotropicSubspaceBias, totalArchitectureRisk]
