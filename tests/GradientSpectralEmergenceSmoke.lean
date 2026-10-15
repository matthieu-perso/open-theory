import OpenTheory.Dynamics.GradientSpectralEmergence

open OpenTheory
open OpenTheory.MultiStepSpectralBias
open OpenTheory.SpikedSampleComplexity
open OpenTheory.GradientSpectralEmergence

-- 1. Raw First-Layer Weight Gradient Update Driven by the Empirical Data Gram Matrix
#check GradientSpectralEmergence.vecDot
#check GradientSpectralEmergence.vecDot_comm
#check GradientSpectralEmergence.vecDot_add_left
#check GradientSpectralEmergence.vecDot_smul_left
#check GradientSpectralEmergence.vecDot_self_nonneg
#check GradientSpectralEmergence.gramAct
#check GradientSpectralEmergence.rawWeightGradStep
#check GradientSpectralEmergence.rawWeightTrajectory
#check GradientSpectralEmergence.rawWeightTrajectory_zero
#check GradientSpectralEmergence.rawWeightTrajectory_succ

-- 2. Exact Eigenmode Power-Iteration Law from Raw Weight Gradients
#check GradientSpectralEmergence.IsSymmetricGram
#check GradientSpectralEmergence.IsGramEigenpair
#check GradientSpectralEmergence.vecDot_gramAct_symm
#check GradientSpectralEmergence.vecDot_gramAct_eigenpair
#check GradientSpectralEmergence.rawWeightGradStep_eigenmode_step
#check GradientSpectralEmergence.rawWeightTrajectory_eigenmode_eq

-- 3. Multi-Mode Subspace Energy Evolution Derived from Raw Weight Updates
#check GradientSpectralEmergence.modalSubspaceEnergy
#check GradientSpectralEmergence.modalSubspaceEnergy_nonneg
#check GradientSpectralEmergence.modalSubspaceEnergy_trajectory_eq
#check GradientSpectralEmergence.signal_subspace_energy_lower_bound_from_gradients
#check GradientSpectralEmergence.bulk_subspace_energy_upper_bound_from_gradients

-- 4. First-Principles Discharge of MultiStepSpectralBias & SpikedSampleComplexity
#check GradientSpectralEmergence.first_principles_gradient_signal_to_bulk_ratio
#check GradientSpectralEmergence.first_principles_gradient_alignment_convergence

-- Concrete numerical verification: 2D diagonal Gram matrix with leading eigenvalue 4 and bulk 1
section ConcreteExample

def G₂ : Matrix (Fin 2) (Fin 2) ℝ :=
  fun i j => if i = j then (if i = 0 then 4 else 1) else 0

lemma G₂_symm : IsSymmetricGram G₂ := by
  intro i j
  unfold G₂
  by_cases hij : i = j
  · subst hij; rfl
  · have hji : j ≠ i := Ne.symm hij
    simp [hij, hji]

def e₀ : Fin 2 → ℝ := fun i => if i = 0 then 1 else 0

lemma e₀_eigen : IsGramEigenpair G₂ e₀ 4 := by
  intro i
  unfold gramAct G₂ e₀
  rw [Fin.sum_univ_two]
  fin_cases i <;> norm_num

def w_init : Fin 2 → ℝ := fun _ => 1

example :
    vecDot (rawWeightTrajectory (1 / 2) 1 G₂ w_init 2) e₀ = 9 := by
  have heq := (rawWeightTrajectory_eigenmode_eq (1 / 2) 1 G₂_symm e₀_eigen w_init 2).2.1
  rw [heq]
  unfold vecDot w_init e₀
  rw [Fin.sum_univ_two]
  norm_num

end ConcreteExample
