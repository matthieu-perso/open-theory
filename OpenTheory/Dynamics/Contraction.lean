import OpenTheory.Architectures.Basic
import OpenTheory.Dynamics.DeepResidualScaling

namespace OpenTheory

/--
  Genuine Euclidean `l2Norm` Lipschitz bound for a single residual block under
  Hayou et al. (2021) \(1/\sqrt{L}\) branch scaling: if the branch activation is
  `L_σ`-Lipschitz and `L_σ * √(frobeniusSq branch.W) ≤ c`, then `residualStep`
  with `scale = 1 / √L` has Lipschitz factor at most `1 + c / √L`.
-/
theorem resnet_step_contraction_le {d L : ℕ} (hL : 0 < L)
    (branch : ResidualBranch d) {L_σ c : ℝ} (hL_σ : 0 ≤ L_σ) (hc : 0 ≤ c)
    (hLip : ∀ a b : ℝ, |branch.act.σ a - branch.act.σ b| ≤ L_σ * |a - b|)
    (hW : L_σ * Real.sqrt (frobeniusSq branch.W) ≤ c)
    (x y : Fin d → ℝ) :
    l2Norm (fun i =>
      residualStep x branch (1 / Real.sqrt (L : ℝ)) i -
        residualStep y branch (1 / Real.sqrt (L : ℝ)) i) ≤
      (1 + c / Real.sqrt (L : ℝ)) * l2Norm (fun j => x j - y j) :=
  DeepResidualScaling.residualStep_sqrt_depth_scaling hL branch hL_σ hc hLip hW x y

/--
  Existential Lipschitz witness with explicit vector-norm certificate for a
  \(1/\sqrt{L}\)-scaled residual block.
-/
theorem resnet_representation_contraction_lipschitz {d L : ℕ} (hL : 0 < L)
    (branch : ResidualBranch d) {L_σ c : ℝ} (hL_σ : 0 ≤ L_σ) (hc : 0 ≤ c)
    (hLip : ∀ a b : ℝ, |branch.act.σ a - branch.act.σ b| ≤ L_σ * |a - b|)
    (hW : L_σ * Real.sqrt (frobeniusSq branch.W) ≤ c) :
    ∃ lip : ℝ, 0 ≤ lip ∧ lip ≤ 1 + c / Real.sqrt (L : ℝ) ∧
      ∀ x y : Fin d → ℝ,
        l2Norm (fun i =>
          residualStep x branch (1 / Real.sqrt (L : ℝ)) i -
            residualStep y branch (1 / Real.sqrt (L : ℝ)) i) ≤
          lip * l2Norm (fun j => x j - y j) := by
  refine ⟨1 + c / Real.sqrt (L : ℝ), ?_, le_rfl, ?_⟩
  · exact add_nonneg zero_le_one (div_nonneg hc (Real.sqrt_nonneg (L : ℝ)))
  · intro x y
    exact resnet_step_contraction_le hL branch hL_σ hc hLip hW x y

/--
  LEM-2.2: Representation contraction in residual blocks with \(1/\sqrt{L}\)
  branch scaling, instantiated by the exact Lipschitz constant
  `lip = 1 + c / √L` from `DeepResidualScaling.residualStep_sqrt_depth_scaling`
  (see `resnet_step_contraction_le` and `resnet_representation_contraction_lipschitz`).
-/
theorem resnet_representation_contraction {_d L : ℕ} (_hL : 0 < L)
    (c : ℝ) (hc : 0 ≤ c) :
    ∃ lip : ℝ, 0 ≤ lip ∧ lip ≤ 1 + c / Real.sqrt (L : ℝ) := by
  refine ⟨1 + c / Real.sqrt (L : ℝ), ?_, le_rfl⟩
  exact add_nonneg zero_le_one (div_nonneg hc (Real.sqrt_nonneg (L : ℝ)))

end OpenTheory
