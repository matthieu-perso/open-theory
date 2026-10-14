import OpenTheory.Generalization.MLPRisk

open OpenTheory

-- Contract: the nonlinear guarantee must not require a residual/gap hypothesis.
#check MLPResidual.residual_le
#check MLPRisk.expected_mse

-- With zero activation curvature, only optimization error remains.
example (T : ℕ) (m : ℕ) (η A L R E K q S : ℝ) :
    MLPRisk.gap T m η A L R 0 E K q S = q ^ T * S := by
  simp [MLPRisk.gap]

-- A hand-computable centered noise law: bias²=9, variance contribution=1.
example : (∑ w : Fin 2, (1 / 2 : ℝ) * (3 + (1 / 2 : ℝ) * ![-2, 2] w) ^ 2) = 10 := by
  have h := MLPRisk.affine_noise_mse (fun _ : Fin 2 => (1 / 2 : ℝ)) ![-2, 2]
    (by norm_num [Fin.sum_univ_two]) (by norm_num [Fin.sum_univ_two])
    2 (by norm_num [Fin.sum_univ_two]) 3 (1 / 2)
  convert h using 1
  norm_num

private def riskNet : TwoLayerNet 1 1 where
  W₁ := fun _ _ => 1
  W₂ := fun _ _ => 1
  act := ⟨id, differentiable_id, ⟨1, fun _ _ => by simp⟩⟩

-- End-to-end application: nonzero initialization, nonzero target, ±1 noise.
-- Removing centering or adding an unproved trajectory hypothesis breaks this.
example :
    (∑ w : Fin 2, (1 / 2 : ℝ) *
      ((MLPTrajectory.atDisplacement riskNet
        (MLPTrajectory.trajectory riskNet (fun _ => 1) (2 + ![-1, 1] w)
          (1 / 2) 1 1)).forward (fun _ => 1) - 2) ^ 2) ≤ 1 := by
  have hK : MLPComparison.kernel riskNet (fun _ => 1) = 1 := by
    norm_num [MLPComparison.kernel, riskNet, FeatureStep.inputEnergy]
  have hf : riskNet.forward (fun _ => 1) = 1 := by
    norm_num [TwoLayerNet.forward, riskNet]
  have h := MLPRisk.expected_mse riskNet (fun _ => 1) 2 (1 / 2) 1
    (A := 1) (L := 1) (M := 0) (Z := 1)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num [hK])
    (by simp [riskNet]) (by simp [riskNet]) (by simp [riskNet])
    (fun _ : Fin 2 => (1 / 2 : ℝ)) ![-1, 1]
    (by intro w; norm_num) (by norm_num [Fin.sum_univ_two])
    (by norm_num [Fin.sum_univ_two]) 1 (by norm_num [Fin.sum_univ_two])
    (by intro w; fin_cases w <;> norm_num) 1
  norm_num [MLPRisk.gap, hK, hf] at h ⊢
  exact h
