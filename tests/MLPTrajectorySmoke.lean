import OpenTheory.Dynamics.MLPTrajectory

open OpenTheory

-- Zero displacement must retain the original prediction and have zero remainder.
example {d m : ℕ} (net : TwoLayerNet d m) (x : Fin d → ℝ) :
    MLPTrajectory.linearizedForward net 0 x = net.forward x := by
  simp [MLPTrajectory.linearizedForward, MLPTrajectory.preShift]

#check MLPTrajectory.forward_remainder_le
#check MLPTrajectory.regularized_step_eq
#check MLPTrajectory.iterate_remainder_le

-- λ=0 recovers the existing unregularized update, without moving the anchor.
example {d m : ℕ} (net : TwoLayerNet d m) (x : Fin d → ℝ)
    (y η : ℝ) (U : Matrix (Fin m) (Fin d) ℝ) (i : Fin m) (j : Fin d) :
    net.W₁ i j + MLPTrajectory.regularizedStep net x y η 0 U i j =
      FeatureStep.firstLayerStep (MLPTrajectory.atDisplacement net U) x y η i j := by
  simpa using MLPTrajectory.regularized_step_eq net x y η 0 U i j

-- Unit input and unit bounds give the hand-computed remainder m T².
example {m : ℕ} (net : TwoLayerNet 1 m) (y : ℝ) (T : ℕ)
    (hW : ∀ i, |net.W₂ 0 i| ≤ 1) (hσ : ∀ u, |deriv net.act.σ u| ≤ 1)
    (hLip : ∀ a b, |deriv net.act.σ a - deriv net.act.σ b| ≤ |a - b|)
    (hres : ∀ t < T, |FeatureStep.outputResidual
      (MLPTrajectory.atDisplacement net (MLPTrajectory.trajectory net (fun _ => 1) y 1 1 t))
      (fun _ => 1) y| ≤ 1) :
    |(MLPTrajectory.atDisplacement net
      (MLPTrajectory.trajectory net (fun _ => 1) y 1 1 T)).forward (fun _ => 1) -
      MLPTrajectory.linearizedForward net
        (MLPTrajectory.trajectory net (fun _ => 1) y 1 1 T) (fun _ => 1)| ≤
      (m : ℝ) * (T : ℝ) ^ 2 := by
  simpa [FeatureStep.inputEnergy] using MLPTrajectory.iterate_remainder_le
    net (fun _ => 1) y 1 1 (A := 1) (L := 1) (R := 1) (M := 1)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) hW hσ (by simpa using hLip) T hres
