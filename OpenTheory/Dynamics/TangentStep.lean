import OpenTheory.Dynamics.FeatureStep
import Mathlib.Analysis.Calculus.Deriv.MeanValue
import Mathlib.Tactic.FieldSimp

/-! One gradient step against its tangent update.

`FeatureStep.preactivation_step` is the exact hidden-unit move. The network
output after that step equals the first-order (tangent) update plus the
activation's remainder. If the derivative of `σ` is `M`-Lipschitz, the mean
value theorem bounds that remainder by `M` times the squared preactivation
move. The constant `M` is the mean-value bound; the integral form improves it
to `M/2`.

Under the checked NTK initialization the output gap is
`O(|f(x)-y|² / √m)`. Under the checked μP initialization it is
`O(|f(x)-y|² / m²)`. Both vanish as the width grows, so one step stays close
to the tangent step. The feature-learning rate `η₁ = m` leaves an output gap
of order `|f(x)-y|²`, which does not vanish.

This is one sample and one step. It does not identify the tangent step with
the kernel-ridge solution, and it does not bound a trained network's risk.
-/

noncomputable section

open scoped BigOperators

open Set

namespace OpenTheory.TangentStep

variable {d m : ℕ}

/-- Output predicted by replacing each activation with its tangent at the
pre-step preactivation. -/
def tangentOutput (net : TwoLayerNet d m) (x : Fin d → ℝ) (y lr1 : ℝ) : ℝ :=
  net.forward x + ∑ i, net.W₂ 0 i * deriv net.act.σ (FeatureStep.preactivation net x i) *
    (FeatureStep.preactivation (FeatureStep.afterFirstLayerStep net x y lr1) x i -
      FeatureStep.preactivation net x i)

theorem forward_eq_sigma_pre (net : TwoLayerNet d m) (x : Fin d → ℝ) :
    net.forward x = ∑ i, net.W₂ 0 i * net.act.σ (FeatureStep.preactivation net x i) := by
  simp [TwoLayerNet.forward, FeatureStep.preactivation]

theorem tangentUpdate_eq (net : TwoLayerNet d m) (x : Fin d → ℝ) (y lr1 : ℝ) :
    (∑ i, net.W₂ 0 i * deriv net.act.σ (FeatureStep.preactivation net x i) *
        (FeatureStep.preactivation (FeatureStep.afterFirstLayerStep net x y lr1) x i -
          FeatureStep.preactivation net x i)) =
      -lr1 * FeatureStep.outputResidual net x y * FeatureStep.inputEnergy x *
        ∑ i, (net.W₂ 0 i * deriv net.act.σ (FeatureStep.preactivation net x i)) ^ 2 := by
  have hterm : ∀ i,
      net.W₂ 0 i * deriv net.act.σ (FeatureStep.preactivation net x i) *
          (FeatureStep.preactivation (FeatureStep.afterFirstLayerStep net x y lr1) x i -
            FeatureStep.preactivation net x i) =
        -lr1 * FeatureStep.outputResidual net x y * FeatureStep.inputEnergy x *
          (net.W₂ 0 i * deriv net.act.σ (FeatureStep.preactivation net x i)) ^ 2 := by
    intro i
    rw [FeatureStep.preactivation_step]
    ring
  simp_rw [hterm, ← Finset.mul_sum]

/-- The step output minus the tangent output is the sum of activation remainders. -/
theorem forward_after_sub_tangent (net : TwoLayerNet d m) (x : Fin d → ℝ) (y lr1 : ℝ) :
    (FeatureStep.afterFirstLayerStep net x y lr1).forward x - tangentOutput net x y lr1 =
      ∑ i, net.W₂ 0 i *
        (net.act.σ (FeatureStep.preactivation (FeatureStep.afterFirstLayerStep net x y lr1) x i) -
          net.act.σ (FeatureStep.preactivation net x i) -
          deriv net.act.σ (FeatureStep.preactivation net x i) *
            (FeatureStep.preactivation (FeatureStep.afterFirstLayerStep net x y lr1) x i -
              FeatureStep.preactivation net x i)) := by
  unfold tangentOutput
  rw [forward_eq_sigma_pre (FeatureStep.afterFirstLayerStep net x y lr1) x,
    forward_eq_sigma_pre net x]
  have hW : (FeatureStep.afterFirstLayerStep net x y lr1).W₂ = net.W₂ := rfl
  have hact : (FeatureStep.afterFirstLayerStep net x y lr1).act = net.act := rfl
  rw [hW, hact, ← sub_sub, ← Finset.sum_sub_distrib, ← Finset.sum_sub_distrib]
  refine Finset.sum_congr rfl fun _ _ => ?_
  ring

theorem abs_forward_after_sub_tangent (net : TwoLayerNet d m) (x : Fin d → ℝ) (y lr1 : ℝ) :
    |(FeatureStep.afterFirstLayerStep net x y lr1).forward x - tangentOutput net x y lr1| ≤
      ∑ i, |net.W₂ 0 i| *
        |net.act.σ (FeatureStep.preactivation (FeatureStep.afterFirstLayerStep net x y lr1) x i) -
          net.act.σ (FeatureStep.preactivation net x i) -
          deriv net.act.σ (FeatureStep.preactivation net x i) *
            (FeatureStep.preactivation (FeatureStep.afterFirstLayerStep net x y lr1) x i -
              FeatureStep.preactivation net x i)| := by
  rw [forward_after_sub_tangent]
  refine (Finset.abs_sum_le_sum_abs _ _).trans ?_
  refine Finset.sum_le_sum fun i _ => ?_
  rw [abs_mul]

/-- If `deriv σ` is `M`-Lipschitz, the activation remainder is at most `M`
times the squared input gap. -/
theorem abs_quadratic_remainder (σ : ℝ → ℝ) (hσ : Differentiable ℝ σ) {M : ℝ}
    (hM : 0 ≤ M) (hLip : ∀ a b, |deriv σ a - deriv σ b| ≤ M * |a - b|) (u v : ℝ) :
    |σ u - σ v - deriv σ v * (u - v)| ≤ M * |u - v| ^ 2 := by
  set d : ℝ := u - v
  let φ : ℝ → ℝ := fun t => σ (v + t * d) - σ v - t * (deriv σ v * d)
  have hφdiff : Differentiable ℝ φ :=
    ((hσ.comp ((differentiable_const _).add (differentiable_id.mul (differentiable_const _)))).sub
        (differentiable_const _)).sub
      (differentiable_id.mul (differentiable_const _))
  obtain ⟨c, hc, hslope⟩ :=
    exists_deriv_eq_slope φ zero_lt_one hφdiff.continuous.continuousOn
      hφdiff.differentiableOn
  have hinner : HasDerivAt (fun t => v + t * d) d c := by
    simpa using HasDerivAt.const_add v ((hasDerivAt_id c).mul_const d)
  have hcomp : HasDerivAt (fun t => σ (v + t * d)) (deriv σ (v + c * d) * d) c := by
    simpa [Function.comp_def] using
      HasDerivAt.comp c ((hσ (v + c * d)).hasDerivAt) hinner
  have hlin : HasDerivAt (fun t => t * (deriv σ v * d)) (deriv σ v * d) c := by
    simpa using (hasDerivAt_id c).mul_const (deriv σ v * d)
  have hφ : HasDerivAt φ ((deriv σ (v + c * d) - deriv σ v) * d) c := by
    have hsub := (hcomp.sub (hasDerivAt_const c (σ v))).sub hlin
    have heq : φ =ᶠ[nhds c]
        ((fun t => σ (v + t * d)) - fun _ => σ v) - fun t => t * (deriv σ v * d) := by
      refine Filter.EventuallyEq.of_eq ?_
      ext t
      simp [φ]
    exact (hsub.congr_of_eventuallyEq heq.symm).congr_deriv (by ring)
  have hderiv : deriv φ c = (deriv σ (v + c * d) - deriv σ v) * d := hφ.deriv
  have hφ1 : deriv φ c = φ 1 := by
    simpa [show φ 0 = 0 by simp [φ], sub_zero] using hslope
  have hlipc : |deriv σ (v + c * d) - deriv σ v| ≤ M * |c| * |d| := by
    have hpt : v + c * d - v = c * d := by ring
    calc
      |deriv σ (v + c * d) - deriv σ v| ≤ M * |(v + c * d) - v| := hLip _ _
      _ = M * |c * d| := by rw [hpt]
      _ = M * |c| * |d| := by rw [abs_mul]; ring
  have hcabs : |c| ≤ 1 := by
    rw [abs_of_pos hc.1]
    exact le_of_lt hc.2
  calc
    |σ u - σ v - deriv σ v * (u - v)| = |φ 1| := by simp [φ, d]
    _ = |(deriv σ (v + c * d) - deriv σ v) * d| := by rw [← hφ1, hderiv]
    _ = |deriv σ (v + c * d) - deriv σ v| * |d| := abs_mul _ _
    _ ≤ (M * |c| * |d|) * |d| := mul_le_mul_of_nonneg_right hlipc (abs_nonneg _)
    _ = M * |c| * (|d| * |d|) := by ring
    _ ≤ M * 1 * (|d| * |d|) := by
        refine mul_le_mul_of_nonneg_right ?_ (mul_nonneg (abs_nonneg _) (abs_nonneg _))
        exact mul_le_mul_of_nonneg_left hcabs hM
    _ = M * |u - v| ^ 2 := by simp [d, sq, mul_one]

theorem abs_forward_quadratic (net : TwoLayerNet d m) (x : Fin d → ℝ) (y lr1 : ℝ) {M : ℝ}
    (hM : 0 ≤ M)
    (hLip : ∀ a b, |deriv net.act.σ a - deriv net.act.σ b| ≤ M * |a - b|) :
    |(FeatureStep.afterFirstLayerStep net x y lr1).forward x - tangentOutput net x y lr1| ≤
      M * ∑ i, |net.W₂ 0 i| *
        |FeatureStep.preactivation (FeatureStep.afterFirstLayerStep net x y lr1) x i -
          FeatureStep.preactivation net x i| ^ 2 := by
  refine (abs_forward_after_sub_tangent net x y lr1).trans ?_
  rw [Finset.mul_sum]
  refine Finset.sum_le_sum fun i _ => ?_
  calc
    |net.W₂ 0 i| *
        |net.act.σ (FeatureStep.preactivation (FeatureStep.afterFirstLayerStep net x y lr1) x i) -
          net.act.σ (FeatureStep.preactivation net x i) -
          deriv net.act.σ (FeatureStep.preactivation net x i) *
            (FeatureStep.preactivation (FeatureStep.afterFirstLayerStep net x y lr1) x i -
              FeatureStep.preactivation net x i)| ≤
        |net.W₂ 0 i| * (M *
          |FeatureStep.preactivation (FeatureStep.afterFirstLayerStep net x y lr1) x i -
            FeatureStep.preactivation net x i| ^ 2) :=
      mul_le_mul_of_nonneg_left
        (abs_quadratic_remainder net.act.σ net.act.differentiable hM hLip _ _) (abs_nonneg _)
    _ = M * (|net.W₂ 0 i| *
        |FeatureStep.preactivation (FeatureStep.afterFirstLayerStep net x y lr1) x i -
          FeatureStep.preactivation net x i| ^ 2) := by ring

theorem ntk_tangent_gap (hd : 0 < (d : ℝ)) (hm : 0 < (m : ℝ))
    (net : TwoLayerNet d m) (x : Fin d → ℝ) (y : ℝ) {L M : ℝ}
    (hL : 0 ≤ L) (hM : 0 ≤ M)
    (hLip : ∀ a b, |deriv net.act.σ a - deriv net.act.σ b| ≤ M * |a - b|)
    (hW : ∀ i, |net.W₂ 0 i| ≤ (NTKScaling.standard hd hm).init_std_W2)
    (hσ : ∀ i, |deriv net.act.σ (FeatureStep.preactivation net x i)| ≤ L) :
    |(FeatureStep.afterFirstLayerStep net x y (NTKScaling.standard hd hm).lr_W1).forward x -
        tangentOutput net x y (NTKScaling.standard hd hm).lr_W1| ≤
      M * (|FeatureStep.outputResidual net x y| ^ 2 * L ^ 2 *
        FeatureStep.inputEnergy x ^ 2 / Real.sqrt (m : ℝ)) := by
  let lr1 : ℝ := (NTKScaling.standard hd hm).lr_W1
  refine (abs_forward_quadratic net x y lr1 hM hLip).trans ?_
  have hscale : (NTKScaling.standard hd hm).init_std_W2 = 1 / Real.sqrt (m : ℝ) := by
    simp [NTKScaling.standard]
  have hstep : ∀ i,
      |FeatureStep.preactivation (FeatureStep.afterFirstLayerStep net x y lr1) x i -
          FeatureStep.preactivation net x i| ≤
        |FeatureStep.outputResidual net x y| * L * FeatureStep.inputEnergy x /
          Real.sqrt (m : ℝ) :=
    fun i => FeatureStep.ntk_preactivation_step_le hd hm net x y i hL (hW i) (hσ i)
  set gap : ℝ := |FeatureStep.outputResidual net x y| * L * FeatureStep.inputEnergy x /
    Real.sqrt (m : ℝ)
  have hterm : ∀ i, |net.W₂ 0 i| *
      |FeatureStep.preactivation (FeatureStep.afterFirstLayerStep net x y lr1) x i -
        FeatureStep.preactivation net x i| ^ 2 ≤
      (1 / Real.sqrt (m : ℝ)) * gap ^ 2 := by
    intro i
    have hsq : |FeatureStep.preactivation (FeatureStep.afterFirstLayerStep net x y lr1) x i -
        FeatureStep.preactivation net x i| ^ 2 ≤ gap ^ 2 :=
      by simpa [sq] using mul_self_le_mul_self (abs_nonneg _) (hstep i)
    have hW' : |net.W₂ 0 i| ≤ 1 / Real.sqrt (m : ℝ) := by rw [← hscale]; exact hW i
    exact mul_le_mul hW' hsq (sq_nonneg _) (div_nonneg zero_le_one (Real.sqrt_nonneg _))
  have hsum : ∑ i, |net.W₂ 0 i| *
      |FeatureStep.preactivation (FeatureStep.afterFirstLayerStep net x y lr1) x i -
        FeatureStep.preactivation net x i| ^ 2 ≤
      (m : ℝ) * ((1 / Real.sqrt (m : ℝ)) * gap ^ 2) := by
    refine (Finset.sum_le_sum fun i _ => hterm i).trans ?_
    simp [Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
  have hgap : (m : ℝ) * ((1 / Real.sqrt (m : ℝ)) * gap ^ 2) =
      |FeatureStep.outputResidual net x y| ^ 2 * L ^ 2 * FeatureStep.inputEnergy x ^ 2 /
        Real.sqrt (m : ℝ) := by
    have hm0 : Real.sqrt (m : ℝ) ≠ 0 := Real.sqrt_ne_zero'.mpr hm
    have hsqsqrt : Real.sqrt (m : ℝ) ^ 2 = (m : ℝ) := Real.sq_sqrt (le_of_lt hm)
    simp only [gap]
    field_simp [hm0]
    ring_nf
    rw [hsqsqrt]
    ring
  have hMsum : M * ∑ i, |net.W₂ 0 i| *
      |FeatureStep.preactivation (FeatureStep.afterFirstLayerStep net x y lr1) x i -
        FeatureStep.preactivation net x i| ^ 2 ≤
      M * ((m : ℝ) * ((1 / Real.sqrt (m : ℝ)) * gap ^ 2)) :=
    mul_le_mul_of_nonneg_left hsum hM
  rw [hgap] at hMsum
  simpa [lr1] using hMsum

theorem mup_tangent_gap (hd : 0 < (d : ℝ)) (hm : 0 < (m : ℝ))
    (net : TwoLayerNet d m) (x : Fin d → ℝ) (y : ℝ) {L M : ℝ}
    (hL : 0 ≤ L) (hM : 0 ≤ M)
    (hLip : ∀ a b, |deriv net.act.σ a - deriv net.act.σ b| ≤ M * |a - b|)
    (hW : ∀ i, |net.W₂ 0 i| ≤ (MuPScaling.standard hd hm).init_std_W2)
    (hσ : ∀ i, |deriv net.act.σ (FeatureStep.preactivation net x i)| ≤ L) :
    |(FeatureStep.afterFirstLayerStep net x y (MuPScaling.standard hd hm).lr_W1).forward x -
        tangentOutput net x y (MuPScaling.standard hd hm).lr_W1| ≤
      M * (|FeatureStep.outputResidual net x y| ^ 2 * L ^ 2 *
        FeatureStep.inputEnergy x ^ 2) / (m : ℝ) ^ 2 := by
  let lr1 : ℝ := (MuPScaling.standard hd hm).lr_W1
  refine (abs_forward_quadratic net x y lr1 hM hLip).trans ?_
  have hscale : (MuPScaling.standard hd hm).init_std_W2 = 1 / (m : ℝ) := by
    simp [MuPScaling.standard]
  have hstep : ∀ i,
      |FeatureStep.preactivation (FeatureStep.afterFirstLayerStep net x y lr1) x i -
          FeatureStep.preactivation net x i| ≤
        |FeatureStep.outputResidual net x y| * L * FeatureStep.inputEnergy x / m :=
    fun i => FeatureStep.mup_preactivation_step_le hd hm net x y i hL (hW i) (hσ i)
  set gap : ℝ := |FeatureStep.outputResidual net x y| * L * FeatureStep.inputEnergy x / m
  have hterm : ∀ i, |net.W₂ 0 i| *
      |FeatureStep.preactivation (FeatureStep.afterFirstLayerStep net x y lr1) x i -
        FeatureStep.preactivation net x i| ^ 2 ≤
      (1 / (m : ℝ)) * gap ^ 2 := by
    intro i
    have hsq : |FeatureStep.preactivation (FeatureStep.afterFirstLayerStep net x y lr1) x i -
        FeatureStep.preactivation net x i| ^ 2 ≤ gap ^ 2 :=
      by simpa [sq] using mul_self_le_mul_self (abs_nonneg _) (hstep i)
    have hW' : |net.W₂ 0 i| ≤ 1 / (m : ℝ) := by rw [← hscale]; exact hW i
    exact mul_le_mul hW' hsq (sq_nonneg _) (div_nonneg zero_le_one (Nat.cast_nonneg _))
  have hsum : ∑ i, |net.W₂ 0 i| *
      |FeatureStep.preactivation (FeatureStep.afterFirstLayerStep net x y lr1) x i -
        FeatureStep.preactivation net x i| ^ 2 ≤
      (m : ℝ) * ((1 / (m : ℝ)) * gap ^ 2) := by
    refine (Finset.sum_le_sum fun i _ => hterm i).trans ?_
    simp [Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
  have hm0 : (m : ℝ) ≠ 0 := ne_of_gt hm
  have hgap : (m : ℝ) * ((1 / (m : ℝ)) * gap ^ 2) =
      |FeatureStep.outputResidual net x y| ^ 2 * L ^ 2 *
        FeatureStep.inputEnergy x ^ 2 / (m : ℝ) ^ 2 := by
    simp only [gap]
    field_simp [hm0]
  have hMsum : M * ∑ i, |net.W₂ 0 i| *
      |FeatureStep.preactivation (FeatureStep.afterFirstLayerStep net x y lr1) x i -
        FeatureStep.preactivation net x i| ^ 2 ≤
      M * ((m : ℝ) * ((1 / (m : ℝ)) * gap ^ 2)) :=
    mul_le_mul_of_nonneg_left hsum hM
  rw [hgap, ← mul_div_assoc] at hMsum
  simpa [lr1] using hMsum

/-- At the feature-learning rate the same remainder stays of order
`|f(x)-y|²`. Width does not remove it. -/
theorem feature_learning_tangent_gap (hm : 0 < m)
    (net : TwoLayerNet d m) (x : Fin d → ℝ) (y : ℝ) {L M : ℝ}
    (hL : 0 ≤ L) (hM : 0 ≤ M)
    (hLip : ∀ a b, |deriv net.act.σ a - deriv net.act.σ b| ≤ M * |a - b|)
    (hW : ∀ i, |net.W₂ 0 i| ≤ 1 / (m : ℝ))
    (hσ : ∀ i, |deriv net.act.σ (FeatureStep.preactivation net x i)| ≤ L) :
    |(FeatureStep.afterFirstLayerStep net x y (FeatureStep.featureLearningRate m)).forward x -
        tangentOutput net x y (FeatureStep.featureLearningRate m)| ≤
      M * |FeatureStep.outputResidual net x y| ^ 2 * L ^ 2 *
        FeatureStep.inputEnergy x ^ 2 := by
  let lr1 : ℝ := FeatureStep.featureLearningRate m
  refine (abs_forward_quadratic net x y lr1 hM hLip).trans ?_
  have hstep : ∀ i,
      |FeatureStep.preactivation (FeatureStep.afterFirstLayerStep net x y lr1) x i -
          FeatureStep.preactivation net x i| ≤
        |FeatureStep.outputResidual net x y| * L * FeatureStep.inputEnergy x :=
    fun i => FeatureStep.feature_learning_preactivation_step_le hm net x y i hL (hW i) (hσ i)
  set gap : ℝ := |FeatureStep.outputResidual net x y| * L * FeatureStep.inputEnergy x
  have hterm : ∀ i, |net.W₂ 0 i| *
      |FeatureStep.preactivation (FeatureStep.afterFirstLayerStep net x y lr1) x i -
        FeatureStep.preactivation net x i| ^ 2 ≤
      (1 / (m : ℝ)) * gap ^ 2 := by
    intro i
    have hsq : |FeatureStep.preactivation (FeatureStep.afterFirstLayerStep net x y lr1) x i -
        FeatureStep.preactivation net x i| ^ 2 ≤ gap ^ 2 :=
      by simpa [sq] using mul_self_le_mul_self (abs_nonneg _) (hstep i)
    exact mul_le_mul (hW i) hsq (sq_nonneg _) (div_nonneg zero_le_one (Nat.cast_nonneg _))
  have hsum : ∑ i, |net.W₂ 0 i| *
      |FeatureStep.preactivation (FeatureStep.afterFirstLayerStep net x y lr1) x i -
        FeatureStep.preactivation net x i| ^ 2 ≤
      (m : ℝ) * ((1 / (m : ℝ)) * gap ^ 2) := by
    refine (Finset.sum_le_sum fun i _ => hterm i).trans ?_
    simp [Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
  have hm0 : (m : ℝ) ≠ 0 := ne_of_gt (Nat.cast_pos.mpr hm)
  have hgap : (m : ℝ) * ((1 / (m : ℝ)) * gap ^ 2) = gap ^ 2 := by
    field_simp [hm0]
  have hMsum : M * ∑ i, |net.W₂ 0 i| *
      |FeatureStep.preactivation (FeatureStep.afterFirstLayerStep net x y lr1) x i -
        FeatureStep.preactivation net x i| ^ 2 ≤
      M * gap ^ 2 := by
    refine (mul_le_mul_of_nonneg_left hsum hM).trans ?_
    rw [hgap]
  have hrewrite : M * gap ^ 2 =
      M * |FeatureStep.outputResidual net x y| ^ 2 * L ^ 2 *
        FeatureStep.inputEnergy x ^ 2 := by
    simp only [gap]
    ring
  rw [hrewrite] at hMsum
  simpa [lr1] using hMsum

end OpenTheory.TangentStep
