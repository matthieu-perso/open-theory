import OpenTheory.Architectures.Basic
import OpenTheory.Parameterization.MuP
import OpenTheory.Parameterization.NTK
import Mathlib.Analysis.Calculus.Deriv.Basic
import Mathlib.Analysis.Calculus.Deriv.Add
import Mathlib.Analysis.Calculus.Deriv.Comp
import Mathlib.Analysis.Calculus.Deriv.Mul
import Mathlib.Analysis.Calculus.Deriv.Pow
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.BigOperators.Group.Finset.Piecewise
import Mathlib.Tactic.Ring

/-! One gradient step on a two-layer network, and what it does to the hidden
preactivations.

The loss on one labeled input is `(1/2) (f(x) - y)²`. The first-layer update is
gradient descent on that loss. The resulting change in hidden preactivation `i`
is exact:

`Δpre_i = -η₁ (f(x) - y) σ'(pre_i) W₂ᵢ ‖x‖²`.

Yang and Hu's dichotomy is read off this factor. If the readout coordinate is
no larger than its initialization scale, the checked NTK rate `η₁ = 1` moves
the preactivation by `O(1/√m)`, and the checked μP rate `η₁ = 1` moves it by
`O(1/m)`. Both vanish as the width grows, so both are kernel-regime steps.
The first-layer rate `η₁ = m`, with a readout coordinate of size `1/m`, moves
that preactivation by a width-independent amount. That is the one-step
feature-learning side of the same dichotomy.

This is not an infinite-width limit, not a proof that training learns a
representation, and not a generalization bound. The kernel-ridge guarantee
applies to these features only while the displacement stays negligible.
-/

noncomputable section

open scoped BigOperators

namespace OpenTheory.FeatureStep

variable {d m : ℕ}

def preactivation (net : TwoLayerNet d m) (x : Fin d → ℝ) (i : Fin m) : ℝ :=
  ∑ j : Fin d, net.W₁ i j * x j

def inputEnergy (x : Fin d → ℝ) : ℝ :=
  ∑ j : Fin d, x j ^ 2

def outputResidual (net : TwoLayerNet d m) (x : Fin d → ℝ) (y : ℝ) : ℝ :=
  net.forward x - y

def sampleLoss (prediction label : ℝ) : ℝ :=
  (1 / 2) * (prediction - label) ^ 2

/-- Replace one first-layer entry. Used to name the partial derivative. -/
def replaceEntry (net : TwoLayerNet d m) (i : Fin m) (j : Fin d) (t : ℝ) : TwoLayerNet d m :=
  { net with W₁ := fun a b => if a = i ∧ b = j then t else net.W₁ a b }

theorem inputEnergy_nonneg (x : Fin d → ℝ) : 0 ≤ inputEnergy x :=
  Finset.sum_nonneg fun _ _ => sq_nonneg _

theorem replaceEntry_self (net : TwoLayerNet d m) (i : Fin m) (j : Fin d) :
    replaceEntry net i j (net.W₁ i j) = net := by
  cases net
  simp only [replaceEntry]
  congr
  ext a b
  split_ifs with h
  · obtain ⟨rfl, rfl⟩ := h
    rfl
  · rfl

theorem preactivation_replace (net : TwoLayerNet d m) (x : Fin d → ℝ) (i : Fin m)
    (j : Fin d) (t : ℝ) :
    preactivation (replaceEntry net i j t) x i =
      preactivation net x i + (t - net.W₁ i j) * x j := by
  unfold preactivation replaceEntry
  rw [← sub_eq_iff_eq_add', ← Finset.sum_sub_distrib]
  have hsum := Finset.sum_eq_single (s := Finset.univ) j
    (f := fun b => (if i = i ∧ b = j then t else net.W₁ i b) * x b - net.W₁ i b * x b)
    (fun b _ hb => by
      rw [ite_eq_right (fun h => hb h.2)]
      ring)
    (fun h => absurd (Finset.mem_univ j) h)
  rw [hsum, ite_eq_left ⟨rfl, rfl⟩]
  ring

theorem preactivation_replace_other (net : TwoLayerNet d m) (x : Fin d → ℝ)
    {i k : Fin m} (j : Fin d) (hk : k ≠ i) (t : ℝ) :
    preactivation (replaceEntry net i j t) x k = preactivation net x k := by
  unfold preactivation replaceEntry
  refine Finset.sum_congr rfl fun b _ => ?_
  have hk' : ¬ (k = i ∧ b = j) := fun h => hk h.1
  simp [hk']

theorem preactivation_hasDerivAt (net : TwoLayerNet d m) (x : Fin d → ℝ) (i : Fin m)
    (j : Fin d) :
    HasDerivAt (fun t => preactivation (replaceEntry net i j t) x i) (x j) (net.W₁ i j) := by
  have hfun : (fun t => preactivation (replaceEntry net i j t) x i) =
      fun t => preactivation net x i + (t - net.W₁ i j) * x j := by
    ext t
    exact preactivation_replace net x i j t
  rw [hfun]
  have hsub : HasDerivAt (fun t => t - net.W₁ i j) 1 (net.W₁ i j) :=
    HasDerivAt.sub_const (net.W₁ i j) (hasDerivAt_id (net.W₁ i j))
  have hmul : HasDerivAt (fun t => (t - net.W₁ i j) * x j) (x j) (net.W₁ i j) := by
    simpa using HasDerivAt.mul_const hsub (x j)
  exact HasDerivAt.const_add (preactivation net x i) hmul

/-- Partial derivative of the network output in one first-layer entry. -/
theorem forward_hasDerivAt_entry (net : TwoLayerNet d m) (x : Fin d → ℝ) (i : Fin m)
    (j : Fin d) :
    HasDerivAt (fun t => (replaceEntry net i j t).forward x)
      (net.W₂ 0 i * deriv net.act.σ (preactivation net x i) * x j) (net.W₁ i j) := by
  have hσ : HasDerivAt (fun t => net.act.σ (preactivation (replaceEntry net i j t) x i))
      (deriv net.act.σ (preactivation net x i) * x j) (net.W₁ i j) := by
    have hpre := preactivation_hasDerivAt net x i j
    have hσat : HasDerivAt net.act.σ (deriv net.act.σ (preactivation net x i))
        (preactivation (replaceEntry net i j (net.W₁ i j)) x i) := by
      rw [replaceEntry_self]
      exact (net.act.differentiable (preactivation net x i)).hasDerivAt
    have hchain : HasDerivAt
        (net.act.σ ∘ fun t => preactivation (replaceEntry net i j t) x i)
        (deriv net.act.σ (preactivation net x i) * x j) (net.W₁ i j) :=
      HasDerivAt.comp_of_eq (net.W₁ i j) hσat hpre rfl
    simpa [Function.comp_def] using hchain
  have hsum : HasDerivAt
      (fun t => ∑ k : Fin m, net.W₂ 0 k *
        net.act.σ (preactivation (replaceEntry net i j t) x k))
      (net.W₂ 0 i * deriv net.act.σ (preactivation net x i) * x j) (net.W₁ i j) := by
    have hterm : ∀ k, HasDerivAt
        (fun t => net.W₂ 0 k * net.act.σ (preactivation (replaceEntry net i j t) x k))
        (if k = i then net.W₂ 0 i * deriv net.act.σ (preactivation net x i) * x j else 0)
        (net.W₁ i j) := by
      intro k
      by_cases hk : k = i
      · rw [hk]
        refine (HasDerivAt.const_mul (net.W₂ 0 i) hσ).congr_deriv ?_
        rw [ite_eq_left rfl]
        ring
      · have hconst : (fun t => net.W₂ 0 k *
            net.act.σ (preactivation (replaceEntry net i j t) x k)) =
            fun _ => net.W₂ 0 k * net.act.σ (preactivation net x k) := by
          ext t
          rw [preactivation_replace_other net x j hk t]
        rw [hconst]
        refine (hasDerivAt_const (net.W₁ i j)
          (net.W₂ 0 k * net.act.σ (preactivation net x k))).congr_deriv ?_
        rw [ite_eq_right hk]
    exact (HasDerivAt.fun_sum (fun k _ => hterm k)).congr_deriv (by
      rw [Finset.sum_ite_eq' (Finset.univ : Finset (Fin m)) i
        (fun _ => net.W₂ 0 i * deriv net.act.σ (preactivation net x i) * x j),
        ite_eq_left (Finset.mem_univ i)])
  have hforward : (fun t => (replaceEntry net i j t).forward x) =
      fun t => ∑ k : Fin m, net.W₂ 0 k *
        net.act.σ (preactivation (replaceEntry net i j t) x k) := by
    ext t
    rfl
  rw [hforward]
  exact hsum

theorem sampleLoss_hasDerivAt (y f : ℝ) :
    HasDerivAt (fun z => sampleLoss z y) (f - y) f := by
  unfold sampleLoss
  have hsub : HasDerivAt (fun z => z - y) 1 f :=
    HasDerivAt.sub_const y (hasDerivAt_id f)
  have hsq : HasDerivAt (fun u => u ^ 2) ((2 : ℝ) * (f - y) ^ (1 : ℕ)) (f - y) :=
    hasDerivAt_pow 2 (f - y)
  have hchain : HasDerivAt (fun z => (z - y) ^ 2)
      ((2 : ℝ) * (f - y) ^ (1 : ℕ) * 1) f := by
    have hcomp := HasDerivAt.comp_of_eq f hsq hsub rfl
    simpa [Function.comp_def] using hcomp
  exact (HasDerivAt.const_mul ((1 : ℝ) / 2) hchain).congr_deriv (by ring)

/-- Partial derivative of the half-squared loss in one first-layer entry. -/
theorem loss_hasDerivAt_entry (net : TwoLayerNet d m) (x : Fin d → ℝ) (y : ℝ)
    (i : Fin m) (j : Fin d) :
    HasDerivAt (fun t => sampleLoss ((replaceEntry net i j t).forward x) y)
      (outputResidual net x y * net.W₂ 0 i * deriv net.act.σ (preactivation net x i) * x j)
      (net.W₁ i j) := by
  have hf := forward_hasDerivAt_entry net x i j
  have hpoint : (replaceEntry net i j (net.W₁ i j)).forward x = net.forward x :=
    congrArg (fun n : TwoLayerNet d m => n.forward x) (replaceEntry_self net i j)
  have hs : HasDerivAt (fun z => sampleLoss z y) (outputResidual net x y) (net.forward x) := by
    simpa [outputResidual] using sampleLoss_hasDerivAt y (net.forward x)
  have hcomp := HasDerivAt.comp_of_eq (net.W₁ i j) hs hf hpoint.symm
  simpa [Function.comp_def, mul_assoc] using hcomp

/-- Simultaneous gradient step on the first layer. Each entry moves by `-η`
times the partial derivative of `sampleLoss`. -/
def firstLayerStep (net : TwoLayerNet d m) (x : Fin d → ℝ) (y lr1 : ℝ) :
    Matrix (Fin m) (Fin d) ℝ :=
  fun a b => net.W₁ a b - lr1 * outputResidual net x y *
    deriv net.act.σ (preactivation net x a) * net.W₂ 0 a * x b

def afterFirstLayerStep (net : TwoLayerNet d m) (x : Fin d → ℝ) (y lr1 : ℝ) :
    TwoLayerNet d m :=
  { net with W₁ := firstLayerStep net x y lr1 }

theorem preactivation_step (net : TwoLayerNet d m) (x : Fin d → ℝ) (y lr1 : ℝ)
    (i : Fin m) :
    preactivation (afterFirstLayerStep net x y lr1) x i =
      preactivation net x i - lr1 * outputResidual net x y *
        deriv net.act.σ (preactivation net x i) * net.W₂ 0 i * inputEnergy x := by
  unfold preactivation afterFirstLayerStep firstLayerStep inputEnergy
  have hrow : ∀ j,
      (net.W₁ i j - lr1 * outputResidual net x y *
        deriv net.act.σ (preactivation net x i) * net.W₂ 0 i * x j) * x j =
      net.W₁ i j * x j - lr1 * outputResidual net x y *
        deriv net.act.σ (preactivation net x i) * net.W₂ 0 i * (x j ^ 2) := by
    intro j
    ring
  simp_rw [hrow, Finset.sum_sub_distrib, ← Finset.mul_sum]
  simp [preactivation]

theorem abs_preactivation_step (net : TwoLayerNet d m) (x : Fin d → ℝ) (y lr1 : ℝ)
    (i : Fin m) :
    |preactivation (afterFirstLayerStep net x y lr1) x i - preactivation net x i| =
      |lr1| * |outputResidual net x y| *
        |deriv net.act.σ (preactivation net x i)| * |net.W₂ 0 i| * inputEnergy x := by
  rw [preactivation_step]
  rw [show preactivation net x i -
        lr1 * outputResidual net x y * deriv net.act.σ (preactivation net x i) *
          net.W₂ 0 i * inputEnergy x -
        preactivation net x i =
      -(lr1 * outputResidual net x y * deriv net.act.σ (preactivation net x i) *
          net.W₂ 0 i * inputEnergy x) by ring]
  rw [abs_neg, abs_mul, abs_mul, abs_mul, abs_mul, abs_of_nonneg (inputEnergy_nonneg x)]

/-- First-layer learning rate that cancels a readout of size `1/m`. -/
def featureLearningRate (m : ℕ) : ℝ := m

theorem ntk_preactivation_step_le (hd : 0 < (d : ℝ)) (hm : 0 < (m : ℝ))
    (net : TwoLayerNet d m) (x : Fin d → ℝ) (y : ℝ) (i : Fin m) {L : ℝ}
    (hL : 0 ≤ L)
    (hW : |net.W₂ 0 i| ≤ (NTKScaling.standard hd hm).init_std_W2)
    (hσ : |deriv net.act.σ (preactivation net x i)| ≤ L) :
    |preactivation (afterFirstLayerStep net x y (NTKScaling.standard hd hm).lr_W1) x i -
        preactivation net x i| ≤
      |outputResidual net x y| * L * inputEnergy x / Real.sqrt (m : ℝ) := by
  rw [abs_preactivation_step]
  have hlr : |(NTKScaling.standard hd hm).lr_W1| = 1 := by
    simp [NTKScaling.standard]
  have hscale : (NTKScaling.standard hd hm).init_std_W2 = 1 / Real.sqrt (m : ℝ) := by
    simp [NTKScaling.standard]
  rw [hlr, hscale] at *
  have hpair : |deriv net.act.σ (preactivation net x i)| * |net.W₂ 0 i| ≤
      L * (1 / Real.sqrt (m : ℝ)) :=
    mul_le_mul hσ hW (abs_nonneg _) hL
  have hnonneg : 0 ≤ |outputResidual net x y| * inputEnergy x :=
    mul_nonneg (abs_nonneg _) (inputEnergy_nonneg x)
  calc
    1 * |outputResidual net x y| * |deriv net.act.σ (preactivation net x i)| *
          |net.W₂ 0 i| * inputEnergy x
        = |outputResidual net x y| * inputEnergy x *
          (|deriv net.act.σ (preactivation net x i)| * |net.W₂ 0 i|) := by ring
    _ ≤ |outputResidual net x y| * inputEnergy x * (L * (1 / Real.sqrt (m : ℝ))) :=
          mul_le_mul_of_nonneg_left hpair hnonneg
    _ = |outputResidual net x y| * L * inputEnergy x / Real.sqrt (m : ℝ) := by ring

theorem mup_preactivation_step_le (hd : 0 < (d : ℝ)) (hm : 0 < (m : ℝ))
    (net : TwoLayerNet d m) (x : Fin d → ℝ) (y : ℝ) (i : Fin m) {L : ℝ}
    (hL : 0 ≤ L)
    (hW : |net.W₂ 0 i| ≤ (MuPScaling.standard hd hm).init_std_W2)
    (hσ : |deriv net.act.σ (preactivation net x i)| ≤ L) :
    |preactivation (afterFirstLayerStep net x y (MuPScaling.standard hd hm).lr_W1) x i -
        preactivation net x i| ≤
      |outputResidual net x y| * L * inputEnergy x / m := by
  rw [abs_preactivation_step]
  have hlr : |(MuPScaling.standard hd hm).lr_W1| = 1 := by
    simp [MuPScaling.standard]
  have hscale : (MuPScaling.standard hd hm).init_std_W2 = 1 / (m : ℝ) := by
    simp [MuPScaling.standard]
  rw [hlr, hscale] at *
  have hpair : |deriv net.act.σ (preactivation net x i)| * |net.W₂ 0 i| ≤ L * (1 / (m : ℝ)) :=
    mul_le_mul hσ hW (abs_nonneg _) hL
  have hnonneg : 0 ≤ |outputResidual net x y| * inputEnergy x :=
    mul_nonneg (abs_nonneg _) (inputEnergy_nonneg x)
  calc
    1 * |outputResidual net x y| * |deriv net.act.σ (preactivation net x i)| *
          |net.W₂ 0 i| * inputEnergy x
        = |outputResidual net x y| * inputEnergy x *
          (|deriv net.act.σ (preactivation net x i)| * |net.W₂ 0 i|) := by ring
    _ ≤ |outputResidual net x y| * inputEnergy x * (L * (1 / (m : ℝ))) :=
          mul_le_mul_of_nonneg_left hpair hnonneg
    _ = |outputResidual net x y| * L * inputEnergy x / m := by ring

theorem feature_learning_preactivation_step_le (hm : 0 < m)
    (net : TwoLayerNet d m) (x : Fin d → ℝ) (y : ℝ) (i : Fin m) {L : ℝ}
    (hL : 0 ≤ L)
    (hW : |net.W₂ 0 i| ≤ 1 / (m : ℝ))
    (hσ : |deriv net.act.σ (preactivation net x i)| ≤ L) :
    |preactivation (afterFirstLayerStep net x y (featureLearningRate m)) x i -
        preactivation net x i| ≤
      |outputResidual net x y| * L * inputEnergy x := by
  rw [abs_preactivation_step]
  have hm0cast : (0 : ℝ) ≤ m := Nat.cast_nonneg m
  have hlr : |featureLearningRate m| = (m : ℝ) := by
    simp [featureLearningRate, abs_of_nonneg hm0cast]
  rw [hlr]
  have hpair : |deriv net.act.σ (preactivation net x i)| * |net.W₂ 0 i| ≤ L * (1 / (m : ℝ)) :=
    mul_le_mul hσ hW (abs_nonneg _) hL
  have hscaled : (m : ℝ) * (|deriv net.act.σ (preactivation net x i)| * |net.W₂ 0 i|) ≤
      (m : ℝ) * (L * (1 / (m : ℝ))) :=
    mul_le_mul_of_nonneg_left hpair hm0cast
  have hnonneg : 0 ≤ |outputResidual net x y| * inputEnergy x :=
    mul_nonneg (abs_nonneg _) (inputEnergy_nonneg x)
  have hm0 : (m : ℝ) ≠ 0 := ne_of_gt (Nat.cast_pos.mpr hm)
  calc
    (m : ℝ) * |outputResidual net x y| * |deriv net.act.σ (preactivation net x i)| *
          |net.W₂ 0 i| * inputEnergy x
        = |outputResidual net x y| * inputEnergy x *
          ((m : ℝ) * (|deriv net.act.σ (preactivation net x i)| * |net.W₂ 0 i|)) := by ring
    _ ≤ |outputResidual net x y| * inputEnergy x * ((m : ℝ) * (L * (1 / (m : ℝ)))) :=
          mul_le_mul_of_nonneg_left hscaled hnonneg
    _ = |outputResidual net x y| * L * inputEnergy x := by
          have hcancel : (m : ℝ) * (L * (1 / (m : ℝ))) = L := by
            rw [mul_left_comm, one_div, mul_inv_cancel₀ hm0, mul_one]
          rw [hcancel]
          ring

/-- A readout coordinate at least `1/m`, a derivative, a residual and an input
that are all at least `c`, and rate `m`, move the preactivation by at least `c³`. -/
theorem feature_learning_preactivation_step_ge (hm : 0 < m)
    (net : TwoLayerNet d m) (x : Fin d → ℝ) (y : ℝ) (i : Fin m) {c : ℝ}
    (hc : 0 ≤ c)
    (hW : 1 / (m : ℝ) ≤ |net.W₂ 0 i|)
    (hσ : c ≤ |deriv net.act.σ (preactivation net x i)|)
    (hr : c ≤ |outputResidual net x y|)
    (hx : c ≤ inputEnergy x) :
    c ^ 3 ≤
      |preactivation (afterFirstLayerStep net x y (featureLearningRate m)) x i -
        preactivation net x i| := by
  rw [abs_preactivation_step]
  have hm0cast : (0 : ℝ) ≤ m := Nat.cast_nonneg m
  have hlr : |featureLearningRate m| = (m : ℝ) := by
    simp [featureLearningRate, abs_of_nonneg hm0cast]
  rw [hlr]
  have hm0 : (m : ℝ) ≠ 0 := ne_of_gt (Nat.cast_pos.mpr hm)
  have hpair : c * (1 / (m : ℝ)) ≤
      |deriv net.act.σ (preactivation net x i)| * |net.W₂ 0 i| :=
    mul_le_mul hσ hW (div_nonneg zero_le_one hm0cast) (le_trans hc hσ)
  have hre : c * c ≤ |outputResidual net x y| * inputEnergy x :=
    mul_le_mul hr hx hc (le_trans hc hr)
  have hcancel : (m : ℝ) * (1 / (m : ℝ)) = 1 := by
    rw [one_div, mul_inv_cancel₀ hm0]
  have hleft : c ^ 3 = (m : ℝ) * (c * (1 / (m : ℝ))) * (c * c) := by
    calc
      c ^ 3 = c * c * c := by ring
      _ = ((m : ℝ) * (1 / (m : ℝ))) * c * c * c := by rw [hcancel, one_mul]
      _ = (m : ℝ) * (c * (1 / (m : ℝ))) * (c * c) := by ring
  have hscaled : (m : ℝ) * (c * (1 / (m : ℝ))) ≤
      (m : ℝ) * (|deriv net.act.σ (preactivation net x i)| * |net.W₂ 0 i|) :=
    mul_le_mul_of_nonneg_left hpair hm0cast
  have hmid :
      (m : ℝ) * (c * (1 / (m : ℝ))) * (c * c) ≤
        (m : ℝ) * (|deriv net.act.σ (preactivation net x i)| * |net.W₂ 0 i|) *
          (|outputResidual net x y| * inputEnergy x) :=
    mul_le_mul hscaled hre (mul_nonneg hc hc)
      (mul_nonneg hm0cast (mul_nonneg (le_trans hc hσ) (le_trans (div_nonneg zero_le_one hm0cast) hW)))
  have hright :
      (m : ℝ) * (|deriv net.act.σ (preactivation net x i)| * |net.W₂ 0 i|) *
          (|outputResidual net x y| * inputEnergy x) =
        (m : ℝ) * |outputResidual net x y| *
          |deriv net.act.σ (preactivation net x i)| * |net.W₂ 0 i| * inputEnergy x := by
    ring
  rw [hleft, ← hright]
  exact hmid

end OpenTheory.FeatureStep
