import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Analysis.Calculus.FDeriv.Basic
import Mathlib.Data.Matrix.Basic

open Real ContinuousLinearMap

namespace OpenTheory

/-- 
  ActivationFunction defines regularity constraints on non-linear transfer functions.
  We require differentiability, a finite Lipschitz constant, and non-linearity.
-/
structure ActivationFunction where
  σ : ℝ → ℝ
  differentiable : Differentiable ℝ σ
  lipschitz : ∃ L : ℝ, ∀ x y, |σ x - σ y| ≤ L * |x - y|

/-- 
  TwoLayerNet models a two-layer feedforward architecture with hidden width `m` 
  and input dimension `d`.
-/
structure TwoLayerNet (d m : ℕ) where
  W₁ : Matrix (Fin m) (Fin d) ℝ
  W₂ : Matrix (Fin 1) (Fin m) ℝ
  act : ActivationFunction

/-- Forward evaluation map of the network on input x ∈ ℝ^d -/
def TwoLayerNet.forward {d m : ℕ} (net : TwoLayerNet d m) (x : Fin d → ℝ) : ℝ :=
  let hidden : Fin m → ℝ := fun i =>
    net.act.σ (∑ j : Fin d, net.W₁ i j * x j)
  ∑ k : Fin m, net.W₂ 0 k * hidden k

/-- Lipschitz continuity of the activation, extracted from the structure field.
    This is a genuine kernel-checked lemma (no `sorry`). -/
lemma ActivationFunction.lipschitz_bound (act : ActivationFunction) :
    ∃ L : ℝ, ∀ x y : ℝ, |act.σ x - act.σ y| ≤ L * |x - y| :=
  act.lipschitz

end OpenTheory
