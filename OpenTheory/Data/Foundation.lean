import OpenTheory.Data.Sample
import Mathlib.Analysis.InnerProductSpace.PiL2
import Mathlib.MeasureTheory.Integral.Bochner.Basic
import Mathlib.Probability.Independence.Basic
import Mathlib.Probability.HasLaw
import Mathlib.Tactic.NormNum

/-! Observations carry no boundedness or distributional assumptions. -/
noncomputable section
open MeasureTheory ProbabilityTheory
open scoped BigOperators

namespace OpenTheory

abbrev Dataset (X Y : Type*) (n : ℕ) := Fin n → X × Y

def squaredLoss (prediction label : ℝ) : ℝ := (prediction - label) ^ 2

namespace Dataset

/-- The empty sample has totalized mean zero; statistical theorems must require `0 < n`. -/
def empiricalRisk {X Y : Type*} {n : ℕ} (D : Dataset X Y n)
    (predict : X → Y) (loss : Y → Y → ℝ) : ℝ :=
  (∑ i, loss (predict (D i).1) (D i).2) / n

def BoundedInputs {X Y : Type*} [Norm X] {n : ℕ}
    (D : Dataset X Y n) (B : ℝ) : Prop := ∀ i, ‖(D i).1‖ ≤ B

theorem empiricalRisk_nonneg {X Y : Type*} {n : ℕ} (D : Dataset X Y n)
    (predict : X → Y) (loss : Y → Y → ℝ) (h : ∀ y z, 0 ≤ loss y z) :
    0 ≤ empiricalRisk D predict loss :=
  div_nonneg (Finset.sum_nonneg fun _ _ => h _ _) (Nat.cast_nonneg n)

end Dataset

abbrev Parameters (d : ℕ) := EuclideanSpace ℝ (Fin d)

def Sample.toDataset {d n : ℕ} (S : Sample d n) : Dataset (Parameters d) ℝ n :=
  fun i => (WithLp.toLp 2 (S.X i), S.Y i)

theorem parameters_norm_sq {d : ℕ} (x : Parameters d) :
    ‖x‖ ^ 2 = ∑ j, (x j) ^ 2 := EuclideanSpace.real_norm_sq_eq x

theorem Sample.toDataset_bounded {d n : ℕ} (S : Sample d n) :
    Dataset.BoundedInputs S.toDataset 1 := by
  intro i
  have h := S.input_bound i
  have hn := norm_nonneg (S.toDataset i).1
  have he := parameters_norm_sq (S.toDataset i).1
  change ‖(S.toDataset i).1‖ ≤ 1
  change ‖(S.toDataset i).1‖ ^ 2 = ∑ j, S.X i j ^ 2 at he
  nlinarith

/-- A population is an actual probability law on labeled observations. -/
structure Population (Z : Type*) [MeasurableSpace Z] where
  law : Measure Z
  probability : IsProbabilityMeasure law

attribute [instance] Population.probability

def Population.risk {Z : Type*} [MeasurableSpace Z] (P : Population Z)
    (loss : Z → ℝ) : ℝ := ∫ z, loss z ∂P.law

/-- Integrability is separate: the totalized Bochner integral alone is not an expectation theorem. -/
def Population.IntegrableLoss {Z : Type*} [MeasurableSpace Z] (P : Population Z)
    (loss : Z → ℝ) : Prop := Integrable loss P.law

/-- Independence and a common measurable pushforward law, not sampling without replacement. -/
structure IIDSample {Ω Z : Type*} [MeasurableSpace Ω] [MeasurableSpace Z]
    (μ : Measure Ω) (P : Population Z) {n : ℕ} (draw : Fin n → Ω → Z) : Prop where
  probability : IsProbabilityMeasure μ
  independent : iIndepFun draw μ
  commonLaw : ∀ i, HasLaw (draw i) P.law μ

end OpenTheory
