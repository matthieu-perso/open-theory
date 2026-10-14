import OpenTheory.Data.Foundation
import Mathlib.MeasureTheory.Integral.Bochner.Basic
import Mathlib.Topology.MetricSpace.Lipschitz
import Mathlib.Tactic.Linarith

/-! Risk envelopes, not sample-complexity/generalization-gap bounds.
The raw observation type and its probability law are arbitrary. Heavy-tailed laws
are permitted when the stated second moments are finite. No IID hypothesis is needed. -/
noncomputable section
open MeasureTheory

namespace OpenTheory

variable {Z E F : Type*} [NormedAddCommGroup E] [NormedAddCommGroup F]

def predictionError (f : E → F) (input : Z → E) (label : Z → F) (z : Z) : ℝ :=
  ‖f (input z) - label z‖ ^ 2

theorem predictionError_le {K : NNReal} {f : E → F} (hf : LipschitzWith K f)
    (input : Z → E) (label : Z → F) (z : Z) :
    predictionError f input label z ≤
      3 * ((K : ℝ) ^ 2 * ‖input z‖ ^ 2 + ‖f 0‖ ^ 2 + ‖label z‖ ^ 2) := by
  have hl : ‖f (input z) - f 0‖ ≤ (K : ℝ) * ‖input z‖ := by
    simpa [dist_eq_norm] using hf.dist_le_mul (input z) 0
  have ht := norm_add_le (f (input z) - f 0) (f 0)
  rw [sub_add_cancel] at ht
  have hs := norm_sub_le (f (input z)) (label z)
  have hn := norm_nonneg (f (input z) - label z)
  have ha : 0 ≤ (K : ℝ) * ‖input z‖ := mul_nonneg K.coe_nonneg (norm_nonneg _)
  have hb := norm_nonneg (f 0)
  have hc := norm_nonneg (label z)
  have he : ‖f (input z) - label z‖ ≤ (K : ℝ) * ‖input z‖ + ‖f 0‖ + ‖label z‖ := by
    linarith
  unfold predictionError
  nlinarith [sq_nonneg ((K : ℝ) * ‖input z‖ - ‖f 0‖),
    sq_nonneg ((K : ℝ) * ‖input z‖ - ‖label z‖), sq_nonneg (‖f 0‖ - ‖label z‖),
    mul_nonneg (sub_nonneg.mpr he) (by positivity :
      0 ≤ (K : ℝ) * ‖input z‖ + ‖f 0‖ + ‖label z‖ + ‖f (input z) - label z‖)]

/-- A measurable loss envelope gives integrability as well as a risk upper bound. -/
theorem Population.risk_le_envelope [MeasurableSpace Z] (P : Population Z)
    (loss bound : Z → ℝ) (hmeas : AEStronglyMeasurable loss P.law)
    (hbound : Integrable bound P.law) (h : ∀ᵐ z ∂P.law, 0 ≤ loss z ∧ loss z ≤ bound z) :
    P.IntegrableLoss loss ∧ P.risk loss ≤ P.risk bound := by
  have hi : Integrable loss P.law := hbound.mono' hmeas (h.mono fun z hz => by
    simpa [Real.norm_eq_abs, abs_of_nonneg hz.1] using hz.2)
  exact ⟨hi, integral_mono_ae hi hbound (h.mono fun _ hz => hz.2)⟩

/-- A second-moment risk bound for any observation type and probability law.
Measurability of the loss is explicit, including for nonseparable feature spaces. -/
theorem Population.risk_le_secondMoments [MeasurableSpace Z] (P : Population Z)
    {K : NNReal} {f : E → F} (hf : LipschitzWith K f) (input : Z → E) (label : Z → F)
    (hmeas : AEStronglyMeasurable (predictionError f input label) P.law)
    (hx : Integrable (fun z => ‖input z‖ ^ 2) P.law)
    (hy : Integrable (fun z => ‖label z‖ ^ 2) P.law) :
    P.IntegrableLoss (predictionError f input label) ∧
      P.risk (predictionError f input label) ≤
        3 * ((K : ℝ) ^ 2 * (∫ z, ‖input z‖ ^ 2 ∂P.law) + ‖f 0‖ ^ 2 +
          ∫ z, ‖label z‖ ^ 2 ∂P.law) := by
  have hbase := ((hx.const_mul ((K : ℝ) ^ 2)).add (integrable_const (‖f 0‖ ^ 2))).add hy
  obtain ⟨hi, hb⟩ := P.risk_le_envelope (predictionError f input label)
    (fun z => 3 * ((K : ℝ) ^ 2 * ‖input z‖ ^ 2 + ‖f 0‖ ^ 2 + ‖label z‖ ^ 2))
    hmeas (hbase.const_mul 3) (Filter.Eventually.of_forall fun z =>
      ⟨sq_nonneg _, predictionError_le hf input label z⟩)
  refine ⟨hi, ?_⟩
  have hxscale : Integrable (fun z => (K : ℝ) ^ 2 * ‖input z‖ ^ 2) P.law := hx.const_mul _
  have hxplus : Integrable (fun z => (K : ℝ) ^ 2 * ‖input z‖ ^ 2 + ‖f 0‖ ^ 2) P.law :=
    hxscale.add (integrable_const _)
  unfold Population.risk at hb
  rw [integral_const_mul, integral_add hxplus hy,
    integral_add hxscale (integrable_const (‖f 0‖ ^ 2)),
    integral_const_mul] at hb
  simpa [Population.risk] using hb

end OpenTheory
