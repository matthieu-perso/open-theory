import Mathlib.Analysis.SpecialFunctions.Pow.Real
import OpenTheory.Architectures.Basic

namespace OpenTheory

/-- Neural Tangent Kernel parameterization.
    `NTKScaling.standard` sets both initialization standard deviations to `1/√fan_in`
    and both learning rates to 1. At infinite width the hidden features stay at
    their random initialization. This is the comparison scaling for the
    feature-learning conjecture. -/
structure NTKScaling (d m : ℕ) where
  init_std_W1 : ℝ
  init_std_W2 : ℝ
  lr_W1 : ℝ
  lr_W2 : ℝ
  h_d_pos : (d : ℝ) > 0
  h_m_pos : (m : ℝ) > 0

noncomputable def NTKScaling.standard {d m : ℕ}
    (hd : (d : ℝ) > 0) (hm : (m : ℝ) > 0) : NTKScaling d m where
  init_std_W1 := 1 / Real.sqrt (d : ℝ)
  init_std_W2 := 1 / Real.sqrt (m : ℝ)
  lr_W1 := 1
  lr_W2 := 1
  h_d_pos := hd
  h_m_pos := hm

lemma NTKScaling.input_dim_pos {d m : ℕ} (s : NTKScaling d m) : (0 : ℝ) < d :=
  s.h_d_pos

lemma NTKScaling.width_pos {d m : ℕ} (s : NTKScaling d m) : (0 : ℝ) < m :=
  s.h_m_pos

end OpenTheory
