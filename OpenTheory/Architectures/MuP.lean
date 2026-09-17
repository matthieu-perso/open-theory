import Mathlib.Analysis.SpecialFunctions.Pow.Real
import OpenTheory.Architectures.Basic

namespace OpenTheory

/-- 
  Maximal Update Parametrization (µP) Coordinate Normalization:
  Guarantees that feature learning does not blow up or vanish in the infinite-width limit.
-/
structure MuPScaling (d m : ℕ) where
  init_std_W1 : ℝ := 1 / Real.sqrt (d : ℝ)
  init_std_W2 : ℝ := 1 / (m : ℝ)
  lr_W1 : ℝ := 1.0
  lr_W2 : ℝ := 1 / (m : ℝ)
  h_d_pos : (d : ℝ) > 0
  h_m_pos : (m : ℝ) > 0

end OpenTheory
