import Mathlib.Analysis.SpecialFunctions.Pow.Real
import OpenTheory.Architectures.Basic

namespace OpenTheory

/-- Maximal Update Parametrization (µP).
    Width enters here, and only here: these fields are the initialization
    standard deviations and the per-layer learning rates. `MuPScaling.standard`
    installs `init_std_W1 = 1/√d`, `init_std_W2 = 1/m`, `lr_W1 = 1`, `lr_W2 = 1/m`,
    the choice for which every layer's coordinates keep a Θ(1) update as `m → ∞`. -/
structure MuPScaling (d m : ℕ) where
  init_std_W1 : ℝ
  init_std_W2 : ℝ
  lr_W1 : ℝ
  lr_W2 : ℝ
  h_d_pos : (d : ℝ) > 0
  h_m_pos : (m : ℝ) > 0

noncomputable def MuPScaling.standard {d m : ℕ}
    (hd : (d : ℝ) > 0) (hm : (m : ℝ) > 0) : MuPScaling d m where
  init_std_W1 := 1 / Real.sqrt (d : ℝ)
  init_std_W2 := 1 / (m : ℝ)
  lr_W1 := 1
  lr_W2 := 1 / (m : ℝ)
  h_d_pos := hd
  h_m_pos := hm

lemma MuPScaling.input_dim_pos {d m : ℕ} (s : MuPScaling d m) : (0 : ℝ) < d :=
  s.h_d_pos

lemma MuPScaling.width_pos {d m : ℕ} (s : MuPScaling d m) : (0 : ℝ) < m :=
  s.h_m_pos

end OpenTheory
