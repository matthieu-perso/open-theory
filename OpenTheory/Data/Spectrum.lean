import Mathlib.Data.Real.Basic

namespace OpenTheory

/-- A gap between the leading eigenvalue of a Gram matrix and the bulk.
    `leading ≥ bulk + gap` with `gap > 0` is what makes a leading subspace unique
    and gives spectral bias a positive rate. -/
structure SpectralGap where
  leading : ℝ
  bulk : ℝ
  gap : ℝ
  gap_pos : 0 < gap
  bulk_nonneg : 0 ≤ bulk
  separated : leading ≥ bulk + gap

end OpenTheory
