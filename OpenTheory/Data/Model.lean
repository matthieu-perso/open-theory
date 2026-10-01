import OpenTheory.Data.Spectrum

namespace OpenTheory

/-- Spiked covariance aligned with the first `signalRank` coordinates.
    The signal strength is `spike`, the noise level is `noise`, and `gap`
    is the resulting spectral gap. Later expressivity and alignment theorems
    use this as the single generative hypothesis. -/
structure SpikedModel (d : ℕ) where
  signalRank : ℕ
  rank_pos : 0 < signalRank
  rank_le : signalRank ≤ d
  spike : ℝ
  noise : ℝ
  spike_pos : 0 < spike
  noise_pos : 0 < noise
  gap : SpectralGap

end OpenTheory
