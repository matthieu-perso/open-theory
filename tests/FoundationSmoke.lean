import OpenTheory.Data.Foundation

open OpenTheory

-- Detect missing averaging, wrong label projection, and wrong squared loss.
example : Dataset.empiricalRisk (fun i : Fin 2 => ((0 : ℝ), if i = 0 then 1 else 3))
    (fun _ : ℝ => (0 : ℝ)) squaredLoss = 5 := by
  norm_num [Dataset.empiricalRisk, squaredLoss, Fin.sum_univ_two]

-- Empty samples have an explicit totalized value, not an implicit nonempty guarantee.
example (D : Dataset ℝ ℝ 0) : Dataset.empiricalRisk D id squaredLoss = 0 := by
  simp [Dataset.empiricalRisk]
