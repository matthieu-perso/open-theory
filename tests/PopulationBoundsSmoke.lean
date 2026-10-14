import OpenTheory.Generalization.PopulationBounds

open OpenTheory

-- Check squared norm, label subtraction, and the observation adapter together.
example : predictionError (fun x : ℝ => 2 * x) id (fun _ => (1 : ℝ)) 2 = 9 := by
  norm_num [predictionError]
