import OpenTheory.Generalization.StatisticalBridge

open OpenTheory

-- Catches lost averaging or the wrong prediction/label order in the adapter.
example : _root_.empiricalRisk 2
    (supervisedLossClass (fun _ : Unit => fun x : ℝ => 2 * x) squaredLoss)
    (fun i : Fin 2 => if i = 0 then (1, 1) else (2, 1)) () = 5 := by
  rw [← Dataset.empiricalRisk_eq_foml]
  norm_num [Dataset.empiricalRisk, squaredLoss, Fin.sum_univ_two]

example (D : Dataset ℝ ℝ 0) (f : ℝ → ℝ) :
    D.empiricalRisk f squaredLoss = 0 := by
  simp [Dataset.empiricalRisk]
