import OpenTheory.Generalization.BallCover
import OpenTheory.Generalization.Covering
import OpenTheory.Optimization.FiniteRidge
import Mathlib.Dynamics.Ergodic.MeasurePreserving
import Mathlib.Tactic

/-! A high-probability excess-risk bound for the actual zero-initialized ridge
gradient-descent iterates. The loss is squared prediction error normalized into
`[0,1]`. Gradient descent minimizes the regularized objective, so the excess
bound pays the optimization gap and the comparator's regularization. The cover
cardinality is the grid estimate from `BallCover`. -/
noncomputable section
open MeasureTheory StatLean.StatisticalLearning Classical
open scoped BigOperators RealInnerProductSpace

namespace OpenTheory.TrainedRidge

def paramBall (d : ℕ) (rad : ℝ) : Set (Parameters d) := {θ | ‖θ‖ ≤ rad}

def boundedSet (d : ℕ) (B T : ℝ) : Set (Parameters d × ℝ) :=
  {z | ‖z.1‖ ≤ B ∧ |z.2| ≤ T}

def scale (R B T : ℝ) : ℝ := (2 * R * B + T) ^ 2

/-- Squared loss divided by the uniform bound on the ball. Outside the bounded
observation set the value is zero, so the loss lies in `[0,1]` everywhere. -/
def normalizedLoss {d : ℕ} (R B T : ℝ) (θ : Parameters d) (z : Parameters d × ℝ) : ℝ :=
  if z ∈ boundedSet d B T ∧ 0 < scale R B T then
    (⟪θ, z.1⟫ - z.2) ^ 2 / scale R B T
  else
    0

def lossLipschitz (R B T : ℝ) : ℝ :=
  if 0 < B ∧ 0 < 2 * R * B + T then 2 * B / (2 * R * B + T) else 0

/-- Parameter radius that moves the normalized loss by at most `ρ`. -/
def netRadius (R B T ρ : ℝ) : ℝ :=
  if 0 < lossLipschitz R B T then ρ / lossLipschitz R B T else 1

def excessBudget (B lam R ε ρ : ℝ) (k : ℕ) (Rscale : ℝ) : ℝ :=
  2 * (ε + 2 * ρ) +
    if 0 < Rscale then
      ((B ^ 2 + lam) * Dataset.uniformFactor B lam ^ k * R ^ 2 + lam * R ^ 2) / Rscale
    else
      0

def iterate {d n : ℕ} (B lam : ℝ) (D : Dataset (Parameters d) ℝ n) (k : ℕ) : Parameters d :=
  (Optim.GD.step (Dataset.uniformStep B lam) (gradient (D.ridgeLoss lam)))^[k] 0

theorem measurableSet_boundedSet {d : ℕ} (B T : ℝ) :
    MeasurableSet (boundedSet d B T) := by
  have h1 : MeasurableSet {z : Parameters d × ℝ | ‖z.1‖ ≤ B} :=
    measurableSet_le (continuous_norm.comp continuous_fst).measurable measurable_const
  have h2 : MeasurableSet {z : Parameters d × ℝ | |z.2| ≤ T} :=
    measurableSet_le ((continuous_abs.comp continuous_snd).measurable) measurable_const
  simpa [boundedSet, Set.ofPred_and] using h1.inter h2

theorem measurable_normalizedLoss {d : ℕ} (R B T : ℝ) (θ : Parameters d) :
    Measurable (normalizedLoss R B T θ) := by
  by_cases hscale : 0 < scale R B T
  · have hcont : Continuous fun z : Parameters d × ℝ =>
        (⟪θ, z.1⟫ - z.2) ^ 2 / scale R B T := by
      have hinner : Continuous fun z : Parameters d × ℝ => ⟪θ, z.1⟫ :=
        Continuous.inner continuous_const continuous_fst
      exact ((hinner.sub continuous_snd).pow 2).div_const (scale R B T)
    have hfun : normalizedLoss R B T θ = fun z =>
        if z ∈ boundedSet d B T then (⟪θ, z.1⟫ - z.2) ^ 2 / scale R B T else 0 := by
      funext z
      by_cases hz : z ∈ boundedSet d B T
      · simp [normalizedLoss, hz, hscale]
      · simp [normalizedLoss, hz]
    rw [hfun]
    exact Measurable.ite (measurableSet_boundedSet B T) hcont.measurable measurable_const
  · have hzero : normalizedLoss R B T θ = fun _ => 0 := by
      funext z
      simp only [normalizedLoss]
      split_ifs with hz
      · exact absurd hz.2 hscale
      · rfl
    rw [hzero]
    exact measurable_const

theorem normalizedLoss_nonneg {d : ℕ} (R B T : ℝ) (θ : Parameters d) (z : Parameters d × ℝ) :
    0 ≤ normalizedLoss R B T θ z := by
  unfold normalizedLoss
  split_ifs with hz
  · exact div_nonneg (sq_nonneg _) hz.2.le
  · exact le_rfl

/-- On the bounded observation set, a parameter in the training ball has
prediction error at most `2RB + T`. -/
theorem prediction_abs_le {d : ℕ} {R B T : ℝ} {θ : Parameters d} {z : Parameters d × ℝ}
    (hR : 0 ≤ R) (hθ : ‖θ‖ ≤ 2 * R) (hx : ‖z.1‖ ≤ B) (hy : |z.2| ≤ T) :
    |⟪θ, z.1⟫ - z.2| ≤ 2 * R * B + T := by
  have hinner : |⟪θ, z.1⟫| ≤ 2 * R * B := by
    calc
      |⟪θ, z.1⟫| ≤ ‖θ‖ * ‖z.1‖ := abs_real_inner_le_norm θ z.1
      _ ≤ 2 * R * B := mul_le_mul hθ hx (norm_nonneg _) (mul_nonneg (by norm_num) hR)
  rcases abs_le.mp hinner with ⟨hlo, hhi⟩
  rcases abs_le.mp hy with ⟨ylo, yhi⟩
  rw [abs_le]
  constructor <;> linarith

theorem normalizedLoss_le_one {d : ℕ} (R B T : ℝ) (hR : 0 ≤ R) (hB : 0 ≤ B) (hT : 0 ≤ T)
    {θ : Parameters d} (hθ : ‖θ‖ ≤ 2 * R) (z : Parameters d × ℝ) :
    normalizedLoss R B T θ z ≤ 1 := by
  unfold normalizedLoss
  split_ifs with hz
  · rcases hz with ⟨⟨hx, hy⟩, hscale⟩
    have herr := prediction_abs_le hR hθ hx hy
    have hnn : 0 ≤ 2 * R * B + T :=
      add_nonneg (mul_nonneg (mul_nonneg (by norm_num) hR) hB) hT
    have hsq : (⟪θ, z.1⟫ - z.2) ^ 2 ≤ scale R B T := by
      have hcmp : |⟪θ, z.1⟫ - z.2| ≤ |2 * R * B + T| := by rwa [abs_of_nonneg hnn]
      simpa [scale] using (sq_le_sq).mpr hcmp
    exact (div_le_one hscale).mpr hsq
  · exact zero_le_one

theorem normalizedLoss_mem_Icc {d : ℕ} (R B T : ℝ) (hR : 0 ≤ R) (hB : 0 ≤ B) (hT : 0 ≤ T)
    {θ : Parameters d} (hθ : ‖θ‖ ≤ 2 * R) (z : Parameters d × ℝ) :
    normalizedLoss R B T θ z ∈ Set.Icc (0 : ℝ) 1 :=
  ⟨normalizedLoss_nonneg R B T θ z, normalizedLoss_le_one R B T hR hB hT hθ z⟩

theorem normalizedLoss_lipschitz {d : ℕ} (R B T ρ : ℝ) (hR : 0 ≤ R) (hB : 0 ≤ B)
    (hT : 0 ≤ T) (hρ : 0 ≤ ρ) {θ φ : Parameters d}
    (hθ : ‖θ‖ ≤ 2 * R) (hφ : ‖φ‖ ≤ 2 * R) (hdist : ‖θ - φ‖ ≤ netRadius R B T ρ)
    (z : Parameters d × ℝ) :
    |normalizedLoss R B T θ z - normalizedLoss R B T φ z| ≤ ρ := by
  by_cases hscale : 0 < scale R B T
  · by_cases hz : z ∈ boundedSet d B T
    · have hloss : normalizedLoss R B T θ z =
          (⟪θ, z.1⟫ - z.2) ^ 2 / scale R B T := by simp [normalizedLoss, hz, hscale]
      have hloss' : normalizedLoss R B T φ z =
          (⟪φ, z.1⟫ - z.2) ^ 2 / scale R B T := by simp [normalizedLoss, hz, hscale]
      rw [hloss, hloss']
      rcases hz with ⟨hx, hy⟩
      by_cases hBpos : 0 < B
      · have hSnn : 0 ≤ 2 * R * B + T :=
          add_nonneg (mul_nonneg (mul_nonneg (by norm_num) hR) hB) hT
        have hSpos : 0 < 2 * R * B + T := by
          have hne : 2 * R * B + T ≠ 0 := by
            intro hzero
            have : scale R B T = 0 := by simp [scale, hzero]
            linarith
          exact lt_of_le_of_ne hSnn (Ne.symm hne)
        have habs (ψ : Parameters d) (hψ : ‖ψ‖ ≤ 2 * R) :
            |⟪ψ, z.1⟫ - z.2| ≤ 2 * R * B + T :=
          prediction_abs_le hR hψ hx hy
        set a := ⟪θ, z.1⟫ - z.2
        set b := ⟪φ, z.1⟫ - z.2
        have hdiff : |a - b| ≤ ‖θ - φ‖ * B := by
          have heq : a - b = ⟪θ - φ, z.1⟫ := by
            simp only [a, b]
            rw [show ⟪θ, z.1⟫ - z.2 - (⟪φ, z.1⟫ - z.2) = ⟪θ, z.1⟫ - ⟪φ, z.1⟫ by ring]
            rw [inner_sub_left]
          rw [heq]
          exact (abs_real_inner_le_norm (θ - φ) z.1).trans
            (mul_le_mul_of_nonneg_left hx (norm_nonneg _))
        have hadd : |a + b| ≤ |a| + |b| := by
          rcases abs_le.mp (le_rfl : |a| ≤ |a|) with ⟨han, hap⟩
          rcases abs_le.mp (le_rfl : |b| ≤ |b|) with ⟨hbn, hbp⟩
          rw [abs_le]
          constructor <;> linarith
        have hsq : |a ^ 2 - b ^ 2| ≤ ‖θ - φ‖ * B * (2 * (2 * R * B + T)) := by
          have hid : a ^ 2 - b ^ 2 = (a - b) * (a + b) := by ring
          rw [hid, abs_mul]
          have hsum : |a| + |b| ≤ 2 * (2 * R * B + T) := by
            linarith [habs θ hθ, habs φ hφ]
          exact mul_le_mul hdiff (hadd.trans hsum) (abs_nonneg _)
            (mul_nonneg (norm_nonneg _) hB)
        have hdiv : |a ^ 2 / scale R B T - b ^ 2 / scale R B T| =
            |a ^ 2 - b ^ 2| / scale R B T := by
          rw [← sub_div, abs_div, abs_of_pos hscale]
        have hLform : lossLipschitz R B T = 2 * B / (2 * R * B + T) := by
          unfold lossLipschitz
          rw [ite_eq_left ⟨hBpos, hSpos⟩]
        have hLpos : 0 < lossLipschitz R B T := by
          rw [hLform]
          exact div_pos (mul_pos (by norm_num) hBpos) hSpos
        have hrad : netRadius R B T ρ = ρ / lossLipschitz R B T := by
          unfold netRadius
          rw [ite_eq_left hLpos]
        rw [hdiv]
        calc
          |a ^ 2 - b ^ 2| / scale R B T
              ≤ (‖θ - φ‖ * B * (2 * (2 * R * B + T))) / scale R B T :=
            div_le_div_of_nonneg_right hsq hscale.le
          _ = lossLipschitz R B T * ‖θ - φ‖ := by
            rw [hLform, scale]
            field_simp [hSpos.ne']
          _ ≤ lossLipschitz R B T * netRadius R B T ρ :=
            mul_le_mul_of_nonneg_left hdist hLpos.le
          _ = ρ := by rw [hrad]; field_simp [hLpos.ne']
      · have hB0 : B = 0 := le_antisymm (le_of_not_gt hBpos) hB
        have hz0 : z.1 = 0 := by
          apply norm_eq_zero.mp
          exact le_antisymm (by simpa [hB0] using hx) (norm_nonneg _)
        have hsame : ⟪θ, z.1⟫ = ⟪φ, z.1⟫ := by simp [hz0]
        rw [hsame, sub_self, abs_zero]
        exact hρ
    · have h0 : normalizedLoss R B T θ z = 0 := by simp [normalizedLoss, hz]
      have h0' : normalizedLoss R B T φ z = 0 := by simp [normalizedLoss, hz]
      simp [h0, h0', hρ]
  · have h0 : normalizedLoss R B T θ z = 0 := by
      simp only [normalizedLoss]
      split_ifs with hz
      · exact absurd hz.2 hscale
      · rfl
    have h0' : normalizedLoss R B T φ z = 0 := by
      simp only [normalizedLoss]
      split_ifs with hz
      · exact absurd hz.2 hscale
      · rfl
    simp [h0, h0', hρ]

theorem sample_unbounded_measure_zero {d n : ℕ} (P : Population (Parameters d × ℝ))
    (B T : ℝ) (hsupp : P.law (boundedSet d B T)ᶜ = 0) :
    sampleLaw P.law n {s | ∃ i, s i ∉ boundedSet d B T} = 0 := by
  have hslice : ∀ i, sampleLaw P.law n
      ((fun s : Fin n → Parameters d × ℝ => s i) ⁻¹' (boundedSet d B T)ᶜ) = 0 := by
    intro i
    exact (MeasurePreserving.measure_preimage
      (measurePreserving_eval_sampleLaw (D := P.law) (n := n) i)
      ((measurableSet_boundedSet B T).compl.nullMeasurableSet)).trans hsupp
  have hsub : {s | ∃ i, s i ∉ boundedSet d B T} ⊆
      ⋃ i, (fun s : Fin n → Parameters d × ℝ => s i) ⁻¹' (boundedSet d B T)ᶜ := by
    rintro s ⟨i, hi⟩
    refine Set.mem_iUnion.mpr ⟨i, ?_⟩
    simpa [Set.mem_preimage, Set.mem_compl_iff] using hi
  refine le_antisymm (le_trans (measure_mono hsub) ?_) bot_le
  refine le_trans (measure_iUnion_le _) ?_
  simp only [hslice, tsum_zero, le_rfl]

theorem empirical_normalized_eq {d n : ℕ} (R B T : ℝ) (hscale : 0 < scale R B T)
    (D : Dataset (Parameters d) ℝ n) (hD : ∀ i, D i ∈ boundedSet d B T) (θ : Parameters d) :
    empRisk (normalizedLoss R B T) D θ =
      (∑ i, (⟪θ, (D i).1⟫ - (D i).2) ^ 2) / (n * scale R B T) := by
  have hpt : ∀ i, normalizedLoss R B T θ (D i) =
      (⟪θ, (D i).1⟫ - (D i).2) ^ 2 / scale R B T := by
    intro i
    simp [normalizedLoss, hD i, hscale]
  simp only [empRisk, hpt, div_eq_mul_inv]
  rw [← Finset.sum_mul, ← mul_assoc, mul_comm ((n : ℝ)⁻¹), mul_assoc, ← mul_inv,
    ← div_eq_mul_inv]

/-- Zero-initialized exact gradient descent on bounded data, run for `k` steps
with the input-uniform step, has normalized excess population risk at most the
stated optimization-plus-deviation budget, outside an explicit Hoeffding tail. -/
theorem trained_ridge_tail {d n : ℕ} (P : Population (Parameters d × ℝ))
    (B T lam R ε ρ : ℝ) (k : ℕ)
    (hB : 0 ≤ B) (hT : 0 ≤ T) (hlam : 0 < lam) (hR : 0 ≤ R)
    (hTR : T ^ 2 ≤ lam * R ^ 2) (hn : 1 ≤ n) (hε : 0 < ε) (hρ : 0 < ρ)
    (hsupp : P.law (boundedSet d B T)ᶜ = 0) :
    sampleLaw P.law n {s | ∃ φ : Parameters d, ‖φ‖ ≤ R ∧
        P.risk (normalizedLoss R B T φ) + excessBudget B lam R ε ρ k (scale R B T) <
          P.risk (normalizedLoss R B T (iterate B lam s k))} ≤
      ENNReal.ofReal (2 * ((BallCover.coverAxis (2 * R) (netRadius R B T ρ) d + 1) ^ d) *
        Real.exp (-2 * n * ε ^ 2)) := by
  have hn0 : 0 < n := by omega
  by_cases hscale : 0 < scale R B T
  · have hδ : 0 < netRadius R B T ρ := by
      unfold netRadius
      split_ifs with hL
      · exact div_pos hρ hL
      · norm_num
    obtain ⟨models, hmodels, hcard, hnet⟩ :=
      BallCover.euclideanBall_net (rad := 2 * R) (δ := netRadius R B T ρ)
        (mul_nonneg (by norm_num) hR) hδ
    have hcover : ∀ h ∈ paramBall d (2 * R), ∃ g ∈ models,
        ∀ z, |normalizedLoss R B T h z - normalizedLoss R B T g z| ≤ ρ := by
      intro h hh
      obtain ⟨g, hg, hdist⟩ := hnet h hh
      exact ⟨g, hg, fun z =>
        normalizedLoss_lipschitz R B T ρ hR hB hT hρ.le hh (hmodels g hg) hdist z⟩
    have htail := Covering.uniform_deviation_tail (P := P) (K := paramBall d (2 * R))
      models (normalizedLoss R B T) hmodels
      (fun h hh z => normalizedLoss_mem_Icc R B T hR hB hT hh z)
      (fun h _ => measurable_normalizedLoss R B T h) ρ ε hcover hn hε
    have hunb := sample_unbounded_measure_zero (n := n) P B T hsupp
    let deviation := {s : Fin n → Parameters d × ℝ | ∃ h ∈ paramBall d (2 * R),
      ε + 2 * ρ < |empRisk (normalizedLoss R B T) s h -
        P.risk (normalizedLoss R B T h)|}
    let unbounded := {s : Fin n → Parameters d × ℝ | ∃ i, s i ∉ boundedSet d B T}
    let excess := {s : Fin n → Parameters d × ℝ | ∃ φ : Parameters d, ‖φ‖ ≤ R ∧
      P.risk (normalizedLoss R B T φ) + excessBudget B lam R ε ρ k (scale R B T) <
        P.risk (normalizedLoss R B T (iterate B lam s k))}
    have hsub : excess ⊆ unbounded ∪ deviation := by
      intro s hs
      by_contra hnone
      have hbounded : ∀ i, s i ∈ boundedSet d B T := by
        have : s ∉ unbounded := fun hu => hnone (Or.inl hu)
        simpa [unbounded, not_exists, not_not] using this
      have hdev : ∀ θ ∈ paramBall d (2 * R),
          |empRisk (normalizedLoss R B T) s θ - P.risk (normalizedLoss R B T θ)| ≤
            ε + 2 * ρ := by
        have : s ∉ deviation := fun hd => hnone (Or.inr hd)
        simpa [deviation, not_exists, not_and, not_lt] using this
      rcases hs with ⟨φ, hφ, hbad⟩
      have hφball : φ ∈ paramBall d (2 * R) := by
        change ‖φ‖ ≤ 2 * R
        linarith
      have hBx : ∀ i, ‖(s i).1‖ ≤ B := fun i => (hbounded i).1
      have hBy : ∀ i, |(s i).2| ≤ T := fun i => (hbounded i).2
      have hη := Dataset.uniformStep_pos hB hlam
      have hstep := Dataset.uniformStep_admissible s hn0 B lam hB hlam hBx
      rcases Dataset.ridge_training_radius s hn0 T hT hBy lam R hlam hR hTR
          (Dataset.uniformStep B lam) hη hstep with ⟨θStar, hmin, hstar, hiter⟩
      have hgrad := Ridge.minimizer_grad_eq_zero (Dataset.observation s) (Dataset.labels s) (1 / n) lam
        (by positivity) hlam hmin
      have hg : gradient (Dataset.ridgeLoss s lam) =
          Ridge.grad (Dataset.observation s) (Dataset.labels s) (1 / n) lam :=
        funext (Ridge.gradient_eq (Dataset.observation s) (Dataset.labels s) (1 / n) lam)
      have hμL : lam ≤ (Ridge.smoothness (Dataset.observation s) (1 / n) lam : ℝ) := by
        simp only [Ridge.smoothness, NNReal.coe_add, coe_nnnorm, Real.norm_eq_abs]
        linarith [norm_nonneg (Ridge.hessian (Dataset.observation s) (1 / n) lam), le_abs_self lam]
      have hdist := Optim.GD.dist_sq_iterate_le_of_step_size (Dataset.uniformStep B lam) lam
        (Ridge.grad (Dataset.observation s) (Dataset.labels s) (1 / n) lam)
        (Ridge.stronglyMonotone (Dataset.observation s) (Dataset.labels s) (1 / n) lam (by positivity))
        ((Ridge.grad_lipschitz (Dataset.observation s) (Dataset.labels s) (1 / n) lam).weaken
          (by simp [Ridge.smoothness]))
        (xStar := θStar) (x := 0) hgrad hlam.le hμL hη hstep k
      have hqnn := Optim.GD.q_nonneg_of_le (Dataset.uniformStep B lam) lam
        (Ridge.smoothness (Dataset.observation s) (1 / n) lam) hlam.le hμL
      have hq := Dataset.data_q_le_uniformFactor s hn0 B lam hB hlam hBx
      have hgap : Dataset.ridgeLoss s lam (iterate B lam s k) - Dataset.ridgeLoss s lam θStar ≤
          ((B ^ 2 + lam) / 2) * Dataset.uniformFactor B lam ^ k * R ^ 2 := by
        have hdistR : ‖iterate B lam s k - θStar‖ ^ 2 ≤
            Dataset.uniformFactor B lam ^ k * R ^ 2 := by
          have hstarSq : ‖θStar‖ ^ 2 ≤ R ^ 2 :=
            (sq_le_sq₀ (norm_nonneg _) hR).mpr (by simpa [abs_of_nonneg hR] using hstar)
          have hpow : (Optim.GD.q (Dataset.uniformStep B lam) lam
              (Ridge.smoothness (Dataset.observation s) (1 / n) lam)) ^ k ≤
              Dataset.uniformFactor B lam ^ k := pow_le_pow_left₀ hqnn hq k
          calc
            _ ≤ (Optim.GD.q (Dataset.uniformStep B lam) lam
                (Ridge.smoothness (Dataset.observation s) (1 / n) lam)) ^ k * ‖θStar‖ ^ 2 := by
              simpa [iterate, hg, zero_sub, norm_neg] using hdist.1
            _ ≤ (Optim.GD.q (Dataset.uniformStep B lam) lam
                (Ridge.smoothness (Dataset.observation s) (1 / n) lam)) ^ k * R ^ 2 :=
              mul_le_mul_of_nonneg_left hstarSq (pow_nonneg hqnn _)
            _ ≤ _ := mul_le_mul_of_nonneg_right hpow (sq_nonneg _)
        have hcoeff : (1 / n) / 2 * ‖(Dataset.observation s)‖ ^ 2 + lam / 2 ≤ (B ^ 2 + lam) / 2 := by
          have hA := Dataset.observation_opNorm_le s B hB hBx
          have hAsq : ‖(Dataset.observation s)‖ ^ 2 ≤ (n : ℝ) * B ^ 2 := by
            have hnn : 0 ≤ Real.sqrt (n : ℝ) * B := mul_nonneg (Real.sqrt_nonneg _) hB
            have := (sq_le_sq₀ (norm_nonneg _) hnn).mpr hA
            simpa [mul_pow, Real.sq_sqrt (Nat.cast_nonneg n)] using this
          have hleft : (1 / n) / 2 * ‖(Dataset.observation s)‖ ^ 2 ≤ B ^ 2 / 2 := by
            calc
              _ ≤ (1 / n) / 2 * ((n : ℝ) * B ^ 2) :=
                mul_le_mul_of_nonneg_left hAsq (by positivity)
              _ = B ^ 2 / 2 := by field_simp
          linarith
        calc
          _ ≤ ((1 / n) / 2 * ‖(Dataset.observation s)‖ ^ 2 + lam / 2) *
              ‖iterate B lam s k - θStar‖ ^ 2 :=
            Ridge.objective_gap_le (Dataset.observation s) (Dataset.labels s) (1 / n) lam (by positivity) hgrad _
          _ ≤ ((B ^ 2 + lam) / 2) * ‖iterate B lam s k - θStar‖ ^ 2 :=
            mul_le_mul_of_nonneg_right hcoeff (sq_nonneg _)
          _ ≤ ((B ^ 2 + lam) / 2) * (Dataset.uniformFactor B lam ^ k * R ^ 2) :=
            mul_le_mul_of_nonneg_left hdistR
              (div_nonneg (by nlinarith [sq_nonneg B, hlam.le]) (by norm_num))
          _ = _ := by ring
      have hemp : empRisk (normalizedLoss R B T) s (iterate B lam s k) -
          empRisk (normalizedLoss R B T) s φ ≤
          ((B ^ 2 + lam) * Dataset.uniformFactor B lam ^ k * R ^ 2 + lam * R ^ 2) /
            scale R B T := by
        set θk := iterate B lam s k
        rw [empirical_normalized_eq R B T hscale s hbounded θk,
          empirical_normalized_eq R B T hscale s hbounded φ]
        have hφnorm : ‖φ‖ ^ 2 ≤ R ^ 2 := (sq_le_sq₀ (norm_nonneg _) hR).mpr hφ
        have hJdiff : Dataset.ridgeLoss s lam θk - Dataset.ridgeLoss s lam φ ≤
            ((B ^ 2 + lam) / 2) * Dataset.uniformFactor B lam ^ k * R ^ 2 := by
          linarith [hmin φ, hgap]
        have hn' : (n : ℝ) ≠ 0 := by exact_mod_cast hn0.ne'
        have hsq : (∑ i, (⟪θk, (s i).1⟫ - (s i).2) ^ 2) / (n : ℝ) -
            (∑ i, (⟪φ, (s i).1⟫ - (s i).2) ^ 2) / (n : ℝ) ≤
            (B ^ 2 + lam) * Dataset.uniformFactor B lam ^ k * R ^ 2 + lam * R ^ 2 := by
          have hk := Dataset.ridgeLoss_eq s lam θk
          have hp := Dataset.ridgeLoss_eq s lam φ
          have halg : (∑ i, (⟪θk, (s i).1⟫ - (s i).2) ^ 2) / (n : ℝ) -
              (∑ i, (⟪φ, (s i).1⟫ - (s i).2) ^ 2) / (n : ℝ) =
              2 * (Dataset.ridgeLoss s lam θk - Dataset.ridgeLoss s lam φ) -
                lam * ‖θk‖ ^ 2 + lam * ‖φ‖ ^ 2 := by
            rw [hk, hp]
            field_simp [hn']
            ring
          nlinarith [sq_nonneg ‖θk‖, hφnorm]
        have hdiv : (∑ i, (⟪θk, (s i).1⟫ - (s i).2) ^ 2) / ((n : ℝ) * scale R B T) -
            (∑ i, (⟪φ, (s i).1⟫ - (s i).2) ^ 2) / ((n : ℝ) * scale R B T) =
            ((∑ i, (⟪θk, (s i).1⟫ - (s i).2) ^ 2) / (n : ℝ) -
              (∑ i, (⟪φ, (s i).1⟫ - (s i).2) ^ 2) / (n : ℝ)) / scale R B T := by
          rw [← div_div, ← div_div, ← sub_div]
        rw [hdiv]
        exact div_le_div_of_nonneg_right hsq hscale.le
      have hpop : P.risk (normalizedLoss R B T (iterate B lam s k)) ≤
          P.risk (normalizedLoss R B T φ) + excessBudget B lam R ε ρ k (scale R B T) := by
        have hiterBall : iterate B lam s k ∈ paramBall d (2 * R) := by
          simpa [paramBall, iterate] using hiter k
        rcases abs_le.mp (hdev _ hiterBall) with ⟨hklo, hkhi⟩
        rcases abs_le.mp (hdev _ hφball) with ⟨hφlo, hφhi⟩
        have hbudget : excessBudget B lam R ε ρ k (scale R B T) =
            2 * (ε + 2 * ρ) +
              ((B ^ 2 + lam) * Dataset.uniformFactor B lam ^ k * R ^ 2 + lam * R ^ 2) /
                scale R B T := by
          simp [excessBudget, ite_eq_left hscale]
        rw [hbudget]
        linarith
      linarith
    calc
      sampleLaw P.law n excess
          ≤ sampleLaw P.law n (unbounded ∪ deviation) := measure_mono hsub
      _ ≤ sampleLaw P.law n unbounded + sampleLaw P.law n deviation := measure_union_le _ _
      _ = sampleLaw P.law n deviation := by rw [hunb, zero_add]
      _ ≤ ENNReal.ofReal (2 * models.card * Real.exp (-2 * n * ε ^ 2)) := by
        simpa [deviation] using htail
      _ ≤ ENNReal.ofReal (2 * ((BallCover.coverAxis (2 * R) (netRadius R B T ρ) d + 1) ^ d) *
          Real.exp (-2 * n * ε ^ 2)) := by
        apply ENNReal.ofReal_le_ofReal
        exact mul_le_mul_of_nonneg_right
          (mul_le_mul_of_nonneg_left (by exact_mod_cast hcard) (by norm_num))
          (Real.exp_nonneg _)
  · have hzero : ∀ (θ : Parameters d) (z : Parameters d × ℝ),
        normalizedLoss R B T θ z = 0 := by
      intro θ z
      simp only [normalizedLoss]
      split_ifs with hz
      · exact absurd hz.2 hscale
      · rfl
    have hempty : {s : Fin n → Parameters d × ℝ | ∃ φ : Parameters d, ‖φ‖ ≤ R ∧
        P.risk (normalizedLoss R B T φ) + excessBudget B lam R ε ρ k (scale R B T) <
          P.risk (normalizedLoss R B T (iterate B lam s k))} = ∅ := by
      ext s
      simp only [Set.mem_empty_iff_false, iff_false]
      intro hbad
      rcases hbad with ⟨φ, _, hlt⟩
      have hpop : ∀ θ : Parameters d, P.risk (normalizedLoss R B T θ) = 0 := by
        intro θ
        simp only [Population.risk]
        rw [show (fun z => normalizedLoss R B T θ z) = fun _ => 0 by
          funext z; exact hzero θ z, integral_zero]
      have hbudget : 0 ≤ excessBudget B lam R ε ρ k (scale R B T) := by
        unfold excessBudget
        rw [ite_eq_right hscale]
        nlinarith
      rw [hpop φ, hpop (iterate B lam s k)] at hlt
      linarith
    simp [hempty, measure_empty]

end OpenTheory.TrainedRidge

