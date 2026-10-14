import OpenTheory.Dynamics.MLPResidual
import OpenTheory.Generalization.PredictionGap

/-! A closed, restricted statistical guarantee for actual nonlinear training.
One fixed input; frozen readout; initialization-centered W₁ regularization;
bounded, centered, finitely supported label noise. No residual or prediction
gap is assumed. This is not a multi-input or unseen-input generalization bound. -/
noncomputable section
open scoped BigOperators
namespace OpenTheory.MLPRisk
open FeatureStep MLPTrajectory MLPComparison MLPEndpoint MLPResidual
variable {d m : ℕ}

def gap (T m : ℕ) (η A L R M E K q S : ℝ) : ℝ :=
  let D : ℝ := T * (η * R * A * L * E)
  let B := M * m * A * D ^ 2
  let G := m * A ^ 2 * L * M * D * E
  B + T * η * (B * |K| + R * G) + q ^ T * S

theorem endpoint_initial_gap (net : TwoLayerNet d m) (x : Fin d → ℝ)
    (y lam : ℝ) (hlam : 0 < lam) :
    |net.forward x - endpoint net x y lam| ≤ |net.forward x - y| := by
  have hK := kernel_nonneg net x
  have hd : 0 < kernel net x + lam := by linarith
  have ha : 0 ≤ kernel net x / (kernel net x + lam) := div_nonneg hK hd.le
  have hb : kernel net x / (kernel net x + lam) ≤ 1 :=
    (div_le_one hd).mpr (by linarith)
  have he : net.forward x - endpoint net x y lam =
      (kernel net x / (kernel net x + lam)) * (net.forward x - y) := by
    unfold endpoint
    ring
  rw [he, abs_mul, abs_of_nonneg ha]
  nlinarith [abs_nonneg (net.forward x - y)]

/-- The deterministic gap is fully discharged from initialization bounds. -/
theorem trained_endpoint_gap (net : TwoLayerNet d m) (x : Fin d → ℝ)
    (y η lam : ℝ) {A L M S : ℝ} (hlam : 0 < lam)
    (hη : 0 ≤ η) (hA : 0 ≤ A) (hL : 0 ≤ L) (hM : 0 ≤ M) (hS : 0 ≤ S)
    (hdecay : |1 - η * lam| ≤ 1)
    (hstep : |1 - η * (kernel net x + lam)| ≤ 1)
    (hW : ∀ i, |net.W₂ 0 i| ≤ A) (hσ : ∀ u, |deriv net.act.σ u| ≤ L)
    (hLip : ∀ a b, |deriv net.act.σ a - deriv net.act.σ b| ≤ M * |a - b|)
    (hinit : |net.forward x - y| ≤ S) (T : ℕ) :
    |(atDisplacement net (trajectory net x y η lam T)).forward x - endpoint net x y lam| ≤
      gap T m η A L (envelope m η A L (inputEnergy x) S T) M (inputEnergy x)
        (kernel net x) |1 - η * (kernel net x + lam)| S := by
  have hE := inputEnergy_nonneg x
  have hR : 0 ≤ envelope m η A L (inputEnergy x) S T := by
    unfold envelope growth
    positivity
  have hc := nonlinear_endpoint_gap net x y η lam hlam hη hA hL hR hM
    hdecay hstep hW hσ hLip T
    (fun t ht => residual_le net x y η lam hη hA hL hS hdecay hW hσ hinit T t ht.le)
  apply hc.trans
  dsimp only [gap]
  exact add_le_add le_rfl (mul_le_mul_of_nonneg_left
    ((endpoint_initial_gap net x y lam hlam).trans hinit) (by positivity))

/-- Exact bias–variance identity for any finite probability distribution. -/
theorem affine_noise_mse {Ω : Type*} [Fintype Ω] (p z : Ω → ℝ)
    (hmass : ∑ w, p w = 1) (hmean : ∑ w, p w * z w = 0)
    (σ : ℝ) (hvar : ∑ w, p w * z w ^ 2 = σ ^ 2) (b a : ℝ) :
    ∑ w, p w * (b + a * z w) ^ 2 = b ^ 2 + a ^ 2 * σ ^ 2 := by
  have he (w : Ω) : p w * (b + a * z w) ^ 2 =
      b ^ 2 * p w + (2 * b * a) * (p w * z w) + a ^ 2 * (p w * z w ^ 2) := by ring
  simp_rw [he, Finset.sum_add_distrib, ← Finset.mul_sum]
  rw [hmass, hmean, hvar]
  ring

/-- Expected squared error of the ACTUAL T-step nonlinear MLP. The expectation
is the finite probability sum with masses `p`. Labels are `target + z`.
All residual and ridge-gap bounds are conclusions, not hypotheses. -/
theorem expected_mse (net : TwoLayerNet d m) (x : Fin d → ℝ)
    (target η lam : ℝ) {A L M Z : ℝ} (hlam : 0 < lam)
    (hη : 0 ≤ η) (hA : 0 ≤ A) (hL : 0 ≤ L) (hM : 0 ≤ M) (hZ : 0 ≤ Z)
    (hdecay : |1 - η * lam| ≤ 1)
    (hstep : |1 - η * (kernel net x + lam)| ≤ 1)
    (hW : ∀ i, |net.W₂ 0 i| ≤ A) (hσ : ∀ u, |deriv net.act.σ u| ≤ L)
    (hLip : ∀ a b, |deriv net.act.σ a - deriv net.act.σ b| ≤ M * |a - b|)
    {Ω : Type*} [Fintype Ω] (p z : Ω → ℝ) (hp : ∀ w, 0 ≤ p w)
    (hmass : ∑ w, p w = 1) (hmean : ∑ w, p w * z w = 0)
    (σ : ℝ) (hvar : ∑ w, p w * z w ^ 2 = σ ^ 2)
    (hz : ∀ w, |z w| ≤ Z) (T : ℕ) :
    let S := |net.forward x - target| + Z
    let R := envelope m η A L (inputEnergy x) S T
    let δ := gap T m η A L R M (inputEnergy x) (kernel net x)
      |1 - η * (kernel net x + lam)| S
    ∑ w, p w * ((atDisplacement net
      (trajectory net x (target + z w) η lam T)).forward x - target) ^ 2 ≤
      2 * ((lam / (kernel net x + lam)) ^ 2 * (net.forward x - target) ^ 2 +
        (kernel net x / (kernel net x + lam)) ^ 2 * σ ^ 2) + 2 * δ ^ 2 := by
  dsimp only
  let S := |net.forward x - target| + Z
  let δ := gap T m η A L (envelope m η A L (inputEnergy x) S T) M (inputEnergy x)
    (kernel net x) |1 - η * (kernel net x + lam)| S
  have hS : 0 ≤ S := by dsimp [S]; positivity
  have hclose (w : Ω) : |(atDisplacement net
      (trajectory net x (target + z w) η lam T)).forward x -
      endpoint net x (target + z w) lam| ≤ δ := by
    apply trained_endpoint_gap net x (target + z w) η lam hlam hη hA hL hM hS
      hdecay hstep hW hσ hLip _ T
    apply (abs_sub_le (net.forward x) target (target + z w)).trans
    simpa [S] using add_le_add (le_refl |net.forward x - target|) (hz w)
  have hδ : 0 ≤ δ := by
    have hE := inputEnergy_nonneg x
    dsimp [δ, gap, envelope, growth]
    positivity
  have hpoint (w : Ω) : ((atDisplacement net
      (trajectory net x (target + z w) η lam T)).forward x - target) ^ 2 ≤
      2 * (endpoint net x (target + z w) lam - target) ^ 2 + 2 * δ ^ 2 := by
    simpa only [Real.norm_eq_abs, sq_abs] using PredictionGap.sq_norm_le_double
      ((atDisplacement net (trajectory net x (target + z w) η lam T)).forward x)
      (endpoint net x (target + z w) lam) target hδ (hclose w)
  have hd : kernel net x + lam ≠ 0 := ne_of_gt (by linarith [kernel_nonneg net x])
  have he (w : Ω) : endpoint net x (target + z w) lam - target =
      lam / (kernel net x + lam) * (net.forward x - target) +
        (kernel net x / (kernel net x + lam)) * z w := by
    unfold endpoint
    field_simp
    ring
  have hr : (∑ w, p w * (endpoint net x (target + z w) lam - target) ^ 2) =
      (lam / (kernel net x + lam)) ^ 2 * (net.forward x - target) ^ 2 +
        (kernel net x / (kernel net x + lam)) ^ 2 * σ ^ 2 := by
    simp_rw [he]
    rw [affine_noise_mse p z hmass hmean σ hvar, mul_pow]
  have hs := Finset.sum_le_sum (s := Finset.univ)
    (fun w _ => mul_le_mul_of_nonneg_left (hpoint w) (hp w))
  have hsum : (∑ w, p w * (2 * (endpoint net x (target + z w) lam - target) ^ 2 +
      2 * δ ^ 2)) = 2 * (∑ w, p w * (endpoint net x (target + z w) lam - target) ^ 2) +
        2 * δ ^ 2 := by
    simp_rw [mul_add, ← mul_assoc, mul_comm (p _) 2, mul_assoc,
      Finset.sum_add_distrib, ← Finset.mul_sum, ← Finset.sum_mul, hmass]
    ring
  rw [hsum, hr] at hs
  exact hs

end OpenTheory.MLPRisk
