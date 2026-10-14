import OpenTheory.Data.Foundation
import Mathlib.Analysis.InnerProductSpace.PiL2
import Mathlib.Tactic

/-! Explicit finite nets for Euclidean balls. The cardinality is a convenient
grid bound, not the optimal covering number. -/
noncomputable section
open scoped BigOperators RealInnerProductSpace

namespace OpenTheory.BallCover

/-- Grid resolution. Flooring each cell gives a coordinate gap of one full step,
so the factor `2` is part of this estimate. -/
def coverAxis (rad δ : ℝ) (d : ℕ) : ℕ :=
  max 1 ⌈2 * rad * Real.sqrt (d : ℝ) / δ⌉₊

def axisCoord (rad : ℝ) (k : ℕ) (i : Fin (k + 1)) : ℝ :=
  -rad + (2 * rad) * (i : ℝ) / (k : ℝ)

def gridPoint {d : ℕ} (rad : ℝ) (k : ℕ) (f : Fin d → Fin (k + 1)) : Parameters d :=
  WithLp.toLp 2 (fun j => axisCoord rad k (f j))

/-- Radial retraction onto the closed ball. -/
def radialProj {d : ℕ} (rad : ℝ) (x : Parameters d) : Parameters d :=
  if ‖x‖ ≤ rad then x else (rad / ‖x‖) • x

theorem radialProj_norm_le {d : ℕ} {rad : ℝ} (hrad : 0 ≤ rad) (x : Parameters d) :
    ‖radialProj rad x‖ ≤ rad := by
  unfold radialProj
  split_ifs with h
  · exact h
  · have hx : rad < ‖x‖ := not_le.mp h
    have hxpos : 0 < ‖x‖ := lt_of_le_of_lt hrad hx
    rw [norm_smul, Real.norm_eq_abs, abs_of_nonneg (div_nonneg hrad hxpos.le)]
    field_simp
    exact le_rfl

theorem radialProj_dist_le {d : ℕ} {rad : ℝ} (hrad : 0 ≤ rad) {x y : Parameters d}
    (hy : ‖y‖ ≤ rad) : ‖y - radialProj rad x‖ ≤ ‖y - x‖ := by
  unfold radialProj
  split_ifs with hx
  · rfl
  · have hxlt : rad < ‖x‖ := not_le.mp hx
    have hxpos : 0 < ‖x‖ := lt_of_le_of_lt hrad hxlt
    set p : Parameters d := (rad / ‖x‖) • x
    have hdiff : ‖y - x‖ ^ 2 - ‖y - p‖ ^ 2 =
        ‖x‖ ^ 2 - ‖p‖ ^ 2 - 2 * ⟪y, x - p⟫ := by
      rw [norm_sub_sq_real, norm_sub_sq_real, inner_sub_right]
      ring
    have hpnorm : ‖p‖ = rad := by
      rw [norm_smul, Real.norm_eq_abs, abs_of_nonneg (div_nonneg hrad hxpos.le)]
      field_simp
    have hinner : ⟪y, x - p⟫ ≤ rad * (‖x‖ - rad) := by
      have hcoe : x - p = (1 - rad / ‖x‖) • x := by
        simp [p, sub_smul, one_smul]
      rw [hcoe, inner_smul_right]
      have hfac : 0 ≤ 1 - rad / ‖x‖ := by
        rw [sub_nonneg, div_le_one hxpos]
        exact hxlt.le
      have hle : ⟪y, x⟫ ≤ ‖y‖ * ‖x‖ := real_inner_le_norm y x
      have h1 : (1 - rad / ‖x‖) * ⟪y, x⟫ ≤ (1 - rad / ‖x‖) * (‖y‖ * ‖x‖) :=
        mul_le_mul_of_nonneg_left hle hfac
      have h2 : (1 - rad / ‖x‖) * (‖y‖ * ‖x‖) ≤ (1 - rad / ‖x‖) * (rad * ‖x‖) :=
        mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_right hy (norm_nonneg _)) hfac
      calc
        (1 - rad / ‖x‖) * ⟪y, x⟫ ≤ (1 - rad / ‖x‖) * (rad * ‖x‖) := le_trans h1 h2
        _ = rad * (‖x‖ - rad) := by field_simp
    have hsq : ‖y - p‖ ^ 2 ≤ ‖y - x‖ ^ 2 := by
      have hsub : ‖y - x‖ ^ 2 - ‖y - p‖ ^ 2 =
          ‖x‖ ^ 2 - rad ^ 2 - 2 * ⟪y, x - p⟫ := by
        rw [hdiff, hpnorm]
      nlinarith [hinner, sq_nonneg (‖x‖ - rad), hsub]
    exact (sq_le_sq₀ (norm_nonneg _) (norm_nonneg _)).mp hsq

theorem coord_abs_le_norm {d : ℕ} (x : Parameters d) (j : Fin d) : |x j| ≤ ‖x‖ := by
  have hsum : (x j) ^ 2 ≤ ‖x‖ ^ 2 := by
    rw [parameters_norm_sq]
    exact Finset.single_le_sum (fun i _ => sq_nonneg (x i)) (Finset.mem_univ j)
  exact abs_le_of_sq_le_sq hsum (norm_nonneg _)

theorem exists_axis_close {rad : ℝ} {k : ℕ} (hk : 1 ≤ k) (hrad : 0 ≤ rad) {t : ℝ}
    (ht : |t| ≤ rad) : ∃ i : Fin (k + 1), |t - axisCoord rad k i| ≤ 2 * rad / k := by
  by_cases hrad0 : rad = 0
  · refine ⟨⟨0, Nat.succ_pos k⟩, ?_⟩
    have ht0 : t = 0 := by
      rcases abs_le.mp ht with ⟨hlo, hhi⟩
      linarith
    simp [axisCoord, ht0, hrad0]
  · have hradpos : 0 < rad := lt_of_le_of_ne hrad (Ne.symm hrad0)
    have hkpos : (0 : ℝ) < k := by exact_mod_cast (Nat.lt_of_lt_of_le (Nat.succ_pos 0) hk)
    let u : ℝ := (t + rad) * (k : ℝ) / (2 * rad)
    have hu0 : 0 ≤ u := by
      apply div_nonneg
      · exact mul_nonneg (by linarith [abs_le.mp ht]) (Nat.cast_nonneg k)
      · linarith
    have huk : u ≤ k := by
      rw [div_le_iff₀ (by linarith : (0 : ℝ) < 2 * rad)]
      nlinarith [abs_le.mp ht]
    let i0 : ℕ := Nat.floor u
    have hi0 : i0 ≤ k := by
      have hcast : (i0 : ℝ) ≤ k := le_trans (Nat.floor_le hu0) huk
      exact Nat.cast_le.mp hcast
    let i : Fin (k + 1) := ⟨i0, Nat.lt_succ_of_le hi0⟩
    have hfloor : 0 ≤ u - i0 ∧ u - i0 ≤ 1 := by
      constructor
      · exact sub_nonneg.mpr (Nat.floor_le hu0)
      · exact le_of_lt (sub_lt_iff_lt_add'.mpr (Nat.lt_floor_add_one u))
    have hrepr : t = -rad + (2 * rad) * u / k := by
      unfold u
      field_simp
      ring
    refine ⟨i, ?_⟩
    have hcoord : axisCoord rad k i = -rad + (2 * rad) * (i0 : ℝ) / k := rfl
    have hdiff : t - axisCoord rad k i = (2 * rad / k) * (u - i0) := by
      rw [hrepr, hcoord]
      field_simp
      ring
    rw [hdiff, abs_of_nonneg (mul_nonneg (div_nonneg (by linarith) hkpos.le) hfloor.1)]
    calc
      (2 * rad / k) * (u - i0) ≤ (2 * rad / k) * 1 :=
        mul_le_mul_of_nonneg_left hfloor.2 (div_nonneg (by linarith) hkpos.le)
      _ = 2 * rad / k := by ring

/-- Every point of the closed ball lies within `δ` of one of at most
`(coverAxis + 1) ^ d` explicitly constructed net points. -/
theorem euclideanBall_net {d : ℕ} {rad δ : ℝ} (hrad : 0 ≤ rad) (hδ : 0 < δ) :
    ∃ models : Finset (Parameters d),
      (∀ g ∈ models, ‖g‖ ≤ rad) ∧
      models.card ≤ (coverAxis rad δ d + 1) ^ d ∧
      ∀ h : Parameters d, ‖h‖ ≤ rad → ∃ g ∈ models, ‖h - g‖ ≤ δ := by
  let k : ℕ := coverAxis rad δ d
  have hk : 1 ≤ k := by
    unfold k coverAxis
    exact le_max_left _ _
  have hkpos : (0 : ℝ) < k := by exact_mod_cast (Nat.lt_of_lt_of_le (Nat.succ_pos 0) hk)
  let ι := Fin d → Fin (k + 1)
  let models : Finset (Parameters d) :=
    (Finset.univ : Finset ι).image (fun f => radialProj rad (gridPoint rad k f))
  refine ⟨models, ?_, ?_, ?_⟩
  · intro g hg
    rcases Finset.mem_image.mp hg with ⟨_, _, rfl⟩
    exact radialProj_norm_le hrad _
  · calc
      models.card ≤ (Finset.univ : Finset ι).card := Finset.card_image_le
      _ = Fintype.card ι := Finset.card_univ
      _ = (k + 1) ^ d := by
        simp [ι, Fintype.card_fin]
      _ = (coverAxis rad δ d + 1) ^ d := rfl
  · intro h hh
    choose f hf using fun j : Fin d =>
      exists_axis_close hk hrad (coord_abs_le_norm h j |>.trans hh)
    refine ⟨radialProj rad (gridPoint rad k f), ?_, ?_⟩
    · exact Finset.mem_image.mpr ⟨f, Finset.mem_univ _, rfl⟩
    · have hgrid : ‖h - gridPoint rad k f‖ ≤ δ := by
        have hsq : ‖h - gridPoint rad k f‖ ^ 2 ≤ δ ^ 2 := by
          rw [parameters_norm_sq]
          have hcoord : ∀ j : Fin d,
              |(h - gridPoint rad k f) j| ≤ 2 * rad / k := by
            intro j
            have happ : (gridPoint rad k f) j = axisCoord rad k (f j) := by
              simp [gridPoint, PiLp.toLp_apply]
            rw [PiLp.sub_apply, happ]
            exact hf j
          have hsum : ∑ j : Fin d, ((h - gridPoint rad k f) j) ^ 2 ≤
              ∑ _ : Fin d, (2 * rad / k) ^ 2 := by
            apply Finset.sum_le_sum
            intro j _
            have hstep : 0 ≤ 2 * rad / k :=
              div_nonneg (mul_nonneg (by norm_num) hrad) hkpos.le
            exact sq_le_sq.mpr ((hcoord j).trans_eq (abs_of_nonneg hstep).symm)
          have hconst : ∑ _ : Fin d, (2 * rad / k) ^ 2 =
              (d : ℝ) * (2 * rad / k) ^ 2 := by simp
          have hscale : (d : ℝ) * (2 * rad / k) ^ 2 ≤ δ ^ 2 := by
            have hceil : 2 * rad * Real.sqrt (d : ℝ) / δ ≤ k := by
              have hnat : (Nat.ceil (2 * rad * Real.sqrt (d : ℝ) / δ) : ℝ) ≤ k := by
                exact_mod_cast (le_max_right 1 (Nat.ceil (2 * rad * Real.sqrt (d : ℝ) / δ)) :
                  Nat.ceil _ ≤ k)
              exact le_trans (Nat.le_ceil _) hnat
            have hmul : 2 * rad * Real.sqrt (d : ℝ) ≤ k * δ :=
              (div_le_iff₀ hδ).mp hceil
            have hdiv : Real.sqrt (d : ℝ) * (2 * rad / k) ≤ δ := by
              have hform : Real.sqrt (d : ℝ) * (2 * rad / k) =
                  (2 * rad * Real.sqrt (d : ℝ)) / k := by
                field_simp
              rw [hform, div_le_iff₀ hkpos]
              linarith
            have hnn : 0 ≤ 2 * rad / k := div_nonneg (mul_nonneg (by norm_num) hrad) hkpos.le
            calc
              (d : ℝ) * (2 * rad / k) ^ 2
                  = (Real.sqrt (d : ℝ)) ^ 2 * (2 * rad / k) ^ 2 := by
                    rw [Real.sq_sqrt (Nat.cast_nonneg d)]
              _ = (Real.sqrt (d : ℝ) * (2 * rad / k)) ^ 2 := by ring
              _ ≤ δ ^ 2 := (sq_le_sq₀ (mul_nonneg (Real.sqrt_nonneg _) hnn)
                  (le_of_lt hδ)).mpr hdiv
          calc
            _ ≤ (d : ℝ) * (2 * rad / k) ^ 2 := by simpa [hconst] using hsum
            _ ≤ δ ^ 2 := hscale
        exact (sq_le_sq₀ (norm_nonneg _) (le_of_lt hδ)).mp hsq
      exact le_trans (radialProj_dist_le hrad hh) hgrid

end OpenTheory.BallCover
