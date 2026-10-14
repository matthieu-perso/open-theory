import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Algebra.Polynomial.Basic
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Data.Fintype.Card
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring
import OpenTheory.Architectures.Basic
import OpenTheory.Data.Model
import OpenTheory.Data.Sample

namespace OpenTheory

/-- The first `k` coordinates of a vector in `ℝ^d`, using `k ≤ d`. -/
def coordinateLift {d k : ℕ} (hk : k ≤ d) (x : Fin d → ℝ) : Fin k → ℝ :=
  fun i => x ⟨i.val, lt_of_lt_of_le i.isLt hk⟩

/-- `σ` agrees with no polynomial. Universal approximation needs this. -/
def NonPolynomial (σ : ℝ → ℝ) : Prop :=
  ∀ p : Polynomial ℝ, ∃ x, σ x ≠ p.eval x

/-- Average absolute error of `net` against `target` on a sample. -/
noncomputable def empiricalApproxError {d m n : ℕ} (net : TwoLayerNet d m)
    (target : (Fin d → ℝ) → ℝ) (sample : Sample d n) : ℝ :=
  ((n : ℝ)⁻¹) * ∑ i : Fin n, |net.forward (sample.X i) - target (sample.X i)|

/-- Average absolute error of a signal-subspace network `net_sig` against a
    signal-subspace target `target : (Fin model.signalRank → ℝ) → ℝ` on the projected sample. -/
noncomputable def empiricalApproxError_signal {d m n : ℕ} (model : SpikedModel d)
    (net_sig : TwoLayerNet model.signalRank m)
    (target : (Fin model.signalRank → ℝ) → ℝ) (sample : Sample d n) : ℝ :=
  ((n : ℝ)⁻¹) * ∑ i : Fin n,
    |net_sig.forward (coordinateLift model.rank_le (sample.X i)) -
      target (coordinateLift model.rank_le (sample.X i))|

lemma empiricalApproxError_le_of_pointwise {d m n : ℕ} (net : TwoLayerNet d m)
    (target : (Fin d → ℝ) → ℝ) (sample : Sample d n) {ε : ℝ}
    (h : ∀ i, |net.forward (sample.X i) - target (sample.X i)| ≤ ε) :
    empiricalApproxError net target sample ≤ ε := by
  have npos : (0 : ℝ) < (n : ℝ) := Nat.cast_pos.mpr sample.n_pos
  unfold empiricalApproxError
  have hsum :
      ∑ i : Fin n, |net.forward (sample.X i) - target (sample.X i)| ≤
        ∑ _i : Fin n, ε :=
    Finset.sum_le_sum fun i _ => h i
  have hconst : ∑ _i : Fin n, ε = (n : ℝ) * ε := by
    simp [Finset.sum_const, Finset.card_univ, Fintype.card_fin]
  have hbound := hsum.trans_eq hconst
  have hmul := mul_le_mul_of_nonneg_left hbound (inv_nonneg.mpr npos.le)
  have hcancel : ((n : ℝ)⁻¹) * ((n : ℝ) * ε) = ε := by
    rw [← mul_assoc, inv_mul_cancel₀ (ne_of_gt npos), one_mul]
  exact hmul.trans_eq hcancel

/-! ### Part 1: Biasless Zero-Input Obstruction -/

/-- At the zero input vector `x = 0`, the forward pass of a biasless two-layer
    network reduces to `(∑ k, W₂ 0 k) * act.σ 0`. -/
theorem forward_zero_eq {d m : ℕ} (net : TwoLayerNet d m) :
    net.forward (fun _ : Fin d => 0) = (∑ k : Fin m, net.W₂ 0 k) * net.act.σ 0 := by
  unfold TwoLayerNet.forward
  dsimp only
  rw [Finset.sum_mul]
  refine Finset.sum_congr rfl fun k _ => ?_
  simp

/-- If `net.act.σ 0 = 0`, then any biasless two-layer network `net : TwoLayerNet d m`
    outputs `0` at the zero input vector `x = 0`. -/
theorem forward_zero_of_act_zero {d m : ℕ} (net : TwoLayerNet d m)
    (hσ₀ : net.act.σ 0 = 0) :
    net.forward (fun _ : Fin d => 0) = 0 := by
  rw [forward_zero_eq, hσ₀, mul_zero]

/-- **Biasless Zero-Sample Obstruction (`biasless_zero_sample_obstruction`)**:
    If `act.σ 0 = 0` and all sample inputs `sample.X i` are zero, then for every
    width `m` and every biasless network `net : TwoLayerNet d m` with `net.act = act`,
    the empirical approximation error against a target is at least `|target 0|`.
    In particular, if `ε < |target 0|`, no biasless network can achieve error `≤ ε`
    without either a nonzero activation value at `0` (`act.σ 0 ≠ 0`) or a signal-subspace
    realizability/separation condition on the sample. -/
theorem biasless_zero_sample_obstruction {d m n : ℕ} (model : SpikedModel d)
    (sample : Sample d n) (act : ActivationFunction) (hσ₀ : act.σ 0 = 0)
    (hX₀ : ∀ i : Fin n, sample.X i = fun _ => 0)
    (target : (Fin model.signalRank → ℝ) → ℝ)
    (net : TwoLayerNet d m) (hact : net.act = act) :
    empiricalApproxError net
      (fun x => target (coordinateLift model.rank_le x)) sample =
      |target (fun _ => 0)| := by
  have npos : (0 : ℝ) < (n : ℝ) := Nat.cast_pos.mpr sample.n_pos
  unfold empiricalApproxError
  have hterm : ∀ i : Fin n,
      |net.forward (sample.X i) - target (coordinateLift model.rank_le (sample.X i))| =
        |target (fun _ => 0)| := by
    intro i
    rw [hX₀ i]
    have hnet₀ : net.forward (fun _ : Fin d => 0) = 0 := by
      apply forward_zero_of_act_zero
      rw [hact]
      exact hσ₀
    have hlift₀ : coordinateLift model.rank_le (fun _ : Fin d => (0 : ℝ)) = fun _ => 0 := rfl
    rw [hnet₀, hlift₀, zero_sub, abs_neg]
  have hsum :
      (∑ i : Fin n,
        |net.forward (sample.X i) - target (coordinateLift model.rank_le (sample.X i))|) =
        (n : ℝ) * |target (fun _ => 0)| := by
    calc
      (∑ i : Fin n,
        |net.forward (sample.X i) - target (coordinateLift model.rank_le (sample.X i))|)
        = ∑ _i : Fin n, |target (fun _ => 0)| := Finset.sum_congr rfl fun i _ => hterm i
      _ = (n : ℝ) * |target (fun _ => 0)| := by
        simp [Finset.sum_const, Finset.card_univ, Fintype.card_fin]
  rw [hsum, ← mul_assoc, inv_mul_cancel₀ (ne_of_gt npos), one_mul]

/-! ### Part 2: Subspace Lifting of Two-Layer Networks (`subspaceLiftNet`) -/

/-- Lift a first-layer weight matrix `W₁ : Matrix (Fin m) (Fin k) ℝ` on the first `k ≤ d`
    signal coordinates to an ambient weight matrix `Matrix (Fin m) (Fin d) ℝ` that puts `0`
    on the remaining `d - k` bulk/noise coordinates. -/
def subspaceLiftWeights {d k m : ℕ} (_hk : k ≤ d)
    (W₁ : Matrix (Fin m) (Fin k) ℝ) : Matrix (Fin m) (Fin d) ℝ :=
  fun r j => if hj : j.val < k then W₁ r ⟨j.val, hj⟩ else 0

/-- Embed a signal-subspace two-layer network `net_sig : TwoLayerNet model.signalRank m`
    into the full ambient space `ℝ^d` by placing zero weights on all noise coordinates
    `j ≥ model.signalRank`. -/
def subspaceLiftNet {d m : ℕ} (model : SpikedModel d)
    (net_sig : TwoLayerNet model.signalRank m) : TwoLayerNet d m where
  W₁ := subspaceLiftWeights model.rank_le net_sig.W₁
  W₂ := net_sig.W₂
  act := net_sig.act

@[simp]
lemma subspaceLiftNet_act {d m : ℕ} (model : SpikedModel d)
    (net_sig : TwoLayerNet model.signalRank m) :
    (subspaceLiftNet model net_sig).act = net_sig.act :=
  rfl

/-- Pre-activation inner product of `subspaceLiftWeights` on `x : Fin d → ℝ` equals
    the inner product of `W₁` on the signal projection `coordinateLift hk x`. -/
theorem subspaceLiftWeights_dot_eq {d k m : ℕ} (hk : k ≤ d)
    (W₁ : Matrix (Fin m) (Fin k) ℝ) (r : Fin m) (x : Fin d → ℝ) :
    (∑ j : Fin d, subspaceLiftWeights hk W₁ r j * x j) =
      ∑ i : Fin k, W₁ r i * coordinateLift hk x i := by
  let emb : Fin k ↪ Fin d :=
    ⟨fun i => ⟨i.val, lt_of_lt_of_le i.isLt hk⟩,
     fun a b h => Fin.ext (by simpa using congr_arg Fin.val h)⟩
  have hmap :
      (∑ i : Fin k, W₁ r i * coordinateLift hk x i) =
        ∑ j ∈ Finset.univ.map emb, subspaceLiftWeights hk W₁ r j * x j := by
    rw [Finset.sum_map]
    refine Finset.sum_congr rfl fun i _ => ?_
    have hi : (emb i).val < k := i.isLt
    simp only [subspaceLiftWeights, hi, ↓reduceDIte, coordinateLift]
    rfl
  rw [hmap]
  symm
  refine Finset.sum_subset (Finset.subset_univ _) fun j _ hj => ?_
  have hnot : ¬ j.val < k := by
    intro hlt
    apply hj
    rw [Finset.mem_map]
    exact ⟨⟨j.val, hlt⟩, Finset.mem_univ _, rfl⟩
  simp [subspaceLiftWeights, hnot]

/-- **Exact Subspace Projection Theorem (`subspaceLiftNet_forward_eq`)**:
    For every input `x : Fin d → ℝ`, the lifted network `subspaceLiftNet model net_sig`
    computes the exact forward pass of `net_sig` on `coordinateLift model.rank_le x`,
    completely eliminating all `d - model.signalRank` noise coordinates. -/
theorem subspaceLiftNet_forward_eq {d m : ℕ} (model : SpikedModel d)
    (net_sig : TwoLayerNet model.signalRank m) (x : Fin d → ℝ) :
    (subspaceLiftNet model net_sig).forward x =
      net_sig.forward (coordinateLift model.rank_le x) := by
  unfold TwoLayerNet.forward subspaceLiftNet
  dsimp only
  refine Finset.sum_congr rfl fun r _ => ?_
  rw [subspaceLiftWeights_dot_eq]

/-- The empirical approximation error of `subspaceLiftNet model net_sig` in `ℝ^d`
    against a lifted signal function equals the signal-space empirical approximation error. -/
theorem empiricalApproxError_subspaceLiftNet {d m n : ℕ} (model : SpikedModel d)
    (net_sig : TwoLayerNet model.signalRank m)
    (target : (Fin model.signalRank → ℝ) → ℝ) (sample : Sample d n) :
    empiricalApproxError (subspaceLiftNet model net_sig)
      (fun x => target (coordinateLift model.rank_le x)) sample =
      empiricalApproxError_signal model net_sig target sample := by
  unfold empiricalApproxError empiricalApproxError_signal
  congr 1
  refine Finset.sum_congr rfl fun i _ => ?_
  rw [subspaceLiftNet_forward_eq]

/-! ### Part 3: Explicit Sample Interpolation and Subspace Universal Approximation -/

/-- **Explicit 1-Neuron-per-Sample Signal Interpolator (`exists_exact_sample_interpolator_of_anchor`)**:
    Suppose `act.σ 0 = 0`, there is an activation anchor `c : ℝ` with `act.σ c ≠ 0`, and
    the projected sample points `z_j = coordinateLift model.rank_le (sample.X j)` admit
    dual separating vectors `v : Fin n → Fin model.signalRank → ℝ` satisfying
    `∑ l, v i l * z_j l = if i = j then c else 0`.
    Then there exists a two-layer network `net : TwoLayerNet d n` of width `m = n` supported
    purely on the signal subspace such that:
    1. `net.act = act`,
    2. `net.forward (sample.X j) = target (coordinateLift model.rank_le (sample.X j))` for all `j`,
    3. `empiricalApproxError net (fun x => target (coordinateLift model.rank_le x)) sample = 0`. -/
theorem exists_exact_sample_interpolator_of_anchor {d n : ℕ} (model : SpikedModel d)
    (sample : Sample d n) (act : ActivationFunction)
    (target : (Fin model.signalRank → ℝ) → ℝ)
    (hσ₀ : act.σ 0 = 0) {c : ℝ} (hc : act.σ c ≠ 0)
    (v : Fin n → Fin model.signalRank → ℝ)
    (hv : ∀ i j : Fin n,
      (∑ l : Fin model.signalRank,
        v i l * coordinateLift model.rank_le (sample.X j) l) =
        if i = j then c else 0) :
    ∃ net : TwoLayerNet d n,
      net.act = act ∧
      (∀ j : Fin n,
        net.forward (sample.X j) =
          target (coordinateLift model.rank_le (sample.X j))) ∧
      empiricalApproxError net
        (fun x => target (coordinateLift model.rank_le x)) sample = 0 := by
  let net_sig : TwoLayerNet model.signalRank n :=
    { W₁ := fun i l => v i l
      W₂ := fun _ i => target (coordinateLift model.rank_le (sample.X i)) / act.σ c
      act := act }
  refine ⟨subspaceLiftNet model net_sig, rfl, ?_, ?_⟩
  · intro j
    rw [subspaceLiftNet_forward_eq]
    unfold TwoLayerNet.forward
    dsimp [net_sig]
    have hsum :
        (∑ k : Fin n,
          (target (coordinateLift model.rank_le (sample.X k)) / act.σ c) *
            act.σ (∑ l : Fin model.signalRank,
              v k l * coordinateLift model.rank_le (sample.X j) l)) =
        ∑ k : Fin n,
          if k = j then target (coordinateLift model.rank_le (sample.X j)) else 0 := by
      refine Finset.sum_congr rfl fun k _ => ?_
      rw [hv k j]
      split_ifs with hkj
      · subst hkj
        exact div_mul_cancel₀ _ hc
      · rw [hσ₀, mul_zero]
    rw [hsum]
    simp
  · unfold empiricalApproxError
    have hzero :
        (∑ j : Fin n,
          |(subspaceLiftNet model net_sig).forward (sample.X j) -
            target (coordinateLift model.rank_le (sample.X j))|) = 0 := by
      refine Finset.sum_eq_zero fun j _ => ?_
      rw [subspaceLiftNet_forward_eq]
      unfold TwoLayerNet.forward
      dsimp [net_sig]
      have hsum :
          (∑ k : Fin n,
            (target (coordinateLift model.rank_le (sample.X k)) / act.σ c) *
              act.σ (∑ l : Fin model.signalRank,
                v k l * coordinateLift model.rank_le (sample.X j) l)) =
          target (coordinateLift model.rank_le (sample.X j)) := by
        calc
          (∑ k : Fin n,
            (target (coordinateLift model.rank_le (sample.X k)) / act.σ c) *
              act.σ (∑ l : Fin model.signalRank,
                v k l * coordinateLift model.rank_le (sample.X j) l))
            = ∑ k : Fin n,
                if k = j then target (coordinateLift model.rank_le (sample.X j)) else 0 := by
              refine Finset.sum_congr rfl fun k _ => ?_
              rw [hv k j]
              split_ifs with hkj
              · subst hkj
                exact div_mul_cancel₀ _ hc
              · rw [hσ₀, mul_zero]
          _ = target (coordinateLift model.rank_le (sample.X j)) := by simp
      rw [hsum, sub_self, abs_zero]
    rw [hzero, mul_zero]

/-- **Signal-Subspace Universal Approximation Transfer (`exists_subspace_approximation_of_signal_net`)**:
    Whenever the target `target : (Fin model.signalRank → ℝ) → ℝ` is approximated within `ε`
    on the projected sample `coordinateLift model.rank_le (sample.X i)` by a signal-subspace
    network `net_sig : TwoLayerNet model.signalRank m` with activation `act`, lifting `net_sig`
    via `subspaceLiftNet model net_sig` produces an ambient network `net : TwoLayerNet d m`
    with `net.act = act` and empirical approximation error at most `ε`. -/
theorem exists_subspace_approximation_of_signal_net {d n : ℕ} (model : SpikedModel d)
    (sample : Sample d n) (act : ActivationFunction)
    (target : (Fin model.signalRank → ℝ) → ℝ) (ε : ℝ)
    (h_sig : ∃ m : ℕ, ∃ net_sig : TwoLayerNet model.signalRank m,
      net_sig.act = act ∧
      ∀ i : Fin n,
        |net_sig.forward (coordinateLift model.rank_le (sample.X i)) -
          target (coordinateLift model.rank_le (sample.X i))| ≤ ε) :
    ∃ m : ℕ, ∃ net : TwoLayerNet d m,
      net.act = act ∧
        empiricalApproxError net
          (fun x => target (coordinateLift model.rank_le x)) sample ≤ ε := by
  obtain ⟨m, net_sig, hact, hpt⟩ := h_sig
  refine ⟨m, subspaceLiftNet model net_sig, ?_, ?_⟩
  · simpa using hact
  · apply empiricalApproxError_le_of_pointwise
    intro i
    rw [subspaceLiftNet_forward_eq]
    exact hpt i

/-- **Constant-Readout Anchor Realizability (`exists_subspace_approximation_of_nonzero_origin`)**:
    If `act.σ 0 ≠ 0` and the target is constant on the sample (or within `ε` of a constant `y₀`),
    a 1-neuron network with zero first-layer weights and readout `y₀ / act.σ 0` achieves
    `empiricalApproxError ≤ ε`. -/
theorem exists_subspace_approximation_of_nonzero_origin {d n : ℕ} (model : SpikedModel d)
    (sample : Sample d n) (act : ActivationFunction) (hσ₀ : act.σ 0 ≠ 0)
    (target : (Fin model.signalRank → ℝ) → ℝ) {y₀ ε : ℝ}
    (hy₀ : ∀ i : Fin n, |y₀ - target (coordinateLift model.rank_le (sample.X i))| ≤ ε) :
    ∃ m : ℕ, ∃ net : TwoLayerNet d m,
      net.act = act ∧
        empiricalApproxError net
          (fun x => target (coordinateLift model.rank_le x)) sample ≤ ε := by
  let net : TwoLayerNet d 1 :=
    { W₁ := fun _ _ => 0
      W₂ := fun _ _ => y₀ / act.σ 0
      act := act }
  refine ⟨1, net, rfl, ?_⟩
  apply empiricalApproxError_le_of_pointwise
  intro i
  have hfwd : net.forward (sample.X i) = y₀ := by
    unfold TwoLayerNet.forward
    simp [net, div_mul_cancel₀ y₀ hσ₀]
  rw [hfwd]
  exact hy₀ i

/-- On a finite sample from a spiked model, any target of the `model.signalRank`
    signal coordinates that is `ε`-realizable on the signal subspace by a two-layer
    network with non-polynomial activation `act` is approximated within `ε` on `ℝ^d`
    by a lifted two-layer network `subspaceLiftNet model net_sig` whose weights vanish
    outside the signal subspace. -/
theorem exists_subspace_approximation {d n : ℕ} (model : SpikedModel d)
    (sample : Sample d n) (act : ActivationFunction) (_hσ : NonPolynomial act.σ)
    (target : (Fin model.signalRank → ℝ) → ℝ) (ε : ℝ) (_hε : 0 < ε)
    (h_realizable : ∃ m : ℕ, ∃ net_sig : TwoLayerNet model.signalRank m,
      net_sig.act = act ∧
      ∀ i : Fin n,
        |net_sig.forward (coordinateLift model.rank_le (sample.X i)) -
          target (coordinateLift model.rank_le (sample.X i))| ≤ ε) :
    ∃ m : ℕ, ∃ net : TwoLayerNet d m,
      net.act = act ∧
        empiricalApproxError net
          (fun x => target (coordinateLift model.rank_le x)) sample ≤ ε :=
  exists_subspace_approximation_of_signal_net model sample act target ε h_realizable

end OpenTheory
