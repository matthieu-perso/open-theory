import OpenTheory.Architectures.Basic
import OpenTheory.Data.Foundation
import OpenTheory.Dynamics.FeatureStep
import OpenTheory.Dynamics.MLPBatchTrajectory
import OpenTheory.Generalization.BallCover
import OpenTheory.Generalization.Covering
import OpenTheory.Generalization.TrajectoryGeneralization
import Mathlib.Analysis.SpecialFunctions.Log.Basic

/-!
# Convolutional Neural Network Generalization, Translation Invariance, and Architectural Separation

This module formalizes four theorem families on vision / convolutional architectures
(`ConvNet`) and their statistical generalization comparison against dense two-layer MLPs
(`TwoLayerNet`):

1. **Patch-Based / Cyclic Convolutional Layer Semantics & Exact Translation Equivariance**:
   - Defines cyclic spatial translation `spatialShift`, general patch extraction `extractPatch`,
     and the canonical translation-equivariant cyclic patch extractor `cyclicPatch`.
   - Defines a multi-channel convolutional architecture `ConvNet k c` with `c` channels of
     shared spatial filters `W : Matrix (Fin c) (Fin k) ℝ`, activation `act : ActivationFunction`,
     and readout weights `α : Fin c → ℝ`, producing feature maps `convFeatureMap` and global
     average pooled output `convForward`.
   - Proves exact translation equivariance of convolutional feature maps
     (`convFeatureMap_shift_equivariant`) and exact translation invariance of global average
     pooling (`convForward_shift_invariant`).

2. **Receptive-Field Locality & Parameter Perturbation Lipschitz Bound**:
   - Defines local patch energy `patchEnergy` and proves `patchEnergy_le_inputEnergy` whenever
     the receptive-field embedding `embed : Fin k → Fin d` is injective (`NeZero d`).
   - Proves `conv_param_diff_le`: perturbing flattened filter parameters `u, v : Parameters (c * k)`
     changes the global-pooled CNN prediction by at most `c * A * L * √E_patch * ‖u - v‖`,
     which depends only on local receptive-field energy `E_patch` and kernel size `k`,
     and is completely independent of the ambient spatial dimension `d`.

3. **High-Probability Out-of-Sample Generalization Bound for CNNs**:
   - Defines `convParamLoss` and `convLossLipConst`, proving `convParamLoss_lipschitz`.
   - Combines `BallCover.euclideanBall_net` with `Covering.uniform_deviation_tail` and
     `Covering.approx_learner_tail` to establish `convBall_loss_cover`,
     `conv_uniform_deviation_tail`, and `conv_approx_erm_excess_risk_tail` over
     `trajectoryBall (c * k) rad` with covering cardinality `(coverAxis rad δ (c * k) + 1) ^ (c * k)`.

4. **Strict Architectural Generalization & Sample Complexity Separation (CNN vs. Dense MLP)**:
   - Proves `cnn_lip_le_mlp_lip` and `cnn_lip_lt_mlp_lip`: local receptive-field energy
     `E_patch ≤ E_global` yields a smaller (and strictly smaller when `E_patch < E_global`)
     parameter-to-loss Lipschitz constant than a dense `c`-channel MLP.
   - Proves `cnn_covering_strictly_smaller_than_mlp`: when `0 < c`, `0 < k < d`, `0 < rad`,
     and `0 < δ`, the CNN covering number bound `(coverAxis rad δ (c * k) + 1) ^ (c * k)` is
     strictly smaller than the dense MLP covering number bound `(coverAxis rad δ (c * d) + 1) ^ (c * d)`.
   - Proves `cnn_vs_mlp_sample_complexity_ratio`: the log-covering sample complexity proxy
     `archComplexity (c * k) C_axis / archComplexity (c * d) C_axis` equals `(k : ℝ) / (d : ℝ) < 1`.
-/

noncomputable section

open MeasureTheory StatLean.StatisticalLearning
open scoped BigOperators RealInnerProductSpace

namespace OpenTheory.ConvolutionalGeneralization

open OpenTheory.FeatureStep
open OpenTheory.MLPTrajectory
open OpenTheory.MLPFixedDesign
open OpenTheory.MLPBatchTrajectory
open OpenTheory.BallCover
open OpenTheory.Covering
open OpenTheory.TrajectoryGeneralization

variable {d k c : ℕ}

/-! ## 1. Patch-Based / Cyclic Convolutional Layer Semantics & Exact Translation Equivariance -/

/-- Cyclic spatial shift of a 1D / spatial signal `x : Fin d → ℝ` by offset `s : Fin d`. -/
def spatialShift [NeZero d] (s : Fin d) (x : Fin d → ℝ) : Fin d → ℝ :=
  fun i => x (i + s)

@[simp]
theorem spatialShift_apply [NeZero d] (s : Fin d) (x : Fin d → ℝ) (i : Fin d) :
    spatialShift s x i = x (i + s) := rfl

/-- Zero shift is the identity on spatial signals. -/
@[simp]
theorem spatialShift_zero [NeZero d] (x : Fin d → ℝ) :
    spatialShift 0 x = x := by
  ext i
  simp [spatialShift]

/-- Composing two spatial shifts adds their shift offsets. -/
theorem spatialShift_add [NeZero d] (s₁ s₂ : Fin d) (x : Fin d → ℝ) :
    spatialShift s₁ (spatialShift s₂ x) = spatialShift (s₁ + s₂) x := by
  ext i
  simp [spatialShift, add_assoc]

/-- General receptive-field patch extractor indexed by an offset map
`offset : Fin d → Fin k → Fin d`. -/
def extractPatch (offset : Fin d → Fin k → Fin d) (x : Fin d → ℝ)
    (p : Fin d) (j : Fin k) : ℝ :=
  x (offset p j)

/-- Canonical translation-equivariant cyclic patch extractor with receptive-field
offsets `embed : Fin k → Fin d`. -/
def cyclicPatch [NeZero d] (embed : Fin k → Fin d) (x : Fin d → ℝ)
    (p : Fin d) (j : Fin k) : ℝ :=
  extractPatch (fun q r => q + embed r) x p j

@[simp]
theorem cyclicPatch_apply [NeZero d] (embed : Fin k → Fin d) (x : Fin d → ℝ)
    (p : Fin d) (j : Fin k) :
    cyclicPatch embed x p j = x (p + embed j) := rfl

/-- Exact translation equivariance of cyclic patch extraction: shifting the input signal `x`
by `s` shifts the extracted patch at position `p` to position `p + s`. -/
theorem cyclicPatch_shift [NeZero d] (embed : Fin k → Fin d) (s : Fin d)
    (x : Fin d → ℝ) (p : Fin d) (j : Fin k) :
    cyclicPatch embed (spatialShift s x) p j = cyclicPatch embed x (p + s) j := by
  simp only [cyclicPatch_apply, spatialShift_apply]
  congr 1
  exact add_right_comm p (embed j) s

/-- Alias for `ActivationFunction` matching architectural notation. -/
abbrev Activation := ActivationFunction

/-- A multi-channel 1D / spatial Convolutional Neural Network (`ConvNet k c`) with kernel
size `k`, channel count `c`, shared filter weights `W : Matrix (Fin c) (Fin k) ℝ`
(which has judgmental type `Fin c → Fin k → ℝ`), nonlinear activation `act : ActivationFunction`,
and channel readout weights `α : Fin c → ℝ`. -/
structure ConvNet (k c : ℕ) where
  W : Matrix (Fin c) (Fin k) ℝ
  act : ActivationFunction
  α : Fin c → ℝ

/-- Convolutional preactivation at channel `ch : Fin c` and spatial location `p : Fin d`. -/
def convPreactivation [NeZero d] (net : ConvNet k c) (embed : Fin k → Fin d)
    (x : Fin d → ℝ) (ch : Fin c) (p : Fin d) : ℝ :=
  ∑ j : Fin k, net.W ch j * cyclicPatch embed x p j

/-- Convolutional feature map at channel `ch : Fin c` and spatial location `p : Fin d`. -/
def convFeatureMap [NeZero d] (net : ConvNet k c) (embed : Fin k → Fin d)
    (x : Fin d → ℝ) (ch : Fin c) (p : Fin d) : ℝ :=
  net.act.σ (∑ j : Fin k, net.W ch j * cyclicPatch embed x p j)

theorem convFeatureMap_eq_sigma_pre [NeZero d] (net : ConvNet k c) (embed : Fin k → Fin d)
    (x : Fin d → ℝ) (ch : Fin c) (p : Fin d) :
    convFeatureMap net embed x ch p = net.act.σ (convPreactivation net embed x ch p) := rfl

/-- Global average pooled forward output of a multi-channel CNN over spatial positions `Fin d`. -/
def convForward [NeZero d] (net : ConvNet k c) (embed : Fin k → Fin d)
    (x : Fin d → ℝ) : ℝ :=
  (1 / (d : ℝ)) * ∑ p : Fin d, ∑ ch : Fin c, net.α ch * convFeatureMap net embed x ch p

/-- Exact translation equivariance of convolutional preactivations. -/
theorem convPreactivation_shift_equivariant [NeZero d] (net : ConvNet k c)
    (embed : Fin k → Fin d) (s : Fin d) (x : Fin d → ℝ) (ch : Fin c) (p : Fin d) :
    convPreactivation net embed (spatialShift s x) ch p =
      spatialShift s (convPreactivation net embed x ch) p := by
  rw [spatialShift_apply]
  unfold convPreactivation
  refine Finset.sum_congr rfl fun j _ => ?_
  rw [cyclicPatch_shift]

/-- **Exact Translation Equivariance (`convFeatureMap_shift_equivariant`)**:
For any spatial shift `s : Fin d`, translating the input `x` by `s` translates every
channel's convolutional feature map by `s`. -/
theorem convFeatureMap_shift_equivariant [NeZero d] (net : ConvNet k c)
    (embed : Fin k → Fin d) (s : Fin d) (x : Fin d → ℝ) (ch : Fin c) (p : Fin d) :
    convFeatureMap net embed (spatialShift s x) ch p =
      spatialShift s (convFeatureMap net embed x ch) p := by
  simp only [convFeatureMap_eq_sigma_pre, spatialShift_apply,
    convPreactivation_shift_equivariant]

/-- Function-level formulation of `convFeatureMap_shift_equivariant`. -/
theorem convFeatureMap_shift_equivariant_fun [NeZero d] (net : ConvNet k c)
    (embed : Fin k → Fin d) (s : Fin d) (x : Fin d → ℝ) (ch : Fin c) :
    convFeatureMap net embed (spatialShift s x) ch =
      spatialShift s (convFeatureMap net embed x ch) := by
  ext p
  exact convFeatureMap_shift_equivariant net embed s x ch p

/-- **Exact Translation Invariance of Global Average Pooling (`convForward_shift_invariant`)**:
For any spatial shift `s : Fin d`, the global-average-pooled CNN output is strictly invariant
under cyclic spatial translations of the input signal. -/
theorem convForward_shift_invariant [NeZero d] (net : ConvNet k c)
    (embed : Fin k → Fin d) (s : Fin d) (x : Fin d → ℝ) :
    convForward net embed (spatialShift s x) = convForward net embed x := by
  unfold convForward
  congr 1
  have hshift : ∀ p : Fin d,
      (∑ ch : Fin c, net.α ch * convFeatureMap net embed (spatialShift s x) ch p) =
        ∑ ch : Fin c, net.α ch * convFeatureMap net embed x ch (p + s) := by
    intro p
    refine Finset.sum_congr rfl fun ch _ => ?_
    rw [convFeatureMap_shift_equivariant, spatialShift_apply]
  simp_rw [hshift]
  exact Equiv.sum_comp (Equiv.addRight s)
    (fun q => ∑ ch : Fin c, net.α ch * convFeatureMap net embed x ch q)

/-! ## 2. Receptive-Field Locality & Parameter Perturbation Lipschitz Bound -/

/-- Local receptive-field patch energy at spatial position `p : Fin d`:
the squared Euclidean norm of the `k`-dimensional patch `cyclicPatch embed x p`. -/
def patchEnergy [NeZero d] (embed : Fin k → Fin d) (x : Fin d → ℝ) (p : Fin d) : ℝ :=
  ∑ j : Fin k, (cyclicPatch embed x p j) ^ 2

theorem patchEnergy_eq_inputEnergy_patch [NeZero d] (embed : Fin k → Fin d)
    (x : Fin d → ℝ) (p : Fin d) :
    patchEnergy embed x p = inputEnergy (cyclicPatch embed x p) := rfl

theorem patchEnergy_nonneg [NeZero d] (embed : Fin k → Fin d)
    (x : Fin d → ℝ) (p : Fin d) :
    0 ≤ patchEnergy embed x p :=
  Finset.sum_nonneg fun _ _ => sq_nonneg _

/-- **Local Patch Energy vs. Global Input Energy (`patchEnergy_le_inputEnergy`)**:
Whenever the receptive-field offset map `embed : Fin k → Fin d` is injective, the local
patch energy at any spatial position `p : Fin d` is at most the global signal energy
`inputEnergy x = ∑ i : Fin d, x i ^ 2`. -/
theorem patchEnergy_le_inputEnergy [NeZero d] (embed : Fin k → Fin d)
    (hembed : Function.Injective embed) (x : Fin d → ℝ) (p : Fin d) :
    patchEnergy embed x p ≤ inputEnergy x := by
  unfold patchEnergy inputEnergy cyclicPatch extractPatch
  let embP : Fin k ↪ Fin d :=
    ⟨fun j => p + embed j, fun j₁ j₂ h => hembed (add_left_cancel h)⟩
  have hmap : (∑ j : Fin k, x (p + embed j) ^ 2) =
      ∑ i ∈ (Finset.univ : Finset (Fin k)).map embP, x i ^ 2 := by
    rw [Finset.sum_map]
    rfl
  rw [hmap]
  exact Finset.sum_le_univ_sum_of_nonneg fun i => sq_nonneg (x i)

/-- Flattening a convolutional filter matrix `U : Matrix (Fin c) (Fin k) ℝ` into
`Parameters (c * k)`. -/
def flattenConv (U : Matrix (Fin c) (Fin k) ℝ) : Parameters (c * k) :=
  flatten U

/-- Unflattening a parameter vector `u : Parameters (c * k)` into a convolutional
filter matrix `Matrix (Fin c) (Fin k) ℝ`. -/
def unflattenConv (u : Parameters (c * k)) : Matrix (Fin c) (Fin k) ℝ :=
  unflatten u

@[simp]
theorem unflattenConv_flattenConv (U : Matrix (Fin c) (Fin k) ℝ) :
    unflattenConv (flattenConv U) = U :=
  unflatten_flatten U

@[simp]
theorem flattenConv_unflattenConv (u : Parameters (c * k)) :
    flattenConv (unflattenConv u) = u :=
  flatten_unflatten u

theorem unflattenConv_sub (u v : Parameters (c * k)) :
    unflattenConv (u - v) = unflattenConv u - unflattenConv v :=
  unflatten_sub u v

/-- Displacing the shared convolutional filter weights `net.W` by `U : Matrix (Fin c) (Fin k) ℝ`. -/
def convAtDisplacement (net : ConvNet k c) (U : Matrix (Fin c) (Fin k) ℝ) : ConvNet k c :=
  { net with W := net.W + U }

@[simp]
theorem convAtDisplacement_zero (net : ConvNet k c) :
    convAtDisplacement net 0 = net := by
  cases net
  simp [convAtDisplacement]

theorem convAtDisplacement_add (net : ConvNet k c) (U V : Matrix (Fin c) (Fin k) ℝ) :
    convAtDisplacement (convAtDisplacement net V) U = convAtDisplacement net (V + U) := by
  cases net
  simp [convAtDisplacement, add_assoc]

theorem convAtDisplacement_sub_cancel (net : ConvNet k c) (U V : Matrix (Fin c) (Fin k) ℝ) :
    convAtDisplacement (convAtDisplacement net V) (U - V) = convAtDisplacement net U := by
  rw [convAtDisplacement_add, add_sub_cancel]

/-- Convolutional filter preactivation shift at channel `ch` and spatial location `p`. -/
def convPreShift [NeZero d] (U : Matrix (Fin c) (Fin k) ℝ) (embed : Fin k → Fin d)
    (x : Fin d → ℝ) (ch : Fin c) (p : Fin d) : ℝ :=
  preShift U (cyclicPatch embed x p) ch

theorem convPreactivation_displacement [NeZero d] (net : ConvNet k c)
    (U : Matrix (Fin c) (Fin k) ℝ) (embed : Fin k → Fin d)
    (x : Fin d → ℝ) (ch : Fin c) (p : Fin d) :
    convPreactivation (convAtDisplacement net U) embed x ch p =
      convPreactivation net embed x ch p + convPreShift U embed x ch p := by
  simp [convPreactivation, convAtDisplacement, convPreShift, preShift,
    add_mul, Finset.sum_add_distrib]

/-- Cauchy-Schwarz bound on the convolutional preactivation shift at position `p`:
`|convPreShift (unflattenConv u) embed x ch p| ≤ ‖u‖ * √(patchEnergy embed x p)`. -/
theorem convPreShift_unflattenConv_le_norm [NeZero d] (u : Parameters (c * k))
    (embed : Fin k → Fin d) (x : Fin d → ℝ) (ch : Fin c) (p : Fin d) :
    |convPreShift (unflattenConv u) embed x ch p| ≤
      ‖u‖ * Real.sqrt (patchEnergy embed x p) := by
  unfold convPreShift unflattenConv
  rw [patchEnergy_eq_inputEnergy_patch]
  exact preShift_unflatten_le_norm u (cyclicPatch embed x p) ch

/-- Single-position convolutional readout perturbation bound at spatial location `p`. -/
theorem convPosition_change_le [NeZero d] (net : ConvNet k c) (embed : Fin k → Fin d)
    (x : Fin d → ℝ) (U : Matrix (Fin c) (Fin k) ℝ) (p : Fin d)
    {A L D : ℝ} (hA : 0 ≤ A) (hL : 0 ≤ L)
    (hα : ∀ ch, |net.α ch| ≤ A) (hσ : ∀ s, |deriv net.act.σ s| ≤ L)
    (hD : ∀ ch, |convPreShift U embed x ch p| ≤ D) :
    |(∑ ch : Fin c, net.α ch * convFeatureMap (convAtDisplacement net U) embed x ch p) -
      (∑ ch : Fin c, net.α ch * convFeatureMap net embed x ch p)| ≤
      c * A * L * D := by
  have hact (a b : ℝ) : |net.act.σ a - net.act.σ b| ≤ L * |a - b| := by
    simpa only [Real.norm_eq_abs] using
      Convex.norm_image_sub_le_of_norm_deriv_le
        (fun u (_ : u ∈ Set.univ) => net.act.differentiable u)
        (fun u _ => show ‖deriv net.act.σ u‖ ≤ L from hσ u)
        (convex_univ : Convex ℝ (Set.univ : Set ℝ)) (Set.mem_univ b) (Set.mem_univ a)
  rw [← Finset.sum_sub_distrib]
  apply (Finset.abs_sum_le_sum_abs _ _).trans
  calc
    _ ≤ ∑ _ : Fin c, A * (L * D) := by
      apply Finset.sum_le_sum
      intro ch _
      rw [← mul_sub, abs_mul]
      apply mul_le_mul (hα ch) _ (abs_nonneg _) hA
      rw [convFeatureMap_eq_sigma_pre, convFeatureMap_eq_sigma_pre]
      apply (hact _ _).trans
      rw [convPreactivation_displacement, add_sub_cancel_left]
      exact mul_le_mul_of_nonneg_left (hD ch) hL
    _ = _ := by
      simp [Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
      ring

/-- **Receptive-Field Locality & Parameter Perturbation Lipschitz Bound (`conv_param_diff_le`)**:
For any two flattened filter displacements `u, v : Parameters (c * k)` and any input signal
`x : Fin d → ℝ` whose local receptive-field energy satisfies `patchEnergy embed x p ≤ E_patch`
at every position `p : Fin d`, perturbing the CNN parameters from `u` to `v` changes the
global-average-pooled output `convForward` by at most `c * A * L * √E_patch * ‖u - v‖`.
Crucially, the factor `(1 / d) * ∑_{p : Fin d}` averages out the `d` spatial positions so that
the Lipschitz constant depends ONLY on local patch energy `E_patch` and NOT on `d`. -/
theorem conv_param_diff_le [NeZero d] (net : ConvNet k c) (embed : Fin k → Fin d)
    (x : Fin d → ℝ) (u v : Parameters (c * k)) {A L E_patch : ℝ}
    (hA : 0 ≤ A) (hL : 0 ≤ L) (_hE : 0 ≤ E_patch)
    (hα : ∀ ch, |net.α ch| ≤ A) (hσ : ∀ s, |deriv net.act.σ s| ≤ L)
    (hpatch : ∀ p : Fin d, patchEnergy embed x p ≤ E_patch) :
    |convForward (convAtDisplacement net (unflattenConv u)) embed x -
      convForward (convAtDisplacement net (unflattenConv v)) embed x| ≤
      c * A * L * Real.sqrt E_patch * ‖u - v‖ := by
  have hd_pos : 0 < d := NeZero.pos d
  have hd_real : (0 : ℝ) < d := Nat.cast_pos.mpr hd_pos
  have hd_ne : (d : ℝ) ≠ 0 := ne_of_gt hd_real
  have h1 : convAtDisplacement net (unflattenConv u) =
      convAtDisplacement (convAtDisplacement net (unflattenConv v)) (unflattenConv (u - v)) := by
    rw [unflattenConv_sub, convAtDisplacement_sub_cancel]
  rw [h1]
  unfold convForward
  rw [← mul_sub, ← Finset.sum_sub_distrib, abs_mul,
    abs_of_nonneg (one_div_nonneg.mpr hd_real.le)]
  let D : ℝ := ‖u - v‖ * Real.sqrt E_patch
  have hpos_bound : ∀ p : Fin d,
      |(∑ ch : Fin c, (convAtDisplacement net (unflattenConv v)).α ch *
          convFeatureMap
            (convAtDisplacement (convAtDisplacement net (unflattenConv v))
              (unflattenConv (u - v))) embed x ch p) -
        (∑ ch : Fin c, (convAtDisplacement net (unflattenConv v)).α ch *
          convFeatureMap (convAtDisplacement net (unflattenConv v)) embed x ch p)| ≤
        c * A * L * D := by
    intro p
    apply convPosition_change_le (convAtDisplacement net (unflattenConv v)) embed x
      (unflattenConv (u - v)) p hA hL hα hσ
    intro ch
    exact (convPreShift_unflattenConv_le_norm (u - v) embed x ch p).trans
      (mul_le_mul_of_nonneg_left (Real.sqrt_le_sqrt (hpatch p)) (norm_nonneg (u - v)))
  have hsum : |∑ p : Fin d,
      ((∑ ch : Fin c, (convAtDisplacement net (unflattenConv v)).α ch *
          convFeatureMap
            (convAtDisplacement (convAtDisplacement net (unflattenConv v))
              (unflattenConv (u - v))) embed x ch p) -
        (∑ ch : Fin c, (convAtDisplacement net (unflattenConv v)).α ch *
          convFeatureMap (convAtDisplacement net (unflattenConv v)) embed x ch p))| ≤
      (d : ℝ) * (c * A * L * D) := by
    apply (Finset.abs_sum_le_sum_abs _ _).trans
    calc
      _ ≤ ∑ _ : Fin d, (c * A * L * D) := Finset.sum_le_sum fun p _ => hpos_bound p
      _ = (d : ℝ) * (c * A * L * D) := by
        simp [Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
  calc
    1 / (d : ℝ) * |∑ p : Fin d, _|
      ≤ 1 / (d : ℝ) * ((d : ℝ) * (c * A * L * D)) :=
        mul_le_mul_of_nonneg_left hsum (one_div_nonneg.mpr hd_real.le)
    _ = c * A * L * Real.sqrt E_patch * ‖u - v‖ := by
        dsimp [D]
        field_simp [hd_ne]

/-! ## 3. High-Probability Out-of-Sample Generalization Bound for CNNs -/

/-- Parameter-indexed CNN observation loss on `z : Z` induced by loss `ℓ : ℝ → ℝ → ℝ`,
input extractor `xOf : Z → Fin d → ℝ`, and label extractor `yOf : Z → ℝ`. -/
def convParamLoss [NeZero d] {Z : Type*} (net : ConvNet k c) (embed : Fin k → Fin d)
    (xOf : Z → Fin d → ℝ) (yOf : Z → ℝ) (ℓ : ℝ → ℝ → ℝ)
    (u : Parameters (c * k)) (z : Z) : ℝ :=
  ℓ (convForward (convAtDisplacement net (unflattenConv u)) embed (xOf z)) (yOf z)

/-- Parameter-to-loss Lipschitz constant for a `c`-channel CNN with local patch energy `E_patch`:
`convLossLipConst c L_ℓ A L E_patch = L_ℓ * (c * A * L * √E_patch)`. -/
def convLossLipConst (c : ℕ) (L_ℓ A L E_patch : ℝ) : ℝ :=
  L_ℓ * (c * A * L * Real.sqrt E_patch)

theorem convLossLipConst_eq_paramLossLipConst (c : ℕ) (L_ℓ A L E_patch : ℝ) :
    convLossLipConst c L_ℓ A L E_patch = paramLossLipConst c L_ℓ A L E_patch := rfl

theorem convLossLipConst_nonneg (c : ℕ) {L_ℓ A L E_patch : ℝ}
    (hLℓ : 0 ≤ L_ℓ) (hA : 0 ≤ A) (hL : 0 ≤ L) :
    0 ≤ convLossLipConst c L_ℓ A L E_patch := by
  unfold convLossLipConst
  positivity

/-- **CNN Parameter-to-Loss Lipschitz Bound (`convParamLoss_lipschitz`)**:
When `ℓ` is `L_ℓ`-Lipschitz in its prediction argument and every patch of `xOf z` has
energy at most `E_patch`, `convParamLoss` is `convLossLipConst c L_ℓ A L E_patch`-Lipschitz
in the flattened filter vector `u : Parameters (c * k)`. -/
theorem convParamLoss_lipschitz [NeZero d] {Z : Type*} (net : ConvNet k c)
    (embed : Fin k → Fin d) (xOf : Z → Fin d → ℝ) (yOf : Z → ℝ) (ℓ : ℝ → ℝ → ℝ)
    {L_ℓ A L E_patch : ℝ} (hLℓ : 0 ≤ L_ℓ) (hA : 0 ≤ A) (hL : 0 ≤ L) (hE : 0 ≤ E_patch)
    (hℓ : ∀ p₁ p₂ y, |ℓ p₁ y - ℓ p₂ y| ≤ L_ℓ * |p₁ - p₂|)
    (hα : ∀ ch, |net.α ch| ≤ A) (hσ : ∀ s, |deriv net.act.σ s| ≤ L)
    (hpatch : ∀ z p, patchEnergy embed (xOf z) p ≤ E_patch)
    (u v : Parameters (c * k)) (z : Z) :
    |convParamLoss net embed xOf yOf ℓ u z - convParamLoss net embed xOf yOf ℓ v z| ≤
      convLossLipConst c L_ℓ A L E_patch * ‖u - v‖ := by
  unfold convParamLoss convLossLipConst
  calc
    |ℓ (convForward (convAtDisplacement net (unflattenConv u)) embed (xOf z)) (yOf z) -
        ℓ (convForward (convAtDisplacement net (unflattenConv v)) embed (xOf z)) (yOf z)|
      ≤ L_ℓ * |convForward (convAtDisplacement net (unflattenConv u)) embed (xOf z) -
          convForward (convAtDisplacement net (unflattenConv v)) embed (xOf z)| := hℓ _ _ _
    _ ≤ L_ℓ * (c * A * L * Real.sqrt E_patch * ‖u - v‖) :=
        mul_le_mul_of_nonneg_left
          (conv_param_diff_le net embed (xOf z) u v hA hL hE hα hσ (hpatch z)) hLℓ
    _ = L_ℓ * (c * A * L * Real.sqrt E_patch) * ‖u - v‖ := by ring

/-- **Explicit Finite Loss Cover for the CNN Parameter Ball (`convBall_loss_cover`)**:
For any radius `rad ≥ 0` and grid resolution `δ > 0`, there is an explicit finite net
`models : Finset (Parameters (c * k))` inside `trajectoryBall (c * k) rad` of cardinality
at most `(coverAxis rad δ (c * k) + 1) ^ (c * k)` that uniformly covers `convParamLoss`
to within `convLossLipConst c L_ℓ A L E_patch * δ`. -/
theorem convBall_loss_cover [NeZero d] {Z : Type*} (net : ConvNet k c)
    (embed : Fin k → Fin d) (xOf : Z → Fin d → ℝ) (yOf : Z → ℝ) (ℓ : ℝ → ℝ → ℝ)
    {L_ℓ A L E_patch rad δ : ℝ}
    (hLℓ : 0 ≤ L_ℓ) (hA : 0 ≤ A) (hL : 0 ≤ L) (hE : 0 ≤ E_patch)
    (hrad : 0 ≤ rad) (hδ : 0 < δ)
    (hℓ : ∀ p₁ p₂ y, |ℓ p₁ y - ℓ p₂ y| ≤ L_ℓ * |p₁ - p₂|)
    (hα : ∀ ch, |net.α ch| ≤ A) (hσ : ∀ s, |deriv net.act.σ s| ≤ L)
    (hpatch : ∀ z p, patchEnergy embed (xOf z) p ≤ E_patch) :
    ∃ models : Finset (Parameters (c * k)),
      (∀ g ∈ models, g ∈ trajectoryBall (c * k) rad) ∧
      models.card ≤ (coverAxis rad δ (c * k) + 1) ^ (c * k) ∧
      ∀ u ∈ trajectoryBall (c * k) rad, ∃ g ∈ models, ∀ z,
        |convParamLoss net embed xOf yOf ℓ u z - convParamLoss net embed xOf yOf ℓ g z| ≤
          convLossLipConst c L_ℓ A L E_patch * δ := by
  obtain ⟨models, hmodels, hcard, hnet⟩ := euclideanBall_net (d := c * k) hrad hδ
  refine ⟨models, hmodels, hcard, ?_⟩
  intro u hu
  obtain ⟨g, hg, hdist⟩ := hnet u hu
  refine ⟨g, hg, fun z => ?_⟩
  have hlip := convParamLoss_lipschitz net embed xOf yOf ℓ hLℓ hA hL hE hℓ hα hσ hpatch u g z
  exact hlip.trans
    (mul_le_mul_of_nonneg_left hdist (convLossLipConst_nonneg c hLℓ hA hL))

/-- **High-Probability Out-of-Sample Uniform Risk Deviation Tail for CNNs
(`conv_uniform_deviation_tail`)**:
Over an IID sample `s ~ sampleLaw P.law n`, every CNN filter parameter vector
`u ∈ trajectoryBall (c * k) rad` simultaneously has empirical risk within
`ε + 2 * convLossLipConst c L_ℓ A L E_patch * δ` of its population risk outside an event of
probability at most `2 * (coverAxis rad δ (c * k) + 1) ^ (c * k) * exp (-2 * n * ε ^ 2)`. -/
theorem conv_uniform_deviation_tail [NeZero d] {Z : Type*} [MeasurableSpace Z]
    (P : Population Z) {n : ℕ} (net : ConvNet k c) (embed : Fin k → Fin d)
    (xOf : Z → Fin d → ℝ) (yOf : Z → ℝ) (ℓ : ℝ → ℝ → ℝ)
    {L_ℓ A L E_patch rad δ ε : ℝ}
    (hLℓ : 0 ≤ L_ℓ) (hA : 0 ≤ A) (hL : 0 ≤ L) (hE : 0 ≤ E_patch)
    (hrad : 0 ≤ rad) (hδ : 0 < δ) (hn : 1 ≤ n) (hε : 0 < ε)
    (hℓ : ∀ p₁ p₂ y, |ℓ p₁ y - ℓ p₂ y| ≤ L_ℓ * |p₁ - p₂|)
    (hrange : ∀ u ∈ trajectoryBall (c * k) rad, ∀ z,
      convParamLoss net embed xOf yOf ℓ u z ∈ Set.Icc (0 : ℝ) 1)
    (hmeas : ∀ u ∈ trajectoryBall (c * k) rad,
      Measurable (convParamLoss net embed xOf yOf ℓ u))
    (hα : ∀ ch, |net.α ch| ≤ A) (hσ : ∀ s, |deriv net.act.σ s| ≤ L)
    (hpatch : ∀ z p, patchEnergy embed (xOf z) p ≤ E_patch) :
    let ρ := convLossLipConst c L_ℓ A L E_patch * δ
    let N_CNN := (coverAxis rad δ (c * k) + 1) ^ (c * k)
    sampleLaw P.law n {s | ∃ u ∈ trajectoryBall (c * k) rad,
      ε + 2 * ρ < |empRisk (convParamLoss net embed xOf yOf ℓ) s u -
        P.risk (convParamLoss net embed xOf yOf ℓ u)|} ≤
      ENNReal.ofReal (2 * (N_CNN : ℝ) * Real.exp (-2 * n * ε ^ 2)) := by
  intro ρ N_CNN
  obtain ⟨models, hmodels, hcard, hcover⟩ :=
    convBall_loss_cover net embed xOf yOf ℓ hLℓ hA hL hE hrad hδ hℓ hα hσ hpatch
  have htail := uniform_deviation_tail P (trajectoryBall (c * k) rad) models
    (convParamLoss net embed xOf yOf ℓ) hmodels hrange hmeas ρ ε hcover hn hε
  apply htail.trans
  apply ENNReal.ofReal_le_ofReal
  have hcard_real : (models.card : ℝ) ≤ (N_CNN : ℝ) := by exact_mod_cast hcard
  have hexp : 0 ≤ Real.exp (-2 * n * ε ^ 2) := Real.exp_nonneg _
  nlinarith

/-- **High-Probability Excess Population Risk Tail for Approximate ERM CNN Learners
(`conv_approx_erm_excess_risk_tail`)**:
Any CNN training algorithm `learner : (Fin n → Z) → Parameters (c * k)` returning parameters in
`trajectoryBall (c * k) rad` that is `η_opt`-approximate ERM satisfies an oracle excess risk
tail bound governed by `N_CNN = (coverAxis rad δ (c * k) + 1) ^ (c * k)`. -/
theorem conv_approx_erm_excess_risk_tail [NeZero d] {Z : Type*} [MeasurableSpace Z]
    (P : Population Z) {n : ℕ} (net : ConvNet k c) (embed : Fin k → Fin d)
    (xOf : Z → Fin d → ℝ) (yOf : Z → ℝ) (ℓ : ℝ → ℝ → ℝ)
    {L_ℓ A L E_patch rad δ ε η_opt : ℝ}
    (hLℓ : 0 ≤ L_ℓ) (hA : 0 ≤ A) (hL : 0 ≤ L) (hE : 0 ≤ E_patch)
    (hrad : 0 ≤ rad) (hδ : 0 < δ) (hn : 1 ≤ n) (hε : 0 < ε)
    (hℓ : ∀ p₁ p₂ y, |ℓ p₁ y - ℓ p₂ y| ≤ L_ℓ * |p₁ - p₂|)
    (hrange : ∀ u ∈ trajectoryBall (c * k) rad, ∀ z,
      convParamLoss net embed xOf yOf ℓ u z ∈ Set.Icc (0 : ℝ) 1)
    (hmeas : ∀ u ∈ trajectoryBall (c * k) rad,
      Measurable (convParamLoss net embed xOf yOf ℓ u))
    (hα : ∀ ch, |net.α ch| ≤ A) (hσ : ∀ s, |deriv net.act.σ s| ≤ L)
    (hpatch : ∀ z p, patchEnergy embed (xOf z) p ≤ E_patch)
    (learner : (Fin n → Z) → Parameters (c * k))
    (htrain : ∀ s, StatLean.StatisticalLearning.IsApproxERM
      (trajectoryBall (c * k) rad) (convParamLoss net embed xOf yOf ℓ) s η_opt (learner s)) :
    let ρ := convLossLipConst c L_ℓ A L E_patch * δ
    let N_CNN := (coverAxis rad δ (c * k) + 1) ^ (c * k)
    sampleLaw P.law n {s | ∃ u ∈ trajectoryBall (c * k) rad,
      P.risk (convParamLoss net embed xOf yOf ℓ u) + 2 * ε + 4 * ρ + η_opt <
        P.risk (convParamLoss net embed xOf yOf ℓ (learner s))} ≤
      ENNReal.ofReal (2 * (N_CNN : ℝ) * Real.exp (-2 * n * ε ^ 2)) := by
  intro ρ N_CNN
  obtain ⟨models, hmodels, hcard, hcover⟩ :=
    convBall_loss_cover net embed xOf yOf ℓ hLℓ hA hL hE hrad hδ hℓ hα hσ hpatch
  have htail := approx_learner_tail P (trajectoryBall (c * k) rad) models
    (convParamLoss net embed xOf yOf ℓ) hmodels hrange hmeas ρ ε η_opt hcover learner htrain hn hε
  apply htail.trans
  apply ENNReal.ofReal_le_ofReal
  have hcard_real : (models.card : ℝ) ≤ (N_CNN : ℝ) := by exact_mod_cast hcard
  have hexp : 0 ≤ Real.exp (-2 * n * ε ^ 2) := Real.exp_nonneg _
  nlinarith

/-! ## 4. Strict Architectural Generalization & Sample Complexity Separation (CNN vs. Dense MLP) -/

/-- **CNN vs. Dense MLP Lipschitz Comparison (`cnn_lip_le_mlp_lip`)**:
Because local patch energy satisfies `E_patch ≤ E_global`, a `c`-channel CNN has a smaller
(or equal) parameter-to-loss Lipschitz constant than a `c`-neuron dense MLP on `Fin d → ℝ`. -/
theorem cnn_lip_le_mlp_lip (c : ℕ) {L_ℓ A L E_patch E_global : ℝ}
    (hLℓ : 0 ≤ L_ℓ) (hA : 0 ≤ A) (hL : 0 ≤ L) (hE : E_patch ≤ E_global) :
    convLossLipConst c L_ℓ A L E_patch ≤
      TrajectoryGeneralization.paramLossLipConst c L_ℓ A L E_global := by
  unfold convLossLipConst TrajectoryGeneralization.paramLossLipConst
  apply mul_le_mul_of_nonneg_left _ hLℓ
  apply mul_le_mul_of_nonneg_left (Real.sqrt_le_sqrt hE) (by positivity)

/-- **Strict CNN vs. Dense MLP Lipschitz Separation (`cnn_lip_lt_mlp_lip`)**:
Whenever local patch energy is strictly smaller than global signal energy (`E_patch < E_global`)
and `c > 0`, `L_ℓ > 0`, `A > 0`, `L > 0`, the CNN parameter-to-loss Lipschitz constant is
**strictly smaller** than the dense MLP parameter-to-loss Lipschitz constant. -/
theorem cnn_lip_lt_mlp_lip {c : ℕ} (hc : 0 < c) {L_ℓ A L E_patch E_global : ℝ}
    (hLℓ : 0 < L_ℓ) (hA : 0 < A) (hL : 0 < L) (hEp : 0 ≤ E_patch)
    (hE : E_patch < E_global) :
    convLossLipConst c L_ℓ A L E_patch <
      TrajectoryGeneralization.paramLossLipConst c L_ℓ A L E_global := by
  unfold convLossLipConst TrajectoryGeneralization.paramLossLipConst
  have hc_pos : (0 : ℝ) < c := Nat.cast_pos.mpr hc
  have hsqrt : Real.sqrt E_patch < Real.sqrt E_global := Real.sqrt_lt_sqrt hEp hE
  exact mul_lt_mul_of_pos_left
    (mul_lt_mul_of_pos_left hsqrt (by positivity)) hLℓ

/-- Monotonicity of `coverAxis rad δ` in the parameter dimension `p` when `0 ≤ rad` and `0 < δ`. -/
theorem coverAxis_mono {rad δ : ℝ} (hrad : 0 ≤ rad) (hδ : 0 < δ) {p q : ℕ} (hpq : p ≤ q) :
    coverAxis rad δ p ≤ coverAxis rad δ q := by
  unfold coverAxis
  apply max_le_max le_rfl
  apply Nat.ceil_mono
  have hpq_real : (p : ℝ) ≤ (q : ℝ) := Nat.cast_le.mpr hpq
  have hsqrt : Real.sqrt (p : ℝ) ≤ Real.sqrt (q : ℝ) := Real.sqrt_le_sqrt hpq_real
  exact div_le_div_of_nonneg_right
    (mul_le_mul_of_nonneg_left hsqrt (mul_nonneg (by norm_num) hrad)) hδ.le

/-- **Strict Covering Number Separation Between CNN and Dense MLP
(`cnn_covering_strictly_smaller_than_mlp`)**:
When channel count `0 < c`, kernel size `0 < k < d`, ball radius `0 ≤ rad`, and resolution `0 < δ`
(including in particular `0 < rad`), the explicit metric-entropy covering number bound of the
`c`-channel CNN `(coverAxis rad δ (c * k) + 1) ^ (c * k)` is **strictly smaller** than that of
the `c`-unit dense MLP `(coverAxis rad δ (c * d) + 1) ^ (c * d)`. -/
theorem cnn_covering_strictly_smaller_than_mlp {c k d : ℕ} {rad δ : ℝ}
    (hc : 0 < c) (_hk : 0 < k) (hkd : k < d) (hrad : 0 < rad) (hδ : 0 < δ) :
    (coverAxis rad δ (c * k) + 1) ^ (c * k) <
      (coverAxis rad δ (c * d) + 1) ^ (c * d) := by
  have hck_lt_cd : c * k < c * d := Nat.mul_lt_mul_of_pos_left hkd hc
  have haxis_le : coverAxis rad δ (c * k) + 1 ≤ coverAxis rad δ (c * d) + 1 :=
    Nat.add_le_add_right (coverAxis_mono hrad.le hδ hck_lt_cd.le) 1
  have hbase_ge_two : 2 ≤ coverAxis rad δ (c * d) + 1 := by
    have h1 : 1 ≤ coverAxis rad δ (c * d) := le_max_left 1 _
    omega
  calc
    (coverAxis rad δ (c * k) + 1) ^ (c * k)
      ≤ (coverAxis rad δ (c * d) + 1) ^ (c * k) := Nat.pow_le_pow_left haxis_le (c * k)
    _ < (coverAxis rad δ (c * d) + 1) ^ (c * d) :=
        Nat.pow_lt_pow_right hbase_ge_two hck_lt_cd

/-- Log-covering sample complexity proxy `archComplexity p C_axis = p * log C_axis` for a
parameter space of dimension `p` and per-coordinate grid factor `C_axis`. -/
def archComplexity (p : ℕ) (C_axis : ℝ) : ℝ :=
  (p : ℝ) * Real.log C_axis

/-- `archComplexity (p : ℕ) ((coverAxis rad δ p + 1 : ℕ) : ℝ)` equals the log of the covering
number bound `((coverAxis rad δ p + 1) ^ p : ℝ)`. -/
theorem archComplexity_eq_log_covering (p : ℕ) (rad δ : ℝ) :
    archComplexity p ((coverAxis rad δ p + 1 : ℕ) : ℝ) =
      Real.log (((coverAxis rad δ p + 1 : ℕ) : ℝ) ^ p) := by
  unfold archComplexity
  rw [Real.log_pow]

/-- **Exact Sample Complexity Ratio Between CNN and Dense MLP
(`cnn_vs_mlp_sample_complexity_ratio`)**:
For any channel count `c > 0`, kernel width `0 < k < d`, and per-axis covering factor `1 < C_axis`,
the ratio of CNN to dense MLP log-covering sample complexity is independent of `c` and `C_axis`
and equals `(k : ℝ) / (d : ℝ) < 1`, establishing a strict `(k / d)`-factor sample complexity
reduction from weight sharing and receptive-field locality. -/
theorem cnn_vs_mlp_sample_complexity_ratio {c k d : ℕ} {C_axis : ℝ}
    (hc : 0 < c) (hk : 0 < k) (hkd : k < d) (hC : 1 < C_axis) :
    archComplexity (c * k) C_axis / archComplexity (c * d) C_axis = (k : ℝ) / (d : ℝ) ∧
      (k : ℝ) / (d : ℝ) < 1 := by
  have hc_ne : (c : ℝ) ≠ 0 := ne_of_gt (Nat.cast_pos.mpr hc)
  have hd_pos : (0 : ℝ) < d := Nat.cast_pos.mpr (lt_trans hk hkd)
  have hlog_pos : 0 < Real.log C_axis := Real.log_pos hC
  have hlog_ne : Real.log C_axis ≠ 0 := ne_of_gt hlog_pos
  refine ⟨?_, ?_⟩
  · unfold archComplexity
    push_cast
    have hnum : (c : ℝ) * (k : ℝ) * Real.log C_axis =
        (k : ℝ) * ((c : ℝ) * Real.log C_axis) := by ring
    have hden : (c : ℝ) * (d : ℝ) * Real.log C_axis =
        (d : ℝ) * ((c : ℝ) * Real.log C_axis) := by ring
    rw [hnum, hden, mul_div_mul_right (k : ℝ) (d : ℝ) (mul_ne_zero hc_ne hlog_ne)]
  · rw [div_lt_one hd_pos]
    exact Nat.cast_lt.mpr hkd

end OpenTheory.ConvolutionalGeneralization
