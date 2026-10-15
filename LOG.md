# OpenTheory Autonomous Execution Log

## Milestone 1: Phase 4 Verification & Zero-Warning Cleanup (Completed)
- Verified `lake build`, `audit_axioms.lean` (335 blueprint references resolved, 0 `sorryAx`), `check_lean_decls.py`, and `export_blueprint.py` (61 nodes).
- Cleaned up all deprecation and unused-variable warnings across `OpenTheory/`.
- Committed at `41b1118`.

## Milestone 2: Phase 5 Frontier Deep Learning Theory Sprint (Completed)
1. **Multi-Layer Transformer Generalization, In-Context Excess Risk & Safety Control (`OpenTheory/Generalization/TransformerGeneralization.lean` + `tests/TransformerGeneralizationSmoke.lean`)**:
   - Proved `transformerPerturbationEnvelope_smul`, `transformer_stack_param_perturbation_le`, and `transformer_depth_scaled_lip_bound` ($L_{\mathrm{stack}} \le L_0 e^{c_0}$ under $1/L$ residual scaling).
   - Proved explicit covering numbers (`transformerBall_loss_cover`) and high-probability uniform generalization / approximate-ERM excess risk tail bounds (`transformer_uniform_deviation_tail`, `transformer_approx_erm_excess_risk_tail`).
   - Proved end-to-end In-Context Learning excess risk decomposition (`icl_end_to_end_excess_risk_bound`, `icl_total_risk_monotone_in_steps`, `icl_end_to_end_target_accuracy`).
   - Proved mechanistic safety intervention and orthogonal head-ablation guarantees (`safety_intervention_tradeoff_bound`, `orthogonal_circuit_ablation_zero_unsafe_and_risk_bound`), certifying exact unsafe-subspace eradication $P_{\mathrm{unsafe}}(\mathrm{MHA}_{\mathrm{keep}}) = 0$ with bounded benign MSE degradation.
2. **Vision / CNN Translation Equivariance, Locality & Architecture Separation (`OpenTheory/Architectures/ConvolutionalGeneralization.lean` + `tests/ConvolutionalGeneralizationSmoke.lean`)**:
   - Proved exact cyclic translation equivariance of convolutional feature maps (`convFeatureMap_shift_equivariant`) and translation invariance of global average pooling (`convForward_shift_invariant`).
   - Proved receptive-field parameter perturbation Lipschitz bounds (`conv_param_diff_le`, `patchEnergy_le_inputEnergy`) depending only on local patch energy $E_{\mathrm{patch}} \le E_{\mathrm{global}}$.
   - Proved CNN covering and generalization tail bounds (`convBall_loss_cover`, `conv_uniform_deviation_tail`, `conv_approx_erm_excess_risk_tail`).
   - Proved strict covering number and sample complexity separation between weight-shared CNNs and dense MLPs (`cnn_lip_lt_mlp_lip`, `cnn_covering_strictly_smaller_than_mlp`, `cnn_vs_mlp_sample_complexity_ratio` $= k/d < 1$).
3. **Momentum, EMA Variance Damping, AdamW Trajectory Ball & Optimizer Selection (`OpenTheory/Optimization/MomentumAdamTrajectory.lean` + `tests/MomentumAdamTrajectorySmoke.lean`)**:
   - Proved exact stationary gradient recovery of Adam's $(1-\beta^T)^{-1}$ bias-corrected EMA (`biasCorrectedEMA_const_grad`) and uniform norm stability (`emaTrajectory_norm_le`, `biasCorrectedEMA_norm_le`).
   - Proved exact $(1-\beta)/(1+\beta) < 1$ mini-batch variance damping (`emaVarianceTrajectory_eq`, `ema_variance_strictly_smaller_than_sgd`) and strict reduction of the stochastic PL noise floor (`momentum_pl_noise_floor_strict_improvement`, `momentum_pl_noise_ball_convergence`).
   - Proved forward invariance of the AdamW decoupled weight-decay parameter ball $B_p(D_{\max}/\lambda_w)$ (`adamw_step_preserves_ball`, `adamw_trajectory_norm_envelope`) and induced horizon-free MLP generalization (`adamw_mlp_generalization_tail`).
   - Proved exact optimizer regime dominance (`adam_momentum_dominates_preconditioned_no_momentum`, `adam_momentum_strictly_beats_scalar_sgd`).
4. **Spiked-Covariance Effective Dimension Collapse & $\mu$P vs. NTK Sample Complexity Separation (`OpenTheory/Generalization/SpikedSampleComplexity.lean` + `tests/SpikedSampleComplexitySmoke.lean`)**:
   - Proved spiked-covariance bias-variance decomposition and strict risk decrease under signal growth + bulk suppression (`spikedExpectedMSE_mono_in_bulk`, `spikedExpectedMSE_strict_decrease_of_signal_and_bulk`).
   - Proved ambient-dimension bottleneck $\Omega(d_{\mathrm{bulk}})$ for unaligned frozen NTK (`unaligned_effectiveDim_scales_with_d_bulk`) vs. collapse to intrinsic spike rank $r + O(\varepsilon d_{\mathrm{bulk}})$ under $\mu$P feature alignment (`mup_aligned_effectiveDim_le_rank_add_eps`).
   - Proved strict sample complexity separation (`mup_vs_ntk_sample_complexity_separation`) and the end-to-end spiked-covariance $\mu$P master guarantee (`mup_spiked_covariance_master_guarantee`).
- Verified full suite: `lake build` (0 warnings in `OpenTheory/`), `audit_axioms.lean` (387 blueprint references resolved, 0 `sorryAx`), `check_lean_decls.py`, `export_blueprint.py` (65 nodes), and Python + Lean smoke tests.
