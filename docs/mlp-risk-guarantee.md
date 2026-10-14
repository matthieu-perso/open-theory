# First closed nonlinear MLP risk guarantee

`MLPRisk.expected_mse` concerns the **actual nonlinear training trajectory**,
not just its Taylor model. It is a restricted fixed-input statistical guarantee,
not a generalization theorem for unseen inputs.

## Setting

- One fixed input `x`, an arbitrary scalar clean target `s`, and labels `s + z`.
- A two-layer scalar-output MLP of arbitrary input dimension and hidden width.
- Train W₁ for T steps from its initialization; freeze W₂. The loss is half
  squared error plus λ/2 times squared displacement from initialization.
- λ > 0, η ≥ 0, readout coordinates bounded by A, activation derivative bounded
  by L and M-Lipschitz. The activation is differentiable as required by the model.
- `|1 − ηλ| ≤ 1` and `|1 − η(K + λ)| ≤ 1`, where K is the initial Jacobian norm².
- A finite noise law with nonnegative masses summing to one, mean zero,
  variance σ², and `|z| ≤ Z`. The same noisy label is used throughout training;
  this is not SGD with fresh noise at each step.

## Explicit bound

Write f₀ for the initial prediction and E = ‖x‖². Define

    S = |f₀ − s| + Z
    g = 1 + η A L E (1 + m A L)
    R = (1 + m A L) S g^T
    D = T η R A L E
    B = M m A D²
    G = m A² L M D E
    q = |1 − η(K + λ)|
    δ = B + T η (B |K| + R G) + q^T S.

Then the probability-weighted expectation over the noisy training label obeys

    E_z [(f_T(x; s+z) − s)²]
      ≤ 2 [(λ/(K+λ))² (f₀−s)² + (K/(K+λ))² σ²] + 2 δ².

The bracket is the exact centered-ridge bias–variance expression. The remaining
term is an explicit bound for nonlinear trajectory error and optimization error.
There is **no assumed residual envelope and no assumed closeness to ridge** in
this theorem: `MLPResidual.residual_le` and `MLPRisk.trained_endpoint_gap` prove
them from the stated initial-data and regularity assumptions.

## Reuse and limitations

The proof reuses the existing nonlinear/linearized trajectory comparison,
the proven ridge endpoint and its geometric convergence, Mathlib's mean-value
bound, and `PredictionGap.sq_norm_le_double`. Finite-sum bias–variance algebra
allows arbitrary scalar targets and avoids extra measurability assumptions.

The envelope R is conservative and can grow exponentially with T. This theorem
does not assert consistency, monotone risk improvement, a favorable width rate,
or an infinite-time nonlinear limit. Such conclusions need sharper estimates
and additional assumptions. It is not a novelty claim about the mathematics.

Next: extend the nonlinear comparison to multiple inputs and connect it to the
already verified `MLPFixedDesign` GD/ridge interfaces. General noise laws,
random-design test risk, jointly trained layers, and deeper networks remain
separate extensions. The frozen blueprint's one-step/scaling claims are unchanged.

The multi-input local estimates are implemented in `MLPBatchBounds`, and their
full finite-horizon accumulation and statistical transfer are now completed in
`MLPBatchTrajectory` and `MLPBatchRisk`: see `MLPBatchRisk.batch_trained_expected_mse`
and `MLPBatchRisk.batch_trained_expected_mse_of_rank`, as documented in
[the proof map](mlp-trajectory-status.md).
