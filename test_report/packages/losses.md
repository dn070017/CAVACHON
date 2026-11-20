# `cavachon/losses`

[← Overview](../UNIT_TEST_REPORT.md)

| Status | Tests | Passed | Open bugs | Coverage | Coupling |
|---|---|---|---|---|---|
| ⏸️ Reviewed, deferred | [`test/losses`](../../test/losses) | 38 | 2 | 100% | Moderate: depends on 3 packages, used by 2 |

- **Depends on:** distributions, layers, utils
- **Used by:** model, modules
- **Mocked in tests:** `ReflectionHandler` (returns a test-chosen distribution class), `ProgressiveScaler`, the distribution classes.

## Open bugs (strict xfails)

> **Reviewed 2026-11-20 — deferred.** Confirmed and accepted for now; no fix is planned. The strict xfails stay in place, so a fix would still be flagged as XPASS.

| Location | Issue | Impact | Decision |
|---|---|---|---|
| `standard_kl_divergence.py`, `gmm_kl_divergence.py` | `event_dims` is read from `y_pred.shape[1]` but slicing is on the last axis, so rank-3 inputs are split wrongly. (2 xfails) | Latent: current inputs are always rank 2 | ⏸️ Deferred |

## Fixed

- ✅ **H1** (`negative_log_data_likelihood.py:76`): `reduce_sum(log_prob(y), axis=-1)` assumed a per-feature `log_prob`. For `MultivariateNormalDiag`, `log_prob` is already per-sample, so this **summed over the batch**: the loss became a scalar inflated by the batch size. It now reduces over the last axis only when `log_prob` has the same rank as `y_true`.
- ✅ `gmm_density_loss.py:88`: the docstring said the weight can be a `ProgressiveScaler`, but `loss * self.weight` failed with one.
- ✅ The `GMMKLDivergence` docstring described the wrong `y_pred` layout.

## Worth a look

- `GMMDensityLoss` is legacy: replaced by the z_hat GMM KL, always created with weight 0. Its removal is deferred (see [model](model.md#reviewed-deferred)).
- `GMMDensityLoss` returns a batch mean under `Reduction.NONE`, so `sample_weight` cannot weight individual samples.
- `NegativeLogDataLikelihood` copies its weight with `float(...)`, so later assignments to a weight Variable have no effect (see C4 in [modules](modules.md)).
