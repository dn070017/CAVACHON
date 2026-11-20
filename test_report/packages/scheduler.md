# `cavachon/scheduler`

[← Overview](../UNIT_TEST_REPORT.md)

| Status | Tests | Passed | Open bugs | Coverage | Coupling |
|---|---|---|---|---|---|
| 🟡 Open bugs, mostly edge cases | [`test/scheduler`](../../test/scheduler) | 80 | 6 | 98% | High: depends on 4 packages, used by 1 |

- **Depends on:** callbacks, config, environment, layers
- **Used by:** workflow
- **Mocked in tests:** the callbacks (Annealing, EarlyStopping, Verbose), `ProgressiveScaler`; mlflow. The scheduler drives a `FakeModel`/`FakeComponent` that records `compile`/`fit`, trainable flags and annealing state for each phase.

**Commit 73eecd0 checked:** passing tests confirm "parent annealing ramps only alpha with the KL tied to it". During parent annealing the parents are frozen with weights 0. The child's alpha and standard KL ramp together, and its z_hat KL is 0. Afterwards alpha is pinned to (1, 1) and the standard KL to 1.

## Open bugs (strict xfails)

| Location | Issue | Impact |
|---|---|---|
| KL schedule | `int()` truncation of `n*(r0+r1)` (`int(10*(0.1+0.7)) = 7`) moves the GMM start one epoch early and cuts the crossfade short. 12 of 144 ratio/epoch combinations are affected. | Off by one epoch for some configs |
| KL schedule | With `kl_annealing_ratio[0] == 1.0`, `kmeans_epoch == n_epochs` is never reached and the regular phase skips k-means, so it never runs. | Edge case |
| k-means init | For cluster std > 1 the inverse softplus is approximated by `std - 0.5` (about 15% too small at std ≈ 3). | Prior starts narrower than the clusters |
| fit phases | The mlflow run is left open if `fit` raises, breaking the next `start_run`. | Edge case |
| `compute_modality_weight` | `constant=True` reads `n_vars` before assignment (UnboundLocalError). | Edge case |
| training order | Not a real topological sort: a child listed before its parent raises KeyError. | Edge case |

## Fixed

- ✅ **M6** (`sequential_training_scheduler.py:235-240`): k-means++ crashed on a collapsed latent (fewer distinct points than clusters: every logit `-inf`), aborting training exactly when posterior collapse happened. It now falls back to choosing uniformly.
- ✅ **M7**: user `callbacks=` or `verbose=` passed to `fit` raised "multiple values for 'callbacks'". Each phase drops them from the forwarded kwargs; user callbacks are still deep-copied into every phase.
- ✅ **`EarlyStoppingConfig` is honoured.** The scheduler had hard-coded `monitor="loss"`, `min_delta=5` and patience `max(10, epochs/20)` from the global `max_regular_training_epochs`, so `patience: 50` became 10. The callback is now built from `monitor`, `mode`, `patience` and `min_delta` (default 5). `early_stopping: true` uses the documented defaults (loss, min, patience 25); `false` disables it.
- ✅ **z_hat annealing:**
  - **KL ramp:** the z_hat GMM KL used `epoch / n`, so the last epoch reached only (n−1)/n and n = 1 gave 0. It now uses `(epoch + 1) / n`.
  - **Samplers:** z is stochastic for the whole training. z_hat is deterministic until its component's z_hat annealing, stochastic from then on. Both flags are set to 0 when `fit` finishes and reset at its start (they are stored in the weights). Before, a frozen parent always passed its *mean* z and z_hat to children.
  - **k-means:** the z_hat prior is k-means-initialised exactly once: at the start of z_hat annealing if it runs, otherwise in the KL or regular phase with a warning. It used to run twice, discarding the first result.
- ✅ Typing: `training_order` is `List[List[str]]`, modality weights are floats.
- ✅ (2026-11-20) The debug-only `OptimizerStateCallback` and `PeriodicTSNECallback` are no longer added to the training phases (both were inactive on Keras 3; see [callbacks](callbacks.md)). `_common_callbacks` now only builds the early-stopping callback.

## Worth a look

- Recompiling a model reuses the existing losses; the scheduler relies on this (see [model](model.md)).
- With the debug callbacks gone, the constructor arguments `outdir`, `distribution_names` and `batch_effect_colnames` are stored but no longer used. They are kept so the workflow call stays unchanged.
