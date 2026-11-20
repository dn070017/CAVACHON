# `cavachon/scheduler`

[← Overview](../UNIT_TEST_REPORT.md)

| Status | Tests | Passed | Open bugs | Coverage | Coupling |
|---|---|---|---|---|---|
| ✅ Clean | [`test/scheduler`](../../test/scheduler) | 121 | 0 | 99% | High: depends on 4 packages, used by 1 |

- **Depends on:** callbacks, config, environment, utils
- **Used by:** workflow
- **Mocked in tests:** the callbacks (Annealing, EarlyStopping, Verbose), `EarlyStoppingConfig`, `GeneralUtils.order_components` (a reference topological sort with the same contract); mlflow. The scheduler drives a `FakeModel`/`FakeComponent` that records `compile`/`fit`, trainable flags and annealing state for each phase. Loss weights are `FakeScaler` (ProgressiveScaler-like) or `FakeVar` (plain Variable).

**Commit 73eecd0 checked:** passing tests confirm "parent annealing ramps only alpha with the KL tied to it". During parent annealing the parents are frozen with weights 0. The child's alpha and standard KL ramp together, and its z_hat KL is 0. Afterwards alpha is pinned to (1, 1) and the standard KL to 1.

## Structure (refactored 2026-11-20)

`SequentialTrainingScheduler.fit` reads as the training flow:

```python
self._compile_model()
schedules = self.compute_component_schedules(epochs)       # one per component, in training order
phases = [p for s in schedules for p in self.planner.plan(s, n_batches)]
progress = PhaseProgress(total=sum(p.epochs for p in phases))
history = [self._run_phase(p, x, n_batches, progress, ...) for p in phases]
```

| Module | Class | Role |
|---|---|---|
| `sequential_training_scheduler.py` | `SequentialTrainingScheduler` | Orchestration: training order, `fit`, `_run_phase` (the one `model.fit` skeleton, inside an mlflow run), callbacks, compile |
| `component_schedule.py` | `ComponentSchedule` | A component's settings resolved once from its config (defaults filled in) and the model |
| `phase_planner.py` | `PhasePlanner` | Which phases a component runs and what each sets up: Parent Annealing → KL Annealing → Regular Training → Z_hat Annealing → Z_hat Training; KL schedule |
| `training_phase.py` | `TrainingPhase`, `PhaseProgress` | One `model.fit` call (name, epochs, setup/teardown, schedule, k-means epochs, early stopping); epoch book-keeping |
| `training_state_controller.py` | `TrainingStateController` | Loss weights, trainable flags, alpha, samplers; keeps progressive scalers across recompilation |
| `gmm_prior_initializer.py` | `GMMPriorInitializer`, `GMMPriorEstimate` | K-means++ GMM prior initialization of z or z_hat; freezes the model only for the prediction and restores every trainable flag and alpha |

`AnnealingCallback` no longer receives the scheduler: it gets a `kmeans_init` callable (`GMMPriorInitializer.initialize` bound to a component and target) and the list of `component_names` (see [callbacks](callbacks.md)).

**Behaviour preserved:** the same seeded end-to-end training on synthetic data (two components; KL annealing, k-means, parent annealing, z_hat annealing and training; 7 phases, real Keras model) gave identical losses in every phase, identical k-means priors for z and z_hat, and the same final alpha and sampler flags before and after the refactor. The only intended change is that mlflow runs are now ended even when a phase raises.

### Design note: one component per training stage

`training_order` is a list of stages, each holding the components trained together (`[[a], [b], ...]`). Components whose parents are already trained could share a stage, and an earlier version did train them jointly. That was dropped for two reasons: summing several components' losses in one stage made the loss normalization across components inconsistent, and joint stages needed noticeably more memory. Each stage now holds exactly one component; the nested structure is kept so stages can hold several components again if needed.

## Fixed

- ✅ (2026-11-20) **The k-means prior std matches the cluster spread.** For cluster std > 1 the inverse softplus was approximated by `std − 0.5`, so `softplus(bias)` came out about 15% too small at std ≈ 3 (the right large-std limit is `std`; the −0.5 only made the two branches meet near std = 1). `GMMPriorInitializer.inverse_softplus` now uses the exact, stable `std + log(1 − exp(−std))` for every std (no overflow for large std; std floored at 1e-6), the same formula as `HierarchicalEncoder.initialize_scale_diag`. Clusters with std ≤ 1 are unchanged up to float rounding.
- ✅ (2026-11-20) **`compute_modality_weight(constant=True)` works:** it read `n_vars` before assigning it (UnboundLocalError). It now gives every modality weight 1.0.
- ✅ (2026-11-20) **The training order is a real topological sort.** The scheduler orders `component_configs` with `GeneralUtils.order_components` itself, instead of relying on `Model.make` having done it, so a child listed before its parent no longer raises KeyError. Unknown parents raise ValueError and cycles AttributeError. Already-ordered configs (everything built through `Model.make`) keep exactly the same order, checked with the end-to-end run. This makes the scheduler usable on its own, outside the workflow; the cost is one more dependency (utils).
- ✅ (2026-11-20) **KL schedule boundaries no longer suffer from float error.** `int(n * (r0 + r1))` truncated `10 * (0.1 + 0.7) = 7.999…` to 7, which moved the GMM start one epoch early and cut the crossfade short (12 of 144 ratio/epoch combinations). `PhasePlanner.epoch_boundary` rounds the product before taking the floor; it is used for the standard-KL end, the GMM start and the k-means epoch. ⚠ For the affected configs the GMM KL now starts one epoch later than before.
- ✅ (2026-11-20) **K-means runs when the standard-KL stage fills the KL annealing phase.** With `kl_annealing_ratio[0] == 1.0`, the k-means epoch equalled the phase length (never reached) and the regular phase skipped k-means because KL annealing had run, so the priors were never initialized. K-means (and the z_hat prior's, when it is initialized early) now runs at the start of the regular training phase, the epoch right after the standard-KL stage.
- ✅ (2026-11-20, refactor) **The mlflow run is ended when a phase raises.** Each phase runs inside a `with self._mlflow_run(...)` block; previously a failing `fit` left the run open and broke the next `start_run`.
- ✅ **M6**: k-means++ crashed on a collapsed latent (fewer distinct points than clusters: every logit `-inf`), aborting training exactly when posterior collapse happened. It now falls back to choosing uniformly.
- ✅ **M7**: user `callbacks=` or `verbose=` passed to `fit` raised "multiple values for 'callbacks'". Each phase drops them from the forwarded kwargs; user callbacks are still deep-copied into every phase.
- ✅ **`EarlyStoppingConfig` is honoured.** The scheduler had hard-coded `monitor="loss"`, `min_delta=5` and patience `max(10, epochs/20)` from the global `max_regular_training_epochs`, so `patience: 50` became 10. The callback is now built from `monitor`, `mode`, `patience` and `min_delta` (default 5). `early_stopping: true` uses the documented defaults (loss, min, patience 25); `false` disables it.
- ✅ **z_hat annealing:**
  - **KL ramp:** the z_hat GMM KL used `epoch / n`, so the last epoch reached only (n−1)/n and n = 1 gave 0. It now uses `(epoch + 1) / n`.
  - **Samplers:** z is stochastic for the whole training. z_hat is deterministic until its component's z_hat annealing, stochastic from then on. Both flags are set to 0 when `fit` finishes and reset at its start (they are stored in the weights). Before, a frozen parent always passed its *mean* z and z_hat to children.
  - **k-means:** the z_hat prior is k-means-initialised exactly once: at the start of z_hat annealing if it runs, otherwise in the KL or regular phase with a warning. It used to run twice, discarding the first result.
- ✅ (2026-11-20) The debug-only `OptimizerStateCallback` and `PeriodicTSNECallback` are no longer added to the training phases (both were inactive on Keras 3; see [callbacks](callbacks.md)).
- ✅ Typing: `training_order` is `List[List[str]]`, modality weights are floats.

## Worth a look

- Recompiling a model reuses the existing losses; the scheduler relies on this (see [model](model.md)).
- The constructor arguments `outdir`, `distribution_names` and `batch_effect_colnames` are stored but no longer used. They are kept so the workflow call stays unchanged.
- The "z_hat prior is never trained" warning now sees the z_hat losses (the model is compiled before the phases are planned). Before the refactor the check ran before the first compile, so it could warn even when the z_hat phases did run.
