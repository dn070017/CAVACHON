# `cavachon/callbacks`

[← Overview](../UNIT_TEST_REPORT.md)

| Status | Tests | Passed | Open bugs | Coverage | Coupling |
|---|---|---|---|---|---|
| 🟡 Open bugs, no wrong results | [`test/callbacks`](../../test/callbacks) | 43 | 6 | 90% | High: depends on 4 packages, used by 1 |

- **Depends on:** dataloader, distributions, environment, layers
- **Used by:** scheduler
- **Mocked in tests:** `DataLoader`, the distributions, `TensorUtils`, layers (tiny real Keras layers); mlflow.

## Open bugs (strict xfails)

| # | Location | Issue | Impact |
|---|---|---|---|
| M3 | `optimizer_state_callback.py:13,21` | **`OptimizerStateCallback` does nothing.** Keras 3 optimizers have no `get_weights`, and a bare `except` hides the error. The restore step also looks up `model._callbacks`, which doesn't exist. The scheduler adds this callback to every phase. (2 xfails) | Feature silently inactive |
| M4 | `periodic_tsne_callback.py:43` | **`PeriodicTSNECallback` never saves.** `save_epochs = set()` is hard-coded, but the scheduler adds the callback whenever `output_dir` is set. | Feature silently inactive |
| — | `periodic_tsne_callback.py` | `batch_size=int` uses the builtin type as the default value instead of an annotation. | Edge case |
| — | `early_stopping_callback.py` | The restored epoch is printed 0-based, one lower than Keras/VerboseCallback labels. | Cosmetic |
| — | `early_stopping_callback.py` | `cumulative_offset` comes before `*args`, so a positional `monitor` argument is swallowed. | Edge case |

## Worth a look

- `AnnealingCallback` freezes the model and relies on the scheduler to unfreeze it.
- `AnnealingCallback` loss patterns match anywhere in the loss name, not only as a prefix.
- `EarlyStoppingConfig` is now honoured; that fix lives in the [scheduler](scheduler.md), which builds the callback.
