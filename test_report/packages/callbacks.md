# `cavachon/callbacks`

[← Overview](../UNIT_TEST_REPORT.md)

| Status | Tests | Passed | Open bugs | Coverage | Coupling |
|---|---|---|---|---|---|
| ⏸️ Reviewed, deferred | [`test/callbacks`](../../test/callbacks) | 45 | 4 | 90% | High: depends on 4 packages, used by 1 |

- **Depends on:** dataloader, distributions, environment, layers
- **Used by:** scheduler
- **Mocked in tests:** `DataLoader`, the distributions, `TensorUtils`, layers (tiny real Keras layers); mlflow.

## Open bugs (strict xfails)

> **Reviewed 2026-11-20 — deferred.** `OptimizerStateCallback` and `PeriodicTSNECallback` are debug tools. They are no longer used anywhere in cavachon (the scheduler stopped adding them to the training phases), so these bugs affect nothing today. The classes, their exports in `cavachon/callbacks/__init__.py` and their strict xfails stay in place.

| # | Location | Issue | Impact | Decision |
|---|---|---|---|---|
| M3 | `optimizer_state_callback.py:13,21` | **`OptimizerStateCallback` does nothing.** Keras 3 optimizers have no `get_weights`, and a bare `except` hides the error. The restore step also looks up `model._callbacks`, which doesn't exist. (2 xfails) | None: unused | ⏸️ Deferred |
| M4 | `periodic_tsne_callback.py:43` | **`PeriodicTSNECallback` never saves.** `save_epochs = set()` is hard-coded. | None: unused | ⏸️ Deferred |
| — | `periodic_tsne_callback.py` | `batch_size=int` uses the builtin type as the default value instead of an annotation. | None: unused | ⏸️ Deferred |

## Fixed

- ✅ `early_stopping_callback.py` (2026-11-20): the restored epoch was printed 0-based (`offset + best_epoch`), one lower than the "Total Epoch" VerboseCallback shows. It now prints `offset + best_epoch + 1`.
- ✅ `early_stopping_callback.py` (2026-11-20): `cumulative_offset` came before `*args`, so `EarlyStoppingCallback("loss")` set the offset to `"loss"` instead of `monitor`. It is now keyword-only; the scheduler already passes it by keyword.

## Worth a look

- `AnnealingCallback` freezes the model and relies on the scheduler to unfreeze it.
- `AnnealingCallback` loss patterns match anywhere in the loss name, not only as a prefix.
- `EarlyStoppingConfig` is now honoured; that fix lives in the [scheduler](scheduler.md), which builds the callback.
