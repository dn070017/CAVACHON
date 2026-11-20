# `cavachon/model`

[← Overview](../UNIT_TEST_REPORT.md)

| Status | Tests | Passed | Open bugs | Coverage | Coupling |
|---|---|---|---|---|---|
| 🟡 Open bugs, no production impact | [`test/model`](../../test/model) | 91 | 6 | 100% | Very high: depends on 7 packages, used by 2 |

- **Depends on:** config, dataloader, environment, layers, losses, modules, utils
- **Used by:** tools, workflow
- **Mocked in tests:**
  - `Component` is replaced by a tiny real Keras model with deterministic z / z_hat, so the parent → child wiring can be checked;
  - the losses, `ProgressiveScaler`, `DataLoader`, `GeneralUtils`, `TensorUtils` and `ToDense` are fakes;
  - a real SGD optimizer checks `train_step` against a hand-computed update.

**Checked and correct:**
- **Wiring:** parent z / z_hat reach the right children, including grandchildren, and multiple parents are concatenated in config order.
- **Loss weighting:** each loss is counted once, and the per-loss metrics are unweighted.
- **Gradients:** the output of `remove_nan_gradients` is what the optimizer applies, and frozen components are excluded.
- **`predict`:** rows stay in order, and each output is written to the correct modality's `.obsm`.

## Open bugs (strict xfails)

| # | Location | Issue | Impact |
|---|---|---|---|
| MD1 | `model.py:645-654` | A component compiled with only the standard KL never trains it: y_true/y_pred are stored under the GMM key. | No (the scheduler always passes both) |
| MD4 | `model.py:940-952`, `1090-1107` | `hierarchical_encode(strict=False)` still concatenates `None` for a missing parent. | No (analysis API only) |
| MD5 | `model.py:362-366` | `predict_x` isn't reset per modality, so x_parameters are computed for `save_x=False` modalities, depending on modality order. | Wasted memory only |
| MD6 | `model.py:471-497` | A weight of 0 still creates the loss. | No |
| MD7 | `model.py:734-736` | The scaler-weighted metric divides by the weight without a zero guard (NaN). | No |
| MD8 | `model.py:462/522` | `compile` pops entries from the caller's `loss_weights`. | No |

## Fixed

- ✅ **MD2** (`model.py:414`): `super.__predict__(...)` should have been `super().predict`, so `predict` on non-MuData input always raised.
- ✅ **MD3** (`model.py:389` + `train_step`): batch keys weren't filtered to the model's inputs; Keras matches dict inputs by position, so a MuData with an unused modality fed e.g. `atac_matrix` in as `rna_matrix`. `Model.select_inputs` now restricts each batch in `predict` and `train_step`.
- ✅ Typing: `components` is a `Mapping[str, Component]`; component configs are ordered with `GeneralUtils.order_components` and set `z_conditional_dims` / `z_hat_conditional_dims` on declared fields.

## Worth a look

- `save_z_hat` in `predict` is built from `save_z` and never used; dead code.
- In `encode`/`decode`, the config is passed as a plain `model_dump()` dict, so the parent lookups always come back `None`; parent inputs are skipped only by accident.
- Missing modality keys (`distribution_names.get`, `n_vars.get`) pass `None` along and fail far from the cause.
- Recompiling reuses the existing losses, so new weights for existing losses are ignored. The scheduler relies on this.
- The losses are evaluated twice per training step, once for the loss and once for the metrics.
