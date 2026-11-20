# `cavachon/modules`

[← Overview](../UNIT_TEST_REPORT.md)

| Status | Tests | Passed | Open bugs | Coverage | Coupling |
|---|---|---|---|---|---|
| 🟡 Open bugs, latent paths | [`test/modules`](../../test/modules) | 181 | 12 | 100% | High: depends on 4 packages, used by 2 |

| Subpackage | Tests | Passed | Open bugs |
|---|---|---|---|
| base | [`test/modules/base`](../../test/modules/base) | 38 | 2 |
| components | [`test/modules/components`](../../test/modules/components) | 98 | 8 |
| parameterizers | [`test/modules/parameterizers`](../../test/modules/parameterizers) | 28 | 2 |
| preprocessors | [`test/modules/preprocessors`](../../test/modules/preprocessors) | 17 | 0 |

- **Depends on:** environment, layers, losses, utils
- **Used by:** model, tools
- **Mocked in tests:** layers and losses as tiny real `tf.keras` layers / loss objects with deterministic outputs, `ProgressiveScaler`, `TensorUtils`, `ReflectionHandler`. Code inside `modules` stays real (e.g. `Component` uses the real `modules.base` encoders).

## Latent: not in the production path

`Component.compile` and `Component.train_step` are never called in production. Training goes through `Model.compile` and `Model.train_step`, driven by the scheduler. C1–C4 therefore affect only standalone Component training.

## Open bugs (strict xfails)

| # | Location | Issue | Impact |
|---|---|---|---|
| C1 | `components/component.py:1351` | `train_step` calls `compiled_metrics.update_state(y_true, y_pred_dict)`, which raises a ValueError on Keras 3. | Latent |
| C3 | `component.py:1183-1212` | `_standard_kl_weights` / `_gmm_kl_weights` Variables are never passed to the losses, so assigning them has no effect. | Latent |
| C4 | `component.py:1242-1254` | `_data_loss_weights` is copied with `float(...)` by NegativeLogDataLikelihood, so assigning it later has no effect. | Latent |
| — | `component.py` | `compile` pops entries from the caller's `loss_weights` dict. | Edge case |
| — | `component.py` | A weight of 0 still creates the loss; its metric becomes 0/0 = NaN. | Edge case |
| — | `component.py` | No `test_step`, so `evaluate()` and validation fail. | Latent |
| — | `component.py` | `__init__` drops `**kwargs`, so Keras kwargs like `trainable=False` are ignored. | Edge case |
| — | `component.py` | `decode()` needs the modality matrices (preprocessor rerun) but documents only batch-effect keys. | Edge case |
| — | `parameterizers/parameterizer.py` | `exp_transform=True` applies `tf.math.exp` to a KerasTensor, so `make` raises on Keras 3. (2 xfails) | Edge case |
| — | `base/hierarchical_encoder.py`, `base/encoder_latent_parameterizer.py` | No `get_config`; Keras 3 auto-config records default constructor arguments. (2 xfails) | Breaks save/clone round trips |

## Fixed

- ✅ (2026-11-21) `Component.compile` created the legacy z_hat density loss with weight 1.0 and reset it to 1.0 on every recompile. It is now kept at 0, as in `Model` (see [model](model.md)).
- ✅ (2026-11-21) **C2:** with only the standard KL compiled, `Component.train_step` fed y only under the GMM key, so `compute_loss` raised KeyError for the standard KL. Every compiled KL loss is now fed, as in `Model.train_step` (MD1 in [model](model.md)).
- ✅ **M1** (`base/decoder_data_parameterizer.py:58-63`): **`n_decoder_layers` was ignored**; the backbone was hard-coded to 2 layers. It is now passed through. ⚠ With the default of 3, every decoder is Dense32 → LN → Dense64 → LN → Dense64 → LN; older models had 2-layer decoders.
- ✅ **M2** (`component.py:494-501` + `base/hierarchical_encoder.py:245,277`): `progressive_iterations` never reached the hierarchical encoder (Component passed `progressive_step=`, which HierarchicalEncoder silently swallowed). Component passes the right name, and HierarchicalEncoder forwards `**kwargs` to `tf.keras.Model`, so unknown keywords raise and `trainable`/`dtype` are honoured.
- ✅ `Component.train_step` now drops NaN gradients through `TensorUtils.remove_nan_gradients`, like `Model.train_step`.
- ✅ Stale ZINB parameterizer `make` docstring defaults.
- ✅ Typing: parameterizer `make()` overrides whose signatures differ on purpose keep strict signatures with a targeted `# type: ignore[override]` (no silent `**kwargs`).

## Worth a look

- **ZINB library size semantics.** The library size is the row sum of *log1p* counts, because NormalizeLibrarySize runs after LogTransform. The decoder computes `exp(min(mean·libsize, 7)) − 1`, so the expected count is capped at about 1,096. This may be intended; worth confirming.
