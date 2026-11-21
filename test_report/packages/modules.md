# `cavachon/modules`

[← Overview](../UNIT_TEST_REPORT.md)

| Status | Tests | Passed | Open bugs | Coverage | Coupling |
|---|---|---|---|---|---|
| ✅ Clean | [`test/modules`](../../test/modules) | 204 | 0 | 100% | High: depends on 4 packages, used by 2 |

| Subpackage | Tests | Passed | Open bugs |
|---|---|---|---|
| base | [`test/modules/base`](../../test/modules/base) | 40 | 0 |
| components | [`test/modules/components`](../../test/modules/components) | 115 | 0 |
| parameterizers | [`test/modules/parameterizers`](../../test/modules/parameterizers) | 32 | 0 |
| preprocessors | [`test/modules/preprocessors`](../../test/modules/preprocessors) | 17 | 0 |

- **Depends on:** environment, layers, losses, utils
- **Used by:** model, tools
- **Mocked in tests:** layers and losses as tiny real `tf.keras` layers / loss objects with deterministic outputs, `ProgressiveScaler`, `TensorUtils`, `ReflectionHandler`. Code inside `modules` stays real (e.g. `Component` uses the real `modules.base` encoders).

## Component as a standalone SDK

`Component.compile` and `Component.train_step` are never called in production: training goes through `Model`, driven by the scheduler. Component exists for developers who want a single-node graph trained on its own (no hierarchy), so its API follows `Model`'s: the same `compile` arguments and loss-weight rules, `train_step`/`test_step` returning the total and each loss's unweighted value. Checked on a real component: `fit` trains (also directly on a multi-modality dataset), and `evaluate` works.

Deliberate differences from `Model`, documented in `Component.compile`:
- **Recompiling rebuilds the losses**, so new weights take effect (and the weights' progress resets). `Model.compile` reuses existing losses and ignores new weights for them, because the scheduler recompiles whenever the trainable variables change and relies on the losses carrying over.
- **Parents are fixed values.** A component with parents takes their z / z_hat as inputs, so in standalone training only its own networks are trained.
- **The z_hat GMM KL is pinned to 0 at compile**, as in Model. There the scheduler switches it on in the z_hat annealing phase; standalone, switch it on with `_z_hat_gmm_kl_weights[name].pin_to(z_hat_kl_weight)`.

## Open bugs (strict xfails)

None.

## Fixed

- ✅ (2026-11-21) **ZINB log-mean ceiling: soft cap at 15 instead of a hard clip at 7** (`IndependentZeroInflatedNegativeBinomialParameterizer`, both `modify_outputs` and `compute_attribution_target`). The log mean m·L was clipped with `tf.where(x > 7, 7, x)` before `exp(x) − 1`:
  - 7 caps real data: e⁷ − 1 ≈ 1096 counts, and in the repo's PBMC 10x Multiome RNA 17 counts in 11 cells exceed it (up to 5585), all immunoglobulin genes (IGKC, IGHA1, IGLC1–3) in plasma cells;
  - the hard clip has zero gradient above 7, so an overshooting output could not be pulled back by the mean term of the loss; downstream, DEG/HDEG pairs above it tied and attributions were 0.
  - Now `soft_cap(x) = cap − softplus(cap − x)` with `MAX_LOG_MEAN = 15` (class attribute): about x below the cap (shift log(1 + e^(x−15)): 3e-4 at 7, 7e-3 at 10), approaching 15 from below, gradient sigmoid(15 − x) > 0 everywhere. 15 only guards `exp` against overflow (float32 overflows above ~88.7): e¹⁵ − 1 ≈ 3.3e6 counts, far above any per-gene count; the ZINB log-likelihood is finite at that mean for dispersions 1e-3 to 1e3. Both are documented in the code.
  - **This changes training numerics** (deliberately): slightly everywhere (the shift above), and fully for log means above 7. Models trained with the old clip were never penalized above 7, so decoding them with the new cap can give large means where their outputs exceed 7.
- ✅ (2026-11-21) `Parameterizer.modify_outputs` applied `tf.math.exp` to a KerasTensor, so `make(exp_transform=True)` raised on Keras 3. The exp is now wrapped in a `Lambda` layer (works on KerasTensors and eager tensors).
- ✅ (2026-11-21) `HierarchicalEncoder` and `EncoderLatentParameterizer` had no `get_config`, so Keras 3 auto-config recorded the default constructor arguments. Both now store their arguments and return them from `get_config`.
- ✅ (2026-11-21) **`Component.compile` / `train_step` follow `Model`.** Fixes, all in `components/component.py`:
  - **C1:** `train_step` no longer calls `compiled_metrics.update_state`, which raised on Keras 3.
  - **C3, C4:** the weight indices (`_standard_kl_weights`, `_gmm_kl_weights`, `_data_loss_weights`) are the losses' own `ProgressiveScaler`s, as in Model, so `pin_to`/`assign` on them changes the loss. They were standalone Variables never passed to the KL losses, and the data loss copied its Variable with `float()`.
  - A weight of 0 creates the loss switched off (scale 1, pinned to 0: reported, not trained, switchable), instead of scale 0 with a 0 / 0 metric.
  - `compile` works on a copy of `loss_weights` and warns about names that match no data loss.
  - New `test_step`, sharing the loss evaluation with `train_step`, so `evaluate()` and `fit(validation_data=...)` work. Each loss is evaluated once per step (it was evaluated twice); metrics use `divide_no_nan`.
  - `__init__` honours Keras's `trainable` and `dtype` (it dropped all `**kwargs`; `trainable` is set last, since setting it pins alpha through the hierarchical encoder).
  - `decode` validates and documents that it needs the modality matrices as well as the batch effects (the preprocessor runs again for the library size). Entries passed as `None` are rejected too, so `Model.decode` gets the same clear error.
  - **`select_inputs`** (as in Model) restricts each batch to the component's inputs in `train_step`, `test_step` and the new `predict_step`, so a dataset with other modalities can be used directly. Keras matches dict inputs by position, so extra entries would otherwise feed the wrong input or fail.
  - **`library_size` and `decode(..., libsize=...)`**, with `requires_library_size(modality)` (see [model](model.md)); `library_size` feeds the preprocessor only the component's own matrices, so a raw multi-modality batch is safe. The steps unpack Keras `(x, y, sample_weight)` batches.
  - **z_hat GMM KL**, as in `Model.compile`: components that learn a z_hat prior get it with `z_hat_kl_weight` as scale (default 1; not created for 0), pinned to 0, indexed in `_z_hat_gmm_kl_weights`, and fed z_hat and its parameters against the z_hat prior.
- ✅ (2026-11-21) `Component.compile` created the legacy z_hat density loss with weight 1.0 and reset it to 1.0 on every recompile. It is now kept at 0, as in `Model` (see [model](model.md)).
- ✅ (2026-11-21) **C2:** with only the standard KL compiled, `Component.train_step` fed y only under the GMM key, so `compute_loss` raised KeyError for the standard KL. Every compiled KL loss is now fed, as in `Model.train_step` (MD1 in [model](model.md)).
- ✅ **M1** (`base/decoder_data_parameterizer.py:58-63`): **`n_decoder_layers` was ignored**; the backbone was hard-coded to 2 layers. It is now passed through. ⚠ With the default of 3, every decoder is Dense32 → LN → Dense64 → LN → Dense64 → LN; older models had 2-layer decoders.
- ✅ **M2** (`component.py:494-501` + `base/hierarchical_encoder.py:245,277`): `progressive_iterations` never reached the hierarchical encoder (Component passed `progressive_step=`, which HierarchicalEncoder silently swallowed). Component passes the right name, and HierarchicalEncoder forwards `**kwargs` to `tf.keras.Model`, so unknown keywords raise and `trainable`/`dtype` are honoured.
- ✅ `Component.train_step` now drops NaN gradients through `TensorUtils.remove_nan_gradients`, like `Model.train_step`.
- ✅ Stale ZINB parameterizer `make` docstring defaults.
- ✅ Typing: parameterizer `make()` overrides whose signatures differ on purpose keep strict signatures with a targeted `# type: ignore[override]` (no silent `**kwargs`).

## Worth a look

- **ZINB library size semantics.** The library size is the row sum of *log1p* counts, because NormalizeLibrarySize runs after LogTransform. The decoder computes `exp(min(mean·libsize, 7)) − 1`, so the expected count is capped at about 1,096. This may be intended; worth confirming.
