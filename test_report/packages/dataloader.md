# `cavachon/dataloader`

[← Overview](../UNIT_TEST_REPORT.md)

| Status | Tests | Passed | Open bugs | Coverage | Coupling |
|---|---|---|---|---|---|
| ✅ Clean | [`test/dataloader`](../../test/dataloader) | 86 | 0 | 100% | Moderate: depends on 3 packages, used by 4 |

- **Depends on:** environment, layers, utils
- **Used by:** callbacks, model, tools, workflow
- **Mocked in tests:** `TensorUtils`, `ReflectionHandler`, `ToDense`/`Binarize` (recording fakes). The package's own `dataloader/modifiers` stay real.

## Fixed

- ✅ **H2** (`dataloader.py:55,189-191`): the mutable default `batch_effect_encoders=dict()` was changed in place, so a second DataLoader reused the first one's encoders and failed with "unseen labels" or **silently encoded with the wrong classes**. The default is now `None`; a dict passed in by the caller is still updated in place.
- ✅ **H3** (`dataloader.py:122-132`): modalities were paired **by row position, not by obs name**. New `DataLoader.check_obs_aligned()` raises a `ValueError` when modalities differ in their cells or cell order. It does not reorder them, because `Model.predict` writes outputs back by row position.
- ✅ `n_vars_batch_effect` always equals the width of the batch-effect tensor:
  - it was computed from the encoder dict, so a reused dict with extra columns inflated it;
  - a reused encoder now fixes how a column is encoded, so subset DataLoaders (as in `tools`) encode the same way as the full one;
  - a batch column missing from `obs` still emits a warning and is skipped, and `n_vars` now agrees;
  - calling `create_dataset()` twice no longer doubles it.
- ✅ `StudenttDataModifier` now sets `modality_name`.
- ✅ `tf.data.experimental.save/load` replaced by `Dataset.save` / `tf.data.Dataset.load`.
- ✅ The ZINB data-modifier docstring no longer promises LogTransform and library-size normalisation (the preprocessor applies them later).

## Worth a look

- `DataLoader.create_dataset` is annotated as returning a dataset but returns `None` (it sets `self.dataset`).
- `batch_effect_colnames` / `distribution_names` must be Mappings: `None` raises AttributeError. Several [tools](tools/README.md) entry points pass `None` by default (T5).
