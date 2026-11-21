# `cavachon/model`

[← Overview](../UNIT_TEST_REPORT.md)

| Status | Tests | Passed | Open bugs | Coverage | Coupling |
|---|---|---|---|---|---|
| ⏸️ Reviewed, deferred | [`test/model`](../../test/model) | 112 | 0 | 100% | Very high: depends on 7 packages, used by 2 |

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

## Reviewed, deferred

> **Reviewed 2026-11-21 — deferred.** No open bugs; one legacy cleanup is deferred.

| Location | Item | Impact | Decision |
|---|---|---|---|
| `Model.compile`, `Component.compile`, `losses/gmm_density_loss.py`, `ComponentSchedule.has_z_hat_density_loss` | **Legacy z_hat GMM density loss.** Replaced by the z_hat GMM KL, it is still created for components that learn a z_hat prior, switched off (weight 0), so it never affects training. It is kept only because the scheduler uses its presence (`has_z_hat_density_loss`) to decide whether the z_hat prior's k-means runs together with the z prior's. Removing it means switching that check to `has_z_hat_prior`, which can move the z_hat k-means for configs where the two differ, so it needs its own check. Documented in the code (GMMDensityLoss docstring, `Model.compile` docstring, `ComponentSchedule`). | None (always weight 0) | ⏸️ Deferred |

## Fixed

- ✅ (2026-11-21) **`Model.evaluate` and `fit(validation_data=...)` work.** Model had no `test_step`, so Keras's default failed with "y_true and y_pred have different structures". `train_step` is split into `_prepare_loss_inputs` and `_evaluate_losses` (each loss evaluated once, not twice), shared by the new `test_step`; the new `predict_step` applies `select_inputs` (also for datasets). All three follow the Keras step API (batches unpacked with `unpack_x_y_sample_weight`, dict of losses returned), as in `Component`. **Training is unchanged:** the seeded 7-phase e2e run gives identical losses, priors, alpha and sampler flags.
- ✅ (2026-11-21) **Explicit library size for `decode`.** `Model.compute_libsize(batch)` / `Component.compute_libsize(batch)` compute it exactly as in training (after the preprocessor's transforms, e.g. the row sum of log1p counts, not the raw count total), and `decode(..., libsize=...)` accepts it per modality, so decoding new z_hat (e.g. generation with a typical library size) needs no data matrix. Without `libsize` it is computed from the matrices as before, and if neither is given a clear ValueError names the key (the decoder used to fall back to 1 silently). Modalities whose decoder does not scale by the library size (`Component.requires_libsize`, e.g. Bernoulli) no longer need their matrix at all. Checked on a real model: decoding with `libsize(...)` and no matrices equals decoding from the data.
- ✅ (2026-11-21) `Model.decode` documents that the batch needs each modality's matrix as well as its batch effect (the preprocessor runs again for the library size), and a missing matrix now raises a clear ValueError naming the key instead of a Keras error about `None`.
- ✅ (2026-11-21) **The legacy z_hat density loss is off by default.** It was replaced by the z_hat GMM KL, and the scheduler always set its weight to 0, but `Model.compile` created it with weight 1.0, so direct `Model` users trained with it. Its weight Variable now starts at 0. The loss object is kept (switched off) because the scheduler checks for it when deciding where the z_hat prior's k-means runs; training through the scheduler is unchanged (identical e2e run).
- ✅ (2026-11-21) **MD1: a component compiled with only the standard KL now trains it.** `train_step` fed the KL inputs under the standard *and* GMM names only when both losses existed; otherwise it always used the GMM name, a leftover of the old "one KL per phase" design. With only the standard KL compiled, its inputs went to a key no loss used, so the component trained as an autoencoder without KL, silently. Every compiled KL loss is now fed. (The scheduler always compiles both, so its training was not affected; checked on a real model that the standard KL is reported and is part of the total.)
- ✅ (2026-11-21) **MD5: `predict` collects `x_parameters` only for modalities with `save_x`.** The flag was set per component and never reset, so once one modality had `save_x`, the following ones in `n_vars` order were collected too (memory only; what was written to `.obsm` was already right).
- ✅ (2026-11-21) **MD6: a weight of 0 in `compile` creates the loss switched off.** The design intent is to keep the losses of frozen parents: their values are logged, but they must not affect the child's training or the total loss used by early stopping. Today the scheduler does this by building every loss with its normal scale and pinning frozen weights to 0 (`ProgressiveScaler` floors progress at 1e-8, so the weight is `scale × 1e-16` and `train_step` can still recover the unweighted value). A weight of 0 passed to `compile` instead built the loss with **scale 0**, which could never be switched on and reported 0 / 0. Now weight 0 builds the loss with scale 1 and pins it to 0, for the KL losses and for the data losses in `loss_weights`. Checked on a real model: the loss reports its real value, the total loss is identical to not having it, and `pin_to(1)` switches it on. The docstring said "weight 0 means the loss is not created"; it now describes this behaviour.
- ✅ (2026-11-21) **MD7: the scaler-weighted metric no longer divides by 0** (`tf.math.divide_no_nan`), like the Variable branch. With MD6 fixed, a true 0 can no longer come from `compile`; this is a safeguard.
- ✅ (2026-11-21) **MD8: `compile` no longer modifies the caller's `loss_weights`.** It consumed the entries with `pop`, so recompiling with the same dict silently fell back to weight 1. It now works on a copy and warns about names that match no data loss (e.g. a typo), which were silently ignored. Existing losses are no longer rebuilt and discarded on every recompile.
- ✅ (2026-11-21) **MD4: `hierarchical_encode(strict=False)`** dropped a missing parent from its parent dicts, but `prepare_component_inputs` takes the parent list from the config and concatenated `None` in its place, so it crashed. Skipping a single parent cannot work anyway: the hierarchical encoder has a fixed input width. A component with a missing parent is now skipped (no `z_hat` output), and so are the components conditioned on its `z_hat`, which matches `decode(strict=False)`.
- ✅ (2026-11-21) **Parent widths are checked** in `hierarchical_encode` (strict or not): each parent `z` / `z_hat` must have its parent's latent width, otherwise a ValueError names the parent, the child, and both widths. Before, only Keras checked the total width, so swapped or wrongly sized parents that happened to add up were accepted silently.
- ✅ (2026-11-21) **The model keeps a private copy of its configs.** `Model.make` deep-copies the configs before building (it used to write `z_conditional_dims` / `z_hat_conditional_dims` into the caller's objects), the model stores them as `_component_configs`, and `component_configs` is a read-only property returning a tuple of copies. The forward pass never read the configs (the wiring is fixed in the functional graph), but `compile`, `train_step`, `predict`, `encode`, `hierarchical_encode` and `decode` do, so changing a config after `Model.make` used to change what they did without changing the graph.
- ✅ **MD2** (`model.py:414`): `super.__predict__(...)` should have been `super().predict`, so `predict` on non-MuData input always raised.
- ✅ **MD3** (`model.py:389` + `train_step`): batch keys weren't filtered to the model's inputs; Keras matches dict inputs by position, so a MuData with an unused modality fed e.g. `atac_matrix` in as `rna_matrix`. `Model.select_inputs` now restricts each batch in `predict` and `train_step`.
- ✅ Typing: `components` is a `Mapping[str, Component]`; component configs are ordered with `GeneralUtils.order_components` and set `z_conditional_dims` / `z_hat_conditional_dims` on declared fields.

## Worth a look

- `Model.compile` reuses existing losses on recompile and ignores new weights for them; the scheduler relies on this. Documented in its docstring; `Component.compile` rebuilds instead (see [modules](modules.md)).
- `save_z_hat` in `predict` is built from `save_z` and never used; dead code.
- In `encode`/`decode`, the config is passed as a plain `model_dump()` dict, so the parent lookups always come back `None`; parent inputs are skipped only by accident.
- Missing modality keys (`distribution_names.get`, `n_vars.get`) pass `None` along and fail far from the cause.
