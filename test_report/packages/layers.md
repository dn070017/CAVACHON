# `cavachon/layers`

[← Overview](../UNIT_TEST_REPORT.md)

| Status | Tests | Passed | Open bugs | Coverage | Coupling |
|---|---|---|---|---|---|
| 🟡 2 low bugs open | [`test/layers`](../../test/layers) | 149 | 2 | 99% | Moderate: depends on 2 packages, used by 7 |

- **Depends on:** environment, utils
- **Used by:** callbacks, dataloader, losses, model, modules, scheduler, tools
- **Mocked in tests:** `TensorUtils`.

## Open bugs (strict xfails)

| Location | Issue | Impact |
|---|---|---|
| `modifiers/to_sparse.py:51-53` | `ToSparse` only converts KerasTensor/SparseTensor; a dense eager `tf.Tensor` is returned unchanged, and the SparseTensor branch is dead code. | Latent (unused) |
| `progressive_scaler.py` | `activate(0)` clamps `current_iteration` to 0 forever, so the weight stays at 1e-16·scale instead of being fully on. | Possibly intended |

## Fixed

- ✅ `binarize.py:42`: the typo `__name___` made an out-of-range threshold raise `AttributeError` instead of warning.
- ✅ `from_config` failed on 8 layers (all parameterizers, both samplers, `ProgressiveScaler`): they took no `**kwargs`, so Keras 3's `trainable`/`dtype` caused a TypeError, breaking `clone_model` and `.keras` save/load.
- ✅ `normalize_library_size.py:61`: an all-zero row produced NaN (0/0).
- ✅ `scale_library_size.py:78`: `tf.ones((tensor.shape[0], 1))` failed inside a `tf.function` with an unknown batch dimension.
- ✅ The mixture grid side `ceil(K**(1/D))` was off by one for some exact powers, such as 3125 = 5^5, because of floating-point error.
- ✅ Samplers sample whenever `training=True` and their `stochastic` flag is on, whether or not their component is frozen (see [scheduler](scheduler.md) for how the flags are set).
- ✅ Stale docstrings: `DuplicateMatrix` (non-existent `threshold`), `ScaleLibrarySize`, Binarize (now says `>=`), `MultivariateNormalDiagModifier`.

## Worth a look

- The mixture `parameters` property only works when the layer was built with an input dimension of 1, and it squeezes away the component axis when K = 1.
- Binarize's default threshold of 1.0 turns values in [0, 1) into 0, which would wipe normalised or TF-IDF ATAC input.
- `MultivariateNormalDiagSampler` honours `trainable`, while `StudenttSampler` ignores it.
- `tf.train.Checkpoint` does not restore the mixture-prior loc/scale weights, because they are kept in Python lists. CAVACHON saves through Keras `model.save_weights`, which does restore them.
