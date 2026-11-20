# `cavachon/utils`

[← Overview](../UNIT_TEST_REPORT.md)

| Status | Tests | Passed | Open bugs | Coverage | Coupling |
|---|---|---|---|---|---|
| ⏸️ Reviewed, deferred | [`test/utils`](../../test/utils) | 188 | 2 | 100% | Low: depends on 1 package, used by 10 |

- **Depends on:** environment
- **Used by:** config, dataloader, filter, layers, losses, model, modules, scheduler, tools, workflow
- **Mocked in tests:** `glob`/`import_module`/`os.path` for `ReflectionHandler`; everything else is pure computation.

Second most widely imported package after environment.

## Open bugs (strict xfails)

> **Reviewed 2026-11-20 — deferred.** Both bugs are confirmed and accepted for now; no fix is planned. Their strict xfails stay in place, so a fix would still be flagged as XPASS.

| Location | Issue | Impact | Decision |
|---|---|---|---|
| `reflection_handler.py` | `_camel_to_snake` does not split acronyms followed by a word (`ZINBDistribution` → `zinbdistribution`). | Edge case | ⏸️ Deferred |
| `reflection_handler.py` | Lookup validation uses `assert`, which `python -O` strips (a miss then becomes IndexError). | Edge case | ⏸️ Deferred |

## Fixed

- ✅ **H4** (`tensor_utils.py:284`): a dense `adata.X` crashed with `'numpy.ndarray' has no attribute 'tocoo'`. `spmatrix_to_sparse_tensor` converts non-sparse input to COO first.
- ✅ **`create_backbone_layers(reverse=True)`** reversed the flat `[Dense, LN, Dense, LN]` list, giving `LN → Dense → … → Dense` with no norm after the last Dense. It now reverses only the widths: encoders are Dense64 → LN → Dense64 → LN → Dense32 → LN. ⚠ **Architecture change**: older checkpoints don't load; retraining was decided.
- ✅ `create_tensor_from_df(encoder_dict=dict())` had the same mutable-default bug as H2.
- ✅ The encoding of a numeric batch column depended on n_obs (the 20% unique-ratio heuristic in `check_is_categorical`); a reused encoder now fixes it.
- ✅ `check_is_categorical` raised a TypeError on string columns with NaN.
- ✅ `anndata_utils.py:42-50`: a NaN group label added an all-NaN pseudo-bulk row.
- ✅ `ReflectionHandler`: the "more than one file" check could never trigger (only `glob(...)[0]` was used); a class name matching files at the same depth in two places now raises.
- ✅ `general_utils`: `tensorflow_compatible_str` accepted a trailing `\n`; an unknown parent gave "None cannot be a node"; duplicate component names were accepted.
- ✅ `tensor_utils.split` added an empty chunk when n was divisible by batch_size; `remove_nan_gradients` changed its input list.
- ✅ Duplicate sparse entries crashed instead of being summed.
- ✅ `np.mat` (removed in NumPy 2) and `AnnData(..., dtype=)` replaced.

## Worth a look

- The `ReflectionHandler` glob is not actually recursive.
