# `cavachon/io`

[← Overview](../UNIT_TEST_REPORT.md)

| Status | Tests | Passed | Open bugs | Coverage | Coupling |
|---|---|---|---|---|---|
| ✅ Clean | [`test/io`](../../test/io) | 33 | 0 | 100% | Low: depends on 1 package, used by 1 |

- **Depends on:** config
- **Used by:** workflow
- **Mocked in tests:** the file readers (`pd.read_csv`, `mmread`, `open`); config objects are plain stand-ins.

## Fixed

- ✅ **H5** (`file_reader.py:106`): reading **two or more samples always failed**, because the per-sample features tables were concatenated (var had n_vars × n_samples rows). Features are taken once, from the first sample; a later sample whose features differ raises a `ValueError`.
- ✅ **H6** (with [config](config.md)): an empty `colnames` is treated like `None`.
- ✅ With `has_headers=True`, `index_col` was ignored and the obs/var index became 0..n.
- ✅ A dense `.mtx` file failed on `.tocsr()`.
- ✅ Deprecated `AnnData(..., dtype=)` removed.

## Worth a look

- `read_table` with `colnames` given and `index_col=None` calls `colnames.index(None)`, which raises ValueError. Marked with `# type: ignore[arg-type]` at `file_reader.py:172`.
