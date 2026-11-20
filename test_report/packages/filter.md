# `cavachon/filter`

[← Overview](../UNIT_TEST_REPORT.md)

| Status | Tests | Passed | Open bugs | Coverage | Coupling |
|---|---|---|---|---|---|
| ✅ Clean | [`test/filter`](../../test/filter) | 109 | 0 | 100% | Low: depends on 1 package, used by 1 |

- **Depends on:** utils
- **Used by:** workflow
- **Mocked in tests:** `GeneralUtils`, `ReflectionHandler`, `FilterConfig`; scanpy stays real.

## Fixed

- ✅ **Default FilterQC `qc_vars`** (`filter_qc.py:96`): the default `["MT"]` was an unanchored regex that matched MTOR, MTHFR and similar genes, **inflating `pct_counts_MT`**, and never matched mouse `mt-` genes. Mitochondrial genes now match `(?i)^MT-` and spike-ins `^ERCC-`.
  - Human `pct_counts_MT` drops; mouse MT filtering now removes cells it used to keep.
  - Column names are unchanged, and configs that set `qc_vars` explicitly behave as before.
- ✅ A modality with no steps entry crashed with `NoneType is not iterable` (`anndata_filter_handler.py:87`).
- ✅ `filter_qc` crashed when `var.index.name` was unset (any AnnData not loaded through `FileReader`); it now matches on `var_names`.
- ✅ `filter_threshold` is optional, as documented: metrics only, every cell kept.
- ✅ scanpy's default `percent_top` includes 500, which raised on modalities with fewer than 500 features; it is now trimmed to the number of features.
- ✅ The filter classes no longer subclass `typing.Callable`.

## Worth a look

- Mutation is inconsistent across steps: FilterCells and FilterGenes work in place, while FilterQC returns a copy but writes QC columns to the input.
