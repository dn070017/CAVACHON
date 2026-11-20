# `cavachon/modality`

[← Overview](../UNIT_TEST_REPORT.md)

| Status | Tests | Passed | Open bugs | Coverage | Coupling |
|---|---|---|---|---|---|
| ✅ Clean | [`test/modality`](../../test/modality) | 34 | 0 | 100% | Low: depends on 1 package, used by 1 |

- **Depends on:** environment
- **Used by:** workflow
- **Mocked in tests:** nothing beyond `Constants` (kept real); anndata and muon stay real.

## Fixed

- ✅ `MultiModality` changed the caller's mapping or MuData in place, and with MuData input dropped non-shared cells from the **original** object. `MultiModality`, `Modality` and the config validators no longer change their inputs; `workflow.py` uses only the returned objects.
- ✅ `Modality` changed the caller's `uns` dict and, for AnnData input, the input's `.uns`.
- ✅ Duplicate `obs_names` produced an unclear pandas error; they now raise a clear one.
