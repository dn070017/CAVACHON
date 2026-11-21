# tools: differential analyses

[← tools](README.md) · [← Overview](../../UNIT_TEST_REPORT.md)

Modules: `differential_analysis.py`, `hierarchical_differential_analysis.py`, `differential_attribution_analysis.py`
Tests: [`test/tools/differential`](../../../test/tools/differential) — 74 passed, 2 open bugs.

**Mocked:** `DataLoader` (a fake that builds a real `tf.data.Dataset` from dense X and `obs["batch"]`), `ReflectionHandler` (returns a fake distribution whose mean equals its parameters), and a deterministic differentiable `FakeModel`: `z[c] = X @ E[c]`, `z_hat = z`, `x = link(Σ z_hat[c] @ W[c]) + batch_effect`.

**Checked and correct:**
- Bayes factor / K values against hand computation on tiny inputs.
- The seeded global random batch effect in `compute_x_means`.
- Hierarchical donor substitution: the same donor cell is used for every donor component, and other components are untouched.
- `DifferentialAttributionAnalysis` integrated gradients: exact for a linear model, completeness holds, a closed-form left-Riemann value for a quadratic link, and signed/absolute contributions each sum to 1 (0 instead of NaN when nothing changes).

## Provenance columns

(2026-11-21) Every DEG/HDEG table records the analysis that produced it, on every row, so a table read back from disk can be matched to its analysis (used by the workflow's differential attribution, see [W1](../workflow.md#w1)):

| Table | Columns added |
|---|---|
| DEG (`between_two_groups`, and both `across_clusters*`) | `Component`, `Modality`, `ZSamplingSize`, `XSamplingSize`, and `UseCluster` when given (new optional `use_cluster` argument; the `across_clusters*` methods pass it) |
| HDEG (`between_clusters`, `across_clusters_pairwise`) | `Component`, `Modality`, `UseCluster`, `NSamples` (next to the existing `RandomSeed` and `DonorComponents`) |

The column names of `Component`, `Modality` and `UseCluster` are `Constants.DIFFERENTIAL_ANALYSIS_*_COLUMN`. Extra columns don't affect the consumers: attribution and enrichment read only their ranking column and the group columns. Checked with real DEG/HDEG tables written to disk and read back through `EnrichmentAnalysis.find_deg_tables`.

## Open bugs (strict xfails)

### T4

`differential_analysis.py:142, 224` — `keep_only_significant` keeps a feature when `|K(A>B)| >= 3.2` or `|K(B>A)| >= 3.2`. Identical decoded means give P(A>B) = P(B>A) = 0, so both K ≈ −16 and the feature is kept as "significant" and sorted to the top.
- Failing input: a 4-cell MuData with every X = 1, clusters a,a,b,b, `across_clusters_pairwise(..., keep_only_significant=True)`. `"a/b"` has 2 rows; it should have 0.
- **Impact: yes**, for features with no difference at all.

### T5

`differential_analysis.py:307-340` (also in [cluster_attribution](cluster_attribution.md#t5)) — `batch_effect_colnames=None` / `distribution_names=None` defaults go straight into `DataLoader`, which calls `.get()` on them, so AttributeError. The hierarchical and differential-attribution subclasses already use `or {}`.
- **Impact:** only when called with defaults (the workflow always passes both). Marked with 6 `# type: ignore[arg-type]` lines in this file.

## Worth a look

- A NaN cluster label, or `across_clusters` with a single cluster, crashes with numpy's "'a' cannot be empty".
- `across_clusters` keys results by the raw cluster value, so int labels give int keys despite `Mapping[str, ...]`; `across_clusters_pairwise` uses f-string keys.
- `compute_differential_attribution(n_steps=0)` silently returns NaN; `n_steps` is not validated.
- `HierarchicalDifferentialAnalysis._normalize_donor_components(None, component)` doesn't check that `component` exists; an unknown one fails later with KeyError.
- `_encode_z_pool` calls `hierarchical_encode` and discards the result.
- The hierarchical `across_clusters_pairwise` has `donor_components` where the base has `z_sampling_size` (`# type: ignore[override]`); a positional call through a base-typed reference would mix them up.
- Docstrings for `across_clusters*` say `z_sampling_size` defaults to 10; the actual default is 5.
