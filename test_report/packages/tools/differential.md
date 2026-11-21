# tools: differential analyses

[← tools](README.md) · [← Overview](../../UNIT_TEST_REPORT.md)

Modules: `differential_analysis.py`, `hierarchical_differential_analysis.py`, `differential_attribution_analysis.py`
Tests: [`test/tools/differential`](../../../test/tools/differential) — 81 passed, 1 open bug.

**Mocked:** `DataLoader` (a fake that builds a real `tf.data.Dataset` from dense X and `obs["batch"]`), `ReflectionHandler` (returns a fake distribution whose mean equals its parameters), and a deterministic differentiable `FakeModel`: `z[c] = X @ E[c]`, `z_hat = z`, `x = link(Σ z_hat[c] @ W[c]) + batch_effect`.

**Checked and correct:**
- Bayes factor / K values against hand computation on tiny inputs.
- The seeded global random batch effect in `compute_x_means`.
- Hierarchical donor substitution: the same donor cell is used for every donor component, and other components are untouched.
- `DifferentialAttributionAnalysis` integrated gradients: exact for a linear model, completeness holds, a closed-form left-Riemann value for a quadratic link, and signed/absolute contributions each sum to 1 (0 instead of NaN when nothing changes).

## Provenance columns

(2026-11-21) Every DEG/HDEG table records the analysis that produced it, on every row, so a table read back from disk can be matched to its analysis (used by the workflow's differential attribution, see [W1](../workflow.md#w1)). Both end with the **same run columns, in the same order**, written by `DifferentialAnalysis.record_run_parameters`:

`Component`, `Modality`, `UseCluster`, `ZSamplingSize`, `XSamplingSize`, `RandomSeed`

| Table | Before the run columns | Notes |
|---|---|---|
| DEG (`between_two_groups`, both `across_clusters*`) | the 6 statistics, `GroupA`, `GroupB` | `UseCluster` (new optional `use_cluster` argument) and the group labels are written when given; the `across_clusters*` methods and the workflow always pass them. |
| HDEG (`between_groups`, `across_clusters_pairwise`) | the 6 statistics, `MeanDelta(Substituted-Original)`, `InterventionType`, `DonorGroup`, `RecipientGroup`, `SamplingStrategy`, `DonorComponents` | `XSamplingSize` is the number of recipient cells actually drawn per round. |

Column names are `Constants.DIFFERENTIAL_ANALYSIS_*_COLUMN` and `Constants.HIERARCHICAL_DIFFERENTIAL_ANALYSIS_{DONOR,RECIPIENT}_GROUP_COLUMN`. Extra columns don't affect the consumers: attribution and enrichment read only their ranking column and the group columns. Checked with real DEG/HDEG tables written to disk and read back through `EnrichmentAnalysis.find_deg_tables`.

**Same sampling parameters** (2026-11-21). DEG and HDEG take `z_sampling_size` (sampling rounds), `x_sampling_size` (cells drawn per round) and `seed`, in the tools and in the config:
- DEG gained `seed` (applied at the start of each comparison, as in HDEG), so its runs are reproducible and `RandomSeed` is recorded.
- HDEG's `n_samples` became `z_sampling_size`, and it gained `x_sampling_size` (default `None`: the recipient group's size, the previous behaviour). The config still accepts `n_samples` with a DeprecationWarning.

**Donor/recipient are "groups".** HDEG uses `donor_group`/`recipient_group` (`between_clusters` → `between_groups`, columns `DonorGroup`/`RecipientGroup`, config fields `donor_group`/`recipient_group`, error messages), since the labels come from any obs column, not only clusters. The old config names were renamed without an alias. `across_clusters_pairwise` keeps its name: it does iterate over the clusters of `use_cluster`. The `between_groups` docstring defines `Mean(A)`, `Mean(B)`, `P` and `K` for the substitution (A = substituted, B = original, paired per recipient cell).

## Open bugs (strict xfails)

### T5

`differential_analysis.py:307-340` (also in [cluster_attribution](cluster_attribution.md#t5)) — `batch_effect_colnames=None` / `distribution_names=None` defaults go straight into `DataLoader`, which calls `.get()` on them, so AttributeError. The hierarchical and differential-attribution subclasses already use `or {}`.
- **Impact:** only when called with defaults (the workflow always passes both). Marked with 6 `# type: ignore[arg-type]` lines in this file.

## Fixed

- ✅ (2026-11-21) <a id="t4"></a>**T4**: `keep_only_significant` kept a feature when `|K(A>B)| >= 3.2` or `|K(B>A)| >= 3.2`, and `sort_output` ranked by `max(|K|)`. Decoded means that are equal in every comparison give P(A>B) = P(B>A) = 0, so both K ≈ −16 (evidence of *no* difference): such features were kept as "significant" and sorted to the top. Mostly-equal ones (both P ≈ 0.02, K ≈ −3.9) too. Ties happen when a feature's output doesn't depend on the compared latent: an HDEG substitution of a component the decoder doesn't use (the whole table), saturated Bernoulli probabilities (exactly 1.0 in float32), features that ignore z.
  - Filter and sort now use the signed Bayes factors, `max(K(A>B|Z), K(B>A|Z))` (`DifferentialAnalysis.bayes_factor_strength`, `keep_significant`, `sort_by_bayes_factor`; also the HDEG sort and the workflow's single-pair filter). Without ties `K(B>A) = −K(A>B)`, so real differences are kept and ordered exactly as before; ties are dropped and sort last.
  - Config templates and `CONFIG_REFERENCE.md` describe the signed rule.

## Worth a look

- A NaN cluster label, or `across_clusters` with a single cluster, crashes with numpy's "'a' cannot be empty".
- `across_clusters` keys results by the raw cluster value, so int labels give int keys despite `Mapping[str, ...]`; `across_clusters_pairwise` uses f-string keys.
- `compute_differential_attribution(n_steps=0)` silently returns NaN; `n_steps` is not validated.
- `HierarchicalDifferentialAnalysis._normalize_donor_components(None, component)` doesn't check that `component` exists; an unknown one fails later with KeyError.
- `_encode_z_pool` calls `hierarchical_encode` and discards the result.
- The hierarchical `across_clusters_pairwise` has `donor_components` where the base has `z_sampling_size` (`# type: ignore[override]`); a positional call through a base-typed reference would mix them up.
- Docstrings for `across_clusters*` say `z_sampling_size` defaults to 10; the actual default is 5.
