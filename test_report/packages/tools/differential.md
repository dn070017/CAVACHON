# tools: differential analyses

[← tools](README.md) · [← Overview](../../UNIT_TEST_REPORT.md)

Modules: `differential_analysis.py`, `hierarchical_differential_analysis.py`, `differential_attribution_analysis.py`
Tests: [`test/tools/differential`](../../../test/tools/differential) — 91 passed, 4 open bugs.

**Mocked:** `DataLoader` (a fake that builds a real `tf.data.Dataset` from dense X and `obs["batch"]`), `ReflectionHandler` (returns a fake distribution whose mean equals its parameters), and a deterministic differentiable `FakeModel`: `z[c] = X @ E[c]`, `z_hat = z`, `x = link(Σ z_hat[c] @ W[c]) + batch_effect`.

**Checked and correct:**
- Bayes factor / K values against hand computation on tiny inputs.
- The seeded global random batch effect in `compute_x_means`.
- Hierarchical donor substitution: the same donor cell is used for every donor component, and other components are untouched.
- `DifferentialAttributionAnalysis` integrated gradients: exact for a linear model, completeness holds, a closed-form left-Riemann value for a quadratic link, and signed/absolute contributions each sum to 1 (0 instead of NaN when nothing changes).

## Provenance columns

(2026-11-21) Every DEG/HDEG table records the analysis that produced it, on every row, so a table read back from disk can be matched to its analysis (used by the workflow's differential attribution, see [W1](../workflow.md#w1)). Both end with the **same run columns, in the same order**, written by `DifferentialAnalysis.record_run_parameters`:

`Component`, `Modality`, `UseCluster`, `ZSamplingSize`, `XSamplingSize`, `RandomSeed`, `SignificanceThreshold`, `LibrarySize` (only when the decoder scales by the library size, see [T13](#t13))

| Table | Before the run columns | Notes |
|---|---|---|
| DEG (`between_two_groups`, both `across_clusters*`) | the 6 statistics, `Significant`, `GroupA`, `GroupB` | `UseCluster` (new optional `use_cluster` argument) and the group labels are written when given; the `across_clusters*` methods and the workflow always pass them. |
| HDEG (`between_groups`, `across_clusters_pairwise`) | the 6 statistics, `Significant`, `MeanDelta(Substituted-Original)`, `InterventionType`, `DonorGroup`, `RecipientGroup`, `SamplingStrategy`, `DonorComponents` | `XSamplingSize` is the number of recipient cells actually drawn per round. |

Column names are `Constants.DIFFERENTIAL_ANALYSIS_*_COLUMN` and `Constants.HIERARCHICAL_DIFFERENTIAL_ANALYSIS_{DONOR,RECIPIENT}_GROUP_COLUMN`. Extra columns don't affect the consumers: attribution and enrichment read only their ranking column and the group columns. Checked with real DEG/HDEG tables written to disk and read back through `EnrichmentAnalysis.find_deg_tables`.

**Same sampling parameters** (2026-11-21). DEG and HDEG take `z_sampling_size` (sampling rounds), `x_sampling_size` (cells drawn per round) and `seed`, in the tools and in the config:
- DEG gained `seed` (applied at the start of each comparison, as in HDEG), so its runs are reproducible and `RandomSeed` is recorded.
- HDEG's `n_samples` became `z_sampling_size`, and it gained `x_sampling_size` (default `None`: the recipient group's size, the previous behaviour). The config still accepts `n_samples` with a DeprecationWarning.

**Donor/recipient are "groups".** HDEG uses `donor_group`/`recipient_group` (`between_clusters` → `between_groups`, columns `DonorGroup`/`RecipientGroup`, config fields `donor_group`/`recipient_group`, error messages), since the labels come from any obs column, not only clusters. The old config names were renamed without an alias. `across_clusters_pairwise` keeps its name: it does iterate over the clusters of `use_cluster`. The `between_groups` docstring defines `Mean(A)`, `Mean(B)`, `P` and `K` for the substitution (A = substituted, B = original, paired per recipient cell).

## Significant features

(2026-11-21) DEG and HDEG are the one place that decides which features are significant: `mark_significant` writes a boolean `Significant` column (`K(A>B|Z)` or `K(B>A|Z)` ≥ `significance_threshold`, default 3.2, new in the tools and in both configs; recorded as `SignificanceThreshold`). Everything downstream reads that column instead of applying its own threshold:
- `keep_only_significant` (tools and the workflow's single-pair mode) keeps the `Significant` rows (`keep_significant`, which also reads `"True"`/`"False"` strings from disk);
- enrichment's `min_num_degs` counts them (see [T12](visualization_enrichment.md#t12));
- differential attribution attributes exactly them: its `column`/`threshold`/`threshold_operator`/`use_absolute` options were removed (they defaulted to `|K| >= 3.2`, which kept tied features, as in T4). A table without the column is skipped with a message.

Checked with real DEG/HDEG tables written to disk and read back: `Significant` parses as bool, and a feature equal in both groups is not significant.

## Open bugs (strict xfails)

Found by the 2026-11-21 review of the differential and attribution tools. T13 is fixed (see below); T15 still needs a decision on the batch effect to decode with.

### T14

`hierarchical_differential_analysis.py:166` and `differential_analysis.py:620` — **the HDEG pairing is only half paired.** The substituted (A) and original (B) decodes use the same recipient cell, but each draws its own random batch effect. P(A>B|Z) mixes the substitution effect with batch-effect noise, which lowers power; the `between_groups` docstring calls the comparison paired.
- Failing input: a donor component the decoder doesn't use, `obs["batch"]` varying → P(A>B|Z) > 0 for every feature (expected 0, a tie in every pair).
- **Impact: yes**, with batch-effect columns.

### T15

`differential_attribution_analysis.py:248-261` — **differential attribution is computed at one arbitrary cell's batch effect and library size.** The "representative batch" is the first cell of the whole dataset (`batch_size=1`, first batch), which may belong to neither group.
- Failing input: the same data with the cells in reverse order → different `Expected(A)` / `Expected(B)` / attributions.
- **Impact: yes**: `Delta`, `Expected(A/B)` and every `attr_*` scale with that cell's library size and carry its batch effect.

### T17

`hierarchical_differential_analysis.py:302-314` — `between_groups` compares the labels as given, so integer-coded clusters (e.g. leiden stored as int) with `donor_group: "1"` from the YAML raise "Unknown donor group". DEG single-pair mode compares as strings (W2), so it works there.
- **Impact:** single-pair HDEG on integer-coded cluster columns.

### T18

`differential_analysis.py:144-150`, `hierarchical_differential_analysis.py:454-461` — a NaN in `use_cluster` becomes its own "cluster" (`NaN == NaN` is False), whose index is empty: numpy's "'a' cannot be empty" in pairwise mode (DEG and HDEG).
- **Impact:** any cells without a label in the cluster column.

## Fixed

- ✅ (2026-11-21) <a id="t13"></a>**T13** (`compute_x_means`): DEG decoded each sampled cell with its own matrix, so for decoders with `libsize_scaling` (ZINB, the RNA default) the mean was scaled by that cell's library size: groups that differed only in sequencing depth looked differential (identical latents, group B 5× deeper → every feature `Significant`). The batch effect was marginalized, the library size was not.
  - DEG and HDEG now decode every cell at **one library size, the median over all cells** (`DifferentialAnalysis.compute_decoding_reference`, which also builds the batch-effect pool in the same pass and replaces HDEG's `_build_batch_effect`; `compute_x_means` / `_compute_substituted_x_means` take `library_size` and pass it to `Model.decode(libsize=...)`). The library size is computed as in training (`Model.library_size`, for the decoded component).
  - The ZINB library size is Σ log1p(counts) and it scales the predicted **log** mean (μ = exp(soft_cap(m·L)) − 1, see [modules](../modules.md)), a design choice kept as is (the network compares log values). With one L for every cell, P/K/`Significant` depend only on m_A vs m_B, so they are the same for any constant L except near the ceiling of the log mean (soft cap at 15, see [modules](../modules.md); it was a hard clip at 7, where pairs above it tied); the median keeps m·L in the range the decoder was trained on. `Mean(A)`/`Mean(B)` are the expected expression at that depth.
  - New run column `LibrarySize` (written when the modality's decoder scales by it; `Constants.DIFFERENTIAL_ANALYSIS_LIBRARY_SIZE_COLUMN`). HDEG wasn't confounded (both decodes of a pair used the recipient cell's library size) and now uses the same median, so DEG and HDEG means are on one scale.
- ✅ (2026-11-21) <a id="t5"></a>**T5** (`differential_analysis.py`, `cluster_analysis.py`, `attribution_analysis.py`): the `batch_effect_colnames=None` / `distribution_names=None` defaults went straight into `DataLoader`, which calls `.get()` on them, so AttributeError (only when the tools are used without them; the workflow always passes both). `None` is now turned into an empty mapping, which DataLoader reads as "use `adata.uns['cavachon']` for every modality":
  - once, in `DifferentialAnalysis.__init__`, so its attributes are always mappings and the three subclasses pass them as they are (the hierarchical and differential-attribution `or {}` were removed); `batch_effect_encoders` defaults to `None` the same way instead of a shared mutable `dict()`;
  - at the `DataLoader` call in `AttributionAnalysis.__init__` and both `ClusterAnalysis.compute_cluster_log_probability*` methods (they don't keep the options).
  - The 12 `# type: ignore[arg-type]` lines are gone. Checked with the real DataLoader: `None` raises, the normalized defaults load.
- ✅ (2026-11-21) <a id="t4"></a>**T4**: `keep_only_significant` kept a feature when `|K(A>B)| >= 3.2` or `|K(B>A)| >= 3.2`, and `sort_output` ranked by `max(|K|)`. Decoded means that are equal in every comparison give P(A>B) = P(B>A) = 0, so both K ≈ −16 (evidence of *no* difference): such features were kept as "significant" and sorted to the top. Mostly-equal ones (both P ≈ 0.02, K ≈ −3.9) too. Ties happen when a feature's output doesn't depend on the compared latent: an HDEG substitution of a component the decoder doesn't use (the whole table), saturated Bernoulli probabilities (exactly 1.0 in float32), features that ignore z.
  - Filter and sort now use the signed Bayes factors, `max(K(A>B|Z), K(B>A|Z))` (`DifferentialAnalysis.bayes_factor_strength`, `keep_significant`, `sort_by_bayes_factor`; also the HDEG sort and the workflow's single-pair filter). Without ties `K(B>A) = −K(A>B)`, so real differences are kept and ordered exactly as before; ties are dropped and sort last.
  - Config templates and `CONFIG_REFERENCE.md` describe the signed rule.

## Worth a look

Design points from the 2026-11-21 review (working as written, but easy to misread):
- **"Z sampling" doesn't sample z.** `encode(training=False)` returns the posterior mean, so `z_sampling_size` is the number of resampling rounds over cells, and P(A>B|Z) measures how much the decoded means of the two groups overlap across cells, not posterior uncertainty.
- **`Delta` is not the DEG difference.** Differential attribution reports f(mean z_A) − f(mean z_B), the decoder at the group-mean latents; the DEG `Mean(A) − Mean(B)` averages the decoded output over cells, and the two differ when the decoder is nonlinear. With a subset of `with_respect_to`, `Expected(A)` keeps the other components at B's latent, so it is not group A's expected expression despite the name.
- **HDEG attribution explains a different intervention than the table.** On HDEG tables the attribution moves components to the donor group's mean z, while HDEG substituted individual donor cells' z. If `with_respect_to` differs from the table's `DonorComponents` it attributes components HDEG never swapped; nothing checks or warns.
- **`contrib_*` is unbounded**: when attributions cancel (+0.5 / −0.49), the signed `contrib_*` blows up (`abs_contrib_*` is fine).
- ✅ (2026-11-21) **Saturated genes** (fixed in [modules](../modules.md)): the ZINB decoder clipped the log-mean at 7 (mean ≤ e⁷ − 1 ≈ 1096), so above it the gradient was 0 (zero attribution) and DEG pairs tied. It is now a soft cap at 15.
- **Repeated work** (time only): each comparison redoes a full-data pass to build the batch-effect pool; HDEG runs `hierarchical_encode` twice per batch and discards the result.
- `seed` calls `np.random.seed`, NumPy's global generator, which other code shares.
- `across_clusters` with a single cluster crashes with numpy's "'a' cannot be empty" (the rest group is empty).
- `across_clusters` keys results by the raw cluster value, so int labels give int keys despite `Mapping[str, ...]`; `across_clusters_pairwise` uses f-string keys.
- `compute_differential_attribution(n_steps=0)` silently returns NaN; `n_steps` is not validated.
- `HierarchicalDifferentialAnalysis._normalize_donor_components(None, component)` doesn't check that `component` exists; an unknown one fails later with KeyError.
- `_encode_z_pool` calls `hierarchical_encode` and discards the result.
- The hierarchical `across_clusters_pairwise` has `donor_components` where the base has `z_sampling_size` (`# type: ignore[override]`); a positional call through a base-typed reference would mix them up.
- Docstrings for `across_clusters*` say `z_sampling_size` defaults to 10; the actual default is 5.
