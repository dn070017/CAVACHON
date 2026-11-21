# tools: interactive visualization and enrichment

[← tools](README.md) · [← Overview](../../UNIT_TEST_REPORT.md)

Modules: `interactive_visualization.py`, `enrichment_analysis.py`
Tests: [`test/tools/visualization_enrichment`](../../../test/tools/visualization_enrichment) — 126 passed, 0 open bugs.

**Mocked:** `AttributionAnalysis`, `ClusterAnalysis`, `DifferentialAnalysis` (in interactive_visualization); `InteractiveVisualization`, `DataFrameUtils` and **gseapy** (in enrichment_analysis). `Figure.show`/`write_html`/`write_image`, `Path.mkdir`, `TemporaryDirectory`, `copytree` and `pd.read_csv` are blocked/recorded. TSNE and UMAP are a deterministic fake; PCA is real.

**Checked and correct:** trace data, labels, colours and axis titles of the returned plotly figures; `write_html` is called only when a filename is given.

## Open bugs (strict xfails)

None.

## Fixed

- ✅ (2026-11-21) <a id="t8"></a>**T8** (`embedding`): `adata.uns.get("cavachon", "").get("name", "")` raised AttributeError for an AnnData not produced by cavachon (no `uns['cavachon']`), e.g. a custom model's latent space. The name (the modality name cavachon stores) only built the default title `Z(<name>)`; the embedding itself always comes from `use_rep`. `uns` is no longer read: the default title is `use_rep` (the obsm key, e.g. `z`), or `Z` for an array. The workflow and `scripts/evaluate_embeddings.py` always pass their own title, so their output is unchanged.
- ✅ (2026-11-21) <a id="t12"></a>**T12** (`EnrichmentAnalysis.run`): one `threshold`/`threshold_operator` pair filtered both the DEG ranking (`column`) and the pathways (`metric`), so one of the two was always wrong; with the template config (3.2, `geq`) no pathway could ever pass `FDR q-val >= 3.2`, so enrichment never wrote output. The threshold was meant to decide **which DEG tables are worth analysing** (author's call), so:
  - significance is defined once, by the differential analysis (`Significant` column, see [differential](differential.md#significant-features)); `min_num_degs` counts those features (either direction) and skips tables with fewer, or without the column (when `min_num_degs > 0`);
  - the **full ranking** in `column` goes to `gseapy.prerank` (GSEA expects the whole ranked list);
  - pathways are always `FDR q-val < 0.05` (`TERM_METRIC` / `TERM_THRESHOLD`; strict, was `<=`); `metric`/`threshold`/`threshold_operator` were removed from `run`, `run_directory`, `visualize` and the config;
  - a table where every feature is significant (filtered with `keep_only_significant`) gives a truncated ranking: `run` warns, and so does the config when a DEG entry has `keep_only_significant` and enrichment runs on DEG tables.
- ✅ (2026-11-21) <a id="t7"></a>**T7** (`bar`): without a `group`, `**kwargs` (documented to go to `fig.update_layout`) were ignored, so titles were lost, e.g. in `neighbors_with_same_annotations(group_by_cluster=False)`. They are now applied in both branches.
- ✅ (2026-11-21) <a id="t9"></a>**T9** (`attribution_score`): every `with_respect_to` figure was written to the same `filename`, so only the last survived. With 2+ variables, each is saved as `<name>_<wrt><ext>` (`a.html` → `a_z1.html`, `a_z2.html`; `/` in a name becomes `_`), via the new `InteractiveVisualization.attribution_score_filename`. A single variable keeps the filename as is, so the workflow (which calls it once per variable) writes the same files as before.
- ✅ (2026-11-21) <a id="t10"></a>**T10** (`prerank_ringplot`): `float(x[:-2])` on gseapy's `"12.35%"` gave 12.3. Only the `%` (and surrounding spaces) is stripped now; numeric values also work.
- ✅ (2026-11-21) <a id="t11"></a>**T11** (`prerank_enrichment_score`): the split enrichment-score and rank-metric traces got the full `Target` column as hover text, so hover labels were offset from their points. Each trace now takes x, y and hover text from the same rows.

## Worth a look

- `embedding(color=...)`: a list crashed at `color in adata.obs`; the type is now `Optional[str]`, but the docstring still mentions Sequence.
- `differential_volcano_plot` adds `LogFC(A/B)`, `K(A!=B|Z)`, `Significant`, `Target` columns to the caller's DataFrame; `Mean(B)=0` gives an infinite LogFC.
- `differential_attribution_radviz` docstring says `logfc_column` defaults to `LogFC(A/B)`; the code default is `"LogFC"` (which is what `DifferentialAttributionAnalysis` produces).
- `prerank_ringplot` crashes with `max() arg is an empty sequence` when no rows pass, and always filters `metric <= threshold`, ignoring `threshold_operator` (so it can disagree with `run`/`_visualize`).
- `attribution_score` with an empty `with_respect_to` raises IndexError.
- In `prerank_enrichment_score`, hit markers use `<= zero_score_index` but the score traces use `<`.
- `_visualize` checks `res2d`/`results` for None only after calling `prerank_ringplot`, which would already have crashed.
- `_safe_filename` can collide (`"A B"` and `"A/B"` both become `A_B`) or produce an empty name (`"!!!"`).
