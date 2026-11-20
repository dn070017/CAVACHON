# tools: interactive visualization and enrichment

[← tools](README.md) · [← Overview](../../UNIT_TEST_REPORT.md)

Modules: `interactive_visualization.py`, `enrichment_analysis.py`
Tests: [`test/tools/visualization_enrichment`](../../../test/tools/visualization_enrichment) — 108 passed, 6 open bugs.

**Mocked:** `AttributionAnalysis`, `ClusterAnalysis`, `DifferentialAnalysis` (in interactive_visualization); `InteractiveVisualization`, `DataFrameUtils` and **gseapy** (in enrichment_analysis). `Figure.show`/`write_html`/`write_image`, `Path.mkdir`, `TemporaryDirectory`, `copytree` and `pd.read_csv` are blocked/recorded. TSNE and UMAP are a deterministic fake; PCA is real.

**Checked and correct:** trace data, labels, colours and axis titles of the returned plotly figures; `write_html` is called only when a filename is given.

## Open bugs (strict xfails)

### T7

`interactive_visualization.py:89-103` — `bar()` ignores `**kwargs` (documented to go to `fig.update_layout`) when `group` is None. `bar(df, x="x", y="y", title="T")` leaves the title as None; `neighbors_with_same_annotations(group_by_cluster=False)` loses its titles.
- **Impact:** cosmetic.

### T8

`interactive_visualization.py:200` — `adata.uns.get("cavachon", "").get("name", "")` raises `'str' object has no attribute 'get'` when uns has no `cavachon` entry, e.g. `embedding(adata_without_uns_cavachon, method="pca")`.
- **Impact:** only for AnnData not produced by cavachon.

### T9

`interactive_visualization.py:540-544` — `attribution_score` writes every figure in `with_respect_to` to the same `filename`; only the last survives (`with_respect_to=["z1","z2"], filename="a.html"` calls `write_html("a.html")` twice).
- **Impact: yes**, with 2+ `with_respect_to`.

### T10

`interactive_visualization.py:848` — `float(x[:-2])` on gseapy's `"Gene %"` format (`"12.35%"`) gives 12.3; it should strip only the `%`.
- **Impact:** small numeric error in ring plots.

### T11

`interactive_visualization.py:956, 969, 1028, 1041` — `prerank_enrichment_score` passes the full `Target` column as hover text to traces that plot only a subset, so hover labels are offset (ranking `[2,1,0,-1,-2]`: the `index >= zero` trace has x `[2,3,4]` but hover text `g0,g1,g2`).
- **Impact:** misleading hover labels.

### T12

`enrichment_analysis.py:286-292` — `run()` applies the **term** threshold/operator ("Metric value threshold for selected terms", default FDR q-val <= 0.05) to the **DEG ranking** column, keeping only genes with K <= 0.05. K values `[5.0, 1.0, -2.0]` pass only `g3` to `gseapy.prerank`.
- Added on purpose in commit a841a53 with `min_num_degs`, but one threshold can't serve both an FDR cutoff and a K cutoff. **Needs the author's call.**
- **Impact: yes**, if unintended: most of the ranked list fed to prerank is discarded.

## Worth a look

- `embedding(color=...)`: a list crashed at `color in adata.obs`; the type is now `Optional[str]`, but the docstring still mentions Sequence.
- `differential_volcano_plot` adds `LogFC(A/B)`, `K(A!=B|Z)`, `Significant`, `Target` columns to the caller's DataFrame; `Mean(B)=0` gives an infinite LogFC.
- `differential_attribution_radviz` docstring says `logfc_column` defaults to `LogFC(A/B)`; the code default is `"LogFC"` (which is what `DifferentialAttributionAnalysis` produces).
- `prerank_ringplot` crashes with `max() arg is an empty sequence` when no rows pass, and always filters `metric <= threshold`, ignoring `threshold_operator` (so it can disagree with `run`/`_visualize`).
- `attribution_score` with an empty `with_respect_to` raises IndexError.
- In `prerank_enrichment_score`, hit markers use `<= zero_score_index` but the score traces use `<`.
- `_visualize` checks `res2d`/`results` for None only after calling `prerank_ringplot`, which would already have crashed.
- `_safe_filename` can collide (`"A B"` and `"A/B"` both become `A_B`) or produce an empty name (`"!!!"`).
