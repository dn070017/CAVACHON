# `cavachon/workflow`

[← Overview](../UNIT_TEST_REPORT.md)

| Status | Tests | Passed | Open bugs | Coverage | Coupling |
|---|---|---|---|---|---|
| 🔴 Open bug affecting results | [`test/workflow`](../../test/workflow) | 89 | 1 | 100% | Very high: depends on 10 packages, used by 1 |

- **Depends on:** config, dataloader, environment, filter, io, modality, model, scheduler, tools, utils
- **Used by:** app
- **Mocked in tests:** all 16 cavachon names it imports, including every `tools` class; `os.makedirs`, `read_h5ad` and `to_csv` are recorders. A few tests read other packages' constructor signatures with `inspect`, to catch argument-order mistakes like M5.

**Checked and correct:** `batch_effect_colnames` and `distribution_names` are passed correctly throughout the workflow.

## Open bugs (strict xfails)

| # | Location | Issue | Impact |
|---|---|---|---|
| W1 | `workflow.py:625-635` | **Differential attribution uses every DEG/HDEG table in the folder**, including tables from other components and modalities, and runs them through its own component and `use_cluster`. | **Yes**: silently wrong attribution output when 2+ DEG/HDEG entries for different components exist |

## Fixed

- ✅ (2026-11-21) **W2** (`perform_differential_analysis`, `perform_differential_attribution_analysis`): group labels (from the config or `str()`-ed from DEG tables) were compared to the raw obs column, so integer-coded annotations gave empty groups. The obs column is now compared as strings. Zero-padded labels (`"01"` vs `1`) still don't match, since a TSV read-back loses the padding.
- ✅ (2026-11-21) **W7** (`visualize_embedding`): a `color_by` of the form `<obsm_key>/<index>` put a `/` in the embedding filename. It is replaced with `_`, as in the other output names.
- ✅ **M5** (`workflow.py:242-251`): positional arguments to `SequentialTrainingScheduler` were swapped (`batch_effect_colnames` ↔ `distribution_names`). They are passed by keyword now.
- ✅ **W3–W6** (`workflow.py:426-502`):
  - DEG filenames no longer contain spaces;
  - a `/` in a label no longer crashes pairwise DEG: `GroupA`/`GroupB` come from the tool instead of being re-parsed from `A/B` keys;
  - single-pair DEG results are keyed by (modality, component, `use_cluster`) like pairwise mode, so they're no longer overwritten. Filenames change from `rna_comp_a_b_a_b.tsv` to `rna_comp_<use_cluster>_a_b.tsv`;
  - the DEG/HDEG output folders are created only when that analysis is configured.

## Worth a look

- `train_model` (`workflow.py:308`) calls `self.model.compile()` **inside** the per-component freeze loop, so it compiles once per component. Looks like an indentation slip; harmless but wasteful.
- `load_model_weights` swallows OSError with only a warning. With `load_weights: true, train: false`, the run continues on random weights.
- `run()` calls `predict()` before `load_model_weights()`, probably to build the variables, at the cost of a full predict pass.
