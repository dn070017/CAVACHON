# `cavachon/workflow`

[← Overview](../UNIT_TEST_REPORT.md)

| Status | Tests | Passed | Open bugs | Coverage | Coupling |
|---|---|---|---|---|---|
| 🔴 Open bug affecting results | [`test/workflow`](../../test/workflow) | 86 | 3 | 100% | Very high: depends on 10 packages, used by 1 |

- **Depends on:** config, dataloader, environment, filter, io, modality, model, scheduler, tools, utils
- **Used by:** app
- **Mocked in tests:** all 16 cavachon names it imports, including every `tools` class; `os.makedirs`, `read_h5ad` and `to_csv` are recorders. A few tests read other packages' constructor signatures with `inspect`, to catch argument-order mistakes like M5.

**Checked and correct:** `batch_effect_colnames` and `distribution_names` are passed correctly throughout the workflow.

## Open bugs (strict xfails)

| # | Location | Issue | Impact |
|---|---|---|---|
| W1 | `workflow.py:572-596` | **Differential attribution uses every DEG/HDEG table in the folder**, including tables from other components and modalities, and runs them through its own component and `use_cluster`. | **Yes**: silently wrong attribution output when 2+ DEG/HDEG entries for different components exist |
| W2 | `workflow.py:609-629`, `441-443` | Group labels from DEG tables are converted to `str`, but the obs column isn't, so integer-coded or zero-padded labels give empty groups. | Only with non-string annotations |
| W7 | `workflow.py:377-390` | A `color_by` of the form `<obsm_key>/<index>` puts a `/` in the embedding filename, so the write fails before the h5mu is saved. | Only with that `color_by` form |

## Fixed

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
