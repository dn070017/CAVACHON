# History

[← Overview](../UNIT_TEST_REPORT.md)

## Scope

The old tests were removed: `test/` (losses, scheduler, utils) and the root-level `test_differential_analysis.py`, `test_enrichment_analysis.py`, `test_hierarchical_differential_analysis.py` and `test_model_encode_decode.py`. The tracked files are recoverable from git. The untracked `test/losses/` was backed up outside the repo.

Packages were covered in order of coupling (see the [overview](../UNIT_TEST_REPORT.md#coupling)):

1. Packages importing fewer than 4 other cavachon packages: distributions, losses, layers, utils, environment, config, io, dataloader, modality, filter, app. These first tests found 80 bugs; after the fix rounds, 7 low-severity ones remain open on purpose.
2. Packages importing exactly 4: callbacks, scheduler, modules.
3. model and workflow.
4. tools.

## Fix rounds

| Date | Round | Details |
|---|---|---|
| 2026-11-13 | H1–H6 (high) | [losses](../packages/losses.md), [dataloader](../packages/dataloader.md), [utils](../packages/utils.md), [io](../packages/io.md), [config](../packages/config.md) |
| 2026-11-13 | Medium items | across the low-coupling packages |
| 2026-11-13 | `create_backbone_layers(reverse=True)` | [utils](../packages/utils.md); changes the encoder architecture, retraining required |
| 2026-11-13 | ZINB mixture weights from logits | [distributions](../packages/distributions.md) |
| 2026-11-13 | Selected low items | across packages |
| 2026-11-13 | M1, M2, M5, M6, M7 | [modules](../packages/modules.md), [workflow](../packages/workflow.md), [scheduler](../packages/scheduler.md) |
| 2026-11-13 | EarlyStoppingConfig, z_hat annealing | [scheduler](../packages/scheduler.md) |
| 2026-11-13 | Config validation, Component NaN gradients, dataset API, constant typo, docstrings | [config](../packages/config.md), [modules](../packages/modules.md), [dataloader](../packages/dataloader.md), [environment](../packages/environment.md) |
| 2026-11-13 | MD2, MD3, W3–W6 | [model](../packages/model.md), [workflow](../packages/workflow.md) |
| 2026-11-14 | ruff check / format | whole repo |
| 2026-11-14 | mypy setup and typing fixes | whole repo, then [tools](../packages/tools/README.md) |

## End-to-end checks

- **After the medium round:** a run on synthetic data used the real code throughout, from config → h5ad → filter → MultiModality → DataLoader → Model. It trained 7 epochs, ran predict and saved the weights. The ATAC cells were stored in reverse order and were aligned correctly. The batch-effect widths matched `n_vars`, and the embeddings were finite. Old layer weights still loaded into the fixed layers.
- **After EarlyStoppingConfig:** a run with `patience: 2` stopped each regular phase after 3 epochs.
- **After the z_hat changes:** the sampler flags reset correctly, and `z_hat` stayed finite.

## Lint and typing

- `pixi run ruff check` and `ruff format --check` are clean on `cavachon/` and `test/`.
- `pixi run mypy` reports no errors in the whole package. The mypy and coverage dev dependencies and the `[tool.mypy]` config are in `pyproject.toml`.
- The type errors exposed a few latent bugs, recorded in the package pages: [io](../packages/io.md) (`read_table`), [dataloader](../packages/dataloader.md) (`create_dataset` return), [model](../packages/model.md) (`save_z_hat`, encode/decode lookups), [tools](../packages/tools/README.md) (T5, T6).
