# `cavachon/environment`

[← Overview](../UNIT_TEST_REPORT.md)

| Status | Tests | Passed | Open bugs | Coverage | Coupling |
|---|---|---|---|---|---|
| ✅ Clean | [`test/environment`](../../test/environment) | 39 | 0 | 100% | Standalone: depends on 0 packages, used by 11 |

- **Depends on:** nothing
- **Used by:** callbacks, config, dataloader, layers, modality, model, modules, scheduler, tools, utils, workflow
- **Mocked in tests:** nothing. `Constants` is the one cavachon name every other package's tests keep real.

The most widely imported package: a change here reaches 11 packages.

## Fixed

- ✅ The constant typo `TENSOR_NAME_ORIGIONAL_X` is now `TENSOR_NAME_ORIGINAL_X`; the old name is kept as a deprecated alias.

## Worth a look

- `Settings` derives every path from `__file__`, which would break once the package is installed into site-packages.
- `cavachon/` has no top-level `__init__.py` (mypy needs `explicit_package_bases`).
