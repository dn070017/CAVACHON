# Unit Test Report

Updated: 2026-11-21 · Branch: `feature/refactor-tests-attribution`

**Whole suite: 1804 passed, 14 open bugs (strict xfails), 0 failures.** `ruff check`, `ruff format --check` and `mypy` are clean on the whole package.

Every package has isolated unit tests: other cavachon packages are mocked, and there is no file, network or subprocess I/O. See [Isolation](guide/isolation.md) and [Conventions and commands](guide/conventions.md). How the suite got here: [History](guide/history.md).

## Status by package

| Package | Status | Coupling | Depends on | Used by | Passed | Open bugs | Coverage |
|---|---|---|---|---|---|---|---|
| [environment](packages/environment.md) | ✅ | 🟢 Standalone | 0 | 11 | 39 | 0 | 100% |
| [distributions](packages/distributions.md) | ✅ | 🟢 Standalone | 0 | 3 | 61 | 0 | 99% |
| [utils](packages/utils.md) | ⏸️ | 🟢 Low | 1 | 10 | 188 | 2 | 100% |
| [modality](packages/modality.md) | ✅ | 🟢 Low | 1 | 1 | 34 | 0 | 100% |
| [filter](packages/filter.md) | ✅ | 🟢 Low | 1 | 1 | 109 | 0 | 100% |
| [io](packages/io.md) | ✅ | 🟢 Low | 1 | 1 | 33 | 0 | 100% |
| [app](packages/app.md) | ✅ | 🟢 Low | 1 | 0 | 11 | 0 | 87% |
| [config](packages/config.md) | ✅ | 🟡 Moderate | 2 | 4 | 209 | 0 | 99% |
| [layers](packages/layers.md) | ⏸️ | 🟡 Moderate | 2 | 6 | 153 | 1 | 99% |
| [losses](packages/losses.md) | ⏸️ | 🟡 Moderate | 3 | 2 | 38 | 2 | 100% |
| [dataloader](packages/dataloader.md) | ✅ | 🟡 Moderate | 3 | 4 | 86 | 0 | 100% |
| [callbacks](packages/callbacks.md) | ⏸️ | 🟠 High | 4 | 1 | 46 | 4 | 89% |
| [scheduler](packages/scheduler.md) | ✅ | 🟠 High | 4 | 1 | 121 | 0 | 99% |
| [modules](packages/modules.md) | 🟡 | 🟠 High | 4 | 2 | 198 | 4 | 100% |
| [model](packages/model.md) | ⏸️ | 🔴 Very high | 7 | 2 | 112 | 0 | 100% |
| [tools](packages/tools/README.md) | 🔴 | 🔴 Very high | 7 | 1 | 266 | 1 | 100% |
| [workflow](packages/workflow.md) | ✅ | 🔴 Very high | 10 | 1 | 100 | 0 | 100% |
| **Total** | | | | | **1804** | **14** | |

**Status**
- ✅ No open bugs.
- 🟡 Open bugs, but none changes results on a normal run: edge cases, latent code paths, or features that are silently inactive.
- 🔴 At least one open bug gives **silently wrong results** on a normal configuration.
- ⏸️ Reviewed and deferred: every open bug (or pending cleanup) has been checked and accepted for now. Open bugs stay as strict xfails, so a later fix is still flagged.

**Coverage** is line coverage of each package from the whole suite. Because other packages are mocked, it is essentially each package's own tests.

## Coupling

"Depends on" counts the other cavachon packages a package imports (fan-out). It decides how much has to be mocked to test the package in isolation. "Used by" counts the packages that import it (fan-in). It decides how far a change or a bug can spread.

| Severity | Depends on | Meaning |
|---|---|---|
| 🟢 Standalone | 0 | Tested with no mocks at all. |
| 🟢 Low | 1 | One boundary to mock. |
| 🟡 Moderate | 2–3 | A few fakes; mostly pure computation. |
| 🟠 High | 4 | Needs fake Keras models/layers to run. |
| 🔴 Very high | 7+ | Orchestrators: most of the package is wiring between mocked collaborators. |

High fan-in packages are where a bug spreads furthest: environment (11), utils (10), layers (6), config and dataloader (4).

<details>
<summary>Dependency list</summary>

| Package | Depends on | Used by |
|---|---|---|
| app | workflow | — |
| callbacks | dataloader, distributions, environment, layers | scheduler |
| config | environment, utils | io, model, scheduler, workflow |
| dataloader | environment, layers, utils | callbacks, model, tools, workflow |
| distributions | — | callbacks, losses, tools |
| environment | — | callbacks, config, dataloader, layers, modality, model, modules, scheduler, tools, utils, workflow |
| filter | utils | workflow |
| io | config | workflow |
| layers | environment, utils | callbacks, dataloader, losses, model, modules, tools |
| losses | distributions, layers, utils | model, modules |
| modality | environment | workflow |
| model | config, dataloader, environment, layers, losses, modules, utils | tools, workflow |
| modules | environment, layers, losses, utils | model, tools |
| scheduler | callbacks, config, environment, utils | workflow |
| tools | dataloader, distributions, environment, layers, model, modules, utils | workflow |
| utils | environment | config, dataloader, filter, layers, losses, model, modules, scheduler, tools, workflow |
| workflow | config, dataloader, environment, filter, io, modality, model, scheduler, tools, utils | app |

</details>

## Needs attention

Open bugs that silently change results on a normal run:

| # | Package | Issue |
|---|---|---|
| [T1](packages/tools/cluster_attribution.md#t1) | tools | Integrated gradients in `AttributionAnalysis` carry an extra factor of alpha, so every attribution score is shrunk. |

## Layout

```
test_report/
├── UNIT_TEST_REPORT.md        this overview
├── guide/
│   ├── isolation.md           mocking policy
│   ├── conventions.md         xfail convention, bug IDs, commands, typing policy
│   └── history.md             scope, fix rounds, end-to-end checks
└── packages/
    ├── <package>.md           one page per cavachon package
    └── tools/                 one page per group of tools modules
```

Each package page has the same sections: status and coupling, mocked dependencies, **Open bugs**, **Fixed**, and **Worth a look** (suspicious behaviour not marked as a bug).
