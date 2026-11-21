# `cavachon/tools`

[← Overview](../../UNIT_TEST_REPORT.md)

| Status | Tests | Passed | Open bugs | Coverage | Coupling |
|---|---|---|---|---|---|
| 🔴 Open bugs affecting results | [`test/tools`](../../../test/tools) | 226 | 14 | 100% | Very high: depends on 7 packages, used by 1 |

- **Depends on:** dataloader, distributions, environment, layers, model, modules, utils
- **Used by:** workflow

| Modules | Report | Tests | Passed | Open bugs |
|---|---|---|---|---|
| `differential_analysis`, `hierarchical_differential_analysis`, `differential_attribution_analysis` | [differential.md](differential.md) | [`test/tools/differential`](../../../test/tools/differential) | 77 | 2 |
| `cluster_analysis`, `attribution_analysis` | [cluster_attribution.md](cluster_attribution.md) | [`test/tools/cluster_attribution`](../../../test/tools/cluster_attribution) | 41 | 6 |
| `interactive_visualization`, `enrichment_analysis` | [visualization_enrichment.md](visualization_enrichment.md) | [`test/tools/visualization_enrichment`](../../../test/tools/visualization_enrichment) | 108 | 6 |

## Mocking

- `DataLoader`, `ReflectionHandler`, the distribution classes, `MultivariateNormalDiagSampler`, `Model` and `Component` are patched where each tools module looks them up.
- Models are small deterministic fakes built from tf ops, so gradients flow. Other tools classes imported by `interactive_visualization` / `enrichment_analysis` are mocks.
- **gseapy is always mocked.** `Figure.write_html` / `write_image` / `show`, `Path.mkdir`, `TemporaryDirectory`, `copytree` and `pd.read_csv` are recorders, so there is no file or network I/O.
- TSNE and UMAP are replaced by a deterministic fake; PCA runs for real.

## Bugs affecting results

| # | Module | Issue |
|---|---|---|
| [T1](cluster_attribution.md#t1) | attribution_analysis | Integrated gradients carry an extra factor of alpha; every attribution is shrunk. |
| [T2](cluster_attribution.md#t2) | cluster_analysis | Child z_hat cluster variance drops cross terms. |
| [T3](cluster_attribution.md#t3) | cluster_analysis | Cluster labels can stop matching their `obsm` columns. |
| [T4](differential.md#t4) | differential_analysis | Unchanged features count as "significant". |
| [T9](visualization_enrichment.md#t9) | interactive_visualization | `attribution_score` overwrites its own figures. |
| [T12](visualization_enrichment.md#t12) | enrichment_analysis | The term threshold is applied to the DEG ranking (needs the author's call). |

## Typing

`embedding(color=...)` is `Optional[str]` (a list never worked), and `differential_volcano_plot` raises a ValueError when `component`/`modality` are missing (it previously crashed later with KeyError). The remaining `type: ignore`s mark T5 (12 lines), T6 and the `across_clusters_pairwise` override.
