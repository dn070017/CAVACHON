# `cavachon/tools`

[← Overview](../../UNIT_TEST_REPORT.md)

| Status | Tests | Passed | Open bugs | Coverage | Coupling |
|---|---|---|---|---|---|
| 🟡 Open bugs, edge cases | [`test/tools`](../../../test/tools) | 289 | 1 | 100% | Very high: depends on 7 packages, used by 1 |

- **Depends on:** dataloader, distributions, environment, layers, model, modules, utils
- **Used by:** workflow

| Modules | Report | Tests | Passed | Open bugs |
|---|---|---|---|---|
| `differential_analysis`, `hierarchical_differential_analysis`, `differential_attribution_analysis` | [differential.md](differential.md) | [`test/tools/differential`](../../../test/tools/differential) | 106 | 0 |
| `cluster_analysis`, `attribution_analysis` | [cluster_attribution.md](cluster_attribution.md) | [`test/tools/cluster_attribution`](../../../test/tools/cluster_attribution) | 57 | 1 |
| `interactive_visualization`, `enrichment_analysis` | [visualization_enrichment.md](visualization_enrichment.md) | [`test/tools/visualization_enrichment`](../../../test/tools/visualization_enrichment) | 126 | 0 |

## Mocking

- `DataLoader`, `ReflectionHandler`, the distribution classes, `MultivariateNormalDiagSampler`, `Model` and `Component` are patched where each tools module looks them up.
- Models are small deterministic fakes built from tf ops, so gradients flow. Other tools classes imported by `interactive_visualization` / `enrichment_analysis` are mocks.
- **gseapy is always mocked.** `Figure.write_html` / `write_image` / `show`, `Path.mkdir`, `TemporaryDirectory`, `copytree` and `pd.read_csv` are recorders, so there is no file or network I/O.
- TSNE and UMAP are replaced by a deterministic fake; PCA runs for real.

## Bugs affecting results

None. T13–T15, T17 and T18 are fixed; the open T16 is an edge case (see [cluster_attribution](cluster_attribution.md#t16)).

## Typing

`embedding(color=...)` is `Optional[str]` (a list never worked), and `differential_volcano_plot` raises a ValueError when `component`/`modality` are missing (it previously crashed later with KeyError). The remaining `type: ignore` marks the `across_clusters_pairwise` override (T5's 12 and T6's were removed with their fixes).
