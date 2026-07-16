# Quick-start config template

This folder contains a ready-to-use CAVACHON config: [`config.yaml`](./config.yaml).

For a complete field-by-field reference, see [`CONFIG_REFERENCE.md`](./CONFIG_REFERENCE.md).

## What the template does

The template integrates two modalities from the same cells:

- **ATAC** (chromatin accessibility)
- **RNA** (gene expression), conditioned on the ATAC component

## How to use it

1. Open [`config.yaml`](./config.yaml).
2. Replace the `${...}` placeholders with your own paths:
   - `${DATASET DIRECTORY}`: folder containing your `.h5ad` files
   - `${RNA H5AD File}` and `${ATAC H5AD File}`: file names of the RNA and ATAC data
   - `${CHECKPOINT DIRECTORY}`: where pretrained/saved model weights go
   - `${OUTPUT DIRECTORY}`: where results are written
3. Run CAVACHON:
   ```python
   from cavachon.workflow import Workflow
   workflow = Workflow("sample_data/config_templates/config.yaml")
   workflow.run()
   ```

## Training settings

All training hyperparameters are set at the `training:` level, so you only need to tune them in one place:

| Setting | Default | What it controls |
|---|---|---|
| `max_regular_training_epochs` | 1000 | How long each component trains |
| `n_parent_annealing_epochs` | 1 | Epochs to blend child component in gradually |
| `n_kl_annealing_epochs` | 25 | Epochs of KL-to-GMM annealing |
| `enable_kmeans_init` | true | Initialize GMM priors with k-means |
| `kl_annealing_ratio` | [0.5, 0.2, 0.3] | Sub-phase ratios within KL annealing |

If you need different values for different components, you can override any of these inside a component block. See [`CONFIG_REFERENCE.md`](./CONFIG_REFERENCE.md) for details.

## Modality settings

For each modality you only need:

- `name`: a short name (e.g. `RNA`, `ATAC`)
- `type`: `RNA` or `ATAC`
- `h5ad`: the file name

The data distribution is chosen automatically from the type.

## Component structure

Components define how modalities are linked:

```yaml
components:
  - name: ATAC
    modalities:
      - name: ATAC
  - name: RNA
    conditioned_on_z_hat:
      - ATAC
    modalities:
      - name: RNA
```

Here RNA is modeled conditional on ATAC. You can add more components or change the hierarchy; just make sure the graph stays acyclic.

## Downstream analyses

The template also includes an `analysis:` block with common post-training analyses:

- **Clustering**: cluster cells in `z` or `z_hat` space (set `use_rep: z` or `use_rep: z_hat`).
- **Visualize embedding**: make t-SNE/UMAP plots colored by annotations or clusters.
- **Differential analysis**: find marker genes between clusters. By default runs all pairwise comparisons; set `group_a` and `group_b` to compare a single pair.
- **Conditional attribution scores**: see which parent latent dimensions drive a child's predictions.
- **Hierarchical differential analysis**: compare gene expression across integrated clusters while controlling for donor components. By default runs all pairwise interventions; set `donor_cluster` and `recipient_cluster` to run a single pair.

Each entry is optional — remove the ones you do not need. The default values are sensible for a first run.

## Cluster-specific differential analysis

Both DEG and HDEG support two modes: **pairwise** (default, all cluster pairs) and **single-pair** (two specific clusters). The mode is chosen by whether you provide explicit cluster labels.

### DEG: differential expression between clusters

DEG compares gene expression between groups of cells defined by a cluster column. It uses Bayesian factors (`K(A>B|Z)`, `K(B>A|Z)`) to quantify differential expression.

**Pairwise mode** (default) — compares every pair of clusters automatically:

```yaml
differential_analysis:
  - modality: RNA
    component: RNA
    use_cluster: cluster_RNA
```

**Single-pair mode** — compare only two specific groups:

```yaml
differential_analysis:
  - modality: RNA
    component: RNA
    use_cluster: cell_type
    group_a: "CD4 Naive"
    group_b: "CD8 Naive"
```

| Field | Description |
|---|---|
| `use_cluster` | Column in `.obs` that defines cluster labels. Defaults to `cluster_{component}`. |
| `group_a` / `group_b` | Cluster labels for single-pair mode. Leave empty for pairwise. |
| `keep_only_significant` | Filter to rows where `|K| >= 3.2` (default `false`). |
| `sort_output` | Sort by `max(|K(A>B|Z)|, |K(B>A|Z)|)` descending (default `true`). |

### HDEG: hierarchical differential analysis between clusters

HDEG goes beyond DEG by performing **interventions**: it substitutes the latent values of donor components from one cluster into another, then measures how gene expression changes. This isolates the effect of parent components (e.g. ATAC) on child expression (e.g. RNA).

Two component concepts appear in the config:

- **`component`** — the component that generates the modality being analyzed (e.g. `RNA`).
- **`donor_components`** — the parent components whose latent values are substituted (e.g. `ATAC`). Defaults to all `conditioned_on_z_hat` parents if omitted.

**Pairwise mode** (default) — runs interventions between every cluster pair, in both directions:

```yaml
hierarchical_differential_analysis:
  - modality: RNA
    component: RNA
    use_cluster: cell_type
    donor_components:
      - ATAC
```

**Single-pair mode** — run the intervention for one specific pair only:

```yaml
hierarchical_differential_analysis:
  - modality: RNA
    component: RNA
    use_cluster: cell_type
    donor_cluster: "CD4 Naive"
    recipient_cluster: "CD8 Naive"
    donor_components:
      - ATAC
```

| Field | Description |
|---|---|
| `use_cluster` | Column in `.obs` that defines cluster labels. |
| `donor_cluster` / `recipient_cluster` | Cluster labels for single-pair mode. Leave empty for pairwise. |
| `donor_components` | Parent components to substitute. Defaults to all parents if omitted. |
| `n_samples` | Number of substituted samples per cell (default `10`). |
| `sort_output` | Sort by `max(|K(A>B|Z)|, |K(B>A|Z)|)` descending (default `true`). |

### Output files

DEG writes one TSV per cluster pair to `differential_analysis/`:

```
rna_cluster_rna_cd4_naive_cd8_naive.tsv
```

HDEG writes one TSV per pair (and direction) to `hierarchical_differential_analysis/`. Filenames distinguish the modality-generating component from the substituted donor components:

```
rna_from_rna_substitute_atac_cd4_naive_to_cd8_naive.tsv
```

When `donor_components` is omitted, the filename uses `all` in place of the component list.

## Tips

- Start from this template and only change paths and the `training:` block.
- If training is slow or runs out of memory, lower `batch_size` (default 512) or `max_regular_training_epochs`.
- For the full list of options, advanced validation rules, and per-component overrides, see [`CONFIG_REFERENCE.md`](./CONFIG_REFERENCE.md).
