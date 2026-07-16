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

## Tips

- Start from this template and only change paths and the `training:` block.
- If training is slow or runs out of memory, lower `batch_size` (default 512) or `max_regular_training_epochs`.
- For the full list of options, advanced validation rules, and per-component overrides, see [`CONFIG_REFERENCE.md`](./CONFIG_REFERENCE.md).
