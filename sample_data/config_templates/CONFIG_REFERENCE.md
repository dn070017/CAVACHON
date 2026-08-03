# Preparing Config File
The input for the CAVACHON model needs to be provided as a list of config files. An example of the minimum specification of the config files is decribed as follows:

```yaml
io:
  datadir: ${DATASET DIRECTORY}
modalities:
  - name: RNA Modality
    type: RNA
    h5ad: ${RNA H5AD File}
  - name: ATAC Modality
    type: ATAC
    h5ad: ${ATAC H5AD File}
model:
  name: CAVACHON
  components:
    - name: ATAC Component
      modalities:
        - name: ATAC
    - name: RNA Component
      conditioned_on_z_hat: 
        - ATAC Component
      modalities:  
        - name: RNA
```
Some other example templates can be found in `sample_data/config_templates`. To use the template, simply replace `${VARIABLE}` with the custom values.

&nbsp;
# Config Hierarchy
The config sould be prepared in a hierarchical structure using [YAML](https://en.wikipedia.org/wiki/YAML) format.
## Configuration Hierarchy
  * `io`: [Inputs and Outputs](#inputs-and-outputs).
  * `analysis`: [Analysis](#analysis)
    * `clustering`: list of [Clustering](#clustering)
    * `visualize_embedding`: list of [Visualize Embedding](#visualize-embedding)
    * `conditional_attribution_scores`: list of [Conditional Attribution Scores](#conditional-attribution-scores)
    * `differential_analysis`: list of [Differential Analysis](#differential-analysis)
    * `hierarchical_differential_analysis`: list of [Hierarchical Differential Analysis](#hierarchical-differential-analysis)
  * `modalities`: list of [Modalities](#modalities).
    * `filters`: list of [Filters](#filters).
  * `samples`: [Samples](#samples) (optional).
    * `modalities`: list of [Modality Files](#modality-files).
      * [Matrix](#matrix-file)
      * [Barcodes](#feature-file)
      * [Features](#feature-file)
  * `model`: [Model](#model)
    * `dataset`: [Dataset](#dataset)
    * `training`: [Training](#training)
      * `optimizer`: [Optimizer](#optimizer)
    * `components`: list of [Components](#components)
      * `modalities`: list of [Modalities (in Component)](#modalities-in-component).

&nbsp;
# Config Specification
## Inputs and Outputs
The configs for inputs and outputs are specified under the field `io`:
* `checkpointdir`:
  * required: `False`.
  * defaults: `./`
  * type: `str`
  * description: the directory for the pretrained checkpoint and the save model weights.
* `datadir`:
  * required: `False`.
  * defaults: `./`
  * type: `str`
  * description: the directory of the input datasets.
* `outdir`:
  * required: `False`.
  * defaults: `./`
  * type: `str`
  * description: the output directory.

[back to top](#config-hierarchy)
&nbsp;
## Analysis
The configs for analysis and visualization are specified under the field `analysis`:
* `clustering`:
  * required: `False`.
  * type: `List[AnalysisClusteringConfig]`
  * description: the config for clustering. See [Clustering](#clustering) for more details.
* `visualize_embedding`:
  * required: `False`.
  * type: `List[AnalysisVisualizeEmbeddingConfig]`
  * description: the config for embedding visualization. See [Visualize Embedding](#visualize-embedding) for more details.
* `differential_analysis`:
  * required: `False`.
  * type: `List[AnalysisDifferentialAnalysisConfig]`
  * description: the config for differential analysis. See [Differential Analysis](#differential-analysis) for more details.
* `conditional_attribution_scores`:
  * required: `False`.
  * defaults: `[]`
  * type: `List[AnalysisAttributionScoreConfig]`
  * description: the config for the attribution scores. See [Conditional Attribution Scores](#conditional-attribution-scores) for more details.
* `hierarchical_differential_analysis`:
  * required: `False`.
  * defaults: `[]`
  * type: `List[AnalysisHierarchicalDifferentialAnalysisConfig]`
  * description: the config for hierarchical differential analysis. See [Hierarchical Differential Analysis](#hierarchical-differential-analysis) for more details.

## Clustering
The config for clustering.
* `modality`:
  * required: `True`
  * type: `str`
  * description: which modality of the outputs of the component to used.
* `component`:
  * required: `True`
  * type: `str`
  * description: the outputs of which component to used.
* `use_rep`:
  * required: `False`
  * defaults: `'z'`
  * type: `str`
  * description: which representation to use for clustering. Must be one of `'z'` or `'z_hat'`.
* `min_n_obs`:
  * required: `False`
  * defaults: `36`
  * type: `int`
  * description: minimum number of observations required for a cluster to be kept. Clusters smaller than this are labeled as `Unassigned` (for `z_hat`) or iteratively removed (for `z`).

[back to top](#config-hierarchy)
&nbsp;

## Visualize Embedding
* `modality`:
  * required: `True`
  * type: `str`
  * description: which modality to visualize.
* `use_rep`:
  * required: `True`
  * type: `str`
  * description: which representation of the modality to visualize.
* `embedding_method`:
  * required: `True`
  * type: `str`
  * description: method used to embed the representation of the modality. Should be one of `'pca'`, `'umap'` or `'tsne'`.
* `color_by`
  * required: `True`
  * type: `str`
  * description: color by which annotation column.
* `interactive`
  * required: `False`
  * type: `bool`
  * description:  whether or not to create interactive visualization. Defaults to False.
  
[back to top](#config-hierarchy)
&nbsp;

## Differential Analysis
The config for differential analysis.

Two modes are supported based on whether `group_a` and `group_b` are provided:

- **Pairwise mode** (default): `group_a` and `group_b` are both empty. Compares every pair of clusters automatically.
- **Single-pair mode**: `group_a` and `group_b` are both specified. Compares only those two groups.

* `modality`:
  * required: `True`
  * type: `str`
  * description: which modality of the outputs of the component to used.
* `component`:
  * required: `True`
  * type: `str`
  * description: the outputs of which component to used.
* `use_cluster`:
  * required: `False`.
  * defaults: `"cluster_{component}"`
  * type: `str`
  * description: cluster column used to define groups. Defaults to the clustering result of the same component.
* `group_a`:
  * required: `False`.
  * defaults: `""`
  * type: `str`
  * description: cluster label of the first group. When set together with `group_b`, runs single-pair mode. Leave empty for pairwise mode.
* `group_b`:
  * required: `False`.
  * defaults: `""`
  * type: `str`
  * description: cluster label of the second group. When set together with `group_a`, runs single-pair mode. Leave empty for pairwise mode.
* `z_sampling_size`:
  * required: `False`.
  * defaults: `5`
  * type: `int`
  * description: number of latent samples to draw per cell for the DEG test.
* `x_sampling_size`:
  * required: `False`.
  * defaults: `1000`
  * type: `int`
  * description: number of decoded samples are drawn per latent sample.
* `batch_size`:
  * required: `False`.
  * defaults: `128`
  * type: `int`
  * description: batch size used during sampling.
* `keep_only_significant`:
  * required: `False`.
  * defaults: `False`
  * type: `bool`
  * description: whether to filter the output to significant results only.
* `sort_output`:
  * required: `False`.
  * defaults: `True`
  * type: `bool`
  * description: sort results by the maximum of the absolute Bayesian factors `K(A>B|Z)` and `K(B>A|Z)` in descending order.

[back to top](#config-hierarchy)
&nbsp;
## Hierarchical Differential Analysis
The config for hierarchical differential analysis.

Two modes are supported based on whether `donor_cluster` and `recipient_cluster` are provided:

- **Pairwise mode** (default): `donor_cluster` and `recipient_cluster` are both empty. Runs interventions between every pair of clusters automatically.
- **Single-pair mode**: `donor_cluster` and `recipient_cluster` are both specified. Runs the intervention for that one pair only.

* `modality`:
  * required: `True`
  * type: `str`
  * description: which modality of the outputs of the component to used.
* `component`:
  * required: `True`
  * type: `str`
  * description: the outputs of which component to used.
* `use_cluster`:
  * required: `False`.
  * defaults: `""`
  * type: `str`
  * description: cluster column used to define donor and recipient groups.
* `donor_cluster`:
  * required: `False`.
  * defaults: `""`
  * type: `str`
  * description: cluster label of the donor group. When set together with `recipient_cluster`, runs single-pair mode. Leave empty for pairwise mode.
* `recipient_cluster`:
  * required: `False`.
  * defaults: `""`
  * type: `str`
  * description: cluster label of the recipient group. When set together with `donor_cluster`, runs single-pair mode. Leave empty for pairwise mode.
* `donor_components`:
  * required: `False`.
  * defaults: `None`
  * type: `List[str] | None`
  * description: parent components whose latent values are substituted to control for hierarchy. Defaults to all `conditioned_on_z_hat` parents.
* `n_samples`:
  * required: `False`.
  * defaults: `10`
  * type: `int`
  * description: number of substituted samples per cell.
* `seed`:
  * required: `False`.
  * defaults: `None`
  * type: `int | None`
  * description: random seed for reproducibility.
* `batch_size`:
  * required: `False`.
  * defaults: `128`
  * type: `int`
  * description: batch size used during sampling.
* `sort_output`:
  * required: `False`.
  * defaults: `True`
  * type: `bool`
  * description: sort results by the maximum of the absolute Bayesian factors `K(A>B|Z)` and `K(B>A|Z)` in descending order.
* `keep_only_significant`:
  * required: `False`.
  * defaults: `False`
  * type: `bool`
  * description: whether to filter the output to significant results only.

[back to top](#config-hierarchy)
&nbsp;
## Hierarchical Differential Analysis
The config for hierarchical differential analysis.
* `modality`:
  * required: `True`
  * type: `str`
  * description: which modality of the outputs of the component to used.
* `component`:
  * required: `True`
  * type: `str`
  * description: the outputs of which component to used.
* `use_cluster`:
  * required: `False`.
  * defaults: `""`
  * type: `str`
  * description: cluster column used to define donor and recipient groups.
* `donor_cluster`:
  * required: `True`
  * type: `str`
  * description: cluster label of the donor group.
* `recipient_cluster`:
  * required: `True`
  * type: `str`
  * description: cluster label of the recipient group.
* `donor_components`:
  * required: `False`.
  * defaults: `None`
  * type: `List[str] | None`
  * description: parent components whose latent values are substituted to control for hierarchy. Defaults to all `conditioned_on_z_hat` parents.
* `n_samples`:
  * required: `False`.
  * defaults: `10`
  * type: `int`
  * description: number of substituted samples per cell.
* `seed`:
  * required: `False`.
  * defaults: `None`
  * type: `int | None`
  * description: random seed for reproducibility.
* `batch_size`:
  * required: `False`.
  * defaults: `128`
  * type: `int`
  * description: batch size used during sampling.

[back to top](#config-hierarchy)
&nbsp;
## Conditional Attribution Scores
The config for conditional attribution scores
* `modality`:
  * required: `True`
  * type: `str`
  * description: which modality of the outputs of the component to used.
* `component`:
  * required: `True`
  * type: `str`
  * description: the outputs of which component to used.
* `with_respect_to`:
  * required: `True`
  * type: `list(str)`
  * description: compute integrated gradient with respect to the latent representation of which component.
* `use_cluster`:
  * required: `False`
  * defaults: `""`
  * type: `str`
  * description: optional cluster column to group cells before computing attribution scores.

[back to top](#config-hierarchy)
&nbsp;
## Modalities
The configs for modalities (or data views) are specified under the field `modalities`. This is used to specified the data distribution and type of the modalities. See also [Filters](#filters).
* `name`:
  * required: `False`.
  * defaults: `modality/{i:02d}`
  * type: `str`
  * description: the name of the modality.
* `type`:
  * required: `True`.
  * type: `str`
  * description: the type of the modality. Currently supports `'atac'` and `'rna'`.
* `dist`:
  * required: `False`.
  * type: `str`
  * defaults:
    1. `'IndependentBernoulli'` for `type:atac`.
    2. `'IndependentZeroInflatedNegativeBinomial'` for `type:rna`.
  * description: the data distribution of the modality. Currently supports `'IndependentBernoulli'` and `'IndependentZeroInflatedNegativeBinomial'` (see `cavachon/distributions` for more details).
* `h5ad`:
  * required: `False`.
  * type: `str`.
  * description: the `h5ad` file name corresponding to the modality in directory `io/datadir` (see [Inputs and Outputs](#inputs-and-outputs)). Alternatively, the data can be loaded with `mtx`, `features` and `barcodes` files specified in [Samples](#samples). Note that `samples` configs will be ignored for the modality if provided with `h5ad`.
* `samples`:
  * required: `False`.
  * type: `List[str]`
  * defaults: `List[]`
  * description: list of sample names that provide data for this modality. Used when loading from `mtx`/`features`/`barcodes` files in [Samples](#samples).
* `filters`:
  * required: `False`.
  * type: `List[FilterConfig]`
  * defaults: `List[]`
  * description: see [Filters](#filters) and `cavachon/config/models/filter_config.py` for more details.
* `batch_effect_colnames`:
  * required: `False`
  * type: `List[str]`
  * defaults: `List[]`
  * description: the column names of the batch effects that needs to be corrected.

[back to top](#config-hierarchy)
## Filters
The filter applied to each modality. Should be provided as a list of `FilterConfig` specification. The filtering steps will be executed sequentially based on the provided order in the list. The config should be put under `modalities -> [one of the modality config] -> filters`. The FilterConfig specification is described as follows: 
* `step`:
  * required: `True`
  * type: `str`
  * description: type of the filtering steps. Currently supports `FilterCells`, `FilterGenes`, and `FilterQC`. (see `cavachon/filter/` for more details)
* `**kwargs`:
  * description: please replace `kwargs` with the arguments passed to `scanpy.pp.filter_cells` (for `FilterCells`), `scanpy.pp.filter_genes` (for `FilterGenes`). For `FilterQC`, please see the following example.
### Filters Examples
```yaml
modalities:
  - name: ${Modality}
  ...
    filters:
      - step: FilterQC 
        qc_vars:
          - ERCC
          - MT
        filter_threshold:
          - field: n_genes_by_counts
            operator: ge
            threshold: 500
          - field: pct_counts_ERCC
            operator: le
            threshold: 0.2
          - field: pct_counts_MT
            operator: le
            threshold: 0.2
      - step: FilterGenes
        min_counts: 25
      - step: FilterGenes
        min_cells: 10
      - step: FilterCells
        min_counts: 5  
```
[back to top](#config-hierarchy)
&nbsp;
## Samples
The configs for the samples (or experiments) files are specified under the field `samples`. Note that one samples can have multiple modalities (e.g. from single-cell multi-omics technology), the files of every samples will be merged into multiple modalities. See also [Modality Files](#modality-files), [Matrix File](#matrix-file) and [Feature File](#feature-file).
* `name`:
  * required: `False`.
  * defaults: `sample/{i:02d}`
  * type: `str`
  * description: the name of the sample
* `modalities`:
  * required: `True`.
  * type: `List[ModalityFileConfig]`.
  * description: see [Modality Files](#modality-files) for more details.

[back to top](#config-hierarchy)
## Modality Files
The configs for the files of a modality from **one sample**. The config should be put under `samples -> modalities`.
* `name`:
  * required: `True`.
  * type: `str`
  * description: the name of the corresponding modality. Must match one of the name specified in [Modalities](#modalities).
* `matrix`:
  * required: `True`.
  * type: `ModalityFileMatrixConfig`.
  * description: the config of matrix file corresponding to the modality in directory io/datadir (see [Matrix File](#matrix-file) and [Inputs and Outputs](#inputs-and-outputs)).
* `barcodes`:
  * required: `True`.
  * type: `ModalityFileFeatureConfig`.
  * description: the config of barcodes file (for the anchor indices) corresponding to the modality in directory io/datadir (see [Feauture File](#feature-file) and [Inputs and Outputs](#inputs-and-outputs)).
* `features`:
  * required: `True`.
  * type: `ModalityFileFeatureConfig`.
  * description: the config of features file (e.g. gene annotations) corresponding to the modality in directory io/datadir (see [Feauture File](#feature-file) and [Inputs and Outputs](#inputs-and-outputs)).

[back to top](#config-hierarchy)
## Matrix File
The configs for the matrix file. Should be put under `samples -> modalities -> matrix`.
* `filename`:
  * required: `True`.
  * type: `str`.
  * description: the matrix file corresponding to the modality in directory `io -> datadir` (see [Inputs and Outputs](#inputs-and-outputs)).
* `transpose`:
  * required: `False`.
  * defaults: `False`
  * type: `bool`.
  * description: if the matrix is transposed (the matrix is transposed if vars as rows, obs as cols).

[back to top](#config-hierarchy)
## Feature File
The configs for the matrix file. Should be put under `samples -> modalities -> features` and `samples -> modalities -> barcodes`.
* `filename`:
  * required: `True`.
  * type: `str`.
  * description: the features file (e.g. gene annotations) or the barcodes file (for the anchor indices) corresponding to the modality in directory `io -> datadir` (see [Inputs and Outputs](#inputs-and-outputs)).
* `has_headers`:
  * required: `False`
  * defaults: `False`
  * type: `bool`
  * description: whether or not the `features` or `barcodes` file have headers.
* `colnames`:
  * required: `False`
  * type: `List[str]`.
  * description: the column names of the `features` or `barcodes` files (if `has_headers=False`)

[back to top](#config-hierarchy)
&nbsp;
## Model
The configs for the model are specified under the field `model`. See also [Components](#components), [Modalities (in Component)](#modalities-in-component), [Training](#training), [Optimizer](#optimizer) and [Dataset](#dataset).
* `name`:
  * required: `False`.
  * defaults: `CAVACHON`
  * type: `str`
  * description: the name of the model.
* `load_weights`:
  * required: `True`.
  * type: `bool`
  * description: whether or not to load the pretrained weights. If `True`, the checkpoint of the pretrained in `checkpoiontdir/model_name` will be load to the Model. See [config for IO](#inputs-and-outputs).
* `save_weights`:
  * required: `False`.
  * type: `bool`
  * description: whether or not to save the weights. If `True`, the weights will be save to `checkpoiontdir/model_name`. See [config for IO](#inputs-and-outputs).
* `components`:
  * required: `True`.
  * type: `List[ComponentConfig]`
  * description: see [Components](#components) and `cavachon/config/models/component_config.py` for more details.
* `training`:
  * required: `False`.
  * type: `TrainingConfig`
  * description: see [Training](#training) and `cavachon/config/models/training_config.py` for more details.
* `dataset`:
  * required: `False`.
  * type: `DatasetConfig`
  * description: see [Dataset](#dataset) and `cavachon/config/models/dataset_config.py` for more details.

[back to top](#config-hierarchy)
## Components
The configs for the components in the model. See also [Modalities (in Component)](#modalities-in-component).
* `name`:
  * required: `False`.
  * defaults: `component/{i:02d}`
  * type: `str`
  * description: the name of the component.
* `n_encoder_layers`:
  * required: `False`.
  * defaults: `3`
  * type: `int`.
  * description: the number of hidden layers used in the encoder neural network.
* `n_latent_dims`:
  * required: `False`.
  * defaults: `5`.
  * type: `int`.
  * description: the dimensionality of the latent space.
* `n_latent_priors`:
  * required: `False`.
  * defaults: `n_latent_dims * 2 + 1`
  * type: `int`.
  * description: the number of components of Gaussian-mixture priors used to compute KL-divergence and perform online clustering.
* `n_parent_annealing_epochs`:
  * required: `False`.
  * defaults: uses `training.n_parent_annealing_epochs` (default `1`) if not set here.
  * type: `int`.
  * description: number of parent annealing epochs used during the training process. During the parent annealing phase, the weight of the child's data likelihood is scaled quadratically from 0 to 1 with `(epoch/n_parent_annealing_epochs)²` while the parent's data weight fades from 1.0 to 0.0.
* `n_kl_annealing_epochs`:
  * required: `False`.
  * defaults: uses `training.n_kl_annealing_epochs` (default `25`) if not set here.
  * type: `int`.
  * description: number of epochs for the standalone KL annealing phase (standard_kl → GMM crossfade) for this component. When > 0 the KL annealing phase runs; when 0 it is skipped.
* `enable_kmeans_init`:
  * required: `False`.
  * defaults: uses `training.enable_kmeans_init` (default `True`) if not set here.
  * type: `bool`.
  * description: whether to run k-means initialization for the GMM priors of this component before GMM training.
* `learn_z_hat_priors`:
  * required: `False`.
  * defaults: uses `training.learn_z_hat_priors` (default `False`) if not set here.
  * type: `bool`.
  * description: whether to learn a direct diagonal GMM density over this component's deterministic `z_hat`. When enabled, a GMM with `n_latent_priors` components is trained on `z_hat`. The loss is a direct density (`-log p_GMM(z_hat)`); there is no sampler, posterior, or KL term. Overrides the training-level default. The loss weight is 0.0 during parent annealing, ramps with the GMM KL crossfade during KL annealing, and stays at 1.0 during regular GMM training. When disabled (default), `z_hat` clustering falls back to the post-hoc analytical `compute_integrated_cluster_log_probability` method.
* `kl_annealing_ratio`:
  * required: `False`.
  * defaults: uses `training.kl_annealing_ratio` (default `[0.5, 0.2, 0.3]`) if not set here.
  * type: `List[float]`.
  * description: ratios for the three sub-phases within KL annealing (standard_kl only, crossfade, gmm only). Ignored when `n_kl_annealing_epochs` is 0.
* `max_regular_training_epochs`:
  * required: `False`.
  * defaults: uses `training.max_regular_training_epochs` (default `500`) if not set here.
  * type: `int`.
  * description: maximum number of regular GMM training epochs for this component.
* `reparameterize_z_hat`:
  * required: `False`.
  * defaults: `True`.
  * type: `bool`.
  * description: whether `z_hat` is treated as a distribution that can be inferred analytically. When `True`, the hierarchical encoder uses `use_bias=False` and `z_hat` clustering is allowed.
* `conditioned_on_z`:
  * required: `False`.
  * defaults: `List[]`
  * type: `List[str]`
  * description: the provided string in the list needs to be the name that matched to one of the specified [Components](#components). The current component will be conditionally independent with the specified components on the latent representation of the later one (**exclude its ancestors**). Note that the conditional independent relationships between components needs to be a **directed acyclic graph**.
* `conditioned_on_z_hat`:
  * required: `False`.
  * defaults: `List[]`
  * type: `List[str]`
  * description: the provided string in the list needs to be the name that matched to one of the specified [Components](#components). The current component will be conditionally independent with the specified components on the latent representation of the later one (**include its ancestors**). Note that the conditional independent relationships between components needs to be a **directed acyclic graph**.
* `modalities`:
  * required: `True`.
  * type: `List[Config]`
  * description: see [Modalities (in Component)](#modalities-in-component)

[back to top](#config-hierarchy)
## Modalities (in Component)
The configs for the modalities in the component.
* `name`:
  * required: `True`.
  * type: `str`.
  * description: the name of the corresponding modality. Must match one of the name specified in [Modalities](#modalities).
* `n_decoder_layers`:
  * required: `False`.
  * defaults: `3`
  * type: `int`.
  * description: the number of hidden layers used in the decoder neural network.
* `save_z`:
  * required: `False`.
  * defaults: `True`.
  * type: `bool`.
  * description: whether or not to save the predicted `z` and `z_hat` to `obsm` of the modality.
* `save_x`:
  * required: `False`.
  * defaults: `True`.
  * type: `bool`.
  * description: whether or not to save the predicted `x_parameters` to `obsm` of the modality. Note that `x_parameters` will not be predicted by defaults if none of the modalities in the component set `save_x`.

[back to top](#config-hierarchy)
## Training
The configs for the training process. See also [Optimizer](#optimizer).
* `train`:
  * required: `False`.
  * defaults: `True`.
  * type: `bool`.
  * description: whether or not to train or finetune the model.
* `early_stopping`:
  * required: `False`.
  * defaults: `True`.
  * type: `bool | EarlyStoppingConfig`
  * description: whether or not to use early stopping when training the model. Ignored if `train=False`. Can be a boolean or a dict with `monitor`, `mode`, and `patience` (e.g. `{monitor: loss, mode: min, patience: 25}`).
* `max_regular_training_epochs`:
  * required: `False`.
  * defaults: `500`.
  * type: `int`.
  * description: default maximum number of regular GMM training epochs. Can be overridden per component with `components[].max_regular_training_epochs`.
* `n_parent_annealing_epochs`:
  * required: `False`.
  * defaults: `1`.
  * type: `int`.
  * description: default number of parent annealing epochs. Can be overridden per component with `components[].n_parent_annealing_epochs`.
* `n_kl_annealing_epochs`:
  * required: `False`.
  * defaults: `25`.
  * type: `int`.
  * description: default number of KL annealing epochs. Can be overridden per component with `components[].n_kl_annealing_epochs`.
* `enable_kmeans_init`:
  * required: `False`.
  * defaults: `True`.
  * type: `bool`.
  * description: default flag for k-means initialization. Can be overridden per component with `components[].enable_kmeans_init`.
* `learn_z_hat_priors`:
  * required: `False`.
  * defaults: `False`.
  * type: `bool`.
  * description: default flag for learning a direct GMM density over the deterministic `z_hat` representation. When enabled, a diagonal GMM with `n_latent_priors` components is trained on the observed `z_hat` values. The density loss weight is 0.0 during parent annealing, ramps with the GMM KL crossfade during KL annealing, and stays at 1.0 during regular GMM training. This is a direct density loss (`-log p_GMM(z_hat)`), not a z_hat sampler or KL term. The learned prior is used for scoring during `z_hat` analysis when enabled; otherwise the legacy post-hoc analytical `compute_integrated_cluster_log_probability` method is used. Can be overridden per component with `components[].learn_z_hat_priors`.
* `kl_annealing_ratio`:
  * required: `False`.
  * defaults: `[0.5, 0.2, 0.3]`.
  * type: `List[float]`.
  * description: default KL annealing sub-phase ratios. Can be overridden per component with `components[].kl_annealing_ratio`.
* `optimizer`:
  * required: `False`.
  * defaults: `OptimizerConfig({'name': 'adam', 'learning_rate': 1e-4})`
  * type: `OptimizerConfig`
  * description: see [Optimizer](#optimizer) and `cavachon/config/OptimizerConfig` for more details.

[back to top](#config-hierarchy)
## Optimizer
The configs for the optimizer.
* `name`:
  * required: `False`.
  * defaults: `adam`
  * type: `str`
  * description: string representation for the Tensorflow Keras optimizer. See [tf.keras.optimizers](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers) for more details.
* `learning_rate`:
  * required: `False`.
  * defaults: `1e-4`
  * type: `float`
  * description: learning rate for the specified optimizers .

[back to top](#config-hierarchy)
## Dataset
The configs for the dataset.
* `batch_size`:
  * required: `False`.
  * defaults: `128`.
  * type: `int`
  * description: batch size used to train and evaluate the model. The higher the value, the more efficient the training process will be but more memory will be used.
* `shuffle`:
  * required: `False`.
  * defaults: `False`.
  * type: `bool`
  * description: whether or not to shuffle the dataset during training.

[back to top](#config-hierarchy)