# Unit Test Report — low-coupling packages

Date: 2026-11-13 · Branch: `feature/attribution_analysis`

## Scope

The old tests were removed: `test/` (losses, scheduler, utils) and the root-level `test_differential_analysis.py`, `test_enrichment_analysis.py`, `test_hierarchical_differential_analysis.py` and `test_model_encode_decode.py`. The tracked files are recoverable from git. The untracked `test/losses/` was backed up outside the repo.

New tests cover every package that imports from fewer than 4 other cavachon packages:

| Packages tested | Out-deps | Test dirs | Passed | xfail (documented bugs) |
|---|---|---|---|---|
| distributions + losses | 0 / 3 | `test/distributions`, `test/losses` | 76 | 11 |
| layers | 2 | `test/layers` | 112 | 15 |
| utils + environment | 1 / 0 | `test/utils`, `test/environment` | 187 | 15 |
| config + io | 2 / 1 | `test/config`, `test/io` | 161 | 22 |
| dataloader + modality | 3 / 1 | `test/dataloader`, `test/modality` | 104 | 12 |
| filter + app | 1 / 1 | `test/filter`, `test/app` | 105 | 5 |
| **Total** | | | **754** | **74** |

The per-group counts above are from before the H1–H6 fixes. After the fixes, 6 xfails became regular passing tests and 3 regression tests were added.

### Isolation (mocking policy)

Every test is a unit test, isolated at the **package boundary**:

- **Other cavachon packages are mocked.** Anything the code under test imports from another `cavachon.<pkg>` is replaced with `MagicMock`s or fakes, using `monkeypatch`. The patch is applied where the name is looked up, for example `cavachon.losses.negative_log_data_likelihood.ReflectionHandler.get_class_by_name`. Tests assert the arguments passed to these mocks.
  - Exception: `cavachon.environment.Constants` stays real, because it only holds plain values.
  - Code inside the package under test stays real. For example, DataLoader runs with its own `dataloader/modifiers`.
- **No real file I/O, temporary files or subprocesses.** These calls are mocked, and the tests assert the paths and arguments they receive:
  - `open`/`yaml.load`, `pd.read_csv`, `mmread`, `mu.read`
  - `write_h5ad`/`read_h5ad`, `tf.data.experimental.save/load`
  - `glob`/`import_module` (for ReflectionHandler), `os.path.realpath`/`isdir`
- **Third-party libraries stay real**: TensorFlow, tfp, numpy, scipy, pandas, anndata, muon, scanpy, pydantic and Keras. The code computes with them, and they serve as numeric references.
- Shared mocks live in each directory's `conftest.py`.

When the code was isolated, a few bugs turned out to belong to a different package than the one whose tests first exposed them. Each was moved to the tests of the package where it lives:

| Bug | Moved from | Moved to |
|---|---|---|
| `tf.Variable` input crash | losses | distributions |
| Dense-X `.tocoo()` crash | dataloader | utils (`test_tensor_utils.py`) |
| Duplicate sparse entries crash | dataloader | utils (`test_tensor_utils.py`) |
| FilterConfig extra-field warning | filter | config |

Isolation also exposed three new bugs, listed below.

These packages were not covered because they import from 4 or more other cavachon packages: `callbacks`, `modules`, `scheduler`, `model`, `tools` and `workflow`.

To run the suite (CPU, about 15s):

```bash
.pixi/envs/default/bin/python -m pytest
```

`pyproject.toml` now sets `testpaths = ["test"]`. Without it, pytest would also collect `test_run_qa_hdeg.py`, a script at the repo root that starts a full workflow.

### How bugs are recorded

Each confirmed bug has a test that asserts the **correct** behaviour and is marked `@pytest.mark.xfail(strict=True, reason="BUG: ...")`. The suite stays green while the bug exists. When a bug is fixed, its test starts passing, which counts as an XPASS failure, and that prompts you to remove the marker. To list them all:

```bash
grep -rn "xfail" test | grep BUG
```

To see the actual failures:

```bash
pytest --runxfail
```

---

## Confirmed issues, by severity

### High: silent wrong results or crashes on normal input. ✅ All fixed (2026-11-13)

| # | Location | Issue | Fix |
|---|---|---|---|
| H1 | `losses/negative_log_data_likelihood.py:76` | `reduce_sum(log_prob(y), axis=-1)` assumes an independent per-feature `log_prob`. For `MultivariateNormalDiag`, `log_prob` is already per-sample, so this **sums over the batch**: the loss becomes a scalar inflated by the batch size. Bernoulli, ZINB and Student-t are correct. | Only reduce over the last axis when `log_prob` has the same rank as `y_true`, i.e. when the distribution returns per-feature values. |
| H2 | `dataloader/dataloader.py:55,189-191` | The mutable default `batch_effect_encoders=dict()` is changed in place. A second DataLoader created without encoders (for example `model.py:387`, or `from_h5mu`) reuses the first loader's encoders. It then fails with "unseen labels" or **silently encodes with the wrong classes**. | The default is now `None`, and a fresh dict is created for each DataLoader. A dict passed in by the caller is still updated in place, as before. |
| H3 | `dataloader/dataloader.py:122-132` | Modalities are paired **by row position, not by obs name**. A MuData whose modalities hold the same cells in different orders produces misaligned RNA/ATAC rows without any error. `MultiModality` aligns the cells, but a raw `mu.MuData` input does not. | New `DataLoader.check_obs_aligned()` raises a `ValueError` when modalities differ in their cells or cell order. The DataLoader does not reorder them, because `Model.predict` writes outputs back into each modality's `.obsm` by row position. |
| H4 | `utils/tensor_utils.py:284`, called from `dataloader.py:122` | A dense `adata.X` (ndarray) crashes with `'numpy.ndarray' has no attribute 'tocoo'`, yet `Modality` accepts dense input. | `spmatrix_to_sparse_tensor` turns non-sparse input into COO first. |
| H5 | `io/file_reader.py:106` | Reading **two or more samples always fails**. The per-sample features tables are concatenated, so var has n_vars × n_samples rows. | Features are taken once, from the first sample. Any later sample whose features differ raises a `ValueError`. |
| H6 | `config/modality_file_config.py:18` + `io/file_reader.py:62-80,164` | The docs say `colnames` is optional, but its default `[]` leads to `df.columns = []`, which raises "Length mismatch". | `read_table` treats an empty `colnames` the same as `None`. |

### Medium

**Found while isolating the tests**

- `config/filter_config.py:9` with `base.py:28-36`: **every valid filter config emits a warning**. `FilterConfig` allows extra fields, but it inherits the "Unexpected field … typo" warning, so `min_counts`, `qc_vars` and the other real parameters all trigger it. Real typos get lost in the noise.
- `config/application_config.py:142-150`: names given in `conditioned_on_z`/`conditioned_on_z_hat` are never checked against the components that exist. The error only shows up later, inside utils/networkx, as "None cannot be a node".
- `dataloader/dataloader.py:181-197`: `n_vars_batch_effect` is computed from the encoder dict, not from the tensor that is returned. A reused encoder dict that holds columns not requested this time inflates `n_vars`, for example 3 when the actual width is 1. `workflow` and `tools/*` both reuse `dataloader.batch_effect_encoders`.

**Dataloader and tensor utilities**

- `utils/tensor_utils.py:182` has the same mutable-default bug as H2, in `create_tensor_from_df(encoder_dict=dict())`.
- `utils/tensor_utils.py:173` uses `reverse=True` in `create_backbone_layers`, which reverses the flat `[Dense, LN, Dense, LN]` list. The result is `LN → Dense → … → Dense` with no norm after the last Dense layer. The intent was presumably to reverse only the layer widths. ⚠ Decoders that use `reverse=True` are affected.
- The way a numeric batch column is encoded **depends on n_obs**, through the 20% unique-ratio heuristic in `dataframe_utils.check_is_categorical`. A subset DataLoader, as used in `tools/*`, can get a width-1 tensor while `n_vars_batch_effect` reports 2.
- A batch-effect column missing from `obs` is silently replaced by a zero column of width 1, while `n_vars_batch_effect` reports 0, so the two disagree.
- `dataframe_utils.py:121`: `check_is_categorical` raises a TypeError on string columns that contain NaN.
- `anndata_utils.py:42-50`: a NaN group label adds an all-NaN pseudo-bulk row.

**Losses and layers**

- `losses/gmm_density_loss.py:88`: the docstring says the weight can be a `ProgressiveScaler`, but `loss * self.weight` fails when given one. Current callers pass a `tf.Variable`, which works.
- Every `from_parameterizer_output` raises `UnboundLocalError` for a `tf.Variable` or numpy input.
- `layers/modifiers/binarize.py:42`: the typo `__name___` means an out-of-range threshold raises `AttributeError` instead of warning.
- `from_config` fails on 8 layers: all parameterizers, both samplers and `ProgressiveScaler`. They take no `**kwargs`, so Keras 3's `trainable`/`dtype` cause a TypeError. This breaks `clone_model` and `.keras` save/load.
- `normalize_library_size.py:61`: a row of all zeros produces NaN (0/0).
- `scale_library_size.py:78`: `tf.ones((tensor.shape[0], 1))` fails inside a `tf.function` when the batch dimension is unknown.

**Utils and environment**

- `utils/reflection_handler.py:50-67`: the "more than one file" check can never trigger, because only `glob(...)[0]` is returned. A filename that exists in two places resolves arbitrarily.

**Filter**

- `filter/filter_qc.py:96`: the default `qc_vars=["MT"]` is an unanchored regex, so it also matches `MTOR`, `MTHFR` and similar genes and **inflates `pct_counts_MT`**. It never matches mouse `mt-` genes.
- `filter/anndata_filter_handler.py:87`: a modality with no steps entry crashes with `NoneType is not iterable`.

**Config**

- `ComponentConfig` and `TrainingConfig` validators change the raw YAML dict (`yaml_raw`). Validating the same dict a second time behaves differently.
- `application_config.py:127` appends each sample without de-duplicating. A sample also listed under the modality, or a re-validated config, is read twice.
- Duplicate component or modality names silently overwrite earlier ones (`application_config.py:74,139`).
- `hierarchical_differential_analysis` entries are never cross-validated, and their `use_cluster` defaults to `""`, which leads to `obs[""]` in `workflow.py:520`.
- An unknown modality `type` is accepted with `dist=""`.
- Unknown top-level YAML keys are silently dropped. For example, a typo `sample:` loses every sample.
- `io/file_reader.py:164-171`: with `has_headers=True`, `index_col` is ignored and the obs/var index becomes 0..n.

**MultiModality**

- `MultiModality` changes the caller's mapping or MuData in place. With MuData input it drops the cells that are not shared from the **original** object.

**Deprecated APIs**

- `AnnData(..., dtype=)` is used in `anndata_utils.py:111` and `file_reader.py:109`.
- `np.mat` is used in `tensor_utils.py:285`. It was removed in NumPy 2.

### Low

**Distributions**

- In the ZINB, the mixture weights come from `sigmoid` and `1 - sigmoid`, which is imprecise in float32. With a large dropout logit, `log_prob(x>0)` is about −48 instead of about −22.
- The KL losses read `event_dims` from `shape[1]` but slice on the last axis, which is wrong for rank-3 inputs.
- `IndependentBernoulliDistribution.from_parameterizer_output` returns a plain tfp `Bernoulli`, not `cls`, and drops `**kwargs`.
- `MixtureMultivariateNormalDiagDistribution` is missing from `distributions/__init__`.

**Layers**

- `ToSparse` does nothing in eager mode, although it is currently unused.
- `ProgressiveScaler.activate(0)` never turns fully on. This may be intended.
- The mixture grid side `ceil(K**(1/D))` is wrong by one for some exact powers, such as 3125 = 5^5, because of floating-point error.

**Utils**

- `reflection_handler`:
  - It uses `assert` for validation, which `python -O` strips.
  - `_camel_to_snake` does not split acronyms or digits (`ZINBDistribution` becomes `zinbdistribution`).
  - Its glob is not actually recursive.
- `general_utils`:
  - `tensorflow_compatible_str` accepts a trailing `\n`, because it uses `match` with `$`.
  - An unknown parent component produces the unhelpful error "None cannot be a node".
  - Duplicate component names are accepted.
- `tensor_utils`:
  - `split` adds an empty chunk when n is divisible by batch_size.
  - `remove_nan_gradients` changes its input list.

**DataLoader and Modality**

- Calling `create_dataset()` a second time doubles `n_vars_batch_effect`.
- Duplicate sparse entries crash instead of being summed.
- `StudenttDataModifier` never sets `modality_name`.
- `Modality` changes the caller's `uns` dict and, for AnnData input, the input's `.uns`.
- Duplicate obs_names produce an unclear pandas error.

**Filter**

- `filter_qc`:
  - It crashes when `var.index.name` is unset, which is true of any AnnData not loaded through `FileReader`.
  - `filter_threshold` is described as optional but is required.
  - scanpy's default `percent_top` includes 500, which raises on modalities with fewer than 500 features.

**Config**

- The default IO paths stay relative, while explicit ones are made absolute.
- When an analysis component is unknown, the error message names the wrong problem.
- `learning_rate: null` gives a raw TypeError.
- A dense `.mtx` file fails on `.tocsr()`.
- Missing `components` and missing component `modalities` are not rejected.

---

## Suspicious but not marked as bugs

Most of these are pinned by passing tests that record the current behaviour.

**Config**

- FilterQC criteria with a misspelled key are skipped, with a warning for `field` and silently for `threshold` and `operator`.
- Operators must be names such as `gt`; the symbol `>` raises AttributeError.
- Relative IO paths are resolved against the working directory, not the config file's directory.
- A component-level `distribution_names` is always overwritten by the modality's `dist`.
- `n_latent_priors ≤ 0` is silently replaced by `2*dims+1`.
- Negative `kl_annealing_ratio` and dims values are accepted.
- An empty YAML file raises AttributeError.
- `EarlyStoppingConfig` is parsed but never used.

**Filter**

- Mutation is inconsistent across steps: FilterCells and FilterGenes work in place, while FilterQC returns a copy but writes QC columns to the input.

**Distributions and losses**

- The `StudenttDistribution` comment promises a finite variance, but df can be exactly 2.
- The `GMMKLDivergence` and Mixture docstrings describe the wrong `y_pred` layouts.
- `GMMDensityLoss` returns a batch mean under `Reduction.NONE`, so `sample_weight` cannot weight individual samples.

**Layers**

- The mixture `parameters` property only works when the layer was built with an input dimension of 1, and it squeezes away the component axis when K = 1.
- Binarize's default threshold of 1.0 turns values in [0, 1) into 0, which would wipe normalised or TF-IDF ATAC input.
- The ZINB data modifier docstring promises LogTransform and library-size normalisation, but only ToDense runs.
- `MultivariateNormalDiagSampler` honours `trainable`, while `StudenttSampler` ignores it.

**App and environment**

- Importing `app/cavachon_application.py` enables GPU memory growth and globally silences `FutureWarning`.
- `Settings` derives every path from `__file__`, which would break once the package is installed into site-packages.
- `cavachon/` has no `__init__.py`.

**Other**

- `tf.data.experimental.save/load` is deprecated.
- The constant has a typo: `TENSOR_NAME_ORIGIONAL_X`.
- Several docstrings are stale, for example in `create_backbone_layers`, `DuplicateMatrix` and `ScaleLibrarySize`.

---

## Suggested fix order

H1–H6 are fixed. Next:

1. **The `tensor_utils.create_tensor_from_df` mutable default**: it has the same pattern as H2. DataLoader always passes a dict explicitly, so it is safe in practice.
2. **`create_backbone_layers(reverse=True)`**: check whether trained decoders depend on the current layer order before changing it, because the change would alter the architecture.
3. **`filter_qc` `qc_vars`**: anchor the patterns (`^MT-`) and match case-insensitively.
4. Add `**kwargs` to the 8 layer constructors and pass them through to `super().__init__`.
