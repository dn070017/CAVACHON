# Unit Test Report

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
| **Total (low-coupling)** | | | **914** | **7** |

The per-group counts above are from before the fixes. After the H1–H6, medium and low fixes, these packages have 914 passed and 7 xfailed. All 7 remaining xfails are low-severity bugs, left open on purpose.

Packages that import from exactly 4 other cavachon packages were added later, under the same isolation rules. See [Coupled packages (out-deps = 4)](#coupled-packages-out-deps--4).

| Packages tested | Test dirs | Passed | xfail | Coverage (own tests only) |
|---|---|---|---|---|
| callbacks | `test/callbacks` | 43 | 6 | 91% |
| scheduler | `test/scheduler` | 72 | 8 | 96% |
| modules (base, parameterizers, preprocessors) | `test/modules/{base,parameterizers,preprocessors}` | 80 | 7 | 99% (modules overall) |
| modules (components) | `test/modules/components` | 93 | 11 | (included above) |
| model | `test/model` | 91 | 6 | 99% |
| workflow | `test/workflow` | 86 | 3 | 100% |
| tools (differential, hierarchical, differential attribution) | `test/tools/differential` | 74 | 2 | 99–100% |
| tools (cluster, attribution) | `test/tools/cluster_attribution` | 41 | 6 | 99–100% |
| tools (interactive visualization, enrichment) | `test/tools/visualization_enrichment` | 108 | 6 | 99–100% |
| **Whole suite (latest)** | | **1653** | **55** | |

Every package now has tests. See [Tools](#tools) for the tools findings.

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

### Medium. ✅ All fixed (2026-11-13)

Every medium item below is fixed.

**Fixes that change behaviour users will see:**

- **Default FilterQC `qc_vars`**
  - Mitochondrial genes now match `(?i)^MT-` and spike-ins match `^ERCC-`.
  - Human `pct_counts_MT` no longer counts MTOR, MTHFR and similar genes, so it drops.
  - Mouse `mt-` genes are now detected, so MT filtering removes cells it used to keep.
  - Column names are unchanged, and configs that set `qc_vars` explicitly behave as before.
- **Stricter configs.** These now raise an error:
  - duplicate modality or component names;
  - an unknown modality `type` with no `dist`;
  - unknown names in `conditioned_on_z`, `conditioned_on_z_hat` or `donor_components`;
  - invalid `hierarchical_differential_analysis` entries.

  Unknown top-level YAML keys now emit a warning. The repo configs still load.
- **`hierarchical_differential_analysis.use_cluster`** defaults to `cluster_<component>`, not `""`.
- **`yaml_raw`** is now the YAML exactly as written. Nothing outside `cavachon/config` reads it.
- **`has_headers: true`** barcodes and features files are now indexed by their first column. No repo config uses this.
- **`n_vars_batch_effect`** always equals the width of the batch-effect tensor. A reused encoder now fixes how a column is encoded, so subset DataLoaders encode the same way as the full one.
- **A batch column missing from `obs`** emits a warning but is still skipped.
- **`MultiModality`, `Modality` and the config validators** no longer change their inputs in place. `workflow.py` uses only the returned objects, so it is unaffected.
- **`ReflectionHandler`** raises an error when a class name matches files at the same depth in more than one place. No current lookup is ambiguous.

**Checked:** an end-to-end run on synthetic data used the real code throughout, from config → h5ad → filter → MultiModality → DataLoader → Model. It trained 7 epochs, ran predict and saved the weights. The ATAC cells were stored in reverse order and were aligned correctly. The batch-effect widths match `n_vars`, and the embeddings are finite. Old layer weights still load into the fixed layers.

**Side observation (not a bug in this code path):** `tf.train.Checkpoint` does not restore the mixture-prior loc/scale weights, because they are kept in Python lists. CAVACHON saves through Keras `model.save_weights`, which does restore them.

**Found while isolating the tests**

- ✅ `config/filter_config.py:9` with `base.py:28-36`: **every valid filter config emits a warning**. `FilterConfig` allows extra fields, but it inherits the "Unexpected field … typo" warning, so `min_counts`, `qc_vars` and the other real parameters all trigger it. Real typos get lost in the noise.
- ✅ `config/application_config.py:142-150`: names given in `conditioned_on_z`/`conditioned_on_z_hat` are never checked against the components that exist. The error only shows up later, inside utils/networkx, as "None cannot be a node".
- ✅ `dataloader/dataloader.py:181-197`: `n_vars_batch_effect` is computed from the encoder dict, not from the tensor that is returned. A reused encoder dict that holds columns not requested this time inflates `n_vars`, for example 3 when the actual width is 1. `workflow` and `tools/*` both reuse `dataloader.batch_effect_encoders`.

**Dataloader and tensor utilities**

- ✅ `utils/tensor_utils.py:182` has the same mutable-default bug as H2, in `create_tensor_from_df(encoder_dict=dict())`.
- ✅ `utils/tensor_utils.py:173` uses `reverse=True` in `create_backbone_layers`, which reverses the flat `[Dense, LN, Dense, LN]` list. The result is `LN → Dense → … → Dense` with no norm after the last Dense layer. The intent was presumably to reverse only the layer widths. ⚠ Decoders that use `reverse=True` are affected. **Fixed: `reverse=True` now reverses only the widths, so the encoders are Dense64 → LN → Dense64 → LN → Dense32 → LN.** ⚠ This changes the encoder architecture. Checkpoints saved before this change will not load, and all models must be retrained (this was decided).
- ✅ The way a numeric batch column is encoded **depends on n_obs**, through the 20% unique-ratio heuristic in `dataframe_utils.check_is_categorical`. A subset DataLoader, as used in `tools/*`, can get a width-1 tensor while `n_vars_batch_effect` reports 2.
- ✅ A batch-effect column missing from `obs` is silently replaced by a zero column of width 1, while `n_vars_batch_effect` reports 0, so the two disagree.
- ✅ `dataframe_utils.py:121`: `check_is_categorical` raises a TypeError on string columns that contain NaN.
- ✅ `anndata_utils.py:42-50`: a NaN group label adds an all-NaN pseudo-bulk row.

**Losses and layers**

- ✅ `losses/gmm_density_loss.py:88`: the docstring says the weight can be a `ProgressiveScaler`, but `loss * self.weight` fails when given one. Current callers pass a `tf.Variable`, which works.
- ✅ Every `from_parameterizer_output` raises `UnboundLocalError` for a `tf.Variable` or numpy input.
- ✅ `layers/modifiers/binarize.py:42`: the typo `__name___` means an out-of-range threshold raises `AttributeError` instead of warning.
- ✅ `from_config` fails on 8 layers: all parameterizers, both samplers and `ProgressiveScaler`. They take no `**kwargs`, so Keras 3's `trainable`/`dtype` cause a TypeError. This breaks `clone_model` and `.keras` save/load.
- ✅ `normalize_library_size.py:61`: a row of all zeros produces NaN (0/0).
- ✅ `scale_library_size.py:78`: `tf.ones((tensor.shape[0], 1))` fails inside a `tf.function` when the batch dimension is unknown.

**Utils and environment**

- ✅ `utils/reflection_handler.py:50-67`: the "more than one file" check can never trigger, because only `glob(...)[0]` is returned. A filename that exists in two places resolves arbitrarily.

**Filter**

- ✅ `filter/filter_qc.py:96`: the default `qc_vars=["MT"]` is an unanchored regex, so it also matches `MTOR`, `MTHFR` and similar genes and **inflates `pct_counts_MT`**. It never matches mouse `mt-` genes.
- ✅ `filter/anndata_filter_handler.py:87`: a modality with no steps entry crashes with `NoneType is not iterable`.

**Config**

- ✅ `ComponentConfig` and `TrainingConfig` validators change the raw YAML dict (`yaml_raw`). Validating the same dict a second time behaves differently.
- ✅ `application_config.py:127` appends each sample without de-duplicating. A sample also listed under the modality, or a re-validated config, is read twice.
- ✅ Duplicate component or modality names silently overwrite earlier ones (`application_config.py:74,139`).
- ✅ `hierarchical_differential_analysis` entries are never cross-validated, and their `use_cluster` defaults to `""`, which leads to `obs[""]` in `workflow.py:520`.
- ✅ An unknown modality `type` is accepted with `dist=""`.
- ✅ Unknown top-level YAML keys are silently dropped. For example, a typo `sample:` loses every sample.
- ✅ `io/file_reader.py:164-171`: with `has_headers=True`, `index_col` is ignored and the obs/var index becomes 0..n.

**MultiModality**

- ✅ `MultiModality` changes the caller's mapping or MuData in place. With MuData input it drops the cells that are not shared from the **original** object.

**Deprecated APIs**

- ✅ `AnnData(..., dtype=)` is used in `anndata_utils.py:111` and `file_reader.py:109`.
- ✅ `np.mat` is used in `tensor_utils.py:285`. It was removed in NumPy 2.

### Low. ✅ Mostly fixed (2026-11-13)

Seven xfails remain, all deliberately left open:

- **`event_dims` in the KL losses:** latent; current inputs are always rank 2.
- **`ToSparse`:** unused.
- **`ProgressiveScaler.activate(0)`:** possibly intended.
- **ReflectionHandler:** `assert` validation and acronyms in `_camel_to_snake`.
- **`embedding_method`:** an omitted field skips its validator.

Fixes in this round that users will notice:

- **IO paths:** omitted IO paths are resolved against the working directory when the config loads.
- **Configs that fail to load:** configs with no components, or a component with no modalities.
- **FilterQC:**
  - it works on AnnData whose var index has no name, matching on `var_names`;
  - `filter_threshold` is optional (metrics only, every cell kept);
  - the default `percent_top` is trimmed to the number of features.
- **Duplicate names:** duplicate `obs_names` and duplicate component names raise clear errors.

**Distributions**

- ✅ **Fixed. This was under-rated: it was a real training problem, not just imprecision.** `IndependentZeroInflatedNegativeBinomialDistribution` built its mixture weights as `probs = sigmoid(logits)` and `1 - probs`. In float32, `sigmoid(l)` rounds to 1.0 once the dropout logit reaches about 17, so the NB component got weight 0. Measured with x=5, mean 3 and dispersion 2:

  | logit | old `log_prob` | fixed / exact |
  |---|---|---|
  | 15 | −18.172 | −18.146 |
  | 17 | **−78.527** | −20.146 |
  | 20 | **−78.527** | −23.146 |
  | 30 | **−78.527** | −33.146 |

  - The gradient with respect to the dropout logit was **NaN** for any non-zero count once the logit reached about 17. `remove_nan_gradients` set it to 0, so a gene whose dropout logit drifted past about 17 got no gradient to pull it back.
  - **Fix:** the weights are now `Categorical(logits=stack([logits, 0]))`. Since `softmax([l, 0]) = [sigmoid(l), sigmoid(−l)]`, the weights are mathematically the same, but they are computed in log space.
  - Parameter meanings, shapes and the parameterizer are unchanged.
  - Models trained before the fix were only affected where a dropout logit exceeded about 17. Retraining is recommended anyway.
  - **Still open (documented in the class docstring, not a bug):** the "zero" component is NB(mean = 1e-7), not an exact point mass at 0. So `mean()` includes a `sigmoid(logits) * 1e-7` term, and `log_prob(0)` is very close to, but not exactly, the textbook ZINB value.
- The KL losses read `event_dims` from `shape[1]` but slice on the last axis, which is wrong for rank-3 inputs.
- ✅ Fixed: `IndependentBernoulliDistribution.from_parameterizer_output` returned a plain tfp `Bernoulli`, not `cls`, and dropped `**kwargs`.
- ✅ Fixed: `MixtureMultivariateNormalDiagDistribution` was missing from `distributions/__init__`.

**Layers**

- `ToSparse` does nothing in eager mode, although it is currently unused.
- `ProgressiveScaler.activate(0)` never turns fully on. This may be intended.
- ✅ The mixture grid side `ceil(K**(1/D))` is wrong by one for some exact powers, such as 3125 = 5^5, because of floating-point error.

**Utils**

- `reflection_handler`:
  - It uses `assert` for validation, which `python -O` strips.
  - `_camel_to_snake` does not split acronyms or digits (`ZINBDistribution` becomes `zinbdistribution`).
  - Its glob is not actually recursive.
- `general_utils`:
  - ✅ `tensorflow_compatible_str` accepts a trailing `\n`, because it uses `match` with `$`.
  - ✅ An unknown parent component produces the unhelpful error "None cannot be a node".
  - ✅ Duplicate component names are accepted.
- `tensor_utils`:
  - ✅ `split` adds an empty chunk when n is divisible by batch_size.
  - ✅ `remove_nan_gradients` changes its input list.

**DataLoader and Modality**

- ✅ Calling `create_dataset()` a second time doubles `n_vars_batch_effect`.
- ✅ Duplicate sparse entries crash instead of being summed.
- ✅ `StudenttDataModifier` never sets `modality_name`.
- ✅ `Modality` changes the caller's `uns` dict and, for AnnData input, the input's `.uns`.
- ✅ Duplicate obs_names produce an unclear pandas error.

**Filter**

- `filter_qc`:
  - ✅ It crashes when `var.index.name` is unset, which is true of any AnnData not loaded through `FileReader`.
  - ✅ `filter_threshold` is described as optional but is required.
  - ✅ scanpy's default `percent_top` includes 500, which raises on modalities with fewer than 500 features.

**Config**

- ✅ The default IO paths stay relative, while explicit ones are made absolute.
- ✅ When an analysis component is unknown, the error message names the wrong problem.
- ✅ `learning_rate: null` gives a raw TypeError.
- ✅ A dense `.mtx` file fails on `.tocsr()`.
- ✅ Missing `components` and missing component `modalities` are not rejected.

---

## Coupled packages (out-deps = 4)

These are `callbacks`, `scheduler` and `modules`. No source code was changed; each bug below is documented by a strict xfail.

**Mocking:** anything these packages import from other cavachon packages is mocked:
- Layers and losses are replaced by tiny real `tf.keras` layers or loss objects with deterministic outputs, so the Keras models still build and train.
- `ProgressiveScaler`, `DataLoader`, the distributions, `TensorUtils` and `ReflectionHandler` are fakes.
- **mlflow is fully mocked.**
- The scheduler drives a `FakeModel`/`FakeComponent` that records `compile`/`fit`, trainable flags and annealing state for each phase.

Code inside the package stays real. For example, `Component` tests may use the real `modules.base` encoders.

**Commit 73eecd0 checked:** passing scheduler tests confirm the intent of "parent annealing ramps only alpha with the KL tied to it". During parent annealing the parents are frozen with weights 0. The child's alpha and standard KL ramp together, and its z_hat KL is 0. Afterwards alpha is pinned to (1, 1) and the standard KL to 1.

### Not in the production path (latent)

`Component.compile` and `Component.train_step` are never called in production. Training goes through `Model.compile` and `Model.train_step`, driven by the scheduler. Component bugs C1, C2, C3, C4 and C8 therefore affect only standalone Component training. They are real, but they don't affect current results.

### Medium (affect current runs or configs). ✅ M1, M2, M5, M6 and M7 fixed (2026-11-13)

| # | Location | Issue |
|---|---|---|
| M1 | `modules/base/decoder_data_parameterizer.py:58-63` | **`n_decoder_layers` is ignored.** The decoder backbone is hard-coded to `n_layers=2, base_n_neurons=32`. `Component.setup_decoders` passes the per-modality value, but it has no effect. The docstring says the default is 3. **Fixed:** `n_layers` is passed to the backbone. **With the default of 3 (your choice), every decoder is now Dense32 → LN → Dense64 → LN → Dense64 → LN instead of 2 layers.** Models trained before this change had 2-layer decoders. |
| M2 | `modules/components/component.py:494-501` + `modules/base/hierarchical_encoder.py:245,277` | **`progressive_iterations` never reaches the hierarchical encoder.** Component passes `progressive_step=`, and HierarchicalEncoder silently swallows unknown `**kwargs` (including `trainable`/`dtype`). The scaler therefore stays at 5000, and the scheduler later overwrites `total_iterations`, which limits the impact. **Fixed:** Component passes `progressive_iterations=`, and HierarchicalEncoder forwards `**kwargs` to `tf.keras.Model`, so unknown keywords now raise an error and `trainable`/`dtype` are honoured. |
| M3 | `callbacks/optimizer_state_callback.py:13,21` | **`OptimizerStateCallback` does nothing.** Keras 3 optimizers have no `get_weights`, and a bare `except` hides the error. The restore step also looks up `model._callbacks`, which doesn't exist. The scheduler adds this callback to every phase. |
| M4 | `callbacks/periodic_tsne_callback.py:43` | **`PeriodicTSNECallback` never saves.** `save_epochs = set()` is hard-coded, but the scheduler adds the callback whenever `output_dir` is set. |
| M5 | `workflow/workflow.py:242-251` | **Positional arguments are swapped.** `SequentialTrainingScheduler(..., outdir, batch_effect_colnames, distribution_names)` is called, but the signature is `(..., outdir, distribution_names, batch_effect_colnames)`. This is harmless today, because the only consumer is the non-saving `PeriodicTSNECallback`. It will matter once M4 is fixed. The workflow package isn't unit-tested, so this was confirmed by reading the code. **Fixed:** the workflow passes `outdir`, `distribution_names` and `batch_effect_colnames` by keyword. A regression test is in `test/workflow/`. |
| M6 | `scheduler/sequential_training_scheduler.py:235-240` | **k-means++ crashes on a collapsed latent.** With fewer distinct points than clusters, every logit is `-inf` and `tf.gather` raises. Training aborts exactly when posterior collapse happens. **Fixed:** when every distance is 0, k-means++ falls back to choosing uniformly. |
| M7 | `scheduler/sequential_training_scheduler.py:727-764` and the other phase runners | **User `callbacks=` (or `verbose=`) passed to `fit` raise "multiple values for 'callbacks'".** They are kept in `kwargs_copy` and also passed explicitly. This is latent: the workflow never passes them. **Fixed:** each phase drops `callbacks` and `verbose` from the kwargs it forwards. User callbacks are still deep-copied into every phase. |
| C1 | `modules/components/component.py:1351` | `Component.train_step` calls `compiled_metrics.update_state(y_true, y_pred_dict)`, which raises a ValueError on Keras 3. *(latent; see above)* |
| C2 | `component.py:1308-1319` | A standard-KL-only setup raises KeyError, because y_true/y_pred are filled only under the GMM key. *(latent)* |
| C3 | `component.py:1183-1212` | The `_standard_kl_weights` and `_gmm_kl_weights` Variables are never passed to the losses, so assigning them has no effect, despite the docstring. *(latent)* |
| C4 | `component.py:1242-1254` | `_data_loss_weights` is copied with `float(...)` by NegativeLogDataLikelihood, so assigning it later has no effect. *(latent)* |

### Low

- **Scheduler:**
  - float truncation in the KL schedule boundaries (`int(10*(0.1+0.7)) = 7`; 12 of 144 ratio/epoch combinations are affected);
  - k-means never runs when `kl_annealing_ratio[0] == 1.0`;
  - the prior std initialisation uses `std - 0.5` as the inverse softplus (about 15% too small at std ≈ 3);
  - the mlflow run is left open if `fit` raises;
  - `compute_modality_weight(constant=True)` raises UnboundLocalError;
  - the training order isn't a real topological sort: a child listed before its parent raises KeyError.
- **Callbacks:**
  - `PeriodicTSNECallback(batch_size=int)` uses the type as a default value;
  - `EarlyStoppingCallback` prints the restored epoch one too low;
  - a positional `monitor` argument is swallowed by `cumulative_offset`.
- **Modules:**
  - `exp_transform=True` crashes `Parameterizer.make` on Keras 3 (`tf.math.exp` on a KerasTensor);
  - HierarchicalEncoder and EncoderLatentParameterizer have no `get_config`, so the auto-config records default values;
  - `Component`:
    - pops entries from the caller's `loss_weights` dict;
    - a weight of 0 still creates the loss, whose metric is then NaN;
    - has no `test_step`, so `evaluate()` fails;
    - drops `**kwargs` in `__init__`;
    - `decode()` requires undocumented matrix keys.

### Worth a look (not marked as bugs)

- **ZINB library size semantics.** The library size is the row sum of *log1p* counts, because NormalizeLibrarySize runs after LogTransform. The decoder computes `exp(min(mean·libsize, 7)) − 1`, so the expected count is capped at about 1,096. This may be intended; worth confirming.
- ✅ **`EarlyStoppingConfig` is now used (fixed 2026-11-13).** The scheduler had ignored the config: it hard-coded `monitor="loss"` and `min_delta=5`, and set patience to `max(10, epochs/20)` from the global `max_regular_training_epochs`. Your `patience: 50` became 10.
  - The callback is now built from `monitor`, `mode`, `patience` and a new `min_delta` field (default 5).
  - `early_stopping: true` uses the documented defaults (loss, min, patience 25), and `false` disables it.
  - `mode`, `patience` and `min_delta` are validated.
  - An end-to-end run with `patience: 2` stopped each regular phase after 3 epochs.
- ✅ **z_hat annealing (fixed 2026-11-13):**
  - **KL ramp:** the z_hat GMM KL ramp used `epoch / n`, so the last annealing epoch only reached (n−1)/n of the target, and n = 1 gave weight 0. It now uses `(epoch + 1) / n`.
  - **Samplers:** samplers sample whenever `training=True` and their `stochastic` flag is on, whether or not their component is frozen. Before, a frozen parent always passed its *mean* z and z_hat to children.
    - z is stochastic for the whole training.
    - z_hat is deterministic until its component's z_hat annealing (when it gets a KL term and a calibrated scale), and stochastic from then on.
    - Both flags are set to 0 when `fit` finishes, and reset at the start of `fit`, since they're stored in the weights.
  - **z_hat k-means:** the z_hat prior is k-means-initialised exactly once:
    - at the start of z_hat annealing, on the z_hat learned during regular training, if annealing runs;
    - otherwise in the KL or regular phase, with a warning that the z_hat prior is then never trained.
    - Previously it ran in the KL or regular phase *and* again at annealing, and the first result was discarded.
- **AnnealingCallback:**
  - it freezes the model and relies on the scheduler to unfreeze it;
  - loss patterns match anywhere in the name, not as a prefix.
- ✅ Fixed: `Component.train_step` didn't drop NaN gradients, unlike `Model.train_step`. It now passes them through `TensorUtils.remove_nan_gradients`.

## Model and workflow

**Rules:** the same isolation rules apply.

- **model:**
  - `Component` is replaced by a tiny real Keras model with deterministic z / z_hat, so the parent → child wiring can be checked.
  - The losses, `ProgressiveScaler`, `DataLoader`, `GeneralUtils` and `TensorUtils` are fakes.
  - A real SGD optimizer checks `train_step` against a hand-computed update.
- **workflow:**
  - All 16 cavachon names it imports are mocked, including every `tools` class.
  - `os.makedirs`, `read_h5ad` and `to_csv` are recorders.
  - A few tests read other packages' constructor signatures with `inspect`, to catch argument-order mistakes like the earlier M5 swap.

**Checked and correct:**
- **Wiring:** parent z / z_hat reach the right children, including grandchildren, and multiple parents are concatenated in config order.
- **Loss weighting:** each loss is counted once, and the per-loss metrics are unweighted.
- **Gradients:** the output of `remove_nan_gradients` is what the optimizer applies, and frozen components are excluded.
- **`predict`:** rows stay in order, and each output is written to the correct modality's `.obsm`.
- **Argument passing:** `batch_effect_colnames` and `distribution_names` are passed correctly throughout the workflow.

### Bugs (strict xfails)

| # | Location | Issue | Production impact |
|---|---|---|---|
| W1 | `workflow.py:572-596` | **Differential attribution uses every DEG/HDEG table in the folder**, including tables from other components and modalities, and runs them through its own component and `use_cluster`. | **Yes**: silently wrong attribution output when 2+ DEG/HDEG entries for different components exist |
| MD3 | `model.py:389` (+ `self(data)` in `train_step`) | Batch keys aren't filtered to the model's inputs. Keras matches dict inputs by position, so a MuData with an unused modality feeds e.g. `atac_matrix` in as `rna_matrix`. | Yes, when the MuData carries a modality no component uses ✅ **Fixed:** `Model.select_inputs` restricts each batch to the model's input names in `predict` and `train_step`. |
| W7 | `workflow.py:377-390` | A `color_by` of the form `<obsm_key>/<index>` puts a `/` in the embedding filename, so the write fails before the h5mu is saved. | Only with that `color_by` form |
| W2 | `workflow.py:609-629`, `441-443` | Group labels from DEG tables are converted to `str`, but the obs column isn't, so integer-coded or zero-padded labels give empty groups. | Only with non-string annotations |
| MD1 | `model.py:645-654` | A component compiled with only the standard KL never trains it: y_true/y_pred are stored under the GMM key. | No (the scheduler always passes both) |
| MD2 | `model.py:414` | `super.__predict__(...)` should be `super().predict`, so `predict` on non-MuData input always raises. | No (the workflow passes MuData) ✅ **Fixed:** the call is now `super().predict(...)`, with the inputs filtered too. |
| MD4 | `model.py:940-952`, `1090-1107` | `hierarchical_encode(strict=False)` still concatenates `None` for a missing parent. | No (analysis API only) |
| MD5 | `model.py:362-366` | `predict_x` isn't reset per modality, so x_parameters are computed for `save_x=False` modalities. Whether it happens depends on modality order. | Wasted memory only |
| MD6–MD8 | `model.py:471-497`, `734-736`, `462/522` | A weight of 0 still creates the loss; its scaler metric divides by 0 and gives NaN; `compile` pops entries from the caller's `loss_weights`. | No |
| W3–W6 | `workflow.py:426-502` | Smaller workflow bugs: <ul><li>DEG filenames keep spaces;</li><li>a `/` in a label crashes pairwise DEG;</li><li>single-pair DEG results on different annotations overwrite each other;</li><li>empty DEG/HDEG directories are always created, which lets a later enrichment-only config pass validation.</li></ul> | Cosmetic / edge cases ✅ **Fixed:**<ul><li>no spaces in filenames;</li><li>`GroupA`/`GroupB` are taken from the tool instead of re-parsed from `A/B` keys;</li><li>single-pair results are keyed by (modality, component, `use_cluster`) like pairwise mode, so they're no longer overwritten. Filenames change from `rna_comp_a_b_a_b.tsv` to `rna_comp_<use_cluster>_a_b.tsv`;</li><li>the output folders are only created when that analysis is configured.</li></ul> |

### Worth a look (not marked as bugs)
- `train_model` (`workflow.py:308`) calls `self.model.compile()` **inside** the per-component freeze loop, so it compiles once per component. This looks like an indentation slip. It's harmless but wasteful.
- `load_model_weights` swallows OSError with only a warning. With `load_weights: true, train: false`, the run then continues on random weights.
- `run()` calls `predict()` before `load_model_weights()`, probably to build the variables, at the cost of a full predict pass.
- In `Model`:
  - `save_z_hat` is built from `save_z` and never used; this is dead code.
  - Recompiling reuses the existing losses, so new weights for existing losses are ignored. The scheduler relies on this.
  - The losses are evaluated twice per training step, once for the loss and once for the metrics.

## Tools

**Rules:** the same isolation rules apply.
- `DataLoader`, `ReflectionHandler`, the distribution classes, `MultivariateNormalDiagSampler`, `Model` and `Component` are patched where each tools module looks them up.
- Models are small deterministic fakes built from tf ops, so gradients flow. Other tools classes imported by `interactive_visualization` / `enrichment_analysis` are mocks.
- **gseapy is always mocked** (`get_library` downloads gene sets). `Figure.write_html` / `write_image` / `show`, `Path.mkdir`, `TemporaryDirectory`, `copytree` and `pd.read_csv` are recorders, so there is no file or network I/O.
- TSNE and UMAP are replaced by a deterministic fake; PCA runs for real.

**Checked and correct:**
- Bayes factor / K values against hand computation, and the seeded global batch effect in `compute_x_means`.
- `DifferentialAttributionAnalysis` integrated gradients: exact for a linear model, completeness holds, and contributions sum to 1 (0 instead of NaN when nothing changes).
- Hierarchical donor substitution: the same donor cell is used for every donor component, and other components are untouched.
- Cluster posteriors: rows logsumexp to 0, match the analytic sigmoid for two 1-D clusters, and don't depend on batch size. Bhattacharyya distance and moment-matched merging are checked by hand.
- Figures: trace data, labels and axis titles; `write_html` is called only when a filename is given.

### Bugs (strict xfails)

| # | Location | Issue | Production impact |
|---|---|---|---|
| T1 | `attribution_analysis.py:146-163` | **Integrated gradients carry an extra factor of alpha.** The target is f(alpha·z) but the gradient is taken w.r.t. z, which gives alpha·f'(alpha·z). Completeness fails: with a linear decoder and steps=10, each attribution is 0.45× the correct value. (`DifferentialAttributionAnalysis` does it correctly.) | **Yes**: every attribution score is shrunk, nonlinearly for nonlinear decoders |
| T2 | `cluster_analysis.py:486` | `_recombine_gmm_parameters` computes the child variance as `(W_r**2) @ (W_child**2)` instead of `(W_r @ W_child)**2`, so cross terms are dropped. W_r=[[1,1]], W_child=[[1],[1]], σ²=1 gives 2, expected 4. | **Yes**: wrong z_hat cluster covariances for children |
| T3 | `cluster_analysis.py:546-566` | `_remove_small_clusters`: an empty cluster that isn't removed keeps its logpy_z column, but labels are renumbered 0..k, so `obs` labels and `obsm` columns disagree. Empty columns are dropped only when something else is removed. | Yes, when a prior component wins no cells |
| T4 | `differential_analysis.py:142, 224` | `keep_only_significant` uses abs(K). Identical decoded means give P(A>B)=P(B>A)=0, K≈-16, so the feature is kept as "significant" and sorted to the top. | Yes, for features with no difference at all |
| T5 | `differential_analysis.py:307-340`, `cluster_analysis.py:101, 149`, `attribution_analysis.py:68-75` | `batch_effect_colnames=None` / `distribution_names=None` defaults go straight into `DataLoader`, which calls `.get()` on them, so AttributeError. The hierarchical / differential-attribution subclasses already use `or {}`. | Only when called with defaults (the workflow always passes both) |
| T6 | `attribution_analysis.py:183` | `compute_attribution_target_batch` returns None for an unknown component; the caller then fails with an unclear AttributeError on `.shape`. | Only on a typo in the component name |
| T7 | `interactive_visualization.py:89-103` | `bar()` ignores `**kwargs` (title, axis titles) when `group` is None. | Cosmetic |
| T8 | `interactive_visualization.py:200` | `adata.uns.get("cavachon", "").get(...)`: `'str' has no attribute 'get'` when uns has no `cavachon` entry. | Only for AnnData not produced by cavachon |
| T9 | `interactive_visualization.py:540-544` | `attribution_score` writes every `with_respect_to` figure to the same filename; only the last survives. | Yes, with 2+ `with_respect_to` |
| T10 | `interactive_visualization.py:848` | `float(x[:-2])` on gseapy's `"12.35%"` gives 12.3; should strip only `%`. | Small numeric error in ring plots |
| T11 | `interactive_visualization.py:956-1041` | `prerank_enrichment_score` passes the full `Target` column as hover text to traces that plot a subset, so hover labels are offset. | Misleading hover labels |
| T12 | `enrichment_analysis.py:286-292` | `run()` applies the **term** threshold (default FDR q-val <= 0.05) to the **DEG ranking** column, keeping only genes with K <= 0.05. Added on purpose in a841a53 with `min_num_degs`, but one threshold can't serve both. Needs the author's call. | **Yes**, if unintended: the ranked list fed to prerank is mostly discarded |

### Worth a look (not marked as bugs)
- `_merge_similar_gmm_components` returns early when `n_clusters <= max_k`, so with the default `max_k=100` the Bhattacharyya `bc_threshold` merge never runs. The agglomerative path can hit a ZeroDivisionError if `fcluster` returns fewer than `max_k` clusters.
- `compute_attribution_target_batch` scales **every** component's z by alpha, not only `with_respect_to`, so the IG baseline is "all latents zero".
- The `HierarchicalDifferentialAnalysis.across_clusters_pairwise` parameter list differs from the base (`donor_components` where the base has `z_sampling_size`): a `type: ignore[override]` marks it.
- `across_clusters` with a single cluster or a NaN label crashes with numpy's "'a' cannot be empty"; results are keyed by the raw label (int keys despite `Mapping[str, ...]`).
- `compute_differential_attribution(n_steps=0)` silently returns NaN.
- `differential_volcano_plot` adds columns to the caller's DataFrame; `prerank_ringplot` crashes on an empty selection and always filters `<=` regardless of `threshold_operator`; `_safe_filename` can collide (`"A B"` and `"A/B"`).
- `compute_neighbors_with_same_annotations` writes `obsm["z_custom"]` into the user's MuData.
- Docstrings: `z_sampling_size` default (10 vs 5), radviz `logfc_column` default, contingency matrix "var" vs obs.

### Typing
`pixi run mypy` now reports no errors in the whole package. In tools, `embedding(color=...)` is annotated `Optional[str]` (a list never worked), `differential_volcano_plot` raises a ValueError when `component`/`modality` are missing (it previously crashed with KeyError later), and the remaining `type: ignore`s mark T5 (12 lines), T6 and the override above.

## Suspicious but not marked as bugs

Most of these are pinned by passing tests that record the current behaviour.

**Config**

- FilterQC criteria with a misspelled key are skipped, with a warning for `field` and silently for `threshold` and `operator`.
- Operators must be names such as `gt`; the symbol `>` raises AttributeError.
- Relative IO paths are resolved against the working directory, not the config file's directory.
- A component-level `distribution_names` is always overwritten by the modality's `dist`.
- ✅ Fixed: `n_latent_priors ≤ 0` was silently replaced by `2*dims+1`. Now `0` (the default) means automatic, `2*dims+1`, and negative values are rejected.
- ✅ Fixed: negative `kl_annealing_ratio` and dims values were accepted.
  - `kl_annealing_ratio` must now be non-negative and sum to 1.
  - `n_latent_dims`, `n_encoder_layers` and `n_decoder_layers` must be at least 1.
  - Epoch counts and `z_hat_kl_weight` must be at least 0, in both the component and the training config.
- ✅ Fixed: an empty YAML file raised AttributeError. An empty or non-mapping file now raises a clear ValueError.
- ✅ Fixed: `EarlyStoppingConfig` was parsed but never used (see the coupled-packages section).

**Filter**

- Mutation is inconsistent across steps: FilterCells and FilterGenes work in place, while FilterQC returns a copy but writes QC columns to the input.

**Distributions and losses**

- The `StudenttDistribution` comment promises a finite variance, but df can be exactly 2.
- ✅ Fixed: the `GMMKLDivergence` and Mixture docstrings described the wrong `y_pred` layouts.
- `GMMDensityLoss` returns a batch mean under `Reduction.NONE`, so `sample_weight` cannot weight individual samples.

**Layers**

- The mixture `parameters` property only works when the layer was built with an input dimension of 1, and it squeezes away the component axis when K = 1.
- Binarize's default threshold of 1.0 turns values in [0, 1) into 0, which would wipe normalised or TF-IDF ATAC input.
- ✅ Fixed (docstring): the ZINB data modifier docstring promised LogTransform and library-size normalisation, but only ToDense runs. It now says those are applied later by the preprocessor modifier.
- `MultivariateNormalDiagSampler` honours `trainable`, while `StudenttSampler` ignores it.

**App and environment**

- Importing `app/cavachon_application.py` enables GPU memory growth and globally silences `FutureWarning`.
- `Settings` derives every path from `__file__`, which would break once the package is installed into site-packages.
- `cavachon/` has no `__init__.py`.

**Other**

- ✅ Fixed: `tf.data.experimental.save/load` was deprecated; `Dataset.save` and `tf.data.Dataset.load` are now used.
- ✅ Fixed: the constant typo `TENSOR_NAME_ORIGIONAL_X`. It is now `TENSOR_NAME_ORIGINAL_X`, and the old name is kept as a deprecated alias.
- ✅ Fixed: several stale docstrings:
  - `create_backbone_layers`: LayerNorm, swish, and defaults 2/32/64;
  - `DuplicateMatrix`: listed a non-existent `threshold`;
  - `ScaleLibrarySize`;
  - Binarize: now says `>=`;
  - `MultivariateNormalDiagModifier`;
  - the ZINB parameterizer `make` defaults.

---

## Suggested fix order

H1–H6 and every medium item are fixed. Remaining:

1. The low-severity items. Each is still documented by a strict xfail.
