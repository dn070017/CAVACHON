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
| workflow (`setup_train_scheduler` only) | `test/workflow` | 2 | 0 | not measured |
| **Whole suite (after M1/M2/M5/M6/M7 and early-stopping fixes)** | | **1222** | **32** | |

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
- **z_hat annealing:**
  - the KL ramp only reaches its target if `n_z_hat_training_epochs > 0`;
  - the z_hat sampler is left stochastic after annealing;
  - the z_hat prior is k-means-initialised twice.
- **AnnealingCallback:**
  - it freezes the model and relies on the scheduler to unfreeze it;
  - loss patterns match anywhere in the name, not as a prefix.
- **`Component.train_step` doesn't drop NaN gradients**, unlike `Model.train_step`. This is latent.

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
- ✅ Fixed: `EarlyStoppingConfig` was parsed but never used (see the coupled-packages section).

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

H1–H6 and every medium item are fixed. Remaining:

1. The low-severity items. Each is still documented by a strict xfail.
