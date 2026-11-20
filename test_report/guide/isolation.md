# Isolation (mocking policy)

[← Overview](../UNIT_TEST_REPORT.md)

Every test is a unit test, isolated at the **package boundary**.

- **Other cavachon packages are mocked.** Anything the code under test imports from another `cavachon.<pkg>` is replaced with `MagicMock`s or small fakes, using `monkeypatch`. The patch is applied where the name is looked up, for example `cavachon.losses.negative_log_data_likelihood.ReflectionHandler.get_class_by_name`. Tests assert the arguments passed to these mocks.
  - Exception: `cavachon.environment.Constants` stays real, because it only holds plain values.
  - Code inside the package under test stays real. For example, DataLoader runs with its own `dataloader/modifiers`, and `Component` tests may use the real `modules.base` encoders.
  - Where a fake has to behave like a model (gradients, training), it is a tiny real `tf.keras` model or tf-op function with deterministic outputs, not a `MagicMock`.
- **No real file I/O, temporary files, subprocesses or network.** These calls are mocked, and the tests assert the paths and arguments they receive:
  - `open`/`yaml.load`, `pd.read_csv`, `mmread`, `mu.read`
  - `write_h5ad`/`read_h5ad`, `Dataset.save`/`tf.data.Dataset.load`
  - `glob`/`import_module` (for ReflectionHandler), `os.path.realpath`/`isdir`, `os.makedirs`
  - plotly `Figure.write_html`/`write_image`/`show`, `Path.mkdir`, `TemporaryDirectory`, `copytree`
  - **mlflow** and **gseapy** (`get_library` downloads gene sets) are always mocked.
- **Third-party libraries stay real**: TensorFlow, tfp, numpy, scipy, pandas, anndata, muon, scanpy, sklearn, plotly, pydantic and Keras. The code computes with them, and they serve as numeric references. TSNE and UMAP are the exception in `tools`: they are replaced by a deterministic fake to keep the tests fast.
- Shared mocks live in each test directory's `conftest.py`.

## Bugs moved to the package where they live

When the code was isolated, a few bugs turned out to belong to a different package than the one whose tests first exposed them. Each was moved:

| Bug | Moved from | Moved to |
|---|---|---|
| `tf.Variable` input crash | losses | [distributions](../packages/distributions.md) |
| Dense-X `.tocoo()` crash | dataloader | [utils](../packages/utils.md) (`test_tensor_utils.py`) |
| Duplicate sparse entries crash | dataloader | [utils](../packages/utils.md) (`test_tensor_utils.py`) |
| FilterConfig extra-field warning | filter | [config](../packages/config.md) |
