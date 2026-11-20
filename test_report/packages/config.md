# `cavachon/config`

[← Overview](../UNIT_TEST_REPORT.md)

| Status | Tests | Passed | Open bugs | Coverage | Coupling |
|---|---|---|---|---|---|
| 🟡 1 low bug open | [`test/config`](../../test/config) | 201 | 1 | 99% | Moderate: depends on 2 packages, used by 4 |

- **Depends on:** environment, utils
- **Used by:** io, model, scheduler, workflow
- **Mocked in tests:** `GeneralUtils`, `ReflectionHandler`; `open`/`yaml.load`, `os.path` lookups.

## Open bugs (strict xfails)

| Location | Issue | Impact |
|---|---|---|
| `analysis_config.py` | `embedding_method` defaults to `''`, which bypasses its validator: an explicit `''` is rejected, an omitted field is accepted. | Edge case |

## Fixed

- ✅ **H6** (with [io](io.md)): `colnames` is documented as optional, but its default `[]` led to "Length mismatch". `read_table` now treats an empty `colnames` like `None`.
- ✅ **Every valid filter config emitted a warning** (`filter_config.py:9` + `base.py:28-36`): `FilterConfig` allows extra fields but inherited the "Unexpected field … typo" warning.
- ✅ **Stricter validation.** These now raise an error:
  - duplicate modality or component names (previously the later one silently won);
  - an unknown modality `type` with no `dist` (previously accepted with `dist=""`);
  - unknown names in `conditioned_on_z`, `conditioned_on_z_hat` or `donor_components` (previously "None cannot be a node" deep in networkx);
  - invalid `hierarchical_differential_analysis` entries;
  - no components, or a component with no modalities;
  - an unknown analysis component, with a message that names the actual problem;
  - `learning_rate: null` (previously a raw TypeError);
  - an empty or non-mapping YAML file (previously AttributeError);
  - negative `kl_annealing_ratio` (must also sum to 1); `n_latent_dims`, `n_encoder_layers`, `n_decoder_layers` below 1; negative epoch counts or `z_hat_kl_weight`;
  - negative `n_latent_priors`. `0` (the default) means automatic, `2*dims+1`.
- ✅ Unknown top-level YAML keys now emit a warning (a typo `sample:` used to drop every sample).
- ✅ `hierarchical_differential_analysis.use_cluster` defaults to `cluster_<component>`, not `""`.
- ✅ `yaml_raw` is the YAML exactly as written; validators no longer change it, so validating twice behaves the same.
- ✅ Samples are de-duplicated (a sample also listed under the modality was read twice).
- ✅ Omitted IO paths are resolved against the working directory, like explicit ones.
- ✅ `EarlyStoppingConfig` gained a validated `min_delta` (default 5); `mode` and `patience` are validated too. See [scheduler](scheduler.md).
- ✅ `ComponentConfig` declares `z_conditional_dims` / `z_hat_conditional_dims` (set by `Model`) instead of receiving undeclared extra fields.

## Worth a look

- FilterQC criteria with a misspelled key are skipped, with a warning for `field` and silently for `threshold` and `operator`.
- Operators must be names such as `gt`; the symbol `>` raises AttributeError.
- Relative IO paths are resolved against the working directory, not the config file's directory.
- A component-level `distribution_names` is always overwritten by the modality's `dist`.
