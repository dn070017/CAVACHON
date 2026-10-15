# Attribution Analysis in CAVACHON

CAVACHON provides two attribution-analysis tools that use Integrated Gradients (IG) to explain how the latent code `z` influences the generative model's predicted expression. Both tools are located under `cavachon/tools/` and are wired into `Workflow`.

---

## 1. Conditional Attribution Analysis (`AttributionAnalysis`)

**File:** `cavachon/tools/attribution_analysis.py`  
**Workflow method:** `Workflow.visualize_conditional_attribution_scores()` → `InteractiveVisualization.attribution_score()`

### What it computes

For a chosen `component` and `modality`, it attributes the predicted expression of each feature to the latent dimensions of one or more `with_respect_to` components.

### Baseline and target

- **Baseline:** `z = 0` (the null latent code). For the target component this is `z_null = 0`; for all other components it is the posterior mean `z` of the cell/cluster being explained.
- **Target:** the posterior latent code `z` of the cell or cluster of interest.

### Integrated Gradients

The straight-line path from baseline to target is parameterized as

```
z(alpha) = z_baseline + alpha * (z - z_baseline),   alpha in [0, 1]
```

For `n_steps` Riemann steps:

1. Sample `alpha` uniformly along `[0, 1]`.
2. Forward the component through `model.hierarchical_encode` and the decoder to obtain the predicted expression mean `f(z(alpha))`.
3. Compute `gradient = tape.gradient(f, z(alpha))` w.r.t. the watched `z(alpha)` tensor.
4. Scale the gradient by `(z - z_baseline)` element-wise and average over the Riemann steps.

The final per-latent-dimension attribution is

```
IG_j(z, z_baseline) = (z_j - z_baseline_j) * mean_alpha( df / dz_j at z(alpha) )
```

By the IG completeness axiom, the sum of all per-dimension attributions equals the change in predicted expression:

```
sum_j IG_j = f(z) - f(z_baseline)
```

### Decoder target

The decoder target is the **libsize-scaled ZINB mean**: the ZINB parameterizer output is scaled by library size, exponent-transformed, floored at `1e-7`, and optionally multiplied by `(1 - dropout_probability)`. This matches the value returned by `Component.decode`, so attributions are on the expression scale rather than the unscaled softmax output.

### Output

`InteractiveVisualization.attribution_score()` writes per-feature attribution scores and produces an interactive visualization. The attribution columns are grouped by `with_respect_to` component.

---

## 2. Differential Attribution Analysis (`DifferentialAttributionAnalysis`)

**File:** `cavachon/tools/differential_attribution_analysis.py`  
**Workflow method:** `Workflow.perform_differential_attribution_analysis()`

### What it computes

Given a DEG or HDEG table (output of `DifferentialAnalysis` or `HierarchicalDifferentialAnalysis`), it attributes the predicted expression difference between Group A and Group B to the `with_respect_to` components.

### Group definitions

- **DEG tables** must contain the columns `GroupA` and `GroupB` (written by `DifferentialAnalysis.between_two_groups`).
- **HDEG tables** contain `DonorCluster` and `RecipientCluster`.

Cells are selected from `mdata[modality].obs[use_cluster]` using these group labels.

### Baseline and target

- **Baseline:** mean posterior `z` of Group B cells: `z_baseline = mean_z(group_b_index)`.
- **Target:** mean posterior `z` of Group A cells: `z_target = mean_z(group_a_index)`.

`Delta = f(z_target) - f(z_baseline)` is the predicted expression change the attributions must reconstruct.

### Per-component aggregation

Latent dimensions are summed per component:

```
attr_c = sum_{j in component c} IG_j(z_target, z_baseline)
```

IG completeness then gives:

```
sum_c attr_c = f(z_target) - f(z_baseline) = Delta
```

### Normalized contributions

For simplex-style interpretation:

- **Signed contribution:** `contrib_c = attr_c / Delta` (sums to 1, can be negative).
- **Absolute contribution:** `abs_contrib_c = |attr_c| / sum_c' |attr_c'|` (always in `[0, 1]`, sums to 1).

When `len(with_respect_to) == 1`, normalized `contrib_*` / `abs_contrib_*` columns are omitted because they would be trivially `±1` or `1`. Only raw `attr_*` and `abs_attr_*` are reported.

If `Delta` or the total absolute attribution is near zero, normalized contributions are reported as `0.0` instead of `NaN`.

### Decoder handling

The decoder expects the full feature dimension of the modality. The analysis therefore keeps the full modality in the forward batch and selects the DEG subset from the decoder output with `tf.gather(..., feature_indices, axis=-1)`.

### Gradient computation details

The IG loop watches **per-component** `z_step` tensors rather than a single concatenated tensor. This avoids a TensorFlow eager-execution gradient disconnect. Gradient calls are performed **inside** the persistent `GradientTape` context; calling `tape.gradient` after exiting the context loses the gradient path through sliced tensors.

### Output

For each DEG/HDEG table the workflow writes:

- `{output_name}.tsv` — attributed DEG table with columns:
  - `Delta` — predicted expression change
  - `attr_<component>` — raw signed attribution per component
  - `abs_attr_<component>` — absolute attribution per component
  - `contrib_<component>` — signed contribution (if ≥ 2 components)
  - `abs_contrib_<component>` — absolute contribution (if ≥ 2 components)
  - `LogFC` — log fold-change copied/computed from the DEG table
- `{output_name}_radviz.html` (if ≥ 2 components) — RadViz projection with:
  - vertices placed on a unit circle, one per `with_respect_to` component
  - point position = weighted average of vertices using `abs_contrib_<component>`
  - color = "Up in A" / "Down in A" from the sign of `LogFC`
  - size = `|LogFC|`

---

## Common implementation notes

- Both tools use `tf.GradientTape` with `persistent=True`.
- Both scale IG by the **input** difference `(z - z_baseline)`, not the output difference, satisfying IG completeness.
- The `with_respect_to` list controls which component latent codes are varied between baseline and target; other component latents are held at their posterior means.
- Empty overlaps between DEG features and modality `var_names` are warned and skipped rather than raising.
