# tools: cluster and attribution analysis

[← tools](README.md) · [← Overview](../../UNIT_TEST_REPORT.md)

Modules: `cluster_analysis.py`, `attribution_analysis.py`
Tests: [`test/tools/cluster_attribution`](../../../test/tools/cluster_attribution) — 57 passed, 1 open bug.

**Mocked:**
- cluster_analysis: `DataLoader` (MagicMock); the two distribution classes are fakes built on real tfp `MultivariateNormalDiag` / `MixtureSameFamily` that record their inputs.
- attribution_analysis: `DataLoader` with a real `tf.data.Dataset`; `MultivariateNormalDiagSampler` returns the loc; `Model` / `Component` are MagicMocks whose networks are small tf-op fakes, so gradients flow.

**Checked and correct:**
- GMM parameter extraction, Bhattacharyya distance, moment-matched merging (including transitive chains and the agglomerative `max_k` cap).
- Cluster posteriors: rows logsumexp to 0, match the analytic sigmoid(2z) for two 1-D clusters, and don't depend on batch size.
- z vs z_hat priors, including the root fallback.
- Integrated z_hat clusters: means, marginal variances and weights of every (parent, child) combination, the encoder called on the cluster means in inference mode; parent z_hat GMMs for root, learned-prior and nested (recursive) parents; E[scale²] over the sigma points for an input-dependent scale.
- KNN proportions and enrichment scores, contingency table, classification metrics.
- Attribution target: parent → child conditional propagation, stopping at the requested component; `selected_variables` order; batch-size invariance.
- Integrated gradients: completeness for linear and nonlinear decoders, and the baseline zeroing only `with_respect_to` (the other components keep their z).

## Open bugs (strict xfails)

### T16

`attribution_analysis.py:167-176` — `compute_integrated_gradient` with a `with_respect_to` that is not upstream of `component` (e.g. decode the parent, attribute to the child, or a sibling): the target doesn't depend on that z, `tape.gradient` returns `None`, and `.numpy()` raises `AttributeError: 'NoneType' object has no attribute 'numpy'`. Config validation only checks that the name is a component. (`DifferentialAttributionAnalysis` gives 0 instead.)
- **Impact:** a config error ends in an unclear crash. Found by the 2026-11-21 review.

## Fixed

- ✅ (2026-11-21) <a id="t1"></a>**T1** (`compute_integrated_gradient`): integrated gradients carried an extra factor of alpha. The target was f(alpha·z) but the gradient was taken w.r.t. z, which gives alpha·f'(alpha·z); with a linear decoder and steps=10 every attribution was 0.45× the correct value, and for nonlinear decoders the later path points were over-weighted. The point on the path, alpha·z, is now the watched input, as in `DifferentialAttributionAnalysis`. With it, the **baseline** changed too: only the z of `with_respect_to` moves from 0 to its value, and every other component keeps its own z (before, all latents were scaled by alpha, so the baseline was "all latents zero"). The attributions of a cell sum to f(z) − f(z with z of `with_respect_to` = 0), up to the left Riemann sum. `compute_attribution_target_batch` lost its `alpha` argument.
- ✅ (2026-11-21) <a id="t2"></a>**T2** (`_recombine_gmm_parameters`): the child variance was `(W_r**2) @ (W_child**2)` instead of `(W_r @ W_child)**2`, dropping the cross terms of the composed map (W_r=[[1,1]], W_child=[[1],[1]], σ²=1 gave 2, not 4). The weights are no longer sliced by hand: the hierarchical encoder is run on the cluster means of every (parent, child) combination, so the model gives the z_hat loc, and the Jacobian of the loc carries the prior variances into z_hat. Each cluster gets the **marginal** variance `(J²)ᵀσ²_x + E[scale_diag²]`, the spread of z_hat over the cells of that cluster (the model's scale_diag alone is the per-cell conditional and would make the clusters far too tight). This also picks up the loc bias and the progressive scaler, which the hand-written version ignored. The z_hat scale_diag depends on the input once its weight is trained (`initialize_scale_diag` only sets its start), so `E[scale_diag²]` is taken over the 2d sigma points `μ ± √(d·σ²) eᵢ` (unscented transform; exact when scale² is at most quadratic, and when it is constant); the scale at the cluster mean alone underestimated the variance by up to ~50%. Checked against a real `HierarchicalEncoder` with a trained-like (random) scale weight by Monte Carlo: means and variances agree within 1% for every combination. `compute_integrated_cluster_log_probability` now raises a ValueError when the component does not set `reparameterize_z_hat` (z_hat is then not a distribution) or has `conditioned_on_z` parents (not supported); `Component.reparameterize_z_hat` stores the flag. The parents now enter with their **z_hat** GMM (`_z_hat_gmm_parameters`): the z prior for a root (z_hat is z), the learned z_hat prior with `learn_z_hat_priors`, otherwise their own integrated GMM, built recursively through their parents and merged. Before, every parent used its z prior, which only held for root parents.
- ✅ (2026-11-21) <a id="t3"></a>**T3** (`_remove_small_clusters`): a cluster that no cell picked kept its logpy_z column when nothing else was removed, while the labels were renumbered, so `obs` labels and `obsm` columns disagreed. Columns of empty clusters are now dropped at the end as well.
- ✅ (2026-11-21) <a id="t6"></a>**T6** (`compute_attribution_target_batch`): an unknown component fell through the loop and returned `None`, and `compute_integrated_gradient` then failed with an unclear AttributeError. It now raises `KeyError("'name' is not a component of the model.")`; the `# type: ignore[return]` is gone.
- ✅ (2026-11-21) **T5**: the `None` defaults of `batch_effect_colnames` / `distribution_names` crashed `DataLoader`; they are now passed as empty mappings. See [differential](differential.md#t5).

## Worth a look

- **Conditional attribution scores scale with depth and expression level** (2026-11-21 review). The plotted score is the mean |IG| over all genes, on the count scale, and the decoder multiplies by the cell's own library size (ZINB), so deeper clusters and highly expressed genes dominate the bar chart. The workflow also calls it one `with_respect_to` at a time, so the cross-component normalization in `attribution_score` never runs.
- Integrated gradients make one gradient call per gene per step (`tape.gradient` on a selector), slow for many genes; a Jacobian would do it in one call.

- `_merge_similar_gmm_components` returns early when `n_clusters <= max_k`, so with the default `max_k=100` the Bhattacharyya `bc_threshold` merge never runs, contradicting both docstrings. The agglomerative path can hit a ZeroDivisionError if `fcluster` returns fewer than `max_k` clusters.
- `compute_integrated_gradient` / `compute_attribution_target_batch` are annotated `np.ndarray` but return `tf.Tensor`.
- `compute_neighbors_with_same_annotations` writes `obsm["z_custom"]` into the user's MuData; with a single cluster the enrichment score divides by zero.
- `compute_contingency_matrix` docstring says "var" but the code uses obs; the F1 score adds 1e-3 to its denominator.
