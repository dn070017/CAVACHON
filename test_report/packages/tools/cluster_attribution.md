# tools: cluster and attribution analysis

[← tools](README.md) · [← Overview](../../UNIT_TEST_REPORT.md)

Modules: `cluster_analysis.py`, `attribution_analysis.py`
Tests: [`test/tools/cluster_attribution`](../../../test/tools/cluster_attribution) — 43 passed, 4 open bugs.

**Mocked:**
- cluster_analysis: `DataLoader` (MagicMock); the two distribution classes are fakes built on real tfp `MultivariateNormalDiag` / `MixtureSameFamily` that record their inputs.
- attribution_analysis: `DataLoader` with a real `tf.data.Dataset`; `MultivariateNormalDiagSampler` returns the loc; `Model` / `Component` are MagicMocks whose networks are small tf-op fakes, so gradients flow.

**Checked and correct:**
- GMM parameter extraction, Bhattacharyya distance, moment-matched merging (including transitive chains and the agglomerative `max_k` cap).
- Cluster posteriors: rows logsumexp to 0, match the analytic sigmoid(2z) for two 1-D clusters, and don't depend on batch size.
- z vs z_hat priors, including the root fallback.
- KNN proportions and enrichment scores, contingency table, classification metrics.
- Attribution target: parent → child conditional propagation, stopping at the requested component; `selected_variables` order; batch-size invariance.

## Open bugs (strict xfails)

### T1

`attribution_analysis.py:146-163` — **integrated gradients carry an extra factor of alpha.** The target is f(alpha·z), but the gradient is taken w.r.t. z, which gives alpha·f'(alpha·z). Completeness (Σ IG = f(z) − f(0)) fails.
- Failing input: a linear decoder, steps=10 → every attribution is 0.45× the correct value (e.g. 0.225 vs 0.5).
- Fix: take the gradient at the point alpha·z (e.g. make the interpolated point the watched variable). `DifferentialAttributionAnalysis` does this correctly.
- **Impact: yes**: every attribution score is shrunk, nonlinearly for nonlinear decoders.

### T2

`cluster_analysis.py:486` — `_recombine_gmm_parameters` computes the child variance as `(W_r**2) @ (W_child**2)` instead of `(W_r @ W_child)**2`, dropping the cross terms of the composed linear map.
- Failing input: W_r=[[1,1]], W_child=[[1],[1]], child σ²=1, no parents → 2, expected 4.
- **Impact: yes**: wrong z_hat cluster covariances for child components.

### T3

`cluster_analysis.py:546-566` — `_remove_small_clusters`: if a cluster is never the argmax and nothing is removed, its logpy_z column stays, but labels are renumbered 0..k, so `obs` labels and `obsm` columns disagree. Empty columns are dropped only when some other cluster is removed.
- Failing input: rows [[.9,.05,.05]×2, [.05,.05,.9]×2], `min_n_obs=1` → labels [0,0,1,1], argmax of the returned logpy_z [0,0,2,2].
- **Impact: yes**, when a prior component wins no cells.

### T6

`attribution_analysis.py:183` — `compute_attribution_target_batch` returns `None` for a component not in `model.component_configs`; `compute_integrated_gradient` then fails with an unclear AttributeError on `.shape`. Marked with `# type: ignore[return]`.
- **Impact:** only on a typo in the component name.

## Fixed

- ✅ (2026-11-21) **T5**: the `None` defaults of `batch_effect_colnames` / `distribution_names` crashed `DataLoader`; they are now passed as empty mappings. See [differential](differential.md#t5).

## Worth a look

- `_merge_similar_gmm_components` returns early when `n_clusters <= max_k`, so with the default `max_k=100` the Bhattacharyya `bc_threshold` merge never runs, contradicting both docstrings. The agglomerative path can hit a ZeroDivisionError if `fcluster` returns fewer than `max_k` clusters.
- `compute_attribution_target_batch` scales **every** component's z by alpha, not only `with_respect_to`, so the IG baseline is "all latents zero".
- `compute_integrated_gradient` / `compute_attribution_target_batch` are annotated `np.ndarray` but return `tf.Tensor`.
- `compute_neighbors_with_same_annotations` writes `obsm["z_custom"]` into the user's MuData; with a single cluster the enrichment score divides by zero.
- `compute_contingency_matrix` docstring says "var" but the code uses obs; the F1 score adds 1e-3 to its denominator; `_recombine_gmm_parameters` uses the parents' z prior for their z_hat, which only holds for root parents.
