from collections.abc import Mapping
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
from scipy.special import expit, logsumexp

from cavachon.tools.cluster_analysis import ClusterAnalysis

from .conftest import FakeLatentModel, make_mdata, make_prior

# two 1-D clusters at -1 and +1, unit scale, equal weights
PRIOR_1D = dict(logits=[0.0, 0.0], mu=[[-1.0], [1.0]], scale=[[1.0], [1.0]])
Z_1D = np.array([[-2.0], [-1.5], [-1.0], [1.0], [1.5], [2.0]], np.float32)


def two_cluster_posterior(z):
    """log p(y | z) for PRIOR_1D: p(y=1 | z) = sigmoid(2 z)."""
    p1 = expit(2 * z[:, 0])
    return np.log(np.stack([1 - p1, p1], axis=1))


def batches_of(z, size):
    return [
        {"rna_matrix": tf.constant(z[i : i + size])} for i in range(0, len(z), size)
    ]


# ---------------------------------------------------------------------------
# _extract_gmm_parameters
# ---------------------------------------------------------------------------


def test_extract_gmm_parameters_splits_logits_loc_scale():
    prior = make_prior(
        logits=[0.0, np.log(3.0)],
        mu=[[1.0, 2.0], [3.0, 4.0]],
        scale=[[0.5, 1.0], [2.0, 3.0]],
    )
    params = ClusterAnalysis._extract_gmm_parameters(prior)
    np.testing.assert_allclose(params["pi"], [0.25, 0.75], rtol=1e-6)
    np.testing.assert_allclose(params["mu"], [[1.0, 2.0], [3.0, 4.0]])
    np.testing.assert_allclose(params["sigma2"], [[0.25, 1.0], [4.0, 9.0]])
    assert params["K"] == 2
    assert params["D"] == 2


# ---------------------------------------------------------------------------
# _compute_bhattacharyya_distance
# ---------------------------------------------------------------------------


def test_bhattacharyya_identical_gaussians_is_zero():
    mu = np.array([1.0, -2.0])
    s2 = np.array([0.5, 3.0])
    assert ClusterAnalysis._compute_bhattacharyya_distance(
        mu, s2, mu, s2
    ) == pytest.approx(0.0)


def test_bhattacharyya_hand_computed():
    # mean term: 0.125 * 4 / (1 + 3) = 0.125; variance term: 0.5 * log(16 / 12)
    d = ClusterAnalysis._compute_bhattacharyya_distance(
        np.array([0.0]), np.array([1.0]), np.array([2.0]), np.array([3.0])
    )
    assert d == pytest.approx(0.125 + 0.5 * np.log(16 / 12))


def test_bhattacharyya_is_symmetric_and_non_negative():
    rng = np.random.default_rng(0)
    mu1, mu2 = rng.normal(size=3), rng.normal(size=3)
    s1, s2 = rng.uniform(0.1, 2, size=3), rng.uniform(0.1, 2, size=3)
    d12 = ClusterAnalysis._compute_bhattacharyya_distance(mu1, s1, mu2, s2)
    d21 = ClusterAnalysis._compute_bhattacharyya_distance(mu2, s2, mu1, s1)
    assert d12 == pytest.approx(d21)
    assert d12 >= 0


# ---------------------------------------------------------------------------
# _merge_similar_gmm_components
# ---------------------------------------------------------------------------


def test_merge_similar_components_moment_matching():
    mu = np.array([[0.0], [0.1], [10.0]])
    sigma2 = np.ones((3, 1))
    pi = np.array([0.25, 0.25, 0.5])
    mu_m, s2_m, pi_m = ClusterAnalysis._merge_similar_gmm_components(
        mu, sigma2, pi, bc_threshold=0.8, max_k=None
    )
    np.testing.assert_allclose(pi_m, [0.5, 0.5])
    np.testing.assert_allclose(mu_m, [[0.05], [10.0]])
    # within variance 1 + between variance 0.05**2
    np.testing.assert_allclose(s2_m, [[1.0025], [1.0]])


def test_merge_keeps_dissimilar_components():
    mu = np.array([[0.0], [5.0], [10.0]])
    sigma2 = np.ones((3, 1))
    pi = np.array([0.2, 0.3, 0.5])
    mu_m, s2_m, pi_m = ClusterAnalysis._merge_similar_gmm_components(
        mu, sigma2, pi, bc_threshold=0.8, max_k=None
    )
    np.testing.assert_allclose(mu_m, mu)
    np.testing.assert_allclose(s2_m, sigma2)
    np.testing.assert_allclose(pi_m, pi)


def test_merge_transitive_chain_forms_one_component():
    # 0 ~ 0.5 ~ 1.0 are pairwise-adjacent links (BC(0, 1) < 0.8 alone)
    mu = np.array([[0.0], [0.5], [1.0]])
    sigma2 = np.full((3, 1), 0.25)
    pi = np.array([1 / 3, 1 / 3, 1 / 3])
    d01 = ClusterAnalysis._compute_bhattacharyya_distance(
        mu[0], sigma2[0], mu[2], sigma2[2]
    )
    assert np.exp(-d01) < 0.8
    mu_m, _, pi_m = ClusterAnalysis._merge_similar_gmm_components(
        mu, sigma2, pi, bc_threshold=0.8, max_k=None
    )
    np.testing.assert_allclose(pi_m, [1.0])
    np.testing.assert_allclose(mu_m, [[0.5]])


def test_merge_agglomerative_cap_reduces_to_max_k():
    mu = np.array([[0.0], [1.0], [100.0]])
    sigma2 = np.full((3, 1), 0.01)
    pi = np.array([0.2, 0.3, 0.5])
    mu_m, s2_m, pi_m = ClusterAnalysis._merge_similar_gmm_components(
        mu, sigma2, pi, bc_threshold=0.99, max_k=2
    )
    assert len(pi_m) == 2
    order = np.argsort(mu_m[:, 0])
    np.testing.assert_allclose(pi_m[order], [0.5, 0.5])
    np.testing.assert_allclose(mu_m[order, 0], [0.6, 100.0])
    # 0.01 + weighted between-variance 0.4*0.36 + 0.6*0.16
    np.testing.assert_allclose(s2_m[order, 0], [0.01 + 0.24, 0.01])
    assert pi_m.sum() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# _build_gmm_distribution
# ---------------------------------------------------------------------------


def test_build_gmm_distribution_parameters(cluster_fakes):
    mu = np.array([[0.0, 1.0], [2.0, 3.0]])
    sigma2 = np.array([[4.0, 1.0], [0.25, 9.0]])
    pi = np.array([0.4, 0.6])
    dist_z, dist_z_y, logpy = ClusterAnalysis._build_gmm_distribution(mu, sigma2, pi)

    (mvn_call,) = cluster_fakes.mvn_calls
    np.testing.assert_allclose(mvn_call["loc"].numpy(), mu)
    np.testing.assert_allclose(mvn_call["scale_diag"].numpy(), np.sqrt(sigma2))

    (mix_params,) = cluster_fakes.mix_calls
    expected = np.concatenate(
        [np.log(pi + 1e-7)[:, None], mu, np.sqrt(sigma2)], axis=-1
    )
    np.testing.assert_allclose(mix_params.numpy(), expected, rtol=1e-6)
    np.testing.assert_allclose(logpy.numpy(), np.log(pi + 1e-7), rtol=1e-6)


# ---------------------------------------------------------------------------
# _remove_small_clusters / _label_small_clusters
# ---------------------------------------------------------------------------


def test_remove_small_clusters_reassigns_and_remaps():
    logpy_z = np.log(
        np.array(
            [
                [0.7, 0.1, 0.2],
                [0.7, 0.1, 0.2],
                [0.2, 0.5, 0.3],  # cluster 1, singleton -> next best is 2
                [0.1, 0.1, 0.8],
                [0.1, 0.1, 0.8],
            ]
        )
    )
    cluster = np.argmax(logpy_z, axis=1)
    out, cluster_final, labels = ClusterAnalysis._remove_small_clusters(
        logpy_z, cluster, min_n_obs=2
    )
    np.testing.assert_allclose(out, logpy_z[:, [0, 2]])
    np.testing.assert_array_equal(cluster_final, [0, 0, 1, 1, 1])
    assert labels == ["Cluster 000"] * 2 + ["Cluster 001"] * 3


def test_remove_small_clusters_noop_when_all_large():
    logpy_z = np.log(np.array([[0.9, 0.1], [0.8, 0.2], [0.1, 0.9], [0.3, 0.7]]))
    cluster = np.argmax(logpy_z, axis=1)
    out, cluster_final, labels = ClusterAnalysis._remove_small_clusters(
        logpy_z, cluster, min_n_obs=2
    )
    np.testing.assert_allclose(out, logpy_z)
    np.testing.assert_array_equal(cluster_final, [0, 0, 1, 1])


def test_remove_small_clusters_all_small_keeps_everything():
    logpy_z = np.log(np.array([[0.9, 0.1], [0.1, 0.9]]))
    cluster = np.array([0, 1])
    out, cluster_final, labels = ClusterAnalysis._remove_small_clusters(
        logpy_z, cluster, min_n_obs=5
    )
    np.testing.assert_allclose(out, logpy_z)
    np.testing.assert_array_equal(cluster_final, [0, 1])
    assert labels == ["Cluster 000", "Cluster 001"]


# regression: an empty cluster kept its logpy_z column when no cluster was
# removed, so the remapped labels no longer indexed the obsm columns
def test_remove_small_clusters_labels_index_logpy_columns_with_empty_cluster():
    # column 1 is never the argmax (empty cluster); nothing is small
    logpy_z = np.log(
        np.array(
            [[0.9, 0.05, 0.05], [0.9, 0.05, 0.05], [0.05, 0.05, 0.9], [0.05, 0.05, 0.9]]
        )
    )
    cluster = np.argmax(logpy_z, axis=1)
    out, cluster_final, _ = ClusterAnalysis._remove_small_clusters(
        logpy_z, cluster, min_n_obs=1
    )
    np.testing.assert_array_equal(cluster_final, [0, 0, 1, 1])
    np.testing.assert_array_equal(np.argmax(out, axis=1), cluster_final)
    np.testing.assert_allclose(out, logpy_z[:, [0, 2]])


def test_label_small_clusters():
    cluster = np.array([0, 0, 0, 3, 5, 5])
    labels = ClusterAnalysis._label_small_clusters(
        cluster, min_n_obs=2, small_cluster_label="NA"
    )
    assert labels == ["Cluster 000"] * 3 + ["NA"] + ["Cluster 005"] * 2


# ---------------------------------------------------------------------------
# _score_and_cluster
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("batch_size", [1, 4, 128])
def test_score_and_cluster_posterior_and_storage(cluster_fakes, batch_size):
    mdata = make_mdata(n_obs=6)
    analysis = ClusterAnalysis(mdata, MagicMock())
    params = ClusterAnalysis._extract_gmm_parameters(make_prior(**PRIOR_1D))
    dist_z, dist_z_y, logpy = ClusterAnalysis._build_gmm_distribution(
        params["mu"], params["sigma2"], params["pi"]
    )
    out = analysis._score_and_cluster(
        Z_1D, dist_z, dist_z_y, logpy.numpy(), "rna", "ck", "lk", 1, batch_size
    )
    np.testing.assert_allclose(out, two_cluster_posterior(Z_1D), atol=1e-5)
    np.testing.assert_allclose(logsumexp(out, axis=1), 0.0, atol=1e-5)
    assert list(mdata.mod["rna"].obs["ck"]) == ["Cluster 000"] * 3 + ["Cluster 001"] * 3
    np.testing.assert_allclose(mdata.mod["rna"].obsm["lk"], out)


def test_score_and_cluster_removes_small_cluster(cluster_fakes):
    mdata = make_mdata(n_obs=6)
    analysis = ClusterAnalysis(mdata, MagicMock())
    z = np.array([[-2.0], [-1.5], [-1.0], [-0.5], [-0.2], [2.0]], np.float32)
    params = ClusterAnalysis._extract_gmm_parameters(make_prior(**PRIOR_1D))
    dist_z, dist_z_y, logpy = ClusterAnalysis._build_gmm_distribution(
        params["mu"], params["sigma2"], params["pi"]
    )
    out = analysis._score_and_cluster(
        z, dist_z, dist_z_y, logpy.numpy(), "rna", "ck", "lk", 2, 128
    )
    assert out.shape == (6, 1)
    assert list(mdata.mod["rna"].obs["ck"]) == ["Cluster 000"] * 6


# ---------------------------------------------------------------------------
# compute_cluster_log_probability / _z_hat
# ---------------------------------------------------------------------------


def test_compute_cluster_log_probability(cluster_fakes):
    mdata = make_mdata(n_obs=6)
    comp = SimpleNamespace(z_prior_parameterizer=make_prior(**PRIOR_1D))
    model = FakeLatentModel({"c": comp}, keys=["c_z"])
    cluster_fakes.DataLoader.return_value = batches_of(Z_1D, 4)
    bec, dn = {"rna": ["batch"]}, {"rna": "dist"}

    out = ClusterAnalysis(mdata, model).compute_cluster_log_probability(
        "rna", "c", bec, dn, batch_size=4, min_n_obs=1
    )

    cluster_fakes.DataLoader.assert_called_once_with(mdata, 4, bec, dn)
    assert model.calls == [False, False]
    np.testing.assert_allclose(out, two_cluster_posterior(Z_1D), atol=1e-5)
    assert (
        list(mdata.mod["rna"].obs["cluster_c"])
        == ["Cluster 000"] * 3 + ["Cluster 001"] * 3
    )
    np.testing.assert_allclose(mdata.mod["rna"].obsm["logpy_z_c"], out)


def test_compute_cluster_log_probability_defaults_give_dataloader_mappings(
    cluster_fakes,
):
    # regression: T5 - the None defaults reached DataLoader, which calls
    # .get() on them. Empty mappings make DataLoader read adata.uns.
    mdata = make_mdata(n_obs=6)
    comp = SimpleNamespace(z_prior_parameterizer=make_prior(**PRIOR_1D))
    model = FakeLatentModel({"c": comp}, keys=["c_z"])
    cluster_fakes.DataLoader.return_value = batches_of(Z_1D, 128)
    ClusterAnalysis(mdata, model).compute_cluster_log_probability(
        "rna", "c", min_n_obs=1
    )
    args = cluster_fakes.DataLoader.call_args.args
    assert isinstance(args[2], Mapping) and args[2] == {}
    assert isinstance(args[3], Mapping) and args[3] == {}


def test_compute_cluster_log_probability_z_hat_uses_z_hat_prior(cluster_fakes):
    mdata = make_mdata(n_obs=6)
    # z prior is deliberately reversed so using it would flip the result
    comp = SimpleNamespace(
        is_root=False,
        z_hat_prior_parameterizer=make_prior(**PRIOR_1D),
        z_prior_parameterizer=make_prior([0.0, 0.0], [[1.0], [-1.0]], [[1.0], [1.0]]),
    )
    model = FakeLatentModel({"c": comp}, keys=["c_z_hat"])
    cluster_fakes.DataLoader.return_value = batches_of(Z_1D, 3)

    out = ClusterAnalysis(mdata, model).compute_cluster_log_probability_z_hat(
        "rna", "c", {}, {}, batch_size=3, min_n_obs=1
    )
    np.testing.assert_allclose(out, two_cluster_posterior(Z_1D), atol=1e-5)
    assert "cluster_c_integrated" in mdata.mod["rna"].obs
    np.testing.assert_allclose(mdata.mod["rna"].obsm["logpy_zhat_c_integrated"], out)


def test_compute_cluster_log_probability_z_hat_root_falls_back_to_z_prior(
    cluster_fakes,
):
    mdata = make_mdata(n_obs=6)
    comp = SimpleNamespace(
        is_root=True,
        z_hat_prior_parameterizer=None,
        z_prior_parameterizer=make_prior(**PRIOR_1D),
    )
    model = FakeLatentModel({"c": comp}, keys=["c_z_hat"])
    cluster_fakes.DataLoader.return_value = batches_of(Z_1D, 128)
    out = ClusterAnalysis(mdata, model).compute_cluster_log_probability_z_hat(
        "rna", "c", {}, {}, min_n_obs=1
    )
    np.testing.assert_allclose(out, two_cluster_posterior(Z_1D), atol=1e-5)


def test_compute_cluster_log_probability_z_hat_without_prior_raises(cluster_fakes):
    comp = SimpleNamespace(
        is_root=False, z_hat_prior_parameterizer=None, z_prior_parameterizer=MagicMock()
    )
    analysis = ClusterAnalysis(make_mdata(), FakeLatentModel({"c": comp}, keys=[]))
    with pytest.raises(ValueError, match="z_hat prior"):
        analysis.compute_cluster_log_probability_z_hat("rna", "c", {}, {})
    cluster_fakes.DataLoader.assert_not_called()


# ---------------------------------------------------------------------------
# _recombine_gmm_parameters / compute_integrated_cluster_log_probability
# ---------------------------------------------------------------------------


class FakeHierarchicalEncoder:
    """z_hat loc = z @ w_z (@ w_z2) + z_hat_conditional @ w_p + bias, and
    scale_diag = scale (+ z @ scale_w_z); records the inputs of each call."""

    def __init__(self, w_z, w_p=None, bias=0.0, scale=0.0, w_z2=None, scale_w_z=None):
        self.scale_w_z = (
            None if scale_w_z is None else tf.constant(scale_w_z, tf.float32)
        )
        self.w_z = tf.constant(w_z, tf.float32)
        self.w_z2 = None if w_z2 is None else tf.constant(w_z2, tf.float32)
        self.w_p = None if w_p is None else tf.constant(w_p, tf.float32)
        self.bias = bias
        self.scale = scale
        self.calls = []

    def __call__(self, inputs, training=False):
        self.calls.append((dict(inputs), training))
        loc = tf.matmul(inputs["z"], self.w_z)
        if self.w_z2 is not None:
            loc = tf.matmul(loc, self.w_z2)
        if self.w_p is not None:
            loc += tf.matmul(inputs["z_hat_conditional"], self.w_p)
        loc += self.bias
        scale = tf.zeros_like(loc) + self.scale
        if self.scale_w_z is not None:
            scale += tf.matmul(inputs["z"], self.scale_w_z)
        return tf.concat([loc, scale], axis=-1)


def gmm(pi, mu, sigma2):
    mu = np.asarray(mu, float)
    return {
        "pi": np.asarray(pi, float),
        "mu": mu,
        "sigma2": np.asarray(sigma2, float),
        "K": len(pi),
        "D": mu.shape[1],
    }


def test_recombine_gmm_parameters_hand_computed():
    parent = gmm([0.3, 0.7], [[1.0], [-1.0]], [[0.5], [2.0]])
    child = gmm([0.4, 0.6], [[2.0], [3.0]], [[1.0], [0.1]])
    # z_hat_loc = 3 * parent + 10 * child + 3.5, z_hat scale_diag = 0.5
    encoder = FakeHierarchicalEncoder(w_z=[[10.0]], w_p=[[3.0]], bias=3.5, scale=0.5)
    mu, s2, pi = ClusterAnalysis._recombine_gmm_parameters(child, [parent], encoder)

    exp_mu, exp_s2, exp_pi = [], [], []
    for p in range(2):  # product order: parent outer, child inner
        for c in range(2):
            exp_mu.append(3 * parent["mu"][p, 0] + 10 * child["mu"][c, 0] + 3.5)
            exp_s2.append(
                0.25 + 9 * parent["sigma2"][p, 0] + 100 * child["sigma2"][c, 0]
            )
            exp_pi.append(parent["pi"][p] * child["pi"][c])
    np.testing.assert_allclose(mu[:, 0], exp_mu, rtol=1e-6)
    np.testing.assert_allclose(s2[:, 0], exp_s2, rtol=1e-5)
    np.testing.assert_allclose(pi, exp_pi)
    assert pi.sum() == pytest.approx(1.0)

    # first call: the cluster means, in inference mode
    inputs, training = encoder.calls[0]
    assert training is False
    np.testing.assert_allclose(inputs["z"].numpy()[:, 0], [2.0, 3.0, 2.0, 3.0])
    np.testing.assert_allclose(
        inputs["z_hat_conditional"].numpy()[:, 0], [1.0, 1.0, -1.0, -1.0]
    )


def test_recombine_gmm_parameters_without_parents():
    child = gmm([1.0], [[2.0]], [[1.0]])
    encoder = FakeHierarchicalEncoder(w_z=[[10.0]], bias=1.0)
    mu, s2, pi = ClusterAnalysis._recombine_gmm_parameters(child, [], encoder)
    np.testing.assert_allclose(mu, [[21.0]])
    np.testing.assert_allclose(s2, [[100.0]])
    np.testing.assert_allclose(pi, [1.0])
    assert "z_hat_conditional" not in encoder.calls[0][0]


# regression: the child variance used (W_r**2) @ (W_b**2) instead of
# (W_r @ W_b)**2, dropping cross terms of the composed linear map
def test_recombine_gmm_parameters_child_variance_of_composed_map():
    # z (1-D, var 1) -> r: [z, z] -> b: z + z = 2 z  => var 4
    child = gmm([1.0], [[0.0]], [[1.0]])
    encoder = FakeHierarchicalEncoder(w_z=[[1.0, 1.0]], w_z2=[[1.0], [1.0]])
    _, s2, _ = ClusterAnalysis._recombine_gmm_parameters(child, [], encoder)
    np.testing.assert_allclose(s2, [[4.0]])


def test_recombine_gmm_parameters_input_dependent_scale():
    # z ~ N(m, s2) per dim, scale = c + z @ B: E[scale**2] is quadratic, so
    # the sigma points give it exactly: (c + m @ B)**2 + (B**2).T @ s2
    child = gmm([1.0], [[1.0, -2.0]], [[0.5, 2.0]])
    b = np.array([[0.3, 0.1], [-0.2, 0.4]])
    encoder = FakeHierarchicalEncoder(w_z=np.eye(2), scale=1.5, scale_w_z=b)
    mu, s2, _ = ClusterAnalysis._recombine_gmm_parameters(child, [], encoder)
    m, v = child["mu"][0], child["sigma2"][0]
    expected = v + (1.5 + m @ b) ** 2 + (b**2).T @ v
    np.testing.assert_allclose(mu, [m], rtol=1e-6)
    np.testing.assert_allclose(s2, [expected], rtol=1e-5)
    # sigma points: mu ± sqrt(d * sigma2) along each input dimension
    sigma_points = encoder.calls[1][0]["z"].numpy()
    np.testing.assert_allclose(
        sigma_points,
        [m + [1.0, 0.0], m + [0.0, 2.0], m - [1.0, 0.0], m - [0.0, 2.0]],
        rtol=1e-6,
    )


def test_recombine_gmm_parameters_multi_dimensional_marginal():
    # loc = z @ A, Var = (A**2).T @ sigma2 per output dim
    a = np.array([[1.0, -2.0], [0.5, 3.0]])
    child = gmm([1.0], [[1.0, -1.0]], [[2.0, 0.5]])
    encoder = FakeHierarchicalEncoder(w_z=a, scale=0.1)
    mu, s2, _ = ClusterAnalysis._recombine_gmm_parameters(child, [], encoder)
    np.testing.assert_allclose(mu, [child["mu"][0] @ a], rtol=1e-6)
    np.testing.assert_allclose(s2, [0.01 + (a**2).T @ child["sigma2"][0]], rtol=1e-5)


def hierarchical_component(**overrides):
    attributes = dict(
        is_root=False,
        conditioned_on_z=[],
        conditioned_on_z_hat=["p"],
        reparameterize_z_hat=True,
        z_hat_prior_parameterizer=None,
    )
    attributes.update(overrides)
    return SimpleNamespace(**attributes)


def root_component(z_prior_parameterizer):
    return SimpleNamespace(is_root=True, z_prior_parameterizer=z_prior_parameterizer)


def test_z_hat_gmm_parameters_of_root_is_z_prior():
    root = root_component(make_prior([0.0, 0.0], [[-1.0], [1.0]], [[2.0], [3.0]]))
    analysis = ClusterAnalysis(make_mdata(), SimpleNamespace(components={"p": root}))
    params = analysis._z_hat_gmm_parameters("p")
    np.testing.assert_allclose(params["mu"], [[-1.0], [1.0]])
    np.testing.assert_allclose(params["sigma2"], [[4.0], [9.0]])


def test_z_hat_gmm_parameters_uses_learned_z_hat_prior():
    parent = hierarchical_component(
        conditioned_on_z_hat=["g"],
        z_prior_parameterizer=make_prior([0.0], [[100.0]], [[1.0]]),
        z_hat_prior_parameterizer=make_prior([0.0], [[5.0]], [[0.5]]),
    )
    analysis = ClusterAnalysis(make_mdata(), SimpleNamespace(components={"p": parent}))
    params = analysis._z_hat_gmm_parameters("p")
    np.testing.assert_allclose(params["mu"], [[5.0]])
    np.testing.assert_allclose(params["sigma2"], [[0.25]])


def test_z_hat_gmm_parameters_integrates_non_root_parent_recursively():
    # g (root, z_hat = z) -> p (z_hat = 2 z_p + z_hat_g) -> c
    grandparent = root_component(
        make_prior([0.0, 0.0], [[-1.0], [1.0]], [[1.0], [1.0]])
    )
    parent = hierarchical_component(
        conditioned_on_z_hat=["g"],
        z_prior_parameterizer=make_prior([0.0], [[3.0]], [[0.5]]),
        hierarchical_encoder=FakeHierarchicalEncoder(w_z=[[2.0]], w_p=[[1.0]]),
    )
    analysis = ClusterAnalysis(
        make_mdata(), SimpleNamespace(components={"g": grandparent, "p": parent})
    )
    params = analysis._z_hat_gmm_parameters("p")
    # loc 2 * 3 + (-1 | 1); var 4 * 0.25 + 1
    np.testing.assert_allclose(params["mu"], [[5.0], [7.0]], rtol=1e-6)
    np.testing.assert_allclose(params["sigma2"], [[2.0], [2.0]], rtol=1e-6)
    np.testing.assert_allclose(params["pi"], [0.5, 0.5], rtol=1e-6)
    assert (params["K"], params["D"]) == (2, 1)


def test_integrated_gmm_parameters_feed_parent_z_hat_gmm():
    # the child encoder receives the parent's z_hat clusters, not its z prior
    parent = hierarchical_component(
        conditioned_on_z_hat=["g"],
        z_prior_parameterizer=make_prior([0.0], [[100.0]], [[1.0]]),
        z_hat_prior_parameterizer=make_prior([0.0], [[5.0]], [[0.5]]),
    )
    encoder = FakeHierarchicalEncoder(w_z=[[1.0]], w_p=[[1.0]])
    child = hierarchical_component(
        z_prior_parameterizer=make_prior([0.0], [[1.0]], [[1.0]]),
        hierarchical_encoder=encoder,
    )
    analysis = ClusterAnalysis(
        make_mdata(), SimpleNamespace(components={"p": parent, "c": child})
    )
    mu, s2, _ = analysis._integrated_gmm_parameters("c")
    np.testing.assert_allclose(
        encoder.calls[0][0]["z_hat_conditional"].numpy(), [[5.0]]
    )
    np.testing.assert_allclose(mu, [[6.0]])
    np.testing.assert_allclose(s2, [[1.25]])


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"conditioned_on_z_hat": []}, "no conditioned_on_z_hat parents"),
        ({"reparameterize_z_hat": False}, "reparameterize_z_hat"),
        ({"conditioned_on_z": ["q"]}, "conditioned_on_z parents"),
    ],
)
def test_compute_integrated_cluster_log_probability_unsupported_raises(
    overrides, match
):
    comp = hierarchical_component(**overrides)
    analysis = ClusterAnalysis(make_mdata(), SimpleNamespace(components={"c": comp}))
    with pytest.raises(ValueError, match=match):
        analysis.compute_integrated_cluster_log_probability("rna", "c")


def test_compute_integrated_cluster_log_probability_end_to_end(cluster_fakes):
    # parent: one cluster at 0; child: clusters at -1 / +1; identity maps
    parent = root_component(make_prior([0.0], [[0.0]], [[1e-3]]))
    child = hierarchical_component(
        z_prior_parameterizer=make_prior(**PRIOR_1D),
        hierarchical_encoder=FakeHierarchicalEncoder(w_z=[[1.0]], w_p=[[1.0]]),
    )
    mdata = make_mdata(n_obs=6)
    mdata.mod["rna"].obsm["z_hat_c"] = Z_1D
    analysis = ClusterAnalysis(
        mdata, SimpleNamespace(components={"p": parent, "c": child})
    )

    out = analysis.compute_integrated_cluster_log_probability("rna", "c", min_n_obs=1)

    # sigma2 = 1e-6 (parent) + 1 (child): effectively PRIOR_1D
    np.testing.assert_allclose(out, two_cluster_posterior(Z_1D), atol=1e-4)
    assert list(mdata.mod["rna"].obs["cluster_c_integrated"]) == (
        ["Cluster 000"] * 3 + ["Cluster 001"] * 3
    )
    np.testing.assert_allclose(mdata.mod["rna"].obsm["logpy_zhat_c_integrated"], out)


# ---------------------------------------------------------------------------
# compute_neighbors_with_same_annotations
# ---------------------------------------------------------------------------


def knn_mdata():
    mdata = make_mdata(n_obs=4, obs={"cl": ["A", "A", "B", "B"]})
    mdata.mod["rna"].obsm["rep"] = np.array([[0.0], [1.0], [3.0], [5.5]])
    return mdata


def test_neighbors_missing_cluster_column_raises():
    with pytest.raises(KeyError, match="missing"):
        ClusterAnalysis(
            knn_mdata(), MagicMock()
        ).compute_neighbors_with_same_annotations("rna", "missing", "rep", 1)


def test_neighbors_hand_computed():
    result = ClusterAnalysis(
        knn_mdata(), MagicMock()
    ).compute_neighbors_with_same_annotations("rna", "cl", "rep", [1, 2])
    assert list(result["Number of Neighbors"]) == [1] * 4 + [2] * 4
    assert list(result["Cluster"]) == ["A", "A", "B", "B"] * 2
    # k=1: 0->1, 1->0, 3->1, 5.5->3 ; k=2: every cell has one same / one other
    np.testing.assert_allclose(
        result["% of KNN Cells with the Same Cluster"], [1, 1, 0, 1, 0.5, 0.5, 0.5, 0.5]
    )
    np.testing.assert_allclose(
        result["KNN Cells with the Same Cluster"], [1, 1, 0, 1, 1, 1, 1, 1]
    )
    assert list(result["Number of Cells"]) == [2] * 8
    # (p - 0.5).clip(0) / 0.5
    np.testing.assert_allclose(
        result["Cell Enrichment Score"], [1, 1, 0, 1, 0, 0, 0, 0]
    )


def test_neighbors_int_and_ndarray_rep():
    mdata = knn_mdata()
    rep = np.array([[0.0], [1.0], [3.0], [5.5]])
    result = ClusterAnalysis(
        mdata, MagicMock()
    ).compute_neighbors_with_same_annotations("rna", "cl", rep, 1)
    np.testing.assert_allclose(
        result["% of KNN Cells with the Same Cluster"], [1, 1, 0, 1]
    )
    np.testing.assert_array_equal(mdata.mod["rna"].obsm["z_custom"], rep)


def test_neighbors_float_k_warns_and_casts():
    with pytest.warns(RuntimeWarning, match="transform float to int"):
        result = ClusterAnalysis(
            knn_mdata(), MagicMock()
        ).compute_neighbors_with_same_annotations("rna", "cl", "rep", [1.0])
    assert list(result["Number of Neighbors"]) == [1] * 4


def test_neighbors_excludes_self_with_duplicate_points():
    mdata = make_mdata(n_obs=4, obs={"cl": ["A", "A", "B", "B"]})
    mdata.mod["rna"].obsm["rep"] = np.array([[0.0], [0.0], [5.0], [5.0]])
    result = ClusterAnalysis(
        mdata, MagicMock()
    ).compute_neighbors_with_same_annotations("rna", "cl", "rep", 1)
    np.testing.assert_allclose(
        result["% of KNN Cells with the Same Cluster"], [1, 1, 1, 1]
    )


# ---------------------------------------------------------------------------
# compute_contingency_matrix / compute_classification_metrics
# ---------------------------------------------------------------------------


def metrics_mdata():
    return make_mdata(
        n_obs=4, obs={"a": ["x", "x", "y", "y"], "b": ["p", "p", "p", "q"]}
    )


def test_contingency_matrix():
    cm = ClusterAnalysis(metrics_mdata(), MagicMock()).compute_contingency_matrix(
        "rna", "a", "b"
    )
    expected = pd.DataFrame([[2, 0], [1, 1]], index=["x", "y"], columns=["p", "q"])
    pd.testing.assert_frame_equal(cm, expected, check_dtype=False)


def test_classification_metrics_hand_computed():
    result = ClusterAnalysis(
        metrics_mdata(), MagicMock()
    ).compute_classification_metrics("rna", "a", "b")
    assert list(result.columns) == [
        "Cluster A",
        "Cluster B",
        "F1 Score",
        "Accuracy",
        "Sensitivity",
        "Specificity",
        "Precision",
    ]
    assert list(zip(result["Cluster A"], result["Cluster B"])) == [
        ("x", "p"),
        ("x", "q"),
        ("y", "p"),
        ("y", "q"),
    ]
    row = result.iloc[0]  # (x, p): TP 2, FP 1, FN 0, TN 1
    assert row["Sensitivity"] == pytest.approx(1.0)
    assert row["Specificity"] == pytest.approx(0.5)
    assert row["Precision"] == pytest.approx(2 / 3)
    assert row["Accuracy"] == pytest.approx(0.75)
    assert row["F1 Score"] == pytest.approx(2 * (2 / 3) / (2 / 3 + 1 + 1e-3))
    row = result.iloc[3]  # (y, q): TP 1, FP 0, FN 1, TN 2
    assert row["Sensitivity"] == pytest.approx(0.5)
    assert row["Specificity"] == pytest.approx(1.0)
    assert row["Precision"] == pytest.approx(1.0)
    assert row["Accuracy"] == pytest.approx(0.75)
