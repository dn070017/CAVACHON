from collections.abc import Mapping
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import tensorflow as tf

from cavachon.tools.attribution_analysis import AttributionAnalysis

from .conftest import FakeAttrModel, FakeComponentNetwork, make_mdata, set_dataset

N_VARS = 4
ENC_W = np.array(
    [[0.5, -1.0], [1.0, 0.0], [0.0, 0.25], [-0.5, 1.0]], np.float32
)  # (n_vars, latent 2)
DEC_A = np.array(
    [[1.0, -2.0, 0.5, 3.0], [0.0, 1.0, -1.0, 2.0]], np.float32
)  # (2, n_vars)
X = np.array(
    [[1.0, 0.0, 2.0, 1.0], [0.0, 3.0, 1.0, 0.0], [2.0, 1.0, 0.0, 1.0]], np.float32
)


def linear_decoder(z_hat):
    return tf.matmul(z_hat, tf.constant(DEC_A))


def single_component_setup(attr_fakes, decoder_fn=linear_decoder):
    mdata = make_mdata(n_obs=3, n_vars=N_VARS)
    net = FakeComponentNetwork(ENC_W, decoder_fn)
    model = FakeAttrModel(
        [SimpleNamespace(name="c", modality_names=["rna"])], {"c": net}
    )
    set_dataset(attr_fakes.DataLoader, X)
    return AttributionAnalysis(mdata, model, {}, {}), model, net


# ---------------------------------------------------------------------------
# constructor
# ---------------------------------------------------------------------------


def test_constructor_builds_single_sample_dataloader(attr_fakes):
    mdata = make_mdata()
    bec, dn, enc = {"rna": ["b"]}, {"rna": "d"}, {"rna": {}}
    analysis = AttributionAnalysis(mdata, MagicMock(), bec, dn, enc)
    attr_fakes.DataLoader.assert_called_once_with(mdata, 1, bec, dn, enc)
    assert analysis.dataloader is attr_fakes.DataLoader.return_value
    assert analysis.mdata is mdata


def test_constructor_defaults_give_dataloader_mappings(attr_fakes):
    # regression: T5 - the None defaults reached DataLoader, which calls
    # .get() on them. Empty mappings make DataLoader read adata.uns.
    AttributionAnalysis(make_mdata(), MagicMock())
    args = attr_fakes.DataLoader.call_args.args
    assert isinstance(args[2], Mapping) and args[2] == {}
    assert isinstance(args[3], Mapping) and args[3] == {}


# ---------------------------------------------------------------------------
# compute_attribution_target_batch
# ---------------------------------------------------------------------------


def test_attribution_target_uses_sampled_z(attr_fakes):
    analysis, model, net = single_component_setup(attr_fakes)
    batch = {"rna_matrix": tf.constant(X)}
    out = analysis.compute_attribution_target_batch(batch, "c", "rna", "c")
    np.testing.assert_allclose(out.numpy(), X @ ENC_W @ DEC_A, rtol=1e-5)

    kwargs = attr_fakes.Model.prepare_component_inputs.call_args.kwargs
    assert kwargs["target_component"] == "c"
    assert kwargs["components"] is model.components
    attr_fakes.Component.prepare_preprocessor_inputs.assert_called_once()
    assert attr_fakes.Component.prepare_preprocessor_inputs.call_args.args[1] == ["rna"]
    assert (
        attr_fakes.Component.prepare_decoder_inputs.call_args.kwargs["modality_name"]
        == "rna"
    )


def test_attribution_target_replaces_z_variable(attr_fakes):
    analysis, _, _ = single_component_setup(attr_fakes)
    batch = {"rna_matrix": tf.constant(X)}
    z_var = tf.constant([[1.0, 2.0], [0.0, -1.0], [3.0, 0.5]])
    out = analysis.compute_attribution_target_batch(
        batch, "c", "rna", "c", z_variable=z_var
    )
    np.testing.assert_allclose(out.numpy(), z_var.numpy() @ DEC_A, rtol=1e-6)


def two_component_setup(attr_fakes, decoder_fn=linear_decoder):
    mdata = make_mdata(n_obs=3, n_vars=N_VARS)
    parent = FakeComponentNetwork(ENC_W, decoder_fn)
    child = FakeComponentNetwork(2 * ENC_W, decoder_fn, parents=["p"])
    configs = [
        SimpleNamespace(name="p", modality_names=["rna"]),
        SimpleNamespace(name="c", modality_names=["rna"]),
    ]
    model = FakeAttrModel(configs, {"p": parent, "c": child})
    set_dataset(attr_fakes.DataLoader, X)
    return AttributionAnalysis(mdata, model, {}, {}), parent, child


def test_attribution_target_propagates_conditionals(attr_fakes):
    analysis, parent, child = two_component_setup(attr_fakes)
    batch = {"rna_matrix": tf.constant(X)}
    z_var = tf.constant(np.ones((3, 2), np.float32))
    out = analysis.compute_attribution_target_batch(
        batch, "c", "rna", "p", z_variable=z_var
    )
    # child z_hat = child z + parent z_hat (= z_variable)
    expected = (X @ (2 * ENC_W) + 1.0) @ DEC_A
    np.testing.assert_allclose(out.numpy(), expected, rtol=1e-5)

    calls = attr_fakes.Model.prepare_component_inputs.call_args_list
    assert [c.kwargs["target_component"] for c in calls] == ["p", "c"]
    np.testing.assert_allclose(calls[1].kwargs["z_hat_conditional"]["p"].numpy(), z_var)


def test_attribution_target_stops_at_requested_component(attr_fakes):
    analysis, parent, child = two_component_setup(attr_fakes)
    out = analysis.compute_attribution_target_batch(
        {"rna_matrix": tf.constant(X)}, "p", "rna", "p"
    )
    np.testing.assert_allclose(out.numpy(), X @ ENC_W @ DEC_A, rtol=1e-5)
    assert len(parent.hier_calls) == 1
    assert child.hier_calls == []


# regression: an unknown component fell through the loop and returned None
def test_attribution_target_unknown_component_raises(attr_fakes):
    analysis, _, _ = single_component_setup(attr_fakes)
    with pytest.raises(KeyError, match="'missing' is not a component"):
        analysis.compute_attribution_target_batch(
            {"rna_matrix": tf.constant(X)}, "missing", "rna", "c"
        )


# ---------------------------------------------------------------------------
# compute_integrated_gradient
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason="BUG: T16 with_respect_to that is not upstream of component has no "
    "gradient; tape.gradient returns None and .numpy() raises AttributeError",
)
def test_integrated_gradient_with_respect_to_not_upstream_raises(attr_fakes):
    analysis, _, _ = two_component_setup(attr_fakes)
    # decode the parent, attribute to the child (downstream of the parent)
    with pytest.raises(ValueError, match="with_respect_to"):
        analysis.compute_integrated_gradient("p", "rna", "c", steps=2)


def test_integrated_gradient_shape_and_selection(attr_fakes):
    analysis, _, _ = single_component_setup(attr_fakes)
    full = analysis.compute_integrated_gradient("c", "rna", "c", steps=4)
    assert tuple(full.shape) == (3, 2, N_VARS)

    sel = analysis.compute_integrated_gradient(
        "c", "rna", "c", steps=4, selected_variables=["gene3", "gene0"]
    )
    assert tuple(sel.shape) == (3, 2, 2)
    np.testing.assert_allclose(sel.numpy(), full.numpy()[..., [3, 0]], rtol=1e-6)


def test_integrated_gradient_batch_size_invariant(attr_fakes):
    analysis, _, _ = single_component_setup(attr_fakes)
    a = analysis.compute_integrated_gradient("c", "rna", "c", steps=3, batch_size=1)
    b = analysis.compute_integrated_gradient("c", "rna", "c", steps=3, batch_size=128)
    np.testing.assert_allclose(a.numpy(), b.numpy(), rtol=1e-6)


def test_integrated_gradient_linear_decoder_is_proportional_to_z_times_weight(
    attr_fakes,
):
    # for a linear map every attribution must be c * z_d * A[d, j] with one c
    analysis, _, _ = single_component_setup(attr_fakes)
    ig = analysis.compute_integrated_gradient("c", "rna", "c", steps=10).numpy()
    reference = (X @ ENC_W)[:, :, None] * DEC_A[None, :, :]
    mask = np.abs(reference) > 1e-6
    ratio = ig[mask] / reference[mask]
    np.testing.assert_allclose(ratio, ratio[0], rtol=1e-5)
    np.testing.assert_allclose(ig[~mask], 0.0, atol=1e-6)


# regression: the gradient was taken w.r.t. z of f(alpha * z), adding an extra
# alpha factor; completeness sum_d IG = f(z) - f(0) failed (0.45x for steps=10)
def test_integrated_gradient_completeness_linear_decoder(attr_fakes):
    analysis, _, _ = single_component_setup(attr_fakes)
    ig = analysis.compute_integrated_gradient("c", "rna", "c", steps=10).numpy()
    z = X @ ENC_W
    np.testing.assert_allclose(ig.sum(axis=1), z @ DEC_A, rtol=1e-4, atol=1e-5)


def test_integrated_gradient_completeness_nonlinear_decoder(attr_fakes):
    # f = z**2 @ A: the left Riemann sum gives (steps - 1) / steps of
    # f(z) - f(0) (with the extra alpha it was about 2/3)
    analysis, _, _ = single_component_setup(
        attr_fakes, decoder_fn=lambda z: tf.square(z) @ tf.constant(DEC_A)
    )
    steps = 50
    ig = analysis.compute_integrated_gradient("c", "rna", "c", steps=steps).numpy()
    z = X @ ENC_W
    expected = (steps - 1) / steps * (z**2 @ DEC_A)
    np.testing.assert_allclose(ig.sum(axis=1), expected, rtol=1e-4, atol=1e-5)


def test_integrated_gradient_baseline_zeroes_only_with_respect_to(attr_fakes):
    # f = (z_c + z_p)**2 @ A, attributed to the parent: the child keeps its
    # z, so sum_d IG = sum_alpha 2 (z_c + alpha z_p) z_p / steps
    #                = 2 z_c z_p + (steps - 1) / steps * z_p**2
    # (-> f(z) - f(z_c, 0)); scaling z_c by alpha too gave z_c z_p + ...
    analysis, _, _ = two_component_setup(
        attr_fakes, decoder_fn=lambda z: tf.square(z) @ tf.constant(DEC_A)
    )
    steps = 10
    ig = analysis.compute_integrated_gradient("c", "rna", "p", steps=steps).numpy()
    z_p = X @ ENC_W
    z_c = X @ (2 * ENC_W)
    expected = (2 * z_c * z_p + (steps - 1) / steps * z_p**2) @ DEC_A
    np.testing.assert_allclose(ig.sum(axis=1), expected, rtol=1e-4, atol=1e-5)


def test_integrated_gradient_zero_latent_gives_zero(attr_fakes):
    mdata = make_mdata(n_obs=2, n_vars=N_VARS)
    net = FakeComponentNetwork(ENC_W, lambda z: tf.square(z) @ tf.constant(DEC_A))
    model = FakeAttrModel(
        [SimpleNamespace(name="c", modality_names=["rna"])], {"c": net}
    )
    set_dataset(attr_fakes.DataLoader, np.zeros((2, N_VARS), np.float32))
    ig = AttributionAnalysis(mdata, model, {}, {}).compute_integrated_gradient(
        "c", "rna", "c", steps=5
    )
    np.testing.assert_allclose(ig.numpy(), 0.0)
