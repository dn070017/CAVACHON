import numpy as np
import pytest
import tensorflow as tf

from cavachon.layers.parameterizers import (
    MixtureMultivariateNormalDiagParameterizerLayer as Mixture,
)


@pytest.mark.parametrize("unit_variance", [False, True])
def test_output_shape(unit_variance):
    layer = Mixture(event_dims=2, n_components=3, unit_variance=unit_variance)
    out = layer(tf.random.normal((4, 5), seed=5))
    assert tuple(out.shape) == (4, 3, 5)


def test_initial_values_input_independent():
    layer = Mixture(event_dims=2, n_components=4)
    out = layer(tf.random.normal((3, 6), seed=6)).numpy()
    # logits init to 0; means = grid; scale = softplus(0.02) + 1e-7
    np.testing.assert_allclose(out[..., 0], 0.0)
    expected_means = np.array([[-1.5, -1.5], [-1.5, 1.5], [1.5, -1.5], [1.5, 1.5]])
    for b in range(3):
        np.testing.assert_allclose(out[b, :, 1:3], expected_means, atol=1e-6)
    np.testing.assert_allclose(out[..., 3:], np.logaddexp(0.0, 0.02) + 1e-7, rtol=1e-6)


def test_unit_variance_scale_is_one_and_no_scale_weights():
    layer = Mixture(event_dims=3, n_components=2, unit_variance=True)
    out = layer(tf.ones((2, 1))).numpy()
    np.testing.assert_allclose(out[..., 4:], 1.0)
    assert not hasattr(layer, "scale_diag_weight")
    # logits weight+bias, loc weight+bias per component
    assert len(layer.trainable_weights) == 2 + 2 * 2


def test_trainable_weight_count_non_unit():
    layer = Mixture(event_dims=2, n_components=3)
    layer(tf.ones((1, 1)))
    assert len(layer.trainable_weights) == 2 + 4 * 3


@pytest.mark.parametrize(
    "k,d,expected",
    [
        (1, 2, [[0.0, 0.0]]),
        (2, 1, [[-1.5], [1.5]]),
        (5, 1, [[-1.5], [-0.75], [0.0], [0.75], [1.5]]),
        (3, 2, [[-1.5, -1.5], [-1.5, 1.5], [1.5, -1.5]]),
    ],
)
def test_grid_positions(k, d, expected):
    coords = Mixture(event_dims=d, n_components=k)._make_grid_positions()
    np.testing.assert_allclose(coords, np.array(expected, dtype=np.float32))


@pytest.mark.parametrize("k,d", [(10, 3), (7, 2), (2, 8)])
def test_grid_positions_distinct_and_bounded(k, d):
    coords = Mixture(event_dims=d, n_components=k)._make_grid_positions(radius=2.0)
    assert coords.shape == (k, d)
    assert len({tuple(c) for c in coords}) == k
    assert np.abs(coords).max() <= 2.0 + 1e-6


@pytest.mark.xfail(
    strict=True,
    reason="BUG: mixture_..._parameterizer_layer.py:50 int(np.ceil(K ** (1/D))) suffers float error "
    "(3125**(1/5)=5.000000000000001 -> side 6), so a perfect-power grid is no longer centred/symmetric",
)
def test_grid_perfect_power_is_centred():
    coords = Mixture(event_dims=5, n_components=3125)._make_grid_positions()
    np.testing.assert_allclose(coords.mean(0), 0.0, atol=1e-6)
    assert coords.min() == -1.5 and coords.max() == 1.5


def test_parameters_property_shape():
    layer = Mixture(event_dims=2, n_components=3)
    p = layer.parameters
    assert tuple(p.shape) == (3, 5)
    np.testing.assert_allclose(p.numpy(), layer(tf.ones((1, 1))).numpy()[0])


def test_logits_depend_on_input_after_weight_change():
    layer = Mixture(event_dims=1, n_components=2)
    layer.build((None, 2))
    layer.logits_weight.assign(np.array([[1.0, -1.0], [0.0, 0.0]], np.float32))
    out = layer(tf.constant([[2.0, 0.0]])).numpy()
    np.testing.assert_allclose(out[0, :, 0], [2.0, -2.0])


def test_get_config_contains_init_args():
    kwargs = dict(event_dims=2, n_components=4, unit_variance=True)
    cfg = Mixture(**kwargs).get_config()
    for k, v in kwargs.items():
        assert cfg[k] == v


def test_from_config_round_trip():
    # regression: __init__ lacked **kwargs, so from_config (trainable/dtype) raised TypeError
    clone = Mixture.from_config(Mixture(**dict(event_dims=2, n_components=4, unit_variance=True)).get_config())
    assert (clone.event_dims, clone.n_components, clone.unit_variance) == (2, 4, True)
