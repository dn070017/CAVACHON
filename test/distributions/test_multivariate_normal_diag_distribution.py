import numpy as np
import pytest
import tensorflow as tf
import tensorflow_probability as tfp

from cavachon.distributions.multivariate_normal_diag_distribution import (
    MultivariateNormalDiagDistribution,
)


def _params(shape=(2, 3)):
    rng = np.random.default_rng(0)
    loc = rng.normal(size=shape).astype(np.float32)
    scale = rng.uniform(0.5, 2.0, size=shape).astype(np.float32)
    return loc, scale


def test_from_tensor_splits_loc_and_scale():
    loc, scale = _params()
    d = MultivariateNormalDiagDistribution.from_parameterizer_output(
        tf.concat([loc, scale], -1)
    )
    assert isinstance(d, MultivariateNormalDiagDistribution)
    assert d.batch_shape == (2,)
    assert d.event_shape == (3,)
    np.testing.assert_allclose(d.mean(), loc)
    np.testing.assert_allclose(d.stddev(), scale, rtol=1e-6)


def test_rank3_batch_shape():
    loc, scale = _params((3, 5, 2))
    d = MultivariateNormalDiagDistribution.from_parameterizer_output(
        tf.concat([loc, scale], -1)
    )
    assert d.batch_shape == (3, 5)
    assert d.event_shape == (2,)


def test_from_mapping_matches_tensor_and_reference():
    loc, scale = _params()
    a = MultivariateNormalDiagDistribution.from_parameterizer_output(
        tf.concat([loc, scale], -1)
    )
    b = MultivariateNormalDiagDistribution.from_parameterizer_output(
        {"loc": tf.constant(loc), "scale_diag": tf.constant(scale)}
    )
    ref = tfp.distributions.MultivariateNormalDiag(loc=loc, scale_diag=scale)
    x = tf.zeros((2, 3))
    np.testing.assert_allclose(a.log_prob(x), ref.log_prob(x), rtol=1e-6)
    np.testing.assert_allclose(b.log_prob(x), ref.log_prob(x), rtol=1e-6)
    # numpy closed form
    np_ref = np.sum(
        -0.5 * np.log(2 * np.pi) - np.log(scale) - 0.5 * (loc / scale) ** 2, -1
    )
    np.testing.assert_allclose(a.log_prob(x), np_ref, rtol=1e-5)


def test_kwargs_forwarded():
    loc, scale = _params()
    d = MultivariateNormalDiagDistribution.from_parameterizer_output(
        tf.concat([loc, scale], -1), name="foo", validate_args=True
    )
    assert d.name == "foo"
    assert d.validate_args


def test_odd_last_dim_raises():
    with pytest.raises(Exception):
        MultivariateNormalDiagDistribution.from_parameterizer_output(tf.ones((2, 3)))


def test_float64_dtype_preserved():
    d = MultivariateNormalDiagDistribution.from_parameterizer_output(
        tf.ones((2, 4), dtype=tf.float64)
    )
    assert d.dtype == tf.float64


# regression: non-Mapping, non-tf.Tensor params (numpy, tf.Variable) used to
# fall through both isinstance branches and raise UnboundLocalError.
@pytest.mark.parametrize(
    "make_params",
    [lambda: np.ones((2, 4), np.float32), lambda: tf.Variable(tf.ones((2, 4)))],
    ids=["numpy", "tf.Variable"],
)
def test_non_tensor_input_accepted(make_params):
    d = MultivariateNormalDiagDistribution.from_parameterizer_output(make_params())
    assert d.batch_shape == (2,)
    assert d.event_shape == (2,)
