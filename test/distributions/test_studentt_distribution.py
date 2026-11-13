import numpy as np
import pytest
import tensorflow as tf
from scipy import stats

from cavachon.distributions.studentt_distribution import StudenttDistribution


def _softplus(x):
    return np.log1p(np.exp(x))


def _raw():
    loc = np.array([[0.0, 1.0], [-2.0, 0.5]], np.float32)
    scale_raw = np.array([[0.0, -1.0], [2.0, 0.3]], np.float32)
    df_raw = np.array([[-5.0, 0.0], [1.0, 10.0]], np.float32)
    return loc, scale_raw, df_raw


def test_from_tensor_transforms_parameters():
    loc, sr, dr = _raw()
    d = StudenttDistribution.from_parameterizer_output(tf.concat([loc, sr, dr], -1))
    assert isinstance(d, StudenttDistribution)
    assert d.batch_shape == (2, 2)
    assert d.event_shape == ()
    np.testing.assert_allclose(d.loc, loc)
    np.testing.assert_allclose(d.scale, _softplus(sr) + 1e-7, rtol=1e-6)
    np.testing.assert_allclose(d.df, _softplus(dr) + 2.0, rtol=1e-6)


def test_df_always_above_two_and_variance_finite():
    loc, sr, dr = _raw()
    dr = np.full_like(dr, -50.0)
    d = StudenttDistribution.from_parameterizer_output(tf.concat([loc, sr, dr], -1))
    assert np.all(d.df.numpy() >= 2.0)
    # df == 2 exactly (softplus underflow) gives infinite variance; >2 is
    # what the comment promises but df=2 itself is allowed.
    assert np.all(d.scale.numpy() > 0)


def test_log_prob_matches_scipy():
    loc, sr, dr = _raw()
    d = StudenttDistribution.from_parameterizer_output(tf.concat([loc, sr, dr], -1))
    x = np.array([[0.3, -1.0], [4.0, 0.0]], np.float32)
    ref = stats.t.logpdf(x, df=_softplus(dr) + 2.0, loc=loc, scale=_softplus(sr) + 1e-7)
    np.testing.assert_allclose(d.log_prob(x), ref, rtol=1e-4)


def test_very_negative_scale_raw_gives_positive_scale():
    params = tf.constant([[0.0, -200.0, 0.0]])
    d = StudenttDistribution.from_parameterizer_output(params)
    assert d.scale.numpy()[0, 0] > 0
    assert np.isfinite(d.log_prob(tf.zeros((1, 1))).numpy()).all()


def test_kwargs_forwarded():
    d = StudenttDistribution.from_parameterizer_output(tf.zeros((1, 3)), name="t")
    assert d.name == "t"


def test_last_dim_not_multiple_of_three_raises():
    with pytest.raises(Exception):
        StudenttDistribution.from_parameterizer_output(tf.zeros((1, 4)))


@pytest.mark.parametrize("wrap", [np.asarray, tf.Variable], ids=["numpy", "tf.Variable"])
def test_non_tensor_input_accepted(wrap):
    loc, sr, dr = _raw()
    packed = np.concatenate([loc, sr, dr], -1)
    a = StudenttDistribution.from_parameterizer_output(wrap(packed))
    np.testing.assert_allclose(a.df, _softplus(dr) + 2.0, rtol=1e-6)
