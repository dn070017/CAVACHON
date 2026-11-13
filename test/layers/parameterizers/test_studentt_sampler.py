import numpy as np
import tensorflow as tf

from cavachon.layers.parameterizers import StudenttSampler

LOC = np.array([[1.0, -1.0]], np.float32)


def _params(raw_scale=0.0, raw_df=0.0, n=1):
    p = np.concatenate([LOC, np.full_like(LOC, raw_scale), np.full_like(LOC, raw_df)], -1)
    return tf.constant(np.repeat(p, n, axis=0))


def test_inference_returns_loc():
    np.testing.assert_array_equal(StudenttSampler()(_params()).numpy(), LOC)


def test_training_sample_shape_and_median():
    out = StudenttSampler()(_params(raw_scale=-1.0, raw_df=5.0, n=20000), training=True).numpy()
    assert out.shape == (20000, 2)
    np.testing.assert_allclose(np.median(out, 0), LOC[0], atol=0.05)


def test_training_scale_matches_softplus():
    raw_df = 50.0  # df ~ 52 -> approx normal with std ~ scale * sqrt(df/(df-2))
    out = StudenttSampler()(_params(raw_scale=0.0, raw_df=raw_df, n=20000), training=True).numpy()
    scale = np.logaddexp(0.0, 0.0)
    df = raw_df + 2.0
    np.testing.assert_allclose(out.std(0), scale * np.sqrt(df / (df - 2)), rtol=0.05)


def test_reparameterized_gradient_flows_to_all_params():
    p = tf.Variable(_params(n=2))
    with tf.GradientTape() as tape:
        y = tf.reduce_sum(StudenttSampler()(p, training=True))
    g = tape.gradient(y, p).numpy()
    np.testing.assert_allclose(g[:, :2], 1.0)
    assert np.isfinite(g).all()


def test_tiny_scale_finite():
    out = StudenttSampler()(_params(raw_scale=-200.0), training=True).numpy()
    assert np.isfinite(out).all()
    np.testing.assert_allclose(out, LOC, atol=1e-5)


def test_get_config_contains_init_args():
    kwargs = dict(name="st")
    cfg = StudenttSampler(**kwargs).get_config()
    for k, v in kwargs.items():
        assert cfg[k] == v


def test_from_config_round_trip():
    # regression: __init__ lacked **kwargs, so from_config (trainable/dtype) raised TypeError
    clone = StudenttSampler.from_config(StudenttSampler(**dict(name="st")).get_config())
    assert clone.name == "st"
