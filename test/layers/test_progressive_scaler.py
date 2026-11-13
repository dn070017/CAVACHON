import numpy as np
import pytest
import tensorflow as tf

from cavachon.layers.progressive_scaler import ProgressiveScaler


def test_default_state():
    s = ProgressiveScaler()
    assert float(s.total_iterations) == 5000.0
    assert float(s.current_iteration) == 1.0
    assert s.numpy() == pytest.approx((1 / 5000) ** 2, rel=1e-5)


def test_total_one_gives_scale():
    s = ProgressiveScaler(total_iterations=1, scale=3.0)
    assert s.numpy() == pytest.approx(3.0)
    np.testing.assert_allclose(s(tf.ones((2, 2))).numpy(), 3.0)


def test_activate_and_quadratic_schedule():
    s = ProgressiveScaler(total_iterations=1, scale=2.0)
    s.activate(4)
    assert float(s.current_iteration) == 0.0
    vals = []
    for _ in range(6):
        s.increment()
        vals.append(s.numpy())
    np.testing.assert_allclose(vals, 2.0 * np.array([1, 4, 9, 16, 16, 16]) / 16.0, rtol=1e-6)


def test_activate_zero_progress_floor():
    s = ProgressiveScaler(total_iterations=10)
    s.activate(10)
    # lower bound on fraction is 1e-8 -> progress 1e-16, never exactly 0
    assert s.numpy() == pytest.approx(1e-16, rel=1e-3)
    assert s.numpy() > 0


def test_decrement_clamps_at_zero():
    s = ProgressiveScaler(total_iterations=3)
    s.activate(3)
    s.decrement()
    assert float(s.current_iteration) == 0.0
    s.increment()
    s.increment()
    s.decrement()
    assert float(s.current_iteration) == 1.0


def test_call_scales_input_and_ignores_training_flag():
    s = ProgressiveScaler(total_iterations=4, scale=1.0)
    s.activate(4)
    s.increment()
    s.increment()
    x = tf.constant([[2.0, -4.0]])
    np.testing.assert_allclose(s(x, training=True).numpy(), x.numpy() * 0.25)
    np.testing.assert_allclose(s(x, training=False).numpy(), x.numpy() * 0.25)
    assert float(s.current_iteration) == 2.0  # call does not mutate


@pytest.mark.parametrize("value", [0.0, 0.5, 1.0, 1.5, 2.0])
def test_pin_to_values(value):
    s = ProgressiveScaler(total_iterations=100, scale=2.0)
    s.pin_to(value)
    assert float(s.total_iterations) == 1.0
    expected = max(value, 1e-16 * 2.0)
    assert s.numpy() == pytest.approx(expected, rel=1e-5)


def test_pin_to_above_scale_is_capped():
    s = ProgressiveScaler(total_iterations=1, scale=2.0)
    s.pin_to(5.0)
    assert s.numpy() == pytest.approx(2.0)


def test_assign_delegates_to_pin_to():
    s = ProgressiveScaler(total_iterations=10, scale=1.0)
    s.assign(0.36)
    assert s.numpy() == pytest.approx(0.36, rel=1e-5)


def test_increment_in_tf_function():
    s = ProgressiveScaler(total_iterations=10)
    s.activate(10)

    @tf.function
    def step():
        s.increment()
        return s(tf.ones(()))

    for _ in range(5):
        out = step()
    assert float(s.current_iteration) == 5.0
    assert float(out) == pytest.approx(0.25)


def test_variables_not_trainable():
    s = ProgressiveScaler()
    assert len(s.trainable_weights) == 0



@pytest.mark.xfail(
    strict=True,
    reason="BUG?: activate(0) clamps current_iteration to 0 forever, so the weight stays at 1e-16*scale "
    "instead of being immediately fully on",
)
def test_activate_zero_iterations_is_fully_on():
    s = ProgressiveScaler(total_iterations=1, scale=1.0)
    s.activate(0)
    s.increment()
    assert s.numpy() == pytest.approx(1.0)


def test_get_config_contains_init_args():
    kwargs = dict(total_iterations=7, scale=0.5, name="ps")
    cfg = ProgressiveScaler(**kwargs).get_config()
    for k, v in kwargs.items():
        assert cfg[k] == v


@pytest.mark.xfail(
    strict=True,
    reason="BUG: progressive_scaler.py:21 __init__ takes no **kwargs, so Keras from_config (which passes "
    "trainable/dtype from get_config) raises TypeError; breaks clone_model / model save-load",
)
def test_from_config_round_trip():
    clone = ProgressiveScaler.from_config(ProgressiveScaler(**dict(total_iterations=7, scale=0.5, name="ps")).get_config())
    assert float(clone.total_iterations) == 7.0 and float(clone.scale) == 0.5
