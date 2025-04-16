import pytest
import tensorflow as tf

from cavachon.layers.latent_integrators.progressive_scaler import ProgressiveScaler


@pytest.fixture
def progressive_scaler_instance():
    return ProgressiveScaler(total_iterations=10)


@pytest.fixture
def input_tensor():
    return tf.random.normal((5, 3), dtype=tf.float32)


def test_progressive_scaler_initialization(progressive_scaler_instance):
    assert progressive_scaler_instance.total_iterations.dtype == tf.float32
    assert progressive_scaler_instance.current_iteration.dtype == tf.float32
    tf.debugging.assert_equal(
        progressive_scaler_instance.total_iterations,
        tf.constant(10.0, dtype=tf.float32),
    )
    tf.debugging.assert_equal(
        progressive_scaler_instance.current_iteration,
        tf.constant(0.0, dtype=tf.float32),
    )
    assert not progressive_scaler_instance.total_iterations.trainable
    assert not progressive_scaler_instance.current_iteration.trainable


def test_progressive_scaler_call_inference(progressive_scaler_instance, input_tensor):
    tf.debugging.assert_equal(
        progressive_scaler_instance(input_tensor, training=False), input_tensor
    )


def test_progressive_scaler_call_training(progressive_scaler_instance, input_tensor):
    total_iterations = progressive_scaler_instance.total_iterations

    current_iter_before = tf.identity(progressive_scaler_instance.current_iteration)
    expected_output_1 = tf.zeros_like(input_tensor)
    output_tensor_1 = progressive_scaler_instance(input_tensor, training=True)

    tf.debugging.assert_near(output_tensor_1, expected_output_1, rtol=1e-6)
    tf.debugging.assert_equal(
        progressive_scaler_instance.current_iteration, current_iter_before + 1.0
    )

    current_iter_before = tf.identity(progressive_scaler_instance.current_iteration)
    expected_alpha_2 = tf.square(current_iter_before / total_iterations)
    expected_output_2 = expected_alpha_2 * input_tensor
    output_tensor_2 = progressive_scaler_instance(input_tensor, training=True)

    tf.debugging.assert_near(output_tensor_2, expected_output_2, rtol=1e-6)
    tf.debugging.assert_equal(
        progressive_scaler_instance.current_iteration, current_iter_before + 1.0
    )

    assert output_tensor_1.shape == input_tensor.shape
    assert output_tensor_2.shape == input_tensor.shape


def test_progressive_scaler_call_training_iteration_cap(input_tensor):
    scaler = ProgressiveScaler(total_iterations=2)
    total_iterations = scaler.total_iterations
    scaler(input_tensor, training=True)
    tf.debugging.assert_equal(
        scaler.current_iteration, tf.constant(1.0, dtype=tf.float32)
    )
    scaler(input_tensor, training=True)
    tf.debugging.assert_equal(
        scaler.current_iteration, tf.constant(2.0, dtype=tf.float32)
    )
    # call 3 (should reach total_iterations, alpha uses 2/2)
    current_iter_before = tf.identity(scaler.current_iteration)  # Should be 2.0
    expected_alpha_3 = tf.square(
        current_iter_before / total_iterations
    )  # (2/2)^2 = 1.0
    expected_output_3 = expected_alpha_3 * input_tensor
    output_tensor_3 = scaler(input_tensor, training=True)
    tf.debugging.assert_near(output_tensor_3, expected_output_3, rtol=1e-6)
    # Iteration should be capped at total_iterations (2.0)
    tf.debugging.assert_equal(scaler.current_iteration, total_iterations)


def test_progressive_scaler_compute_alpha(progressive_scaler_instance):
    total_iterations = progressive_scaler_instance.total_iterations

    progressive_scaler_instance.current_iteration.assign(1.0)
    expected_alpha = tf.square(1.0 / total_iterations)
    tf.debugging.assert_near(
        progressive_scaler_instance.compute_alpha(), expected_alpha, rtol=1e-6
    )

    mid_iteration = total_iterations / 2.0
    progressive_scaler_instance.current_iteration.assign(mid_iteration)
    expected_alpha_mid = tf.square(mid_iteration / total_iterations)
    tf.debugging.assert_near(
        progressive_scaler_instance.compute_alpha(), expected_alpha_mid, rtol=1e-6
    )

    progressive_scaler_instance.current_iteration.assign(total_iterations)
    expected_alpha_end = tf.square(total_iterations / total_iterations)  # (1.0)^2 = 1.0
    tf.debugging.assert_near(
        progressive_scaler_instance.compute_alpha(), expected_alpha_end, rtol=1e-6
    )

    progressive_scaler_instance.current_iteration.assign(total_iterations + 5.0)
    expected_alpha_beyond = tf.constant(
        1.0, dtype=tf.float32
    )  # (capped_alpha)^2 = (1.0)^2 = 1.0
    tf.debugging.assert_near(
        progressive_scaler_instance.compute_alpha(), expected_alpha_beyond, rtol=1e-6
    )


def test_progressive_scaler_step(progressive_scaler_instance):
    progressive_scaler_instance.total_iterations.assign(2.0)
    total_iterations = progressive_scaler_instance.total_iterations

    progressive_scaler_instance.step()
    tf.debugging.assert_equal(
        progressive_scaler_instance.current_iteration,
        tf.constant(1.0, dtype=tf.float32),
    )
    progressive_scaler_instance.step()
    tf.debugging.assert_equal(
        progressive_scaler_instance.current_iteration,
        tf.constant(2.0, dtype=tf.float32),
    )
    progressive_scaler_instance.step()
    tf.debugging.assert_equal(
        progressive_scaler_instance.current_iteration,
        total_iterations,
    )
    progressive_scaler_instance.step()
    tf.debugging.assert_equal(
        progressive_scaler_instance.current_iteration,
        total_iterations,
    )


def test_progressive_scaler_reset(progressive_scaler_instance):
    progressive_scaler_instance.step()
    tf.debugging.assert_equal(
        progressive_scaler_instance.current_iteration,
        tf.constant(1.0, dtype=tf.float32),
    )
    progressive_scaler_instance.step()
    tf.debugging.assert_equal(
        progressive_scaler_instance.current_iteration,
        tf.constant(2.0, dtype=tf.float32),
    )
    progressive_scaler_instance.reset()
    tf.debugging.assert_equal(
        progressive_scaler_instance.current_iteration,
        tf.constant(0.0, dtype=tf.float32),
    )
    progressive_scaler_instance.step()
    tf.debugging.assert_equal(
        progressive_scaler_instance.current_iteration,
        tf.constant(1.0, dtype=tf.float32),
    )


def test_progressive_scaler_alpha_clamping(input_tensor):
    scaler = ProgressiveScaler(total_iterations=5)
    total_iterations = scaler.total_iterations

    scaler.current_iteration.assign(6.0)

    # expected alpha should be 1.0 (clamped during calculation)
    # alpha = (6 / 5) -> clamped to 1.0 -> squared = 1.0
    expected_alpha = tf.constant(1.0, dtype=tf.float32)
    expected_output = expected_alpha * input_tensor
    output_tensor = scaler(input_tensor, training=True)

    tf.debugging.assert_near(output_tensor, expected_output, rtol=1e-6)
    tf.debugging.assert_equal(scaler.current_iteration, total_iterations)
