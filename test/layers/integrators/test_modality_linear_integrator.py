import tensorflow as tf

from cavachon.layers.integrators.modality_linear_integrator import (
    ModalityLinearIntegrator,
)


def test_modality_linear_integrator_defaults():
    modality_keys = ["modality1", "modality2"]
    output_dims = 5

    integrator = ModalityLinearIntegrator(modality_keys=modality_keys)

    inputs = {
        "modality1": tf.random.normal([10, 3]),
        "modality2": tf.random.normal([10, 3]),
    }

    output = integrator(inputs)
    assert output.shape[-1] == output_dims


def test_modality_linear_integrator_output_dim():
    modality_keys = ["modality1", "modality2"]
    output_dims = 10

    integrator = ModalityLinearIntegrator(
        modality_keys=modality_keys, output_dims=output_dims
    )

    inputs = {
        "modality1": tf.random.normal([10, 3]),
        "modality2": tf.random.normal([10, 3]),
    }

    output = integrator(inputs)
    assert output.shape[-1] == output_dims


def test_modality_linear_integrator_output_dim_different_modality_dims():
    modality_keys = ["modality1", "modality2"]
    output_dims = 10

    integrator = ModalityLinearIntegrator(
        modality_keys=modality_keys, output_dims=output_dims
    )

    inputs = {
        "modality1": tf.random.normal([10, 3]),
        "modality2": tf.random.normal([10, 7]),
    }

    output = integrator(inputs)

    assert output.shape[-1] == output_dims
