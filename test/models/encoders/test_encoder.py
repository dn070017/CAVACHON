import pytest
import tensorflow as tf

from cavachon.environment.constants import Constants
from cavachon.layers.modifiers.distribution_preset_modifier import (
    DistributionPresetModifier,
)
from cavachon.models.encoders.encoder import Encoder


@pytest.fixture
def n_latent_dims():
    return 5


@pytest.fixture
def encoder(n_latent_dims):
    modality_names = ["modality1", "modality2"]
    modifiers = {
        "modality1": DistributionPresetModifier(),
        "modality2": DistributionPresetModifier(),
    }
    n_reduced_dims = 512

    encoder = Encoder(
        modality_names,
        modifiers,
        n_reduced_dims=n_reduced_dims,
        n_latent_dims=n_latent_dims,
    )

    return encoder


@pytest.fixture
def batch_size():
    return 10


@pytest.fixture
def labels(batch_size):
    return tf.random.normal((batch_size, 5))


@pytest.fixture
def inputs(batch_size):
    return {
        f"modality1_{Constants.TENSOR_NAME_X_MODEL}": tf.random.normal(
            (batch_size, 100)
        ),
        f"modality2_{Constants.TENSOR_NAME_X_MODEL}": tf.random.normal(
            (batch_size, 100)
        ),
    }


def test_encoder_call(n_latent_dims, encoder, batch_size, inputs):
    outputs = encoder(inputs)

    assert isinstance(outputs, dict)
    assert Constants.MODEL_OUTPUTS_Z in outputs
    assert Constants.MODEL_OUTPUTS_Z_PARAMS in outputs

    z = outputs[Constants.MODEL_OUTPUTS_Z]
    z_params = outputs[Constants.MODEL_OUTPUTS_Z_PARAMS]

    assert z.shape == (batch_size, n_latent_dims)
    assert z_params.shape == (batch_size, n_latent_dims * 2)


def test_encoder_train_step(encoder):
    with pytest.raises(NotImplementedError):
        encoder.train_step(None)


def test_encoder_test_step(encoder):
    with pytest.raises(NotImplementedError):
        encoder.test_step(None)


def test_encoder_fit(encoder, inputs, labels):
    encoder.compile()
    with pytest.raises(NotImplementedError):
        encoder.fit(inputs, labels)


def test_encoder_evaluate(encoder, inputs, labels):
    encoder.compile()
    with pytest.raises(NotImplementedError):
        encoder.evaluate(inputs, labels)


def test_encoder_predict(n_latent_dims, encoder, batch_size, inputs):
    outputs = encoder.predict(inputs)

    assert isinstance(outputs, dict)
    assert Constants.MODEL_OUTPUTS_Z in outputs
    assert Constants.MODEL_OUTPUTS_Z_PARAMS in outputs

    z = outputs[Constants.MODEL_OUTPUTS_Z]
    z_params = outputs[Constants.MODEL_OUTPUTS_Z_PARAMS]

    assert z.shape == (batch_size, n_latent_dims)
    assert z_params.shape == (batch_size, n_latent_dims * 2)
