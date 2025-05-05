import os

import pytest
import tensorflow as tf

from cavachon.environment.constants import Constants
from cavachon.models.decoders.decoder import Decoder


@pytest.fixture
def n_latent_dims():
    return 5


@pytest.fixture
def x_parameterizers():
    return {
        "modality1": tf.keras.layers.Dense(10),
        "modality2": tf.keras.layers.Dense(10),
    }


@pytest.fixture
def decoder(n_latent_dims, x_parameterizers):
    modality_names = ["modality1", "modality2"]
    n_layers = 3

    decoder = Decoder(
        modality_names,
        x_parameterizers,
        n_layers=n_layers,
        n_latent_dims=n_latent_dims,
        name="test_decoder",
    )

    return decoder


@pytest.fixture
def batch_size():
    return 10


@pytest.fixture
def inputs(batch_size):
    return {
        "modality1": tf.random.normal((batch_size, 5)),
        "modality2": tf.random.normal((batch_size, 5)),
    }


def test_decoder_call(decoder, batch_size, inputs):
    outputs = decoder(inputs)

    assert isinstance(outputs, dict)
    for modality_name in decoder.modality_names:
        assert f"{modality_name}_{Constants.MODEL_OUTPUTS_X_PARAMS}" in outputs

    assert outputs[decoder.modality_names[0]].shape == (batch_size, 5)
    assert outputs[decoder.modality_names[1]].shape == (batch_size, 5)
    assert outputs[
        f"{decoder.modality_names[0]}_{Constants.MODEL_OUTPUTS_X_PARAMS}"
    ].shape == (batch_size, 10)
    assert outputs[
        f"{decoder.modality_names[1]}_{Constants.MODEL_OUTPUTS_X_PARAMS}"
    ].shape == (batch_size, 10)


def test_decoder_train_step(decoder):
    with pytest.raises(NotImplementedError):
        decoder.train_step(None)


def test_decoder_test_step(decoder):
    with pytest.raises(NotImplementedError):
        decoder.test_step(None)


def test_decoder_fit(decoder, inputs):
    decoder.compile()
    with pytest.raises(NotImplementedError):
        decoder.fit(inputs)


def test_decoder_evaluate(decoder, inputs):
    decoder.compile()
    with pytest.raises(NotImplementedError):
        decoder.evaluate(inputs)


def test_decoder_predict(decoder, batch_size, inputs):
    outputs = decoder.predict(inputs)

    assert isinstance(outputs, dict)
    for modality_name in decoder.modality_names:
        assert f"{modality_name}_{Constants.MODEL_OUTPUTS_X_PARAMS}" in outputs

    assert outputs[decoder.modality_names[0]].shape == (batch_size, 5)
    assert outputs[
        f"{decoder.modality_names[0]}_{Constants.MODEL_OUTPUTS_X_PARAMS}"
    ].shape == (batch_size, 10)


def test_decoder_save_and_load(tmp_path, decoder, inputs):
    outputs = decoder.predict(inputs)
    model_path = os.path.join(tmp_path, "test_decoder.keras")
    decoder.save(model_path)
    loaded_decoder = tf.keras.models.load_model(model_path)
    loaded_outputs = loaded_decoder.predict(inputs)

    assert isinstance(loaded_outputs, dict)
    for modality_name in decoder.modality_names:
        assert f"{modality_name}_{Constants.MODEL_OUTPUTS_X_PARAMS}" in loaded_outputs

    tf.debugging.assert_near(
        outputs[decoder.modality_names[0]], loaded_outputs[decoder.modality_names[0]]
    )
    tf.debugging.assert_near(
        outputs[f"{decoder.modality_names[0]}_{Constants.MODEL_OUTPUTS_X_PARAMS}"],
        loaded_outputs[
            f"{decoder.modality_names[0]}_{Constants.MODEL_OUTPUTS_X_PARAMS}"
        ],
    )


def test_decoder_save_and_load_weights(tmp_path, decoder, inputs):
    outputs = decoder.predict(inputs)
    weights_path = os.path.join(tmp_path, "test_decoder.weights.h5")
    decoder.save_weights(weights_path)
    decoder.load_weights(weights_path)
    loaded_outputs = decoder.predict(inputs)

    tf.debugging.assert_near(
        outputs[decoder.modality_names[0]], loaded_outputs[decoder.modality_names[0]]
    )
    tf.debugging.assert_near(
        outputs[f"{decoder.modality_names[0]}_{Constants.MODEL_OUTPUTS_X_PARAMS}"],
        loaded_outputs[
            f"{decoder.modality_names[0]}_{Constants.MODEL_OUTPUTS_X_PARAMS}"
        ],
    )
