import numpy as np
import pandas as pd
import scipy.sparse
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

from cavachon.utils.tensor_utils import TensorUtils


def test_create_tensor_from_df():
    sample_dataframe = pd.DataFrame(
        {
            "A": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "B": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
        }
    )
    expected_tensor = tf.convert_to_tensor(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [1.0, 0.0, 2.0],
            [1.0, 0.0, 3.0],
            [1.0, 0.0, 4.0],
            [1.0, 0.0, 5.0],
            [0.0, 1.0, 6.0],
            [0.0, 1.0, 7.0],
            [0.0, 1.0, 8.0],
            [0.0, 1.0, 9.0],
            [0.0, 1.0, 10.0],
            [0.0, 1.0, 11.0],
        ]
    )
    expected_encoded_labels = np.array([0.0, 1.0])

    tensor, encoder_dict = TensorUtils.create_tensor_from_df(
        sample_dataframe, ["A", "B"]
    )

    assert encoder_dict["B"] is None, "'B' should not be one-hot encoded."
    assert isinstance(encoder_dict["A"], LabelEncoder), "'A' should be one-hot encoded."
    assert np.array_equal(expected_encoded_labels, encoder_dict["A"].classes_), (
        "The classes of the LabelEncoder is incorrect."
    )
    assert tf.reduce_all(tf.equal(expected_tensor, tensor)), (
        f"The created Tensor is not the same. Expected {expected_tensor}, but got {tensor}."
    )


def test_create_tensor_from_df_empty():
    df = pd.DataFrame(
        {"A": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]}
    )
    tensor, encoder_dict = TensorUtils.create_tensor_from_df(df, ["C"])
    tensor_true = tf.convert_to_tensor(
        [
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
        ]
    )
    assert "A" not in encoder_dict, "'A' should not be one-hot encoded."
    assert "C" not in encoder_dict, "'C' should not be one-hot encoded."
    assert tf.reduce_all(tf.equal(tensor_true, tensor)), (
        f"The created Tensor is not the same. Expected {tensor_true}, but got {tensor}."
    )


def test_create_one_hot_encoded_tensor():
    data = pd.Series([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    expected_encoded_labels = np.array([0.0, 1.0])

    tensor, encoder = TensorUtils.create_one_hot_encoded_tensor(data)

    tensor_true = tf.convert_to_tensor(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ]
    )

    assert np.array_equal(expected_encoded_labels, encoder.classes_), (
        "The classes of the LabelEncoder is incorrect."
    )
    assert tf.reduce_all(tf.equal(tensor_true, tensor)), (
        f"The created Tensor is not the same. Expected {tensor_true}, but got {tensor}."
    )


def test_spmatrix_to_sparse_tensor():
    matrix = np.matrix([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    csr_matrix = scipy.sparse.csr_matrix(matrix)
    sparse_tensor = TensorUtils.spmatrix_to_sparse_tensor(csr_matrix)
    tensor = tf.sparse.to_dense(sparse_tensor)

    sparse_tensor_true = tf.SparseTensor(
        indices=[[0, 2], [2, 0]], values=[1.0, 1.0], dense_shape=[3, 3]
    )
    tensor_true = tf.sparse.to_dense(sparse_tensor_true)

    assert tf.reduce_all(tf.equal(tensor_true, tensor)), (
        f"The created Tensor is not the same. Expected {tensor_true}, but got {tensor}."
    )


# Add tests for the new/updated functions in TensorUtils
def test_is_sparse_tensor():
    # Test regular tensor
    regular_tensor = tf.constant([1, 2, 3])
    assert not TensorUtils.is_sparse_tensor(regular_tensor)

    # Test sparse tensor
    indices = [[0, 0], [1, 2]]
    values = [1.0, 2.0]
    dense_shape = [3, 4]
    sparse_tensor = tf.SparseTensor(
        indices=indices, values=values, dense_shape=dense_shape
    )
    assert TensorUtils.is_sparse_tensor(sparse_tensor)

    # Test other types
    assert not TensorUtils.is_sparse_tensor(5)
    assert not TensorUtils.is_sparse_tensor([1, 2, 3])
    assert not TensorUtils.is_sparse_tensor(np.array([1, 2, 3]))


def test_max_n_neurons():
    # Create some layers
    layer1 = tf.keras.layers.Dense(128)
    layer2 = tf.keras.layers.Dense(256)
    layer3 = tf.keras.layers.Conv2D(32, (3, 3))

    # Test with no Dense layers
    assert TensorUtils.max_n_neurons([layer3]) == 0

    # Test with multiple Dense layers
    assert TensorUtils.max_n_neurons([layer1, layer2, layer3]) == 256
    assert TensorUtils.max_n_neurons([layer1, layer3]) == 128


def test_remove_nan_gradients():
    # Create some gradients with nans and infs
    grad1 = tf.constant([1.0, 2.0, float("nan"), 4.0])
    grad2 = tf.constant([float("inf"), -float("inf"), 3.0, 4.0])
    grad3 = tf.constant([12.0, -15.0, 5.0, 6.0])  # For testing clipping
    grads = [grad1, grad2, grad3, None]  # Include None to test that case

    # Apply the function
    processed_grads = TensorUtils.remove_nan_gradients(grads, clip_value=10)

    # Check results
    assert tf.reduce_all(
        tf.equal(processed_grads[0], tf.constant([1.0, 2.0, 0.0, 4.0]))
    )
    assert tf.reduce_all(
        tf.equal(processed_grads[1], tf.constant([0.0, 0.0, 3.0, 4.0]))
    )
    assert tf.reduce_all(
        tf.equal(processed_grads[2], tf.constant([10.0, -10.0, 5.0, 6.0]))
    )

    assert processed_grads[3] is None


def test_create_backbone_layers():
    # Test default parameters
    model = TensorUtils.create_backbone_layers()

    # Check model structure
    assert len(model.layers) == 6
    assert isinstance(model.layers[0], tf.keras.layers.Dense)
    assert model.layers[0].units == 128
    assert isinstance(model.layers[1], tf.keras.layers.LayerNormalization)

    assert isinstance(model.layers[2], tf.keras.layers.Dense)
    assert model.layers[2].units == 256
    assert isinstance(model.layers[3], tf.keras.layers.LayerNormalization)

    assert isinstance(model.layers[4], tf.keras.layers.Dense)
    assert model.layers[4].units == 512
    assert isinstance(model.layers[5], tf.keras.layers.LayerNormalization)

    # Test with custom parameters
    model = TensorUtils.create_backbone_layers(
        n_layers=2,
        base_n_neurons=64,
        max_n_neurons=100,
        rate=2,
        activation="relu",
        reverse=True,
        name="custom_backbone",
    )

    # Check model structure
    assert len(model.layers) == 4
    assert model.name == "custom_backbone"
    assert isinstance(model.layers[0], tf.keras.layers.LayerNormalization)
    assert isinstance(model.layers[1], tf.keras.layers.Dense)
    assert model.layers[1].units == 100
    assert isinstance(model.layers[2], tf.keras.layers.LayerNormalization)
    assert isinstance(model.layers[3], tf.keras.layers.Dense)
    assert model.layers[3].units == 64


def test_split():
    # Create a tensor
    x = tf.constant([[i, i + 1] for i in range(300)])

    # Split it
    splits = TensorUtils.split(x, batch_size=128)

    # Check results
    assert len(splits) == 3
    assert splits[0].shape == (128, 2)
    assert splits[1].shape == (128, 2)
    assert splits[2].shape == (44, 2)  # 300 % 128 = 44

    # Check that the content is correct
    assert tf.reduce_all(tf.equal(splits[0][0], tf.constant([0, 1])))
    assert tf.reduce_all(tf.equal(splits[1][0], tf.constant([128, 129])))
    assert tf.reduce_all(tf.equal(splits[2][0], tf.constant([256, 257])))
