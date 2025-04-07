import numpy as np
import pandas as pd
import pytest
import scipy
import tensorflow as tf
from anndata import AnnData
from muon import MuData
from sklearn.preprocessing import LabelEncoder

from cavachon.dataset.batch_effect_process_configs import BatchEffectProcessConfig
from cavachon.dataset.dataset_creator import DatasetCreator
from cavachon.environment.constants import Constants
from cavachon.utils.tensor_utils import TensorUtils


@pytest.fixture
def mdata():
    obs_df = pd.DataFrame(
        {
            "categorical": ["G1", "G2", "G3", "G1", "G2", "G3", "G1", "G2", "G3", "G1"],
            "continuous": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        }
    )

    adata_A = AnnData(
        X=scipy.sparse.coo_matrix(
            np.array(
                [
                    [0, 0, 3, 0, 0, 0, 0],
                    [0, 0, 0, 4, 0, 0, 0],
                    [5, 0, 0, 0, 6, 0, 0],
                    [0, 7, 0, 0, 0, 0, 0],
                    [0, 0, 0, 8, 0, 0, 0],
                    [0, 0, 0, 0, 0, 9, 0],
                    [0, 0, 0, 0, 0, 0, 10],
                    [0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 9, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 7, 0],
                ]
            )
        ),
        obs=obs_df.copy(),
        var=pd.DataFrame({"A": ["A1", "A2", "A3", "A4", "A5", "A6", "A7"]}),
    )

    adata_B = AnnData(
        X=np.array(
            [
                [0, 5, 3, 0, 0],
                [0, 4, 0, 5, 0],
                [5, 0, 6, 6, 0],
                [0, 0, 7, 7, 3],
                [0, 1, 0, 8, 0],
                [0, 9, 5, 5, 0],
                [0, 0, 8, 0, 10],
                [0, 11, 0, 0, 0],
                [12, 0, 7, 0, 0],
                [0, 2, 7, 13, 0],
            ]
        ),
        obs=obs_df.copy(),
        var=pd.DataFrame({"B": ["B1", "B2", "B3", "B4", "B5"]}),
    )

    mdata = MuData({"A": adata_A, "B": adata_B})
    return mdata


@pytest.fixture
def modality_names():
    return ["A", "B"]


@pytest.fixture
def distribution_names():
    return {"A": "IndependentZeroInflatedNegativeBinomial", "B": "IndependentBernoulli"}


def test_dataset_creator_without_batch_effect(
    mdata, modality_names, distribution_names
):
    dataset_creator = DatasetCreator(
        mdata=mdata,
        modality_names=modality_names,
        distribution_names=distribution_names,
    )
    assert dataset_creator.mdata == mdata
    assert dataset_creator.modality_names == ["A", "B"]
    assert dataset_creator.distribution_names == {
        "A": "IndependentZeroInflatedNegativeBinomial",
        "B": "IndependentBernoulli",
    }
    assert dataset_creator.batch_effect_process_configs == {"A": [], "B": []}


def test_dataset_creator_with_batch_effect(mdata, modality_names, distribution_names):
    batch_effect_colnames = {
        "A": ["categorical", "continuous"],
        "B": ["categorical", "continuous"],
    }
    dataset_creator = DatasetCreator(
        mdata=mdata,
        modality_names=modality_names,
        distribution_names=distribution_names,
        batch_effect_colnames=batch_effect_colnames,
    )
    categorical_encoder = LabelEncoder()
    categorical_encoder.classes_ = ["G1", "G2", "G3"]
    batch_effect_process_configs = [
        BatchEffectProcessConfig(
            colname="categorical",
            categorical=True,
            encoder=categorical_encoder,
            n_vars=3,
        ),
        BatchEffectProcessConfig(
            colname="continuous", categorical=False, encoder=None, n_vars=1
        ),
    ]
    assert dataset_creator.mdata == mdata
    assert dataset_creator.modality_names == ["A", "B"]
    assert dataset_creator.distribution_names == {
        "A": "IndependentZeroInflatedNegativeBinomial",
        "B": "IndependentBernoulli",
    }
    assert modality_names == list(dataset_creator.batch_effect_process_configs.keys())
    assert (
        dataset_creator.batch_effect_process_configs["A"]
        == batch_effect_process_configs
    )
    assert (
        dataset_creator.batch_effect_process_configs["B"]
        == batch_effect_process_configs
    )


def test_create_base_dataset_without_batch_effect(
    mdata, modality_names, distribution_names
):
    dataset_creator = DatasetCreator(
        mdata=mdata,
        modality_names=modality_names,
        distribution_names=distribution_names,
    )
    batch_dataset = dataset_creator.create_base_dataset().batch(10)
    assert isinstance(batch_dataset, tf.data.Dataset)
    assert len(batch_dataset) == 1
    expected_A_X = TensorUtils.spmatrix_to_sparse_tensor(
        scipy.sparse.coo_matrix(
            np.array(
                [
                    [0, 0, 3, 0, 0, 0, 0],
                    [0, 0, 0, 4, 0, 0, 0],
                    [5, 0, 0, 0, 6, 0, 0],
                    [0, 7, 0, 0, 0, 0, 0],
                    [0, 0, 0, 8, 0, 0, 0],
                    [0, 0, 0, 0, 0, 9, 0],
                    [0, 0, 0, 0, 0, 0, 10],
                    [0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 9, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 7, 0],
                ]
            )
        )
    )
    expected_B_X = tf.convert_to_tensor(
        np.array(
            [
                [0, 5, 3, 0, 0],
                [0, 4, 0, 5, 0],
                [5, 0, 6, 6, 0],
                [0, 0, 7, 7, 3],
                [0, 1, 0, 8, 0],
                [0, 9, 5, 5, 0],
                [0, 0, 8, 0, 10],
                [0, 11, 0, 0, 0],
                [12, 0, 7, 0, 0],
                [0, 2, 7, 13, 0],
            ]
        )
    )
    expected_batch = tf.zeros((10, 1))
    for batch_data in batch_dataset:
        assert tf.reduce_all(
            tf.equal(batch_data[f"A_{Constants.TENSOR_NAME_BATCH}"], expected_batch)
        )
        assert tf.reduce_all(
            tf.equal(batch_data[f"B_{Constants.TENSOR_NAME_BATCH}"], expected_batch)
        )

        assert isinstance(
            batch_data[f"A_{Constants.TENSOR_NAME_X_MODEL}"], tf.SparseTensor
        )
        assert isinstance(
            batch_data[f"A_{Constants.TENSOR_NAME_X_OBSERVED}"], tf.SparseTensor
        )
        assert tf.reduce_all(
            tf.equal(
                batch_data[f"A_{Constants.TENSOR_NAME_X_MODEL}"].shape,
                expected_A_X.shape,
            )
        )
        assert tf.reduce_all(
            tf.equal(
                batch_data[f"A_{Constants.TENSOR_NAME_X_MODEL}"].indices,
                expected_A_X.indices,
            )
        )
        assert tf.reduce_all(
            tf.equal(
                batch_data[f"A_{Constants.TENSOR_NAME_X_MODEL}"].values,
                expected_A_X.values,
            )
        )
        assert tf.reduce_all(
            tf.equal(
                batch_data[f"A_{Constants.TENSOR_NAME_X_OBSERVED}"].shape,
                expected_A_X.shape,
            )
        )
        assert tf.reduce_all(
            tf.equal(
                batch_data[f"A_{Constants.TENSOR_NAME_X_OBSERVED}"].indices,
                expected_A_X.indices,
            )
        )
        assert tf.reduce_all(
            tf.equal(
                batch_data[f"A_{Constants.TENSOR_NAME_X_OBSERVED}"].values,
                expected_A_X.values,
            )
        )
        assert isinstance(batch_data[f"B_{Constants.TENSOR_NAME_X_MODEL}"], tf.Tensor)
        assert isinstance(
            batch_data[f"B_{Constants.TENSOR_NAME_X_OBSERVED}"], tf.Tensor
        )
        assert tf.reduce_all(
            tf.equal(batch_data[f"B_{Constants.TENSOR_NAME_X_MODEL}"], expected_B_X)
        )
        assert tf.reduce_all(
            tf.equal(batch_data[f"B_{Constants.TENSOR_NAME_X_MODEL}"], expected_B_X)
        )
        assert tf.reduce_all(
            tf.equal(batch_data[f"B_{Constants.TENSOR_NAME_X_OBSERVED}"], expected_B_X)
        )
        assert tf.reduce_all(
            tf.equal(batch_data[f"B_{Constants.TENSOR_NAME_X_OBSERVED}"], expected_B_X)
        )


def test_create_base_dataset_with_batch_effect(
    mdata, modality_names, distribution_names
):
    batch_effect_colnames = {
        "A": ["categorical", "continuous"],
        "B": ["categorical", "continuous"],
    }
    dataset_creator = DatasetCreator(
        mdata=mdata,
        modality_names=modality_names,
        distribution_names=distribution_names,
        batch_effect_colnames=batch_effect_colnames,
    )
    batch_dataset = dataset_creator.create_base_dataset().batch(10)
    assert isinstance(batch_dataset, tf.data.Dataset)
    assert len(batch_dataset) == 1
    expected_A_X = TensorUtils.spmatrix_to_sparse_tensor(
        scipy.sparse.coo_matrix(
            np.array(
                [
                    [0, 0, 3, 0, 0, 0, 0],
                    [0, 0, 0, 4, 0, 0, 0],
                    [5, 0, 0, 0, 6, 0, 0],
                    [0, 7, 0, 0, 0, 0, 0],
                    [0, 0, 0, 8, 0, 0, 0],
                    [0, 0, 0, 0, 0, 9, 0],
                    [0, 0, 0, 0, 0, 0, 10],
                    [0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 9, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 7, 0],
                ]
            )
        )
    )
    expected_B_X = tf.convert_to_tensor(
        np.array(
            [
                [0, 5, 3, 0, 0],
                [0, 4, 0, 5, 0],
                [5, 0, 6, 6, 0],
                [0, 0, 7, 7, 3],
                [0, 1, 0, 8, 0],
                [0, 9, 5, 5, 0],
                [0, 0, 8, 0, 10],
                [0, 11, 0, 0, 0],
                [12, 0, 7, 0, 0],
                [0, 2, 7, 13, 0],
            ]
        )
    )
    expected_batch = tf.convert_to_tensor(
        np.array(
            [
                [1, 0, 0, 1],
                [0, 1, 0, 2],
                [0, 0, 1, 3],
                [1, 0, 0, 4],
                [0, 1, 0, 5],
                [0, 0, 1, 6],
                [1, 0, 0, 7],
                [0, 1, 0, 8],
                [0, 0, 1, 9],
                [1, 0, 0, 10],
            ]
        ),
        dtype=tf.float32,
    )
    for batch_data in batch_dataset:
        print(expected_batch)
        print(batch_data[f"A_{Constants.TENSOR_NAME_BATCH}"])
        assert tf.reduce_all(
            tf.equal(batch_data[f"A_{Constants.TENSOR_NAME_BATCH}"], expected_batch)
        )
        assert tf.reduce_all(
            tf.equal(batch_data[f"B_{Constants.TENSOR_NAME_BATCH}"], expected_batch)
        )

        assert isinstance(
            batch_data[f"A_{Constants.TENSOR_NAME_X_MODEL}"], tf.SparseTensor
        )
        assert isinstance(
            batch_data[f"A_{Constants.TENSOR_NAME_X_OBSERVED}"], tf.SparseTensor
        )
        assert tf.reduce_all(
            tf.equal(
                batch_data[f"A_{Constants.TENSOR_NAME_X_MODEL}"].shape,
                expected_A_X.shape,
            )
        )
        assert tf.reduce_all(
            tf.equal(
                batch_data[f"A_{Constants.TENSOR_NAME_X_MODEL}"].indices,
                expected_A_X.indices,
            )
        )
        assert tf.reduce_all(
            tf.equal(
                batch_data[f"A_{Constants.TENSOR_NAME_X_MODEL}"].values,
                expected_A_X.values,
            )
        )
        assert tf.reduce_all(
            tf.equal(
                batch_data[f"A_{Constants.TENSOR_NAME_X_OBSERVED}"].shape,
                expected_A_X.shape,
            )
        )
        assert tf.reduce_all(
            tf.equal(
                batch_data[f"A_{Constants.TENSOR_NAME_X_OBSERVED}"].indices,
                expected_A_X.indices,
            )
        )
        assert tf.reduce_all(
            tf.equal(
                batch_data[f"A_{Constants.TENSOR_NAME_X_OBSERVED}"].values,
                expected_A_X.values,
            )
        )
        assert isinstance(batch_data[f"B_{Constants.TENSOR_NAME_X_MODEL}"], tf.Tensor)
        assert isinstance(
            batch_data[f"B_{Constants.TENSOR_NAME_X_OBSERVED}"], tf.Tensor
        )
        assert tf.reduce_all(
            tf.equal(batch_data[f"B_{Constants.TENSOR_NAME_X_MODEL}"], expected_B_X)
        )
        assert tf.reduce_all(
            tf.equal(batch_data[f"B_{Constants.TENSOR_NAME_X_MODEL}"], expected_B_X)
        )
        assert tf.reduce_all(
            tf.equal(batch_data[f"B_{Constants.TENSOR_NAME_X_OBSERVED}"], expected_B_X)
        )
        assert tf.reduce_all(
            tf.equal(batch_data[f"B_{Constants.TENSOR_NAME_X_OBSERVED}"], expected_B_X)
        )
