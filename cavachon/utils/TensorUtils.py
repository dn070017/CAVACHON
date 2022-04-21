import numpy as np
import pandas as pd
import tensorflow as tf

from cavachon.utils.DataFrameUtils import DataFrameUtils
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder
from typing import Dict, List, Tuple

class TensorUtils:
  """TensorUtils
  Utility functions for Tensorflow Tensor
  """

  @staticmethod
  def create_tensor_from_df(
      df: pd.DataFrame,
      colnames: List[str] = []) -> Tuple[tf.Tensor, Dict[str, LabelEncoder]]:
    """Create a Tensorflow Tensor from column data (specified with 
    `colnames`) in the provided DataFrame. If the column data is a 
    categorical variable, transform it with one-hot encoded Tensor. If 
    it is a continous variable, simply transform it into a Tensorflow 
    Tensor. If no valid column data is provided, return Tensor which is 
    a zero vector.

    Args:
      df (pd.DataFrame): input DataFrame.

      colnames (List[str], optional): columns of the DataFrame that are
      used to create the Tensor. Defaults to [].

    Returns:
      Tuple[tf.Tensor, Dict[str, LabelEncoder]]: the first element is 
      the one-hot encoded Tensor. The second element is the dictionary
      of LabelEncoder used to map the categorical variable into scalar 
      representation, where the keys are the column names and the values
      are the correponding LabelEncoder. 
    """
    # if no valid batch effect column is provided, use zero vector for batch effect
    encoder_dict = dict()
    n_obs, n_features = df.shape
    tensor_list = []

    for colname in colnames:
      if colname not in df.columns:
        continue
      coldata = df[colname]
      if DataFrameUtils.check_is_categorical(coldata):
        # if the column is a categorical variable, use one hot encoded tensor
        encoded_tensor, encoder = TensorUtils.create_one_hot_encoded_tensor(coldata)
        encoder_dict.setdefault(colname, encoder)
        tensor_list.append(encoded_tensor)
      else:
        # if the column is a continous variable, 
        tensor = tf.reshape(tf.convert_to_tensor(coldata, tf.float32), (-1, 1))
        tensor_list.append(tensor)
    
    if len(tensor_list) == 0:
      tensor_list.append(tf.zeros((n_obs, 1)))

    return tf.concat(tensor_list, axis=1), encoder_dict

  @staticmethod
  def create_one_hot_encoded_tensor(data: pd.Series) -> Tuple[tf.Tensor, LabelEncoder]:
    """Create a one hot encoded Tensor from a Series variable.

    Args:
      data (pd.Series): data variable.

    Returns:
      Tuple[tf.Tensor, LabelEncoder]: the first element is the one-hot 
      encoded Tensor, the second element is the LabelEncoder used to map
      the categorical variable into scalar representation.
    """
    encoder = LabelEncoder()
    encoded_array = encoder.fit_transform(data)
    n_class = len(encoder.classes_)
    encoded_tensor = tf.cast(tf.one_hot(encoded_array, n_class), tf.float32)

    return encoded_tensor, encoder

  @staticmethod
  def csr_to_sparse_tensor(csr_matrix: csr_matrix) -> tf.SparseTensor:
    """Create a SparseTensor out of a scipy csr matrix.

    Args:
      matrix (csr_matrix): the provided matrix

    Returns:
      tf.SparseTensor: the created SparseTensor.
    """
    coo_matrix = csr_matrix.tocoo()
    indices = np.mat([coo_matrix.row, coo_matrix.col]).transpose()
    sparse_tensor = tf.SparseTensor(indices, coo_matrix.data, coo_matrix.shape)
    return tf.cast(sparse_tensor, tf.float32)