from cavachon.dataloader.DataLoader import DataLoader
from cavachon.distributions.MultivariateNormalDiag import MultivariateNormalDiag
from cavachon.distributions.MixtureMultivariateNormalDiag import MixtureMultivariateNormalDiag
from sklearn.metrics.cluster import contingency_matrix
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from typing import Sequence, Union

import muon as mu
import numpy as np
import pandas as pd
import scanpy
import tensorflow as tf
import warnings

class ClusterAnalysis:
  """ClusterAnalysis

  Cluster analysis including online multi-facet (soft) clustering, 
  K-nearest neighbor analysis.

  Attibutes
  ---------
  mdata: muon.MuData
      the MuData for analysis.

  model: tf.keras.Model
      the trained generative model.

  """
  def __init__(
      self,
      mdata: mu.MuData,
      model: tf.keras.Model):
    """Constructor for ClusterAnalysis

    Parameters
    ----------
    mdata: muon.MuData
        the MuData for analysis.

    model: tf.keras.Model
        the trained generative model.
    
    """
    self.mdata = mdata
    self.model = model

  def compute_cluster_log_probability(
      self,
      modality: str,
      component: str,
      batch_size: int = 128) -> np.array:
    """Compute the log probability of a sample being assingned to each
    cluster in the latent space of the specified component.

    Parameters
    ----------
    modality: str
        the result will be stored in the obs and obsm of this modality
        in the self.mdata.

    component : str
        which latent space of the component to used for the clustering.

    batch_size : int, optional
        batch size used for the forward pass. Defaults to 128

    Returns
    -------
    np.array
        logpy_z, the log probability of a sample `i` being assigned to 
        each cluster `j`. 
    """
    z_prior_parameterizer = self.model.components[component].z_prior_parameterizer
    z_prior_parameters = tf.squeeze(z_prior_parameterizer(tf.ones((1, 1))))
    dist_z_y = MultivariateNormalDiag.from_parameterizer_output(z_prior_parameters[..., 1:])  
    dist_z = MixtureMultivariateNormalDiag.from_parameterizer_output(z_prior_parameters)
    logpy = tf.math.log(tf.math.softmax(z_prior_parameters[..., 0]) + 1e-7)

    logpy_z = list()
    dataloader = DataLoader(self.mdata, batch_size=batch_size)
    for batch_data in tqdm(dataloader):
      outputs = self.model(batch_data, training=False)
      z = outputs[f'{component}/z']
      logpz_y = dist_z_y.log_prob(tf.expand_dims(z, -2))
      logpz = tf.expand_dims(dist_z.log_prob(z), -1)
      logpy_z.append(logpy + logpz_y - logpz)

    logpy_z = np.vstack(logpy_z)
    cluster = tf.argmax(logpy_z, axis=-1).numpy()
    self.mdata.mod[modality].obsm[f'logpy_z_{component}'] = logpy_z
    self.mdata.mod[modality].obs[f'cluster_{component}'] = [f'Cluster {x:03d}'for x in cluster]
    
    return logpy_z

  def compute_neighbors_with_same_annotations(
      self,
      modality: str,
      use_cluster: str,
      use_rep: Union[str, np.array],
      n_neighbors: Union[int, Sequence[int]] = list(range(5, 25))) -> pd.DataFrame:
    """Perform K-nearest neighbor analysis.

    Parameters
    ----------
    modality: str
        the modality to used.
    
    use_cluster: str
        the column name of the clusters in the obs of modality.
    
    use_rep: Union[str, np.array]
        the key of obsm of modality to used to compute the distance 
        within and between clusters. Alternatively, the array will be 
        used if provided with np.array,
    
    n_neighbors: Union[int, Sequence[int]], optional
        the number of neighbors to be analyzed, Defaults to 
        list(range(5, 25))

    Returns
    -------
    pd.DataFrame
        analysis result for K-nearest neighbor. The DataFrame contains
        3 columns, the first column is the number of neighbors (K), the 
        second column is the cluster identifiers, the third column 
        specify the proportion of KNN samples with the same cluster, 

    Raises
    ------
    KeyError
        if use_cluster is not in the obs of the modality in self.mdata.
        (please perform compute_cluster_log_probability first for 
        unsupervised clustering).
    """
    if use_cluster not in self.mdata.mod[modality].obs:
      message = f'{use_cluster} not in obs DataFrame of the modality.'
      raise KeyError(message)
    
    proportions_series = list()
    clusters_series = list()
    n_neighbors_series = list()
    if isinstance(n_neighbors, (int, float)):
      n_neighbors = [n_neighbors]

    if isinstance(use_rep, np.ndarray):
      self.mdata[modality].obsm['z_custom'] = use_rep
      use_rep = 'z_custom'

    for k in tqdm(n_neighbors):
      if isinstance(k, float):
        k = int(k)
        message = 'Expect int for element in n_neighbors, transform float to int.'
        warnings.warn(message, RuntimeWarning)

      scanpy.pp.neighbors(self.mdata[modality], n_neighbors=k + 1, use_rep=use_rep)
      proportions_k = list()
      for i, j in enumerate(self.mdata[modality].obsp['distances']):
        cluster = self.mdata[modality].obs.iloc[i][use_cluster]
        neighbor_clusters = self.mdata[modality].obs.iloc[j.indices][use_cluster]
        proportions_k.append((neighbor_clusters == cluster).sum()/ k)
        clusters_series += [cluster]
      
      proportions_series.append(np.array(proportions_k))
      n_neighbors_series.append(np.array([k] * len(proportions_k)))

    return pd.DataFrame({ 
        'Number of Neighbors': np.concatenate(n_neighbors_series),
        'Cluster': clusters_series,
        '% of KNN Cells with the Same Cluster': np.concatenate(proportions_series)})
  
  def compute_contigency_matrix(
      self,
      modality: str,
      cluster_colname_a: str,
      cluster_colname_b: str) -> pd.DataFrame:
    """Compute the contigency matrix of two clustering results.

    Parameters
    ----------
    modality: str
        the modality to used.

    cluster_colname_a: str
        column name for the first clustering assignments in var.
    
    cluster_colname_b: str
        column name for the second clustering assignments in var.

    Returns
    -------
    pd.DataFrame
        contigency matrix where the indices are the cluster names in 
        the first clustering assignment, the column names are the 
        second clustering assignment.
    
    """
    encoder_cluster_a = LabelEncoder()
    encoder_cluster_b = LabelEncoder()

    encoded_array_a = encoder_cluster_a.fit_transform(self.mdata[modality].obs[cluster_colname_a])
    encoded_array_b = encoder_cluster_b.fit_transform(self.mdata[modality].obs[cluster_colname_b])

    contigency_matrix = pd.DataFrame(contingency_matrix(encoded_array_a, encoded_array_b))
    contigency_matrix.index = encoder_cluster_a.classes_
    contigency_matrix.columns = encoder_cluster_b.classes_

    return contigency_matrix

  def compute_classification_metrics(
      self,
      modality: str,
      cluster_colname_a: str,
      cluster_colname_b: str) -> pd.DataFrame:
    """Compute the classification metrics if using the second 
    clustering assignment to predict the first cluster assigment.

    Parameters
    ----------
    modality: str
        the modality to used.

    cluster_colname_a: str
        column name for the first clustering assignments in var.
    
    cluster_colname_b: str
        column name for the second clustering assignments in var.

    Returns
    -------
    pd.DataFrame
        classification metrics where the column names are:
        1. Cluster A
        2. Cluster B
        3. F1 Score
        4. Accuracy
        5. Sensitivity
        6. Specificity
        7. Precision
    
    """
    contigency_matrix = self.compute_contigency_matrix(
        modality,
        cluster_colname_a,
        cluster_colname_b)
    
    result = []
    for target_a in contigency_matrix.index:
      for target_b in contigency_matrix.columns:
        FP = contigency_matrix[target_b].sum() - contigency_matrix[target_b][target_a]
        TP = contigency_matrix[target_b][target_a]
        FN = contigency_matrix.loc[target_a].sum() - contigency_matrix[target_b][target_a]
        TN = (np.sum(contigency_matrix.values) - 
            contigency_matrix[target_b].sum() - 
            contigency_matrix.loc[target_a].sum() + 
            contigency_matrix[target_b][target_a])

        sensitivity = TP / (TP + FN)
        specificity = TN / (TN + FP)
        precision = TP / (TP + FP)
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        f1 = 2 * precision * sensitivity / (precision + sensitivity + 1e-3)

        result.append([target_a, target_b, f1, accuracy, sensitivity, specificity, precision])
    
    colnames = [
        'Cluster A',
        'Cluster B',
        'F1 Score',
        'Accuracy',
        'Sensitivity',
        'Specificity',
        'Precision'
    ]

    return pd.DataFrame(result, columns = colnames)
