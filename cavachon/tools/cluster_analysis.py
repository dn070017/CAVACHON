import warnings
from itertools import product
from typing import Dict, List, Optional, Sequence, Union

import muon as mu
import numpy as np
import pandas as pd
import scanpy
import tensorflow as tf
from sklearn.metrics.cluster import contingency_matrix
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from cavachon.dataloader.dataloader import DataLoader
from cavachon.distributions.mixture_multivariate_normal_diag_distribution import (
    MixtureMultivariateNormalDiagDistribution,
)
from cavachon.distributions.multivariate_normal_diag_distribution import (
    MultivariateNormalDiagDistribution,
)


class ClusterAnalysis:
    """ClusterAnalysis

    Cluster analysis including online multi-facet (soft) clustering,
    K-nearest neighbor analysis.

    Attributes
    ----------
    mdata: muon.MuData
        the MuData for analysis.

    model: tf.keras.Model
        the trained generative model.

    """

    def __init__(self, mdata: mu.MuData, model: tf.keras.Model):
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
        batch_effect_colnames: Optional[Dict[str, List[str]]] = None,
        distribution_names: Optional[Dict[str, str]] = None,
        batch_size: int = 128,
        min_n_obs=36,
    ) -> np.array:
        """Compute the log probability of a sample being assigned to
        each cluster in the latent space of the specified component.

        Parameters
        ----------
        modality: str
            the result will be stored in the obs and obsm of this
            modality in the self.mdata.

        component : str
            which latent space of the component to used for the
            clustering.

        batch_effect_colnames: Dict[str, List[str]], optional
            the batch effect columns for each modality. Defaults to
            None.

        distribution_names: Dict[str, str], optional
            the distribution names for each modality. Defaults to
            None.

        batch_size : int, optional
            batch size used for the forward pass. Defaults to 128

        Returns
        -------
        np.array
            logpy_z, the log probability of a sample `i` being assigned
            to each cluster `j`.
        """
        z_prior_parameterizer = self.model.components[component].z_prior_parameterizer
        z_prior_parameters = tf.squeeze(z_prior_parameterizer(tf.ones((1, 1))))
        dist_z_y = MultivariateNormalDiagDistribution.from_parameterizer_output(
            z_prior_parameters[..., 1:]
        )
        dist_z = MixtureMultivariateNormalDiagDistribution.from_parameterizer_output(
            z_prior_parameters
        )
        logpy = tf.math.log(tf.math.softmax(z_prior_parameters[..., 0]) + 1e-7)

        logpy_z = list()
        dataloader = DataLoader(
            self.mdata, batch_size, batch_effect_colnames, distribution_names
        )
        for batch_data in tqdm(dataloader):
            outputs = self.model(batch_data, training=False)
            z = outputs[f"{component}_z"]
            logpz_y = dist_z_y.log_prob(tf.expand_dims(z, -2))
            logpz = tf.expand_dims(dist_z.log_prob(z), -1)
            logpy_z.append(logpy + logpz_y - logpz)

        logpy_z = np.vstack(logpy_z)
        cluster = tf.argmax(logpy_z, axis=-1).numpy()
        self.mdata.mod[modality].obs[f"cluster_{component}"] = [
            f"Cluster {x:03d}" for x in cluster
        ]
        _keep = self.mdata.mod[modality].obs[f"cluster_{component}"].value_counts()

        while (_keep < min_n_obs).sum() > 0:
            keep = []
            for i, x in zip(_keep.index[:-1], _keep):
                if x > min_n_obs:
                    keep.append(int(i.split(" ")[1]))
            logpy_z = logpy_z[:, keep]
            cluster = tf.argmax(logpy_z, axis=-1).numpy()
            self.mdata.mod[modality].obsm[f"logpy_z_{component}"] = logpy_z
            self.mdata.mod[modality].obs[f"cluster_{component}"] = [
                f"Cluster {x:03d}" for x in cluster
            ]
            _keep = self.mdata.mod[modality].obs[f"cluster_{component}"].value_counts()

        return logpy_z

    @staticmethod
    def _extract_gmm_parameters(prior_parameterizer):
        """Extract GMM parameters from a trained z_prior_parameterizer.

        Uses the ``parameters`` property to retrieve the learned prior
        parameters without a forward pass.

        Parameters
        ----------
        prior_parameterizer : MixtureMultivariateNormalDiagParameterizerLayer
            the trained prior parameterizer layer.

        Returns
        -------
        dict
            keys: ``pi`` (mixing weights), ``mu`` (means), ``sigma2``
            (variances), ``K`` (n_components), ``D`` (event_dims).
        """
        parameters = prior_parameterizer.parameters.numpy()
        logits = parameters[:, 0]
        n_clusters = len(logits)
        params = parameters[:, 1:]
        n_dims = params.shape[1] // 2
        mu = params[:, :n_dims]
        scale = params[:, n_dims:]
        return {
            "pi": tf.nn.softmax(tf.convert_to_tensor(logits)).numpy(),
            "mu": mu,
            "sigma2": scale**2,
            "K": n_clusters,
            "D": n_dims,
        }

    @staticmethod
    def _remove_small_clusters(logpy_z, cluster, min_n_obs):
        """Iteratively remove clusters with fewer than ``min_n_obs``
        observations and re-assign their members to the remaining clusters.
        Remaps cluster indices to be contiguous after removal.

        Parameters
        ----------
        logpy_z : np.ndarray
            log-probability matrix of shape (n_samples, n_clusters).
        cluster : np.ndarray
            initial hard cluster assignments (n_samples,).
        min_n_obs : int
            minimum number of observations required to keep a cluster.

        Returns
        -------
        tuple
            (logpy_z, cluster_final, final_labels) where
            ``cluster_final`` contains contiguous integer labels and
            ``final_labels`` are formatted ``"Cluster NNN"`` strings.
        """
        cluster_labels = [f"Cluster {x:03d}" for x in cluster]
        temp_series = pd.Series(cluster_labels)
        _keep = temp_series.value_counts()
        while (_keep < min_n_obs).sum() > 0:
            keep = []
            for i, x in zip(_keep.index, _keep):
                if x >= min_n_obs:
                    keep.append(int(i.split(" ")[1]))
            if len(keep) == 0:
                break
            logpy_z = logpy_z[:, keep]
            cluster = tf.argmax(logpy_z, axis=-1).numpy()
            cluster_labels = [f"Cluster {x:03d}" for x in cluster]
            temp_series = pd.Series(cluster_labels)
            _keep = temp_series.value_counts()

        unique_clusters = np.unique(cluster)
        remap = {old: new for new, old in enumerate(unique_clusters)}
        cluster_final = np.array([remap[c] for c in cluster])
        final_labels = [f"Cluster {x:03d}" for x in cluster_final]
        return logpy_z, cluster_final, final_labels

    def compute_integrated_cluster_log_probability(
        self,
        modality: str,
        component: str,
        batch_size: int = 128,
        min_n_obs: int = 36,
    ) -> np.array:
        """Compute the log probability of a sample being assigned to
        each *integrated* cluster in z_hat space for a hierarchical
        component.

        Unlike ``compute_cluster_log_probability`` (which clusters in
        ``z`` space of a single component), this method projects the
        GMM parameters of every parent component **and** the child
        component through the learned hierarchical-encoder weights
        (``r_network`` and ``b_network``) to build a joint Gaussian
        mixture in z_hat space.  Samples are then scored against that
        joint mixture.

        Results are stored in::

            mdata.mod[modality].obs["cluster_{component}_integrated"]
            mdata.mod[modality].obsm["logpy_zhat_{component}_integrated"]

        Parameters
        ----------
        modality : str
            the modality whose obsm contains the pre-computed
            ``z_hat_{component}`` array and where results are saved.
        component : str
            name of the **child** (hierarchical) component.
        batch_size : int, optional
            number of samples to process at once while scoring.
            Defaults to 128.
        min_n_obs : int, optional
            clusters with fewer than this many observations are
            iteratively removed. Defaults to 36.

        Returns
        -------
        np.ndarray
            ``logpy_zhat``, log-probability of sample *i* being assigned
            to each integrated cluster *j*.

        Raises
        ------
        ValueError
            if ``component`` has no ``conditioned_on_z_hat`` parents.
        """
        comp = self.model.components[component]
        parent_names = comp.conditioned_on_z_hat

        if not parent_names:
            raise ValueError(
                f"Component '{component}' has no conditioned_on_z_hat parents. "
                "Integrated clustering requires hierarchical components."
            )

        parent_params = []
        parent_dims = []
        for parent_name in parent_names:
            parent_parameterizer = self.model.components[
                parent_name
            ].z_prior_parameterizer
            p_params = self._extract_gmm_parameters(parent_parameterizer)
            parent_params.append(p_params)
            parent_dims.append(p_params["D"])

        child_parameterizer = comp.z_prior_parameterizer
        child_params = self._extract_gmm_parameters(child_parameterizer)

        hierarchical_encoder = comp.hierarchical_encoder
        W_r = hierarchical_encoder.r_network.get_weights()[0]
        W_b = hierarchical_encoder.b_network.get_weights()[0]

        W_parent_parts: list = []
        offset = 0
        for dim in parent_dims:
            W_parent_parts.append(W_b[offset : offset + dim, :])
            offset += dim
        W_child_raw = W_b[offset:, :]
        W_child_effective = W_child_raw @ W_r
        cluster_ranges = [range(p["K"]) for p in parent_params] + [
            range(child_params["K"])
        ]
        n_clusters = np.prod([len(r) for r in cluster_ranges], dtype=int)
        D_out = W_b.shape[1]

        mu_zhat = np.zeros((n_clusters, D_out))
        sigma2_zhat = np.zeros((n_clusters, D_out))
        pi_zhat = np.zeros(n_clusters)

        for idx, indices in enumerate(product(*cluster_ranges)):
            parent_idx = indices[:-1]
            child_idx = indices[-1]

            for p_idx, W_p, p_params in zip(parent_idx, W_parent_parts, parent_params):
                mu_zhat[idx] += W_p.T @ p_params["mu"][p_idx]
            mu_zhat[idx] += W_child_effective.T @ child_params["mu"][child_idx]

            for p_idx, W_p, p_params in zip(parent_idx, W_parent_parts, parent_params):
                sigma2_zhat[idx] += (W_p**2).T @ p_params["sigma2"][p_idx]
            sigma2_zhat[idx] += (W_child_effective**2).T @ child_params["sigma2"][
                child_idx
            ]

            pi_zhat[idx] = child_params["pi"][child_idx]
            for p_idx, p_params in zip(parent_idx, parent_params):
                pi_zhat[idx] *= p_params["pi"][p_idx]
        z_hat_all = self.mdata.mod[modality].obsm[
            f"z_hat_{component}"
        ]
        N = z_hat_all.shape[0]
        logpy = np.log(pi_zhat + 1e-7)
        logpy_zhat = np.zeros((N, n_clusters))

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            z_hat_batch = z_hat_all[start:end]
            for k in range(n_clusters):
                diff = z_hat_batch - mu_zhat[k]
                log_likelihood = -0.5 * np.sum(
                    np.log(2 * np.pi * sigma2_zhat[k])
                    + (diff**2) / sigma2_zhat[k],
                    axis=1,
                )
                logpy_zhat[start:end, k] = logpy[k] + log_likelihood

        cluster = tf.argmax(logpy_zhat, axis=-1).numpy()
        logpy_zhat, _, final_labels = self._remove_small_clusters(
            logpy_zhat, cluster, min_n_obs
        )

        cluster_key = f"cluster_{component}_integrated"
        logpy_key = f"logpy_zhat_{component}_integrated"
        self.mdata.mod[modality].obs[cluster_key] = final_labels
        self.mdata.mod[modality].obsm[logpy_key] = logpy_zhat

        return logpy_zhat

    def compute_neighbors_with_same_annotations(
        self,
        modality: str,
        use_cluster: str,
        use_rep: Union[str, np.array],
        n_neighbors: Union[int, Sequence[int]] = list(range(5, 25)),
    ) -> pd.DataFrame:
        """Perform K-nearest neighbor analysis.

        Parameters
        ----------
        modality: str
            the modality to used.

        use_cluster: str
            the column name of the clusters in the obs of modality.

        use_rep: Union[str, np.array]
            the key of obsm of modality to used to compute the distance
            within and between clusters. Alternatively, the array will
            be used if provided with np.array,

        n_neighbors: Union[int, Sequence[int]], optional
            the number of neighbors to be analyzed, Defaults to
            list(range(5, 25))

        Returns
        -------
        pd.DataFrame
            analysis result for K-nearest neighbor. The DataFrame
            contains 3 columns, the first column is the number of
            neighbors (K), the second column is the cluster
            identifiers, the third column specify the proportion of KNN
            samples with the same cluster,

        Raises
        ------
        KeyError
            if use_cluster is not in the obs of the modality in
            self.mdata. (please perform compute_cluster_log_probability
            first for unsupervised clustering).
        """
        if use_cluster not in self.mdata.mod[modality].obs:
            message = f"{use_cluster} not in obs DataFrame of the modality."
            raise KeyError(message)

        proportions_series = list()
        clusters_series = list()
        n_neighbors_series = list()
        if isinstance(n_neighbors, (int, float)):
            n_neighbors = [n_neighbors]

        if isinstance(use_rep, np.ndarray):
            self.mdata[modality].obsm["z_custom"] = use_rep
            use_rep = "z_custom"

        for k in tqdm(n_neighbors):
            if isinstance(k, float):
                k = int(k)
                message = (
                    "Expect int for element in n_neighbors, transform float to int."
                )
                warnings.warn(message, RuntimeWarning)

            scanpy.pp.neighbors(
                self.mdata[modality], n_neighbors=k + 1, use_rep=use_rep
            )
            proportions_k = list()
            for i, j in enumerate(self.mdata[modality].obsp["distances"]):
                cluster = self.mdata[modality].obs.iloc[i][use_cluster]
                neighbor_clusters = self.mdata[modality].obs.iloc[j.indices][
                    use_cluster
                ]
                proportions_k.append((neighbor_clusters == cluster).sum() / k)
                clusters_series += [cluster]

            proportions_series.append(np.array(proportions_k))
            n_neighbors_series.append(np.array([k] * len(proportions_k)))

        analysis_result = pd.DataFrame(
            {
                "Number of Neighbors": np.concatenate(n_neighbors_series),
                "Cluster": clusters_series,
                "% of KNN Cells with the Same Cluster": np.concatenate(
                    proportions_series
                ),
            }
        )

        analysis_result["KNN Cells with the Same Cluster"] = (
            analysis_result["% of KNN Cells with the Same Cluster"]
            * analysis_result["Number of Neighbors"]
        )
        num_cells = self.mdata[modality].obs["cell_type"].value_counts()
        total_num_cells = num_cells.sum()
        analysis_result["Number of Cells"] = analysis_result["Cluster"].map(num_cells)
        analysis_result["Cell Enrichment Score"] = (
            analysis_result["% of KNN Cells with the Same Cluster"]
            - analysis_result["Number of Cells"] / total_num_cells
        ).clip(lower=0) / (1 - analysis_result["Number of Cells"] / total_num_cells)

        return analysis_result

    def compute_contingency_matrix(
        self, modality: str, cluster_colname_a: str, cluster_colname_b: str
    ) -> pd.DataFrame:
        """Compute the contingency matrix of two clustering results.

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
            contingency matrix where the indices are the cluster names
            in the first clustering assignment, the column names are
            the second clustering assignment.

        """
        encoder_cluster_a = LabelEncoder()
        encoder_cluster_b = LabelEncoder()

        encoded_array_a = encoder_cluster_a.fit_transform(
            self.mdata[modality].obs[cluster_colname_a]
        )
        encoded_array_b = encoder_cluster_b.fit_transform(
            self.mdata[modality].obs[cluster_colname_b]
        )

        contigency_matrix = pd.DataFrame(
            contingency_matrix(encoded_array_a, encoded_array_b)
        )
        contigency_matrix.index = encoder_cluster_a.classes_
        contigency_matrix.columns = encoder_cluster_b.classes_

        return contigency_matrix

    def compute_classification_metrics(
        self, modality: str, cluster_colname_a: str, cluster_colname_b: str
    ) -> pd.DataFrame:
        """Compute the classification metrics if using the second
        clustering assignment to predict the first cluster assignment.

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
        contingency_matrix = self.compute_contingency_matrix(
            modality, cluster_colname_a, cluster_colname_b
        )

        result = []
        for target_a in contingency_matrix.index:
            for target_b in contingency_matrix.columns:
                FP = (
                    contingency_matrix[target_b].sum()
                    - contingency_matrix[target_b][target_a]
                )
                TP = contingency_matrix[target_b][target_a]
                FN = (
                    contingency_matrix.loc[target_a].sum()
                    - contingency_matrix[target_b][target_a]
                )
                TN = (
                    np.sum(contingency_matrix.values)
                    - contingency_matrix[target_b].sum()
                    - contingency_matrix.loc[target_a].sum()
                    + contingency_matrix[target_b][target_a]
                )

                sensitivity = TP / (TP + FN)
                specificity = TN / (TN + FP)
                precision = TP / (TP + FP)
                accuracy = (TP + TN) / (TP + TN + FP + FN)
                f1 = 2 * precision * sensitivity / (precision + sensitivity + 1e-3)

                result.append(
                    [
                        target_a,
                        target_b,
                        f1,
                        accuracy,
                        sensitivity,
                        specificity,
                        precision,
                    ]
                )

        colnames = [
            "Cluster A",
            "Cluster B",
            "F1 Score",
            "Accuracy",
            "Sensitivity",
            "Specificity",
            "Precision",
        ]

        return pd.DataFrame(result, columns=colnames)
