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
        params = self._extract_gmm_parameters(z_prior_parameterizer)
        dist_z, dist_z_y, logpy = self._build_gmm_distribution(
            params["mu"], params["sigma2"], params["pi"]
        )

        z_all = []
        dataloader = DataLoader(
            self.mdata, batch_size, batch_effect_colnames, distribution_names
        )
        for batch_data in tqdm(dataloader):
            outputs = self.model(batch_data, training=False)
            z_all.append(outputs[f"{component}_z"].numpy())
        z_all = np.vstack(z_all)

        logpy_z = self._score_and_cluster(
            z_all,
            dist_z,
            dist_z_y,
            logpy.numpy(),
            modality,
            f"cluster_{component}",
            f"logpy_z_{component}",
            min_n_obs,
            batch_size,
        )

        return logpy_z

    def compute_cluster_log_probability_z_hat(
        self,
        modality: str,
        component: str,
        batch_effect_colnames: Optional[Dict[str, List[str]]] = None,
        distribution_names: Optional[Dict[str, str]] = None,
        batch_size: int = 128,
        min_n_obs=36,
    ) -> np.array:
        """Compute direct learned-prior cluster log probabilities in z_hat space."""
        component_obj = self.model.components[component]
        prior = component_obj.z_hat_prior_parameterizer
        if prior is None:
            raise ValueError(
                f"Component '{component}' does not have z_hat prior parameterizer."
            )

        params = self._extract_gmm_parameters(prior)
        dist_z, dist_z_y, logpy = self._build_gmm_distribution(
            params["mu"], params["sigma2"], params["pi"]
        )

        z_hat_all = []
        dataloader = DataLoader(
            self.mdata, batch_size, batch_effect_colnames, distribution_names
        )
        for batch_data in tqdm(dataloader):
            outputs = self.model(batch_data, training=False)
            z_hat_all.append(outputs[f"{component}_z_hat"].numpy())
        z_hat_all = np.vstack(z_hat_all)

        logpy_zhat = self._score_and_cluster(
            z_hat_all,
            dist_z,
            dist_z_y,
            logpy.numpy(),
            modality,
            f"cluster_{component}_integrated",
            f"logpy_zhat_{component}_integrated",
            min_n_obs,
            batch_size,
        )

        return logpy_zhat

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
    def _compute_bhattacharyya_distance(
        mu1: np.ndarray, sigma2_1: np.ndarray, mu2: np.ndarray, sigma2_2: np.ndarray
    ) -> float:
        """Compute Bhattacharyya distance between two diagonal Gaussians.

        D_B = 0.125 * sum((μ₁-μ₂)² / (σ₁²+σ₂²)) + 0.5 * sum(log((σ₁²+σ₂²)² / (4*σ₁²*σ₂²)))

        Parameters
        ----------
        mu1 : np.ndarray, shape (D,)
            Mean of first Gaussian.
        sigma2_1 : np.ndarray, shape (D,)
            Variance of first Gaussian.
        mu2 : np.ndarray, shape (D,)
            Mean of second Gaussian.
        sigma2_2 : np.ndarray, shape (D,)
            Variance of second Gaussian.

        Returns
        -------
        float
            Bhattacharyya distance (non-negative).
        """
        diff_sq = (mu1 - mu2) ** 2
        sigma_sum = sigma2_1 + sigma2_2
        term1 = 0.125 * np.sum(diff_sq / sigma_sum)
        term2 = 0.5 * np.sum(np.log(sigma_sum**2 / (4 * sigma2_1 * sigma2_2)))
        return term1 + term2

    @staticmethod
    def _merge_similar_gmm_components(
        mu: np.ndarray,
        sigma2: np.ndarray,
        pi: np.ndarray,
        bc_threshold: float = 0.8,
        max_k: Optional[int] = 100,
    ):
        """Merge similar GMM components using Bhattacharyya coefficient.

        Single-pass merging: compute pairwise BC, build graph of similar pairs,
        find connected components, merge all clusters within each component.

        Parameters
        ----------
        mu : np.ndarray, shape (K, D)
            Means of GMM components.
        sigma2 : np.ndarray, shape (K, D)
            Variances of GMM components.
        pi : np.ndarray, shape (K,)
            Mixing weights of GMM components.
        bc_threshold : float, optional
            Merge clusters with BC > bc_threshold (i.e., D_B < -log(bc_threshold)).
            bc_threshold=0.8 corresponds to D_B < 0.22. Defaults to 0.8.
        max_k : int, optional
            Safety cap on final number of clusters. If set, uses agglomerative
            clustering to reduce to max_k components. Defaults to 100.

        Returns
        -------
        tuple
            (mu_merged, sigma2_merged, pi_merged) with merged components.
        """
        n_clusters = len(pi)

        if max_k is not None and n_clusters <= max_k:
            return mu, sigma2, pi

        diff = mu[:, np.newaxis, :] - mu[np.newaxis, :, :]
        diff_sq = diff ** 2
        sigma_sum = sigma2[:, np.newaxis, :] + sigma2[np.newaxis, :, :]

        term1 = 0.125 * np.sum(diff_sq / sigma_sum, axis=2)
        term2 = 0.5 * np.sum(
            np.log(sigma_sum ** 2 / (4 * sigma2[:, np.newaxis, :] * sigma2[np.newaxis, :, :])),
            axis=2,
        )
        D = term1 + term2
        BC = np.exp(-D)

        adjacency = (BC > bc_threshold).astype(int)
        np.fill_diagonal(adjacency, 0)

        labels = np.full(n_clusters, -1, dtype=int)
        current_label = 0

        for i in tqdm(range(n_clusters), desc="Finding connected components"):
            if labels[i] == -1:
                queue = [i]
                labels[i] = current_label
                while queue:
                    node = queue.pop(0)
                    neighbors = np.where(adjacency[node] > 0)[0]
                    for neighbor in neighbors:
                        if labels[neighbor] == -1:
                            labels[neighbor] = current_label
                            queue.append(neighbor)
                current_label += 1

        print(f"[Merging] Found {current_label} connected components from {n_clusters} initial clusters")
        component_sizes = [np.sum(labels == i) for i in range(current_label)]
        print(f"[Merging] Component sizes: min={min(component_sizes)}, max={max(component_sizes)}, mean={np.mean(component_sizes):.1f}")

        if max_k is not None and current_label > max_k:
            from scipy.cluster.hierarchy import fcluster, linkage
            from scipy.spatial.distance import squareform

            Z = linkage(squareform(D), method="average")
            labels = fcluster(Z, t=max_k, criterion="maxclust") - 1
            current_label = max_k

        mu_merged = []
        sigma2_merged = []
        pi_merged = []

        for label in tqdm(range(current_label), desc="Merging cluster distributions"):
            mask = labels == label
            pi_cluster = pi[mask]
            pi_sum = np.sum(pi_cluster)

            mu_m = np.average(mu[mask], axis=0, weights=pi_cluster)
            sigma2_m = np.average(
                sigma2[mask] + (mu[mask] - mu_m) ** 2, axis=0, weights=pi_cluster
            )

            mu_merged.append(mu_m)
            sigma2_merged.append(sigma2_m)
            pi_merged.append(pi_sum)

        return np.array(mu_merged), np.array(sigma2_merged), np.array(pi_merged)

    @staticmethod
    def _build_gmm_distribution(mu: np.ndarray, sigma2: np.ndarray, pi: np.ndarray):
        """Build TensorFlow GMM distributions from parameters.

        Parameters
        ----------
        mu : np.ndarray, shape (K, D)
            Means of GMM components.
        sigma2 : np.ndarray, shape (K, D)
            Variances of GMM components.
        pi : np.ndarray, shape (K,)
            Mixing weights of GMM components.

        Returns
        -------
        tuple
            (dist_z, dist_z_y, logpy) where:
            - dist_z: MixtureMultivariateNormalDiagDistribution
            - dist_z_y: MultivariateNormalDiagDistribution (batched)
            - logpy: log mixing weights, shape (K,)
        """
        mu_tf = tf.convert_to_tensor(mu, dtype=tf.float32)
        sigma2_tf = tf.convert_to_tensor(sigma2, dtype=tf.float32)
        pi_tf = tf.convert_to_tensor(pi, dtype=tf.float32)

        dist_z_y = MultivariateNormalDiagDistribution(loc=mu_tf, scale_diag=tf.sqrt(sigma2_tf))

        logits = tf.math.log(pi_tf + 1e-7)
        params = tf.concat([tf.expand_dims(logits, -1), mu_tf, tf.sqrt(sigma2_tf)], axis=-1)
        dist_z = MixtureMultivariateNormalDiagDistribution.from_parameterizer_output(params)

        logpy = tf.math.log(pi_tf + 1e-7)

        return dist_z, dist_z_y, logpy

    def _score_and_cluster(
        self,
        z: np.ndarray,
        dist_z,
        dist_z_y,
        logpy: np.ndarray,
        modality: str,
        cluster_key: str,
        logpy_key: str,
        min_n_obs: int,
        batch_size: int = 128,
    ):
        """Score data against GMM, apply Bayes theorem, remove small clusters.

        Parameters
        ----------
        z : np.ndarray, shape (N, D)
            Latent representations to score.
        dist_z : MixtureMultivariateNormalDiagDistribution
            Full mixture distribution.
        dist_z_y : MultivariateNormalDiagDistribution
            Individual cluster distributions.
        logpy : np.ndarray, shape (K,)
            Log mixing weights.
        modality : str
            Modality name for storing results.
        cluster_key : str
            Key for storing cluster labels in obs.
        logpy_key : str
            Key for storing log probabilities in obsm.
        min_n_obs : int
            Minimum observations to keep a cluster.
        batch_size : int, optional
            Batch size for scoring. Defaults to 128.

        Returns
        -------
        np.ndarray
            logpy_z, shape (N, K), posterior log probabilities.
        """
        N = z.shape[0]
        logpy_z = np.zeros((N, len(logpy)))

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            z_batch = tf.convert_to_tensor(z[start:end], dtype=tf.float32)
            logpz_y = dist_z_y.log_prob(tf.expand_dims(z_batch, -2))
            logpz = tf.expand_dims(dist_z.log_prob(z_batch), -1)
            logpy_z[start:end] = (logpy + logpz_y - logpz).numpy()

        cluster = tf.argmax(logpy_z, axis=-1).numpy()
        initial_k = len(np.unique(cluster))
        logpy_z, cluster_final, final_labels = self._remove_small_clusters(
            logpy_z, cluster, min_n_obs
        )

        final_k = len(np.unique(cluster_final))
        print(
            f"[Clustering] {cluster_key}: initial K = {initial_k}, "
            f"min_n_obs = {min_n_obs}, final K = {final_k}, "
            f"removed {initial_k - final_k} cluster(s) "
            f"(stored in obs['{cluster_key}'], obsm['{logpy_key}'])"
        )

        self.mdata.mod[modality].obs[cluster_key] = final_labels
        self.mdata.mod[modality].obsm[logpy_key] = logpy_z

        return logpy_z

    @staticmethod
    def _recombine_gmm_parameters(child_params, parent_params, hierarchical_encoder):
        """Recombine child and parent GMM parameters through hierarchical encoder.

        Computes the integrated GMM parameters in z_hat space by projecting
        parent and child GMM parameters through the learned hierarchical encoder
        weights.

        Parameters
        ----------
        child_params : dict
            Child component GMM parameters (from _extract_gmm_parameters).
        parent_params : list of dict
            Parent component GMM parameters (from _extract_gmm_parameters).
        hierarchical_encoder : HierarchicalEncoder
            The hierarchical encoder with r_network and b_network.

        Returns
        -------
        tuple
            (mu_zhat, sigma2_zhat, pi_zhat) for the integrated GMM.
        """
        W_r = hierarchical_encoder.r_network.get_weights()[0]
        W_b = hierarchical_encoder.b_network.get_weights()[0]

        parent_dims = [p["D"] for p in parent_params]
        W_parent_parts = []
        offset = 0
        for dim in parent_dims:
            W_parent_parts.append(W_b[offset : offset + dim, :])
            offset += dim
        W_child_raw = W_b[offset:, :]
        W_child_effective = W_r @ W_child_raw
        W_child_effective_sigma2 = (W_r**2) @ (W_child_raw**2)

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
            sigma2_zhat[idx] += W_child_effective_sigma2.T @ child_params["sigma2"][
                child_idx
            ]

            pi_zhat[idx] = child_params["pi"][child_idx]
            for p_idx, p_params in zip(parent_idx, parent_params):
                pi_zhat[idx] *= p_params["pi"][p_idx]

        return mu_zhat, sigma2_zhat, pi_zhat

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

    @staticmethod
    def _label_small_clusters(
        cluster: np.ndarray,
        min_n_obs: int,
        small_cluster_label: str = "Unassigned",
    ) -> List[str]:
        """Assign ``small_cluster_label`` to clusters with fewer than
        ``min_n_obs`` observations.

        Unlike ``_remove_small_clusters``, this method does not remove
        any clusters or recompute hard assignments. Observations that
        belong to a cluster with fewer than ``min_n_obs`` observations
        are labeled with ``small_cluster_label``, while all other
        observations keep their original formatted cluster label.

        Parameters
        ----------
        cluster : np.ndarray
            initial hard cluster assignments (n_samples,).

        min_n_obs : int
            minimum number of observations required to keep a cluster.

        small_cluster_label : str, optional
            label to assign to observations in small clusters. Defaults
            to "Unassigned".

        Returns
        -------
        List[str]
            formatted cluster labels, one per observation.

        """
        unique, counts = np.unique(cluster, return_counts=True)
        small_clusters = set(unique[counts < min_n_obs])
        labels = [
            small_cluster_label if c in small_clusters else f"Cluster {c:03d}"
            for c in cluster
        ]
        return labels

    def compute_integrated_cluster_log_probability(
        self,
        modality: str,
        component: str,
        batch_size: int = 128,
        min_n_obs: int = 36,
        small_cluster_label: str = "Unassigned",
        bc_threshold: float = 0.8,
        max_k: Optional[int] = 100,
    ) -> np.array:
        """Compute the log probability of a sample being assigned to
        each *integrated* cluster in z_hat space for a hierarchical
        component.

        This is the legacy analytical/post-hoc z_hat route that recombines
        parent/child priors through hierarchical encoder weights.

        Unlike ``compute_cluster_log_probability`` (which clusters in
        ``z`` space of a single component), this method projects the
        GMM parameters of every parent component **and** the child
        component through the learned hierarchical-encoder weights
        (``r_network`` and ``b_network``) to build a joint Gaussian
        mixture in z_hat space.  Samples are then scored against that
        joint mixture.

        Similar integrated clusters are merged using Bhattacharyya
        coefficient (BC) before scoring, reducing the combinatorial
        explosion from the Cartesian product of parent and child clusters.
        BC = exp(-D_B), where D_B is Bhattacharyya distance.

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
            clusters with fewer than this many observations are removed
            and their members reassigned to remaining clusters.
            Defaults to 36.
        small_cluster_label : str, optional
            (Deprecated, unused — small clusters are now removed and
            reassigned via ``_remove_small_clusters``.)
        bc_threshold : float, optional
            Merge clusters with BC > bc_threshold (i.e., D_B < -log(bc_threshold)).
            bc_threshold=0.8 corresponds to D_B < 0.22. Defaults to 0.8.
        max_k : int, optional
            Safety cap on final number of clusters. Defaults to 100.

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
        for parent_name in parent_names:
            parent_parameterizer = self.model.components[
                parent_name
            ].z_prior_parameterizer
            parent_params.append(self._extract_gmm_parameters(parent_parameterizer))

        child_parameterizer = comp.z_prior_parameterizer
        child_params = self._extract_gmm_parameters(child_parameterizer)

        mu_zhat, sigma2_zhat, pi_zhat = self._recombine_gmm_parameters(
            child_params, parent_params, comp.hierarchical_encoder
        )

        n_initial = len(pi_zhat)
        print(f"[Integrated Clustering] Initial clusters (Cartesian product): {n_initial}")
        print("[Integrated Clustering] Weight distribution before merging:")
        print(f"  Top 5 weights: {np.sort(pi_zhat)[-5:][::-1]}")
        print(f"  Sum of top 5: {np.sum(np.sort(pi_zhat)[-5:]):.4f}")
        print(f"  Max weight: {np.max(pi_zhat):.4f}")

        mu_zhat, sigma2_zhat, pi_zhat = self._merge_similar_gmm_components(
            mu_zhat, sigma2_zhat, pi_zhat, bc_threshold, max_k
            )

        n_merged = len(pi_zhat)
        print(f"[Integrated Clustering] After merging (bc_threshold={bc_threshold}): {n_merged} clusters")
        if n_initial > n_merged:
            print(f"[Integrated Clustering] Merged {n_initial - n_merged} redundant clusters")
        print("[Integrated Clustering] Weight distribution after merging:")
        print(f"  Top 5 weights: {np.sort(pi_zhat)[-5:][::-1]}")
        print(f"  Sum of top 5: {np.sum(np.sort(pi_zhat)[-5:]):.4f}")
        print(f"  Max weight: {np.max(pi_zhat):.4f}")

        dist_z, dist_z_y, logpy = self._build_gmm_distribution(
            mu_zhat, sigma2_zhat, pi_zhat
        )

        z_hat_all = self.mdata.mod[modality].obsm[f"z_hat_{component}"]

        logpy_zhat = self._score_and_cluster(
            z_hat_all,
            dist_z,
            dist_z_y,
            logpy.numpy(),
            modality,
            f"cluster_{component}_integrated",
            f"logpy_zhat_{component}_integrated",
            min_n_obs,
            batch_size,
        )

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
