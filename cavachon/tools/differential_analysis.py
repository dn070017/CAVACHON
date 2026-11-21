import warnings
from itertools import combinations
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import muon as mu
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from cavachon.dataloader.dataloader import DataLoader
from cavachon.environment.constants import Constants
from cavachon.utils.reflection_handler import ReflectionHandler


class DifferentialAnalysis:
    """DifferentialAnalysis

    Differential analysis between two groups of samples.

    Attributes
    ----------
    mdata: muon.MuData
        the MuData for analysis.

    model: tf.keras.Model
        the trained generative model.

    """

    # default K above which a difference counts as significant
    SIGNIFICANCE_THRESHOLD = 3.2

    def __init__(
        self,
        mdata: mu.MuData,
        model: tf.keras.Model,
        batch_effect_colnames: Optional[Dict[str, List[str]]] = None,
        distribution_names: Optional[Dict[str, str]] = None,
        batch_effect_encoders: Optional[Dict[str, Dict[str, LabelEncoder]]] = None,
    ) -> None:
        """Constructor for DifferentialAnalysis.

        Parameters
        ----------
        mdata: muon.MuData
            the MuData for analysis.

        model: tf.keras.Model
            the trained generative model.

        batch_effect_colnames: Dict[str, List[str]], optional
            the batch effect columns for each modality. Modalities not
            listed (or None) use adata.uns['cavachon'], as in
            DataLoader. Defaults to None.

        distribution_names: Dict[str, str], optional
            the distribution names for each modality. Modalities not
            listed (or None) use adata.uns['cavachon'], as in
            DataLoader. Defaults to None.

        batch_effect_encoders: Dict[str, Dict[str, LabelEncoder]], optional
            the fitted batch effect encoders of each modality. Defaults
            to None (fitted from the data).

        """
        self.mdata = mdata
        self.model = model
        # DataLoader takes mappings (empty: everything from adata.uns)
        self.batch_effect_colnames: Mapping[str, List[str]] = (
            batch_effect_colnames or {}
        )
        self.distribution_names: Mapping[str, str] = distribution_names or {}
        self.batch_effect_encoders: Dict[str, Dict[str, LabelEncoder]] = (
            batch_effect_encoders or {}
        )

    def across_clusters_pairwise(
        self,
        component: str,
        modality: str,
        use_cluster: str,
        z_sampling_size: int = 5,
        x_sampling_size: int = 1000,
        batch_size: int = 128,
        keep_only_significant: bool = False,
        sort_output: bool = True,
        seed: Optional[int] = None,
        significance_threshold: float = SIGNIFICANCE_THRESHOLD,
    ) -> Mapping[str, pd.DataFrame]:
        """Perform differential analysis between every groups of
        samples.

        Parameters
        ----------
        component: str
            generative result of `modality` from which component to
            used.

        modality: str
            which modality to used from the generative result of
            `component`.

        use_cluster: str
            the cluster annotation in the obs of `modality`.

        z_sampling_size: int, optional
            how many z to sample, by default 10.

        x_sampling_size: int, optional
            how many x to sample, by default 1000.

        batch_size: int, optional
            batch size used for the forward pass, by default 128.

        keep_only_significant: bool, optional
            keep only the significantly differentially expressed
            entity. Default to False.

        seed: Optional[int], optional
            NumPy random seed, applied at the start of every comparison.
            Defaults to None.

        significance_threshold: float, optional
            K above which a feature is marked ``Significant`` (see
            ``between_two_groups``). Defaults to 3.2.

        Returns
        -------
        Mapping[str, pd.DataFrame]
            analysis result for differential analysis. The keys are the
            pair of cluster, such as "cluster A/cluster B". The values
            are the following DataFrame, containing 6 columns:
            1.  expected values of groups A
            2.  expected values of groups B
            3.  the probability P(A>B|Z)
            4.  the probability P(B>A|Z)
            5.  the Bayesian factor of K(A>B|Z)
            6.  the Bayesian factor of K(B>A|Z)
        """
        results: Dict[str, pd.DataFrame] = dict()
        obs = self.mdata[modality].obs
        unique_clusters = obs[use_cluster].unique()
        for cluster_a, cluster_b in tqdm(
            combinations(unique_clusters, r=2),
            desc="Across clusters pairwise",
        ):
            index_a = obs[obs[use_cluster] == cluster_a].index
            index_b = obs[obs[use_cluster] == cluster_b].index
            deg = self.between_two_groups(
                group_a_index=index_a,
                group_b_index=index_b,
                component=component,
                modality=modality,
                z_sampling_size=z_sampling_size,
                x_sampling_size=x_sampling_size,
                batch_size=batch_size,
                desc=f"Between {cluster_a} and {cluster_b}",
                sort_output=sort_output,
                group_a_label=str(cluster_a),
                group_b_label=str(cluster_b),
                use_cluster=use_cluster,
                seed=seed,
                significance_threshold=significance_threshold,
            )
            if keep_only_significant:
                deg = self.keep_significant(deg)

            results.setdefault(f"{cluster_a}/{cluster_b}", deg)

        return results

    def across_clusters(
        self,
        component: str,
        modality: str,
        use_cluster: str,
        z_sampling_size: int = 5,
        x_sampling_size: int = 1000,
        batch_size: int = 128,
        keep_only_significant: bool = False,
        sort_output: bool = True,
        seed: Optional[int] = None,
        significance_threshold: float = SIGNIFICANCE_THRESHOLD,
    ) -> Mapping[str, pd.DataFrame]:
        """Perform differential analysis between one specific group and
        the rest of the samples.

        Parameters
        ----------
        component: str
            generative result of `modality` from which component to
            used.

        modality: str
            which modality to used from the generative result of
            `component`.

        use_cluster: str
            the cluster annotation in the obs of `modality`.

        z_sampling_size: int, optional
            how many z to sample, by default 10.

        x_sampling_size: int, optional
            how many x to sample, by default 1000.

        batch_size: int, optional
            batch size used for the forward pass, by default 128.

        keep_only_significant: bool, optional
            keep only the significantly differentially expressed
            entity. Default to False.

        seed: Optional[int], optional
            NumPy random seed, applied at the start of every comparison.
            Defaults to None.

        significance_threshold: float, optional
            K above which a feature is marked ``Significant`` (see
            ``between_two_groups``). Defaults to 3.2.

        Returns
        -------
        Mapping[str, pd.DataFrame]
            analysis result for differential analysis. The keys are the
            cluster name. The values are the following DataFrame,
            containing 6 columns:
            1.  expected values of groups A
            2.  expected values of groups B
            3.  the probability P(A>B|Z)
            4.  the probability P(B>A|Z)
            5.  the Bayesian factor of K(A>B|Z)
            6.  the Bayesian factor of K(B>A|Z)
        """
        results: Dict[str, pd.DataFrame] = dict()
        obs = self.mdata[modality].obs
        unique_clusters = obs[use_cluster].unique()
        for cluster in tqdm(unique_clusters, desc="Across clusters"):
            index_a = obs[obs[use_cluster] == cluster].index
            index_b = obs[obs[use_cluster] != cluster].index

            deg = self.between_two_groups(
                group_a_index=index_a,
                group_b_index=index_b,
                component=component,
                modality=modality,
                z_sampling_size=z_sampling_size,
                x_sampling_size=x_sampling_size,
                batch_size=batch_size,
                desc=f"Between {cluster} and others",
                sort_output=sort_output,
                group_a_label=str(cluster),
                group_b_label="others",
                use_cluster=use_cluster,
                seed=seed,
                significance_threshold=significance_threshold,
            )
            if keep_only_significant:
                deg = self.keep_significant(deg)

            results.setdefault(cluster, deg)

        return results

    def between_two_groups(
        self,
        group_a_index: Union[pd.Index, Sequence[str]],
        group_b_index: Union[pd.Index, Sequence[str]],
        component: str,
        modality: str,
        z_sampling_size: int = 10,
        x_sampling_size: int = 2500,
        batch_size: int = 128,
        desc: str = "",
        sort_output: bool = True,
        group_a_label: Optional[str] = None,
        group_b_label: Optional[str] = None,
        use_cluster: Optional[str] = None,
        seed: Optional[int] = None,
        significance_threshold: float = SIGNIFICANCE_THRESHOLD,
    ) -> pd.DataFrame:
        """Perform the differential analysis between two groups.

        Parameters
        ----------
        group_a_index : Union[pd.Index, Sequence[str]]
            index of group one. Needs to meet the index in the obs of
            the modality.

        group_b_index : Union[pd.Index, Sequence[str]]
            index of group two. Needs to meet the index in the obs of
            the modality.

        component : str
            generative result of `modality` from which component to
            used.

        modality : str
            which modality to used from the generative result of
            `component`.

        z_sampling_size: int, optional
            how many z to sample, by default 10.

        x_sampling_size: int, optional
            how many x to sample, by default 2500.

        batch_size: int, optional
            batch size used for the forward pass. Defaults to 128.

        desc: str, optional
            description for the progress bar.

        sort_output: bool, optional
            Sort the result by the larger of the Bayesian factors
            K(A>B|Z) and K(B>A|Z) in descending order (see
            ``bayes_factor_strength``). Default to True.

        group_a_label: Optional[str], optional
            Human-readable label for group A. If provided, a ``GroupA``
            column is added to the result.

        group_b_label: Optional[str], optional
            Human-readable label for group B. If provided, a ``GroupB``
            column is added to the result.

        use_cluster: Optional[str], optional
            the obs column the group labels come from. If provided, a
            ``UseCluster`` column is added to the result.

        seed: Optional[int], optional
            NumPy random seed, applied before sampling. Defaults to None.

        significance_threshold: float, optional
            a feature is marked ``Significant`` when ``K(A>B|Z)`` or
            ``K(B>A|Z)`` reaches it (see ``mark_significant``). Defaults
            to 3.2.

        Returns
        -------
        pd.DataFrame
            analysis result for differential analysis. The DataFrame
            contains the original 6 columns, ``Significant``, ``GroupA``
            and ``GroupB`` when labels are provided, then the parameters
            of the run (see ``record_run_parameters``): ``Component``,
            ``Modality``, ``UseCluster`` (when provided),
            ``ZSamplingSize``, ``XSamplingSize``, ``RandomSeed`` and
            ``SignificanceThreshold``. The parameters let a table read
            back from disk be matched to the analysis that produced it.

        """
        if seed is not None:
            np.random.seed(seed)

        x_means_a_list: List[np.ndarray] = []
        x_means_b_list: List[np.ndarray] = []

        modality_names = self.mdata.mod.keys()

        # holds a list of per-batch tensors first, then their concatenation
        batch_effect: Dict[str, Any] = dict()
        dataloader = DataLoader(
            self.mdata,
            batch_size,
            self.batch_effect_colnames,
            self.distribution_names,
            self.batch_effect_encoders,
        )

        for batch in dataloader:
            for modality_name in modality_names:
                if modality_name not in batch_effect:
                    batch_effect.setdefault(modality_name, [])
                batch_effect_key = f"{modality_name}_{Constants.TENSOR_NAME_BATCH}"
                batch_effect[modality_name].append(batch.get(batch_effect_key))

        for modality_name in batch_effect.keys():
            batch_effect[modality_name] = tf.concat(batch_effect[modality_name], axis=0)

        for _ in tqdm(range(z_sampling_size), desc=desc, leave=False):
            mdata_group_a = self.sample_mdata_x(
                index=group_a_index, x_sampling_size=x_sampling_size
            )
            mdata_group_b = self.sample_mdata_x(
                index=group_b_index, x_sampling_size=x_sampling_size
            )
            dataloader_group_a = DataLoader(
                mdata_group_a,
                batch_size,
                self.batch_effect_colnames,
                self.distribution_names,
                self.batch_effect_encoders,
            )
            dataloader_group_b = DataLoader(
                mdata_group_b,
                batch_size,
                self.batch_effect_colnames,
                self.distribution_names,
                self.batch_effect_encoders,
            )
            batch_size = min(
                dataloader_group_a.batch_size, dataloader_group_b.batch_size
            )
            x_means_a_list.append(
                self.compute_x_means(
                    dataset=dataloader_group_a.dataset,
                    component=component,
                    modality=modality,
                    batch_effect=batch_effect,
                    batch_size=batch_size,
                )
            )
            x_means_b_list.append(
                self.compute_x_means(
                    dataset=dataloader_group_b.dataset,
                    component=component,
                    modality=modality,
                    batch_effect=batch_effect,
                    batch_size=batch_size,
                )
            )

        x_means_a = np.vstack(x_means_a_list)
        x_means_b = np.vstack(x_means_b_list)
        index = self.mdata.mod[modality].var.index

        result = self.compute_bayesian_factor(x_means_a, x_means_b, index)
        self.mark_significant(result, significance_threshold)
        if sort_output:
            result = self.sort_by_bayes_factor(result)

        if group_a_label is not None:
            result[Constants.DIFFERENTIAL_ANALYSIS_GROUP_A_COLUMN] = group_a_label
        if group_b_label is not None:
            result[Constants.DIFFERENTIAL_ANALYSIS_GROUP_B_COLUMN] = group_b_label
        self.record_run_parameters(
            result,
            component=component,
            modality=modality,
            use_cluster=use_cluster,
            z_sampling_size=z_sampling_size,
            x_sampling_size=x_sampling_size,
            seed=seed,
            significance_threshold=significance_threshold,
        )

        return result

    @staticmethod
    def bayes_factor_strength(result: pd.DataFrame) -> pd.Series:
        """Evidence of a difference in either direction:
        ``max(K(A>B|Z), K(B>A|Z))``.

        The Bayes factors are taken with their sign. Without ties
        ``K(B>A|Z) = -K(A>B|Z)``, so this equals the largest ``|K|``;
        but a feature whose decoded means are equal in every comparison
        has ``P(A>B|Z) = P(B>A|Z) = 0`` and both K about -16, which is
        strong evidence of *no* difference, not of a difference.
        """
        return result[["K(A>B|Z)", "K(B>A|Z)"]].max(axis=1)

    @classmethod
    def mark_significant(
        cls, result: pd.DataFrame, threshold: float = SIGNIFICANCE_THRESHOLD
    ) -> None:
        """Add the ``Significant`` column (in place): True when
        ``K(A>B|Z) >= threshold`` or ``K(B>A|Z) >= threshold`` (see
        ``bayes_factor_strength``).

        This is the one definition of a significant feature; filters
        (``keep_only_significant``) and later analyses (the enrichment's
        ``min_num_degs``) read the column instead of applying their own
        threshold.
        """
        result[Constants.DIFFERENTIAL_ANALYSIS_SIGNIFICANT_COLUMN] = (
            cls.bayes_factor_strength(result) >= threshold
        )

    @staticmethod
    def keep_significant(result: pd.DataFrame) -> pd.DataFrame:
        """Keep the features marked ``Significant`` (see
        ``mark_significant``). Works on tables read back from disk too,
        where the column may hold ``"True"``/``"False"`` strings."""
        significant = result[Constants.DIFFERENTIAL_ANALYSIS_SIGNIFICANT_COLUMN]
        return result.loc[significant.astype(str).str.lower() == "true"]

    @classmethod
    def sort_by_bayes_factor(cls, result: pd.DataFrame) -> pd.DataFrame:
        """Sort the features by ``bayes_factor_strength``, strongest
        difference first (ties between the groups sort last)."""
        sort_key = cls.bayes_factor_strength(result)
        return result.loc[sort_key.sort_values(ascending=False).index]

    @staticmethod
    def record_run_parameters(
        result: pd.DataFrame,
        component: str,
        modality: str,
        use_cluster: Optional[str],
        z_sampling_size: int,
        x_sampling_size: int,
        seed: Optional[int],
        significance_threshold: float,
    ) -> None:
        """Record the parameters of a DEG/HDEG run on every row of its
        result (in place), with the same columns for both analyses:
        ``Component``, ``Modality``, ``UseCluster`` (skipped when None),
        ``ZSamplingSize``, ``XSamplingSize``, ``RandomSeed`` (empty
        when None) and ``SignificanceThreshold``."""
        result[Constants.DIFFERENTIAL_ANALYSIS_COMPONENT_COLUMN] = component
        result[Constants.DIFFERENTIAL_ANALYSIS_MODALITY_COLUMN] = modality
        if use_cluster is not None:
            result[Constants.DIFFERENTIAL_ANALYSIS_USE_CLUSTER_COLUMN] = use_cluster
        result[Constants.DIFFERENTIAL_ANALYSIS_Z_SAMPLING_SIZE_COLUMN] = z_sampling_size
        result[Constants.DIFFERENTIAL_ANALYSIS_X_SAMPLING_SIZE_COLUMN] = x_sampling_size
        result[Constants.DIFFERENTIAL_ANALYSIS_RANDOM_SEED_COLUMN] = seed
        result[Constants.DIFFERENTIAL_ANALYSIS_SIGNIFICANCE_THRESHOLD_COLUMN] = (
            significance_threshold
        )

    def sample_mdata_x(
        self, index: Union[pd.Index, Sequence[str]], x_sampling_size: int = 2500
    ) -> mu.MuData:
        """sample x from the mdata of samples with provided index.

        Parameters
        ----------
        index : Union[pd.Index, Sequence[str]]
            the samples to be sampled from.

        x_sampling_size: int, optional
            how many x to sample, by default 2500.

        Returns
        -------
        mu.MuData
            MuData with sampled data.

        """
        index_sampled = np.random.choice(index, x_sampling_size, replace=True)

        adata_dict = dict()
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=FutureWarning)
            warnings.simplefilter(action="ignore", category=UserWarning)
            for mod, adata in self.mdata.mod.items():
                adata_sampled = adata[index_sampled].copy()
                adata_sampled.obs_names_make_unique()
                adata_dict[mod] = adata_sampled

        return mu.MuData(adata_dict)

    def compute_x_means(
        self,
        dataset: tf.data.Dataset,
        component: str,
        modality: str,
        batch_effect: Mapping[str, tf.Tensor],
        training: bool = False,
        batch_size: int = 128,
    ) -> np.ndarray:
        """Compute the means of generative data.

        Parameters
        ----------
        dataset: tf.data.Dataset
            input dataset.

        component: str
            generative result of `modality` from which component to
            used.

        modality: str
            which modality to used from the generative result of
            `component`.

        batch_effect: Mapping[str, tf.Tensor]
            the batch effect tensors. The keys for the mapping are the
            modality names, the values are the corresponding batch
            effect tensors.

        training: bool
            if True, the forward pass will perform sampling with
            reparameterization. Otherwise, the mean value of the latent
            distribution is used. Defaults to False.

        batch_size: int, optional
            batch size used for the forward pass. Defaults to 128.

        Returns
        -------
        np.ndarray
            the means of the generative data distribution, where index
            i specify the samples, index j specify the means of the
            data distribution of the variables.

        """
        modality_names = self.mdata.mod.keys()
        dist_x_z_name = self.model.components.get(component).distribution_names.get(
            modality
        )
        dist_x_z_class = ReflectionHandler.get_class_by_name(
            dist_x_z_name, "distributions", "Distribution"
        )

        x_means = []
        for batch in dataset.batch(batch_size):
            encode_outputs = self.model.encode(batch, training=training)
            z = encode_outputs[Constants.MODEL_OUTPUTS_Z]

            hier_outputs = self.model.hierarchical_encode(batch, z, training=training)
            z_hat = hier_outputs[Constants.MODEL_OUTPUTS_Z_HAT]

            for modality_name in modality_names:
                batch_effect_key = f"{modality_name}_{Constants.TENSOR_NAME_BATCH}"
                n_obs_batch = batch[batch_effect_key].shape[0]

            random_batch_index = np.random.choice(
                np.arange(self.mdata.n_obs), n_obs_batch
            )
            for modality_name in modality_names:
                batch_effect_key = f"{modality_name}_{Constants.TENSOR_NAME_BATCH}"
                batch[batch_effect_key] = tf.gather(
                    batch_effect[modality_name], random_batch_index, axis=0
                )
            decode_outputs = self.model.decode(
                batch, z_hat, components=[component], training=training
            )
            x_parameters = decode_outputs[Constants.MODEL_OUTPUTS_X_PARAMS].get(
                f"{component}_{modality}_{Constants.MODEL_OUTPUTS_X_PARAMS}"
            )
            dist_x_z = dist_x_z_class.from_parameterizer_output(x_parameters)
            x_means.append(dist_x_z.mean().numpy())
        return np.vstack(x_means)

    def compute_bayesian_factor(
        self,
        x_means_a: np.ndarray,
        x_means_b: np.ndarray,
        index: Union[pd.Index, Sequence[str]],
    ) -> pd.DataFrame:
        """Compute the Bayesian factor of two groups.

        Parameters
        ----------
        x_means_a : np.ndarray
            the means of the generative data distribution of the first
            group.

        x_means_b : np.ndarray
            the means of the generative data distribution of the second
            group.

        index : Union[pd.Index, Sequence[str]]
            the index of the output DataFrame (var.index).

        Returns
        -------
        pd.DataFrame
            analysis result for differential analysis. The DataFrame
            contains 6 columns:
            1.  expected values of groups A
            2.  expected values of groups B
            3.  the probability P(A>B|Z)
            4.  the probability P(B>A|Z)
            5.  the Bayesian factor of K(A>B|Z)
            6.  the Bayesian factor of K(B>A|Z)

        """
        p_a_gt_b = np.mean(x_means_a > x_means_b, 0)
        p_a_leq_b = 1.0 - p_a_gt_b
        bayesian_factor_a_gt_b = np.log(p_a_gt_b + 1e-7) - np.log(p_a_leq_b + 1e-7)
        p_b_gt_a = np.mean(x_means_b > x_means_a, 0)
        p_b_leq_a = 1.0 - p_b_gt_a
        bayesian_factor_b_gt_a = np.log(p_b_gt_a + 1e-7) - np.log(p_b_leq_a + 1e-7)

        return pd.DataFrame(
            {
                "Mean(A)": np.mean(x_means_a, axis=0),
                "Mean(B)": np.mean(x_means_b, axis=0),
                "P(A>B|Z)": p_a_gt_b,
                "P(B>A|Z)": p_b_gt_a,
                "K(A>B|Z)": bayesian_factor_a_gt_b,
                "K(B>A|Z)": bayesian_factor_b_gt_a,
            },
            index=index,
        )
