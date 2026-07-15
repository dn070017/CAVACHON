import warnings
from itertools import combinations
from typing import Dict, List, Mapping, Optional, Sequence, Union

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

    def __init__(
        self,
        mdata: mu.MuData,
        model: tf.keras.Model,
        batch_effect_colnames: Optional[Dict[str, List[str]]] = None,
        distribution_names: Optional[Dict[str, str]] = None,
        batch_effect_encoders: Dict[str, Dict[str, LabelEncoder]] = dict(),
    ) -> None:
        """Constructor for DifferentialAnalysis.

        Parameters
        ----------
        mdata: muon.MuData
            the MuData for analysis.

        model: tf.keras.Model
            the trained generative model.

        batch_effect_colnames: Dict[str, List[str]], optional
            the batch effect columns for each modality. Defaults to
            None.

        distribution_names: Dict[str, str], optional
            the distribution names for each modality. Defaults to
            None.

        """
        self.mdata = mdata
        self.model = model
        self.batch_effect_colnames = batch_effect_colnames
        self.distribution_names = distribution_names
        self.batch_effect_encoders = batch_effect_encoders

    def across_clusters_pairwise(
        self,
        component: str,
        modality: str,
        use_cluster: str,
        z_sampling_size: int = 5,
        x_sampling_size: int = 1000,
        batch_size: int = 128,
        keep_only_significant: bool = False,
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
        results = dict()
        obs = self.mdata[modality].obs
        unique_clusters = obs[use_cluster].unique()
        for cluster_a, cluster_b in combinations(unique_clusters, r=2):
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
            )
            if keep_only_significant:
                deg = deg.loc[
                    (deg["K(A>B|Z)"].abs() >= 3.2) | (deg["K(B>A|Z)"].abs() >= 3.2)
                ]

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
        results = dict()
        obs = self.mdata[modality].obs
        unique_clusters = obs[use_cluster].unique()
        for cluster in unique_clusters:
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
            )
            if keep_only_significant:
                deg = deg.loc[
                    (deg["K(A>B|Z)"].abs() >= 3.2) | (deg["K(B>A|Z)"].abs() >= 3.2)
                ]

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
        x_means_a = []
        x_means_b = []

        modality_names = self.mdata.mod.keys()

        batch_effect = dict()
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

        for _ in tqdm(range(z_sampling_size), desc=desc):
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
            x_means_a.append(
                self.compute_x_means(
                    dataset=dataloader_group_a.dataset,
                    component=component,
                    modality=modality,
                    batch_effect=batch_effect,
                    batch_size=batch_size,
                )
            )
            x_means_b.append(
                self.compute_x_means(
                    dataset=dataloader_group_b.dataset,
                    component=component,
                    modality=modality,
                    batch_effect=batch_effect,
                    batch_size=batch_size,
                )
            )

        x_means_a = np.vstack(x_means_a)
        x_means_b = np.vstack(x_means_b)
        index = self.mdata.mod[modality].var.index

        return self.compute_bayesian_factor(x_means_a, x_means_b, index)

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
        x_means_a: np.array,
        x_means_b: np.array,
        index: Union[pd.Index, Sequence[str]],
    ) -> pd.DataFrame:
        """Compute the Bayesian factor of two groups.

        Parameters
        ----------
        x_means_a : np.array
            the means of the generative data distribution of the first
            group.

        x_means_b : np.array
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
