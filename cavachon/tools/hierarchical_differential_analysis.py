# pyright: reportMissingImports=false, reportMissingModuleSource=false, reportUnknownParameterType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportAny=false, reportDeprecated=false, reportUnusedVariable=false, reportMissingTypeArgument=false

import warnings
from collections.abc import Mapping

import muon as mu
import numpy as np
import pandas as pd
import tensorflow as tf

from cavachon.dataloader.dataloader import DataLoader
from cavachon.environment.constants import Constants
from cavachon.tools.differential_analysis import DifferentialAnalysis
from cavachon.utils.reflection_handler import ReflectionHandler


class HierarchicalDifferentialAnalysis(DifferentialAnalysis):
    """Differential analysis with hierarchy-safe latent substitution."""

    def _make_dataloader(self, mdata: mu.MuData, batch_size: int) -> DataLoader:
        return DataLoader(
            mdata,
            batch_size,
            self.batch_effect_colnames or {},
            self.distribution_names or {},
            self.batch_effect_encoders or {},
        )

    def _require_dataset(self, dataloader: DataLoader) -> tf.data.Dataset:
        dataset = dataloader.dataset
        if dataset is None:
            raise ValueError("Failed to create dataset for differential analysis.")
        return dataset

    def _get_cluster_mdata(
        self, cluster: str, use_cluster: str, modality: str
    ) -> mu.MuData:
        cluster_index = self.mdata[modality].obs[
            self.mdata[modality].obs[use_cluster] == cluster
        ].index

        adata_dict = dict()
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=FutureWarning)
            warnings.simplefilter(action="ignore", category=UserWarning)
            for mod, adata in self.mdata.mod.items():
                adata_cluster = adata[cluster_index].copy()
                adata_cluster.obs_names_make_unique()
                adata_dict[mod] = adata_cluster

        return mu.MuData(adata_dict)

    def _encode_z_pool(
        self, mdata: mu.MuData, component: str, batch_size: int
    ) -> np.ndarray:
        dataloader = self._make_dataloader(mdata, batch_size)

        donor_z_pool = []
        dataset = self._require_dataset(dataloader)
        for batch in dataset.batch(batch_size):
            encode_outputs = self.model.encode(batch, training=False)
            z = encode_outputs[Constants.MODEL_OUTPUTS_Z]
            self.model.hierarchical_encode(batch, z, training=False)
            donor_z_pool.append(z[component].numpy())

        if not donor_z_pool:
            return np.empty((0, 0), dtype=np.float32)

        return np.vstack(donor_z_pool)

    def _build_batch_effect(self, batch_size: int) -> Mapping[str, tf.Tensor]:
        batch_effect = dict()
        dataloader = self._make_dataloader(self.mdata, batch_size)

        for batch in dataloader:
            for modality_name in self.mdata.mod.keys():
                if modality_name not in batch_effect:
                    batch_effect.setdefault(modality_name, [])
                batch_effect_key = f"{modality_name}_{Constants.TENSOR_NAME_BATCH}"
                batch_effect[modality_name].append(batch.get(batch_effect_key))

        for modality_name in batch_effect.keys():
            batch_effect[modality_name] = tf.concat(batch_effect[modality_name], axis=0)

        return batch_effect

    def _compute_substituted_x_means(
        self,
        dataset: tf.data.Dataset,
        component: str,
        modality: str,
        donor_z_pool: np.ndarray,
        batch_effect: Mapping[str, tf.Tensor],
        training: bool = False,
        batch_size: int = 128,
    ) -> np.ndarray:
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

            self.model.hierarchical_encode(batch, z, training=training)

            n_obs_batch = 0
            for modality_name in modality_names:
                batch_effect_key = f"{modality_name}_{Constants.TENSOR_NAME_BATCH}"
                n_obs_batch = batch[batch_effect_key].shape[0]

            indices = np.random.choice(len(donor_z_pool), size=n_obs_batch, replace=True)
            donor_z_sample = tf.constant(donor_z_pool[indices], dtype=tf.float32)
            z_substituted = dict(z)
            z_substituted[component] = donor_z_sample

            hier_outputs_substituted = self.model.hierarchical_encode(
                batch, z_substituted, training=training
            )
            z_hat_substituted = hier_outputs_substituted[Constants.MODEL_OUTPUTS_Z_HAT]

            random_batch_index = np.random.choice(np.arange(self.mdata.n_obs), n_obs_batch)
            for modality_name in modality_names:
                batch_effect_key = f"{modality_name}_{Constants.TENSOR_NAME_BATCH}"
                batch[batch_effect_key] = tf.gather(
                    batch_effect[modality_name], random_batch_index, axis=0
                )

            decode_outputs = self.model.decode(
                batch, z_hat_substituted, components=[component], training=training
            )
            x_parameters = decode_outputs[Constants.MODEL_OUTPUTS_X_PARAMS].get(
                f"{component}_{modality}_{Constants.MODEL_OUTPUTS_X_PARAMS}"
            )
            dist_x_z = dist_x_z_class.from_parameterizer_output(x_parameters)
            x_means.append(dist_x_z.mean().numpy())

        return np.vstack(x_means)

    def between_clusters(
        self,
        donor_cluster: str,
        recipient_cluster: str,
        component: str,
        modality: str,
        use_cluster: str,
        n_samples: int = 10,
        seed: int | None = None,
        batch_size: int = 128,
    ) -> pd.DataFrame:
        obs = self.mdata[modality].obs
        cluster_labels = obs[use_cluster]
        if hasattr(cluster_labels, "cat"):
            available_clusters = cluster_labels.cat.categories
        else:
            available_clusters = pd.Index(cluster_labels.unique())

        if donor_cluster not in available_clusters:
            raise ValueError(f"Unknown donor cluster: {donor_cluster}")
        if recipient_cluster not in available_clusters:
            raise ValueError(f"Unknown recipient cluster: {recipient_cluster}")

        if seed is not None:
            np.random.seed(seed)

        recipient_index = obs[obs[use_cluster] == recipient_cluster].index

        if len(recipient_index) == 0:
            raise ValueError(f"Recipient cluster '{recipient_cluster}' is empty.")

        donor_mdata = self._get_cluster_mdata(donor_cluster, use_cluster, modality)
        donor_z_pool = self._encode_z_pool(donor_mdata, component, batch_size)
        if len(donor_z_pool) == 0:
            raise ValueError(f"Donor cluster '{donor_cluster}' is empty.")

        batch_effect = self._build_batch_effect(batch_size)

        x_means_substituted = []
        x_means_original = []
        for _ in range(n_samples):
            recipient_mdata = self.sample_mdata_x(
                recipient_index, x_sampling_size=len(recipient_index)
            )
            recipient_dataloader = self._make_dataloader(recipient_mdata, batch_size)

            recipient_dataset = self._require_dataset(recipient_dataloader)

            x_means_substituted.append(
                self._compute_substituted_x_means(
                    dataset=recipient_dataset,
                    component=component,
                    modality=modality,
                    donor_z_pool=donor_z_pool,
                    batch_effect=batch_effect,
                    batch_size=batch_size,
                )
            )
            x_means_original.append(
                self.compute_x_means(
                    dataset=recipient_dataset,
                    component=component,
                    modality=modality,
                    batch_effect=batch_effect,
                    batch_size=batch_size,
                )
            )

        x_means_substituted = np.vstack(x_means_substituted)
        x_means_original = np.vstack(x_means_original)
        var_index = self.mdata.mod[modality].var.index

        result = self.compute_bayesian_factor(
            x_means_substituted, x_means_original, var_index
        )
        result["InterventionType"] = "donor_pool_z_substitution"
        result["DonorCluster"] = donor_cluster
        result["RecipientCluster"] = recipient_cluster
        result["SamplingStrategy"] = "donor_pool"
        result["RandomSeed"] = seed
        result["MeanDelta(Substituted-Original)"] = (
            result["Mean(A)"] - result["Mean(B)"]
        )

        return result
