# pyright: reportMissingImports=false, reportMissingModuleSource=false, reportUnknownParameterType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportAny=false, reportDeprecated=false, reportUnusedVariable=false, reportMissingTypeArgument=false

import warnings
from collections.abc import Mapping
from itertools import combinations
from typing import Any

import muon as mu
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

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

    def _get_group_mdata(
        self, group: str, use_cluster: str, modality: str
    ) -> mu.MuData:
        group_index = (
            self.mdata[modality]
            .obs[self.mdata[modality].obs[use_cluster] == group]
            .index
        )

        adata_dict = dict()
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=FutureWarning)
            warnings.simplefilter(action="ignore", category=UserWarning)
            for mod, adata in self.mdata.mod.items():
                adata_group = adata[group_index].copy()
                adata_group.obs_names_make_unique()
                adata_dict[mod] = adata_group

        return mu.MuData(adata_dict)

    def _normalize_donor_components(
        self, donor_components: list[str] | None, component: str
    ) -> list[str]:
        if donor_components is None:
            return [component]

        seen: set[str] = set()
        result: list[str] = []
        for name in donor_components:
            if name not in seen:
                seen.add(name)
                result.append(name)

        if not result:
            raise ValueError("donor_components must not be empty.")

        valid = set(self.model.components.keys())
        unknown = seen - valid
        if unknown:
            raise ValueError(
                f"Unknown donor component(s): {sorted(unknown)}. "
                f"Valid components: {sorted(valid)}"
            )

        return result

    def _encode_z_pool(
        self, mdata: mu.MuData, donor_components: list[str], batch_size: int
    ) -> dict[str, np.ndarray]:
        dataloader = self._make_dataloader(mdata, batch_size)

        pools: dict[str, list[np.ndarray]] = {name: [] for name in donor_components}
        dataset = self._require_dataset(dataloader)
        for batch in dataset.batch(batch_size):
            encode_outputs = self.model.encode(batch, training=False)
            z = encode_outputs[Constants.MODEL_OUTPUTS_Z]
            self.model.hierarchical_encode(batch, z, training=False)
            for name in donor_components:
                pools[name].append(z[name].numpy())

        result: dict[str, np.ndarray] = {}
        for name in donor_components:
            if pools[name]:
                result[name] = np.vstack(pools[name])
            else:
                result[name] = np.empty((0, 0), dtype=np.float32)

        return result

    def _build_batch_effect(self, batch_size: int) -> Mapping[str, tf.Tensor]:
        # holds a list of per-batch tensors first, then their concatenation
        batch_effect: dict[str, Any] = dict()
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
        donor_z_pools: dict[str, np.ndarray],
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

            first_pool = next(iter(donor_z_pools.values()))
            indices = np.random.choice(len(first_pool), size=n_obs_batch, replace=True)

            z_substituted = dict(z)
            for name, pool in donor_z_pools.items():
                z_substituted[name] = tf.constant(pool[indices], dtype=tf.float32)

            hier_outputs_substituted = self.model.hierarchical_encode(
                batch, z_substituted, training=training
            )
            z_hat_substituted = hier_outputs_substituted[Constants.MODEL_OUTPUTS_Z_HAT]

            random_batch_index = np.random.choice(
                np.arange(self.mdata.n_obs), n_obs_batch
            )
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

    def between_groups(
        self,
        donor_group: str,
        recipient_group: str,
        component: str,
        modality: str,
        use_cluster: str,
        z_sampling_size: int = 10,
        x_sampling_size: int | None = None,
        seed: int | None = None,
        batch_size: int = 128,
        donor_components: list[str] | None = None,
        desc: str = "",
        sort_output: bool = True,
    ) -> pd.DataFrame:
        """Compute differential expression between donor and recipient groups.

        Performs latent substitution: samples ``z`` from the donor group for each
        component in ``donor_components``, substitutes them into recipient-cell
        encodings, and compares the resulting decoded expression against the
        unmodified recipient expression.

        Parameters
        ----------
        donor_group : str
            Label (in ``obs[use_cluster]``) of the group from which latent
            representations are drawn.
        recipient_group : str
            Label (in ``obs[use_cluster]``) of the group whose cells provide
            recipient encodings and baseline expression.
        component : str
            **DEG target**: model component whose output distribution is decoded and
            compared. Controls distribution lookup and ``model.decode(components=
            [component], ...)``. NOT the donor selection unless also listed in
            ``donor_components``.
        modality : str
            **DEG target modality**: omics layer within ``component`` whose mean
            expression is compared (e.g. ``"RNA"``).
        use_cluster : str
            Column in ``mdata[modality].obs`` containing the group labels.
        z_sampling_size : int, optional
            Monte-Carlo sampling rounds. Default ``10``.
        x_sampling_size : int | None, optional
            Recipient cells drawn (with replacement) per round. Defaults to
            ``None``: as many as the recipient group has.
        seed : int | None, optional
            NumPy random seed. Default ``None``.
        batch_size : int, optional
            Batch size for encoding and decoding. Default ``128``.
        donor_components : list[str] | None, optional
            Names of model components whose ``z`` latent vectors are replaced by
            samples drawn from the donor group. Duplicates are removed preserving
            order. Defaults to ``[component]`` when ``None``, reproducing the
            original single-component substitution behaviour.

            Controls **which latent dimensions are swapped**; independent of the
            DEG decode target (``component`` / ``modality``).

        sort_output : bool, optional
            Sort the result by the maximum of the absolute Bayesian factors
            ``K(A>B|Z)`` and ``K(B>A|Z)`` in descending order. Default ``True``.

        Returns
        -------
        pd.DataFrame
            One row per feature of ``modality``. In each of the
            ``z_sampling_size`` rounds, ``x_sampling_size`` recipient cells
            are drawn; every drawn cell is decoded twice: **A** with the
            ``z`` of ``donor_components`` replaced by the ``z`` of a random
            donor cell, and **B** unchanged (each decode with a batch
            effect drawn at random from all cells). Over these
            (round, cell) pairs:

            - ``Mean(A)`` / ``Mean(B)``: mean of the decoded means of A /
              B, i.e. the expected expression of the recipient group with
              / without the donor's latent;
            - ``P(A>B|Z)``: fraction of pairs in which A > B (the same
              cell, so the comparison is paired); ``P(B>A|Z)`` likewise.
              Ties count for neither, so they may sum to less than 1;
            - ``K(A>B|Z)``: log Bayes factor
              ``log(P(A>B|Z) + 1e-7) - log(1 - P(A>B|Z) + 1e-7)``, natural
              log; ``|K| >= 3.2`` is the usual significance threshold.
              ``K(B>A|Z)`` likewise;
            - ``MeanDelta(Substituted-Original)``: ``Mean(A) - Mean(B)``.

            Then the intervention: ``InterventionType``, ``DonorGroup``,
            ``RecipientGroup``, ``SamplingStrategy``, ``DonorComponents``
            (comma-separated sorted component names); and the parameters of
            the run, as in ``DifferentialAnalysis.between_two_groups``:
            ``Component``, ``Modality``, ``UseCluster``,
            ``ZSamplingSize``, ``XSamplingSize`` (the number actually
            drawn) and ``RandomSeed``. The parameters let a table read back
            from disk be matched to the analysis that produced it.

        Raises
        ------
        ValueError
            If groups are unknown/empty or ``donor_components`` contains unknown
            names or resolves to an empty list.
        """
        obs = self.mdata[modality].obs
        group_labels = obs[use_cluster]
        if hasattr(group_labels, "cat"):
            available_groups = group_labels.cat.categories
        else:
            available_groups = pd.Index(group_labels.unique())

        if donor_group not in available_groups:
            raise ValueError(f"Unknown donor group: {donor_group}")
        if recipient_group not in available_groups:
            raise ValueError(f"Unknown recipient group: {recipient_group}")

        if seed is not None:
            np.random.seed(seed)

        normalized_donor_components = self._normalize_donor_components(
            donor_components, component
        )

        recipient_index = obs[obs[use_cluster] == recipient_group].index

        if len(recipient_index) == 0:
            raise ValueError(f"Recipient group '{recipient_group}' is empty.")
        if x_sampling_size is None:
            x_sampling_size = len(recipient_index)

        donor_mdata = self._get_group_mdata(donor_group, use_cluster, modality)
        donor_z_pools = self._encode_z_pool(
            donor_mdata, normalized_donor_components, batch_size
        )

        if all(len(v) == 0 for v in donor_z_pools.values()):
            raise ValueError(f"Donor group '{donor_group}' is empty.")

        batch_effect = self._build_batch_effect(batch_size)

        x_means_substituted_list: list[np.ndarray] = []
        x_means_original_list: list[np.ndarray] = []
        for _ in tqdm(range(z_sampling_size), desc=desc, leave=False):
            recipient_mdata = self.sample_mdata_x(
                recipient_index, x_sampling_size=x_sampling_size
            )
            recipient_dataloader = self._make_dataloader(recipient_mdata, batch_size)

            recipient_dataset = self._require_dataset(recipient_dataloader)

            x_means_substituted_list.append(
                self._compute_substituted_x_means(
                    dataset=recipient_dataset,
                    component=component,
                    modality=modality,
                    donor_z_pools=donor_z_pools,
                    batch_effect=batch_effect,
                    batch_size=batch_size,
                )
            )
            x_means_original_list.append(
                self.compute_x_means(
                    dataset=recipient_dataset,
                    component=component,
                    modality=modality,
                    batch_effect=batch_effect,
                    batch_size=batch_size,
                )
            )

        x_means_substituted = np.vstack(x_means_substituted_list)
        x_means_original = np.vstack(x_means_original_list)
        var_index = self.mdata.mod[modality].var.index

        result = self.compute_bayesian_factor(
            x_means_substituted, x_means_original, var_index
        )
        result["MeanDelta(Substituted-Original)"] = (
            result["Mean(A)"] - result["Mean(B)"]
        )
        result["InterventionType"] = "donor_pool_z_substitution"
        result[Constants.HIERARCHICAL_DIFFERENTIAL_ANALYSIS_DONOR_GROUP_COLUMN] = (
            donor_group
        )
        result[Constants.HIERARCHICAL_DIFFERENTIAL_ANALYSIS_RECIPIENT_GROUP_COLUMN] = (
            recipient_group
        )
        result["SamplingStrategy"] = "donor_pool"
        result["DonorComponents"] = ",".join(sorted(normalized_donor_components))
        self.record_run_parameters(
            result,
            component=component,
            modality=modality,
            use_cluster=use_cluster,
            z_sampling_size=z_sampling_size,
            x_sampling_size=x_sampling_size,
            seed=seed,
        )

        if sort_output:
            sort_key = result[["K(A>B|Z)", "K(B>A|Z)"]].abs().max(axis=1)
            result = result.loc[sort_key.sort_values(ascending=False).index]

        return result

    def across_clusters_pairwise(  # type: ignore[override]  # deliberately different parameters (donor substitution), not a drop-in for the base method
        self,
        component: str,
        modality: str,
        use_cluster: str,
        donor_components: list[str] | None = None,
        z_sampling_size: int = 10,
        x_sampling_size: int | None = None,
        seed: int | None = None,
        batch_size: int = 128,
        sort_output: bool = True,
    ) -> Mapping[str, pd.DataFrame]:
        """Perform counterfactual intervention between every pair of clusters.

        For each unordered pair of clusters (A, B), runs two interventions:
        ``A -> B`` (substitute A's z into B) and ``B -> A`` (substitute B's
        z into A), using ``between_groups``. The ``donor_components``
        parameter controls which model components' z latent vectors are
        substituted.

        Parameters
        ----------
        component : str
            DEG target: model component whose output distribution is decoded.
        modality : str
            DEG target modality within ``component``.
        use_cluster : str
            Column in ``mdata[modality].obs`` containing cluster labels.
        donor_components : list[str] | None, optional
            Model components whose z vectors are replaced by donor samples.
            Defaults to ``[component]``.
        z_sampling_size : int, optional
            Monte-Carlo sampling rounds per intervention. Default ``10``.
        x_sampling_size : int | None, optional
            Recipient cells drawn per round. Defaults to ``None``: the
            recipient cluster's size.
        seed : int | None, optional
            NumPy random seed, applied at the start of every intervention.
            Default ``None``.
        batch_size : int, optional
            Batch size for encoding/decoding. Default ``128``.

        sort_output : bool, optional
            Sort each result by the maximum of the absolute Bayesian factors
            ``K(A>B|Z)`` and ``K(B>A|Z)`` in descending order. Default ``True``.

        Returns
        -------
        Mapping[str, pd.DataFrame]
            Keys are ``"ClusterA->ClusterB"`` and ``"ClusterB->ClusterA"``.
            Values are Bayesian-factor tables with intervention metadata
            (same columns as ``between_groups``).
        """
        obs = self.mdata[modality].obs
        unique_clusters = obs[use_cluster].unique()
        results = {}

        for cluster_a, cluster_b in tqdm(
            combinations(unique_clusters, r=2),
            desc="Across clusters pairwise",
        ):
            for donor, recipient in ((cluster_a, cluster_b), (cluster_b, cluster_a)):
                key = f"{donor}->{recipient}"
                results[key] = self.between_groups(
                    donor_group=donor,
                    recipient_group=recipient,
                    component=component,
                    modality=modality,
                    use_cluster=use_cluster,
                    donor_components=donor_components,
                    z_sampling_size=z_sampling_size,
                    x_sampling_size=x_sampling_size,
                    seed=seed,
                    batch_size=batch_size,
                    desc=key,
                    sort_output=sort_output,
                )

        return results
