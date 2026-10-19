import gc
import warnings
from typing import List, Mapping, Optional, Sequence, Union, cast

import muon as mu
import numpy as np
import pandas as pd
import tensorflow as tf

from cavachon.dataloader.dataloader import DataLoader
from cavachon.environment.constants import Constants
from cavachon.tools.differential_analysis import DifferentialAnalysis
from cavachon.utils.dataframe_utils import DataFrameUtils, ThresholdOperator
from cavachon.utils.reflection_handler import ReflectionHandler


class DifferentialAttributionAnalysis(DifferentialAnalysis):
    """Differential attribution analysis for DEGs.

    Computes Integrated-Gradients attributions for the change in expected
    expression of each DEG between two groups. The group difference is
    represented by the difference in the per-component latent means
    (``z_baseline`` for the reference group and ``z_target`` for the case
    group). The attributions are grouped by component, summed over latent
    dimensions, and normalized so that the signed contributions sum to one
    when multiple components are selected in ``with_respect_to``. With a
    single component, raw attributions are reported instead.

    """

    def compute_group_latent_mean(
        self,
        index: Union[pd.Index, Sequence[str]],
        batch_size: int = 128,
    ) -> Mapping[str, tf.Tensor]:
        """Compute the mean latent code ``z`` for a group of cells.

        Parameters
        ----------
        index : Union[pd.Index, Sequence[str]]
            Cell identifiers matching ``mdata.obs_names``.

        batch_size : int, optional
            Batch size used by the encoder. Defaults to 128.

        Returns
        -------
        Mapping[str, tf.Tensor]
            Component-keyed mean ``z`` tensors.

        """
        if len(index) == 0:
            raise ValueError("Cannot compute group latent mean for an empty index.")

        adata_dict = dict()
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=FutureWarning)
            warnings.simplefilter(action="ignore", category=UserWarning)
            for mod, adata in self.mdata.mod.items():
                adata_dict[mod] = adata[index].copy()

        mdata_group = mu.MuData(adata_dict)
        dataloader = DataLoader(
            mdata_group,
            batch_size,
            self.batch_effect_colnames or {},
            self.distribution_names or {},
            self.batch_effect_encoders,
        )

        sums = dict()
        counts = dict()
        for batch in dataloader:
            encode_outputs = self.model.encode(batch, training=False)
            z = encode_outputs[Constants.MODEL_OUTPUTS_Z]
            for component_name, z_component in z.items():
                sums.setdefault(component_name, [])
                counts.setdefault(component_name, [])
                sums[component_name].append(
                    tf.reduce_sum(z_component, axis=0, keepdims=True)
                )
                counts[component_name].append(
                    tf.cast(tf.shape(z_component)[0], tf.float32)
                )

        means = dict()
        for component_name in sums.keys():
            total_sum = tf.add_n(sums[component_name])
            total_count = tf.add_n(counts[component_name])
            means[component_name] = total_sum / total_count

        gc.collect()
        return means

    def compute_differential_attribution_target_batch(
        self,
        batch: Mapping[str, tf.Tensor],
        component: str,
        modality: str,
        z_dict: Mapping[str, tf.Tensor],
    ) -> tf.Tensor:
        """Decode the expected expression using the provided ``z`` dict.

        Parameters
        ----------
        batch : Mapping[str, tf.Tensor]
            A batch used for conditional inputs, batch effects and library
            size. The modality data itself is only used for preprocessor
            outputs such as library size.

        component : str
            Name of the component whose modality is being decoded.

        modality : str
            Name of the modality being decoded.

        z_dict : Mapping[str, tf.Tensor]
            Component-keyed ``z`` tensors.

        Returns
        -------
        tf.Tensor
            Mean of the generative distribution, shape ``(batch_size,
            n_features)``.

        """
        hier_outputs = self.model.hierarchical_encode(
            batch, z_dict, training=False
        )
        z_hat = hier_outputs[Constants.MODEL_OUTPUTS_Z_HAT]
        decode_outputs = self.model.decode(
            batch, z_hat, components=[component], training=False
        )
        x_parameters = decode_outputs[Constants.MODEL_OUTPUTS_X_PARAMS].get(
            f"{component}_{modality}_{Constants.MODEL_OUTPUTS_X_PARAMS}"
        )
        dist_x_z_name = self.model.components.get(component).distribution_names.get(
            modality
        )
        dist_x_z_class = ReflectionHandler.get_class_by_name(
            dist_x_z_name, "distributions", "Distribution"
        )
        dist_x_z = dist_x_z_class.from_parameterizer_output(x_parameters)
        return dist_x_z.mean()

    def compute_differential_attribution(
        self,
        deg: pd.DataFrame,
        component: str,
        modality: str,
        group_a_index: Union[pd.Index, Sequence[str]],
        group_b_index: Union[pd.Index, Sequence[str]],
        with_respect_to: List[str],
        batch_size: int = 128,
        n_steps: int = 32,
    ) -> Optional[pd.DataFrame]:
        """Compute per-component attributions for a DEG table.

        Parameters
        ----------
        deg : pd.DataFrame
            DEG table whose index contains feature names. Only features
            present in ``mdata.mod[modality].var_names`` are attributed.

        component : str
            Component that generates ``modality``.

        modality : str
            Modality of the DEGs.

        group_a_index : Union[pd.Index, Sequence[str]]
            Cells defining the target (case) group.

        group_b_index : Union[pd.Index, Sequence[str]]
            Cells defining the baseline (control/reference) group.

        with_respect_to : List[str]
            Component names whose latent changes should be attributed.

        batch_size : int, optional
            Encoder batch size. Defaults to 128.

        n_steps : int, optional
            Number of Riemann-sum steps for Integrated Gradients. Defaults
            to 32.

        Returns
        -------
        Optional[pd.DataFrame]
            Augmented DEG table with per-component attribution columns:
            ``Delta``, ``attr_<comp>``, ``abs_attr_<comp>``,
            ``contrib_<comp>`` and ``abs_contrib_<comp>`` (the latter two are
            omitted when ``with_respect_to`` contains a single component).
            Returns ``None`` when no DEG features overlap with the modality.

        """
        if not with_respect_to:
            raise ValueError("with_respect_to must contain at least one component.")

        invalid_components = set(with_respect_to).difference(self.model.components.keys())
        if invalid_components:
            raise ValueError(
                f"Unknown components in with_respect_to: {sorted(invalid_components)}"
            )

        modality_obj = self.mdata.mod.get(modality)
        if modality_obj is None:
            raise ValueError(f"Modality '{modality}' not found in mdata.")

        valid_features = modality_obj.var_names.intersection(deg.index)
        if len(valid_features) == 0:
            warnings.warn(
                f"No DEG features overlap with {modality} var names for component {component}; skipping.",
                UserWarning,
            )
            return None

        deg = deg.loc[valid_features].copy()

        # Provide a log-fold-change column for visualization if absent.
        if "LogFC" not in deg.columns:
            if "Mean(A)" in deg.columns and "Mean(B)" in deg.columns:
                deg["LogFC"] = np.log1p(deg["Mean(A)"]) - np.log1p(deg["Mean(B)"])

        # The decoder expects the full modality feature dimension, so we keep
        # all features and select the DEG subset from the decoder output.
        feature_indices = modality_obj.var_names.get_indexer(valid_features)
        if np.any(feature_indices < 0):
            missing = valid_features[feature_indices < 0].tolist()
            raise ValueError(f"DEG features not found in modality: {missing}")
        feature_indices = tf.constant(feature_indices, dtype=tf.int32)

        adata_dict = dict()
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=FutureWarning)
            warnings.simplefilter(action="ignore", category=UserWarning)
            for mod, adata in self.mdata.mod.items():
                adata_dict[mod] = adata.copy()

        mdata_subset = mu.MuData(adata_dict)

        z_baseline = self.compute_group_latent_mean(group_b_index, batch_size=batch_size)
        z_target = self.compute_group_latent_mean(group_a_index, batch_size=batch_size)

        # Representative batch for decoder conditional inputs.
        representative_dataloader = DataLoader(
            mdata_subset,
            batch_size=1,
            batch_effect_colnames=self.batch_effect_colnames or {},
            distribution_names=self.distribution_names or {},
            batch_effect_encoders=self.batch_effect_encoders,
        )
        representative_batch = None
        for batch in representative_dataloader:
            representative_batch = batch
            break
        assert representative_batch is not None
        del mdata_subset
        gc.collect()

        delta_z = {
            comp_name: z_target[comp_name] - z_baseline[comp_name]
            for comp_name in with_respect_to
        }

        # Compute full prediction at baseline and target (deterministic).
        z_dict_baseline = {
            name: z_baseline[name] for name in z_baseline.keys()
        }
        z_dict_target = {
            name: (z_target[name] if name in with_respect_to else z_baseline[name])
            for name in z_baseline.keys()
        }

        f_baseline = tf.gather(
            self.compute_differential_attribution_target_batch(
                representative_batch, component, modality, z_dict_baseline
            ),
            feature_indices,
            axis=-1,
        )
        f_target = tf.gather(
            self.compute_differential_attribution_target_batch(
                representative_batch, component, modality, z_dict_target
            ),
            feature_indices,
            axis=-1,
        )

        n_features = int(tf.shape(f_baseline)[-1].numpy().item())

        # Accumulate gradients along the IG path. We keep per-component
        # watched tensors because slicing a concatenated watched tensor and
        # passing the slices through Keras models drops the gradient path.
        accumulated_gradients = {
            comp_name: tf.zeros(
                (1, n_features, tf.shape(z_baseline[comp_name])[-1]),
                dtype=tf.float32,
            )
            for comp_name in with_respect_to
        }

        for step in range(n_steps):
            alpha = float(step) / float(n_steps)
            with tf.GradientTape(persistent=True) as tape:
                z_step = {
                    comp_name: z_baseline[comp_name] + alpha * delta_z[comp_name]
                    for comp_name in with_respect_to
                }
                for comp_name in with_respect_to:
                    z_step[comp_name] = tf.identity(z_step[comp_name])
                    tape.watch(z_step[comp_name])

                z_dict_step = {
                    comp_name: (
                        z_step[comp_name]
                        if comp_name in with_respect_to
                        else z_baseline[comp_name]
                    )
                    for comp_name in self.model.components.keys()
                }

                target = tf.gather(
                    self.compute_differential_attribution_target_batch(
                        representative_batch, component, modality, z_dict_step
                    ),
                    feature_indices,
                    axis=-1,
                )

                # Gradient calls must stay inside the GradientTape context;
                # slicing/indexing the target outside the tape creates new
                # tensors that are no longer connected to the watched source.
                for feature_index in range(n_features):
                    for comp_name in with_respect_to:
                        gradient = tape.gradient(
                            target[0, feature_index], z_step[comp_name]
                        )
                        if gradient is not None:
                            gradient_tensor = tf.convert_to_tensor(gradient)
                            accumulated_gradients[comp_name] = tf.tensor_scatter_nd_add(
                                accumulated_gradients[comp_name],
                                indices=[[0, feature_index]],
                                updates=tf.expand_dims(gradient_tensor[0], axis=0),
                            )
            del tape
            gc.collect()

        # Scale by delta_z and divide by number of steps (Riemann sum).
        attributions = dict()
        for comp_name in with_respect_to:
            attributions[comp_name] = (
                accumulated_gradients[comp_name]
                * tf.expand_dims(delta_z[comp_name], axis=1)
                / float(n_steps)
            )[0].numpy()

        # Sum attributions per component.
        attr_columns = dict()
        abs_attr_columns = dict()
        for comp_name in with_respect_to:
            comp_attr = attributions[comp_name].sum(axis=1)
            attr_columns[f"attr_{comp_name}"] = comp_attr
            abs_attr_columns[f"abs_attr_{comp_name}"] = np.abs(comp_attr)

        delta = (f_target - f_baseline)[0].numpy()

        result = deg.copy()
        result["Delta"] = delta
        result["Expected(A)"] = f_target[0].numpy()
        result["Expected(B)"] = f_baseline[0].numpy()
        for comp_name in with_respect_to:
            result[f"attr_{comp_name}"] = attr_columns[f"attr_{comp_name}"]
            result[f"abs_attr_{comp_name}"] = abs_attr_columns[f"abs_attr_{comp_name}"]

        if len(with_respect_to) == 1:
            # With a single attributed component the normalized contributions
            # would be trivially +/-1, so report only raw attributions.
            return result

        signed_sum = np.sum(list(attr_columns.values()), axis=0)
        abs_sum = np.sum(list(abs_attr_columns.values()), axis=0)

        # Guard against division by zero on a per-gene basis. When the total
        # signed or absolute attribution is essentially zero, the normalized
        # contribution is reported as zero rather than NaN.
        signed_sum = np.where(np.abs(signed_sum) < 1e-12, 0.0, signed_sum)
        abs_sum = np.where(abs_sum < 1e-12, 0.0, abs_sum)

        for comp_name in with_respect_to:
            signed_denom = np.where(
                np.abs(signed_sum) < 1e-12, 1.0, signed_sum
            )
            abs_denom = np.where(abs_sum < 1e-12, 1.0, abs_sum)
            result[f"contrib_{comp_name}"] = np.where(
                np.abs(signed_sum) < 1e-12,
                0.0,
                attr_columns[f"attr_{comp_name}"] / signed_denom,
            )
            result[f"abs_contrib_{comp_name}"] = np.where(
                abs_sum < 1e-12,
                0.0,
                abs_attr_columns[f"abs_attr_{comp_name}"] / abs_denom,
            )

        return result

    def from_deg_file(
        self,
        deg_file: str,
        component: str,
        modality: str,
        group_a_index: Union[pd.Index, Sequence[str]],
        group_b_index: Union[pd.Index, Sequence[str]],
        with_respect_to: List[str],
        column: str = "K(A>B|Z)",
        threshold: float = 3.2,
        threshold_operator: ThresholdOperator = "geq",
        use_absolute: bool = True,
        min_num_degs: int = 0,
        batch_size: int = 128,
        n_steps: int = 32,
    ) -> Optional[pd.DataFrame]:
        """Load a DEG file, filter it, and compute attributions.

        Parameters
        ----------
        deg_file : str
            Path to a DEG result file.

        component : str
            Component that generates ``modality``.

        modality : str
            Modality of the DEGs.

        group_a_index : Union[pd.Index, Sequence[str]]
            Cells defining the target (case) group.

        group_b_index : Union[pd.Index, Sequence[str]]
            Cells defining the baseline (control/reference) group.

        with_respect_to : List[str]
            Component names whose latent changes should be attributed.

        column : str, optional
            Column used to filter significant DEGs. Defaults to
            ``K(A>B|Z)``.

        threshold : float, optional
            Threshold applied to ``column``. Defaults to 3.2.

        threshold_operator : ThresholdOperator, optional
            Operator used to compare ``column`` to ``threshold``. One of
            ``leq``, ``geq``, ``lt``, ``gt``. Defaults to ``geq``.

        use_absolute : bool, optional
            Whether to compare ``|column|`` to ``threshold``. Defaults to
            ``True``.

        min_num_degs : int, optional
            Minimum number of DEGs required to compute attributions.
            Defaults to 0.

        batch_size : int, optional
            Encoder batch size. Defaults to 128.

        n_steps : int, optional
            Number of Riemann-sum steps. Defaults to 32.

        Returns
        -------
        Optional[pd.DataFrame]
            Filtered and attributed DEG table, or ``None`` if too few DEGs
            pass the threshold or no features overlap with the modality.

        """
        deg = pd.read_csv(deg_file, sep=None, engine="python", index_col=0)
        deg = deg.loc[
            DataFrameUtils.filter_by_threshold(
                deg[column], threshold, threshold_operator, use_absolute
            )
        ]
        if len(deg) < min_num_degs:
            print(
                f"Skipping differential attribution for {component}/{modality}: "
                f"only {len(deg)} DEGs (min_num_degs={min_num_degs})"
            )
            return None
        return self.compute_differential_attribution(
            deg=deg,
            component=component,
            modality=modality,
            group_a_index=group_a_index,
            group_b_index=group_b_index,
            with_respect_to=with_respect_to,
            batch_size=batch_size,
            n_steps=n_steps,
        )

