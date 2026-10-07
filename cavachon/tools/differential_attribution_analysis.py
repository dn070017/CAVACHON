import warnings
from typing import List, Mapping, Sequence, Union

import muon as mu
import numpy as np
import pandas as pd
import tensorflow as tf

from cavachon.dataloader.dataloader import DataLoader
from cavachon.environment.constants import Constants
from cavachon.tools.differential_analysis import DifferentialAnalysis
from cavachon.utils.reflection_handler import ReflectionHandler


class DifferentialAttributionAnalysis(DifferentialAnalysis):
    """Differential attribution analysis for DEGs.

    Computes Integrated-Gradients attributions for the change in expected
    expression of each DEG between two groups. The group difference is
    represented by the difference in the per-component latent means
    (``z_baseline`` for the reference group and ``z_target`` for the case
    group). The attributions are grouped by component, summed over latent
    dimensions, and normalized so that the signed contributions sum to one
    (within the user-selected ``with_respect_to`` components).

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
    ) -> pd.DataFrame:
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
        pd.DataFrame
            Augmented DEG table with per-component attribution columns:
            ``Delta``, ``attr_<comp>``, ``abs_attr_<comp>``,
            ``contrib_<comp>`` and ``abs_contrib_<comp>``.

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
            raise ValueError(
                "No DEG features overlap with the modality var names."
            )

        deg = deg.loc[valid_features].copy()

        # Provide a log-fold-change column for visualization if absent.
        if "LogFC" not in deg.columns:
            if "Mean(A)" in deg.columns and "Mean(B)" in deg.columns:
                deg["LogFC"] = np.log1p(deg["Mean(A)"]) - np.log1p(deg["Mean(B)"])

        # Work on a feature subset so the decoder only produces DEGs.
        adata_dict = dict()
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=FutureWarning)
            warnings.simplefilter(action="ignore", category=UserWarning)
            for mod, adata in self.mdata.mod.items():
                if mod == modality:
                    adata_dict[mod] = adata[:, valid_features].copy()
                else:
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

        # Build concatenated selected z slices and their baseline/target values.
        selected_dims = []
        for comp_name in with_respect_to:
            selected_dims.append(
                tf.shape(z_baseline[comp_name])[-1].numpy().item()
            )

        z_baseline_selected = tf.concat(
            [z_baseline[comp] for comp in with_respect_to], axis=-1
        )
        z_target_selected = tf.concat(
            [z_target[comp] for comp in with_respect_to], axis=-1
        )
        delta_z_selected = z_target_selected - z_baseline_selected

        # Compute full prediction at baseline and target (deterministic).
        z_dict_baseline = {
            name: (z_baseline[name] if name in with_respect_to else z_baseline[name])
            for name in z_baseline.keys()
        }
        z_dict_target = {
            name: (z_target[name] if name in with_respect_to else z_baseline[name])
            for name in z_baseline.keys()
        }

        f_baseline = self.compute_differential_attribution_target_batch(
            representative_batch, component, modality, z_dict_baseline
        )
        f_target = self.compute_differential_attribution_target_batch(
            representative_batch, component, modality, z_dict_target
        )

        n_features = int(tf.shape(f_baseline)[-1].numpy().item())

        # Accumulate gradients along the IG path.
        accumulated_gradients = tf.zeros(
            (1, n_features, tf.shape(z_baseline_selected)[-1]), dtype=tf.float32
        )

        for step in range(n_steps):
            alpha = float(step) / float(n_steps)
            with tf.GradientTape(persistent=True) as tape:
                z_step = z_baseline_selected + alpha * delta_z_selected
                z_step = tf.identity(z_step)
                tape.watch(z_step)

                # Reconstruct per-component z dict.
                z_dict_step = dict()
                start = 0
                for comp_name in self.model.components.keys():
                    if comp_name in with_respect_to:
                        end = start + selected_dims[with_respect_to.index(comp_name)]
                        z_dict_step[comp_name] = z_step[:, start:end]
                        start = end
                    else:
                        z_dict_step[comp_name] = z_baseline[comp_name]

                target = self.compute_differential_attribution_target_batch(
                    representative_batch, component, modality, z_dict_step
                )

            for feature_index in range(n_features):
                gradient = tape.gradient(
                    target[0, feature_index], z_step
                )
                if gradient is not None:
                    accumulated_gradients = tf.tensor_scatter_nd_add(
                        accumulated_gradients,
                        indices=[[0, feature_index]],
                        updates=tf.expand_dims(gradient[0], axis=0),
                    )
            del tape

        # Scale by delta_z and divide by number of steps (Riemann sum).
        attributions = (
            accumulated_gradients
            * tf.expand_dims(delta_z_selected, axis=1)
            / float(n_steps)
        )
        attributions = attributions[0].numpy()

        # Sum attributions per component.
        attr_columns = dict()
        abs_attr_columns = dict()
        start = 0
        for i, comp_name in enumerate(with_respect_to):
            end = start + selected_dims[i]
            comp_attr = attributions[:, start:end].sum(axis=1)
            attr_columns[f"attr_{comp_name}"] = comp_attr
            abs_attr_columns[f"abs_attr_{comp_name}"] = np.abs(comp_attr)
            start = end

        delta = (f_target - f_baseline)[0].numpy()
        signed_sum = np.sum(list(attr_columns.values()), axis=0)
        abs_sum = np.sum(list(abs_attr_columns.values()), axis=0)

        # Guard against division by zero on a per-gene basis.
        signed_sum = np.where(np.abs(signed_sum) < 1e-12, np.nan, signed_sum)
        abs_sum = np.where(abs_sum < 1e-12, np.nan, abs_sum)

        contrib_columns = dict()
        abs_contrib_columns = dict()
        for comp_name in with_respect_to:
            contrib_columns[f"contrib_{comp_name}"] = (
                attr_columns[f"attr_{comp_name}"] / signed_sum
            )
            abs_contrib_columns[f"abs_contrib_{comp_name}"] = (
                abs_attr_columns[f"abs_attr_{comp_name}"] / abs_sum
            )

        result = deg.copy()
        result["Delta"] = delta
        result["Expected(A)"] = f_target[0].numpy()
        result["Expected(B)"] = f_baseline[0].numpy()
        for comp_name in with_respect_to:
            result[f"attr_{comp_name}"] = attr_columns[f"attr_{comp_name}"]
            result[f"abs_attr_{comp_name}"] = abs_attr_columns[f"abs_attr_{comp_name}"]
            result[f"contrib_{comp_name}"] = contrib_columns[f"contrib_{comp_name}"]
            result[f"abs_contrib_{comp_name}"] = abs_contrib_columns[f"abs_contrib_{comp_name}"]

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
        batch_size: int = 128,
        n_steps: int = 32,
    ) -> pd.DataFrame:
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
            Absolute-value threshold applied to ``column``. Defaults to 3.2.

        batch_size : int, optional
            Encoder batch size. Defaults to 128.

        n_steps : int, optional
            Number of Riemann-sum steps. Defaults to 32.

        Returns
        -------
        pd.DataFrame
            Filtered and attributed DEG table.

        """
        deg = pd.read_csv(deg_file, sep=None, engine="python", index_col=0)
        deg = deg.loc[deg[column].abs() >= threshold]
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

