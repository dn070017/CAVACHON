from typing import Dict, List, Mapping, Optional, Sequence, Union

import muon as mu
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from cavachon.dataloader.dataloader import DataLoader
from cavachon.environment.constants import Constants
from cavachon.layers.parameterizers.multivariate_normal_diag_sampler import (
    MultivariateNormalDiagSampler,
)
from cavachon.model.model import Model
from cavachon.modules.components.component import Component
from cavachon.utils.tensor_utils import TensorUtils


class AttributionAnalysis:
    """AttributionAnalysis

    Attribution analysis of the latent representation of the component
    to the outputs.

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
    ):
        """Constructor for ContributionAnalysis.

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
            the distribution names for each modality. Defaults to None.

        """
        self.mdata = mdata
        self.model = model
        self.dataloader = DataLoader(
            self.mdata, 1, batch_effect_colnames, distribution_names
        )

    def compute_delta_x(
        self,
        component: str,
        modality: str,
        exclude_component: str,
        batch_size: int = 128,
    ) -> np.ndarray:
        """Compute the x - x_baseline in the integrated gradients. The
        baseline is the mean of the outputs modality from the component
        without using the latent representation z of the exclude
        component.

        Parameters
        ----------
        component : str
            the outputs of which component to used.

        modality : str
            which modality of the outputs of the component to used.

        exclude_component: str
            which component to exclude (the latent representation z
            will not be used in the forward pass)

        batch_size : int, optional
            batch size used for the forward pass. Defaults to 128

        Returns
        -------
        np.ndarray
            x - x_baseline in the integrated gradients.

        """
        delta_x = []
        progress_message = "Computing delta x"
        for batch in tqdm(
            self.dataloader.dataset.batch(batch_size), desc=progress_message
        ):
            x_means_null_batch = self.compute_attribution_target_batch(
                batch=batch,
                component=component,
                modality=modality,
                with_respect_to=exclude_component,
                alpha=0.0,
            )

            x_means_full_batch = self.compute_attribution_target_batch(
                batch=batch,
                component=component,
                modality=modality,
                with_respect_to=exclude_component,
                alpha=1.0,
            )

            delta_x.append(x_means_full_batch - x_means_null_batch)

        return np.vstack(delta_x)

    def compute_integrated_gradient(
        self,
        component: str,
        modality: str,
        with_respect_to: str,
        steps: int = 10,
        selected_variables: Optional[Sequence[str]] = None,
        batch_size: int = 128,
    ) -> np.ndarray:
        """Compute the integrated gradients of ∂rho_m/∂z_m.

        Parameters
        ----------
        component: str
            the outputs of which component to used.

        modality: str
            which modality of the outputs of the component to used.

        with_respect_to: str
            compute integrated gradients with respect to the latent
            representation of which component.

        steps: int, optional
            steps in integrated gradients. Defaults to 10.

        selected_variables: Optional[Sequence[str]], optional
            the variables to used. The provided variables needs to
            match the indices of mdata[modality].var. All variables
            will be used if provided with None. Defaults to None.

        batch_size: int, optional
            batch size used for the forward pass. Defaults to 128

        Returns
        -------
        np.ndarray
            integrated gradients of ∂rho_m/∂z_m.

        """
        if selected_variables is not None:
            selected_indices = [
                self.mdata[modality].var.index.get_loc(var)
                for var in selected_variables
            ]
        else:
            selected_indices = range(self.mdata[modality].var.shape[0])

        delta_x = self.compute_delta_x(
            component=component,
            modality=modality,
            exclude_component=with_respect_to,
            batch_size=batch_size,
        )
        delta_x = tf.gather(delta_x, selected_indices, axis=1)

        delta_x_split = TensorUtils.split(delta_x, batch_size=batch_size)

        progress_message = "Computing integrated gradient"
        integrated_gradients = []
        for batch, batch_delta_x in tqdm(
            zip(self.dataloader.dataset.batch(batch_size), delta_x_split),
            desc=progress_message,
        ):
            outputs = self.model(batch, training=False)
            unintegrated_gradients_batch = None
            for alpha in tqdm([x / steps for x in range(steps)]):
                with tf.GradientTape(
                    watch_accessed_variables=False, persistent=True
                ) as tape:
                    z_variable = tf.Variable(
                        outputs.get(f"{with_respect_to}/{Constants.MODEL_OUTPUTS_Z}")
                    )
                    tape.watch(z_variable)

                    attribution_target = self.compute_attribution_target_batch(
                        batch=batch,
                        component=component,
                        modality=modality,
                        with_respect_to=with_respect_to,
                        z_variable=z_variable,
                        alpha=alpha,
                    )

                    gradients_tmp = []
                    for index in selected_indices:
                        selector = np.zeros((attribution_target.shape[1], 1))
                        selector[index][0] = 1
                        selector = tf.convert_to_tensor(selector, dtype=tf.float32)

                        gradients_j_feature = tape.gradient(
                            tf.matmul(attribution_target, selector), z_variable
                        ).numpy()
                        gradients_tmp.append(gradients_j_feature)

                    gradients = tf.stack(gradients_tmp, axis=-1)
                    scaled_gradients = (
                        1 / steps * tf.expand_dims(batch_delta_x, 1) * gradients
                    )

                    if unintegrated_gradients_batch is None:
                        unintegrated_gradients_batch = scaled_gradients
                    else:
                        unintegrated_gradients_batch += scaled_gradients

            integrated_gradients.append(unintegrated_gradients_batch)
        integrated_gradients = tf.concat(integrated_gradients, 0)

        return integrated_gradients

    def compute_attribution_target_batch(
        self,
        batch: Mapping[str, tf.Tensor],
        component: str,
        modality: str,
        with_respect_to: str,
        z_variable: Union[tf.Tensor, tf.Variable, None] = None,
        alpha: float = 1.0,
    ) -> np.ndarray:
        """Compute the means of generative data in each batch with
        selected indices.

        Parameters
        ----------
        batch: Mapping[str, tf.Tensor]
            batch inputs.

        component: str
            generative result of `modality` from which component to
            used.

        modality: str
            which modality to used from the generative result of
            `component`.

        with_respect_to: str
            compute integrated gradietn with respect to the latent
            representation of which component.

        z_variable: Union[tf.Tensor, tf.Variable, None], optional
            replace z with provided z_variable. Use the original z if
            provided with None. Defaults to None.

        alpha: float, optional
            scaling factor of z. Defaults to 1.0.

        Returns
        -------
        np.ndarray
            the means of the generative data distribution, where index
            i specify the samples, index j specify the means of the
            data distribution of the variables.

        """
        z_conditional = dict()
        z_hat_conditional = dict()
        for component_config in self.model.component_configs:
            component_name = component_config.get("name")
            component_network = self.model.components.get(component_name)
            modality_names = component_config.get(
                Constants.CONFIG_FIELD_COMPONENT_MODALITY_NAMES
            )
            component_inputs = Model.prepare_component_inputs(
                batch=batch,
                component_config=component_config,
                target_component=component_name,
                components=self.model.components,
                z_conditional=z_conditional,
                z_hat_conditional=z_hat_conditional,
            )

            preprocessor_inputs = Component.prepare_preprocessor_inputs(
                component_inputs, modality_names
            )
            preprocessor_outputs = component_network.preprocessor(
                preprocessor_inputs, training=False
            )
            z_parameters = component_network.encoder(
                preprocessor_outputs.get(component_network.preprocessor.matrix_key),
                training=False,
            )
            if component_name == with_respect_to and z_variable is not None:
                z = alpha * z_variable
            else:
                z_sampler = MultivariateNormalDiagSampler()
                z = alpha * z_sampler(z_parameters, training=False)

            hierarchical_encoder_inputs = Component.prepare_hierarchical_encoder_inputs(
                component_inputs, z
            )
            z_hat = component_network.hierarchical_encoder(hierarchical_encoder_inputs)

            z_conditional.setdefault(component_name, z)
            z_hat_conditional.setdefault(component_name, z_hat)

            if component_name == component:
                decoder_inputs = Component.prepare_decoder_inputs(
                    batch=batch,
                    modality_name=modality,
                    z_hat=z_hat,
                    preprocessor_outputs=dict(),
                )
                decoder = component_network.decoders.get(modality)
                attribution_target = decoder.compute_attribution_target(decoder_inputs)

                return attribution_target
