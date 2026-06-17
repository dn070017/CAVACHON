import warnings
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union

import muon as mu
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from cavachon.config.config_mapping.component_config_mapping import (
    ComponentConfigMapping,
)
from cavachon.dataloader.dataloader import DataLoader
from cavachon.environment.constants import Constants
from cavachon.layers.modifiers import ToDense
from cavachon.losses.kl_divergence import KLDivergence
from cavachon.losses.negative_log_data_likelihood import NegativeLogDataLikelihood
from cavachon.losses.vanilla_kl_divergence import VanillaKLDivergence
from cavachon.modules.components.component import Component
from cavachon.utils.general_utils import GeneralUtils
from cavachon.utils.tensor_utils import TensorUtils


class Model(tf.keras.Model):
    """Model

    Main CAVACHON model. It consists of multiple Components and the
    dependency between them.

    Attributes
    ----------
    components: Mapping[str, Component]
        the components which makes up the model.

    component_configs: List[ComponentConfigMapping]
        the config used to create the components in the model.

    """

    def __init__(
        self,
        inputs: Mapping[Any, tf.keras.Input],
        outputs: Mapping[Any, tf.Tensor],
        components: Mapping[str, Component],
        component_configs: List[ComponentConfigMapping],
        name: str = "model",
        **kwargs,
    ):
        """Constuctor for Model. Should not be called directly most of
        the time. Please use make() to create the model.

        Parameters
        ----------
        inputs: Mapping[Any, tf.keras.Input]):
            inputs for building tf.keras.Model using Tensorflow
            functional API. By defaults, expect to have keys
            'z_hat_conditional', `modality_name`_matrix, and
            `modality_name`_libsize (if applicable).

        outputs: Mapping[Any, tf.keras.Input]):
            outputs for building tf.keras.Model using Tensorflow
            functional API. By defaults, the keys are:
            1.  `component_names`_z
            2.  `component_names`_z_hat
            3.  `component_names`_z_parameters
            4.  `component_names`_`modality_nanes`_x_parameters.

        components: Mapping[str, Component]
            the components which makes up the model.

        component_configs: List[ComponentConfigMapping]
            the config used to create the components in the model.

        name: str, optional:
            Name for the tensorflow model. Defaults to 'model'.

        kwargs: Mapping[str, Any]
            additional parameters for custom models.

        """
        super().__init__(inputs=inputs, outputs=outputs, name=name)
        self.components: List[Component] = components
        self.component_configs: List[ComponentConfigMapping] = component_configs

    @classmethod
    def setup_inputs(
        cls,
        modality_names: List[str],
        n_vars: Mapping[str, int],
        n_vars_batch_effect: Mapping[str, int],
        **kwargs,
    ) -> Mapping[Any, tf.keras.Input]:
        """Builder function for setting up inputs. Developers can
        overwrite this function to create custom Model.

        Parameters
        ----------
        modality_names: str
            names of the modalities used in the model.

        n_vars: Mapping[str, int]
            number of variables for the inputs data distribution. It
            should be the size of last dimensions of inputs Tensor. The
            keys are the modality names, and the values are the
            corresponding number of variables.

        n_vars_batch_effect: Mapping[str, int]
            number of variables for the batch effect tensor. It should
            be the size of last dimensions of batch effect Tensor. The
            keys are the modality names, and the values are the
            corresponding number of variables.

        kwargs: Mapping[str, Any]
            additional parameters used for custom setup_inputs()

        Returns
        -------
        Mapping[Any, tf.keras.Input]:
            inputs for building tf.keras.Model using Tensorflow
            functional API, where keys are `modality_name`_matrix,
            values are the tf.keras.Input.

        """
        inputs = dict()
        for modality_name in modality_names:
            modality_matrix_key = f"{modality_name}_{Constants.TENSOR_NAME_X}"
            modality_batch_key = f"{modality_name}_{Constants.TENSOR_NAME_BATCH}"
            inputs.setdefault(
                modality_matrix_key,
                tf.keras.Input(
                    shape=(n_vars.get(modality_name),),
                    name=f"{modality_name}_{Constants.TENSOR_NAME_X}",
                ),
            )
            inputs.setdefault(
                modality_batch_key,
                tf.keras.Input(
                    shape=(n_vars_batch_effect.get(modality_name),),
                    name=f"{modality_name}_{Constants.TENSOR_NAME_BATCH}",
                ),
            )

        return inputs

    @classmethod
    def setup_components(
        cls, component_configs: List[ComponentConfigMapping], **kwargs
    ) -> Tuple:
        """Builder function for setting up components. Developers can
        overwrite this function to create custom Model.

        Parameters
        ----------
        component_configs: List[ComponentConfigMapping]
            the config used to create the components in the model.

        kwargs: Mapping[str, Any]
            additional parameters used for custom setup_components()

        Returns
        -------
        Tuple
            1.  The first element is the mapping of created components,
                where the keys are the component names, values are the
                created components.
            2.  The second element is the component configs but
                reordered based on the number of breadth first search
                successors (topological sort) in the dependency direct
                acyclic graph.
            3.  The third element is the list of names of all
                modalities used in the model. The last element is the
                Mapping of number of variables for each modality, where
                the keys are the modality names.

        """
        component_configs = GeneralUtils.order_components(component_configs)
        components = dict()
        modality_names = set()
        distributions = dict()
        n_vars = dict()
        n_vars_batch_effect = dict()
        for component_config in component_configs:
            modality_names = modality_names.union(
                set(
                    component_config.get(
                        Constants.CONFIG_FIELD_COMPONENT_MODALITY_NAMES
                    )
                )
            )
            distributions.update(
                component_config.get(
                    Constants.CONFIG_FIELD_COMPONENT_MODALITY_DIST_NAMES
                )
            )
            n_vars.update(component_config.get(Constants.CONFIG_FIELD_COMPONENT_N_VARS))
            n_vars_batch_effect.update(component_config.get("n_vars_batch_effect"))

            component_name = component_config.get("name")
            conditional_dims_config = Model.prepare_conditional_dims_config(
                component_config, components
            )
            component_config.update(conditional_dims_config)

            components.setdefault(component_name, Component.make(**component_config))

        return (
            components,
            component_configs,
            modality_names,
            n_vars,
            n_vars_batch_effect,
        )

    @classmethod
    def setup_outputs(
        cls,
        inputs: Mapping[Any, tf.keras.Input],
        components: List[Component],
        component_configs: List[ComponentConfigMapping],
        **kwargs,
    ) -> Mapping[Any, tf.Tensor]:
        """Builder function for setting up outputs. Developers can
        overwrite this function to create custom Model.

        Parameters
        ----------
        inputs: Mapping[Any, tf.keras.Input]
            inputs created using setup_inputs()

        components: Mapping[str, Component]
            components created by setup_components().

        component_configs: List[ComponentConfigMapping]
            the config used to create the components in the model.

        kwargs: Mapping[str, Any]
            additional parameters used for custom setup_outputs()

        Returns
        -------
        Mapping[Any, tf.Tensor]
            outputs for building tf.keras.Model using Tensorflow
            functional API.

        """
        z_conditional = dict()
        z_hat_conditional = dict()
        outputs = dict()
        for component_config in component_configs:
            component_name = component_config.get("name")
            component = components.get(component_name)
            component_inputs = Model.prepare_component_inputs(
                inputs,
                component_config,
                component_name,
                components,
                z_conditional,
                z_hat_conditional,
            )

            results = component(component_inputs)
            for key, result in results.items():
                outputs.setdefault(f"{component_name}_{key}", result)

            z_conditional.setdefault(
                component_name, results.get(Constants.MODEL_OUTPUTS_Z)
            )
            z_hat_conditional.setdefault(
                component_name, results.get(Constants.MODEL_OUTPUTS_Z_HAT)
            )

        return outputs

    @classmethod
    def make(
        cls,
        component_configs: List[ComponentConfigMapping],
        name: str = "cavachon",
        **kwargs,
    ) -> tf.keras.Model:
        """Make the tf.keras.Model using the functional API of
        Tensorflow.

        Parameters
        ----------
        component_configs: Union[Iterable[Dict[str, Any]], Dict[str, Dict[str, Any]]]
            the config used to create the components in the model.

        name: str, optional:
            name for the tensorflow model. Defaults to 'component'.

        kwargs: Mapping[str, Any]
            additional parameters used for the builder functions.

        Returns
        -------
        tf.keras.Model
            created model using Tensorflow functional API.

        """
        components, component_configs, modality_names, n_vars, n_vars_batch_effect = (
            cls.setup_components(component_configs=component_configs, **kwargs)
        )

        inputs = cls.setup_inputs(
            modality_names=modality_names,
            n_vars=n_vars,
            n_vars_batch_effect=n_vars_batch_effect,
            **kwargs,
        )

        outputs = cls.setup_outputs(
            inputs=inputs,
            components=components,
            component_configs=component_configs,
            **kwargs,
        )

        return cls(
            inputs=inputs,
            outputs=outputs,
            name=name,
            components=components,
            component_configs=component_configs,
        )

    def predict(
        self,
        x: Union[Mapping[str, tf.Tensor], mu.MuData],
        batch_size: int = None,
        **kwargs,
    ):
        """Predict based on Mapping[str, tf.Tensor] (with the same
        format constructed by the dataset of DataLoader) or mu.MuData.
        If provided with mu.MuData, the predicted z and x_parameters
        will be stored in the obsm of each modality.

        Parameters
        ----------
        x: Union[Mapping[str, tf.Tensor], mu.MuData]
            inputs.

        batch_size: int, optional
            batch size. If provided with None, will automatically set
            to 1. Defaults to None.

        kwargs: Mapping[str, Any]
            Additional parameters used to compile the model.

        """
        if batch_size is None:
            batch_size = 1
        if issubclass(type(x), mu.MuData):
            outputs = dict()
            use_which_component = dict()
            field_save_x = Constants.CONFIG_FIELD_COMPONENT_MODALITY_SAVE_X
            field_save_z = Constants.CONFIG_FIELD_COMPONENT_MODALITY_SAVE_Z
            save_x = dict()
            save_z = dict()
            save_z_hat = dict()
            for component_config in self.component_configs:
                component_name = component_config.name
                outputs.setdefault(f"{component_name}_z", list())
                outputs.setdefault(f"{component_name}_z_hat", list())
                modality_names = component_config.get(
                    Constants.CONFIG_FIELD_COMPONENT_N_VARS
                ).keys()
                predict_x = False

                for modality_name in modality_names:
                    if component_config.get(field_save_x).get(modality_name):
                        predict_x = True

                    use_which_component.setdefault(modality_name, [])
                    use_which_component.get(modality_name).append(component_name)
                    save_x.setdefault(
                        f"{component_name}_{modality_name}",
                        component_config.get(field_save_x).get(modality_name),
                    )
                    save_z.setdefault(
                        f"{component_name}_{modality_name}",
                        component_config.get(field_save_z).get(modality_name),
                    )
                    save_z_hat.setdefault(
                        f"{component_name}_{modality_name}",
                        component_config.get(field_save_z).get(modality_name),
                    )
                    if predict_x:
                        outputs.setdefault(
                            f"{component_name}_{modality_name}_x_parameters", list()
                        )

            dataloader = DataLoader(x, batch_size=batch_size)
            for batch in tqdm(dataloader):
                result = self.predict_on_batch(batch)
                for key in outputs:
                    outputs[key].append(result.get(key))
            for key in outputs:
                outputs[key] = np.vstack(outputs[key])

            for modality_name, component_names in use_which_component.items():
                for component_name in component_names:
                    if save_z.get(f"{component_name}_{modality_name}"):
                        x.mod[modality_name].obsm[f"z_{component_name}"] = outputs.get(
                            f"{component_name}_z"
                        )
                    if save_z.get(f"{component_name}_{modality_name}"):
                        x.mod[modality_name].obsm[f"z_hat_{component_name}"] = (
                            outputs.get(f"{component_name}_z_hat")
                        )
                    if save_x.get(f"{component_name}_{modality_name}"):
                        x.mod[modality_name].obsm[f"x_parameters_{component_name}"] = (
                            outputs.get(
                                f"{component_name}_{modality_name}_x_parameters"
                            )
                        )

            return outputs
        else:
            return super.__predict__(x=x, batch_size=batch_size, **kwargs)

    def compile(
        self,
        vanilla_kl_weights: Optional[Mapping[str, float]] = None,
        gmm_kl_weights: Optional[Mapping[str, float]] = None,
        **kwargs,
    ) -> None:
        """Compile the model before training. Note that the 'metrics'
        will be ignored in Model because of the incompatibility with
        Tensorflow API. The 'loss' will be setup automatically if not
        provided.

        Parameters
        ----------
        vanilla_kl_weights: Mapping[str, float], optional
            per-component weights for vanilla N(0,1) KL divergence.
            Components with weight > 0 get a vanilla KL loss. A weight
            of 0 means the loss is not created for that component.
            Defaults to None (no vanilla KL for any component).

        gmm_kl_weights: Mapping[str, float], optional
            per-component weights for GMM KL divergence. Components
            with weight > 0 get a GMM KL loss. A weight of 0 means the
            loss is not created for that component. When both
            vanilla_kl_weights and gmm_kl_weights are absent for a
            component, defaults to GMM KL with weight 1.0.

        kwargs: Mapping[str, Any]
            additional parameters used to compile the model.

        """
        vanilla_kl_weights = vanilla_kl_weights or {}
        gmm_kl_weights = gmm_kl_weights or {}

        # Create two separate weight variables
        if not hasattr(self, "_vanilla_kl_weights"):
            self._vanilla_kl_weights = {}
        if not hasattr(self, "_gmm_kl_weights"):
            self._gmm_kl_weights = {}

        loss_weights = kwargs.get("loss_weights", dict())
        kwargs.pop("loss_weights", None)

        if "loss" not in kwargs:
            loss = dict()
            for component_config in self.component_configs:
                component_name = component_config.get("name")

                kl_divergence_name = (
                    f"{component_name}_{Constants.MODEL_LOSS_KL_POSTFIX}"
                )

                vanilla_w = vanilla_kl_weights.get(component_name, 0.0)
                gmm_w = gmm_kl_weights.get(component_name, 0.0)

                has_vanilla = component_name in vanilla_kl_weights
                has_gmm = component_name in gmm_kl_weights

                # Default: if neither dict specifies this component,
                # use GMM KL at 1.0 (backwards compatible with develop)
                if not has_vanilla and not has_gmm:
                    has_gmm = True
                    gmm_w = 1.0

                if has_vanilla:
                    if component_name not in self._vanilla_kl_weights:
                        self._vanilla_kl_weights[component_name] = tf.Variable(
                            vanilla_w,
                            trainable=False,
                            dtype=tf.float32,
                            name=f"{component_name}_vanilla_kl_weight",
                        )
                    else:
                        self._vanilla_kl_weights[component_name].assign(vanilla_w)

                    loss.setdefault(
                        f"{component_name}_vanilla_kl_divergence",
                        VanillaKLDivergence(
                            weight_var=self._vanilla_kl_weights[component_name],
                            name=f"{component_name}_vanilla_kl_divergence",
                        ),
                    )

                if has_gmm:
                    if component_name not in self._gmm_kl_weights:
                        self._gmm_kl_weights[component_name] = tf.Variable(
                            gmm_w,
                            trainable=False,
                            dtype=tf.float32,
                            name=f"{component_name}_gmm_kl_weight",
                        )
                    else:
                        self._gmm_kl_weights[component_name].assign(gmm_w)

                    loss.setdefault(
                        f"{component_name}_gmm_kl_divergence",
                        KLDivergence(
                            weight_var=self._gmm_kl_weights[component_name],
                            name=f"{component_name}_gmm_kl_divergence",
                        ),
                    )

                if not hasattr(self, "_data_loss_weights"):
                    self._data_loss_weights = {}
                if component_name not in self._data_loss_weights:
                    self._data_loss_weights[component_name] = {}
                distribution_names = component_config.get(
                    Constants.CONFIG_FIELD_COMPONENT_MODALITY_DIST_NAMES
                )
                for modality_name in component_config.get("modality_names"):
                    nldl_name = (
                        f"{component_name}_{modality_name}_"
                        f"{Constants.MODEL_LOSS_DATA_POSTFIX}"
                    )
                    var = tf.Variable(
                        loss_weights.pop(nldl_name, 1.0),
                        trainable=False, dtype=tf.float32,
                        name=f"{component_name}_{modality_name}_data_weight",
                    )
                    self._data_loss_weights[component_name][modality_name] = var
                    loss.setdefault(
                        nldl_name,
                        NegativeLogDataLikelihood(
                            distribution_names.get(modality_name),
                            var,
                            name=nldl_name,
                        ),
                    )
            kwargs.setdefault("loss", loss)
        else:
            message = "".join(
                (
                    "Please make sure the provided custom losses are properly used in ",
                    f"train_step() of {self.__class__.__name__}.",
                )
            )
            warnings.warn(message, RuntimeWarning)

        if "metrics" in kwargs:
            message = "".join(
                (
                    f"{self.__class__.__name__} directly uses the loss as evaluation metrics. ",
                    "The custom metrics provided to compile() will be ignored.",
                )
            )
            warnings.warn(message, RuntimeWarning)
            kwargs.pop("metrics")

        super().compile(**kwargs)

    def train_step(self, data: Mapping[Any, tf.Tensor]) -> Mapping[str, float]:
        """Training step for one iteration. The trainable variables in
        the Model will be trained once after calling this function.

        Parameters
        ----------
        data: Mapping[Any, tf.Tensor]
            input data with structure specified with self.inputs.

        Returns
        -------
        Mapping[str, float]
            losses trained in the training iteration, where the keys
            are the names of the losses.

        """
        with tf.GradientTape() as tape:
            results = self(data, training=True)
            y_true = dict()
            y_pred = dict()

            for component_config in self.component_configs:
                component_name = component_config.get("name")

                kl_divergence_name = (
                    f"{component_name}_{Constants.MODEL_LOSS_KL_POSTFIX}"
                )
                
                modality_names = component_config.get(
                    Constants.CONFIG_FIELD_COMPONENT_MODALITY_NAMES
                )

                # Get the prior parameters and z data (same for all phases)
                prior_params = results.get(
                    f"{component_name}_{Constants.MODEL_OUTPUTS_Z_PRIOR_PARAMS}"
                )
                z_key = f"{component_name}_{Constants.MODEL_OUTPUTS_Z}"
                z_params_key = f"{component_name}_{Constants.MODEL_OUTPUTS_Z_PARAMS}"
                z_concat = tf.keras.layers.Lambda(lambda x: tf.concat(x, axis=-1))(
                    [results.get(z_key), results.get(z_params_key)]
                )
                # Check which KL losses are compiled
                vanilla_kl_name = f"{component_name}_vanilla_kl_divergence"
                gmm_kl_name = f"{component_name}_gmm_kl_divergence"

                if vanilla_kl_name in self.loss and gmm_kl_name in self.loss:
                    # PHASE 2: Both losses active
                    y_true.setdefault(vanilla_kl_name, prior_params)
                    y_pred.setdefault(vanilla_kl_name, z_concat)
                    y_true.setdefault(gmm_kl_name, prior_params)
                    y_pred.setdefault(gmm_kl_name, z_concat)
                else:
                    # PHASE 1 or 3: Single loss (standard name)
                    y_true.setdefault(kl_divergence_name, prior_params)
                    y_pred.setdefault(kl_divergence_name, z_concat)

                for modality_name in modality_names:
                    nldl_name = f"{component_name}_{modality_name}_{Constants.MODEL_LOSS_DATA_POSTFIX}"
                    modality_key = f"{modality_name}_{Constants.TENSOR_NAME_X}"
                    data = ToDense(modality_key)(data)
                    y_true.setdefault(nldl_name, data.get(modality_key))
                    y_pred.setdefault(
                        nldl_name,
                        results.get(
                            f"{component_name}_{modality_name}_{Constants.MODEL_OUTPUTS_X_PARAMS}"
                        ),
                    )

            loss = self.compute_loss(x=None, y=y_true, y_pred=y_pred)
            gradients = tape.gradient(loss, self.trainable_variables)
            gradients = TensorUtils.remove_nan_gradients(gradients)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

            loss_metrics = {"loss": loss}
            for key in y_true:
                loss_fn = self.loss.get(key)
                if loss_fn:
                    loss_value = loss_fn(y_true[key], y_pred[key])
                    loss_metrics[key] = loss_value

        return loss_metrics

    def __setattr__(self, name: str, value: Any) -> None:
        """Overwrite __setattr__ function, so that every time setting
        trainable to False, it automatically set alpha in the
        progressive_scaler of every components to 1.0.

        Parameters
        ----------
        name: str
            name of the attributes

        value: Any
            new value of the attributes.

        """
        super().__setattr__(name, value)
        if name == "trainable":
            if not value:
                for component_name in self.components.keys():
                    self.components[component_name].trainable = value

    @staticmethod
    def prepare_conditionals(
        for_dims: bool = True,
        z_conditional: Mapping[str, tf.Tensor] = None,
        z_hat_conditional: Mapping[str, tf.Tensor] = None,
    ) -> Iterable:
        """Prepare iterable conditionals used in
        `prepare_conditional_dims_config` and
        `prepare_component_inputs`. This function should not be used
        directly by the user.

        Parameters
        ----------
        for_dims: bool, optional
            whether the function is called by
            `prepare_conditional_dims_config`. Defaults to True.

        z_conditional: Mapping[str, tf.Tensor], optional
            Tensor of z_conditional, keys should be the component names
            that the current component condition on (z), value is the
            corresponding z Tensor. Ignored if `for_dims=True`. Default
            to None.

        z_hat_conditional: Mapping[str, tf.Tensor], optional
            Tensor of z_hat_conditional, keys should be the component
            names that the current component condition on (z_hat),
            value is the corresponding z_hat Tensor. Ignored if
            `for_dims=True`. Default to None.

        Returns
        -------
        Iterable
            if `for_dims=True`, return zip(
                ['conditioned_on_z', 'conditioned_on_z_hat'],
                ['z_conditional_dims', 'z_hat_conditonal_dims'])
            else, return zip(
                ['z_conditional', 'z_hat_conditional'],
                ['conditioned_on_z', 'conditioned_on_z_hat'],
                [{`modality_names`: z}, {`modality_names`: z_hat}])

        """

        conditional_input_keys = [
            Constants.MODULE_INPUTS_CONDITIONED_Z,
            Constants.MODULE_INPUTS_CONDITIONED_Z_HAT,
        ]

        conditional_config_keys = [
            Constants.CONFIG_FIELD_COMPONENT_CONDITION_Z,
            Constants.CONFIG_FIELD_COMPONENT_CONDITION_Z_HAT,
        ]

        conditional_dims_keys = [
            Constants.MODEL_INPUTS_Z_CONDITIONAL_DIMS,
            Constants.MODEL_INPUTS_Z_HAT_CONDITIONAL_DIMS,
        ]

        conditional_tensor_dicts = [z_conditional, z_hat_conditional]
        if for_dims:
            conditionals = zip(conditional_config_keys, conditional_dims_keys)
        else:
            conditionals = zip(
                conditional_input_keys,
                conditional_config_keys,
                conditional_tensor_dicts,
            )

        return conditionals

    @staticmethod
    def prepare_conditional_dims_config(
        component_config: ComponentConfigMapping, components: Mapping[str, Component]
    ) -> Dict[str, int]:
        """Prepare the config for conditional dimensions used in
        `setup_components`. This function should not be used directly
        by the user.

        Parameters
        ----------
        component_config: List[ComponentConfigMapping]
            the config used to create the current component.

        components: Mapping[str, Component]
            the components which makes up the model.

        Returns
        -------
        Dict[str, int]
            keys are the `z_conditional_dims` and
            `z_hat_conditional_dims`, values are the corresponding
            Tensor dimensions.

        """
        conditional_dims_config = dict()

        conditionals = Model.prepare_conditionals()
        for config_key, dims_key in conditionals:
            conditional_component_names = component_config.get(config_key, [])
            if len(conditional_component_names) == 0:
                conditional_dims_config.setdefault(dims_key, None)
            else:
                conditional_dims = 0
                for conditional_component_name in conditional_component_names:
                    component = components.get(conditional_component_name)
                    conditional_dims += component.z_prior_parameterizer.event_dims
                conditional_dims_config.setdefault(dims_key, conditional_dims)

        return conditional_dims_config

    @staticmethod
    def prepare_component_inputs(
        batch: Mapping[str, tf.Tensor],
        component_config: ComponentConfigMapping,
        target_component: str,
        components: Mapping[str, Component],
        z_conditional: Mapping[str, tf.Tensor] = dict(),
        z_hat_conditional: Mapping[str, tf.Tensor] = dict(),
    ) -> Dict[str, tf.Tensor]:
        """Prepare the inputs for the component used in `setup_outputs`.

        Parameters
        ----------
        batch: Mapping[str, tf.Tensor]
            batch inputs.

        component_config: List[ComponentConfigMapping]
            the config used to create the current component.

        target_component: str
            the target component name.

        components: Mapping[str, Component]
            the components which makes up the model.

        z_conditional: Mapping[str, tf.Tensor], optional
            the keys are the name of the conditioned component (z), the
            values are the corresponding z Tensor. Defaults to {}.

        z_hat_conditional: Mapping[str, tf.Tensor], optional
            the keys are the name of the conditioned component (z_hat),
            the values are the corresponding z_hat Tensor. Defaults to
            {}.

        Returns
        -------
        Dict[str, tf.Tensors]
            keys are the `{modality_name}_matrix`,
            `{modality_name}_batch_effect`, `z_conditional`,
            `z_hat_conditional`, values are the corresponding Tensors.

        """
        component_inputs = dict()
        for modality_name in components.get(target_component).modality_names:
            modality_matrix_key = f"{modality_name}_{Constants.TENSOR_NAME_X}"
            modality_batch_key = f"{modality_name}_{Constants.TENSOR_NAME_BATCH}"
            component_inputs.setdefault(
                modality_matrix_key, batch.get(modality_matrix_key)
            )
            component_inputs.setdefault(
                modality_batch_key, batch.get(modality_batch_key)
            )

        conditionals = Model.prepare_conditionals(
            False, z_conditional, z_hat_conditional
        )

        for input_key, config_key, tensor_dict in conditionals:
            conditional_tensors = []
            conditional_component_names = component_config.get(config_key, [])
            if len(conditional_component_names) != 0:
                for conditional_component_name in conditional_component_names:
                    conditional_tensors.append(
                        tensor_dict.get(conditional_component_name)
                    )
                conditional_tensors = tf.keras.layers.Lambda(
                    lambda x: tf.concat(x, axis=-1)
                )(conditional_tensors)
                component_inputs.setdefault(input_key, conditional_tensors)

        return component_inputs

    def encode(
        self,
        batch: Mapping[str, tf.Tensor],
        components: Optional[List[str]] = None,
        training: bool = False,
    ) -> Mapping[str, Mapping[str, tf.Tensor]]:
        """Encode requested components into latent outputs.

        Parameters
        ----------
        batch: Mapping[str, tf.Tensor]
            Raw input batch.

        components: List[str], optional
            Component names to encode. Defaults to all components.

        training: bool
            Whether to run the sublayers in training mode.

        Returns
        -------
        Mapping[str, Mapping[str, tf.Tensor]]
            Mapping with ``z_parameters`` and ``z`` outputs keyed by
            component name.

        Raises
        ------
        ValueError
            Raised when unknown component names are requested.

        """
        requested_components = set(components or self.components.keys())
        unknown_components = requested_components.difference(self.components.keys())
        if unknown_components:
            raise ValueError(
                f"Unknown component names: {sorted(unknown_components)}"
            )

        outputs = {
            Constants.MODEL_OUTPUTS_Z_PARAMS: dict(),
            Constants.MODEL_OUTPUTS_Z: dict(),
        }
        for component_config in self.component_configs:
            component_name = component_config.get("name")
            if component_name not in requested_components:
                continue

            component_input_config = dict(component_config)
            component_input_config.update(
                {
                    Constants.CONFIG_FIELD_COMPONENT_CONDITION_Z: [],
                    Constants.CONFIG_FIELD_COMPONENT_CONDITION_Z_HAT: [],
                }
            )
            component_batch = Model.prepare_component_inputs(
                batch,
                component_input_config,
                component_name,
                self.components,
                {},
                {},
            )
            component_outputs = self.components.get(component_name).encode(
                component_batch, training=training
            )
            outputs[Constants.MODEL_OUTPUTS_Z_PARAMS][component_name] = (
                component_outputs.get(Constants.MODEL_OUTPUTS_Z_PARAMS)
            )
            outputs[Constants.MODEL_OUTPUTS_Z][component_name] = component_outputs.get(
                Constants.MODEL_OUTPUTS_Z
            )

        return outputs

    def hierarchical_encode(
        self,
        batch: Mapping[str, tf.Tensor],
        z: Mapping[str, tf.Tensor],
        z_hat_seed: Optional[Mapping[str, tf.Tensor]] = None,
        components: Optional[List[str]] = None,
        strict: bool = True,
        training: bool = False,
    ) -> Mapping[str, Mapping[str, tf.Tensor]]:
        """Hierarchically encode requested latents into z_hat.

        Parameters
        ----------
        batch: Mapping[str, tf.Tensor]
            Raw input batch used for conditional inputs.

        z: Mapping[str, tf.Tensor]
            Component-keyed latent samples from ``encode``.

        z_hat_seed: Mapping[str, tf.Tensor], optional
            Pre-seeded ``z_hat`` values. Defaults to ``{}``.

        components: List[str], optional
            Component names to process. Defaults to all components.

        strict: bool
            If ``True``, missing parent ``z`` or ``z_hat`` raises
            ``ValueError``. If ``False``, missing parents are skipped.

        training: bool
            Whether to run the sublayers in training mode.

        Returns
        -------
        Mapping[str, Mapping[str, tf.Tensor]]
            Mapping with ``z_hat`` outputs keyed by component name.

        Raises
        ------
        ValueError
            Raised when unknown components, missing ``z``, or missing
            required parent values are requested in strict mode.

        """
        requested_components = set(components or self.components.keys())
        unknown_components = requested_components.difference(self.components.keys())
        if unknown_components:
            raise ValueError(
                f"Unknown component names: {sorted(unknown_components)}"
            )

        accumulated_z_hat = dict(z_hat_seed or {})
        outputs = {Constants.MODEL_OUTPUTS_Z_HAT: dict()}
        for component_config in self.component_configs:
            component_name = component_config.get("name")
            if component_name not in requested_components:
                continue

            if component_name not in z:
                raise ValueError(f"Missing z for component '{component_name}'.")

            z_conditional = dict()
            for parent_name in component_config.get(
                Constants.CONFIG_FIELD_COMPONENT_CONDITION_Z, []
            ):
                if parent_name in z:
                    z_conditional[parent_name] = z.get(parent_name)
                elif strict:
                    raise ValueError(
                        "Missing required parent z for component "
                        f"'{component_name}': '{parent_name}'."
                    )

            z_hat_conditional = dict()
            for parent_name in component_config.get(
                Constants.CONFIG_FIELD_COMPONENT_CONDITION_Z_HAT, []
            ):
                if parent_name in accumulated_z_hat:
                    z_hat_conditional[parent_name] = accumulated_z_hat.get(parent_name)
                elif strict:
                    raise ValueError(
                        "Missing required parent z_hat for component "
                        f"'{component_name}': '{parent_name}'."
                    )

            component_batch = Model.prepare_component_inputs(
                batch,
                component_config,
                component_name,
                self.components,
                z_conditional,
                z_hat_conditional,
            )
            component_outputs = self.components.get(component_name).hierarchical_encode(
                component_batch,
                z.get(component_name),
                training=training,
            )
            component_z_hat = component_outputs.get(Constants.MODEL_OUTPUTS_Z_HAT)
            accumulated_z_hat[component_name] = component_z_hat
            outputs[Constants.MODEL_OUTPUTS_Z_HAT][component_name] = component_z_hat

        return outputs

    def decode(
        self,
        batch: Mapping[str, tf.Tensor],
        z_hat: Mapping[str, tf.Tensor],
        components: Optional[List[str]] = None,
        strict: bool = True,
        training: bool = False,
    ) -> Mapping[str, Mapping[str, tf.Tensor]]:
        """Decode requested component ``z_hat`` tensors into ``x`` params.

        Parameters
        ----------
        batch: Mapping[str, tf.Tensor]
            Raw input batch used for conditional inputs.

        z_hat: Mapping[str, tf.Tensor]
            Component-keyed hierarchical latents. Pass ``z_hat`` here,
            not raw ``z``.

        components: List[str], optional
            Component names to decode. Defaults to all components.

        strict: bool
            If ``True``, missing ``z_hat`` raises ``ValueError``. If
            ``False``, missing components are skipped.

        training: bool
            Whether to run the sublayers in training mode.

        Returns
        -------
        Mapping[str, Mapping[str, tf.Tensor]]
            Mapping with ``x_parameters`` outputs keyed by component and
            modality.

        Raises
        ------
        ValueError
            Raised when unknown components or required ``z_hat`` values
            are missing.

        """
        requested_components = set(components or self.components.keys())
        unknown_components = requested_components.difference(self.components.keys())
        if unknown_components:
            raise ValueError(
                f"Unknown component names: {sorted(unknown_components)}"
            )

        outputs = {Constants.MODEL_OUTPUTS_X_PARAMS: dict()}
        for component_config in self.component_configs:
            component_name = component_config.get("name")
            if component_name not in requested_components:
                continue

            component_z_hat = z_hat.get(component_name)
            if component_z_hat is None:
                if strict:
                    raise ValueError(
                        f"Missing z_hat for component '{component_name}'."
                    )
                continue

            component_input_config = dict(component_config)
            component_input_config.update(
                {
                    Constants.CONFIG_FIELD_COMPONENT_CONDITION_Z: [],
                    Constants.CONFIG_FIELD_COMPONENT_CONDITION_Z_HAT: [],
                }
            )
            component_batch = Model.prepare_component_inputs(
                batch,
                component_input_config,
                component_name,
                self.components,
                {},
                {},
            )
            component_outputs = self.components.get(component_name).decode(
                component_batch,
                component_z_hat,
                training=training,
            )
            for key, value in component_outputs.items():
                outputs[Constants.MODEL_OUTPUTS_X_PARAMS][
                    f"{component_name}_{key}"
                ] = value

        return outputs
