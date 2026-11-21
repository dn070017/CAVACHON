import copy
import warnings
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)

import muon as mu
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from cavachon.config.component_config import ComponentConfig
from cavachon.dataloader.dataloader import DataLoader
from cavachon.environment.constants import Constants
from cavachon.layers.modifiers import ToDense
from cavachon.losses.gmm_density_loss import GMMDensityLoss
from cavachon.losses.gmm_kl_divergence import GMMKLDivergence
from cavachon.losses.negative_log_data_likelihood import NegativeLogDataLikelihood
from cavachon.losses.standard_kl_divergence import StandardKLDivergence
from cavachon.layers.progressive_scaler import ProgressiveScaler
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

    component_configs: Tuple[ComponentConfig, ...]
        the configs the components were built with (read-only copies;
        the model keeps its own private copy).

    """

    def __init__(
        self,
        inputs: Mapping[Any, tf.keras.Input],
        outputs: Mapping[Any, tf.Tensor],
        components: Mapping[str, Component],
        component_configs: List[ComponentConfig],
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

        component_configs: List[ComponentConfig]
            the config used to create the components in the model.

        name: str, optional:
            Name for the tensorflow model. Defaults to 'model'.

        kwargs: Mapping[str, Any]
            additional parameters for custom models.

        """
        super().__init__(inputs=inputs, outputs=outputs, name=name)
        self.components: Mapping[str, Component] = components
        # a private copy: the model's structure must not change when the
        # caller (or anyone else) later modifies the configs passed in
        self._component_configs: List[ComponentConfig] = [
            copy.deepcopy(config) for config in component_configs
        ]

    @property
    def component_configs(self) -> Tuple[ComponentConfig, ...]:
        """The configs the components were built with, in training
        order. Read-only: a tuple of copies, so modifying it does not
        change the model."""
        return tuple(copy.deepcopy(config) for config in self._component_configs)

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
        inputs: Dict[str, Any] = dict()
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
        cls, component_configs: List[ComponentConfig], **kwargs
    ) -> Tuple:
        """Builder function for setting up components. Developers can
        overwrite this function to create custom Model.

        Parameters
        ----------
        component_configs: List[ComponentConfig]
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
        components: Dict[str, Component] = dict()
        modality_names: set = set()
        distributions = dict()
        n_vars = dict()
        n_vars_batch_effect = dict()
        for component_config in component_configs:
            modality_names = modality_names.union(set(component_config.modality_names))
            distributions.update(component_config.distribution_names)
            n_vars.update(component_config.n_vars)
            n_vars_batch_effect.update(component_config.n_vars_batch_effect)

            component_name = component_config.name
            conditional_dims_config = Model.prepare_conditional_dims_config(
                component_config, components
            )
            component_config.z_conditional_dims = conditional_dims_config.get(
                "z_conditional_dims"
            )
            component_config.z_hat_conditional_dims = conditional_dims_config.get(
                "z_hat_conditional_dims"
            )

            components.setdefault(
                component_name, Component.make(**component_config.model_dump())
            )

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
        components: Mapping[str, Component],
        component_configs: List[ComponentConfig],
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

        component_configs: List[ComponentConfig]
            the config used to create the components in the model.

        kwargs: Mapping[str, Any]
            additional parameters used for custom setup_outputs()

        Returns
        -------
        Mapping[Any, tf.Tensor]
            outputs for building tf.keras.Model using Tensorflow
            functional API.

        """
        z_conditional: Dict[str, Any] = dict()
        z_hat_conditional: Dict[str, Any] = dict()
        outputs: Dict[str, Any] = dict()
        for component_config in component_configs:
            component_name = component_config.name
            component = components.get(component_name)
            assert component is not None
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
        component_configs: List[ComponentConfig],
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
        # work on copies: setup_components writes the conditional dims
        # into the configs, which must not change the caller's objects
        component_configs = [copy.deepcopy(config) for config in component_configs]
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
        batch_size: Optional[int] = None,
        extra_outputs: Sequence[str] = (),
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

        extra_outputs: Sequence[str], optional
            with mu.MuData, more model outputs to collect besides the
            z and z_hat of every component, e.g.
            ``"{component}_z_parameters"`` (the loc and scale_diag of
            the posterior). They are returned, not stored in the obsm.
            Defaults to ().

        kwargs: Mapping[str, Any]
            Additional parameters used to compile the model.

        """
        if batch_size is None:
            batch_size = 1
        if issubclass(type(x), mu.MuData):
            # issubclass(type(x), ...) does not narrow the Union for mypy.
            mdata = cast(mu.MuData, x)
            outputs: Dict[str, Any] = dict()
            use_which_component: Dict[str, List[str]] = dict()
            save_x: Dict[str, Optional[bool]] = dict()
            save_z: Dict[str, Optional[bool]] = dict()
            save_z_hat: Dict[str, Optional[bool]] = dict()
            for component_config in self._component_configs:
                component_name = component_config.name
                outputs.setdefault(f"{component_name}_z", list())
                outputs.setdefault(f"{component_name}_z_hat", list())
                modality_names = component_config.n_vars.keys()

                for modality_name in modality_names:
                    use_which_component.setdefault(modality_name, [])
                    use_which_component[modality_name].append(component_name)
                    save_x.setdefault(
                        f"{component_name}_{modality_name}",
                        component_config.save_x.get(modality_name),
                    )
                    save_z.setdefault(
                        f"{component_name}_{modality_name}",
                        component_config.save_z.get(modality_name),
                    )
                    save_z_hat.setdefault(
                        f"{component_name}_{modality_name}",
                        component_config.save_z.get(modality_name),
                    )
                    # x_parameters are only collected for the modalities
                    # that save them
                    if component_config.save_x.get(modality_name):
                        outputs.setdefault(
                            f"{component_name}_{modality_name}_x_parameters", list()
                        )

            for key in extra_outputs:
                outputs.setdefault(key, list())

            dataloader = DataLoader(mdata, batch_size=batch_size)
            for batch in tqdm(dataloader):
                result = self.predict_on_batch(self.select_inputs(batch))
                for key in outputs:
                    outputs[key].append(result.get(key))
            for key in outputs:
                outputs[key] = np.vstack(outputs[key])

            for modality_name, component_names in use_which_component.items():
                for component_name in component_names:
                    if save_z.get(f"{component_name}_{modality_name}"):
                        mdata.mod[modality_name].obsm[f"z_{component_name}"] = (
                            outputs.get(f"{component_name}_z")
                        )
                    if save_z.get(f"{component_name}_{modality_name}"):
                        mdata.mod[modality_name].obsm[f"z_hat_{component_name}"] = (
                            outputs.get(f"{component_name}_z_hat")
                        )
                    if save_x.get(f"{component_name}_{modality_name}"):
                        mdata.mod[modality_name].obsm[
                            f"x_parameters_{component_name}"
                        ] = outputs.get(
                            f"{component_name}_{modality_name}_x_parameters"
                        )

            return outputs
        else:
            return super().predict(
                x=self.select_inputs(x), batch_size=batch_size, **kwargs
            )

    def compile(
        self,
        standard_kl_weights: Optional[Mapping[str, float]] = None,
        gmm_kl_weights: Optional[Mapping[str, float]] = None,
        **kwargs,
    ) -> None:
        """Compile the model before training. Note that the 'metrics'
        will be ignored in Model because of the incompatibility with
        Tensorflow API. The 'loss' will be setup automatically if not
        provided.

        Parameters
        ----------
        standard_kl_weights: Mapping[str, float], optional
            per-component weights for standard N(0,1) KL divergence.
            Components listed here get a standard KL loss. A weight of
            0 creates the loss switched off: its scale is 1 and its
            weight is pinned to 0, so it does not affect training (or
            the total loss used by early stopping) but its value is
            still reported, and it can be switched on later with
            ``pin_to``. Defaults to None (no standard KL for any
            component).

        gmm_kl_weights: Mapping[str, float], optional
            per-component weights for GMM KL divergence, with the same
            meaning of 0 as ``standard_kl_weights``. When both
            standard_kl_weights and gmm_kl_weights are absent for a
            component, defaults to GMM KL with weight 1.0.

        kwargs: Mapping[str, Any]
            additional parameters used to compile the model.
            ``loss_weights`` maps data-loss names
            (``{component}_{modality}_negative_log_data_likelihood``)
            to weights, with the same meaning of 0; the mapping is not
            modified, and unknown names emit a warning.

        Losses that already exist (on recompilation) are reused, so
        weights passed for them are ignored: the scheduler recompiles
        whenever the trainable variables change and relies on the losses
        (and the progress of their ProgressiveScaler weights) carrying
        over. Component.compile differs: it rebuilds its losses on every
        compile.

        Components that learn a z_hat prior also get the legacy z_hat
        GMM density loss, switched off (weight 0; see GMMDensityLoss),
        and the z_hat GMM KL divergence, pinned to 0 until the
        scheduler's z_hat annealing phase.

        """
        standard_kl_weights = standard_kl_weights or {}
        gmm_kl_weights = gmm_kl_weights or {}

        # Scheduler-facing index: maps component name → ProgressiveScaler
        # so the scheduler can call pin_to / activate / increment directly.
        # Populated below from the loss's .weight attribute; NOT used for
        # determining which losses to create or what scale to initialize.
        if not hasattr(self, "_standard_kl_weights"):
            self._standard_kl_weights = {}
        if not hasattr(self, "_gmm_kl_weights"):
            self._gmm_kl_weights = {}
        if not hasattr(self, "_z_hat_density_weights"):
            self._z_hat_density_weights: Dict[str, tf.Variable] = {}
        if not hasattr(self, "_z_hat_gmm_kl_weights"):
            self._z_hat_gmm_kl_weights = {}

        # a copy: entries are consumed below, the caller's mapping is kept
        loss_weights: Dict[str, float] = dict(kwargs.pop("loss_weights", None) or {})

        if "loss" not in kwargs:
            loss = getattr(self, "loss", None)
            loss = loss if isinstance(loss, dict) else {}
            for component_config in self._component_configs:
                component_name = component_config.name

                standard_w = standard_kl_weights.get(component_name, 0.0)
                gmm_w = gmm_kl_weights.get(component_name, 0.0)

                has_standard = component_name in standard_kl_weights
                has_gmm = component_name in gmm_kl_weights

                # Default: if neither dict specifies this component,
                # use GMM KL at 1.0 (backwards compatible with develop)
                if not has_standard and not has_gmm:
                    has_gmm = True
                    gmm_w = 1.0

                if has_standard:
                    name = (
                        f"{component_name}_{Constants.MODEL_LOSS_STANDARD_KL_POSTFIX}"
                    )
                    if name not in loss:
                        loss[name] = StandardKLDivergence(
                            weight=self._initial_scale(standard_w), name=name
                        )
                        self._switch_off_if_zero(loss[name], standard_w)
                    # scheduler index
                    self._standard_kl_weights[component_name] = loss[name].weight

                if has_gmm:
                    name = f"{component_name}_{Constants.MODEL_LOSS_GMM_KL_POSTFIX}"
                    if name not in loss:
                        loss[name] = GMMKLDivergence(
                            weight=self._initial_scale(gmm_w), name=name
                        )
                        self._switch_off_if_zero(loss[name], gmm_w)
                    # scheduler index
                    self._gmm_kl_weights[component_name] = loss[name].weight

                # scheduler index (per-component, per-modality ProgressiveScaler)
                if not hasattr(self, "_data_loss_weights"):
                    self._data_loss_weights: Dict[str, Dict[str, Any]] = {}
                if component_name not in self._data_loss_weights:
                    self._data_loss_weights[component_name] = {}
                distribution_names = component_config.distribution_names
                for modality_name in component_config.modality_names:
                    nldl_name = (
                        f"{component_name}_{modality_name}_"
                        f"{Constants.MODEL_LOSS_DATA_POSTFIX}"
                    )
                    weight = loss_weights.pop(nldl_name, 1.0)
                    if nldl_name not in loss:
                        loss[nldl_name] = NegativeLogDataLikelihood(
                            cast(str, distribution_names.get(modality_name)),
                            self._initial_scale(weight),
                            name=nldl_name,
                        )
                        self._switch_off_if_zero(loss[nldl_name], weight)
                    # scheduler index
                    self._data_loss_weights[component_name][modality_name] = loss[
                        nldl_name
                    ].weight

                # z_hat GMM density loss: legacy, replaced by the z_hat GMM
                # KL below. Kept (switched off, weight 0) because the
                # scheduler checks for it (ComponentSchedule
                # .has_z_hat_density_loss); never used for training.
                # Removal is deferred (see GMMDensityLoss).
                component = self.components.get(component_name)
                assert component is not None
                if (
                    component.learn_z_hat_priors
                    and component.z_hat_prior_parameterizer is not None
                ):
                    loss_name = (
                        f"{component_name}_"
                        f"{Constants.MODEL_LOSS_Z_HAT_GMM_DENSITY_POSTFIX}"
                    )
                    # Reuse existing Variable on recompile
                    if component_name not in self._z_hat_density_weights:
                        self._z_hat_density_weights[component_name] = tf.Variable(
                            0.0,
                            trainable=False,
                            dtype=tf.float32,
                            name=f"{component_name}_z_hat_density_weight",
                        )
                    weight_var = self._z_hat_density_weights[component_name]
                    loss.setdefault(
                        loss_name,
                        GMMDensityLoss(weight=weight_var, name=loss_name),
                    )

                    # z_hat GMM KL divergence (stochastic z_hat). The
                    # scale is the target weight; the scheduler keeps it
                    # pinned at 0 until the z_hat annealing phase.
                    z_hat_kl_name = (
                        f"{component_name}_{Constants.MODEL_LOSS_Z_HAT_GMM_KL_POSTFIX}"
                    )
                    z_hat_kl_weight = getattr(component_config, "z_hat_kl_weight", None)
                    z_hat_kl_weight = (
                        1.0 if z_hat_kl_weight is None else z_hat_kl_weight
                    )
                    if z_hat_kl_weight > 0 and z_hat_kl_name not in loss:
                        loss[z_hat_kl_name] = GMMKLDivergence(
                            weight=z_hat_kl_weight, name=z_hat_kl_name
                        )
                        loss[z_hat_kl_name].weight.pin_to(0.0)
                    if z_hat_kl_name in loss:
                        self._z_hat_gmm_kl_weights[component_name] = loss[
                            z_hat_kl_name
                        ].weight
            if loss_weights:
                warnings.warn(
                    "loss_weights has no data loss named "
                    f"{sorted(loss_weights)}; these weights are ignored.",
                    RuntimeWarning,
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

    @staticmethod
    def _initial_scale(weight: float) -> float:
        """Scale a loss is created with: the weight, or 1 for a weight
        of 0 (the loss is then switched off by ``_switch_off_if_zero``,
        and can be switched on later). Same rule as
        Component.initial_loss_scale."""
        return 1.0 if weight == 0 else float(weight)

    @staticmethod
    def _switch_off_if_zero(loss_fn: Any, weight: float) -> None:
        """Pin a new loss's weight to 0 when it was requested with
        weight 0. Its effective weight is then scale * 1e-16, so it
        does not affect training but train_step still reports its
        unweighted value. Same rule as Component.switch_off_if_zero."""
        if weight == 0:
            loss_fn.weight.pin_to(0.0)

    def select_inputs(self, data: Any) -> Any:
        """Keep only the entries of a batch that are inputs of the model.

        Keras matches dict inputs by position when the keys do not match
        the input names, so extra entries (e.g. the matrix of a modality
        that no component uses) would be fed to the wrong input.

        Parameters
        ----------
        data: Any
            a batch. Only Mapping batches are filtered, other inputs are
            returned as they are.

        Returns
        -------
        Any
            the batch restricted to the input names of the model.

        """
        inputs = self.input
        if isinstance(data, Mapping) and isinstance(inputs, Mapping):
            return {key: data[key] for key in inputs}
        return data

    def train_step(self, data: Any) -> Mapping[str, Any]:
        """Training step for one iteration (Keras ``train_step``). The
        trainable variables in the Model will be trained once after
        calling this function.

        Parameters
        ----------
        data: Any
            a batch: the inputs (a Mapping with the structure of
            self.inputs), optionally packed as (x, y, sample_weight) by
            Keras. The targets of every loss are taken from the inputs.

        Returns
        -------
        Mapping[str, Any]
            the total loss (``loss``) and the unweighted value of each
            loss, keyed by loss name.

        """
        x, _, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        with tf.GradientTape() as tape:
            results = self(self.select_inputs(x), training=True)
            y_true, y_pred = self._prepare_loss_inputs(x, results)
            loss, loss_metrics = self._evaluate_losses(y_true, y_pred)
        trainable = self.trainable_variables
        gradients = tape.gradient(loss, trainable)
        gradients = TensorUtils.remove_nan_gradients(gradients)
        self.optimizer.apply_gradients(zip(gradients, trainable))
        return loss_metrics

    def test_step(self, data: Any) -> Mapping[str, Any]:
        """Evaluation step (Keras ``test_step``, used by ``evaluate`` and
        ``fit(validation_data=...)``): the same losses as
        ``train_step``, in inference mode and without training.

        Parameters
        ----------
        data: Any
            a batch, as for ``train_step``.

        Returns
        -------
        Mapping[str, Any]
            the total loss (``loss``) and the unweighted value of each
            loss, keyed by loss name.

        """
        x, _, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        results = self(self.select_inputs(x), training=False)
        y_true, y_pred = self._prepare_loss_inputs(x, results)
        _, loss_metrics = self._evaluate_losses(y_true, y_pred)
        return loss_metrics

    def predict_step(self, data: Any) -> Any:
        """Prediction step (Keras ``predict_step``): the outputs of the
        model, with the batch restricted to its inputs first (see
        select_inputs), so datasets with extra entries are safe."""
        x, _, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        return self(self.select_inputs(x), training=False)

    def _prepare_loss_inputs(
        self, data: Mapping[Any, tf.Tensor], results: Mapping[str, tf.Tensor]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """y_true and y_pred of every compiled loss, keyed by loss name,
        in the order the losses are summed."""
        y_true: Dict[str, Any] = dict()
        y_pred: Dict[str, Any] = dict()

        for component_config in self._component_configs:
            component_name = component_config.name

            modality_names = component_config.modality_names

            # Get the prior parameters and z data (same for all phases)
            prior_params = results.get(
                f"{component_name}_{Constants.MODEL_OUTPUTS_Z_PRIOR_PARAMS}"
            )
            z_key = f"{component_name}_{Constants.MODEL_OUTPUTS_Z}"
            z_params_key = f"{component_name}_{Constants.MODEL_OUTPUTS_Z_PARAMS}"
            z_concat = tf.keras.layers.Lambda(lambda x: tf.concat(x, axis=-1))(
                [results.get(z_key), results.get(z_params_key)]
            )
            # feed every KL loss that is compiled (standard, GMM or both)
            for kl_name in (
                f"{component_name}_{Constants.MODEL_LOSS_STANDARD_KL_POSTFIX}",
                f"{component_name}_{Constants.MODEL_LOSS_GMM_KL_POSTFIX}",
            ):
                if kl_name in self.loss:
                    y_true.setdefault(kl_name, prior_params)
                    y_pred.setdefault(kl_name, z_concat)

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

            # z_hat GMM density loss (deterministic prior density)
            z_hat_density_name = (
                f"{component_name}_{Constants.MODEL_LOSS_Z_HAT_GMM_DENSITY_POSTFIX}"
            )
            if z_hat_density_name in self.loss:
                z_hat_prior_params = results.get(
                    f"{component_name}_{Constants.MODEL_OUTPUTS_Z_HAT_PRIOR_PARAMS}"
                )
                z_hat = results.get(f"{component_name}_{Constants.MODEL_OUTPUTS_Z_HAT}")
                if z_hat_prior_params is not None and z_hat is not None:
                    y_true.setdefault(z_hat_density_name, z_hat_prior_params)
                    y_pred.setdefault(z_hat_density_name, z_hat)

            # z_hat GMM KL divergence (stochastic z_hat)
            z_hat_kl_name = (
                f"{component_name}_{Constants.MODEL_LOSS_Z_HAT_GMM_KL_POSTFIX}"
            )
            if z_hat_kl_name in self.loss:
                z_hat_concat = tf.keras.layers.Lambda(lambda x: tf.concat(x, axis=-1))(
                    [
                        results.get(
                            f"{component_name}_{Constants.MODEL_OUTPUTS_Z_HAT}"
                        ),
                        results.get(
                            f"{component_name}_{Constants.MODEL_OUTPUTS_Z_HAT_PARAMS}"
                        ),
                    ]
                )
                y_true.setdefault(
                    z_hat_kl_name,
                    results.get(
                        f"{component_name}_{Constants.MODEL_OUTPUTS_Z_HAT_PRIOR_PARAMS}"
                    ),
                )
                y_pred.setdefault(z_hat_kl_name, z_hat_concat)

        return y_true, y_pred

    def _evaluate_losses(
        self, y_true: Mapping[str, Any], y_pred: Mapping[str, Any]
    ) -> Tuple[tf.Tensor, Dict[str, Any]]:
        """Evaluate every compiled loss once. Returns the total loss
        (weighted losses plus regularization losses) and the metrics:
        the total and each loss's unweighted value (weighted value
        divided by its weight; a weight of exactly 0 reports 0)."""
        weighted: Dict[str, tf.Tensor] = dict()
        for key in y_true:
            loss_fn = self.loss.get(key)
            if loss_fn is not None:
                weighted[key] = loss_fn(y_true[key], y_pred[key])

        loss = tf.add_n(list(weighted.values()))
        if self.losses:
            loss += tf.add_n(self.losses)

        loss_metrics: Dict[str, Any] = {"loss": loss}
        for key, value in weighted.items():
            weight = getattr(self.loss[key], "weight", None)
            if isinstance(weight, ProgressiveScaler):
                # a weight pinned to 0 is scale * 1e-16, so this recovers
                # the unweighted loss; a true 0 reports 0
                loss_metrics[key] = tf.math.divide_no_nan(value, weight(tf.ones(())))
            elif isinstance(weight, tf.Variable):
                loss_metrics[key] = tf.where(
                    tf.greater(weight, 0.0), value / weight, tf.zeros_like(value)
                )
            else:
                loss_metrics[key] = value
        return loss, loss_metrics

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
        z_conditional: Optional[Mapping[str, tf.Tensor]] = None,
        z_hat_conditional: Optional[Mapping[str, tf.Tensor]] = None,
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
        conditionals: Iterable[Tuple[Any, ...]]
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
        component_config: ComponentConfig, components: Mapping[str, Component]
    ) -> Dict[str, Optional[int]]:
        """Prepare the config for conditional dimensions used in
        `setup_components`. This function should not be used directly
        by the user.

        Parameters
        ----------
        component_config: List[ComponentConfig]
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
        conditional_dims_config: Dict[str, Optional[int]] = dict()

        conditionals = Model.prepare_conditionals()
        for config_key, dims_key in conditionals:
            conditional_component_names = (
                getattr(component_config, config_key, None) or []
            )
            if len(conditional_component_names) == 0:
                conditional_dims_config.setdefault(dims_key, None)
            else:
                conditional_dims = 0
                for conditional_component_name in conditional_component_names:
                    component = components.get(conditional_component_name)
                    assert component is not None
                    conditional_dims += component.z_prior_parameterizer.event_dims
                conditional_dims_config.setdefault(dims_key, conditional_dims)

        return conditional_dims_config

    @staticmethod
    def prepare_component_inputs(
        batch: Mapping[str, tf.Tensor],
        component_config: Union[ComponentConfig, Mapping[str, Any]],
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

        component_config: List[ComponentConfig]
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
        component_inputs: Dict[str, Any] = dict()
        target = components.get(target_component)
        assert target is not None
        for modality_name in target.modality_names:
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
            conditional_component_names = (
                getattr(component_config, config_key, None) or []
            )
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
            raise ValueError(f"Unknown component names: {sorted(unknown_components)}")

        outputs: Dict[str, Dict[str, Any]] = {
            Constants.MODEL_OUTPUTS_Z_PARAMS: dict(),
            Constants.MODEL_OUTPUTS_Z: dict(),
        }
        for component_config in self._component_configs:
            component_name = component_config.name
            if component_name not in requested_components:
                continue

            component_input_config = component_config.model_dump()
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
            component = self.components.get(component_name)
            assert component is not None
            component_outputs = component.encode(
                component_batch, training=training and component.trainable
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
            If ``True``, a missing parent ``z`` or ``z_hat`` raises
            ``ValueError``. If ``False``, a component with a missing
            parent is skipped (no ``z_hat`` output), and so are the
            components conditioned on its ``z_hat``.

        training: bool
            Whether to run the sublayers in training mode.

        Returns
        -------
        Mapping[str, Mapping[str, tf.Tensor]]
            Mapping with ``z_hat`` outputs keyed by component name.

        Raises
        ------
        ValueError
            Raised when unknown components or missing ``z`` are
            requested, when a parent tensor's width differs from the
            parent's latent dimension, or when parent values are missing
            in strict mode.

        """
        requested_components = set(components or self.components.keys())
        unknown_components = requested_components.difference(self.components.keys())
        if unknown_components:
            raise ValueError(f"Unknown component names: {sorted(unknown_components)}")

        accumulated_z_hat = dict(z_hat_seed or {})
        outputs: Dict[str, Dict[str, Any]] = {Constants.MODEL_OUTPUTS_Z_HAT: dict()}
        for component_config in self._component_configs:
            component_name = component_config.name
            if component_name not in requested_components:
                continue

            if component_name not in z:
                raise ValueError(f"Missing z for component '{component_name}'.")

            z_conditional, missing_z = self._collect_parents(
                component_name, component_config.conditioned_on_z, z, "z"
            )
            z_hat_conditional, missing_z_hat = self._collect_parents(
                component_name,
                component_config.conditioned_on_z_hat,
                accumulated_z_hat,
                "z_hat",
            )
            missing = [("z", name) for name in missing_z] + [
                ("z_hat", name) for name in missing_z_hat
            ]
            if missing:
                if strict:
                    kind, parent_name = missing[0]
                    raise ValueError(
                        f"Missing required parent {kind} for component "
                        f"'{component_name}': '{parent_name}'."
                    )
                # without all its parents the component cannot be encoded
                # (the hierarchical encoder has a fixed input width), so
                # it is skipped, and so are its children
                continue

            component_batch = Model.prepare_component_inputs(
                batch,
                component_config,
                component_name,
                self.components,
                z_conditional,
                z_hat_conditional,
            )
            component = self.components.get(component_name)
            assert component is not None
            component_outputs = component.hierarchical_encode(
                component_batch,
                z=z.get(component_name),
                training=training and component.trainable,
            )
            component_z_hat = component_outputs.get(Constants.MODEL_OUTPUTS_Z_HAT)
            accumulated_z_hat[component_name] = component_z_hat
            outputs[Constants.MODEL_OUTPUTS_Z_HAT][component_name] = component_z_hat

        return outputs

    def _collect_parents(
        self,
        component_name: str,
        parent_names: Iterable[str],
        available: Mapping[str, tf.Tensor],
        kind: str,
    ) -> Tuple[Dict[str, tf.Tensor], List[str]]:
        """Collect the parent tensors of a component and check their
        widths against the ones the model was built with.

        Parameters
        ----------
        component_name: str
            the child component.

        parent_names: Iterable[str]
            names of the parents conditioned on (z or z_hat).

        available: Mapping[str, tf.Tensor]
            component-keyed tensors to take the parents from.

        kind: str
            "z" or "z_hat" (for error messages).

        Returns
        -------
        Tuple[Dict[str, tf.Tensor], List[str]]
            the parent tensors found, and the names of missing parents.

        Raises
        ------
        ValueError
            if a parent tensor's last dimension differs from the
            parent's latent dimension.
        """
        found: Dict[str, tf.Tensor] = dict()
        missing: List[str] = list()
        for parent_name in parent_names:
            if parent_name not in available:
                missing.append(parent_name)
                continue
            tensor = available[parent_name]
            expected = self.components[parent_name].z_prior_parameterizer.event_dims
            width = tensor.shape[-1]
            if width is not None and width != expected:
                raise ValueError(
                    f"Parent {kind} '{parent_name}' of component "
                    f"'{component_name}': expected width {expected}, got {width}."
                )
            found[parent_name] = tensor
        return found, missing

    def decode(
        self,
        batch: Mapping[str, tf.Tensor],
        z_hat: Mapping[str, tf.Tensor],
        components: Optional[List[str]] = None,
        strict: bool = True,
        training: bool = False,
        libsize: Optional[Mapping[str, tf.Tensor]] = None,
    ) -> Mapping[str, Mapping[str, tf.Tensor]]:
        """Decode requested component ``z_hat`` tensors into ``x`` params.

        Parameters
        ----------
        batch: Mapping[str, tf.Tensor]
            Raw input batch. For every decoded component it must contain
            each modality's ``{modality_name}_batch_effect``, and the
            ``{modality_name}_matrix`` of modalities whose decoder scales
            by the library size, unless ``libsize`` gives it (see
            Component.decode).

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

        libsize: Mapping[str, tf.Tensor], optional
            library size of each modality, with shape (batch_size, 1),
            keyed by modality name and shared by the components (see
            ``compute_libsize``). Defaults to None: computed from the
            modality matrices in ``batch``.

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
            raise ValueError(f"Unknown component names: {sorted(unknown_components)}")

        outputs: Dict[str, Dict[str, Any]] = {Constants.MODEL_OUTPUTS_X_PARAMS: dict()}
        for component_config in self._component_configs:
            component_name = component_config.name
            if component_name not in requested_components:
                continue

            component_z_hat = z_hat.get(component_name)
            if component_z_hat is None:
                if strict:
                    raise ValueError(f"Missing z_hat for component '{component_name}'.")
                continue

            component_input_config = component_config.model_dump()
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
            component = self.components.get(component_name)
            assert component is not None
            component_outputs = component.decode(
                component_batch,
                component_z_hat,
                training=training and component.trainable,
                libsize=libsize,
            )
            for key, value in component_outputs.items():
                outputs[Constants.MODEL_OUTPUTS_X_PARAMS][f"{component_name}_{key}"] = (
                    value
                )

        return outputs

    def compute_libsize(
        self,
        batch: Mapping[str, tf.Tensor],
        components: Optional[List[str]] = None,
    ) -> Dict[str, tf.Tensor]:
        """Library size of each modality whose decoder scales by it,
        computed as in training (see Component.compute_libsize), keyed by
        modality name. Use it to pass ``libsize`` to ``decode``, e.g.
        a typical library size from training data when decoding new
        z_hat.

        Parameters
        ----------
        batch: Mapping[str, tf.Tensor]
            Raw input batch with the matrices of the modalities used by
            the requested components.

        components: List[str], optional
            Component names to compute it with. Defaults to all
            components.

        Returns
        -------
        Dict[str, tf.Tensor]
            library size, with shape (batch_size, 1), keyed by modality
            name.

        Raises
        ------
        ValueError
            Raised when unknown components are requested or a modality
            matrix is missing.
        """
        requested_components = set(components or self.components.keys())
        unknown_components = requested_components.difference(self.components.keys())
        if unknown_components:
            raise ValueError(f"Unknown component names: {sorted(unknown_components)}")
        libsize: Dict[str, tf.Tensor] = dict()
        for component_config in self._component_configs:
            component_name = component_config.name
            if component_name not in requested_components:
                continue
            component = self.components.get(component_name)
            assert component is not None
            for modality_name, value in component.compute_libsize(batch).items():
                libsize.setdefault(modality_name, value)
        return libsize
