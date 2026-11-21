import warnings
from collections import OrderedDict
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union, cast

import tensorflow as tf

from cavachon.environment.constants import Constants
from cavachon.layers.modifiers.to_dense import ToDense
from cavachon.layers.progressive_scaler import ProgressiveScaler
from cavachon.layers.parameterizers.mixture_multivariate_normal_diag_parameterizer_layer import (
    MixtureMultivariateNormalDiagParameterizerLayer,
)
from cavachon.layers.parameterizers.multivariate_normal_diag_sampler import (
    MultivariateNormalDiagSampler,
)
from cavachon.losses.gmm_density_loss import GMMDensityLoss
from cavachon.losses.gmm_kl_divergence import GMMKLDivergence
from cavachon.losses.negative_log_data_likelihood import (
    NegativeLogDataLikelihood,
)
from cavachon.losses.standard_kl_divergence import StandardKLDivergence
from cavachon.modules.base.encoder_latent_parameterizer import (
    EncoderLatentParameterizer,
)
from cavachon.modules.base.decoder_data_parameterizer import (
    DecoderDataParameterizer,
)
from cavachon.modules.base.hierarchical_encoder import HierarchicalEncoder
from cavachon.modules.preprocessors import Preprocessor
from cavachon.utils.tensor_utils import TensorUtils


class Component(tf.keras.Model):
    """Component

    Main component used in CAVACHON model. By defaults, it includes a
    Preprocessor, an EncoderLatentParameterizer, a HierarchicalEncoder,
    a Sampler, multiple DecoderDataParameterizer (for each modality)
    and a Parameterizer of the priors for latent distributions. Among
    the modules, the Preprocessor and HierarchicalEncoder are
    implemented using the Tensorflow functional API.

    Attributes
    ----------
    modality_names: List[str]
        names of the modalities used in the component.

    distribution_names: Mapping[str, str]
        names of the distributions for each modality. The keys are the
        names of the modalities, and the values are the corresponding
        distribution names. This is used to automatically find the
        parameterizer in modules.parameterizers for data distributions.

    preprocessor: tf.keras.Model
        preprocessor for the input data.

    encoder: tf.keras.Model
        encoder neural network.

    z_prior_parameterizer: tf.keras.layers.Layer
        parameterizer used for the priors in latent distributions. Used
        when computing the GMMKLDivergence.

    hierarchical_encoder: tf.keras.Model
        hierarchical encoder used to encode z_hat hierarchically
        through the dependency between components.

    decoders: Mapping[str, tf.keras.Model]
        decoder neural networks. The keys are the name of the modality,
        the values are the corresponding decoder neural network.

    name: str, optional
        Name for the tensorflow model. Defaults to 'component'.
    """

    def __init__(
        self,
        inputs: Mapping[Any, tf.keras.Input],
        outputs: Mapping[Any, tf.Tensor],
        modality_names: List[Any],
        distribution_names: Mapping[str, str],
        preprocessor: tf.keras.Model,
        encoder: tf.keras.Model,
        z_prior_parameterizer: tf.keras.layers.Layer,
        hierarchical_encoder: tf.keras.Model,
        z_sampler: Union[tf.keras.layers.Layer, tf.keras.Model],
        decoders: Mapping[str, tf.keras.Model],
        z_hat_prior_parameterizer: Optional[tf.keras.layers.Layer] = None,
        z_hat_sampler: Optional[Union[tf.keras.layers.Layer, tf.keras.Model]] = None,
        conditioned_on_z: List[str] = [],
        conditioned_on_z_hat: List[str] = [],
        learn_z_hat_priors: bool = False,
        name: str = "component",
        **kwargs,
    ):
        """Constructor for Component. Should not be called directly
        most of the time. Please use make() to create the model.

        Parameters
        ----------
        inputs: Mapping[Any, tf.keras.Input]):
            inputs for building tf.keras.Model using Tensorflow
            functional API. By defaults, expect to have keys
            `modality_name`_matrix, 'z_conditional' (if applicable),
            'z_hat_conditional' (if applicable) and
            `modality_name`_libsize (if applicable).

        outputs: Mapping[Any, tf.Tensor]
            outputs for building tf.keras.Model using Tensorflow
            functional API. By defaults, the keys are `model_name`_z,
            `model_name`_z_hat, `model_name`_z_parameters and
            `modality_names`_x_parameters.

        modality_names: List[str]
            names of the modalities used in the component.

        distribution_names: str
            names of the distributions for each modality, this is used
            to automatically find the parameterizer in
            modules.parameterizers for data distributions.

        preprocessor: tf.keras.Model
            preprocessor for the input data.

        encoder: tf.keras.Model
            encoder neural network.

        z_prior_parameterizer: tf.keras.layers.Layer
            parameterizer used for the priors in latent distributions.
            Used when computing the GMMKLDivergence.

        hierarchical_encoder: tf.keras.Model
            hierarchical encoder used to encode z_hat hierarchically
            through the dependency between components.

        z_sampler: Union[tf.keras.layers.Layer, tf.keras.Model]
            sampler used to transform z_parameters into z.

        decoders: Mapping[str, tf.keras.Model]
            decoder neural networks. The keys are the name of the
            modality, the values are the corresponding decoder neural
            network.

        z_hat_sampler: Union[tf.keras.layers.Layer, tf.keras.Model], optional
            sampler used to transform z_hat_parameters into z_hat.
            Defaults to None.

        conditioend_on_z: str, optional
            The current component will be conditionally independent
            with the specified components on the latent representation
            of the later one (exclude its ancestors). Note that the
            conditional independent relationships between components
            needs to be a directed acyclic graph. Defaults to [].

        conditioend_on_z_hat: str, optional
            The current component will be conditionally independent
            with the specified components on the latent representation
            of the later one (include its ancestors). Note that the
            conditional independent relationships between components
            needs to be a directed acyclic graph. Defaults to [].

        name: str, optional:
            Name for the tensorflow model. Defaults to 'component'.

        kwargs: Mapping[str, Any]
            additional parameters for custom components.

        """
        # make() forwards every config field in kwargs; of the ones
        # tf.keras.Model understands, dtype is passed on here and
        # trainable is set at the end (setting it pins alpha through
        # self.hierarchical_encoder, see __setattr__)
        keras_kwargs = {k: kwargs[k] for k in ("dtype",) if k in kwargs}
        super().__init__(inputs=inputs, outputs=outputs, name=name, **keras_kwargs)
        self.modality_names = modality_names
        self.preprocessor = preprocessor
        self.distribution_names = distribution_names
        self.encoder = encoder
        self.z_prior_parameterizer = z_prior_parameterizer
        self.z_hat_prior_parameterizer = z_hat_prior_parameterizer
        self.learn_z_hat_priors = learn_z_hat_priors
        self.hierarchical_encoder = hierarchical_encoder
        self.z_sampler = z_sampler
        self.z_hat_sampler = z_hat_sampler
        self.decoders = decoders
        self.conditioned_on_z = conditioned_on_z
        self.conditioned_on_z_hat = conditioned_on_z_hat
        # target weight of the z_hat GMM KL divergence (see compile)
        self.z_hat_kl_weight = kwargs.get("z_hat_kl_weight")
        if "trainable" in kwargs:
            self.trainable = kwargs["trainable"]

    @property
    def is_root(self) -> bool:
        """Whether the component has no parents. For a root component
        z_hat is z itself: there is no hierarchical encoder in the
        forward pass, no z_hat prior (the z prior is used) and no z_hat
        annealing."""
        return not self.conditioned_on_z and not self.conditioned_on_z_hat

    @staticmethod
    def is_root_inputs(inputs: Mapping[Any, tf.Tensor]) -> bool:
        """Whether the inputs carry no conditional z / z_hat, i.e. they
        belong to a root component."""
        return (
            Constants.MODULE_INPUTS_CONDITIONED_Z not in inputs
            and Constants.MODULE_INPUTS_CONDITIONED_Z_HAT not in inputs
        )

    @classmethod
    def setup_inputs(
        cls,
        modality_names: List[str],
        n_vars: Mapping[str, int],
        n_vars_batch_effect: Mapping[str, int],
        z_conditional_dims: Optional[int] = None,
        z_hat_conditional_dims: Optional[int] = None,
        **kwargs,
    ) -> Mapping[Any, tf.keras.Input]:
        """Builder function for setting up inputs. Developers can
        overwrite this function to create custom Component.

        Parameters
        ----------
        modality_names: List[str]
            names of the modalities used in the component.

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

        z_conditional_dims: int, optional
            dimension of z from the components of the dependency. None
            if the component does not depends on any other components.
            Defaults to None.

        z_hat_conditional_dims: int, optional
            dimension of z_hat from the components of the dependency.
            None if the component does not depends on any other
            components. Defaults to None.

        kwargs: Mapping[str, Any]
            additional parameters used for custom setup_inputs()

        Returns
        -------
        Mapping[Any, tf.keras.Input]:
            the inputs for the entire component.

        """
        inputs: Dict[Any, tf.keras.Input] = dict()

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

        if z_conditional_dims is not None:
            z_conditional = tf.keras.Input(
                shape=(z_conditional_dims,), name=Constants.MODULE_INPUTS_CONDITIONED_Z
            )
            inputs.setdefault(Constants.MODULE_INPUTS_CONDITIONED_Z, z_conditional)

        if z_hat_conditional_dims is not None:
            z_hat_conditional = tf.keras.Input(
                shape=(z_hat_conditional_dims,),
                name=Constants.MODULE_INPUTS_CONDITIONED_Z_HAT,
            )
            inputs.setdefault(
                Constants.MODULE_INPUTS_CONDITIONED_Z_HAT, z_hat_conditional
            )

        return inputs

    @classmethod
    def setup_preprocessor(
        cls,
        modality_names: List[str],
        distribution_names: Mapping[str, str],
        n_vars: Mapping[str, int],
        n_dims: int = 1024,
        name: str = "preprocessor",
        **kwargs,
    ) -> tf.keras.Model:
        """Builder function for setting up preprocessor. Developers can
        overwrite this function to create custom Component.

        Parameters
        ----------
        modality_names: str
            names of the modalities used in the component.

        distribution_names: Mapping[str, str]
            names of the distributions for each modality. The keys are
            the names of the modalities, and the values are the
            corresponding distribution names.

        n_vars: Mapping[str, int]
            number of variables for the inputs data distribution. It
            should be the size of last dimensions of inputs Tensor. The
            keys are the modality names, and the values are the
            corresponding number of variables.

        n_dims: int
            number of dimensions to reduced before concatenating the
            inputs.

        name: str, optional:
            Name for the tensorflow model. Defaults to 'preprocessor'.

        kwargs: Mapping[str, Any]
            additional parameters used for custom setup_preprocessors()

        Returns
        -------
        tf.keras.Model
            created preprocessor.

        """
        return Preprocessor.make(
            modality_names=modality_names,
            distribution_names=distribution_names,
            n_vars=n_vars,
            n_dims=n_dims,
            name=name,
        )

    @classmethod
    def setup_decoders(
        cls,
        modality_names: List[str],
        distribution_names: Mapping[str, str],
        n_vars: Mapping[str, int],
        n_decoder_layers: Union[int, Mapping[str, int]] = 3,
        name: str = "decoder",
        **kwargs,
    ) -> Mapping[Any, tf.keras.Model]:
        """Builder function for setting up decoders. Developers can
        overwrite this function to create custom Component.

        Parameters
        ----------
        modality_names: str
            names of the modalities used in the component.

        distribution_names: Mapping[str, str]
            names of the distributions for each modality. The keys are
            the names of the modalities, and the values are the
            corresponding distribution names.

        n_vars: Mapping[str, int]
            number of variables for the inputs data distribution. It
            should be the size of last dimensions of inputs Tensor. The
            keys are the modality names, and the values are the
            corresponding number of variables.

        n_decoder_layers: Union[int, Mapping[str, int]]
            number of decoder layers for each modality. If provided
            with a Mapping, the keys are the modality names, and the
            values are the corresponding number of variables. Defaults
            to 3.

        name: str, optional:
            Name for the tensorflow model. Defaults to 'preprocessor'.

        kwargs: Mapping[str, Any]
            additional parameters used for custom setup_decoders()

        Returns
        -------
        Mapping[Any, tf.keras.Model]:
            the keys for the Mapping is the names of the modalities,
            and the values are the corresponding decoders for the
            modalities.

        """

        decoders: Dict[Any, tf.keras.Model] = dict()

        if isinstance(n_decoder_layers, int):
            default_n_decoder_layers = n_decoder_layers
            n_decoder_layers = dict()
        else:
            default_n_decoder_layers = 3

        for modality_name in modality_names:
            decoders.setdefault(
                modality_name,
                DecoderDataParameterizer(
                    distribution_name=cast(str, distribution_names.get(modality_name)),
                    n_vars=cast(int, n_vars.get(modality_name)),
                    n_layers=n_decoder_layers.get(
                        modality_name, default_n_decoder_layers
                    ),
                    name=f"{name}_{modality_name}",
                ),
            )

        return decoders

    @classmethod
    def setup_encoder(
        cls,
        n_latent_dims: int = 5,
        n_encoder_layers: int = 3,
        name: str = "encoder",
        **kwargs,
    ) -> tf.keras.Model:
        """Builder function for setting up encoder. Developers can
        overwrite this function to create custom Component.

        Parameters
        ----------
        n_latent_dims: int
            number of latent dimensions. Defaults to 5.

        n_encoder_layers: int
            number of encoder layers. Defaults to 3.

        name: str, optional:
            Name for the tensorflow model. Defaults to 'encoder'.

        kwargs: Mapping[str, Any]
            additional parameters used for custom setup_encoder()

        Returns
        -------
        tf.keras.Model:
            created encoder.

        """

        return EncoderLatentParameterizer(
            n_layers=n_encoder_layers, n_latent_dims=n_latent_dims, name=name
        )

    @classmethod
    def setup_hierarchical_encoder(
        cls,
        n_latent_dims: int = 5,
        is_conditioned_on_z: bool = False,
        is_conditioned_on_z_hat: bool = False,
        progressive_iterations: int = 5000,
        name: str = "hierarchical_encoder",
        **kwargs,
    ) -> tf.keras.Model:
        """Builder function for setting up hierarchical encoder.
        Developers can overwrite this function to create custom
        Component.

        Parameters
        ----------
        n_latent_dims: int
            number of latent dimensions. Defaults to 5.

        conditioned_on_z: bool, optional
            use latent representation from the conditioned components.
            Defaults to False.

        conditioned_on_z_hat: bool, optional
            use transformed latent representation (contains information
            of all ancestor of conditioned components) from the
            conditioned components. Defaults to False.

        progressive_iterations: int, optional
            total iterations for progressive training. Defaults to
            5000.

        name: str, optional:
            Name for the tensorflow model. Defaults to 'encoder'.

        kwargs: Mapping[str, Any]
            additional parameters used for custom setup_encoders()

        Returns
        -------
        tf.keras.Model:
            created hierarchical encoder.

        """
        return HierarchicalEncoder(
            n_latent_dims=n_latent_dims,
            is_conditioned_on_z=is_conditioned_on_z,
            is_conditioned_on_z_hat=is_conditioned_on_z_hat,
            progressive_iterations=progressive_iterations,
            use_bias=kwargs.get("use_bias", False),
            name=name,
        )

    @classmethod
    def setup_z_prior_parameterizer(
        cls,
        n_latent_dims: int = 5,
        n_latent_priors: int = 11,
        name: str = "z_prior_parameterizer",
        **kwargs,
    ) -> tf.keras.Model:
        """Builder function for setting up the parameterizer of the
        priors for latent distributions. Developers can overwrite this
        function to create custom Component.

        Parameters
        ----------
        n_latent_dims: int
            number of latent dimensions. Defaults to 5.

        n_latent_priors: int
            number of priors for the latent distributions. To make the
            model identifiable, set this number to at least
            2 * n_latent_dims + 1. Defaults to 11.

        name: str, optional:
            Name for the tensorflow model. Defaults to
            'z_prior_parameterizer'.

        kwargs: Mapping[str, Any]
            additional parameters used for custom
            setup_z_prior_parameterizer()

        Returns
        -------
        tf.keras.Model:
            created parameterizer of the priors for latent
            distributions.

        """
        return MixtureMultivariateNormalDiagParameterizerLayer(
            event_dims=n_latent_dims, n_components=n_latent_priors, name=name
        )

    @classmethod
    def setup_z_hat_prior_parameterizer(
        cls,
        n_latent_dims: int = 5,
        n_latent_priors: int = 11,
        name: str = "z_hat_prior_parameterizer",
        **kwargs,
    ) -> tf.keras.Model:
        """Builder function for setting up the parameterizer of the
        priors for z_hat (deterministic latent). Developers can
        overwrite this function to create custom Component.

        Parameters
        ----------
        n_latent_dims: int
            number of latent dimensions. Defaults to 5.

        n_latent_priors: int
            number of priors for the z_hat distributions. Defaults to
            11.

        name: str, optional:
            Name for the tensorflow model. Defaults to
            'z_hat_prior_parameterizer'.

        kwargs: Mapping[str, Any]
            additional parameters used for custom
            setup_z_hat_prior_parameterizer()

        Returns
        -------
        tf.keras.Model:
            created parameterizer of the priors for z_hat
            distributions.

        """
        return MixtureMultivariateNormalDiagParameterizerLayer(
            event_dims=n_latent_dims, n_components=n_latent_priors, name=name
        )

    @classmethod
    def setup_z_sampler(
        cls, name: str = "z_sampler", **kwargs
    ) -> tf.keras.layers.Layer:
        """Builder function for setting up the sampler for the latent
        distributions. Developers can overwrite this function to create
        custom Component.

        Parameters
        ----------
        name: str, optional:
            Name for the tensorflow model. Defaults to
            'z_sampler'.

        kwargs: Mapping[str, Any]
            additional parameters used for custom
            setup_z_sampler()

        Returns
        -------
        tf.keras.Model:
            created sampler for the latent distributions.

        """
        return MultivariateNormalDiagSampler(name=name)

    @classmethod
    def setup_z_hat_sampler(
        cls, name: str = "z_hat_sampler", **kwargs
    ) -> tf.keras.layers.Layer:
        """Builder function for setting up the sampler for the z_hat
        distributions. The sampler starts deterministic (returns the
        loc) and is switched to stochastic by the scheduler in the
        z_hat annealing phase. Developers can overwrite this function
        to create custom Component.

        Parameters
        ----------
        name: str, optional:
            Name for the tensorflow model. Defaults to
            'z_hat_sampler'.

        kwargs: Mapping[str, Any]
            additional parameters used for custom
            setup_z_hat_sampler()

        Returns
        -------
        tf.keras.Model:
            created sampler for the z_hat distributions.

        """
        return MultivariateNormalDiagSampler(name=name, stochastic=False)

    @classmethod
    def setup_outputs(
        cls,
        inputs: Mapping[Any, tf.keras.Input],
        modality_names: List[str],
        preprocessor: tf.keras.Model,
        encoder: tf.keras.Model,
        hierarchical_encoder: tf.keras.Model,
        z_prior_parameterizer: tf.keras.layers.Layer,
        z_sampler: Union[tf.keras.Model, tf.keras.layers.Layer],
        decoders: Mapping[str, tf.keras.Model],
        z_hat_prior_parameterizer: Optional[tf.keras.layers.Layer] = None,
        z_hat_sampler: Optional[Union[tf.keras.Model, tf.keras.layers.Layer]] = None,
        **kwargs,
    ) -> Mapping[Any, tf.Tensor]:
        """Builder function for setting up outputs. Developers can
        overwrite this function to create custom Component.

        Parameters
        ----------
        inputs: Mapping[Any, tf.keras.Input]
            inputs created using setup_inputs()

        hierarchical_encoder_inputs: Mapping[str, tf.keras.Input]
            inputs for hierarchical encoder created using
            setup_inputs()

        modality_names: List[str]
            names of the modalities used in the component.

        preprocessor: tf.keras.Model
            preprocessor created using setup_preprocessor()

        encoder (tf.keras.Model)
            encoder created using setup_encoder()

        hierarchical_encoder (tf.keras.Model)
            hierarchical encoder created using
            setup_hierarchical_encoder()

        z_sampler: Union[tf.keras.Model, tf.keras.layers.Layer]
            sampler for latent distributions using setup_z_sampler()

        decoders: Mapping[str, tf.keras.Model]
            decoders created using setup_decoders()

        z_hat_sampler: Union[tf.keras.Model, tf.keras.layers.Layer], optional
            sampler for z_hat distributions using setup_z_hat_sampler()

        kwargs: Mapping[str, Any]
            additional parameters used for custom setup_outputs()

        Returns
        -------
        Mapping[Any, tf.Tensor]
            outputs for building tf.keras.Model using Tensorflow
            functional API.

        """
        outputs: Dict[Any, tf.Tensor] = dict()

        preprocessor_inputs = Component.prepare_preprocessor_inputs(
            inputs, modality_names
        )
        preprocessor_outputs = preprocessor(preprocessor_inputs)
        z_parameters = encoder(preprocessor_outputs.get(preprocessor.matrix_key))
        z = z_sampler(z_parameters)
        if Component.is_root_inputs(inputs):
            # without parents z_hat is z itself (and its prior the z prior)
            z_hat_parameters = z_parameters
            z_hat = z
        else:
            hierarchical_encoder_inputs = Component.prepare_hierarchical_encoder_inputs(
                inputs, z
            )
            z_hat_parameters = hierarchical_encoder(hierarchical_encoder_inputs)
            assert z_hat_sampler is not None
            z_hat = z_hat_sampler(z_hat_parameters)

        outputs.setdefault(Constants.MODEL_OUTPUTS_Z, z)
        outputs.setdefault(Constants.MODEL_OUTPUTS_Z_HAT, z_hat)
        outputs.setdefault(Constants.MODEL_OUTPUTS_Z_PARAMS, z_parameters)
        outputs.setdefault(Constants.MODEL_OUTPUTS_Z_HAT_PARAMS, z_hat_parameters)

        dummy_input = tf.keras.layers.Lambda(
            lambda x: tf.ones((1, 1)), name="z_prior_dummy_input"
        )(z)
        outputs.setdefault(
            Constants.MODEL_OUTPUTS_Z_PRIOR_PARAMS,
            z_prior_parameterizer(dummy_input),
        )

        if z_hat_prior_parameterizer is not None:
            dummy_input_z_hat = tf.keras.layers.Lambda(
                lambda x: tf.ones((1, 1)), name="z_hat_prior_dummy_input"
            )(z)
            outputs.setdefault(
                Constants.MODEL_OUTPUTS_Z_HAT_PRIOR_PARAMS,
                z_hat_prior_parameterizer(dummy_input_z_hat),
            )

        for modality_name in modality_names:
            decoder_inputs = Component.prepare_decoder_inputs(
                inputs, modality_name, z_hat, preprocessor_outputs
            )
            decoder = decoders.get(modality_name)
            assert decoder is not None
            x_parameters = decoder(decoder_inputs)
            outputs.setdefault(
                f"{modality_name}_{Constants.MODEL_OUTPUTS_X_PARAMS}", x_parameters
            )

        return outputs

    @classmethod
    def make(
        cls,
        modality_names: List[str],
        distribution_names: Mapping[str, str],
        n_vars: Mapping[str, int],
        n_vars_batch_effect: Mapping[str, int],
        n_latent_dims: int = 5,
        n_latent_priors: int = 11,
        n_encoder_layers: int = 3,
        n_decoder_layers: Union[int, Mapping[str, int]] = 3,
        z_conditional_dims: Optional[int] = None,
        z_hat_conditional_dims: Optional[int] = None,
        conditioned_on_z: List[str] = [],
        conditioned_on_z_hat: List[str] = [],
        progressive_iterations: int = 5000,
        name: str = "component",
        **kwargs,
    ) -> tf.keras.Model:
        """Make the tf.keras.Model using the functional API of
        Tensorflow.

        Parameters
        ----------
        modality_names: str
            names of the modalities used in the component.

        distribution_names: Mapping[str, str]
            names of the distributions for each modality. The keys are
            the names of the modalities, and the values are the
            corresponding distribution names.

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

        n_latent_dims: int, optional
            number of latent dimensions. Defaults to 5.

        n_latent_priors: int
            number of priors for the latent distributions. To make the
            model identifiable, set this number to at least
            2 * n_latent_dims + 1. Defaults to 11.

        n_encoder_layers: int
            number of encoder layers. Defaults to 3.

        n_decoder_layers: Union[int, Mapping[str, int]]
            number of decoder layers for each modality. If provided
            with a Mapping, the keys are the modality names, and the
            values are the corresponding number of decoder layers.
            Defaults to 3.

        z_conditional_dims: int, optional
            dimension of z from the components of the dependency. None
            if the component does not depends on any other components.
            Defaults to None.

        z_hat_conditional_dims: int, optional
            dimension of z_hat from the components of the dependency.
            None if the component does not depends on any other
            components. Defaults to None.

        conditioned_on_z: str, optional
            The current component will be conditionally independent
            with the specified components on the latent representation
            of the later one (exclude its ancestors). Note that the
            conditional independent relationships between components
            needs to be a directed acyclic graph. Defaults to [].

        conditioned_on_z_hat: str, optional
            The current component will be conditionally independent
            with the specified components on the latent representation
            of the later one (include its ancestors). Note that the
            conditional independent relationships between components
            needs to be a directed acyclic graph. Defaults to [].

        progressive_iterations: int, optional
            total iterations for progressive training. Defaults to 5000.

        name: str, optional:
            Name for the tensorflow model. Defaults to 'component'.

        kwargs: Mapping[str, Any]
            additional parameters used for the builder functions.

        Returns
        -------
        tf.keras.Model
            created model using Tensorflow functional API.

        """
        inputs = cls.setup_inputs(
            modality_names=modality_names,
            n_vars=n_vars,
            n_vars_batch_effect=n_vars_batch_effect,
            z_conditional_dims=z_conditional_dims,
            z_hat_conditional_dims=z_hat_conditional_dims,
            **kwargs,
        )

        encoder = cls.setup_encoder(
            n_latent_dims=n_latent_dims,
            n_encoder_layers=n_encoder_layers,
            name=f"{name}_encoder",
            **kwargs,
        )

        preprocessor = cls.setup_preprocessor(
            modality_names=modality_names,
            distribution_names=distribution_names,
            n_vars=n_vars,
            n_dims=encoder.max_n_neurons,
            name=f"{name}_preprocessor",
            **kwargs,
        )

        reparameterize_z_hat = kwargs.get("reparameterize_z_hat", True)
        use_bias = not reparameterize_z_hat

        hierarchical_encoder = cls.setup_hierarchical_encoder(
            n_latent_dims=n_latent_dims,
            is_conditioned_on_z=z_conditional_dims is not None,
            is_conditioned_on_z_hat=z_hat_conditional_dims is not None,
            progressive_iterations=progressive_iterations,
            use_bias=use_bias,
            name=f"{name}_hierarchical_encoder",
            **kwargs,
        )

        z_prior_parameterizer = cls.setup_z_prior_parameterizer(
            n_latent_dims=n_latent_dims,
            n_latent_priors=n_latent_priors,
            name=f"{name}_z_prior_parameterizer",
            **kwargs,
        )

        learn_z_hat_priors = kwargs.pop("learn_z_hat_priors", False) or False
        if z_conditional_dims is None and z_hat_conditional_dims is None:
            # root component: z_hat is z, so the z prior is the z_hat prior
            learn_z_hat_priors = False
        z_hat_prior_parameterizer = None
        if learn_z_hat_priors:
            z_hat_prior_parameterizer = cls.setup_z_hat_prior_parameterizer(
                n_latent_dims=n_latent_dims,
                n_latent_priors=n_latent_priors,
                name=f"{name}_z_hat_prior_parameterizer",
                **kwargs,
            )

        z_sampler = cls.setup_z_sampler(name=f"{name}_z_sampler", **kwargs)
        z_hat_sampler = cls.setup_z_hat_sampler(name=f"{name}_z_hat_sampler", **kwargs)

        decoders = cls.setup_decoders(
            modality_names=modality_names,
            distribution_names=distribution_names,
            n_vars=n_vars,
            n_decoder_layers=n_decoder_layers,
            name=name,
            **kwargs,
        )

        outputs = cls.setup_outputs(
            inputs=inputs,
            modality_names=modality_names,
            preprocessor=preprocessor,
            encoder=encoder,
            hierarchical_encoder=hierarchical_encoder,
            z_prior_parameterizer=z_prior_parameterizer,
            z_hat_prior_parameterizer=z_hat_prior_parameterizer,
            z_sampler=z_sampler,
            z_hat_sampler=z_hat_sampler,
            decoders=decoders,
            **kwargs,
        )

        return cls(
            inputs=inputs,
            outputs=outputs,
            modality_names=modality_names,
            distribution_names=distribution_names,
            preprocessor=preprocessor,
            encoder=encoder,
            z_prior_parameterizer=z_prior_parameterizer,
            z_hat_prior_parameterizer=z_hat_prior_parameterizer,
            hierarchical_encoder=hierarchical_encoder,
            z_sampler=z_sampler,
            z_hat_sampler=z_hat_sampler,
            decoders=decoders,
            conditioned_on_z=conditioned_on_z,
            conditioned_on_z_hat=conditioned_on_z_hat,
            learn_z_hat_priors=learn_z_hat_priors,
            name=name,
            **kwargs,
        )

    def encode(
        self,
        batch: Mapping[str, tf.Tensor],
        training: bool = False,
    ) -> Mapping[str, tf.Tensor]:
        """Encode a batch into latent parameters and samples.

        Parameters
        ----------
        batch: Mapping[str, tf.Tensor]
            Input tensors. Each modality must provide
            ``{modality_name}_matrix``.

        training: bool, optional
            Whether to run the encoder path in training mode. Defaults
            to False.

        Returns
        -------
        Mapping[str, tf.Tensor]
            Mapping with keys ``z_parameters`` and ``z``.

        Raises
        ------
        ValueError
            If a required modality matrix key is missing from ``batch``.
        """
        for modality_name in self.modality_names:
            modality_key = f"{modality_name}_{Constants.TENSOR_NAME_X}"
            if modality_key not in batch:
                raise ValueError(
                    f"Missing required input key '{modality_key}' in batch."
                )

        preprocessor_inputs = Component.prepare_preprocessor_inputs(
            batch, self.modality_names
        )
        preprocessor_outputs = self.preprocessor(preprocessor_inputs, training=training)
        z_parameters = self.encoder(
            preprocessor_outputs.get(self.preprocessor.matrix_key),
            training=training,
        )
        z = self.z_sampler(z_parameters, training=training)

        return {
            Constants.MODEL_OUTPUTS_Z_PARAMS: z_parameters,
            Constants.MODEL_OUTPUTS_Z: z,
        }

    def hierarchical_encode(
        self,
        batch: Mapping[str, tf.Tensor],
        z: tf.Tensor,
        training: bool = False,
    ) -> Mapping[str, tf.Tensor]:
        """Transform z into z_hat using the hierarchical encoder.

        Parameters
        ----------
        batch: Mapping[str, tf.Tensor]
            Input tensors. Must contain z_conditional and/or
            z_hat_conditional keys if the component is conditioned.

        z: tf.Tensor
            Latent sample from encode(). Expected shape
            (batch_size, n_latent_dims).

        training: bool, optional
            Whether to run in training mode. Defaults to False.

        Returns
        -------
        Mapping[str, tf.Tensor]
            Mapping with keys ``z_hat_parameters`` and ``z_hat``. For a
            root component z_hat is z and only ``z_hat`` is returned.

        Raises
        ------
        tf.errors.InvalidArgumentError
            If ``z`` does not have rank 2.
        """
        tf.debugging.assert_rank(z, 2, message="Expected 'z' to have rank 2.")
        if self.is_root:
            return {Constants.MODEL_OUTPUTS_Z_HAT: z}
        hierarchical_encoder_inputs = Component.prepare_hierarchical_encoder_inputs(
            batch, z
        )
        z_hat_parameters = self.hierarchical_encoder(
            hierarchical_encoder_inputs, training=training
        )
        assert self.z_hat_sampler is not None
        z_hat = self.z_hat_sampler(z_hat_parameters, training=training)
        return {
            Constants.MODEL_OUTPUTS_Z_HAT_PARAMS: z_hat_parameters,
            Constants.MODEL_OUTPUTS_Z_HAT: z_hat,
        }

    def decode(
        self,
        batch: Mapping[str, tf.Tensor],
        z_hat: tf.Tensor,
        training: bool = False,
        libsize: Optional[Mapping[str, tf.Tensor]] = None,
    ) -> Mapping[str, tf.Tensor]:
        """Decode a transformed latent sample into modality parameters.

        Parameters
        ----------
        batch: Mapping[str, tf.Tensor]
            Input tensors. Each modality must provide
            ``{modality_name}_batch_effect``. The modality matrices are
            only needed to compute the library size of modalities whose
            decoder scales by it (see ``requires_library_size``) when
            ``libsize`` does not give it.

        z_hat: tf.Tensor
            Hierarchically transformed latent tensor to decode.
            Expected shape is
            ``(batch_size, n_latent_dims)``.

        training: bool, optional
            Whether to run the decoder path in training mode. Defaults
            to False.

        libsize: Mapping[str, tf.Tensor], optional
            library size of each modality, with shape (batch_size, 1),
            for decoders that scale by it. Use ``library_size`` to
            compute it the way training does (e.g. from training data,
            when decoding new z_hat). Defaults to None: computed from
            the modality matrices in ``batch``.

        Returns
        -------
        Mapping[str, tf.Tensor]
            Mapping with keys
            ``{modality_name}_x_parameters`` for each modality.

        Raises
        ------
        ValueError
            If a modality batch effect is missing from ``batch``, or a
            library size is needed but neither ``libsize`` nor the
            modality matrices in ``batch`` provide it.

        ValueError
            If no decoder is registered for a modality name.

        tf.errors.InvalidArgumentError
            If ``z_hat`` does not have rank 2.
        """
        for modality_name in self.modality_names:
            key = f"{modality_name}_{Constants.TENSOR_NAME_BATCH}"
            # Model.decode passes missing entries as None
            if batch.get(key) is None:
                raise ValueError(f"Missing required input key '{key}' in batch.")

        tf.debugging.assert_rank(z_hat, 2, message="Expected 'z_hat' to have rank 2.")
        libsize = dict(libsize or {})
        missing = [
            modality_name
            for modality_name in self.modality_names
            if self.requires_library_size(modality_name)
            and modality_name not in libsize
        ]
        if missing:
            computed = self.library_size(batch, training=training)
            libsize.update({m: computed[m] for m in missing})

        outputs = dict()
        for modality_name in self.modality_names:
            decoder = self.decoders.get(modality_name)
            if decoder is None:
                raise ValueError(f"No decoder found for modality '{modality_name}'.")
            libsize_key = (
                f"{modality_name}_{Constants.TENSOR_NAME_X}_"
                f"{Constants.TENSOR_NAME_LIBSIZE}"
            )
            preprocessor_outputs = (
                {libsize_key: libsize[modality_name]}
                if self.requires_library_size(modality_name)
                else {}
            )
            decoder_inputs = Component.prepare_decoder_inputs(
                batch, modality_name, z_hat, preprocessor_outputs
            )
            x_parameters = decoder(decoder_inputs, training=training)
            outputs[f"{modality_name}_{Constants.MODEL_OUTPUTS_X_PARAMS}"] = (
                x_parameters
            )

        return outputs

    def requires_library_size(self, modality_name: str) -> bool:
        """Whether the decoder of a modality scales its outputs by the
        library size (e.g. IndependentZeroInflatedNegativeBinomial)."""
        decoder = self.decoders.get(modality_name)
        x_parameterizer = getattr(decoder, "x_parameterizer", None)
        return bool(getattr(x_parameterizer, "libsize_scaling", False))

    def library_size(
        self, batch: Mapping[str, tf.Tensor], training: bool = False
    ) -> Dict[str, tf.Tensor]:
        """Library size of each modality whose decoder scales by it,
        computed by the preprocessor exactly as in training. Note that
        it is taken after the preprocessor's transforms (e.g. the row
        sum of log1p counts for modalities that are log-transformed),
        not the raw count total.

        Parameters
        ----------
        batch: Mapping[str, tf.Tensor]
            Input tensors with the ``{modality_name}_matrix`` of every
            modality of the component.

        training: bool, optional
            Whether to run the preprocessor in training mode. Defaults
            to False.

        Returns
        -------
        Dict[str, tf.Tensor]
            library size, with shape (batch_size, 1), keyed by modality
            name (only modalities whose decoder needs it).

        Raises
        ------
        ValueError
            If a modality matrix is missing from ``batch``.
        """
        for modality_name in self.modality_names:
            key = f"{modality_name}_{Constants.TENSOR_NAME_X}"
            if batch.get(key) is None:
                raise ValueError(
                    f"Missing required input key '{key}' in batch: the library "
                    "size is computed from the modality matrices (or pass it "
                    "to decode as libsize)."
                )
        # only this component's matrices: Keras matches dict inputs by
        # position, so other entries of a raw batch would be misfed
        preprocessor_inputs = {
            f"{m}_{Constants.TENSOR_NAME_X}": batch[f"{m}_{Constants.TENSOR_NAME_X}"]
            for m in self.modality_names
        }
        preprocessor_outputs = self.preprocessor(preprocessor_inputs, training=training)
        libsize: Dict[str, tf.Tensor] = dict()
        for modality_name in self.modality_names:
            key = (
                f"{modality_name}_{Constants.TENSOR_NAME_X}_"
                f"{Constants.TENSOR_NAME_LIBSIZE}"
            )
            if (
                self.requires_library_size(modality_name)
                and key in preprocessor_outputs
            ):
                libsize[modality_name] = preprocessor_outputs[key]
        return libsize

    def compile(
        self,
        standard_kl_weights: Optional[Mapping[str, float]] = None,
        gmm_kl_weights: Optional[Mapping[str, float]] = None,
        **kwargs,
    ) -> None:
        """Compile the component before standalone training, the same way
        Model.compile does for a single component. Note that the
        'metrics' will be ignored because of the incompatibility with
        Tensorflow API. The 'loss' will be setup automatically if not
        provided.

        Parameters
        ----------
        standard_kl_weights: Mapping[str, float], optional
            weight of the standard N(0,1) KL divergence, keyed by the
            component name. Listed components get the loss. A weight of
            0 creates the loss switched off: its scale is 1 and its
            weight is pinned to 0, so it does not affect training but
            its value is still reported, and it can be switched on
            later with ``pin_to``. Defaults to None.

        gmm_kl_weights: Mapping[str, float], optional
            weight of the GMM KL divergence, with the same meaning of 0.
            When both are absent, defaults to GMM KL with weight 1.0.

        kwargs: Mapping[str, Any]
            Additional parameters used to compile the model.
            ``loss_weights`` maps data-loss names
            (``{modality}_negative_log_data_likelihood``) to weights,
            with the same meaning of 0; the mapping is not modified,
            and unknown names emit a warning.

        The loss weights are indexed in ``_standard_kl_weights``,
        ``_gmm_kl_weights``, ``_data_loss_weights`` and
        ``_z_hat_gmm_kl_weights`` (the losses' own ProgressiveScaler, so
        ``pin_to``/``assign`` on them changes the loss) and
        ``_z_hat_density_weights``.

        A component that learns a z_hat prior (it has parents) also
        gets, as in Model.compile:

        1. the z_hat GMM KL divergence, with ``z_hat_kl_weight`` as its
           scale, pinned to 0 at compile. In Model the scheduler
           switches it on in the z_hat annealing phase; when training
           the component on its own, switch it on with
           ``_z_hat_gmm_kl_weights[name].pin_to(z_hat_kl_weight)``.
        2. the legacy z_hat GMM density loss, switched off (weight 0;
           see GMMDensityLoss).

        The parents' z / z_hat are inputs of the component, so in
        standalone training they are fixed values: only this
        component's networks are trained.

        Recompiling rebuilds every loss from the arguments given, so new
        weights take effect and the indices point to the new losses
        (their progress is reset). This differs from Model.compile,
        which reuses existing losses and ignores the new weights for
        them (the scheduler relies on that).
        """
        standard_kl_weights = standard_kl_weights or {}
        gmm_kl_weights = gmm_kl_weights or {}

        if not hasattr(self, "_standard_kl_weights"):
            self._standard_kl_weights: Dict[str, Any] = {}
        if not hasattr(self, "_gmm_kl_weights"):
            self._gmm_kl_weights: Dict[str, Any] = {}
        if not hasattr(self, "_z_hat_density_weights"):
            self._z_hat_density_weights: Dict[str, tf.Variable] = {}
        if not hasattr(self, "_data_loss_weights"):
            self._data_loss_weights: Dict[str, Dict[str, Any]] = {}
        if not hasattr(self, "_z_hat_gmm_kl_weights"):
            self._z_hat_gmm_kl_weights: Dict[str, Any] = {}

        # a copy: entries are consumed below, the caller's mapping is kept
        loss_weights: Dict[str, float] = dict(kwargs.pop("loss_weights", None) or {})

        if "loss" not in kwargs:
            loss: OrderedDict[str, tf.keras.losses.Loss] = OrderedDict()

            # Default component name for weight dicts and Variable storage
            comp_name = self.name or "component"

            has_standard = comp_name in standard_kl_weights
            has_gmm = comp_name in gmm_kl_weights
            gmm_w = gmm_kl_weights.get(comp_name, 0.0)
            if not has_standard and not has_gmm:
                has_gmm = True
                gmm_w = 1.0

            if has_standard:
                standard_w = standard_kl_weights[comp_name]
                name = Constants.MODEL_LOSS_STANDARD_KL_POSTFIX
                loss[name] = StandardKLDivergence(
                    weight=self.initial_loss_scale(standard_w), name=name
                )
                self.switch_off_if_zero(loss[name], standard_w)
                self._standard_kl_weights[comp_name] = loss[name].weight

            if has_gmm:
                name = Constants.MODEL_LOSS_GMM_KL_POSTFIX
                loss[name] = GMMKLDivergence(
                    weight=self.initial_loss_scale(gmm_w), name=name
                )
                self.switch_off_if_zero(loss[name], gmm_w)
                self._gmm_kl_weights[comp_name] = loss[name].weight

            if self.learn_z_hat_priors and self.z_hat_prior_parameterizer is not None:
                loss_name = (
                    f"{self.name}_{Constants.MODEL_LOSS_Z_HAT_GMM_DENSITY_POSTFIX}"
                )
                # legacy z_hat GMM density loss, replaced by the z_hat GMM
                # KL: kept switched off (weight 0), never used for training.
                # Removal is deferred (see GMMDensityLoss).
                if comp_name not in self._z_hat_density_weights:
                    self._z_hat_density_weights[comp_name] = tf.Variable(
                        0.0,
                        trainable=False,
                        dtype=tf.float32,
                        name=f"{comp_name}_z_hat_density_weight",
                    )
                else:
                    self._z_hat_density_weights[comp_name].assign(0.0)
                weight_var = self._z_hat_density_weights[comp_name]
                loss[loss_name] = GMMDensityLoss(weight=weight_var, name=loss_name)

                # z_hat GMM KL divergence (stochastic z_hat), as in Model:
                # the scale is the target weight, pinned to 0 until it is
                # switched on
                z_hat_kl_weight = (
                    1.0 if self.z_hat_kl_weight is None else self.z_hat_kl_weight
                )
                if z_hat_kl_weight > 0:
                    name = Constants.MODEL_LOSS_Z_HAT_GMM_KL_POSTFIX
                    loss[name] = GMMKLDivergence(weight=z_hat_kl_weight, name=name)
                    loss[name].weight.pin_to(0.0)
                    self._z_hat_gmm_kl_weights[comp_name] = loss[name].weight

            self._data_loss_weights[comp_name] = {}
            for modality_name in self.modality_names:
                nldl_name = f"{modality_name}_{Constants.MODEL_LOSS_DATA_POSTFIX}"
                weight = loss_weights.pop(nldl_name, 1.0)
                loss[nldl_name] = NegativeLogDataLikelihood(
                    cast(str, self.distribution_names.get(modality_name)),
                    self.initial_loss_scale(weight),
                    name=nldl_name,
                )
                self.switch_off_if_zero(loss[nldl_name], weight)
                self._data_loss_weights[comp_name][modality_name] = loss[
                    nldl_name
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
    def initial_loss_scale(weight: float) -> float:
        """Scale a loss is created with: the weight, or 1 for a weight
        of 0 (the loss is then switched off by ``switch_off_if_zero``,
        and can be switched on later). Model.compile uses the same rule."""
        return 1.0 if weight == 0 else float(weight)

    @staticmethod
    def switch_off_if_zero(loss_fn: Any, weight: float) -> None:
        """Pin a new loss's weight to 0 when it was requested with
        weight 0. Its effective weight is then scale * 1e-16, so it
        does not affect training but its unweighted value can still be
        reported. Model.compile uses the same rule."""
        if weight == 0:
            loss_fn.weight.pin_to(0.0)

    def train_step(self, data: Mapping[Any, tf.Tensor]) -> Mapping[str, float]:
        """Training step for one iteration. The trainable variables in
        the Component will be trained once after calling this function.

        Parameters
        ----------
        data: Mapping[Any, tf.Tensor]
            input data with structure specified with self.inputs.

        Returns
        -------
        Mapping[str, float]
            the total loss (``loss``) and the unweighted value of each
            loss, keyed by loss name.

        """
        x, _, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        with tf.GradientTape() as tape:
            results = self(self.select_inputs(x), training=True)
            y_true, y_pred = self._prepare_loss_inputs(x, results)
            loss, loss_metrics = self._evaluate_losses(y_true, y_pred)
        gradients = tape.gradient(loss, self.trainable_variables)
        # same as Model.train_step: NaN gradients are replaced by 0.
        gradients = TensorUtils.remove_nan_gradients(gradients)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss_metrics

    def test_step(self, data: Mapping[Any, tf.Tensor]) -> Mapping[str, float]:
        """Evaluation step (``evaluate`` and ``validation_data`` of
        ``fit``): the same losses as ``train_step``, without training.

        Parameters
        ----------
        data: Mapping[Any, tf.Tensor]
            input data with structure specified with self.inputs.

        Returns
        -------
        Mapping[str, float]
            the total loss (``loss``) and the unweighted value of each
            loss, keyed by loss name.

        """
        x, _, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        results = self(self.select_inputs(x), training=False)
        y_true, y_pred = self._prepare_loss_inputs(x, results)
        _, loss_metrics = self._evaluate_losses(y_true, y_pred)
        return loss_metrics

    def predict_step(self, data: Any) -> Any:
        """Prediction step (``predict``): the outputs of the component,
        with the batch restricted to its inputs first (see
        select_inputs)."""
        x, _, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        return self(self.select_inputs(x), training=False)

    def select_inputs(self, data: Any) -> Any:
        """Keep only the entries of a batch that are inputs of the
        component, as Model.select_inputs does.

        Keras matches dict inputs by position when the keys do not match
        the input names, so extra entries (e.g. the matrix of a modality
        the component does not use) would be fed to the wrong input.

        Parameters
        ----------
        data: Any
            a batch. Only Mapping batches are filtered, other inputs are
            returned as they are.

        Returns
        -------
        Any
            the batch restricted to the input names of the component.

        """
        inputs = self.input
        if isinstance(data, Mapping) and isinstance(inputs, Mapping):
            return {key: data[key] for key in inputs}
        return data

    def _prepare_loss_inputs(
        self, data: Mapping[Any, tf.Tensor], results: Mapping[str, tf.Tensor]
    ) -> Tuple[Dict[str, tf.Tensor], Dict[str, tf.Tensor]]:
        """y_true and y_pred of every compiled loss, keyed by loss name."""
        y_true: Dict[str, tf.Tensor] = dict()
        y_pred: Dict[str, tf.Tensor] = dict()
        prior_params = results.get(Constants.MODEL_OUTPUTS_Z_PRIOR_PARAMS)
        z_concat = tf.keras.layers.Lambda(lambda x: tf.concat(x, axis=-1))(
            [
                results.get(Constants.MODEL_OUTPUTS_Z),
                results.get(Constants.MODEL_OUTPUTS_Z_PARAMS),
            ]
        )

        # feed every KL loss that is compiled (standard, GMM or both)
        for kl_name in (
            Constants.MODEL_LOSS_STANDARD_KL_POSTFIX,
            Constants.MODEL_LOSS_GMM_KL_POSTFIX,
        ):
            if kl_name in self.loss:
                y_true.setdefault(kl_name, prior_params)
                y_pred.setdefault(kl_name, z_concat)

        if self.learn_z_hat_priors and self.z_hat_prior_parameterizer is not None:
            z_hat_prior_params = results.get(Constants.MODEL_OUTPUTS_Z_HAT_PRIOR_PARAMS)
            z_hat = results.get(Constants.MODEL_OUTPUTS_Z_HAT)
            if z_hat_prior_params is not None and z_hat is not None:
                loss_name = (
                    f"{self.name}_{Constants.MODEL_LOSS_Z_HAT_GMM_DENSITY_POSTFIX}"
                )
                y_true.setdefault(loss_name, z_hat_prior_params)
                y_pred.setdefault(loss_name, z_hat)

        z_hat_kl_name = Constants.MODEL_LOSS_Z_HAT_GMM_KL_POSTFIX
        if z_hat_kl_name in self.loss:
            y_true.setdefault(
                z_hat_kl_name, results.get(Constants.MODEL_OUTPUTS_Z_HAT_PRIOR_PARAMS)
            )
            y_pred.setdefault(
                z_hat_kl_name,
                tf.keras.layers.Lambda(lambda x: tf.concat(x, axis=-1))(
                    [
                        results.get(Constants.MODEL_OUTPUTS_Z_HAT),
                        results.get(Constants.MODEL_OUTPUTS_Z_HAT_PARAMS),
                    ]
                ),
            )

        for modality_name in self.modality_names:
            nldl_name = f"{modality_name}_{Constants.MODEL_LOSS_DATA_POSTFIX}"
            modality_key = f"{modality_name}_{Constants.TENSOR_NAME_X}"
            data = ToDense(modality_key)(data)
            y_true.setdefault(nldl_name, data.get(modality_key))
            y_pred.setdefault(
                nldl_name,
                results.get(f"{modality_name}_{Constants.MODEL_OUTPUTS_X_PARAMS}"),
            )
        return y_true, y_pred

    def _evaluate_losses(
        self, y_true: Mapping[str, tf.Tensor], y_pred: Mapping[str, tf.Tensor]
    ) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        """Evaluate every compiled loss once. Returns the total loss
        (weighted losses plus regularization losses) and the metrics:
        the total and each loss's unweighted value (weighted value
        divided by its weight, 0 for a weight of exactly 0), as in
        Model.train_step."""
        weighted: Dict[str, tf.Tensor] = dict()
        for key in y_true:
            loss_fn = self.loss.get(key)
            if loss_fn is not None:
                weighted[key] = loss_fn(y_true[key], y_pred[key])

        loss = tf.add_n(list(weighted.values()))
        if self.losses:
            loss += tf.add_n(self.losses)

        loss_metrics: Dict[str, tf.Tensor] = {"loss": loss}
        for key, value in weighted.items():
            weight = getattr(self.loss[key], "weight", None)
            if isinstance(weight, ProgressiveScaler):
                # a weight pinned to 0 is scale * 1e-16, so this recovers
                # the unweighted loss
                loss_metrics[key] = tf.math.divide_no_nan(value, weight(tf.ones(())))
            elif isinstance(weight, tf.Variable):
                loss_metrics[key] = tf.math.divide_no_nan(value, weight)
            else:
                loss_metrics[key] = value
        return loss, loss_metrics

    def __setattr__(self, name: str, value: Any) -> None:
        """Overwrite __setattr__ function, so that every time setting
        trainable to False, it automatically set alpha in the
        progressive_scaler to 1.0.

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
                self.set_progressive_scaler_iteration(1, 1)

        return

    def set_progressive_scaler_iteration(
        self, current_iteration: float = 1, total_iterations: float = 1
    ) -> None:
        """Set the alpha values (current iteration and total
        iterations) for the hierarchical encoder.

        Parameters
        ----------
        current_iteration: float, optional
            current iteration. Defaults to 1.

        total_iterations: float, optional
            total iterations, Defaults to 1.
        """
        total_iterations = float(total_iterations)
        current_iteration = float(current_iteration)
        progressive_scaler = self.hierarchical_encoder.progressive_scaler
        progressive_scaler.total_iterations.assign(total_iterations)
        progressive_scaler.current_iteration.assign(current_iteration)

        return

    @staticmethod
    def prepare_preprocessor_inputs(
        batch: Mapping[str, tf.Tensor], modality_names: List[str]
    ) -> Mapping[str, tf.Tensor]:
        """Prepare the inputs for the preprocessor. The function
        removes batch_effect, z_conditional and z_hat_conditional from
        batch inputs to prevent warning messages.

        Parameters
        ----------
        batch: Mapping[str, tf.Tensor]
            batch inputs.

        modality_names: List[str]
            names of the modalities used in the component.

        Returns
        -------
        Mapping[str, tf.Tensor]
            keys are `modality_name`_matrix, values are the
            corresponding tensors.
        """
        preprocessor_inputs: Dict[str, tf.Tensor] = dict()
        preprocessor_inputs.update(batch)
        preprocessor_inputs.pop(Constants.MODULE_INPUTS_CONDITIONED_Z, None)
        preprocessor_inputs.pop(Constants.MODULE_INPUTS_CONDITIONED_Z_HAT, None)

        for modality_name in modality_names:
            modality_batch_key = f"{modality_name}_{Constants.TENSOR_NAME_BATCH}"
            preprocessor_inputs.pop(modality_batch_key, None)

        return preprocessor_inputs

    @staticmethod
    def prepare_hierarchical_encoder_inputs(
        batch: Mapping[str, tf.Tensor], z: tf.Tensor
    ) -> Mapping[str, tf.Tensor]:
        """Prepare inputs for hierarchical encoder. The function puts
        z, z_conditional and z_hat_conditional together in a dictionary
        as the inputs for hierarchical encoder.

        Parameters
        ----------
        batch: Mapping[str, tf.Tensor]
            batch inputs.

        z: tf.Tensor
            outputs of z_sampler.

        Returns
        -------
        Mapping[str, tf.Tensor]
            keys are `z`, `z_conditional` (if applicable),
            `z_hat_conditional` (if applicable). values are the
            corresponding tensors.
        """
        hierarchical_encoder_inputs: Dict[str, tf.Tensor] = dict()
        hierarchical_encoder_inputs.setdefault(Constants.MODEL_OUTPUTS_Z, z)
        if Constants.MODULE_INPUTS_CONDITIONED_Z in batch:
            hierarchical_encoder_inputs.setdefault(
                Constants.MODULE_INPUTS_CONDITIONED_Z,
                batch.get(Constants.MODULE_INPUTS_CONDITIONED_Z),
            )
        if Constants.MODULE_INPUTS_CONDITIONED_Z_HAT in batch:
            hierarchical_encoder_inputs.setdefault(
                Constants.MODULE_INPUTS_CONDITIONED_Z_HAT,
                batch.get(Constants.MODULE_INPUTS_CONDITIONED_Z_HAT),
            )

        return hierarchical_encoder_inputs

    @staticmethod
    def prepare_decoder_inputs(
        batch: Mapping[str, tf.Tensor],
        modality_name: str,
        z_hat: tf.Tensor,
        preprocessor_outputs: Mapping[str, tf.Tensor] = dict(),
    ) -> Mapping[str, tf.Tensor]:
        """Prepare inputs for decoder of `modality_name`. The function
        concatenates z_hat and `modality_name`_batch_effect into
        'matrix', and combined with 'libsize' (if applicable) into a
        dictionary as the inputs for decoder.

        Parameters
        ----------
        batch: Mapping[str, tf.Tensor]
            batch inputs.

        modality_name : str
            name of the target modality.

        z_hat: tf.Tensor
            outputs of hierarchical_encoder.

        preprocessor_outputs : Mapping[str, tf.Tensor], optional
            outputs of preprocessor, Defaults to dict().

        Returns
        -------
        Mapping[str, tf.Tensor]
            keys are 'matrix' and 'libsize', values are the
            concatenation of z_hat and `modality_name`_batch_effect,
            and library size tensor.

        """

        decoder_inputs: Dict[str, tf.Tensor] = dict()
        modality_batch_key = f"{modality_name}_{Constants.TENSOR_NAME_BATCH}"
        decoder_inputs.setdefault(
            Constants.TENSOR_NAME_X,
            tf.keras.layers.Lambda(lambda x: tf.concat(x, axis=-1))(
                [z_hat, batch.get(modality_batch_key)]
            ),
        )
        libsize_key = (
            f"{modality_name}_{Constants.TENSOR_NAME_X}_{Constants.TENSOR_NAME_LIBSIZE}"
        )
        if libsize_key in preprocessor_outputs:
            decoder_inputs.setdefault(
                Constants.TENSOR_NAME_LIBSIZE, preprocessor_outputs.get(libsize_key)
            )

        return decoder_inputs
