import itertools
import os
from collections import defaultdict
from copy import deepcopy
from typing import Any, List, Mapping, Optional, Tuple

import mlflow
import muon as mu
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from cavachon.dataloader.dataloader import DataLoader
from cavachon.distributions.mixture_multivariate_normal_diag_distribution import (
    MixtureMultivariateNormalDiagDistribution,
)
from cavachon.distributions.multivariate_normal_diag_distribution import (
    MultivariateNormalDiagDistribution,
)
from cavachon.environment.constants import Constants


class PeriodicTSNECallback(tf.keras.callbacks.Callback):
    """
    new test callback to save logpy_z and z per n epoch
    then append the callback in fit()
    """

    def __init__(
        self,
        mdata: mu.MuData,  # we need to pass mdata into it
        component: str,
        outdir: str,
        batch_size = int,
        every: int = 100,
        batch_effect_colnames: Optional[Mapping[str, List[str]]] = None,
        distribution_names: Optional[Mapping[str, str]] = None,
    ):
        super().__init__()
        self.mdata = mdata
        self.component = component
        self.every = int(every)
        self.batch_effect_colnames = batch_effect_colnames
        self.distribution_names = distribution_names
        self.batch_size = batch_size
        self.output_dir = outdir
        os.makedirs(self.output_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        # here we adjust the freq of saving the snaphot
        if (epoch + 1) % self.every != 0:
            return

        component = self.component

        # extract mean latent z
        outputs = self.model.predict(self.mdata, batch_size=self.batch_size)
        z_full = outputs[f"{self.component}_z"]

        # compute logpy_z (copied from cluster_analysis.py)
        z_prior_parameterizer = self.model.components[component].z_prior_parameterizer
        z_prior_parameters = tf.squeeze(z_prior_parameterizer(tf.ones((1, 1))))

        dist_z_y = MultivariateNormalDiagDistribution.from_parameterizer_output(
            z_prior_parameters[..., 1:]
        )
        dist_z = MixtureMultivariateNormalDiagDistribution.from_parameterizer_output(
            z_prior_parameters
        )
        logpy = tf.math.log(tf.math.softmax(z_prior_parameters[..., 0]) + 1e-7)

        logpy_z_parts = []
        dataloader = DataLoader(
            self.mdata,
            self.batch_size,
            self.batch_effect_colnames,
            self.distribution_names,
        )
        for batch_data in tqdm(dataloader, desc=f"Epoch {epoch}: computing logpy_z"):
            outs = self.model(batch_data, training=False)
            z = outs[f"{self.component}_z"]
            logpz_y = dist_z_y.log_prob(tf.expand_dims(z, -2))
            logpz = tf.expand_dims(dist_z.log_prob(z), -1)
            logpy_z_parts.append(logpy + logpz_y - logpz)

        logpy_z = np.vstack([x.numpy() for x in logpy_z_parts])

        # 3) Save z and logpy_z in the configured results directory
        np.save(os.path.join(self.output_dir, f"{epoch}_z.h5"), z_full)
        np.save(os.path.join(self.output_dir, f"{epoch}_logpy_z.h5"), logpy_z)


# -------------------------------------


class SequentialTrainingScheduler:
    """SequentialTrainingScheduler

    Training scheduler that sets the loss weight and stop the gradient
    of trained components sequentially during training process.

    Attributes
    ----------
    model : tf.keras.Model
        input model that needs to be trained.

    component_configs: List[ComponentConfigMapping]
        the config used to create the components in the model.

    optimizer: str
        optimizer used to train the model (only the attributes of the
        optimizer will be used)

    early_stopping: bool
        whether or not to use early stopping when training the model.

    training_order: Mapping[int, List[str]]
        the training order of the components, the keys are the training
        order, the values are lists of the components trained in the
        corresponding order.

    modality_weight: Mapping[str, Mapping[str, int]]
        the weight of the data distribution by component. The keys are
        the component names, the values are the mapping where the keys
        are the modality names and the values are the weight.

    """

    def __init__(
        self,
        model: tf.keras.Model,
        mdata,
        optimizer: str = "adam",
        learning_rate: float = 1e-4,
        early_stopping: bool = True,
        batch_size: int = 128,
        outdir: Optional[str] = None,
    ):
        """Constructor for SequentialTrainingScheduler.

        Parameters
        ----------
        model: tf.keras.Model
            input model that needs to be trained.

        optimizer: tf.keras.optimizers.Optimizer, optional
            optimizer used to train the model (only the attributes of
            the optimizer will be used if provided with Optimizer).
            Defaults to 'adam'.

        learning_rate: float, optional
            learning rate for the optimizer. Defaults to 1e-4.

        early_stopping: bool, optional
            whether or not to use early stopping when training the model.

        """
        self.model = model
        self.mdata = mdata
        self.component_configs = self.model.component_configs
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.early_stopping = early_stopping
        self.training_order = self.compute_component_training_order()
        self.modality_weight = self.compute_modality_weight()
        self.batch_size = batch_size
        self.outdir = outdir

    def compute_component_training_order(self) -> Mapping[int, List[str]]:
        """Compute the training order of the components based on the
        order of topological sort of the input dependency graph.

        Returns
        -------
        Mapping[int, List[str]]
            the training order of the components, the keys are the
            training order, the values are lists of the components
            trained in the corresponding order.
        """
        training_order = list()
        component_order = dict()
        self.run_progressive_training = dict()
        component_configs = self.component_configs
        training_order.append([])
        for component_config in component_configs:
            component_name = component_config.get("name")
            conditioned_on_z = component_config.get(
                Constants.CONFIG_FIELD_COMPONENT_CONDITION_Z, []
            )
            conditioned_on_z_hat = component_config.get(
                Constants.CONFIG_FIELD_COMPONENT_CONDITION_Z_HAT, []
            )
            if len(conditioned_on_z) == 0 and len(conditioned_on_z_hat) == 0:
                self.run_progressive_training[component_name] = False
            else:
                self.run_progressive_training[component_name] = True

            conditioned_on = itertools.chain(conditioned_on_z, conditioned_on_z_hat)
            order = max([0] + [component_order[x] + 1 for x in conditioned_on])
            if order > len(training_order) - 1:
                training_order.append([])
            training_order[order].append(component_name)
            component_order.setdefault(component_name, order)

        return [[x] for xs in training_order for x in xs]  # training_order

    def compute_modality_weight(
        self, constant: bool = False
    ) -> Mapping[str, Mapping[str, int]]:
        """Compute the weight of the data distribution for each
        modality.

        Parameters
        ----------
        constant : bool, optional
            whether or not to use constant weight for the data
            distribution. Defaults to False.

        Returns
        -------
        Mapping[str, Mapping[str, int]]
            the weight of the data distribution by component. The keys
            are the component names, the values are the mapping where
            the keys are the modality names and the values are the
            weight.
        """
        modality_weight_by_component = defaultdict(dict)
        for component_config in self.component_configs:
            component_name = component_config.get("name")
            modality_weight = dict()
            if not constant:
                n_vars = component_config.get(Constants.CONFIG_FIELD_COMPONENT_N_VARS)
                total_vars = 0
                total_scaled_weight = 0
                for modality_name, n_var in n_vars.items():
                    total_vars += n_var
                for modality_name, n_var in n_vars.items():
                    scaled_weight = total_vars / n_var
                    total_scaled_weight += scaled_weight
                    modality_weight.setdefault(modality_name, scaled_weight)
                for modality_name, scaled_weight in modality_weight.items():
                    modality_weight[modality_name] = scaled_weight / total_scaled_weight
            else:
                for modality_name in n_vars.keys():
                    modality_weight.setdefault(modality_name, 1.0)
            modality_weight_by_component.setdefault(component_name, modality_weight)

        return modality_weight_by_component

    def fit(self, x: tf.data.Dataset, **kwargs) -> List[tf.keras.callbacks.History]:
        """Fit self.model sequentially.

        Parameters
        ----------
        x: tf.data.Dataset
            input dataset created by DataLoader.

        **kwargs: Mapping[str, Any]
            additional arguments passed to self.model.fit.

        Returns
        -------
        List[tf.keras.callbacks.History]
            history of model.fit in each step.
        """

        n_batches = len(x)
        learning_rate = self.learning_rate
        history = []

        experiment_name = self.model.name
        mlflow.set_experiment(experiment_name)
        experiment = mlflow.get_experiment_by_name(experiment_name)

        for component_order, train_components in enumerate(self.training_order):
            # progressive training
            if self.run_progressive_training.get(train_components[0]):
                loss_weights, max_n_progressive_epochs = (
                    self.setup_component_and_loss_weights(
                        train_components=train_components,
                        n_batches=n_batches,
                        initial_iteration=0.0,
                    )
                )

                if max_n_progressive_epochs != 0:
                    run_name = f"Training/{component_order}/Progressive/{'/'.join(train_components)}"
                    mlflow.start_run(
                        experiment_id=experiment.experiment_id, run_name=run_name
                    )
                    mlflow.tensorflow.autolog(
                        log_every_n_steps=1,
                        log_every_epoch=False,
                        log_models=False,
                        checkpoint=False,
                        checkpoint_save_best_only=False,
                        registered_model_name=f"Model/{run_name}",
                    )
                    optimizer = tf.keras.optimizers.get(self.optimizer).__class__(
                        learning_rate=learning_rate
                    )
                    self.model.compile(optimizer=optimizer, loss_weights=loss_weights)
                    kwargs_progressive = deepcopy(kwargs)
                    kwargs_progressive.pop("epochs", None)
                    history.append(
                        self.model.fit(
                            x, epochs=max_n_progressive_epochs, **kwargs_progressive
                        )
                    )
                    mlflow.end_run()

            # non-progressive training
            run_name = f"Training/{component_order}/{'/'.join(train_components)}"
            mlflow.start_run(experiment_id=experiment.experiment_id, run_name=run_name)
            mlflow.tensorflow.autolog(
                log_every_epoch=True,
                log_models=False,
                checkpoint=False,
                checkpoint_save_best_only=False,
                registered_model_name=f"Model/{run_name}",
            )
            loss_weights, max_n_progressive_epochs = (
                self.setup_component_and_loss_weights(
                    train_components=train_components,
                    n_batches=n_batches,
                    initial_iteration=None,
                )
            )

            optimizer = tf.keras.optimizers.get(self.optimizer).__class__(
                learning_rate=learning_rate
            )
            self.model.compile(optimizer=optimizer, loss_weights=loss_weights)
            callbacks = deepcopy(kwargs.get("callbacks", []))
            if self.early_stopping:
                callbacks.append(
                    tf.keras.callbacks.EarlyStopping(
                        monitor="loss",
                        min_delta=5,
                        patience=max(10, int(kwargs.get("epochs", 1) / 20)),
                        restore_best_weights=True,
                        verbose=1,
                    )
                )
            callbacks.append(
                PeriodicTSNECallback(
                    mdata=self.mdata,
                    component=train_components[0],
                    outdir=os.path.join(self.model.name, "tsne_snapshots"),
                    batch_size=self.batch_size,
                    every=100,
                    batch_effect_colnames=getattr(self, "batch_effect_colnames", None),
                    distribution_names=getattr(self, "distribution_names", None),
                    )
                )
            history.append(self.model.fit(x, callbacks=callbacks, **kwargs))
            mlflow.end_run()

        return history

    def setup_component_and_loss_weights(
        self,
        train_components: List[str],
        n_batches=int,
        initial_iteration: Optional[int] = None,
    ) -> Tuple[Any]:
        """Setup the trainable attributes for each component and the
        alpha (with current_iteration and total_iterations) in
        progressive scaler and set the loss weights properly (for
        components that take more than one modality).

        Parameters
        ----------
        train_components : List[str]
            list of component names for components that need to be
            trained in this sequential step.

        n_batches: int
            number of batches needed to processed in one epoch.

        initial_iteration : Optional[int], optional
            initial iteration for progressive scaler when the training
            starts. The progressive training will be turn off if
            provided with None. Defaults to None.

        Returns
        -------
        Tuple[Any]
            The first element in the tuple is the loss weights, the
            second element is the maximum number of progressive epochs.
        """
        loss_weights = dict()
        max_n_progressive_epochs = 0
        for component_config in self.component_configs:
            component_name = component_config.get("name")
            component = self.model.components.get(component_name)
            if component_name in train_components:
                component.trainable = True
                weight_scale = 1.0

                n_progressive_epochs = float(
                    component_config.get(
                        Constants.CONFIG_FIELD_COMPONENT_N_PROGRESSIVE_EPOCHS
                    )
                )
                progressive_iterations = n_batches * n_progressive_epochs
                max_n_progressive_epochs = max(
                    max_n_progressive_epochs, n_progressive_epochs
                )

                # For non-progressive training
                if initial_iteration is None or initial_iteration <= 0:
                    initial_iteration = progressive_iterations

                component.set_progressive_scaler_iteration(
                    current_iteration=initial_iteration,
                    total_iterations=progressive_iterations,
                )
            else:
                component.trainable = False
                weight_scale = 0.0
                component.set_progressive_scaler_iteration(
                    current_iteration=1.0, total_iterations=1.0
                )

            loss_weights.setdefault(
                f"{component_name}_{Constants.MODEL_LOSS_KL_POSTFIX}",
                1.0 * weight_scale,
            )
            for modality_name in component_config.get(
                Constants.CONFIG_FIELD_COMPONENT_MODALITY_NAMES
            ):
                loss_weights.setdefault(
                    f"{component_name}_{modality_name}_{Constants.MODEL_LOSS_DATA_POSTFIX}",
                    self.modality_weight.get(component_name).get(modality_name)
                    * weight_scale,
                )

        return loss_weights, int(max_n_progressive_epochs)
