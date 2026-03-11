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
        batch_size=int,
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
        # below is start from 500
        # if epoch < 499 or ((epoch - 499) % self.every) != 0:
        # if (epoch + 1) % self.every != 0:
        save_epochs = {0, 249, 599, 839}
        if epoch not in save_epochs:
            return

        component = self.component

        # First temporarily freeze the model
        # this prevents layers from updating internal states (like Batch Norm) during the predict call
        original_trainable_state = self.model.trainable
        self.model.trainable = False

        # Then, extract mean latent z
        # By calling predict with verbose=0 and the model's training state as False,
        # Model will return the Mean (μ) and skip the sampling (ϵ) step.
        # This gives the stable "Mean Z"
        outputs = self.model.predict(self.mdata, batch_size=self.batch_size, verbose=0)
        # outputs = self.model.predict(self.mdata, batch_size=self.batch_size)
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
            # use training=False here as well to ensure we get the MEAN Z
            # and don't accidentally update any model weights.
            outs = self.model(batch_data, training=False)
            z = outs[f"{self.component}_z"]
            logpz_y = dist_z_y.log_prob(tf.expand_dims(z, -2))
            logpz = tf.expand_dims(dist_z.log_prob(z), -1)
            logpy_z_parts.append(logpy + logpz_y - logpz)

        logpy_z = np.vstack([x.numpy() for x in logpy_z_parts])

        # restore the model state back
        self.model.trainable = original_trainable_state

        # Save z and logpy_z in the configured results directory
        np.save(
            os.path.join(self.output_dir, f"{epoch + 1}_z.h5"), z_full
        )  # because of zero indexing
        np.save(os.path.join(self.output_dir, f"{epoch + 1}_logpy_z.h5"), logpy_z)
        np.save(
            os.path.join(self.output_dir, f"{epoch + 1}_prior_params.npy"),
            z_prior_parameters.numpy(),
        )


class KLAnnealingCallback(tf.keras.callbacks.Callback):
    """Callback to anneal KL loss weight during training.

    For vanilla phase: decreases weight from start_weight → 0
    For GMM phase: increases weight from 0 → end_weight
    """

    def __init__(
        self,
        loss_name: str,
        start_weight: float,
        end_weight: float,
        total_epochs: int,
        anneal_epochs: int,
        phase: str = "vanilla",  # "vanilla" or "gmm"
    ):
        """
        Parameters
        ----------
        loss_name: str
            Name of the KL loss to modify (e.g., "RNA_kl_divergence")
        start_weight: float
            Starting weight value
        end_weight: float
            Ending weight value
        total_epochs: int
            Total number of epochs in this phase
        anneal_epochs: int
            Number of epochs over which to anneal
        phase: str
            "vanilla" (decrease at end) or "gmm" (increase at start)
        """
        super().__init__()
        self.loss_name = loss_name
        self.start_weight = start_weight
        self.end_weight = end_weight
        self.total_epochs = total_epochs
        self.anneal_epochs = anneal_epochs
        self.phase = phase

        # Calculate when annealing starts/ends
        if phase == "vanilla":
            # Anneal in last 15% of vanilla phase
            self.anneal_start = total_epochs - anneal_epochs
            self.anneal_end = total_epochs
        else:  # gmm
            # Anneal in first 15% of GMM phase
            self.anneal_start = 0
            self.anneal_end = anneal_epochs

    def on_epoch_begin(self, epoch, logs=None):
        """Update KL weight at the beginning of each epoch"""

        # Check if we're in annealing period
        if self.anneal_start <= epoch < self.anneal_end:
            # Linear interpolation
            progress = (epoch - self.anneal_start) / self.anneal_epochs
            current_weight = self.start_weight + progress * (
                self.end_weight - self.start_weight
            )

            print(f"  [Annealing] Epoch {epoch}: KL weight = {current_weight:.4f}")
        elif epoch < self.anneal_start:
            current_weight = self.start_weight
        else:
            current_weight = self.end_weight

        # Update the loss weight
        loss_fn = self.model.loss.get(self.loss_name)
        if loss_fn:
            loss_fn.weight = current_weight


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
        distribution_names: Optional[Mapping[str, str]] = None,
        batch_effect_colnames: Optional[Mapping[str, List[str]]] = None,
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
        self.output_dir = outdir
        self.distribution_names = distribution_names
        self.batch_effect_colnames = batch_effect_colnames

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
            # Set up loss weights for progressive training
            loss_weights, max_n_progressive_epochs = (
                self.setup_component_and_loss_weights(
                    train_components=train_components,
                    n_batches=n_batches,
                    initial_iteration=0.0,
                )
            )

            # Force progressive training even if max_n_progressive_epochs is 0
            if max_n_progressive_epochs == 0:
                component_config = [
                    c
                    for c in self.component_configs
                    if c.get("name") == train_components[0]
                ][0]
                max_n_progressive_epochs = component_config.get(
                    "n_progressive_epochs", 100
                )

            # Split progressive epochs: 50% vanilla KL, 50% GMM KL
            vanilla_progressive_epochs = int(max_n_progressive_epochs * 0.50)
            gmm_progressive_epochs = (
                max_n_progressive_epochs - vanilla_progressive_epochs
            )

            # Calculate annealing epochs (15% of each phase)
            vanilla_anneal_epochs = int(vanilla_progressive_epochs * 0.15)
            gmm_anneal_epochs = int(gmm_progressive_epochs * 0.15)

            # Print training plan
            print(f"\n{'=' * 70}")
            print(f"Training Component: {train_components[0]}")
            print(f"Total Progressive Epochs: {max_n_progressive_epochs}")
            print(f"  → Vanilla KL Phase: {vanilla_progressive_epochs} epochs")
            print(
                f"     - Constant (beta=3.0): {vanilla_progressive_epochs - vanilla_anneal_epochs} epochs"
            )
            print(f"     - Annealing (3.0→0.0): {vanilla_anneal_epochs} epochs")
            print(f"  → GMM KL Phase: {gmm_progressive_epochs} epochs")
            print(f"     - Annealing (0.0→1.0): {gmm_anneal_epochs} epochs")
            print(
                f"     - Constant (beta=1.0): {gmm_progressive_epochs - gmm_anneal_epochs} epochs"
            )
            print(f"{'=' * 70}\n")

            # PHASE 1: VANILLA KL
            run_name = f"Training/{component_order}/Progressive/VanillaKL/{'/'.join(train_components)}"
            mlflow.start_run(experiment_id=experiment.experiment_id, run_name=run_name)
            mlflow.tensorflow.autolog(
                log_every_n_steps=None,  # 1
                log_every_epoch=True,
                log_models=False,
                checkpoint=False,
                checkpoint_save_best_only=False,
                registered_model_name=f"Model/{run_name}",
            )
            print(
                f"[VANILLA KL] Starting training for {vanilla_progressive_epochs} epochs..."
            )

            optimizer = tf.keras.optimizers.get(self.optimizer).__class__(
                learning_rate=learning_rate
            )
            self.model.compile(
                use_vanilla_kl=True,  # ← Turn ON vanilla KL,
                optimizer=optimizer,
                loss_weights=loss_weights,
            )

            kwargs_progressive = deepcopy(kwargs)
            kwargs_progressive.pop("epochs", None)

            callbacks_vanilla = deepcopy(kwargs.get("callbacks", []))
            callbacks_vanilla.append(
                KLAnnealingCallback(
                    loss_name=f"{train_components[0]}_kl_divergence",
                    start_weight=3.0,
                    end_weight=0.0,
                    total_epochs=vanilla_progressive_epochs,
                    anneal_epochs=vanilla_anneal_epochs,
                    phase="vanilla",
                )
            )
            callbacks_vanilla.append(
                PeriodicTSNECallback(
                    mdata=self.mdata,
                    component=train_components[0],
                    outdir=os.path.join(
                        self.output_dir, "tsne_snapshots_vanilla"
                    ),  # ← vanilla folder
                    batch_size=self.batch_size,
                    every=100,
                    batch_effect_colnames=self.batch_effect_colnames,
                    distribution_names=self.distribution_names,
                )
            )

            history.append(
                self.model.fit(
                    x,
                    epochs=vanilla_progressive_epochs,
                    callbacks=callbacks_vanilla,
                    **kwargs_progressive,
                )
            )
            print(
                f"[VANILLA KL] Completed! Final loss: {history[-1].history['loss'][-1]:.4f}\n"
            )
            mlflow.end_run()

            # PHASE 2: GMM KL
            run_name = f"Training/{component_order}/Progressive/GMMKL/{'/'.join(train_components)}"
            mlflow.start_run(experiment_id=experiment.experiment_id, run_name=run_name)
            mlflow.tensorflow.autolog(
                log_every_n_steps=None,  # 1
                log_every_epoch=True,
                log_models=False,
                checkpoint=False,
                checkpoint_save_best_only=False,
                registered_model_name=f"Model/{run_name}",
            )
            print(f"[GMM KL] Starting training for {gmm_progressive_epochs} epochs...")
            optimizer = tf.keras.optimizers.get(self.optimizer).__class__(
                learning_rate=learning_rate
            )
            self.model.compile(
                use_vanilla_kl=False,  # ← Turn OFF vanilla, use GMM
                optimizer=optimizer,
                loss_weights=loss_weights,
            )

            kwargs_progressive = deepcopy(kwargs)
            kwargs_progressive.pop("epochs", None)

            callbacks_gmm = deepcopy(kwargs.get("callbacks", []))
            callbacks_gmm.append(
                KLAnnealingCallback(
                    loss_name=f"{train_components[0]}_kl_divergence",
                    start_weight=0.0,
                    end_weight=1.0,
                    total_epochs=gmm_progressive_epochs,
                    anneal_epochs=gmm_anneal_epochs,
                    phase="gmm",
                )
            )
            callbacks_gmm.append(
                PeriodicTSNECallback(
                    mdata=self.mdata,
                    component=train_components[0],
                    outdir=os.path.join(
                        self.output_dir, "tsne_snapshots_gmm"
                    ),  # ← gmm folder
                    batch_size=self.batch_size,
                    every=100,
                    batch_effect_colnames=self.batch_effect_colnames,
                    distribution_names=self.distribution_names,
                )
            )
            history.append(
                self.model.fit(
                    x,
                    epochs=gmm_progressive_epochs,
                    callbacks=callbacks_gmm,
                    **kwargs_progressive,
                )
            )
            print(
                f"[GMM KL] Completed! Final loss: {history[-1].history['loss'][-1]:.4f}\n"
            )
            mlflow.end_run()

            # non-progressive training
            """
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
                        min_delta=50,  # 5
                        patience=1100,  # max(10, int(kwargs.get("epochs", 1) / 20)),
                        restore_best_weights=True,
                        verbose=1,
                    )
                )
            callbacks.append(
                PeriodicTSNECallback(
                    mdata=self.mdata,
                    component=train_components[0],
                    outdir=os.path.join(self.output_dir, "tsne_snapshots"),
                    batch_size=self.batch_size,
                    every=100,
                    batch_effect_colnames=self.batch_effect_colnames,
                    distribution_names=self.distribution_names,
                )
            )
            history.append(self.model.fit(x, callbacks=callbacks, **kwargs))
            mlflow.end_run()
            """

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
