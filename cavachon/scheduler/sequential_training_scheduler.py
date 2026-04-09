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
        save_epochs = {0, 139, 279, 419, 699}
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


class DualKLAnnealingCallback(tf.keras.callbacks.Callback):
    """Crossfade between vanilla and GMM KL during transition phase.

    Vanilla KL: 3.0 → 0.0 (linear decrease)
    GMM KL: 0.0 → 1.0 (linear increase)
    Both losses active simultaneously during transition.
    """

    def __init__(self, total_epochs: int):
        """
        Parameters
        ----------
        total_epochs: int
            Total number of epochs in the transition phase.
            Used to calculate progress (0 to 1).
        """
        super().__init__()
        self.total_epochs = total_epochs

    def on_epoch_begin(self, epoch, logs=None):
        """Update both KL weights at the beginning of each epoch.

        Parameters
        ----------
        epoch: int
            Current epoch number (0-indexed within this phase).
        logs: dict, optional
            Training logs (not used).
        """

        # Calculate progress through transition (0.0 to 1.0)
        progress = epoch / self.total_epochs

        # Linear crossfade
        beta_vanilla = 3.0 * (1.0 - progress)  # Decreases: 3.0 → 0.0
        beta_gmm = 1.0 * progress  # Increases: 0.0 → 1.0

        # Update both weight variables in the model
        self.model._vanilla_kl_weight_var.assign(beta_vanilla)
        self.model._gmm_kl_weight_var.assign(beta_gmm)

        # Print progress
        total_weight = beta_vanilla + beta_gmm
        print(
            f"  [Crossfade] Epoch {epoch}: vanilla={beta_vanilla:.4f}, gmm={beta_gmm:.4f}, sum={total_weight:.4f}"
        )


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

    def _kmeans_plus_plus(
        self, X: tf.Tensor, n_clusters: int, seed: int = None
    ) -> tf.Tensor:
        """K-Means++ seeding using TensorFlow ops.

        Selects n_clusters initial centers from data X using the K-means++
        algorithm, which spreads centers probabilistically to cover all
        natural clusters.

        Parameters
        ----------
        X: tf.Tensor
            Data tensor of shape (n_samples, n_dimensions)
        n_clusters: int
            Number of centers to select
        seed: int, optional
            Random seed for reproducibility

        Returns
        -------
        tf.Tensor
            Centers of shape (n_clusters, n_dimensions)
        """
        n_samples = tf.shape(X)[0]

        # 1. Choose first center uniformly at random
        first_idx = tf.random.uniform(
            [], minval=0, maxval=n_samples, dtype=tf.int32, seed=seed
        )
        centers = [tf.gather(X, first_idx)]
        # 2. Iteratively choose remaining centers
        for i in range(1, n_clusters):
            # Stack current centers: (num_current_centers, n_dims)
            current_centers = tf.stack(centers)
            # Calculate squared distances to nearest center
            # Expand for broadcasting: (n_samples, 1, n_dims) - (1, num_centers, n_dims)
            distances_sq = tf.reduce_sum(
                tf.square(tf.expand_dims(X, 1) - tf.expand_dims(current_centers, 0)),
                axis=2,
            )  # Shape: (n_samples, num_centers)
            # For each point, find distance to NEAREST center
            min_distances_sq = tf.reduce_min(
                distances_sq, axis=1
            )  # Shape: (n_samples,)

            # 3. Probabilistic selection: P(x) ∝ D(x)²
            # Points far from existing centers are more likely to be chosen
            logits = tf.math.log(
                tf.expand_dims(min_distances_sq, 0)
            )  # Shape: (1, n_samples)
            next_idx = tf.random.categorical(logits, num_samples=1, seed=seed)[0, 0]

            centers.append(tf.gather(X, next_idx))
        return tf.stack(centers)  # Shape: (n_clusters, n_dimensions)

    def _compute_cluster_assignments(
        self, X: tf.Tensor, centers: tf.Tensor
    ) -> tf.Tensor:
        """Assign each data point to its nearest center.

        Parameters
        ----------
        X: tf.Tensor
            Data tensor of shape (n_samples, n_dimensions)
        centers: tf.Tensor
            Cluster centers of shape (n_clusters, n_dimensions)

        Returns
        -------
        tf.Tensor
            Cluster assignments of shape (n_samples,)
            Each value is an integer in [0, n_clusters-1]
        """
        # Calculate squared Euclidean distances
        # (n_samples, 1, n_dims) - (1, n_clusters, n_dims)
        distances_sq = tf.reduce_sum(
            tf.square(tf.expand_dims(X, 1) - tf.expand_dims(centers, 0)), axis=2
        )  # Shape: (n_samples, n_clusters)

        # Assign to nearest center
        assignments = tf.argmin(
            distances_sq, axis=1, output_type=tf.int32
        )  # Shape: (n_samples,)

        return assignments

    def initialize_gmm_priors_with_kmeans(
        self,
        component_name: str,
        seed: int = 42,
        add_noise: bool = True,
        noise_std: float = 0.1,
    ):
        """
        Initialize GMM prior means and logits using K-means++ on vanilla latent space.

        This should be called after vanilla phase and before transition phase.
        Uses K-means++ to find cluster centers and initializes:
        - loc_bias (means): at cluster centers
        - logits_bias: based on cluster sizes

        Parameters
        ----------
        component_name: str
            Name of the component (e.g., "RNA")
        seed: int
            Random seed for reproducibility
        add_noise: bool
            Whether to add small noise to centers (prevents identical initialization)
        noise_std: float
            Standard deviation of noise to add to centers
        """
        # 1. Extract latent representations from end of vanilla phase
        self.model.trainable = False
        outputs = self.model.predict(self.mdata, batch_size=self.batch_size, verbose=0)
        z = outputs[f"{component_name}_z"]  # Shape: (n_samples, event_dims)
        z_tensor = tf.constant(z, dtype=tf.float32)

        n_samples = z.shape[0]
        event_dims = z.shape[1]

        # 2. Get number of GMM components
        prior_layer = self.model.components[component_name].z_prior_parameterizer
        n_clusters = prior_layer.n_components

        # 3. Run K-means++ to find cluster centers
        centers = self._kmeans_plus_plus(z_tensor, n_clusters, seed=seed)

        # 4. Compute cluster assignments and sizes
        assignments = self._compute_cluster_assignments(z_tensor, centers)
        assignments_np = assignments.numpy()

        # Count points in each cluster
        cluster_counts = np.zeros(n_clusters, dtype=np.int32)
        for k in range(n_clusters):
            cluster_counts[k] = np.sum(assignments_np == k)
        for k in range(n_clusters):
            pct = 100.0 * cluster_counts[k] / n_samples

        # 5. Optionally add small noise to centers
        if add_noise:
            noise = tf.random.normal(
                shape=centers.shape,
                mean=0.0,
                stddev=noise_std,
                seed=seed,
                dtype=tf.float32,
            )
            centers = centers + noise
            print(f"  Added Gaussian noise (std={noise_std}) to centers")
        else:
            print("  No noise added to centers")

        # Initialize GMM prior parameters
        # 6a. Initialize means (loc_bias)
        for k in range(n_clusters):
            center = centers[k : k + 1, :]  # Shape: (1, event_dims)
            prior_layer.loc_bias[k].assign(center)

        # 6b. Initialize logits based on cluster sizes
        # Convert counts to log-probabilities
        cluster_proportions = cluster_counts / n_samples
        # Avoid log(0) for empty clusters (shouldn't happen with k-means++)
        cluster_proportions = np.maximum(cluster_proportions, 1e-7)
        logits = np.log(cluster_proportions).astype(np.float32)
        logits_tensor = tf.constant(logits.reshape(1, n_clusters), dtype=tf.float32)
        prior_layer.logits_bias.assign(logits_tensor)
        
        # 6c. Initialize stds based on cluster spreads
        # Compute std for each cluster
        cluster_stds = np.zeros((n_clusters, event_dims), dtype=np.float32)
        for k in range(n_clusters):
            points_in_cluster = z[assignments_np == k]
            if len(points_in_cluster) > 1:
                # Compute std per dimension
                cluster_stds[k] = np.std(points_in_cluster, axis=0)
            else:
                # Fallback for empty/tiny clusters
                cluster_stds[k] = 0.5
            # Add minimum threshold to prevent collapse
            # cluster_stds[k] = np.maximum(cluster_stds[k], 0.02) 

        # Convert std to scale_diag_bias value (inverse of softplus)
        # softplus(x) = log(1 + exp(x))
        # Inverse: x = log(exp(std) - 1)
        # But simpler approximation for std > 0.5: x ≈ std
        for k in range(n_clusters):
            std_value = cluster_stds[k]
            # Inverse softplus (approximate)
            # For std > 1: bias ≈ std
            # For std < 1: bias ≈ log(exp(std) - 1)
            bias_value = np.where(
                std_value > 1.0,
                std_value - 0.5,  # Approximation
                np.log(np.exp(std_value) - 1 + 1e-7),  # Exact inverse
            )
            bias_tensor = tf.constant(
                bias_value.reshape(1, event_dims), dtype=tf.float32
            )
            prior_layer.scale_diag_bias[k].assign(bias_tensor)
            
        self.model.trainable = True

    def fit(self, x: tf.data.Dataset, **kwargs) -> List[tf.keras.callbacks.History]:
        """Fit self.model sequentially with three-phase training.
        Phase 1: Vanilla KL only (beta=3.0 constant)
        Phase 2: Transition with crossfade (vanilla 3.0→0.0, GMM 0.0→1.0)
        Phase 3: GMM KL only (beta=1.0 constant)

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

            # Split progressive epochs:
            # 3 PHASES: 50% vanilla / 20% transition / 30% GMM
            vanilla_epochs = int(max_n_progressive_epochs * 0.50)
            transition_epochs = int(max_n_progressive_epochs * 0.20)
            gmm_epochs = max_n_progressive_epochs - vanilla_epochs - transition_epochs

            # Print training plan
            print(f"\n{'=' * 70}")
            print(f"Training Component: {train_components[0]}")
            print(f"Total Progressive Epochs: {max_n_progressive_epochs}")
            print(f"  → Phase 1 - Vanilla Only: {vanilla_epochs} epochs (beta=3.0)") 
            print(
                f"  → Phase 2 - Transition:   {transition_epochs} epochs (crossfade 3.0→0.0 / 0.0→1.0)"
            )
            print(f"  → Phase 3 - GMM Only:     {gmm_epochs} epochs (beta=1.0)")
            print(f"{'=' * 70}\n")

            # PHASE 1: VANILLA KL ONLY
            run_name = f"Training/{component_order}/Progressive/Phase1_VanillaOnly/{'/'.join(train_components)}"
            mlflow.start_run(experiment_id=experiment.experiment_id, run_name=run_name)
            mlflow.tensorflow.autolog(
                log_every_n_steps=None,
                log_every_epoch=True,
                log_models=False,
                checkpoint=False,
                checkpoint_save_best_only=False,
                registered_model_name=f"Model/{run_name}",
            )

            print(f"[PHASE 1: VANILLA ONLY] Starting {vanilla_epochs} epochs...")

            optimizer = tf.keras.optimizers.get(self.optimizer).__class__(
                learning_rate=learning_rate
            )
            self.model.compile(
                use_vanilla_kl=True,  # ← Turn ON vanilla KL,
                use_both_kl=False,
                optimizer=optimizer,
                loss_weights=loss_weights,
            )

            # Set initial weight for vanilla phase
            self.model._vanilla_kl_weight_var.assign(3.0) ##
            self.model._gmm_kl_weight_var.assign(0.0)

            kwargs_progressive = deepcopy(kwargs)
            kwargs_progressive.pop("epochs", None)

            callbacks_phase1 = deepcopy(kwargs.get("callbacks", []))
            callbacks_phase1.append(
                PeriodicTSNECallback(
                    mdata=self.mdata,
                    component=train_components[0],
                    outdir=os.path.join(
                        self.output_dir, "tsne_snapshots_phase1_vanilla"
                    ),
                    batch_size=self.batch_size,
                    every=100,
                    batch_effect_colnames=self.batch_effect_colnames,
                    distribution_names=self.distribution_names,
                )
            )

            history.append(
                self.model.fit(
                    x,
                    epochs=vanilla_epochs,
                    callbacks=callbacks_phase1,
                    **kwargs_progressive,
                )
            )
            print(
                f"[PHASE 1] Completed! Final loss: {history[-1].history['loss'][-1]:.4f}\n"
            )
            mlflow.end_run()

            # Initialize GMM priors using K-means++ on vanilla latent space
            self.initialize_gmm_priors_with_kmeans(
                component_name=train_components[0],
                seed=42,
                add_noise=True,
                noise_std=0.1,
            )

            # PHASE 2: TRANSITION (Both losses active)
            run_name = f"Training/{component_order}/Progressive/Phase2_Transition/{'/'.join(train_components)}"
            mlflow.start_run(experiment_id=experiment.experiment_id, run_name=run_name)
            mlflow.tensorflow.autolog(
                log_every_n_steps=None,
                log_every_epoch=True,
                log_models=False,
                checkpoint=False,
                checkpoint_save_best_only=False,
                registered_model_name=f"Model/{run_name}",
            )
            print(
                f"[PHASE 2: TRANSITION] Starting {transition_epochs} epochs with annealing..."
            )
            optimizer = tf.keras.optimizers.get(self.optimizer).__class__(
                learning_rate=learning_rate
            )
            self.model.compile(
                use_vanilla_kl=False,  # ← Turn OFF vanilla,
                use_both_kl=True,  # ← Both losses active
                optimizer=optimizer,
                loss_weights=loss_weights,
            )

            # Set initial weights for transition
            self.model._vanilla_kl_weight_var.assign(3.0)
            self.model._gmm_kl_weight_var.assign(0.0)

            kwargs_progressive = deepcopy(kwargs)
            kwargs_progressive.pop("epochs", None)

            callbacks_phase2 = deepcopy(kwargs.get("callbacks", []))
            callbacks_phase2.append(
                DualKLAnnealingCallback(total_epochs=transition_epochs)
            )
            callbacks_phase2.append(
                PeriodicTSNECallback(
                    mdata=self.mdata,
                    component=train_components[0],
                    outdir=os.path.join(
                        self.output_dir, "tsne_snapshots_phase2_transition"
                    ),
                    batch_size=self.batch_size,
                    every=100,
                    batch_effect_colnames=self.batch_effect_colnames,
                    distribution_names=self.distribution_names,
                )
            )
            history.append(
                self.model.fit(
                    x,
                    epochs=transition_epochs,
                    callbacks=callbacks_phase2,
                    **kwargs_progressive,
                )
            )
            print(
                f"[GMM KL] Completed! Final loss: {history[-1].history['loss'][-1]:.4f}\n"
            )
            print(
                f"[PHASE 2] Completed! Final loss: {history[-1].history['loss'][-1]:.4f}\n"
            )
            mlflow.end_run()
            
            # ========== test
            # Re-initialize GMM priors after transition phase
            self.initialize_gmm_priors_with_kmeans(
                component_name=train_components[0],
                seed=42,
                add_noise=True,
                noise_std=0.1,
            )
            # =========

            # PHASE 3: GMM KL only
            run_name = f"Training/{component_order}/Progressive/Phase3_GMMOnly/{'/'.join(train_components)}"
            mlflow.start_run(experiment_id=experiment.experiment_id, run_name=run_name)
            mlflow.tensorflow.autolog(
                log_every_n_steps=None,
                log_every_epoch=True,
                log_models=False,
                checkpoint=False,
                checkpoint_save_best_only=False,
                registered_model_name=f"Model/{run_name}",
            )
            print(f"[PHASE 3: GMM ONLY] Starting {gmm_epochs} epochs...")

            optimizer = tf.keras.optimizers.get(self.optimizer).__class__(
                learning_rate=learning_rate * 0.5 # slow down for gmm
            )
            self.model.compile(
                use_vanilla_kl=False,
                use_both_kl=False,
                optimizer=optimizer,
                loss_weights=loss_weights,
            )
            # Set initial weights for GMM phase
            self.model._vanilla_kl_weight_var.assign(0.0)
            self.model._gmm_kl_weight_var.assign(1.0)

            kwargs_progressive = deepcopy(kwargs)
            kwargs_progressive.pop("epochs", None)

            callbacks_phase3 = deepcopy(kwargs.get("callbacks", []))
            callbacks_phase3.append(
                PeriodicTSNECallback(
                    mdata=self.mdata,
                    component=train_components[0],
                    outdir=os.path.join(self.output_dir, "tsne_snapshots_phase3_gmm"),
                    batch_size=self.batch_size,
                    every=100,
                    batch_effect_colnames=self.batch_effect_colnames,
                    distribution_names=self.distribution_names,
                )
            )
            history.append(
                self.model.fit(
                    x,
                    epochs=gmm_epochs,
                    callbacks=callbacks_phase3,
                    **kwargs_progressive,
                )
            )

            print(
                f"[PHASE 3] Completed! Final loss: {history[-1].history['loss'][-1]:.4f}\n"
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
