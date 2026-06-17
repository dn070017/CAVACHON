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
        mdata: mu.MuData,
        component: str,
        outdir: str,
        batch_size=int,
        batch_effect_colnames: Optional[Mapping[str, List[str]]] = None,
        distribution_names: Optional[Mapping[str, str]] = None,
    ):
        super().__init__()
        self.mdata = mdata
        self.component = component
        self.batch_effect_colnames = batch_effect_colnames
        self.distribution_names = distribution_names
        self.batch_size = batch_size
        self.output_dir = outdir
        os.makedirs(self.output_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        # EDIT HERE TO SPECIFY WHICH EPOCHS TO SAVE:
        save_epochs = set()  # Empty set = never save (disabled)
        # save_epochs = {0, 349, 699, 840, 1199}  # Example: save at these epochs

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


class AnnealingCallback(tf.keras.callbacks.Callback):
    """Unified annealing callback for both parent-component annealing
    and intra-component KL annealing (standard_kl → GMM).

    The callback is driven by a schedule(epoch) function that returns
    a dict mapping loss-name substring patterns to target weights.
    For each loss, the longest matching pattern determines its weight
    (so ``child1_vanilla_kl_divergence`` wins over ``child1``).

    K-means initialization can be triggered at a specific epoch.
    """

    def __init__(
        self,
        schedule,
        kmeans_epoch=None,
        scheduler=None,
        component_name=None,
    ):
        super().__init__()
        self.schedule = schedule
        self.kmeans_epoch = kmeans_epoch
        self.scheduler = scheduler
        self.component_name = component_name
        self._kmeans_done = False

    def on_epoch_begin(self, epoch, logs=None):
        targets = self.schedule(epoch)
        for loss_name, loss_fn in self.model.loss.items():
            if not hasattr(loss_fn, "weight") or not hasattr(
                loss_fn.weight, "assign"
            ):
                continue
            best_match = None
            best_len = 0
            for pattern, weight in targets.items():
                if pattern in loss_name and len(pattern) > best_len:
                    best_match = weight
                    best_len = len(pattern)
            if best_match is not None:
                loss_fn.weight.assign(tf.cast(best_match, tf.float32))

        if (
            self.kmeans_epoch is not None
            and epoch == self.kmeans_epoch
            and not self._kmeans_done
        ):
            print(f"\n{'=' * 70}")
            print(
                f"K-MEANS INITIALIZATION AT EPOCH {epoch} - "
                f"{self.component_name}"
            )
            print(f"{'=' * 70}\n")
            self.model.trainable = False
            self.scheduler.initialize_gmm_priors_with_kmeans(
                component_name=self.component_name,
                seed=42,
                add_noise=True,
                noise_std=0.1,
            )
            self.model.trainable = True
            self._kmeans_done = True


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

    def _get_parent_component(self, child_name: str) -> Optional[str]:
        """Get parent component name for the child."""
        for component_config in self.component_configs:
            if component_config.get("name") == child_name:
                conditioned_on_z_hat = component_config.get(
                    Constants.CONFIG_FIELD_COMPONENT_CONDITION_Z_HAT, []
                )
                if len(conditioned_on_z_hat) > 0:
                    return conditioned_on_z_hat
        return []

    # ==================================================================
    # Public API
    # ==================================================================

    def fit(
        self,
        x: tf.data.Dataset,
        enable_kl_annealing: bool = True,
        kl_annealing_ratios: Tuple[float, float, float] = (0.5, 0.2, 0.3),
        **kwargs,
    ) -> List[tf.keras.callbacks.History]:
        """Fit model with multi-phase hierarchical training.

        Compiles the model once and uses Variable-backed loss weights
        so callbacks can adjust weights at runtime without recompilation.

        Parameters
        ----------
        x: tf.data.Dataset
            input dataset created by DataLoader.

        enable_kl_annealing: bool, optional
            whether to enable intra-component KL annealing
            (standard_kl → GMM). Defaults to True.

        kl_annealing_ratios: Tuple[float, float, float], optional
            ratios for the three sub-phases within progressive/training
            epochs when KL annealing is enabled:
            (standard_kl_only, annealing, gmm_only). Defaults to (0.5, 0.2, 0.3).

        **kwargs: Mapping[str, Any]
            additional arguments passed to self.model.fit.

        Returns
        -------
        List[tf.keras.callbacks.History]
            history of model.fit in each step.
        """
        n_batches = len(x)
        history = []
        max_n_epochs = kwargs.get("epochs", 100)

        self._compile_model(self.learning_rate)
        experiment = self._mlflow_experiment()
        is_single_component = len(self.training_order) == 1

        for component_order, train_components in enumerate(
            self.training_order
        ):
            component_name = train_components[0]
            self.setup_component_and_loss_weights(
                train_components, n_batches
            )

            # --- Parent annealing ---
            if self.run_progressive_training.get(component_name):
                parent_names = self._get_parent_component(component_name)
                if parent_names:
                    n_prog_epochs = self._get_progressive_epochs(
                        component_name
                    )
                    if n_prog_epochs > 0:
                        self.setup_component_and_loss_weights(
                            parent_names + [component_name], n_batches
                        )
                        self._run_parent_annealing_phase(
                            component_name=component_name,
                            component_order=component_order,
                            parent_names=parent_names,
                            n_prog_epochs=n_prog_epochs,
                            x=x,
                            history=history,
                            experiment=experiment,
                            enable_kl_annealing=enable_kl_annealing,
                            kl_annealing_ratios=kl_annealing_ratios,
                            kwargs=kwargs,
                        )

            # --- Regular training ---
            self.setup_component_and_loss_weights(
                train_components, n_batches
            )
            self._run_regular_training_phase(
                component_name=component_name,
                component_order=component_order,
                x=x,
                history=history,
                experiment=experiment,
                enable_kl_annealing=enable_kl_annealing,
                kl_annealing_ratios=kl_annealing_ratios,
                max_n_epochs=max_n_epochs,
                is_single_component=is_single_component,
                kwargs=kwargs,
            )

        return history

    # ==================================================================
    # Phase runners
    # ==================================================================

    def _run_parent_annealing_phase(
        self,
        component_name,
        component_order,
        parent_names,
        n_prog_epochs,
        x,
        history,
        experiment,
        enable_kl_annealing,
        kl_annealing_ratios,
        kwargs,
    ):
        """Run the parent→child annealing phase (progressive epochs)."""
        self._print_phase_header(
            "PARENT ANNEALING",
            f"Parents: {', '.join(parent_names)} → Child: {component_name}",
            n_prog_epochs,
            "standard_kl (child, β=3.0) + GMM (parent, β=1.0)"
            if enable_kl_annealing
            else "GMM only (both, β=1.0)",
        )

        run_name = (
            f"Training/{component_order}/ParentAnnealing/"
            f"{'-'.join(parent_names)}_to_{component_name}"
        )
        self._mlflow_start_run(run_name, experiment)

        kwargs_prog = deepcopy(kwargs)
        kwargs_prog.pop("epochs", None)
        callbacks_prog = deepcopy(kwargs.get("callbacks", []))

        schedule, kmeans_epoch = self._make_parent_annealing_schedule(
            parent_names=parent_names,
            component_name=component_name,
            n_prog_epochs=n_prog_epochs,
            enable_kl_annealing=enable_kl_annealing,
            kl_annealing_ratios=kl_annealing_ratios,
        )
        callbacks_prog.append(
            AnnealingCallback(
                schedule=schedule,
                kmeans_epoch=kmeans_epoch,
                scheduler=self,
                component_name=component_name,
            )
        )

        history.append(
            self.model.fit(
                x,
                epochs=n_prog_epochs,
                callbacks=callbacks_prog,
                **kwargs_prog,
            )
        )

        mlflow.end_run()

        # Freeze parents and zero their weights
        for pn in parent_names:
            self.model.components[pn].trainable = False
            print(f"  Frozen parent: {pn}")
        self._zero_component_variables(parent_names)

    def _run_regular_training_phase(
        self,
        component_name,
        component_order,
        x,
        history,
        experiment,
        enable_kl_annealing,
        kl_annealing_ratios,
        max_n_epochs,
        is_single_component,
        kwargs,
    ):
        """Run the final training phase for a single component."""
        is_child = self.run_progressive_training.get(component_name)
        has_parents = is_child and bool(
            self._get_parent_component(component_name)
        )
        do_kl_annealing = enable_kl_annealing and not has_parents

        # Build phase description
        if is_child:
            stage = "CHILD" if do_kl_annealing else "CHILD GMM"
        else:
            stage = "PARENT" if do_kl_annealing else "PARENT GMM"
        kl_mode = (
            "KL ANNEALING" if do_kl_annealing else "TRAINING"
        )
        self._print_phase_header(
            f"{stage} {kl_mode}",
            f"Component: {component_name}",
            max_n_epochs,
        )

        run_name = (
            f"Training/{component_order}/KLAnnealing/{component_name}"
            if do_kl_annealing
            else f"Training/{component_order}/{component_name}"
        )
        self._mlflow_start_run(run_name, experiment)

        callbacks = deepcopy(kwargs.get("callbacks", []))

        if do_kl_annealing:
            schedule = self._make_final_kl_schedule(
                component_name, max_n_epochs, kl_annealing_ratios
            )
            callbacks.append(
                AnnealingCallback(
                    schedule=schedule,
                    kmeans_epoch=int(
                        max_n_epochs
                        * (kl_annealing_ratios[0] + kl_annealing_ratios[1])
                    ),
                    scheduler=self,
                    component_name=component_name,
                )
            )
        else:
            # No KL annealing, but run k-means at epoch 0
            callbacks.append(
                AnnealingCallback(
                    schedule=lambda epoch: {},
                    kmeans_epoch=0,
                    scheduler=self,
                    component_name=component_name,
                )
            )

        callbacks.extend(
            self._common_callbacks(
                kwargs, is_single_component, component_name
            )
        )

        kwargs_copy = deepcopy(kwargs)
        kwargs_copy.pop("epochs", None)
        history.append(
            self.model.fit(
                x,
                epochs=max_n_epochs,
                callbacks=callbacks,
                **kwargs_copy,
            )
        )

        mlflow.end_run()

    # ==================================================================
    # Schedule builders
    # ==================================================================

    def _make_parent_annealing_schedule(
        self,
        parent_names,
        component_name,
        n_prog_epochs,
        enable_kl_annealing,
        kl_annealing_ratios,
    ):
        """Return (schedule_fn, kmeans_epoch) for the parent annealing phase."""
        if not enable_kl_annealing:
            def schedule(epoch):
                p = epoch / n_prog_epochs
                result = {}
                for pn in parent_names:
                    result[pn] = 1.0 - p
                result[component_name] = p
                return result
            return schedule, None

        standard_kl_end = int(n_prog_epochs * kl_annealing_ratios[0])
        gmm_kl_start = int(
            n_prog_epochs
            * (kl_annealing_ratios[0] + kl_annealing_ratios[1])
        )

        def schedule(epoch):
            p = epoch / n_prog_epochs
            result = {}
            for pn in parent_names:
                result[pn] = 1.0 - p
            result[component_name] = p
            if epoch < standard_kl_end:
                result[f"{component_name}_vanilla_kl_divergence"] = 3.0 * p
                result[f"{component_name}_gmm_kl_divergence"] = 0.0
            elif epoch < gmm_kl_start:
                t = (epoch - standard_kl_end) / (
                    gmm_kl_start - standard_kl_end
                )
                result[f"{component_name}_vanilla_kl_divergence"] = (
                    3.0 * (1.0 - t) * p
                )
                result[f"{component_name}_gmm_kl_divergence"] = (
                    1.0 * t * p
                )
            else:
                result[f"{component_name}_vanilla_kl_divergence"] = 0.0
                result[f"{component_name}_gmm_kl_divergence"] = 1.0 * p
            return result

        return schedule, gmm_kl_start

    def _make_final_kl_schedule(
        self, component_name, total_epochs, kl_annealing_ratios
    ):
        """Return a schedule_fn for the final-phase KL annealing."""
        standard_kl_end = int(total_epochs * kl_annealing_ratios[0])
        gmm_kl_start = int(
            total_epochs
            * (kl_annealing_ratios[0] + kl_annealing_ratios[1])
        )

        def schedule(epoch):
            if epoch < standard_kl_end:
                return {
                    f"{component_name}_vanilla_kl_divergence": 3.0,
                    f"{component_name}_gmm_kl_divergence": 0.0,
                }
            elif epoch < gmm_kl_start:
                t = (epoch - standard_kl_end) / (
                    gmm_kl_start - standard_kl_end
                )
                return {
                    f"{component_name}_vanilla_kl_divergence": 3.0
                    * (1.0 - t),
                    f"{component_name}_gmm_kl_divergence": 1.0 * t,
                }
            else:
                return {
                    f"{component_name}_vanilla_kl_divergence": 0.0,
                    f"{component_name}_gmm_kl_divergence": 1.0,
                }

        return schedule

    # ==================================================================
    # Utilities
    # ==================================================================

    def _compile_model(self, learning_rate):
        """Compile the model once with all loss-backing Variables."""
        all_components = [c.get("name") for c in self.component_configs]
        optimizer = tf.keras.optimizers.get(self.optimizer).__class__(
            learning_rate=learning_rate
        )
        self.model.compile(
            vanilla_kl_weights={c: 0.0 for c in all_components},
            gmm_kl_weights={c: 1.0 for c in all_components},
            optimizer=optimizer,
        )

    def _mlflow_experiment(self):
        """Set up MLflow experiment and return the experiment object."""
        experiment_name = self.model.name
        mlflow.set_experiment(experiment_name)
        return mlflow.get_experiment_by_name(experiment_name)

    def _mlflow_start_run(self, run_name, experiment):
        """Start an MLflow run with TensorFlow autologging."""
        mlflow.start_run(
            experiment_id=experiment.experiment_id, run_name=run_name
        )
        mlflow.tensorflow.autolog(
            log_every_epoch=True,
            log_models=False,
            checkpoint=False,
        )

    def _get_progressive_epochs(self, component_name):
        """Get progressive epochs from component config."""
        for c in self.component_configs:
            if c.get("name") == component_name:
                return c.get(
                    Constants.CONFIG_FIELD_COMPONENT_N_PROGRESSIVE_EPOCHS, 0
                )
        return 0

    @staticmethod
    def _print_phase_header(title, subtitle, epochs, kl_mode=None):
        """Print a formatted phase header."""
        print(f"\n{'═' * 70}")
        print(title)
        print(f"  {subtitle}")
        print(f"  Epochs: {epochs}")
        if kl_mode:
            print(f"  KL mode: {kl_mode}")
        print(f"{'═' * 70}\n")

    def _common_callbacks(
        self, kwargs, is_single_component, component_name
    ):
        """Return common callbacks: PeriodicTSNE + EarlyStopping."""
        callbacks = []
        if is_single_component and self.output_dir:
            callbacks.append(
                PeriodicTSNECallback(
                    mdata=self.mdata,
                    component=component_name,
                    outdir=os.path.join(self.output_dir, "snapshots"),
                    batch_size=self.batch_size,
                    batch_effect_colnames=self.batch_effect_colnames,
                    distribution_names=self.distribution_names,
                )
            )
        if self.early_stopping:
            callbacks.append(
                tf.keras.callbacks.EarlyStopping(
                    monitor="loss",
                    min_delta=5,
                    patience=max(
                        10, int(kwargs.get("epochs", 1) / 20)
                    ),
                    restore_best_weights=True,
                    verbose=1,
                )
            )
        return callbacks

    def _zero_component_variables(
        self, component_names: List[str]
    ) -> None:
        """Set all loss-weight Variables for the given components to 0."""
        for comp_name in component_names:
            for attr in (
                "_gmm_kl_weights",
                "_vanilla_kl_weights",
            ):
                d = getattr(self.model, attr, {})
                if comp_name in d:
                    d[comp_name].assign(0.0)
            for var in self.model._data_loss_weights.get(
                comp_name, {}
            ).values():
                var.assign(0.0)

    def setup_component_and_loss_weights(
        self,
        train_components: List[str],
        n_batches: int,
    ) -> None:
        """Set trainable flags, progressive scaler state, and initial
        loss weights (via Variable assignment — no loss_weights dict).

        Parameters
        ----------
        train_components : List[str]
            component names to enable for training.
        n_batches: int
            number of batches per epoch (for progressive scaler).
        """
        for component_config in self.component_configs:
            comp_name = component_config.get("name")
            component = self.model.components.get(comp_name)

            if comp_name in train_components:
                component.trainable = True
                weight_scale = 1.0

                n_prog_epochs = float(
                    component_config.get(
                        Constants.CONFIG_FIELD_COMPONENT_N_PROGRESSIVE_EPOCHS,
                        0,
                    )
                )
                progressive_iterations = n_batches * n_prog_epochs
                component.set_progressive_scaler_iteration(
                    current_iteration=progressive_iterations,
                    total_iterations=progressive_iterations,
                )
            else:
                component.trainable = False
                weight_scale = 0.0
                component.set_progressive_scaler_iteration(
                    current_iteration=1.0, total_iterations=1.0
                )

            # KL Variables
            if comp_name in getattr(
                self.model, "_gmm_kl_weights", {}
            ):
                self.model._gmm_kl_weights[comp_name].assign(
                    1.0 * weight_scale
                )
            if comp_name in getattr(
                self.model, "_vanilla_kl_weights", {}
            ):
                self.model._vanilla_kl_weights[comp_name].assign(0.0)

            # Data-loss Variables (scaled by modality weight)
            for mod_name, var in self.model._data_loss_weights.get(
                comp_name, {}
            ).items():
                mod_w = self.modality_weight.get(comp_name, {}).get(
                    mod_name, 1.0
                )
                var.assign(mod_w * weight_scale)
