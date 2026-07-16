import itertools
import os
from collections import defaultdict
from copy import deepcopy
from typing import Any, List, Mapping, Optional, Tuple, Union

import mlflow
import numpy as np
import tensorflow as tf

from cavachon.callbacks import (
    AnnealingCallback,
    EarlyStoppingCallback,
    OptimizerStateCallback,
    PeriodicTSNECallback,
    VerboseCallback,
)
from cavachon.config.models.training_config import EarlyStoppingConfig
from cavachon.environment.constants import Constants
from cavachon.layers.progressive_scaler import ProgressiveScaler


class SequentialTrainingScheduler:
    """SequentialTrainingScheduler

    Training scheduler that sets the loss weight and stop the gradient
    of trained components sequentially during training process.

    Attributes
    ----------
    model : tf.keras.Model
        input model that needs to be trained.

    component_configs: List[ComponentConfig]
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
        early_stopping: Union[bool, EarlyStoppingConfig] = True,
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

        early_stopping: Union[bool, EarlyStoppingConfig], optional
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
            component_name = component_config.name
            conditioned_on_z = component_config.conditioned_on_z
            conditioned_on_z_hat = component_config.conditioned_on_z_hat
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
            component_name = component_config.name
            modality_weight = dict()
            if not constant:
                n_vars = component_config.n_vars
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
        ci = self.component_order.index(component_name)
        color = VerboseCallback._COLORS[ci % len(VerboseCallback._COLORS)]
        print(
            f"\033[91mPerform GMM K-means Initialization for "
            f"{color}{VerboseCallback._BOLD}[{component_name}]\033[91m."
            f"{VerboseCallback._RESET}"
        )

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

        cluster_counts = np.zeros(n_clusters, dtype=np.int32)
        for k in range(n_clusters):
            cluster_counts[k] = np.sum(assignments_np == k)
        # for k in range(n_clusters):
        #     pct = 100.0 * cluster_counts[k] / n_samples
        #     print(f"    cluster {k}: {cluster_counts[k]} samples ({pct:.1f}%)")

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
            # print(f"  Added Gaussian noise (std={noise_std}) to centers")

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
            if component_config.name == child_name:
                conditioned_on_z_hat = component_config.conditioned_on_z_hat
                if len(conditioned_on_z_hat) > 0:
                    return conditioned_on_z_hat
        return []

    # ==================================================================
    # Public API
    # ==================================================================

    def fit(
        self,
        x: tf.data.Dataset,
        **kwargs,
    ) -> List[tf.keras.callbacks.History]:
        """Fit model with multi-phase hierarchical training.

        Compiles the model once and uses Variable-backed loss weights
        so callbacks can adjust weights at runtime without recompilation.

        Per-component ``n_kl_annealing_epochs``, ``enable_kmeans_init``,
        and ``kl_annealing_ratio`` are read from
        ``self.component_configs``.

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
        history = []
        fallback_epochs = kwargs.get("epochs", 100)

        self._compile_model(self.learning_rate)
        experiment = self._mlflow_experiment()
        is_single_component = len(self.training_order) == 1

        # Pre-compute total cumulative epochs across all phases
        cumulative_total = 0
        for tc in self.training_order:
            cn = tc[0]
            comp_max_epochs = self._get_max_regular_training_epochs(
                cn, fallback_epochs
            )
            is_child = self.run_progressive_training.get(cn)
            has_parents = is_child and bool(self._get_parent_component(cn))
            if has_parents:
                cumulative_total += self._get_parent_annealing_epochs(cn)
                kl_ep = self._get_kl_annealing_epochs(cn)
                if kl_ep > 0:
                    cumulative_total += kl_ep
            elif (kl_ep := self._get_kl_annealing_epochs(cn)) > 0:
                cumulative_total += kl_ep
            cumulative_total += comp_max_epochs

        cumulative_offset = 0
        phase_number = 1
        self.component_order = [
            c[0] for c in self.training_order
        ]

        for component_order, train_components in enumerate(
            self.training_order
        ):
            component_name = train_components[0]
            self.setup_component_and_loss_weights(
                train_components
            )
            self._compile_model(self.learning_rate)  # retrace with current trainable_variables

            # --- Parent annealing ---
            if self.run_progressive_training.get(component_name):
                parent_names = self._get_parent_component(component_name)
                if parent_names:
                    n_prog_epochs = self._get_parent_annealing_epochs(
                        component_name
                    )
                    if n_prog_epochs > 0:
                        # Parents stay frozen; only set their data weights to 1.0
                        # so the schedule can fade them 1→0 during annealing.
                        for pn in parent_names:
                            self._set_component_weights(
                                pn, data_scale=1.0, gmm_kl=1.0, standard_kl=0.0,
                            )
                        before = len(history)
                        comp_kl_epochs = self._get_kl_annealing_epochs(
                            component_name
                        )
                        comp_enable_kmeans_init = self._get_enable_kmeans_init(
                            component_name
                        )
                        comp_kl_annealing_ratio = self._get_kl_annealing_ratios(
                            component_name
                        )
                        self._run_parent_annealing_phase(
                            component_name=component_name,
                            component_order=component_order,
                            parent_names=parent_names,
                            n_prog_epochs=n_prog_epochs,
                            n_batches=n_batches,
                            x=x.take(n_batches * n_prog_epochs),
                            history=history,
                            experiment=experiment,
                            kl_annealing_enabled=comp_kl_epochs > 0,
                            kl_annealing_ratios=comp_kl_annealing_ratio,
                            enable_kmeans_init=comp_enable_kmeans_init,
                            cumulative_offset=cumulative_offset,
                            cumulative_total=cumulative_total,
                            phase_number=phase_number,
                            kwargs=kwargs,
                        )
                        actual = len(history[-1].epoch) if len(history) > before else n_prog_epochs
                        cumulative_total -= (n_prog_epochs - actual)
                        cumulative_offset += actual
                        phase_number += 1

            # --- KL annealing (when enabled for this component) ---
            comp_kl_epochs = self._get_kl_annealing_epochs(component_name)
            if comp_kl_epochs > 0:
                comp_enable_kmeans_init = self._get_enable_kmeans_init(
                    component_name
                )
                comp_kl_annealing_ratio = self._get_kl_annealing_ratios(
                    component_name
                )
                self.setup_component_and_loss_weights(
                    train_components
                )
                self._compile_model(self.learning_rate)
                before = len(history)
                self._run_kl_annealing_phase(
                    component_name=component_name,
                    component_order=component_order,
                    x=x.take(n_batches * comp_kl_epochs),
                    history=history,
                    experiment=experiment,
                    n_epochs=comp_kl_epochs,
                    kl_annealing_ratios=comp_kl_annealing_ratio,
                    enable_kmeans_init=comp_enable_kmeans_init,
                    cumulative_offset=cumulative_offset,
                    cumulative_total=cumulative_total,
                    phase_number=phase_number,
                    kwargs=kwargs,
                )
                actual = len(history[-1].epoch) if len(history) > before else comp_kl_epochs
                cumulative_total -= (comp_kl_epochs - actual)
                cumulative_offset += actual
                phase_number += 1

            # --- Regular GMM training ---
            self.setup_component_and_loss_weights(
                train_components
            )
            self._compile_model(self.learning_rate)
            before = len(history)
            kmeans_initialized_in_kl_phase = comp_kl_epochs > 0
            comp_max_epochs = self._get_max_regular_training_epochs(
                component_name, fallback_epochs
            )
            self._run_gmm_training_phase(
                component_name=component_name,
                component_order=component_order,
                x=x.take(n_batches * comp_max_epochs),
                history=history,
                experiment=experiment,
                n_epochs=comp_max_epochs,
                is_single_component=is_single_component,
                enable_kmeans_init=self._get_enable_kmeans_init(component_name)
                and not kmeans_initialized_in_kl_phase,
                cumulative_offset=cumulative_offset,
                cumulative_total=cumulative_total,
                phase_number=phase_number,
                kwargs=kwargs,
            )
            actual = len(history[-1].epoch) if len(history) > before else comp_max_epochs
            cumulative_total -= (comp_max_epochs - actual)
            cumulative_offset += actual
            phase_number += 1

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
        n_batches,
        x,
        history,
        experiment,
        kl_annealing_enabled,
        kl_annealing_ratios,
        enable_kmeans_init,
        cumulative_offset,
        cumulative_total,
        phase_number,
        kwargs,
    ):
        """Run the parent→child annealing phase (progressive epochs).

        Activates the progressive scaler so alpha fades 0→1 over the
        phase, then pins it to 1.0 afterwards.  Freezes parent
        components and zeros their loss weights on completion.
        """
        run_name = (
            f"Training/{component_order}/ParentAnnealing/"
            f"{'-'.join(parent_names)}_to_{component_name}"
        )
        self._mlflow_start_run(run_name, experiment)

        self._set_component_progressive(
            component_name, active=True,
            n_batches=n_batches, n_epochs=n_prog_epochs,
        )

        kwargs_prog = deepcopy(kwargs)
        kwargs_prog.pop("epochs", None)
        callbacks_prog = deepcopy(kwargs.get("callbacks", []))

        schedule, kmeans_epoch = self._make_parent_annealing_schedule(
            parent_names=parent_names,
            component_name=component_name,
            n_prog_epochs=n_prog_epochs,
            kl_annealing_enabled=kl_annealing_enabled,
            kl_annealing_ratios=kl_annealing_ratios,
        )

        callbacks_prog.append(
            VerboseCallback(
                labels=[(pn, 0) for pn in parent_names]
                + [(component_name, 1)],
                phase="Parent Annealing",
                loss_prefixes={
                    **{pn: 0 for pn in parent_names},
                    component_name: 1,
                },
                phase_epochs=n_prog_epochs,
                cumulative_offset=cumulative_offset,
                cumulative_total=cumulative_total,
                phase_number=phase_number,
                component_order=self.component_order,
            )
        )

        callbacks_prog.append(
            AnnealingCallback(
                schedule=schedule,
                kmeans_epoch=kmeans_epoch if enable_kmeans_init else None,
                scheduler=self,
                component_name=component_name,
            )
        )
        callbacks_prog.append(OptimizerStateCallback())

        history.append(
            self.model.fit(
                x,
                epochs=n_prog_epochs,
                callbacks=callbacks_prog,
                verbose=0,
                **kwargs_prog,
            )
        )

        mlflow.end_run()

        self._set_component_progressive(component_name, active=False)

        # Zero parent loss weights (parents were already frozen)
        self._zero_component_variables(parent_names)

    def _run_kl_annealing_phase(
        self,
        component_name,
        component_order,
        x,
        history,
        experiment,
        n_epochs,
        kl_annealing_ratios,
        enable_kmeans_init,
        cumulative_offset,
        cumulative_total,
        phase_number,
        kwargs,
    ):
        """Run KL annealing (standard_kl → GMM) for a root component."""
        run_name = (
            f"Training/{component_order}/KLAnnealing/{component_name}"
        )
        self._mlflow_start_run(run_name, experiment)

        schedule = self._make_final_kl_schedule(
            component_name, n_epochs, kl_annealing_ratios
        )
        gmm_start = int(
            n_epochs
            * (kl_annealing_ratios[0] + kl_annealing_ratios[1])
        )

        callbacks = deepcopy(kwargs.get("callbacks", []))
        callbacks.append(
            VerboseCallback(
                labels=[(component_name, component_order)],
                phase="KL Annealing",
                loss_prefixes={component_name: component_order},
                phase_epochs=n_epochs,
                cumulative_offset=cumulative_offset,
                cumulative_total=cumulative_total,
                phase_number=phase_number,
                component_order=self.component_order,
            )
        )
        callbacks.append(
            AnnealingCallback(
                schedule=schedule,
                kmeans_epoch=gmm_start if enable_kmeans_init else None,
                scheduler=self,
                component_name=component_name,
            )
        )
        callbacks.append(OptimizerStateCallback())

        kwargs_copy = deepcopy(kwargs)
        kwargs_copy.pop("epochs", None)
        history.append(
            self.model.fit(
                x,
                epochs=n_epochs,
                callbacks=callbacks,
                verbose=0,
                **kwargs_copy,
            )
        )

        mlflow.end_run()

    def _run_gmm_training_phase(
        self,
        component_name,
        component_order,
        x,
        history,
        experiment,
        n_epochs,
        is_single_component,
        enable_kmeans_init,
        cumulative_offset,
        cumulative_total,
        phase_number,
        kwargs,
    ):
        """Run regular GMM training for a single component."""
        run_name = f"Training/{component_order}/{component_name}"
        self._mlflow_start_run(run_name, experiment)

        callbacks = deepcopy(kwargs.get("callbacks", []))
        callbacks.append(
            VerboseCallback(
                labels=[(component_name, component_order)],
                phase="Regular Training",
                loss_prefixes={component_name: component_order},
                phase_epochs=n_epochs,
                cumulative_offset=cumulative_offset,
                cumulative_total=cumulative_total,
                phase_number=phase_number,
                component_order=self.component_order,
            )
        )
        callbacks.append(
            AnnealingCallback(
                schedule=lambda epoch: {},
                kmeans_epoch=0 if enable_kmeans_init else None,
                scheduler=self,
                component_name=component_name,
            )
        )
        callbacks.append(OptimizerStateCallback())
        callbacks.extend(
            self._common_callbacks(
                kwargs, is_single_component, component_name,
                cumulative_offset=cumulative_offset,
            )
        )

        kwargs_copy = deepcopy(kwargs)
        kwargs_copy.pop("epochs", None)
        history.append(
            self.model.fit(
                x,
                epochs=n_epochs,
                callbacks=callbacks,
                verbose=0,
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
        kl_annealing_enabled,
        kl_annealing_ratios,
    ):
        """Return (schedule_fn, kmeans_epoch) for the parent annealing phase.

        When *kl_annealing_enabled* is True the child uses only standard
        KL (no GMM KL) — the standard→GMM crossfade happens in the
        separate KL annealing phase.  When False, the child uses GMM KL
        throughout.
        """
        if not kl_annealing_enabled:
            def schedule(epoch):
                p = epoch / n_prog_epochs
                result = {}
                for pn in parent_names:
                    result[pn] = 1.0 - p
                    result[f"{pn}_{Constants.MODEL_LOSS_GMM_KL_POSTFIX}"] = 1.0
                    result[f"{pn}_{Constants.MODEL_LOSS_STANDARD_KL_POSTFIX}"] = 0.0
                result[component_name] = p
                return result
            return schedule, None

        def schedule(epoch):
            p = epoch / n_prog_epochs
            result = {}
            for pn in parent_names:
                result[pn] = 1.0 - p
                result[f"{pn}_{Constants.MODEL_LOSS_GMM_KL_POSTFIX}"] = 1.0
                result[f"{pn}_{Constants.MODEL_LOSS_STANDARD_KL_POSTFIX}"] = 0.0
            result[component_name] = p
            result[f"{component_name}_{Constants.MODEL_LOSS_STANDARD_KL_POSTFIX}"] = 3.0 * p
            result[f"{component_name}_{Constants.MODEL_LOSS_GMM_KL_POSTFIX}"] = 0.0
            return result

        return schedule, None

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
                    f"{component_name}_{Constants.MODEL_LOSS_STANDARD_KL_POSTFIX}": 3.0,
                    f"{component_name}_{Constants.MODEL_LOSS_GMM_KL_POSTFIX}": 0.0,
                }
            elif epoch < gmm_kl_start:
                t = (epoch - standard_kl_end) / (
                    gmm_kl_start - standard_kl_end
                )
                return {
                    f"{component_name}_{Constants.MODEL_LOSS_STANDARD_KL_POSTFIX}": 3.0
                    * (1.0 - t),
                    f"{component_name}_{Constants.MODEL_LOSS_GMM_KL_POSTFIX}": 1.0 * t,
                }
            else:
                return {
                    f"{component_name}_{Constants.MODEL_LOSS_STANDARD_KL_POSTFIX}": 0.0,
                    f"{component_name}_{Constants.MODEL_LOSS_GMM_KL_POSTFIX}": 1.0,
                }

        return schedule

    # ==================================================================
    # Utilities
    # ==================================================================

    def _compile_model(self, learning_rate):
        """Compile the model.  Creates a new optimizer when the trainable
        variable set has changed, otherwise reuses the existing one so
        momentum is preserved within a component's phases.

        ProgressiveScaler state (current_iteration, total_iterations) is
        saved before recompile and restored afterwards so that pinned
        weights (e.g. zeroed parents) survive recompilation.
        """
        all_components = [c.name for c in self.component_configs]
        current_tv = {id(v) for v in self.model.trainable_variables} if hasattr(self.model, 'trainable_variables') else set()
        if getattr(self.model, 'optimizer', None) is not None and \
           getattr(self, '_last_trainable', None) == current_tv:
            optimizer = self.model.optimizer
        else:
            optimizer = tf.keras.optimizers.get(self.optimizer).__class__(
                learning_rate=learning_rate
            )
        self._last_trainable = current_tv

        # Save ProgressiveScaler state before recompile
        saved_std = {}
        saved_gmm = {}
        saved_data = {}
        d = getattr(self.model, '_standard_kl_weights', {})
        for cn in all_components:
            if cn in d and isinstance(d[cn], ProgressiveScaler):
                saved_std[cn] = (
                    float(d[cn].current_iteration),
                    float(d[cn].total_iterations),
                )
        d = getattr(self.model, '_gmm_kl_weights', {})
        for cn in all_components:
            if cn in d and isinstance(d[cn], ProgressiveScaler):
                saved_gmm[cn] = (
                    float(d[cn].current_iteration),
                    float(d[cn].total_iterations),
                )
        d = getattr(self.model, '_data_loss_weights', {})
        for cn, mods in d.items():
            for mod, w in mods.items():
                if isinstance(w, ProgressiveScaler):
                    saved_data.setdefault(cn, {})[mod] = (
                        float(w.current_iteration),
                        float(w.total_iterations),
                    )

        # Read scales (and data weight scales) from existing ProgressiveScalers
        std_w = {}
        gmm_w = {}
        data_w = {}
        for cn in all_components:
            d = getattr(self.model, '_standard_kl_weights', {})
            std_w[cn] = float(d[cn].scale) if cn in d else 3.0
            d = getattr(self.model, '_gmm_kl_weights', {})
            gmm_w[cn] = float(d[cn].scale) if cn in d else 1.0
            d = getattr(self.model, '_data_loss_weights', {})
            for mod, w in d.get(cn, {}).items():
                if isinstance(w, ProgressiveScaler):
                    data_w[f"{cn}_{mod}_{Constants.MODEL_LOSS_DATA_POSTFIX}"] = float(w.scale)

        compile_kwargs = dict(
            standard_kl_weights=std_w,
            gmm_kl_weights=gmm_w,
            optimizer=optimizer,
        )
        if data_w:
            compile_kwargs["loss_weights"] = data_w
        self.model.compile(**compile_kwargs)

        # Restore ProgressiveScaler state after recompile
        for cn, (ci, ti) in saved_std.items():
            w = getattr(self.model, '_standard_kl_weights', {}).get(cn)
            if w is not None and isinstance(w, ProgressiveScaler):
                w.current_iteration.assign(ci)
                w.total_iterations.assign(ti)
        for cn, (ci, ti) in saved_gmm.items():
            w = getattr(self.model, '_gmm_kl_weights', {}).get(cn)
            if w is not None and isinstance(w, ProgressiveScaler):
                w.current_iteration.assign(ci)
                w.total_iterations.assign(ti)
        for cn, mods in saved_data.items():
            for mod, (ci, ti) in mods.items():
                w = getattr(self.model, '_data_loss_weights', {}).get(cn, {}).get(mod)
                if w is not None and isinstance(w, ProgressiveScaler):
                    w.current_iteration.assign(ci)
                    w.total_iterations.assign(ti)

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

    def _get_parent_annealing_epochs(self, component_name):
        """Get parent annealing epochs from component config."""
        for c in self.component_configs:
            if c.name == component_name:
                return c.n_parent_annealing_epochs
        return 0

    def _get_kl_annealing_epochs(self, component_name):
        """Get KL annealing epochs from component config."""
        for c in self.component_configs:
            if c.name == component_name:
                return c.n_kl_annealing_epochs
        return 0

    def _get_enable_kmeans_init(self, component_name):
        """Get enable_kmeans_init flag from component config."""
        for c in self.component_configs:
            if c.name == component_name:
                return c.enable_kmeans_init
        return True

    def _get_max_regular_training_epochs(
        self, component_name: str, fallback_epochs: int
    ) -> int:
        """Get max regular training epochs from component config.

        Falls back to ``fallback_epochs`` when the component does not
        specify its own value.
        """
        for c in self.component_configs:
            if c.name == component_name:
                return c.max_regular_training_epochs or fallback_epochs
        return fallback_epochs

    def _get_kl_annealing_ratios(self, component_name):
        """Get KL annealing ratios from component config."""
        for c in self.component_configs:
            if c.name == component_name:
                return c.kl_annealing_ratio
        return (0.5, 0.2, 0.3)

    def _common_callbacks(
        self, kwargs, is_single_component, component_name,
        cumulative_offset=0,
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
                EarlyStoppingCallback(
                    cumulative_offset=cumulative_offset,
                    monitor="loss",
                    min_delta=5,
                    patience=max(
                        10, int(kwargs.get("epochs", 1) / 20)
                    ),
                    restore_best_weights=True,
                    verbose=0,
                )
            )
        return callbacks

    def _set_component_weights(
        self,
        comp_name: str,
        data_scale: float = 1.0,
        gmm_kl: float = 1.0,
        standard_kl: float = 0.0,
    ) -> None:
        """Assign all loss-weight Variables for one component.

        Parameters
        ----------
        comp_name : str
            Component name.
        data_scale : float
            Scale factor applied to data-loss modality weights.
            Use 0.0 to zero data losses.
        gmm_kl : float
            Value assigned to the GMM KL divergence weight Variable.
        standard_kl : float
            Value assigned to the standard KL divergence weight Variable.
        """
        model = self.model
        if comp_name in getattr(model, '_gmm_kl_weights', {}):
            w = model._gmm_kl_weights[comp_name]
            if isinstance(w, ProgressiveScaler):
                w.pin_to(float(gmm_kl))
            else:
                w.assign(float(gmm_kl))
        if comp_name in getattr(model, '_standard_kl_weights', {}):
            w = model._standard_kl_weights[comp_name]
            if isinstance(w, ProgressiveScaler):
                w.pin_to(float(standard_kl))
            else:
                w.assign(float(standard_kl))
        for mod_name, var in model._data_loss_weights.get(comp_name, {}).items():
            mod_w = self.modality_weight.get(comp_name, {}).get(mod_name, 1.0)
            if isinstance(var, ProgressiveScaler):
                var.pin_to(mod_w * float(data_scale))
            else:
                var.assign(mod_w * float(data_scale))

    def _set_component_progressive(
        self,
        comp_name: str,
        active: bool,
        n_batches: int = 1,
        n_epochs: int = 0,
    ) -> None:
        """Activate or deactivate the progressive scaler for one component.

        Parameters
        ----------
        comp_name : str
            Component name.
        active : bool
            If True, resets the scaler to start from iteration 0.0
            with total = n_batches * n_epochs.  If False, pins the
            scaler at alpha = 1.0.
        n_batches : int
            Batches per epoch (only used when active=True).
        n_epochs : int
            Number of progressive epochs (only used when active=True).
        """
        component = self.model.components[comp_name]
        if active:
            total = float(n_batches) * float(n_epochs)
            component.set_progressive_scaler_iteration(0.0, total)
        else:
            component.set_progressive_scaler_iteration(1.0, 1.0)

    def _zero_component_variables(
        self, component_names: List[str]
    ) -> None:
        """Set all loss-weight Variables for the given components to 0."""
        for comp_name in component_names:
            self._set_component_weights(comp_name, data_scale=0.0, gmm_kl=0.0, standard_kl=0.0)

    def setup_component_and_loss_weights(
        self,
        train_components: List[str],
    ) -> None:
        """Set trainable flags, recursive sublayer freeze, and initial
        loss weights for the given components.

        Progressive scaler is left at alpha = 1.0; explicit activation
        is done by ``_set_component_progressive(active=True)`` in the
        parent-annealing phase runner.

        Parameters
        ----------
        train_components : List[str]
            component names to enable for training.  All others are
            frozen and their loss weights zeroed.
        """
        for component_config in self.component_configs:
            comp_name = component_config.name
            component = self.model.components.get(comp_name)

            should_train = comp_name in train_components
            component.trainable = should_train
            self._set_component_progressive(comp_name, active=False)

            visited = set()
            def _set_trainable(layer):
                if id(layer) in visited:
                    return
                visited.add(id(layer))
                if hasattr(layer, 'trainable'):
                    layer.trainable = should_train
                for sub in getattr(layer, 'submodules', []):
                    _set_trainable(sub)
                for sub in getattr(layer, 'layers', []):
                    _set_trainable(sub)
            _set_trainable(component)

            self._set_component_weights(
                comp_name,
                data_scale=1.0 if should_train else 0.0,
                gmm_kl=1.0 if should_train else 0.0,
                standard_kl=0.0,
            )
