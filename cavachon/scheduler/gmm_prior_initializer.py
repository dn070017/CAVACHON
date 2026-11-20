from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Literal, Optional, Tuple

import numpy as np
import tensorflow as tf

from cavachon.callbacks import VerboseCallback

Target = Literal["z", "z_hat"]


@dataclass
class GMMPriorEstimate:
    """GMMPriorEstimate

    GMM prior parameters estimated from a latent representation.

    Attributes
    ----------
    locs: tf.Tensor
        cluster centers, with shape (n_clusters, event_dims).

    logits: np.ndarray
        log of the cluster proportions, with shape (1, n_clusters).

    scale_biases: np.ndarray
        biases of the scale_diag (before softplus), with shape
        (n_clusters, event_dims).

    """

    locs: tf.Tensor
    logits: np.ndarray
    scale_biases: np.ndarray


class GMMPriorInitializer:
    """GMMPriorInitializer

    Initializes the GMM prior (means, logits and stds) of a component
    with K-means++ on its latent representation z or z_hat.

    Attributes
    ----------
    model: cavachon.model.Model
        the model whose priors are initialized.

    mdata: muon.MuData
        data used to compute the latent representation.

    batch_size: int
        batch size of the prediction.

    component_order: List[str]
        training order of the components (for the message colors).

    seed: int
        random seed of K-means++ and of the noise.

    add_noise: bool
        whether to add Gaussian noise to the centers.

    noise_std: float
        std of the noise added to the centers.

    """

    # (output key suffix, prior parameterizer attribute) of each target
    TARGETS: Dict[str, Tuple[str, str]] = {
        "z": ("_z", "z_prior_parameterizer"),
        "z_hat": ("_z_hat", "z_hat_prior_parameterizer"),
    }

    def __init__(
        self,
        model: Any,
        mdata: Any,
        batch_size: int,
        component_order: List[str],
        seed: int = 42,
        add_noise: bool = True,
        noise_std: float = 0.1,
    ):
        """Constructor for GMMPriorInitializer.

        Parameters
        ----------
        model: cavachon.model.Model
            the model whose priors are initialized.

        mdata: muon.MuData
            data used to compute the latent representation.

        batch_size: int
            batch size of the prediction.

        component_order: List[str]
            training order of the components (for the message colors).

        seed: int, optional
            random seed of K-means++ and of the noise. Defaults to 42.

        add_noise: bool, optional
            whether to add Gaussian noise to the centers. Defaults to
            True.

        noise_std: float, optional
            std of the noise added to the centers. Defaults to 0.1.
        """
        self.model = model
        self.mdata = mdata
        self.batch_size = batch_size
        self.component_order = component_order
        self.seed = seed
        self.add_noise = add_noise
        self.noise_std = noise_std

    def initialize(self, component_name: str, target: Target = "z") -> np.ndarray:
        """Initialize the GMM prior of z (or z_hat) of a component with
        K-means++ on that latent representation.

        The model is frozen while the representation is predicted, and
        every trainable flag and alpha is restored afterwards.

        Parameters
        ----------
        component_name: str
            component name.

        target: Literal["z", "z_hat"], optional
            which latent representation and prior to use. Defaults to
            "z".

        Returns
        -------
        np.ndarray
            the clustered latent representation, with shape
            (n_samples, event_dims).
        """
        self._announce(component_name, target)
        with self._frozen_for_prediction():
            z = self._predict_representation(component_name, target)
        prior_layer = self._prior_layer(component_name, target)
        estimate = self.estimate(z, prior_layer.n_components)
        self.apply(prior_layer, estimate)
        return z

    def estimate(self, z: np.ndarray, n_clusters: int) -> GMMPriorEstimate:
        """Estimate the GMM prior parameters of ``n_clusters`` clusters
        from ``z`` with K-means++.

        Parameters
        ----------
        z: np.ndarray
            latent representation, with shape (n_samples, event_dims).

        n_clusters: int
            number of GMM components.

        Returns
        -------
        GMMPriorEstimate
            centers (optionally with noise), log proportions and scale
            biases of the clusters.
        """
        z_tensor = tf.constant(z, dtype=tf.float32)
        n_samples, event_dims = z.shape[0], z.shape[1]

        centers = self.kmeans_plus_plus(z_tensor, n_clusters, seed=self.seed)
        assignments = self.assign_clusters(z_tensor, centers).numpy()
        cluster_counts = np.array(
            [np.sum(assignments == k) for k in range(n_clusters)], dtype=np.int32
        )

        if self.add_noise:
            centers = centers + tf.random.normal(
                shape=centers.shape,
                mean=0.0,
                stddev=self.noise_std,
                seed=self.seed,
                dtype=tf.float32,
            )

        # log of the cluster proportions (empty clusters get 1e-7)
        proportions = np.maximum(cluster_counts / n_samples, 1e-7)
        logits = np.log(proportions).astype(np.float32).reshape(1, n_clusters)

        # per-cluster std, mapped to the scale_diag bias (before softplus)
        scale_biases = np.zeros((n_clusters, event_dims), dtype=np.float32)
        for k in range(n_clusters):
            members = z[assignments == k]
            std = (
                np.std(members, axis=0)
                if len(members) > 1
                else np.full(event_dims, 0.5)
            )
            scale_biases[k] = np.where(
                std > 1.0,
                std - 0.5,  # approximation of the inverse softplus
                np.log(np.exp(std) - 1 + 1e-7),  # exact inverse softplus
            )

        return GMMPriorEstimate(locs=centers, logits=logits, scale_biases=scale_biases)

    @staticmethod
    def apply(prior_layer: Any, estimate: GMMPriorEstimate) -> None:
        """Write the estimated parameters into a mixture prior layer.

        Parameters
        ----------
        prior_layer: MixtureMultivariateNormalDiagParameterizerLayer
            the prior parameterizer (loc_bias, logits_bias and
            scale_diag_bias).

        estimate: GMMPriorEstimate
            the estimated parameters.
        """
        n_clusters, event_dims = estimate.scale_biases.shape
        for k in range(n_clusters):
            prior_layer.loc_bias[k].assign(estimate.locs[k : k + 1, :])
            prior_layer.scale_diag_bias[k].assign(
                tf.constant(
                    estimate.scale_biases[k].reshape(1, event_dims), dtype=tf.float32
                )
            )
        prior_layer.logits_bias.assign(tf.constant(estimate.logits, dtype=tf.float32))

    @staticmethod
    def kmeans_plus_plus(
        X: tf.Tensor, n_clusters: int, seed: Optional[int] = None
    ) -> tf.Tensor:
        """K-Means++ seeding using TensorFlow ops.

        Selects n_clusters initial centers from data X: the first
        uniformly, the next ones with probability proportional to the
        squared distance to the nearest center chosen so far.

        Parameters
        ----------
        X: tf.Tensor
            data, with shape (n_samples, n_dimensions).

        n_clusters: int
            number of centers to select.

        seed: int, optional
            random seed for reproducibility.

        Returns
        -------
        tf.Tensor
            centers, with shape (n_clusters, n_dimensions).
        """
        n_samples = tf.shape(X)[0]
        first_idx = tf.random.uniform(
            [], minval=0, maxval=n_samples, dtype=tf.int32, seed=seed
        )
        centers = [tf.gather(X, first_idx)]
        for _ in range(1, n_clusters):
            current_centers = tf.stack(centers)
            # squared distance to the nearest center: (n_samples, )
            min_distances_sq = tf.reduce_min(
                tf.reduce_sum(
                    tf.square(
                        tf.expand_dims(X, 1) - tf.expand_dims(current_centers, 0)
                    ),
                    axis=2,
                ),
                axis=1,
            )
            # If every point coincides with a center (e.g. a collapsed
            # latent space with fewer distinct points than clusters),
            # all distances are 0, so fall back to choosing uniformly.
            weights = tf.cond(
                tf.reduce_sum(min_distances_sq) > 0,
                lambda: min_distances_sq,
                lambda: tf.ones_like(min_distances_sq),
            )
            logits = tf.math.log(tf.expand_dims(weights, 0))
            next_idx = tf.random.categorical(logits, num_samples=1, seed=seed)[0, 0]
            centers.append(tf.gather(X, next_idx))
        return tf.stack(centers)

    @staticmethod
    def assign_clusters(X: tf.Tensor, centers: tf.Tensor) -> tf.Tensor:
        """Assign each data point to its nearest center.

        Parameters
        ----------
        X: tf.Tensor
            data, with shape (n_samples, n_dimensions).

        centers: tf.Tensor
            cluster centers, with shape (n_clusters, n_dimensions).

        Returns
        -------
        tf.Tensor
            index of the nearest center of each point, with shape
            (n_samples, ) and dtype int32.
        """
        distances_sq = tf.reduce_sum(
            tf.square(tf.expand_dims(X, 1) - tf.expand_dims(centers, 0)), axis=2
        )
        return tf.argmin(distances_sq, axis=1, output_type=tf.int32)

    @contextmanager
    def _frozen_for_prediction(self) -> Iterator[None]:
        """Freeze the model for the prediction, then restore the
        trainable flag of the model, of every component and of every
        nested layer, and the alpha of every hierarchical encoder
        (toggling ``trainable`` pins alpha to 1)."""
        model = self.model
        model_trainable = model.trainable
        layer_flags = [
            (layer, layer.trainable)
            for component in model.components.values()
            for layer in self._nested_layers(component)
        ]
        alphas = {
            name: (
                float(
                    component.hierarchical_encoder.progressive_scaler.current_iteration
                ),
                float(
                    component.hierarchical_encoder.progressive_scaler.total_iterations
                ),
            )
            for name, component in model.components.items()
        }
        model.trainable = False
        try:
            yield
        finally:
            model.trainable = model_trainable
            for layer, trainable in layer_flags:
                layer.trainable = trainable
            for name, (current, total) in alphas.items():
                model.components[name].set_progressive_scaler_iteration(
                    current_iteration=current, total_iterations=total
                )

    @staticmethod
    def _nested_layers(component: Any) -> List[Any]:
        """The component and every layer nested in it, reached through
        ``submodules`` and ``layers`` (outermost first)."""
        visited = set()
        ordered: List[Any] = []

        def visit(current: Any) -> None:
            if id(current) in visited:
                return
            visited.add(id(current))
            if hasattr(current, "trainable"):
                ordered.append(current)
            for sub in getattr(current, "submodules", []):
                visit(sub)
            for sub in getattr(current, "layers", []):
                visit(sub)

        visit(component)
        return ordered

    def _predict_representation(
        self, component_name: str, target: Target
    ) -> np.ndarray:
        """Predict z (or z_hat) of a component on the whole data."""
        suffix, _ = self.TARGETS[target]
        outputs = self.model.predict(self.mdata, batch_size=self.batch_size, verbose=0)
        return outputs[f"{component_name}{suffix}"]

    def _prior_layer(self, component_name: str, target: Target) -> Any:
        """The prior parameterizer of z (or z_hat) of a component."""
        _, attr = self.TARGETS[target]
        return getattr(self.model.components[component_name], attr)

    def _announce(self, component_name: str, target: Target) -> None:
        """Print which prior is being initialized, in the component's
        color."""
        index = self.component_order.index(component_name)
        color = VerboseCallback._COLORS[index % len(VerboseCallback._COLORS)]
        print(
            f"\033[91mPerform GMM K-means Initialization ({target} prior) for "
            f"{color}{VerboseCallback._BOLD}[{component_name}]\033[91m."
            f"{VerboseCallback._RESET}"
        )
