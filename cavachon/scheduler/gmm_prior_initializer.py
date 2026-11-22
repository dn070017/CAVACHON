import os
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

    assignments: np.ndarray
        cluster of each sample, with shape (n_samples, ).

    seeds: np.ndarray
        the K-means++ seeds the Lloyd iterations started from, with
        shape (n_clusters, event_dims).

    """

    locs: tf.Tensor
    logits: np.ndarray
    scale_biases: np.ndarray
    assignments: np.ndarray
    seeds: np.ndarray


class GMMPriorInitializer:
    """GMMPriorInitializer

    Initializes the GMM prior (means, logits and stds) of a component
    with K-means (K-means++ seeds, then Lloyd iterations) on the
    posterior means of its latent representation z or z_hat. Each
    cluster's prior variance is moment-matched to the posteriors of
    its samples: the variance of their means plus their mean posterior
    variance, so the prior covers the sampled latents, not only their
    means (with only the former, dimensions where the means barely
    differ but the posteriors are wide get a far too narrow prior).

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

    outdir: str, optional
        if set, the latents and the fitted prior of every
        initialization are saved to
        ``{outdir}/kmeans_initialization/{component}/{target}.npz``.

    """

    # upper bound of the Lloyd iterations (they stop earlier once the
    # assignments no longer change)
    MAX_LLOYD_ITERATIONS = 100

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
        outdir: Optional[str] = None,
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

        outdir: str, optional
            where to save the latents and the fitted prior of every
            initialization (see the class attributes). Defaults to None
            (not saved).
        """
        self.model = model
        self.mdata = mdata
        self.batch_size = batch_size
        self.component_order = component_order
        self.seed = seed
        self.add_noise = add_noise
        self.noise_std = noise_std
        self.outdir = outdir

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
            z, z_std = self._predict_representation(component_name, target)
        prior_layer = self._prior_layer(component_name, target)
        estimate = self.estimate(z, prior_layer.n_components, z_std)
        self.apply(prior_layer, estimate)
        if self.outdir is not None:
            self.save(component_name, target, z, z_std, estimate)
        return z

    def estimate(
        self,
        z: np.ndarray,
        n_clusters: int,
        z_std: Optional[np.ndarray] = None,
    ) -> GMMPriorEstimate:
        """Estimate the GMM prior parameters of ``n_clusters`` clusters
        from the posterior means ``z`` with K-means: K-means++ seeds,
        then Lloyd iterations, so the centers are cluster means.

        Parameters
        ----------
        z: np.ndarray
            posterior means of the latent representation, with shape
            (n_samples, event_dims).

        n_clusters: int
            number of GMM components.

        z_std: np.ndarray, optional
            posterior standard deviations, with the shape of ``z``. Each
            cluster's prior variance is the variance of its means plus
            its mean posterior variance. Defaults to None: the variance
            of the means only (0.5 std for a single-sample cluster).

        Returns
        -------
        GMMPriorEstimate
            centers (optionally with noise), log proportions and scale
            biases of the clusters, the assignments and the seeds.
        """
        z_tensor = tf.constant(z, dtype=tf.float32)
        n_samples, event_dims = z.shape[0], z.shape[1]
        seeds = self.kmeans_plus_plus(z_tensor, n_clusters, seed=self.seed).numpy()
        centers, assignments = self.lloyd(z, seeds)
        cluster_counts = np.array(
            [np.sum(assignments == k) for k in range(n_clusters)], dtype=np.int32
        )
        locs = tf.constant(centers, dtype=tf.float32)
        if self.add_noise:
            locs = locs + tf.random.normal(
                shape=locs.shape,
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
            members = assignments == k
            if z_std is not None and members.any():
                # moment-matched to the members' posteriors
                variance = np.var(z[members], axis=0) + np.mean(
                    np.square(z_std[members]), axis=0
                )
                std = np.sqrt(variance)
            elif members.sum() > 1:
                std = np.std(z[members], axis=0)
            else:
                std = np.full(event_dims, 0.5)
            scale_biases[k] = self.inverse_softplus(std)
        return GMMPriorEstimate(
            locs=locs,
            logits=logits,
            scale_biases=scale_biases,
            assignments=assignments,
            seeds=seeds,
        )

    @classmethod
    def lloyd(cls, z: np.ndarray, centers: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Lloyd iterations of K-means from ``centers``: assign every
        sample to its nearest center, move each center to the mean of
        its samples (an empty cluster keeps its center), until the
        assignments no longer change or ``MAX_LLOYD_ITERATIONS``.

        Parameters
        ----------
        z: np.ndarray
            data, with shape (n_samples, n_dimensions).

        centers: np.ndarray
            initial centers, with shape (n_clusters, n_dimensions).

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            the centers and the assignment of every sample.
        """
        z = np.asarray(z, dtype=np.float32)
        z_tensor = tf.constant(z)
        centers = np.array(centers, dtype=np.float32)
        assignments = cls.assign_clusters(z_tensor, tf.constant(centers)).numpy()
        for _ in range(cls.MAX_LLOYD_ITERATIONS):
            for k in range(centers.shape[0]):
                members = assignments == k
                if members.any():
                    centers[k] = z[members].mean(axis=0)
            updated = cls.assign_clusters(z_tensor, tf.constant(centers)).numpy()
            if np.array_equal(updated, assignments):
                break
            assignments = updated
        return centers, assignments

    def save(
        self,
        component_name: str,
        target: Target,
        z: np.ndarray,
        z_std: Optional[np.ndarray],
        estimate: GMMPriorEstimate,
    ) -> str:
        """Save the latents and the fitted prior of an initialization to
        ``{outdir}/kmeans_initialization/{component}/{target}.npz``:
        ``z_mean`` and ``z_std`` (the posteriors, in the order of the
        cells of ``mdata``; ``z_std`` is empty when unknown), ``obs_names``, ``assignments``, ``seeds``
        (the K-means++ seeds) and the prior written to the model,
        ``prior_loc``, ``prior_std`` and ``prior_logits``.

        Returns
        -------
        str
            the path of the saved file.
        """
        directory = os.path.join(
            str(self.outdir), "kmeans_initialization", component_name
        )
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f"{target}.npz")
        prior_std = np.log1p(np.exp(estimate.scale_biases.astype(np.float64)))
        np.savez(
            path,
            z_mean=z,
            # empty when the model gave no posterior parameters
            z_std=np.empty((0,), dtype=np.float32) if z_std is None else z_std,
            obs_names=np.asarray(self.mdata.obs_names, dtype=str),
            assignments=estimate.assignments,
            seeds=estimate.seeds,
            prior_loc=np.asarray(estimate.locs),
            prior_std=prior_std,
            prior_logits=estimate.logits,
        )
        return path

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
    def inverse_softplus(std: np.ndarray) -> np.ndarray:
        """Bias whose softplus is ``std``: log(exp(std) - 1), written
        as std + log(1 - exp(-std)) so it neither overflows for large
        std nor loses precision for small std. std is floored at 1e-6
        (a cluster of identical points has std 0).

        Parameters
        ----------
        std: np.ndarray
            target standard deviations.

        Returns
        -------
        np.ndarray
            the scale_diag biases, as float32.
        """
        std = np.maximum(np.asarray(std, dtype=np.float64), 1e-6)
        return (std + np.log(-np.expm1(-std))).astype(np.float32)

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
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Predict z (or z_hat) of a component on the whole data: the
        posterior means and standard deviations (None when the model
        doesn't output the posterior parameters)."""
        suffix, _ = self.TARGETS[target]
        parameters_key = f"{component_name}{suffix}_parameters"
        outputs = self.model.predict(
            self.mdata,
            batch_size=self.batch_size,
            verbose=0,
            output_keys=[f"{component_name}{suffix}", parameters_key],
        )
        z = outputs[f"{component_name}{suffix}"]
        parameters = outputs.get(parameters_key)
        if parameters is None or np.size(parameters) == 0:
            return z, None
        # parameters: loc and scale_diag of the posterior
        z_std = np.asarray(parameters)[:, z.shape[1] : 2 * z.shape[1]]
        return z, z_std

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
