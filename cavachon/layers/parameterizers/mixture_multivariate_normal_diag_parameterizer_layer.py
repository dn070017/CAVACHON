import numpy as np
import tensorflow as tf


class MixtureMultivariateNormalDiagParameterizerLayer(tf.keras.layers.Layer):
    """MixtureMultivariateNormalDiagParameterizerLayer

    Parameterizer for mixture of multivariate normal distributions with
    diagonal covariance matrix (logits, loc and scale_diag).

    """

    def __init__(
        self,
        event_dims: int,
        n_components: int,
        unit_variance: bool = False,
        name: str = "mixture_multivariate_normal_diag_parameterizer_layer",
    ):
        """Constructor for MultivariateNormalDiagParameterizerLayer

        Parameters
        ----------
        event_dims: int
            number of event dimensions for the multivariate normal
            distributions with diagonal covariance matrix.

        n_components: int
            number of components in the mixture distributions.

        unit_variance: bool, optional
            use unit variance. Defaults to False.

        name: str, optional
            Name for the tensorflow layer. Defaults to
            'mixture_multivariate_normal_diag_parameterizer_layer'.

        """
        super().__init__(name=name)
        self.event_dims: int = event_dims
        self.n_components: int = n_components
        self.unit_variance: bool = unit_variance

        return

    def _make_grid_positions(self, radius: float = 1.5) -> np.ndarray:
        K = self.n_components
        D = self.event_dims
        # side length of the grid along each dimension (hypercube)
        side = int(np.ceil(K ** (1.0 / D)))
        if side < 1:
            side = 1

        coords = []
        for idx in range(K):
            # represent idx in base `side` with D digits
            digits = []
            tmp = idx
            for _ in range(D):
                digits.append(tmp % side)
                tmp //= side
            # digits is in reverse order; reverse back
            digits = digits[::-1]
            digits = np.array(digits, dtype=np.float32)

            # center the grid around 0 and scale to roughly [-1, 1]* radius (depends on radius)
            center = (side - 1) / 2.0
            if center > 0:
                coord = (digits - center) / center
            else:
                coord = np.zeros_like(digits)

            coord = coord * radius
            coords.append(coord)

        coords = np.stack(coords, axis=0)  # (K, D)
        return coords

    def build(self, input_shape: tf.TensorShape) -> None:
        """Create necessary tf.Variable for the first time being called.
        (see tf.keras.layers.Layer)

        Parameters
        ----------
        input_shape: tf.TensorShape
            input shape of tf.Tensor.

        """
        self.logits_weight = self.add_weight(
            name=f"{self.name}_logits_weight",
            shape=(int(input_shape[-1]), self.n_components),
            initializer=tf.keras.initializers.Constant(0.0),
        )
        self.logits_bias = self.add_weight(
            name=f"{self.name}_logits_bias",
            shape=(1, self.n_components),
            initializer=tf.keras.initializers.Constant(0.0),
        )

        self.loc_weight = []
        self.loc_bias = []
        if not self.unit_variance:
            self.scale_diag_weight = []
            self.scale_diag_bias = []
        # --- NEW: compute grid coordinates for all components ---
        grid_coords = self._make_grid_positions()  # shape (K, event_dims)
        # --------------------------------------------------------

        for i in range(self.n_components):
            # means should not depend on input -> weights = 0
            self.loc_weight.append(
                self.add_weight(
                    name=f"{self.name}_loc_weight_{i}",
                    shape=(int(input_shape[-1]), self.event_dims),
                    initializer=tf.keras.initializers.Constant(0.0),
                )
            )

            # bias = initial mean for component i (grid position)
            self.loc_bias.append(
                self.add_weight(
                    name=f"{self.name}_loc_bias_{i}",
                    shape=(1, self.event_dims),
                    initializer=tf.keras.initializers.Constant(
                        grid_coords[i][None, :]  # shape (1, D)
                    ),
                )
            )
            ####

            if not self.unit_variance:
                self.scale_diag_weight.append(
                    self.add_weight(
                        name=f"{self.name}_scale_diag_weight_{i}",
                        shape=(int(input_shape[-1]), self.event_dims),
                        initializer=tf.keras.initializers.Constant(0.0),  # test
                    )
                )
                self.scale_diag_bias.append(
                    self.add_weight(
                        name=f"{self.name}_scale_diag_bias_{i}",
                        shape=(1, self.event_dims),
                        initializer=tf.keras.initializers.Constant(0.02),  # it was 0.5
                    )
                )

        return

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        """Parameterize mixture of multivariate normal distributions
        with diagonal covariance matrix with loc and scale_diag using
        the given tf.Tensor.

        Parameters
        ----------
        inputs: tf.Tensor
            inputs Tensor.

        Returns
        -------
        tf.Tensor
            logits, loc and scale_diag for normal distributions with
            diagonal covariance matrix, with shape
            (batch, n_components, event_dims * 2 + 1), where
            1.  results[..., n_components, 0] are the logits
            2.  results[..., n_components, 1:event_dims+1] are the loc
                (mean)
            3.  results[..., n_components, event_dims+1:] are the
                scale_diag (std)

        """

        # shape: (batch, n_components, 1)
        logits = tf.expand_dims(
            tf.matmul(inputs, self.logits_weight) + self.logits_bias, -1
        )
        means = []
        scale_diag = []
        for i in range(self.n_components):
            loc_weight = self.loc_weight[i]
            loc_bias = self.loc_bias[i]
            if not self.unit_variance:
                scale_diag_weight = self.scale_diag_weight[i]
                scale_diag_bias = self.scale_diag_bias[i]

            mean = tf.matmul(inputs, loc_weight) + loc_bias

            (means.append(mean),)
            if not self.unit_variance:
                scale_diag.append(
                    tf.math.softplus(
                        tf.matmul(inputs, scale_diag_weight) + scale_diag_bias
                    )
                    + 1e-7
                )
            else:
                scale_diag.append(tf.ones_like(mean))

        # shape: (batch, n_components, event_dims)
        means = tf.stack(means, axis=1)
        scale_diag = tf.stack(scale_diag, axis=1)

        # shape: (batch, n_components, event_dims * 2 + 1)
        return tf.concat([logits, means, scale_diag], axis=-1)

    @property
    def parameters(self) -> tf.Tensor:
        """Getter function for parameters, return the parameters of the
        priors. Used when the parameterizer is trained without the use
        of observed data (each parameter is itself a trainable
        parameters)

        Returns
        -------
        tf.Tensor
            logits, loc and scale_diag for normal distributions with
            diagonal covariance matrix, with shape
            (n_components, event_dims * 2 + 1), where
            1.  results[n_components, 0] are the logits
            2.  results[n_components, 1:event_dims+1] are the loc (mean)
            3.  results[n_components, event_dims+1:] are the scale_diag
                (std)

        """
        return tf.squeeze(self(tf.ones((1, 1))))
