import tensorflow as tf


class OrderedLogisticParameterizerLayer(tf.keras.layers.Layer):
    """OrderedLogisticParameterizerLayer

    Parameterizer for OrderedLogistic distributions (score and
    cutpoints).

    """

    N_CUTPOINTS = 4

    def __init__(
        self,
        event_dims: int,
        use_shared_cutpoints: bool = False,
        name: str = "ordered_logistic_parameterizer_layer",
    ):
        """Constructor for OrderedLogisticParameterizerLayer

        Parameters
        ----------
        event_dims: int
            number of event dimensions for the ordered logistic
            distribution.

        use_shared_cutpoints: bool
            use cutpoints shared across all samples instead of modeling
            the cutpoints independently for each sample. The cutpoints
            are per-feature in both cases.

        name: str, optional
            Name for the tensorflow layer.
        """
        super().__init__(name=name)
        self.event_dims: int = event_dims
        self.use_shared_cutpoints: bool = use_shared_cutpoints
        return

    def build(self, input_shape: tf.TensorShape) -> None:
        n_cut = self.N_CUTPOINTS

        self.score_weight = self.add_weight(
            name=f"{self.name}_score_weight",
            shape=(int(input_shape[-1]), self.event_dims),
        )
        self.score_bias = self.add_weight(
            name=f"{self.name}_score_bias", shape=(1, self.event_dims)
        )

        if self.use_shared_cutpoints:
            self.cutpoint_weight = self.add_weight(
                name=f"{self.name}_cutpoint_weight",
                shape=(1, self.event_dims * n_cut),
            )
        else:
            self.cutpoint_weight = self.add_weight(
                name=f"{self.name}_cutpoint_weight",
                shape=(int(input_shape[-1]), self.event_dims * n_cut),
            )

        # start with roughly even regions: raw [-1.5, 0, 0, 0] gives
        # cutpoints near [-1.5, -0.8, -0.1, 0.6] after the step transform
        init = tf.tile(
            tf.constant([[-1.5, 0.0, 0.0, 0.0]], dtype=tf.float32),
            [self.event_dims, 1],
        )
        self.cutpoint_bias = self.add_weight(
            name=f"{self.name}_cutpoint_bias",
            shape=(1, self.event_dims * n_cut),
            initializer=tf.keras.initializers.Constant(
                tf.reshape(init, (1, -1)).numpy()
            ),
        )
        return

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        """Parameterize ordered logistic distributions with score and
        cutpoints using the given tf.Tensor.

        Returns
        -------
        tf.Tensor
            shape (batch, event_dims * 5), where
            1.  results[..., 0:event_dims] are the scores
            2.  results[..., event_dims:] are the raw cutpoints,
                four per feature.
        """
        score = tf.matmul(inputs, self.score_weight) + self.score_bias

        if self.use_shared_cutpoints:
            batch_size = tf.shape(inputs)[0]
            cutpoints = self.cutpoint_weight + self.cutpoint_bias
            cutpoints = tf.repeat(cutpoints, repeats=batch_size, axis=0)
        else:
            cutpoints = tf.matmul(inputs, self.cutpoint_weight) + self.cutpoint_bias

        # constraints applied in OrderedLogisticDistribution.constrain
        return tf.concat([score, cutpoints], axis=-1)
