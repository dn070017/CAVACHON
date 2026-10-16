import tensorflow as tf


class StudenttParameterizerLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        event_dims: int,
        name: str = "studentt_parameterizer_layer",
    ):
        super().__init__(name=name)
        self.event_dims = event_dims

    def build(self, input_shape: tf.TensorShape) -> None:
        # event_dims * 4 outputs: loc, scale, df, skewness (raw, unconstrained)
        self.weight = self.add_weight(
            name=f"{self.name}_weight",
            shape=(int(input_shape[-1]), self.event_dims * 4),
        )
        self.bias = self.add_weight(
            name=f"{self.name}_bias", shape=(1, self.event_dims * 4)
        )

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        # Linear map only. Constraints are applied in StudenttDistribution.constrain.
        return tf.matmul(inputs, self.weight) + self.bias
