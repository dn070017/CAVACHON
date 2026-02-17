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
        # We need event_dims * 3 outputs: one set each for loc, scale, and df
        self.weight = self.add_weight(
            name=f"{self.name}_weight",
            shape=(int(input_shape[-1]), self.event_dims * 3),
        )
        self.bias = self.add_weight(
            name=f"{self.name}_bias", shape=(1, self.event_dims * 3)
        )

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        # Simple linear transformation. The activation (softplus)
        # happens inside the Distribution class.
        return tf.matmul(inputs, self.weight) + self.bias
