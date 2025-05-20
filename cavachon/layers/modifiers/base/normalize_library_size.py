import tensorflow as tf


@tf.keras.utils.register_keras_serializable()
class NormalizeLibrarySize(tf.keras.layers.Layer):
    """NormalizedLibrarySize

    Modifier used to normalize (read / library size) the tf.Tensor with
    library size (the reduced sum of the last dimension)

    """

    def __init__(self, **kwargs):
        """Constructor for NormalizeLibrarySize"""
        super().__init__(**kwargs)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Normalize tf.Tensor stored in input with library size (the sum
        of values in the last dimension)

        Parameters
        ----------
        inputs: tf.Tensor
            input tf.Tensor.

        Returns
        -------
        tf.Tensor
            library size normalized tf.Tensor.

        """
        libsize = tf.expand_dims(tf.reduce_sum(inputs, axis=-1), -1)
        outputs = inputs / libsize

        return outputs
