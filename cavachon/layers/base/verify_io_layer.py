from typing import Dict, List

import tensorflow as tf

from cavachon.utils.tensor_utils import TensorUtils


@tf.keras.utils.register_keras_serializable()
class VerifyIOLayer(tf.keras.layers.Layer):
    """VerifyIOLayer

    VerifyIOLayer is a Keras layer used to verify the input and output
    tensors of a model. It provide functionality to check inputs and
    outputs are of the correct type and that the expected keys are
    present in the input dictionary.
    """

    def __init__(
        self, input_keys: List[str] | None, output_keys: List[str] | None, **kwargs
    ):
        """Constructor for VerifyIOLayer.

        Parameters
        ----------
        input_keys: List[str] | None, optional
            The expected keys in the input dictionary. Defaults to None.

        output_keys: List[str] | None, optional
            The expected keys in the output dictionary. Defaults to None.

        """
        super().__init__(**kwargs)
        self.input_keys = input_keys
        self.output_keys = output_keys

    def verify_tensor_like(
        self,
        value: Dict[str, TensorUtils.TensorLike] | TensorUtils.TensorLike,
        name: str,
    ):
        """Verify that the input is a tf.Tensor or tf.keras.KerasTensor.

        Parameters
        ----------
        value: Dict[str, TensorUtils.TensorLike] | TensorUtils.TensorLike
            The input value to verify.

        name: str
            The name of the input value.

        Raises
        ------
        TypeError
            If the input is not a tf.Tensor or tf.keras.KerasTensor.
        """
        if not isinstance(value, (tf.Tensor, tf.keras.KerasTensor)):
            raise TypeError(
                f"{name} must be either tf.Tensor or tf.keras.KerasTensor. "
                f"({self.__class__.__name__}: {self.name})"
            )

    def verify_io_keys(
        self,
        value: Dict[str, TensorUtils.TensorLike] | TensorUtils.TensorLike,
        expected_keys: List[str],
        name: str,
    ):
        """Verify that the input dictionary contains the expected keys.

        Parameters
        ----------
        value: Dict[str, TensorUtils.TensorLike] | TensorUtils.TensorLike
            the input dictionary to verify.

        expected_keys: List[str]
            the expected keys in the dictionary.

        name: str
            the name of the dictionary (show in error message).

        Raises
        ------
        TypeError
            if the input is not a dictionary.

        KeyError
            if any of the expected keys are not present in the dictionary.
        """
        if not isinstance(value, dict):
            raise TypeError(
                f"{name} must be a dictionary with string keys "
                f"({', '.join(expected_keys)}) and tf.Tensor values. "
                f"({self.__class__.__name__}: {self.name})"
            )
        for k in expected_keys:
            if k not in value:
                raise KeyError(
                    f"'{k}' is not in {name}. ({self.__class__.__name__}: {self.name})"
                )

    def verify_inputs(
        self, inputs: Dict[str, TensorUtils.TensorLike] | TensorUtils.TensorLike
    ):
        """Verify the inputs to the layer.

        Parameters
        ----------
        inputs: Dict[str, TensorUtils.TensorLike] | TensorUtils.TensorLike
            The inputs to verify.

        Raises
        ------
        TypeError
            If the input is not a dictionary.

        KeyError
            If any of the expected keys are not present in the input dictionary.
        """
        if self.input_keys is None:
            self.verify_tensor_like(inputs, "inputs")
        else:
            self.verify_io_keys(inputs, self.input_keys, "inputs")

    def verify_outputs(
        self, outputs: Dict[str, TensorUtils.TensorLike] | TensorUtils.TensorLike
    ):
        """Verify the outputs of the layer.

        Parameters
        ----------
        outputs: Dict[str, TensorUtils.TensorLike] | TensorUtils.TensorLike
            The outputs to verify.

        Raises
        ------
        TypeError
            If the output is not a dictionary.

        KeyError
            If any of the expected keys are not present in the output dictionary.
        """
        if self.output_keys is None:
            self.verify_tensor_like(outputs, "outputs")
        else:
            self.verify_io_keys(outputs, self.output_keys, "outputs")
