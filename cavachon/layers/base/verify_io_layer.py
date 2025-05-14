from typing import Any, Dict, List

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
        self,
        expected_input_keys: List[str] | None = None,
        expected_output_keys: List[str] | None = None,
        *args,
        **kwargs,
    ):
        """Constructor for VerifyIOLayer.

        Parameters
        ----------
        expected_input_keys: List[str] | None, optional
            the expected keys in the input dictionary. Defaults to None.

        expected_output_keys: List[str] | None, optional
            the expected keys in the output dictionary. Defaults to None.

        """
        super().__init__(*args, **kwargs)
        self.expected_input_keys = expected_input_keys
        self.expected_output_keys = expected_output_keys

    def verify_tensor_like(
        self,
        value: Dict[str, TensorUtils.TensorLike] | TensorUtils.TensorLike,
        name: str,
    ):
        """Verify that the input is a tf.Tensor or tf.keras.KerasTensor.

        Parameters
        ----------
        value: Dict[str, TensorUtils.TensorLike] | TensorUtils.TensorLike
            the input value to verify.

        name: str
            the name of the input value.

        Raises
        ------
        TypeError
            if the input is not a tf.Tensor or tf.keras.KerasTensor.
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
            the inputs to verify.

        Raises
        ------
        TypeError
            if the input is not a dictionary.

        KeyError
            if any of the expected keys are not present in the input dictionary.
        """
        if self.expected_input_keys is None:
            self.verify_tensor_like(inputs, "inputs")
        else:
            self.verify_io_keys(inputs, self.expected_input_keys, "inputs")

    def verify_outputs(
        self, outputs: Dict[str, TensorUtils.TensorLike] | TensorUtils.TensorLike
    ):
        """Verify the outputs of the layer.

        Parameters
        ----------
        outputs: Dict[str, TensorUtils.TensorLike] | TensorUtils.TensorLike
            the outputs to verify.

        Raises
        ------
        TypeError
            if the output is not a dictionary.

        KeyError
            if any of the expected keys are not present in the output dictionary.
        """
        if self.expected_output_keys is None:
            self.verify_tensor_like(outputs, "outputs")
        else:
            self.verify_io_keys(outputs, self.expected_output_keys, "outputs")

    def get_config(self) -> Dict[str, Any]:
        """Returns the configuration of the layer.

        Returns
        -------
        Dict[str, Any]
            a dictionary containing the configuration of the layer.

        """
        config = super().get_config()
        config.update(
            {
                "expected_input_keys": self.expected_input_keys,
                "expected_output_keys": self.expected_output_keys,
            }
        )
        return config
