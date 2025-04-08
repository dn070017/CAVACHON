from abc import ABC, abstractmethod
from typing import Mapping

import tensorflow as tf


class Distribution(ABC):
    """Abstract class of Distribution, which defines an interface of
    from_parameterizer_output().

    """

    @classmethod
    @abstractmethod
    def from_parameterizer_output(
        cls, params: tf.Tensor | Mapping[str, tf.Tensor], **kwargs
    ):
        """Create Tensorflow Probability Distribution from the outputs
        of parameterizers.

        Parameters
        ----------
        params: tf.Tensor | Mapping[str, tf.Tensor]
            parameters for the distribution created by parameterizers.
            Alternatively, a mapping of tf.Tensor with parameter name
            as keys can be provided.

        Returns
        -------
        tfp.distributions.Distribution
            created Tensorflow Probability Distribution like instance.

        See Also
        --------
        Parameterizer: the modules used to create parameters for
        distributions.

        """
        pass
