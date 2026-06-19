import tensorflow as tf


class ProgressiveScaler(tf.keras.layers.Layer):
    """ProgressiveScaler

    ProgressiveScaler used to scale the inputs during training. The input
    tensor will be scale as current_iteration/iteration * tensor. Do
    nothing in inference mode.

    Attributes
    ----------
    total_iterations: tf.Variable
        total iterations in the progressive training.

    current_iteration: tf.Variable
        current iterations in the progressive training.

    """

    def __init__(self, total_iterations: int = 5000, scale: float = 1.0, name: str = "progressive_scaler"):
        """Constructor for ProgressiveScaler

        Parameters
        ----------
        total_iterations: int, optional
            total iterations in the progressive training. Defaults to 5000.

        scale: float, optional
            multiplier applied to the final output. Defaults to 1.0.

        name: str, optional
            Name for the tensorflow layer. Defaults to 'progressive_scaler'.

        """
        super().__init__(name=name)
        self.total_iterations = tf.Variable(total_iterations, trainable=False, dtype=tf.float32)
        self.current_iteration = tf.Variable(tf.ones(()), trainable=False)
        self.scale = tf.constant(float(scale), dtype=tf.float32)

    def call(self, inputs: tf.Tensor, training: bool = False, **kwargs) -> tf.Tensor:
        """Forward pass (stateless — no mutation)."""
        progress = self._progress()
        return self.scale * progress * inputs

    def increment(self):
        """Advance one iteration, clamp to [0, total_iterations]."""
        self.current_iteration.assign_add(1.0)
        self._clip_current()

    def decrement(self):
        """Regress one iteration, clamp to [0, total_iterations]."""
        self.current_iteration.assign_sub(1.0)
        self._clip_current()

    def _clip_current(self):
        """Clamp current_iteration into [0, total_iterations]."""
        self.current_iteration.assign(
            tf.clip_by_value(self.current_iteration, 0.0, self.total_iterations)
        )

    def _progress(self):
        """Return (current/total)² ∈ [0, 1] — the scaling progress.

        Lower-bound clipped at 1e-8 (not 0) so that train_step can
        recover the raw unweighted loss via weighted_loss / weight_scalar
        without hitting division-by-zero.
        """
        fraction = self.current_iteration / tf.maximum(self.total_iterations, 1.0)
        fraction = tf.clip_by_value(fraction, 1e-8, 1.0)
        return fraction ** 2

    def numpy(self):
        """Return current effective weight = scale × progress as a Python float."""
        return float(self.scale * self._progress())

    def assign(self, value):
        """MutableVariable-compatible setter. Delegates to pin_to."""
        self.pin_to(float(value))

    def pin_to(self, value: float):
        """Pin output to *value* (sets progress = value / scale)."""
        target = float(value) / max(float(self.scale), 1e-7)
        self.total_iterations.assign(1.0)
        if target <= 0.0:
            self.current_iteration.assign(0.0)
        elif target >= 1.0:
            self.current_iteration.assign(1.0)
        else:
            self.current_iteration.assign(tf.sqrt(float(target)))

    def activate(self, total_iterations: float):
        """Begin progressive scaling from 0→1 over *total_iterations* steps."""
        self.total_iterations.assign(float(total_iterations))
        self.current_iteration.assign(0.0)
