import tensorflow as tf


class EarlyStoppingCallback(tf.keras.callbacks.EarlyStopping):
    """EarlyStopping with custom red message using the cumulative epoch."""

    def __init__(self, *args, cumulative_offset: int = 0, **kwargs):
        # keyword-only, so positional arguments (e.g. monitor) still reach
        # tf.keras.callbacks.EarlyStopping
        super().__init__(*args, **kwargs)
        self._cumulative_offset = cumulative_offset

    def on_train_end(self, logs=None):
        stopped_epoch = self.stopped_epoch
        super().on_train_end(logs)
        if stopped_epoch > 0 and self.restore_best_weights:
            # 1-based, matching the "Total Epoch" shown by VerboseCallback
            epoch = self._cumulative_offset + self.best_epoch + 1
            print(f"\033[91mRestoring Model Weights from Epoch {epoch}\033[0m")
