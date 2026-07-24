import tensorflow as tf


class OptimizerStateCallback(tf.keras.callbacks.Callback):
    """Snapshot optimizer state each epoch; restore on early stop."""
    def __init__(self):
        super().__init__()
        self._opt_states = {}

    def on_epoch_end(self, epoch, logs=None):
        try:
            if hasattr(self.model, 'optimizer') and self.model.optimizer is not None:
                self._opt_states[epoch] = self.model.optimizer.get_weights()
        except Exception:
            pass

    def on_train_end(self, logs=None):
        try:
            if not hasattr(self.model, 'optimizer') or self.model.optimizer is None:
                return
            for cb in getattr(self.model, '_callbacks', []) or []:
                if isinstance(cb, tf.keras.callbacks.EarlyStopping) and \
                   getattr(cb, 'best_epoch', None) is not None:
                    best = cb.best_epoch
                    if best in self._opt_states:
                        self.model.optimizer.set_weights(self._opt_states[best])
                    break
        except Exception:
            pass
