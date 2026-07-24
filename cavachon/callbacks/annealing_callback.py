import tensorflow as tf

from cavachon.layers.progressive_scaler import ProgressiveScaler


class AnnealingCallback(tf.keras.callbacks.Callback):
    """Unified annealing callback for both parent-component annealing
    and intra-component KL annealing (standard_kl → GMM).

    The callback is driven by a schedule(epoch) function that returns
    a dict mapping loss-name substring patterns to target weights.
    For each loss, the longest matching pattern determines its weight
    (so ``child1_standard_kl_divergence`` wins over ``child1``).

    K-means initialization can be triggered at a specific epoch.
    """

    def __init__(
        self,
        schedule,
        kmeans_epoch=None,
        scheduler=None,
        component_name=None,
    ):
        super().__init__()
        self.schedule = schedule
        self.kmeans_epoch = kmeans_epoch
        self.scheduler = scheduler
        self.component_name = component_name
        self._kmeans_done = False

    def on_epoch_begin(self, epoch, logs=None):
        targets = self.schedule(epoch)
        for loss_name, loss_fn in self.model.loss.items():
            if not hasattr(loss_fn, "weight") or not hasattr(
                loss_fn.weight, "assign"
            ):
                continue
            best_match = None
            best_len = 0
            for pattern, weight in targets.items():
                if pattern in loss_name and len(pattern) > best_len:
                    best_match = weight
                    best_len = len(pattern)
            if best_match is not None:
                loss_fn.weight.assign(tf.cast(best_match, tf.float32))

        if (
            self.kmeans_epoch is not None
            and epoch == self.kmeans_epoch
            and not self._kmeans_done
        ):
            saved_trainable = {
                name: comp.trainable
                for name, comp in self.model.components.items()
            }
            self.model.trainable = False
            self.scheduler.initialize_gmm_priors_with_kmeans(
                component_name=self.component_name,
                seed=42,
                add_noise=True,
                noise_std=0.1,
            )
            for name, was_trainable in saved_trainable.items():
                self.model.components[name].trainable = was_trainable
            self._kmeans_done = True

    def on_batch_end(self, batch, logs=None):
        """Increment all progressive scalers after each training batch."""
        try:
            comp = self.model.components[self.component_name]
            ps = comp.hierarchical_encoder.progressive_scaler
            if float(ps.total_iterations) > 1.0:
                ps.increment()
        except Exception:
            pass
        try:
            for cn in self.scheduler.component_order:
                for wdict in ('_standard_kl_weights', '_gmm_kl_weights', '_data_loss_weights'):
                    w = getattr(self.model, wdict, {}).get(cn)
                    if wdict == '_data_loss_weights' and isinstance(w, dict):
                        for v in w.values():
                            if isinstance(v, ProgressiveScaler) and float(v.total_iterations) > 1.0:
                                v.increment()
                    elif w is not None and isinstance(w, ProgressiveScaler) and float(w.total_iterations) > 1.0:
                        w.increment()
        except Exception:
            pass
