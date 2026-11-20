import tensorflow as tf

from cavachon.environment.constants import Constants
from cavachon.layers.progressive_scaler import ProgressiveScaler


class AnnealingCallback(tf.keras.callbacks.Callback):
    """Unified annealing callback for both parent-component annealing
    and intra-component KL annealing (standard_kl → GMM).

    The callback is driven by a schedule(epoch) function that returns
    a dict mapping loss-name substring patterns to target weights.
    For each loss, the longest matching pattern determines its weight
    (so ``child1_standard_kl_divergence`` wins over ``child1``). The
    z_hat GMM KL divergence is only matched by patterns that contain
    its postfix, so component-wide patterns (e.g. ``child1``) never
    switch it on outside the z_hat annealing phase.

    K-means initialization can be triggered at a specific epoch through
    ``kmeans_init`` (e.g. ``GMMPriorInitializer.initialize`` bound to a
    component and target), which restores the trainable flags itself.

    Parameters
    ----------
    schedule: Callable[[int], Mapping[str, float]]
        loss-name pattern -> weight for each epoch.

    kmeans_epoch: int, optional
        epoch at which ``kmeans_init`` is called (once).

    kmeans_init: Callable[[], Any], optional
        initializes the GMM prior.

    component_name: str, optional
        component whose alpha is advanced per batch.

    component_names: Sequence[str], optional
        components whose progressive loss weights are advanced per
        batch.
    """

    def __init__(
        self,
        schedule,
        kmeans_epoch=None,
        kmeans_init=None,
        component_name=None,
        component_names=(),
    ):
        super().__init__()
        self.schedule = schedule
        self.kmeans_epoch = kmeans_epoch
        self.kmeans_init = kmeans_init
        self.component_name = component_name
        self.component_names = list(component_names)
        self._kmeans_done = False

    def on_epoch_begin(self, epoch, logs=None):
        targets = self.schedule(epoch)
        for loss_name, loss_fn in self.model.loss.items():
            if not hasattr(loss_fn, "weight") or not hasattr(loss_fn.weight, "assign"):
                continue
            best_match = None
            best_len = 0
            is_z_hat_kl = Constants.MODEL_LOSS_Z_HAT_GMM_KL_POSTFIX in loss_name
            for pattern, weight in targets.items():
                if (
                    is_z_hat_kl
                    and Constants.MODEL_LOSS_Z_HAT_GMM_KL_POSTFIX not in pattern
                ):
                    continue
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
            if self.kmeans_init is not None:
                self.kmeans_init()
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
            for cn in self.component_names:
                for wdict in (
                    "_standard_kl_weights",
                    "_gmm_kl_weights",
                    "_data_loss_weights",
                ):
                    w = getattr(self.model, wdict, {}).get(cn)
                    if wdict == "_data_loss_weights" and isinstance(w, dict):
                        for v in w.values():
                            if (
                                isinstance(v, ProgressiveScaler)
                                and float(v.total_iterations) > 1.0
                            ):
                                v.increment()
                    elif (
                        w is not None
                        and isinstance(w, ProgressiveScaler)
                        and float(w.total_iterations) > 1.0
                    ):
                        w.increment()
        except Exception:
            pass
