import time

import tensorflow as tf


class VerboseCallback(tf.keras.callbacks.Callback):
    """Print a colored banner identifying the active training phase.

    Colors cycle per component so different components stand out
    in the log stream without modifying Keras metric names.
    """

    _COLORS = ["\033[34m", "\033[32m", "\033[35m", "\033[36m", "\033[33m"]
    _RED = "\033[31m"
    _BOLD = "\033[1m"
    _RESET = "\033[0m"

    def __init__(self, labels, phase="Regular Training",
                 loss_prefixes=None, phase_epochs=None,
                 cumulative_offset=0, cumulative_total=None,
                 phase_number=1, component_order=None,
                 debug=True):
        super().__init__()
        self.labels = labels
        self.phase = phase
        self.loss_prefixes = loss_prefixes or {}
        self.phase_epochs = phase_epochs
        self.cumulative_offset = cumulative_offset
        self.cumulative_total = cumulative_total
        self.phase_number = phase_number
        self.component_order = component_order or []
        self.debug = debug
        self._epoch_start = None

    def _strip_component_prefix(self, key):
        """Remove the leading component name from a metric key."""
        for comp_name in self.component_order:
            prefix = comp_name + "_"
            if key.startswith(prefix):
                return key[len(prefix):]
        return key

    def on_epoch_begin(self, epoch, logs=None):
        self._epoch_start = time.time()
        parts = []
        for name, ci in self.labels:
            c = self._COLORS[ci % len(self._COLORS)]
            parts.append(f"{c}{self._BOLD}{name}{self._RESET}")
        label = " → ".join(parts)
        cumul = self.cumulative_offset + epoch + 1
        cumul_str = f" (Total Epoch: {cumul}/{self.cumulative_total})" if self.cumulative_total else ""
        epoch_str = f" Phase Epoch: {epoch + 1}/{self.phase_epochs}" if self.phase_epochs else ""
        print(
            f"{self._RED}Phase {self.phase_number}. {self.phase}{self._RESET} "
            f"[{label}]"
            f"{self._RED}{epoch_str}{cumul_str}{self._RESET}"
        )

    def on_epoch_end(self, epoch, logs=None):
        elapsed = time.time() - self._epoch_start if self._epoch_start else 0
        if not logs:
            return

        loss_val = logs.get("loss")
        comp_metrics = {cn: {} for cn in self.component_order}

        for k, v in logs.items():
            if not isinstance(v, (int, float)) or k == "loss":
                continue
            for cn in self.component_order:
                if k.startswith(cn + "_"):
                    display = self._strip_component_prefix(k)
                    comp_metrics[cn][display] = v
                    break

        if loss_val is not None:
            print(f"→ loss={loss_val:.3f}")

        for cn in self.component_order:
            metrics = comp_metrics[cn]
            comp = self.model.components.get(cn) if hasattr(self.model, 'components') else None
            is_trainable = comp.trainable if comp else False

            if metrics:
                items = "  ".join(f"{k}={v:.3f}" for k, v in metrics.items())
                if is_trainable:
                    ci = self.component_order.index(cn)
                    c = self._COLORS[ci % len(self._COLORS)]
                    print(f"→ {c}{self._BOLD}{cn} (Training): {items}{self._RESET}")
                else:
                    print(f"→ {cn} (Frozen): {items}")
            elif is_trainable and self.debug and comp is not None:
                ci = self.component_order.index(cn)
                c = self._COLORS[ci % len(self._COLORS)]
                print(f"→ {c}{self._BOLD}{cn} (Training): (no metrics){self._RESET}")
            else:
                continue

            if self.debug and comp is not None:
                ci = self.component_order.index(cn)
                c = self._COLORS[ci % len(self._COLORS)] if is_trainable else ""
                tag = "(Training): " if is_trainable else "(Frozen): "
                self._print_debug_line(cn, comp, c, tag)

        print(f"→ {elapsed:.1f}s")

    def _print_debug_line(self, cn, comp, color, tag="(Training): "):
        """Print per-component weight debug info aligned under metrics."""
        indent = " " * (len("→ ") + len(cn) + len(tag) + 1)

        parts = []
        try:
            gmm = self.model._gmm_kl_weights.get(cn)
            parts.append(f"GMM weight={gmm.numpy():.2f}" if gmm is not None else "GMM weight=N/A")
        except Exception:
            parts.append("GMM weight=N/A")

        try:
            std = self.model._standard_kl_weights.get(cn)
            parts.append(f"KL weight={std.numpy():.2f}" if std is not None else "KL weight=N/A")
        except Exception:
            parts.append("KL weight=N/A")

        try:
            dw = self.model._data_loss_weights.get(cn, {})
            data_val = sum(float(v.numpy()) for v in dw.values()) if dw else 0.0
            parts.append(f"data weight={data_val:.2f}")
        except Exception:
            parts.append("data weight=N/A")

        try:
            ps = comp.hierarchical_encoder.progressive_scaler
            cur = float(ps.current_iteration.numpy())
            tot = float(ps.total_iterations.numpy())
            alpha = min(cur / max(tot, 1e-7), 1.0)
            parts.append(f"\u03b1\u00b2={(alpha * alpha):.2f}")
        except Exception:
            parts.append("\u03b1\u00b2=N/A")

        print(f"{color}{indent}{'  '.join(parts)}{self._RESET if color else ''}")
