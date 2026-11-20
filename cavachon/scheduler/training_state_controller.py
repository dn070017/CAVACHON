from typing import Any, Dict, List, Mapping, Optional, Tuple

from cavachon.environment.constants import Constants

# maximum weight of the standard KL (the ELBO, β = 1)
STANDARD_KL_WEIGHT = 1.0


class TrainingStateController:
    """TrainingStateController

    Sets the training state of the model between and within phases:
    the loss weights of each component, which components (and which
    of their modules) are trainable, the alpha of the hierarchical
    encoders and the stochastic flags of the samplers. It also keeps
    the progressive scalers of the loss weights across recompilation.

    Loss weights are either ProgressiveScaler (pinned with ``pin_to``)
    or plain Variables (``assign``).

    Attributes
    ----------
    model: cavachon.model.Model
        the model being trained.

    modality_weight: Mapping[str, Mapping[str, float]]
        weight of the data loss of each modality, by component.

    """

    def __init__(
        self,
        model: Any,
        modality_weight: Mapping[str, Mapping[str, float]],
    ):
        """Constructor for TrainingStateController.

        Parameters
        ----------
        model: cavachon.model.Model
            the model being trained.

        modality_weight: Mapping[str, Mapping[str, float]]
            weight of the data loss of each modality, by component.
        """
        self.model = model
        self.modality_weight = modality_weight

    # ------------------------------------------------------------------
    # loss weights
    # ------------------------------------------------------------------

    def set_loss_weights(
        self,
        component_name: str,
        data_scale: float = 1.0,
        gmm_kl: float = 1.0,
        standard_kl: float = 0.0,
        z_hat_density: Optional[float] = None,
        z_hat_gmm_kl: Optional[float] = None,
    ) -> None:
        """Set every loss weight of one component.

        Parameters
        ----------
        component_name: str
            component name.

        data_scale: float, optional
            scale applied to the modality weights of the data losses.
            Use 0.0 to switch the data losses off. Defaults to 1.0.

        gmm_kl: float, optional
            weight of the GMM KL divergence. Defaults to 1.0.

        standard_kl: float, optional
            weight of the standard KL divergence. Defaults to 0.0.

        z_hat_density: float, optional
            if not None, weight of the z_hat GMM density loss (only
            exists when ``learn_z_hat_priors`` is enabled).

        z_hat_gmm_kl: float, optional
            if not None, weight of the z_hat GMM KL divergence (only
            exists when ``learn_z_hat_priors`` is enabled).
        """
        model = self.model
        weights = getattr(model, "_gmm_kl_weights", {})
        if component_name in weights:
            self._assign(weights[component_name], gmm_kl)
        weights = getattr(model, "_standard_kl_weights", {})
        if component_name in weights:
            self._assign(weights[component_name], standard_kl)
        weights = getattr(model, "_z_hat_density_weights", {})
        if z_hat_density is not None and component_name in weights:
            self._assign(weights[component_name], z_hat_density)
        weights = getattr(model, "_z_hat_gmm_kl_weights", {})
        if z_hat_gmm_kl is not None and component_name in weights:
            weights[component_name].pin_to(float(z_hat_gmm_kl))
        data_weights = model._data_loss_weights.get(component_name, {})
        for modality_name, weight in data_weights.items():
            modality_weight = self.modality_weight.get(component_name, {}).get(
                modality_name, 1.0
            )
            self._assign(weight, modality_weight * float(data_scale))

    def zero_loss_weights(self, component_names: List[str]) -> None:
        """Set every loss weight of the given components to 0."""
        for component_name in component_names:
            self.set_loss_weights(
                component_name,
                data_scale=0.0,
                gmm_kl=0.0,
                standard_kl=0.0,
                z_hat_density=0.0,
                z_hat_gmm_kl=0.0,
            )

    def start_standard_kl_ramp(self, component_name: str, n_iterations: int) -> None:
        """Ramp the standard KL weight from 0 to its scale over
        ``n_iterations`` batches (advanced by AnnealingCallback)."""
        self.model._standard_kl_weights[component_name].activate(n_iterations)

    def pin_standard_kl(self, component_name: str, value: float) -> None:
        """Pin the standard KL weight of a component to ``value``."""
        self.model._standard_kl_weights[component_name].pin_to(value)

    @staticmethod
    def _assign(weight: Any, value: float) -> None:
        """Pin a ProgressiveScaler, or assign a plain Variable."""
        if hasattr(weight, "pin_to"):
            weight.pin_to(float(value))
        else:
            weight.assign(float(value))

    # ------------------------------------------------------------------
    # trainable flags
    # ------------------------------------------------------------------

    def train_only(self, train_components: List[str]) -> None:
        """Make the given components (and all their modules) trainable
        and freeze every other component. The loss weights of the
        trained components are reset to the regular training values
        (data 1, GMM KL 1); those of frozen components are zeroed. All
        alphas are pinned to 1.

        Parameters
        ----------
        train_components: List[str]
            component names to train.
        """
        for config in self.model.component_configs:
            component_name = config.name
            component = self.model.components.get(component_name)
            should_train = component_name in train_components
            component.trainable = should_train
            self.pin_alpha(component_name)
            self.set_trainable_recursively(component, should_train)
            self.set_loss_weights(
                component_name,
                data_scale=1.0 if should_train else 0.0,
                gmm_kl=1.0 if should_train else 0.0,
                standard_kl=0.0,
                z_hat_density=0.0,  # GMMDensityLoss is disabled
                z_hat_gmm_kl=0.0,  # enabled only in the z_hat annealing phase
            )

    def freeze_encoder_path(self, component_name: str) -> None:
        """Freeze the preprocessor, encoder and z prior of a component.

        The hierarchical encoder, decoders and z_hat prior stay
        trainable. The samplers are not affected: whether they sample
        is controlled by their stochastic flag.
        """
        component = self.model.components.get(component_name)
        for module in (
            component.preprocessor,
            component.encoder,
            component.z_prior_parameterizer,
        ):
            module.trainable = False

    @staticmethod
    def set_trainable_recursively(layer: Any, trainable: bool) -> None:
        """Set ``trainable`` on a layer and every nested layer, reached
        through ``submodules`` and ``layers``."""
        visited = set()

        def visit(current: Any) -> None:
            if id(current) in visited:
                return
            visited.add(id(current))
            if hasattr(current, "trainable"):
                current.trainable = trainable
            for sub in getattr(current, "submodules", []):
                visit(sub)
            for sub in getattr(current, "layers", []):
                visit(sub)

        visit(layer)

    # ------------------------------------------------------------------
    # alpha and samplers
    # ------------------------------------------------------------------

    def start_alpha_ramp(self, component_name: str, n_iterations: int) -> None:
        """Ramp the alpha of a component's hierarchical encoder from 0
        to 1 over ``n_iterations`` batches."""
        self.model.components[component_name].set_progressive_scaler_iteration(
            0.0, float(n_iterations)
        )

    def pin_alpha(self, component_name: str) -> None:
        """Pin the alpha of a component's hierarchical encoder to 1."""
        self.model.components[component_name].set_progressive_scaler_iteration(1.0, 1.0)

    def set_samplers_stochastic(self, z: bool, z_hat: bool) -> None:
        """Set the stochastic flag of the z and z_hat samplers of every
        component. Samplers sample in training mode regardless of
        whether their component is frozen, so a frozen parent passes
        sampled z (and z_hat once it is stochastic) to its children."""
        for component in self.model.components.values():
            for attr, value in (("z_sampler", z), ("z_hat_sampler", z_hat)):
                sampler = getattr(component, attr, None)
                stochastic = getattr(sampler, "stochastic", None)
                if stochastic is not None:
                    stochastic.assign(1.0 if value else 0.0)

    def set_z_hat_sampler_stochastic(self, component_name: str) -> None:
        """Make the z_hat sampler of one component stochastic."""
        self.model.components[component_name].z_hat_sampler.stochastic.assign(1.0)

    # ------------------------------------------------------------------
    # recompilation
    # ------------------------------------------------------------------

    def compile_loss_weights(self) -> Dict[str, Any]:
        """Return the loss-weight arguments of ``Model.compile``, read
        from the scales of the existing loss weights, so recompiling
        keeps them. The z_hat density weights keep their current
        values.

        Returns
        -------
        Dict[str, Any]
            ``standard_kl_weights``, ``gmm_kl_weights`` and, when any
            data loss exists, ``loss_weights``.
        """
        model = self.model
        standard_kl_weights: Dict[str, float] = {}
        gmm_kl_weights: Dict[str, float] = {}
        data_weights: Dict[str, float] = {}
        for config in model.component_configs:
            name = config.name
            weights = getattr(model, "_standard_kl_weights", {})
            standard_kl_weights[name] = (
                float(weights[name].scale) if name in weights else STANDARD_KL_WEIGHT
            )
            weights = getattr(model, "_gmm_kl_weights", {})
            gmm_kl_weights[name] = (
                float(weights[name].scale) if name in weights else 1.0
            )
            data = getattr(model, "_data_loss_weights", {}).get(name, {})
            for modality_name, weight in data.items():
                if self._is_scaler(weight):
                    key = f"{name}_{modality_name}_{Constants.MODEL_LOSS_DATA_POSTFIX}"
                    data_weights[key] = float(weight.scale)
            weights = getattr(model, "_z_hat_density_weights", {})
            if name in weights:
                weight = weights[name]
                value = (
                    float(weight.scale)
                    if self._is_scaler(weight)
                    else float(weight.numpy())
                )
                weight.assign(value)

        compile_kwargs: Dict[str, Any] = dict(
            standard_kl_weights=standard_kl_weights,
            gmm_kl_weights=gmm_kl_weights,
        )
        if data_weights:
            compile_kwargs["loss_weights"] = data_weights
        return compile_kwargs

    def snapshot_scalers(self) -> List[Tuple[Tuple[str, ...], float, float]]:
        """Save (current, total) iterations of every progressive
        loss-weight scaler, so pinned or ramping weights survive
        recompilation."""
        return [
            (path, float(w.current_iteration), float(w.total_iterations))
            for path, w in self._scalers()
        ]

    def restore_scalers(
        self, snapshot: List[Tuple[Tuple[str, ...], float, float]]
    ) -> None:
        """Restore the iterations saved by ``snapshot_scalers``."""
        scalers = dict(self._scalers())
        for path, current, total in snapshot:
            weight = scalers.get(path)
            if weight is not None:
                weight.current_iteration.assign(current)
                weight.total_iterations.assign(total)

    def _scalers(self) -> List[Tuple[Tuple[str, ...], Any]]:
        """All progressive loss-weight scalers, keyed by
        (weight dict, component[, modality])."""
        model = self.model
        names = [config.name for config in model.component_configs]
        scalers: List[Tuple[Tuple[str, ...], Any]] = []
        for attr in (
            "_standard_kl_weights",
            "_gmm_kl_weights",
            "_z_hat_density_weights",
        ):
            weights = getattr(model, attr, {})
            for name in names:
                if name in weights and self._is_scaler(weights[name]):
                    scalers.append(((attr, name), weights[name]))
        for name, data in getattr(model, "_data_loss_weights", {}).items():
            for modality_name, weight in data.items():
                if self._is_scaler(weight):
                    scalers.append(
                        (("_data_loss_weights", name, modality_name), weight)
                    )
        return scalers

    @staticmethod
    def _is_scaler(weight: Any) -> bool:
        """Whether a loss weight is a ProgressiveScaler."""
        return hasattr(weight, "current_iteration") and hasattr(weight, "pin_to")
