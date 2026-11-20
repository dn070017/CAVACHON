from dataclasses import dataclass
from typing import Any, List, Tuple

# KL annealing stages used when a component sets no kl_annealing_ratio:
# standard KL only, crossfade to the GMM KL, GMM KL only.
DEFAULT_KL_ANNEALING_RATIO: Tuple[float, float, float] = (0.5, 0.2, 0.3)


@dataclass(frozen=True)
class ComponentSchedule:
    """ComponentSchedule

    Training settings of one component, resolved once from its
    ComponentConfig (defaults filled in) and the model, so the
    scheduler can decide which phases to run without looking the
    config up repeatedly.

    Attributes
    ----------
    name: str
        component name.

    order: int
        position of the component in the training order.

    parents: List[str]
        components the component is conditioned on (z, then z_hat).

    n_parent_annealing_epochs: int
        epochs of the parent annealing phase (children only).

    n_kl_annealing_epochs: int
        epochs of the KL annealing phase.

    kl_annealing_ratios: Tuple[float, float, float]
        fractions of the KL annealing phase spent on the standard KL,
        the crossfade and the GMM KL.

    enable_kmeans_init: bool
        whether the GMM priors are initialized with k-means.

    n_regular_training_epochs: int
        epochs of the regular (GMM) training phase.

    n_z_hat_annealing_epochs: int
        epochs of the z_hat annealing phase.

    n_z_hat_training_epochs: int
        epochs of the z_hat training phase.

    has_z_hat_prior: bool
        the component learns a z_hat GMM prior.

    has_z_hat_density_loss: bool
        the model compiled a z_hat density loss for the component.

    z_hat_annealing_enabled: bool
        the z_hat annealing phase runs for the component.

    """

    name: str
    order: int
    parents: List[str]
    n_parent_annealing_epochs: int
    n_kl_annealing_epochs: int
    kl_annealing_ratios: Tuple[float, float, float]
    enable_kmeans_init: bool
    n_regular_training_epochs: int
    n_z_hat_annealing_epochs: int
    n_z_hat_training_epochs: int
    has_z_hat_prior: bool
    has_z_hat_density_loss: bool
    z_hat_annealing_enabled: bool

    @classmethod
    def from_config(
        cls,
        component_config: Any,
        order: int,
        model: Any,
        default_epochs: int,
    ) -> "ComponentSchedule":
        """Resolve the training settings of a component.

        Parameters
        ----------
        component_config: ComponentConfig
            config of the component.

        order: int
            position of the component in the training order.

        model: cavachon.model.Model
            the model, used to check the z_hat prior and its losses.

        default_epochs: int
            regular training epochs when the config sets none (the
            ``epochs`` passed to ``fit``).

        Returns
        -------
        ComponentSchedule
            the resolved settings.
        """
        name = component_config.name
        component = model.components.get(name)
        has_z_hat_prior = bool(
            getattr(component, "learn_z_hat_priors", False)
            and getattr(component, "z_hat_prior_parameterizer", None) is not None
        )
        n_z_hat_annealing_epochs = (
            getattr(component_config, "n_z_hat_annealing_epochs", None) or 0
        )
        return cls(
            name=name,
            order=order,
            parents=list(component_config.conditioned_on_z)
            + list(component_config.conditioned_on_z_hat),
            n_parent_annealing_epochs=component_config.n_parent_annealing_epochs,
            n_kl_annealing_epochs=component_config.n_kl_annealing_epochs,
            kl_annealing_ratios=component_config.kl_annealing_ratio
            or DEFAULT_KL_ANNEALING_RATIO,
            enable_kmeans_init=component_config.enable_kmeans_init,
            n_regular_training_epochs=component_config.max_regular_training_epochs
            or default_epochs,
            n_z_hat_annealing_epochs=n_z_hat_annealing_epochs,
            n_z_hat_training_epochs=(
                getattr(component_config, "n_z_hat_training_epochs", None) or 0
            ),
            has_z_hat_prior=has_z_hat_prior,
            has_z_hat_density_loss=has_z_hat_prior
            and name in getattr(model, "_z_hat_density_weights", {}),
            z_hat_annealing_enabled=bool(
                n_z_hat_annealing_epochs > 0
                and has_z_hat_prior
                and name in getattr(model, "_z_hat_gmm_kl_weights", {})
            ),
        )

    @property
    def runs_parent_annealing(self) -> bool:
        """Whether the parent annealing phase runs."""
        return bool(self.parents) and self.n_parent_annealing_epochs > 0

    @property
    def runs_kl_annealing(self) -> bool:
        """Whether the KL annealing phase runs."""
        return self.n_kl_annealing_epochs > 0

    @property
    def runs_z_hat_training(self) -> bool:
        """Whether the z_hat training phase runs (after z_hat
        annealing)."""
        return self.z_hat_annealing_enabled and self.n_z_hat_training_epochs > 0
