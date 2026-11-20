import math
import warnings
from typing import Any, Callable, List, Mapping, Optional, Tuple

import numpy as np

from cavachon.environment.constants import Constants
from cavachon.scheduler.component_schedule import ComponentSchedule
from cavachon.scheduler.gmm_prior_initializer import GMMPriorInitializer
from cavachon.scheduler.training_phase import Schedule, TrainingPhase
from cavachon.scheduler.training_state_controller import (
    STANDARD_KL_WEIGHT,
    TrainingStateController,
)


class PhasePlanner:
    """PhasePlanner

    Turns the training settings of a component into the phases it is
    trained with, in order:

    1. Parent Annealing (children with parents): the child z enters
       z_hat through alpha, ramped from 0 to 1 together with the
       standard KL, while the parents are frozen.
    2. KL Annealing: standard KL, crossfade, then GMM KL; k-means
       initializes the GMM prior at the end of the standard-KL stage.
    3. Regular Training: GMM KL, with early stopping.
    4. Z_hat Annealing (learned z_hat prior): the encoder path is
       frozen, z_hat is sampled, and the z_hat GMM KL is ramped up.
    5. Z_hat Training: the z_hat GMM KL at its target, with early
       stopping.

    Attributes
    ----------
    model: cavachon.model.Model
        the model being trained.

    state: TrainingStateController
        sets loss weights, trainable flags, alpha and samplers.

    prior_initializer: GMMPriorInitializer
        initializes the GMM priors with k-means.

    compile_model: Callable[[], None]
        recompiles the model (after the trainable variables change).

    z_hat_init_std_ratio: float
        initial std of z_hat at the start of the z_hat annealing phase,
        relative to the std of the deterministic z_hat in each
        dimension.

    """

    def __init__(
        self,
        model: Any,
        state: TrainingStateController,
        prior_initializer: GMMPriorInitializer,
        compile_model: Callable[[], None],
        z_hat_init_std_ratio: float,
    ):
        """Constructor for PhasePlanner.

        Parameters
        ----------
        model: cavachon.model.Model
            the model being trained.

        state: TrainingStateController
            sets loss weights, trainable flags, alpha and samplers.

        prior_initializer: GMMPriorInitializer
            initializes the GMM priors with k-means.

        compile_model: Callable[[], None]
            recompiles the model.

        z_hat_init_std_ratio: float
            initial std of z_hat at the start of the z_hat annealing
            phase, relative to the std of the deterministic z_hat.
        """
        self.model = model
        self.state = state
        self.prior_initializer = prior_initializer
        self.compile_model = compile_model
        self.z_hat_init_std_ratio = z_hat_init_std_ratio

    def plan(self, schedule: ComponentSchedule, n_batches: int) -> List[TrainingPhase]:
        """Return the phases a component is trained with, in order.

        Parameters
        ----------
        schedule: ComponentSchedule
            training settings of the component.

        n_batches: int
            batches per epoch.

        Returns
        -------
        List[TrainingPhase]
            the phases of the component.
        """
        phases: List[TrainingPhase] = []
        if schedule.runs_parent_annealing:
            phases.append(self.parent_annealing(schedule, n_batches))
        if schedule.runs_kl_annealing:
            phases.append(self.kl_annealing(schedule))
        phases.append(self.regular_training(schedule))
        if schedule.z_hat_annealing_enabled:
            phases.append(self.z_hat_annealing(schedule))
            if schedule.runs_z_hat_training:
                phases.append(self.z_hat_training(schedule))
        return phases

    # ------------------------------------------------------------------
    # phases
    # ------------------------------------------------------------------

    def parent_annealing(
        self, schedule: ComponentSchedule, n_batches: int
    ) -> TrainingPhase:
        """Parent annealing phase.

        The child z enters z_hat through the progressive scaler, whose
        alpha grows from 0 to 1 over the phase (per batch); the parents
        enter at full strength. The child loss is the plain VAE loss
        (data weight 1, standard KL) except that the standard KL weight
        follows the same alpha: the reconstruction gradient reaching the
        child encoder scales with alpha, so a constant KL would dominate
        while alpha is small and collapse the child z. Parents are
        frozen with zero loss weights. No k-means initialization here:
        the GMM priors are initialized in the KL annealing phase (or the
        regular training phase without KL annealing).
        """
        name = schedule.name
        parents = schedule.parents
        n_iterations = n_batches * schedule.n_parent_annealing_epochs

        def setup() -> None:
            self.state.set_loss_weights(
                name,
                data_scale=1.0,
                gmm_kl=0.0,
                standard_kl=0.0,
                z_hat_density=0.0,
                z_hat_gmm_kl=0.0,
            )
            self.state.start_alpha_ramp(name, n_iterations)
            # standard KL weight = scale × alpha, advanced per batch
            # together with alpha by AnnealingCallback.on_batch_end
            self.state.start_standard_kl_ramp(name, n_iterations)

        def teardown() -> None:
            self.state.pin_alpha(name)
            self.state.pin_standard_kl(name, STANDARD_KL_WEIGHT)
            self.state.zero_loss_weights(parents)

        return TrainingPhase(
            name="Parent Annealing",
            run_name=(
                f"Training/{schedule.order}/ParentAnnealing/"
                f"{'-'.join(parents)}_to_{name}"
            ),
            component_name=name,
            epochs=schedule.n_parent_annealing_epochs,
            labels=[(parent, 0) for parent in parents] + [(name, 1)],
            loss_prefixes={**{parent: 0 for parent in parents}, name: 1},
            reset_before=True,
            setup=setup,
            teardown=teardown,
            schedule=self.constant_schedule,
        )

    def kl_annealing(self, schedule: ComponentSchedule) -> TrainingPhase:
        """KL annealing phase: standard KL, crossfade to the GMM KL,
        then GMM KL only. K-means initializes the GMM prior at the end
        of the standard-KL stage, while the GMM KL weight is still 0
        (when that stage fills the whole phase, k-means runs at the
        start of the regular training phase instead). No early
        stopping, since the loss changes by design."""
        name = schedule.name
        n_epochs = schedule.n_kl_annealing_epochs
        ratios = schedule.kl_annealing_ratios
        kmeans_epoch = self.kl_kmeans_epoch(schedule)
        if kmeans_epoch == 0:
            warnings.warn(
                f"Kmeans initialization epoch is 0 for component "
                f"'{name}': ratio[0] ({ratios[0]}) × "
                f"n_kl_annealing_epochs ({n_epochs}) = 0. "
                f"Kmeans will fire at epoch 0, which may effectively "
                f"disable initialization if ratio[0] is too small."
            )

        def setup() -> None:
            self.state.set_loss_weights(
                name, data_scale=1.0, gmm_kl=1.0, standard_kl=0.0, z_hat_density=0.0
            )

        return TrainingPhase(
            name="KL Annealing",
            run_name=f"Training/{schedule.order}/KLAnnealing/{name}",
            component_name=name,
            epochs=n_epochs,
            labels=[(name, schedule.order)],
            loss_prefixes={name: schedule.order},
            reset_before=True,
            setup=setup,
            schedule=self.make_kl_schedule(name, n_epochs, ratios),
            kmeans_epoch=kmeans_epoch,
            z_hat_kmeans_epoch=(
                kmeans_epoch if self._initializes_z_hat_prior_early(schedule) else None
            ),
        )

    def regular_training(self, schedule: ComponentSchedule) -> TrainingPhase:
        """Regular training phase with the GMM KL and early stopping.
        K-means initializes the GMM prior at its start when the KL
        annealing phase did not (no KL annealing, or a standard-KL
        stage that fills the whole KL annealing phase)."""
        name = schedule.name
        if schedule.enable_kmeans_init and not schedule.runs_kl_annealing:
            warnings.warn(
                f"Kmeans initialization is enabled but KL annealing is "
                f"disabled for component '{name}'. The latent "
                f"space may not be structured by standard KL "
                f"regularization before kmeans fires, which could result "
                f"in poor cluster priors. Consider enabling KL annealing "
                f"for better initialization quality."
            )
        run_kmeans = (
            schedule.enable_kmeans_init and self.kl_kmeans_epoch(schedule) is None
        )

        def setup() -> None:
            # GMMDensityLoss is kept but disabled: z_hat is regularized
            # by the z_hat GMM KL in the z_hat annealing phase instead.
            self.state.set_loss_weights(
                name,
                data_scale=1.0,
                gmm_kl=1.0,
                standard_kl=0.0,
                z_hat_density=0.0,
                z_hat_gmm_kl=0.0,
            )

        return TrainingPhase(
            name="Regular Training",
            run_name=f"Training/{schedule.order}/{name}",
            component_name=name,
            epochs=schedule.n_regular_training_epochs,
            labels=[(name, schedule.order)],
            loss_prefixes={name: schedule.order},
            reset_before=True,
            setup=setup,
            schedule=self.constant_schedule,
            kmeans_epoch=0 if run_kmeans else None,
            z_hat_kmeans_epoch=(
                0
                if run_kmeans and self._initializes_z_hat_prior_early(schedule)
                else None
            ),
            early_stopping=True,
        )

    def z_hat_annealing(self, schedule: ComponentSchedule) -> TrainingPhase:
        """Z_hat annealing phase.

        The z_hat prior is initialized with k-means on the
        deterministic z_hat learned so far, and z_hat starts with a
        small constant std relative to its spread. The encoder path
        (preprocessor, encoder and z prior) is frozen so z keeps what
        it learned. z_hat is sampled, and the hierarchical encoder, the
        decoders and the z_hat prior are trained while the z_hat GMM KL
        weight is ramped from 0 to ``z_hat_kl_weight``. No early
        stopping, since the loss rises with the KL weight by design.
        The frozen encoder path and the stochastic z_hat are kept for
        the z_hat training phase.
        """
        name = schedule.name
        n_epochs = schedule.n_z_hat_annealing_epochs
        target_weight = float(self.model._z_hat_gmm_kl_weights[name].scale)
        z_hat_kl_name = f"{name}_{Constants.MODEL_LOSS_Z_HAT_GMM_KL_POSTFIX}"

        def setup() -> None:
            z_hat = self.prior_initializer.initialize(name, target="z_hat")
            self.model.components[name].hierarchical_encoder.initialize_scale_diag(
                self.z_hat_init_std_ratio * np.std(z_hat, axis=0)
            )
            self.state.freeze_encoder_path(name)
            self.state.set_z_hat_sampler_stochastic(name)
            self.state.set_loss_weights(
                name,
                data_scale=1.0,
                gmm_kl=1.0,
                standard_kl=0.0,
                z_hat_density=0.0,
                z_hat_gmm_kl=0.0,
            )
            self.compile_model()

        # epoch is 0-based, so (epoch + 1) / n makes the last annealing
        # epoch use the full target weight (e.g. 0.25, 0.5, 0.75, 1.0).
        def ramp(epoch: int) -> Mapping[str, float]:
            progress = min(1.0, (epoch + 1) / n_epochs)
            return {z_hat_kl_name: target_weight * progress}

        return TrainingPhase(
            name="Z_hat Annealing",
            run_name=f"Z_hat Annealing/{schedule.order}/{name}",
            component_name=name,
            epochs=n_epochs,
            labels=[(name, schedule.order)],
            loss_prefixes={name: schedule.order},
            setup=setup,
            schedule=ramp,
        )

    def z_hat_training(self, schedule: ComponentSchedule) -> TrainingPhase:
        """Z_hat training phase: continues from the z_hat annealing
        phase (encoder path frozen, stochastic z_hat) with the z_hat GMM
        KL weight fixed at ``z_hat_kl_weight``, using early stopping."""
        name = schedule.name
        target_weight = float(self.model._z_hat_gmm_kl_weights[name].scale)

        def setup() -> None:
            self.state.set_loss_weights(
                name,
                data_scale=1.0,
                gmm_kl=1.0,
                standard_kl=0.0,
                z_hat_density=0.0,
                z_hat_gmm_kl=target_weight,
            )

        return TrainingPhase(
            name="Z_hat Training",
            run_name=f"Z_hat Training/{schedule.order}/{name}",
            component_name=name,
            epochs=schedule.n_z_hat_training_epochs,
            labels=[(name, schedule.order)],
            loss_prefixes={name: schedule.order},
            setup=setup,
            early_stopping=True,
        )

    # ------------------------------------------------------------------
    # schedules
    # ------------------------------------------------------------------

    @staticmethod
    def constant_schedule(epoch: int) -> Mapping[str, float]:
        """Schedule that changes no loss weight."""
        return {}

    @staticmethod
    def make_kl_schedule(
        component_name: str,
        total_epochs: int,
        kl_annealing_ratios: Tuple[float, float, float],
    ) -> Schedule:
        """Return the KL annealing schedule: standard KL only for the
        first ``ratio[0]`` of the epochs, a linear crossfade to the GMM
        KL for the next ``ratio[1]``, then GMM KL only."""
        standard_kl = f"{component_name}_{Constants.MODEL_LOSS_STANDARD_KL_POSTFIX}"
        gmm_kl = f"{component_name}_{Constants.MODEL_LOSS_GMM_KL_POSTFIX}"
        # GMMDensityLoss disabled (replaced by the z_hat GMM KL)
        density = f"{component_name}_{Constants.MODEL_LOSS_Z_HAT_GMM_DENSITY_POSTFIX}"
        standard_kl_end = PhasePlanner.epoch_boundary(
            total_epochs, kl_annealing_ratios[0]
        )
        gmm_kl_start = PhasePlanner.epoch_boundary(
            total_epochs, kl_annealing_ratios[0] + kl_annealing_ratios[1]
        )

        def schedule(epoch: int) -> Mapping[str, float]:
            if epoch < standard_kl_end:
                return {standard_kl: STANDARD_KL_WEIGHT, gmm_kl: 0.0, density: 0.0}
            if epoch < gmm_kl_start:
                t = (epoch - standard_kl_end) / (gmm_kl_start - standard_kl_end)
                return {
                    standard_kl: STANDARD_KL_WEIGHT * (1.0 - t),
                    gmm_kl: 1.0 * t,
                    density: 0.0,
                }
            return {standard_kl: 0.0, gmm_kl: 1.0, density: 0.0}

        return schedule

    @staticmethod
    def epoch_boundary(total_epochs: int, fraction: float) -> int:
        """First epoch after the first ``fraction`` of ``total_epochs``
        (floor of their product). The product is rounded first, so
        floating-point error such as 10 * (0.1 + 0.7) = 7.999... still
        gives 8."""
        return math.floor(round(total_epochs * fraction, 6))

    @staticmethod
    def kl_kmeans_epoch(schedule: ComponentSchedule) -> Optional[int]:
        """Epoch of the KL annealing phase at which k-means initializes
        the GMM prior: the end of the standard-KL stage. None when
        k-means is disabled, there is no KL annealing, or the
        standard-KL stage fills the whole phase (k-means then runs at
        the start of the regular training phase)."""
        if not (schedule.enable_kmeans_init and schedule.runs_kl_annealing):
            return None
        epoch = PhasePlanner.epoch_boundary(
            schedule.n_kl_annealing_epochs, schedule.kl_annealing_ratios[0]
        )
        return epoch if epoch < schedule.n_kl_annealing_epochs else None

    @staticmethod
    def _initializes_z_hat_prior_early(schedule: ComponentSchedule) -> bool:
        """The z_hat prior is initialized once: at the start of the
        z_hat annealing phase (on the z_hat learned so far) if it runs,
        otherwise together with the z prior in the KL annealing or
        regular training phase."""
        return (
            schedule.has_z_hat_density_loss
            and schedule.enable_kmeans_init
            and not schedule.z_hat_annealing_enabled
        )
