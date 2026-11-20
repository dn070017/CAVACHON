import itertools
import warnings
from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
from functools import partial
from typing import Any, Dict, Iterator, List, Mapping, Optional, Union

import mlflow
import tensorflow as tf

from cavachon.callbacks import (
    AnnealingCallback,
    EarlyStoppingCallback,
    VerboseCallback,
)
from cavachon.config.training_config import EarlyStoppingConfig
from cavachon.utils.general_utils import GeneralUtils
from cavachon.scheduler.component_schedule import ComponentSchedule
from cavachon.scheduler.gmm_prior_initializer import GMMPriorInitializer
from cavachon.scheduler.phase_planner import PhasePlanner
from cavachon.scheduler.training_phase import PhaseProgress, TrainingPhase
from cavachon.scheduler.training_state_controller import (
    STANDARD_KL_WEIGHT as STANDARD_KL_WEIGHT,
)
from cavachon.scheduler.training_state_controller import TrainingStateController


class SequentialTrainingScheduler:
    """SequentialTrainingScheduler

    Training scheduler that trains the components one at a time, in
    dependency order, each through a sequence of phases (see
    PhasePlanner): parent annealing, KL annealing, regular training,
    z_hat annealing and z_hat training. Components that are not being
    trained are frozen with zero loss weights.

    Attributes
    ----------
    model : tf.keras.Model
        input model that needs to be trained.

    component_configs: List[ComponentConfig]
        the config used to create the components in the model.

    optimizer: str
        optimizer used to train the model (only the attributes of the
        optimizer will be used)

    early_stopping: Union[bool, EarlyStoppingConfig]
        early stopping settings (EarlyStoppingConfig), or False if early
        stopping is disabled.

    training_order: List[List[str]]
        the training order of the components, each element is a list
        of the components trained in the corresponding stage (see
        compute_component_training_order).

    component_order: List[str]
        the components in training order.

    modality_weight: Mapping[str, Mapping[str, float]]
        the weight of the data distribution by component. The keys are
        the component names, the values are the mapping where the keys
        are the modality names and the values are the weight.

    state: TrainingStateController
        sets loss weights, trainable flags, alpha and samplers.

    prior_initializer: GMMPriorInitializer
        initializes the GMM priors with k-means.

    planner: PhasePlanner
        builds the training phases of each component.

    z_hat_init_std_ratio: float
        class attribute, initial std of z_hat at the start of the z_hat
        annealing phase, relative to the std of the deterministic z_hat
        in each dimension.

    """

    z_hat_init_std_ratio: float = 0.1

    def __init__(
        self,
        model: tf.keras.Model,
        mdata,
        optimizer: str = "adam",
        learning_rate: float = 1e-4,
        early_stopping: Union[bool, EarlyStoppingConfig] = True,
        batch_size: int = 128,
        outdir: Optional[str] = None,
        distribution_names: Optional[Mapping[str, str]] = None,
        batch_effect_colnames: Optional[Mapping[str, List[str]]] = None,
    ):
        """Constructor for SequentialTrainingScheduler.

        Parameters
        ----------
        model: tf.keras.Model
            input model that needs to be trained.

        mdata: muon.MuData
            the training data, used for the k-means initialization.

        optimizer: tf.keras.optimizers.Optimizer, optional
            optimizer used to train the model (only the attributes of
            the optimizer will be used if provided with Optimizer).
            Defaults to 'adam'.

        learning_rate: float, optional
            learning rate for the optimizer. Defaults to 1e-4.

        early_stopping: Union[bool, EarlyStoppingConfig], optional
            early stopping settings for the regular (GMM) training phase
            and the z_hat training phase. True uses the defaults of
            EarlyStoppingConfig, False disables early stopping.

        batch_size: int, optional
            batch size of the k-means prediction. Defaults to 128.

        outdir, distribution_names, batch_effect_colnames: optional
            kept for the workflow's call; not used.

        """
        self.model = model
        self.mdata = mdata
        # ordered again here (Model.make already orders them), so the
        # scheduler also works on configs that are not ordered
        self.component_configs = GeneralUtils.order_components(
            self.model.component_configs
        )
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        if early_stopping is True:
            early_stopping = EarlyStoppingConfig()
        self.early_stopping = early_stopping
        self.training_order = self.compute_component_training_order()
        self.component_order = [stage[0] for stage in self.training_order]
        self.modality_weight = self.compute_modality_weight()
        self.batch_size = batch_size
        self.output_dir = outdir
        self.distribution_names = distribution_names
        self.batch_effect_colnames = batch_effect_colnames

        self.state = TrainingStateController(model, self.modality_weight)
        self.prior_initializer = GMMPriorInitializer(
            model, mdata, batch_size, self.component_order
        )
        self.planner = PhasePlanner(
            model,
            self.state,
            self.prior_initializer,
            self._compile_model,
            self.z_hat_init_std_ratio,
        )

    def compute_component_training_order(self) -> List[List[str]]:
        """Compute the training order of the components based on the
        order of topological sort of the input dependency graph.

        ``self.component_configs`` is topologically sorted by
        ``GeneralUtils.order_components`` (parents before children,
        config order kept otherwise; unknown parents raise ValueError
        and cycles AttributeError). Each component's stage is one more
        than its latest parent's.

        Design note: the order is a list of stages, each holding the
        components trained together. Components whose parents are
        already trained could share a stage, and an earlier version
        did train them jointly. That was dropped for two reasons:
        summing several components' losses in one stage made the loss
        normalization across components inconsistent, and joint stages
        needed noticeably more memory. Each stage now holds exactly one
        component (``[[a], [b], ...]``). The nested structure is kept
        so stages can hold several components again if needed.

        Returns
        -------
        List[List[str]]
            the training order of the components, each element is a
            list of the components trained in the corresponding stage.
        """
        training_order: List[List[str]] = [[]]
        component_order: Dict[str, int] = dict()
        for component_config in self.component_configs:
            component_name = component_config.name
            conditioned_on = itertools.chain(
                component_config.conditioned_on_z,
                component_config.conditioned_on_z_hat,
            )
            order = max([0] + [component_order[x] + 1 for x in conditioned_on])
            if order > len(training_order) - 1:
                training_order.append([])
            training_order[order].append(component_name)
            component_order.setdefault(component_name, order)

        # one component per stage (see the design note above)
        return [[x] for xs in training_order for x in xs]

    def compute_modality_weight(
        self, constant: bool = False
    ) -> Mapping[str, Mapping[str, float]]:
        """Compute the weight of the data distribution for each
        modality.

        Parameters
        ----------
        constant : bool, optional
            whether to give every modality the same weight (1.0)
            instead of weights inversely proportional to their number
            of variables (normalized to sum to 1). Defaults to False.

        Returns
        -------
        Mapping[str, Mapping[str, float]]
            the weight of the data distribution by component. The keys
            are the component names, the values are the mapping where
            the keys are the modality names and the values are the
            weight.
        """
        modality_weight_by_component: defaultdict[str, Dict[str, float]] = defaultdict(
            dict
        )
        for component_config in self.component_configs:
            component_name = component_config.name
            modality_weight: Dict[str, float] = dict()
            n_vars = component_config.n_vars
            if not constant:
                total_vars = 0
                total_scaled_weight = 0
                for modality_name, n_var in n_vars.items():
                    total_vars += n_var
                for modality_name, n_var in n_vars.items():
                    scaled_weight = total_vars / n_var
                    total_scaled_weight += scaled_weight
                    modality_weight.setdefault(modality_name, scaled_weight)
                for modality_name, scaled_weight in modality_weight.items():
                    modality_weight[modality_name] = scaled_weight / total_scaled_weight
            else:
                for modality_name in n_vars.keys():
                    modality_weight.setdefault(modality_name, 1.0)
            modality_weight_by_component.setdefault(component_name, modality_weight)

        return modality_weight_by_component

    def compute_component_schedules(
        self, default_epochs: int
    ) -> List[ComponentSchedule]:
        """Resolve the training settings of every component, in
        training order.

        Parameters
        ----------
        default_epochs: int
            regular training epochs of components whose config sets
            none.

        Returns
        -------
        List[ComponentSchedule]
            the settings of each component.
        """
        configs = {config.name: config for config in self.component_configs}
        return [
            ComponentSchedule.from_config(
                configs[name], order, self.model, default_epochs
            )
            for order, name in enumerate(self.component_order)
        ]

    # ==================================================================
    # Training
    # ==================================================================

    def fit(
        self,
        x: tf.data.Dataset,
        **kwargs,
    ) -> List[tf.keras.callbacks.History]:
        """Fit model with multi-phase hierarchical training.

        Each component is trained through the phases built by
        PhasePlanner; each phase is one ``model.fit`` call with its own
        loss weights, trainable modules and callbacks. Loss weights are
        Variables, so callbacks can adjust them without recompiling.

        Parameters
        ----------
        x: tf.data.Dataset
            input dataset created by DataLoader.

        **kwargs: Mapping[str, Any]
            additional arguments passed to self.model.fit. ``epochs``
            is the regular training length of components that set no
            ``max_regular_training_epochs``.

        Returns
        -------
        List[tf.keras.callbacks.History]
            history of model.fit in each phase.
        """
        n_batches = len(x)
        # z is sampled during the whole training; z_hat only from the
        # z_hat annealing phase of its component on. The flags are
        # stored in the weights, so they are reset here in case of
        # resumed training.
        self.state.set_samplers_stochastic(z=True, z_hat=False)
        # compile first: the z_hat losses checked by the schedules only
        # exist once the model is compiled
        self._compile_model()
        schedules = self.compute_component_schedules(kwargs.get("epochs", 100))
        self._warn_untrained_z_hat_priors(schedules)
        experiment = self._mlflow_experiment()

        phases = [
            phase
            for schedule in schedules
            for phase in self.planner.plan(schedule, n_batches)
        ]
        progress = PhaseProgress(total=sum(phase.epochs for phase in phases))
        history = [
            self._run_phase(phase, x, n_batches, progress, experiment, kwargs)
            for phase in phases
        ]

        # training is completed: z and z_hat are the posterior means.
        self.state.set_samplers_stochastic(z=False, z_hat=False)
        return history

    def _run_phase(
        self,
        phase: TrainingPhase,
        x: tf.data.Dataset,
        n_batches: int,
        progress: PhaseProgress,
        experiment: Any,
        kwargs: Mapping[str, Any],
    ) -> tf.keras.callbacks.History:
        """Run one phase as a ``model.fit`` call and advance the epoch
        book-keeping by the epochs it actually ran."""
        with self._mlflow_run(phase.run_name, experiment):
            if phase.reset_before:
                self.state.train_only([phase.component_name])
                self._compile_model()
            phase.setup()

            fit_kwargs = deepcopy(dict(kwargs))
            # callbacks are merged into the phase callbacks and verbose
            # is set explicitly, so they must not be passed twice.
            for key in ("epochs", "callbacks", "verbose"):
                fit_kwargs.pop(key, None)
            history = self.model.fit(
                x.take(n_batches * phase.epochs),
                epochs=phase.epochs,
                callbacks=self._phase_callbacks(phase, progress, kwargs),
                verbose=0,
                **fit_kwargs,
            )
        phase.teardown()
        progress.advance(phase.epochs, len(history.epoch))
        return history

    def _phase_callbacks(
        self,
        phase: TrainingPhase,
        progress: PhaseProgress,
        kwargs: Mapping[str, Any],
    ) -> List[tf.keras.callbacks.Callback]:
        """Callbacks of a phase: copies of the user callbacks, progress
        display, annealing (and k-means), and early stopping."""
        name = phase.component_name
        callbacks = deepcopy(list(kwargs.get("callbacks", [])))
        callbacks.append(
            VerboseCallback(
                labels=phase.labels,
                phase=phase.name,
                loss_prefixes=phase.loss_prefixes,
                phase_epochs=phase.epochs,
                cumulative_offset=progress.offset,
                cumulative_total=progress.total,
                phase_number=progress.phase_number,
                component_order=self.component_order,
            )
        )
        if phase.schedule is not None:
            callbacks.append(
                AnnealingCallback(
                    schedule=phase.schedule,
                    kmeans_epoch=phase.kmeans_epoch,
                    kmeans_init=(
                        partial(self.prior_initializer.initialize, name, target="z")
                        if phase.kmeans_epoch is not None
                        else None
                    ),
                    component_name=name,
                    component_names=self.component_order,
                )
            )
        if phase.z_hat_kmeans_epoch is not None:
            callbacks.append(
                AnnealingCallback(
                    schedule=PhasePlanner.constant_schedule,
                    kmeans_epoch=phase.z_hat_kmeans_epoch,
                    kmeans_init=partial(
                        self.prior_initializer.initialize, name, target="z_hat"
                    ),
                    component_name=name,
                    component_names=self.component_order,
                )
            )
        if phase.early_stopping and self.early_stopping:
            early_stopping = self.early_stopping
            callbacks.append(
                EarlyStoppingCallback(
                    cumulative_offset=progress.offset,
                    monitor=early_stopping.monitor,
                    mode=early_stopping.mode,
                    min_delta=early_stopping.min_delta,
                    patience=early_stopping.patience,
                    restore_best_weights=True,
                    verbose=0,
                )
            )
        return callbacks

    def _warn_untrained_z_hat_priors(self, schedules: List[ComponentSchedule]) -> None:
        """Warn about learned z_hat priors that no phase trains."""
        for schedule in schedules:
            if schedule.has_z_hat_prior and not schedule.z_hat_annealing_enabled:
                warnings.warn(
                    f"Component '{schedule.name}' has learn_z_hat_priors enabled "
                    "but no z_hat annealing epochs: its z_hat prior is only "
                    "initialized with k-means and is never trained. Set "
                    "n_z_hat_annealing_epochs > 0 to learn it."
                )

    # ==================================================================
    # Compilation and logging
    # ==================================================================

    def _compile_model(self) -> None:
        """Compile the model. Creates a new optimizer when the trainable
        variable set has changed, otherwise reuses the existing one so
        momentum is preserved within a component's phases.

        The progressive loss-weight scalers are saved before and
        restored after compiling, so pinned or ramping weights (e.g.
        zeroed parents) survive recompilation.
        """
        current_variables = (
            {id(v) for v in self.model.trainable_variables}
            if hasattr(self.model, "trainable_variables")
            else set()
        )
        if (
            getattr(self.model, "optimizer", None) is not None
            and getattr(self, "_last_trainable", None) == current_variables
        ):
            optimizer = self.model.optimizer
        else:
            optimizer = tf.keras.optimizers.get(self.optimizer).__class__(
                learning_rate=self.learning_rate
            )
        self._last_trainable = current_variables

        snapshot = self.state.snapshot_scalers()
        self.model.compile(optimizer=optimizer, **self.state.compile_loss_weights())
        self.state.restore_scalers(snapshot)

    def _mlflow_experiment(self) -> Any:
        """Set up MLflow experiment and return the experiment object."""
        experiment_name = self.model.name
        mlflow.set_experiment(experiment_name)
        return mlflow.get_experiment_by_name(experiment_name)

    @contextmanager
    def _mlflow_run(self, run_name: str, experiment: Any) -> Iterator[None]:
        """MLflow run with TensorFlow autologging, ended even if the
        phase raises."""
        mlflow.start_run(experiment_id=experiment.experiment_id, run_name=run_name)
        mlflow.tensorflow.autolog(
            log_every_epoch=True,
            log_models=False,
            checkpoint=False,
        )
        try:
            yield
        finally:
            mlflow.end_run()
