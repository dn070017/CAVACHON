from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Tuple

Schedule = Callable[[int], Mapping[str, float]]


@dataclass
class TrainingPhase:
    """TrainingPhase

    One ``model.fit`` call of the sequential training: what is trained,
    for how long, and which callbacks run.

    Attributes
    ----------
    name: str
        phase name shown by VerboseCallback (e.g. "KL Annealing").

    run_name: str
        name of the mlflow run.

    component_name: str
        the component trained in the phase.

    epochs: int
        planned number of epochs.

    labels: List[Tuple[str, int]]
        (component, level) pairs shown by VerboseCallback.

    loss_prefixes: Dict[str, int]
        component name -> level, used by VerboseCallback to group the
        losses.

    reset_before: bool
        train only the phase's component (TrainingStateController
        .train_only) and recompile before ``setup``.

    setup: Callable[[], None]
        sets the loss weights, alpha, frozen modules etc. of the phase.

    teardown: Callable[[], None]
        runs after ``model.fit``.

    schedule: Callable[[int], Mapping[str, float]], optional
        loss-name pattern -> weight for each epoch, applied by an
        AnnealingCallback. None means the phase has no
        AnnealingCallback.

    kmeans_epoch: int, optional
        epoch at which the z prior is initialized with k-means.

    z_hat_kmeans_epoch: int, optional
        epoch at which the z_hat prior is initialized with k-means
        (a second AnnealingCallback).

    early_stopping: bool
        whether the phase uses early stopping (when configured).

    """

    name: str
    run_name: str
    component_name: str
    epochs: int
    labels: List[Tuple[str, int]]
    loss_prefixes: Dict[str, int]
    reset_before: bool = False
    setup: Callable[[], None] = field(default=lambda: None)
    teardown: Callable[[], None] = field(default=lambda: None)
    schedule: Optional[Schedule] = None
    kmeans_epoch: Optional[int] = None
    z_hat_kmeans_epoch: Optional[int] = None
    early_stopping: bool = False


@dataclass
class PhaseProgress:
    """PhaseProgress

    Epoch book-keeping across phases, shown by VerboseCallback and
    EarlyStoppingCallback.

    Attributes
    ----------
    total: int
        total epochs of all phases; reduced when a phase stops early.

    offset: int
        epochs run before the current phase.

    phase_number: int
        1-based number of the current phase.

    """

    total: int
    offset: int = 0
    phase_number: int = 1

    def advance(self, planned_epochs: int, actual_epochs: int) -> None:
        """Move to the next phase after one that ran ``actual_epochs``
        of its ``planned_epochs``."""
        self.total -= planned_epochs - actual_epochs
        self.offset += actual_epochs
        self.phase_number += 1
