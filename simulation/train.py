"""Train (trainset) model and its status state machine.

Status vocabulary (from the config): Available, Maintenance, Inducted,
Running, Delayed. A separate ``location`` attribute (DEPOT / TRANSIT /
MAINLINE) removes the ambiguity of "Delayed", which can mean either a train
whose preparation overran in the depot or a running train held up in service.

Allowed transitions::

    Available   -> Inducted | Maintenance | Delayed
    Delayed     -> Inducted | Maintenance | Running | Available
    Maintenance -> Available
    Inducted    -> Running
    Running     -> Delayed | Available (withdrawn) | Maintenance (failure)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

from .exceptions import IllegalStateTransitionError, TrainNotReadyError

logger = logging.getLogger(__name__)


class TrainStatus(str, Enum):
    AVAILABLE = "Available"
    MAINTENANCE = "Maintenance"
    INDUCTED = "Inducted"
    RUNNING = "Running"
    DELAYED = "Delayed"


class TrainLocation(str, Enum):
    DEPOT = "DEPOT"
    TRANSIT = "TRANSIT"      # between depot and mainline (induction route / reception)
    MAINLINE = "MAINLINE"


ALLOWED_TRANSITIONS: dict[TrainStatus, tuple[TrainStatus, ...]] = {
    TrainStatus.AVAILABLE: (TrainStatus.INDUCTED, TrainStatus.MAINTENANCE, TrainStatus.DELAYED),
    TrainStatus.DELAYED: (TrainStatus.INDUCTED, TrainStatus.MAINTENANCE, TrainStatus.RUNNING,
                          TrainStatus.AVAILABLE),
    TrainStatus.MAINTENANCE: (TrainStatus.AVAILABLE,),
    TrainStatus.INDUCTED: (TrainStatus.RUNNING,),
    TrainStatus.RUNNING: (TrainStatus.DELAYED, TrainStatus.AVAILABLE, TrainStatus.MAINTENANCE),
}


@dataclass
class Train:
    """A single trainset.

    Attributes:
        train_id: unique identifier, e.g. ``TS07``.
        depot_location: stabling position, e.g. ``SL03-B``.
        readiness_time: second-of-day at which preparation completes.
        min_prep_time_s: minimum preparation time required before induction.
        maintenance_required: True if the train must not be inducted until
            maintenance is complete.
        induction_priority: 1 (highest) .. N (lowest).
        scheduled_induction_time: planned induction time (second-of-day) or
            None if the train is not in today's plan.
        km_since_maintenance: odometer since last maintenance (synthetic).
    """

    train_id: str
    depot_location: str
    readiness_time: int
    min_prep_time_s: int
    maintenance_required: bool = False
    induction_priority: int = 3
    scheduled_induction_time: int | None = None
    km_since_maintenance: float = 0.0
    status: TrainStatus = TrainStatus.AVAILABLE
    location: TrainLocation = TrainLocation.DEPOT
    history: list[tuple[int, str, str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.train_id:
            raise ValueError("train_id must be a non-empty string")
        if self.min_prep_time_s < 0 or self.readiness_time < 0:
            raise ValueError(f"{self.train_id}: times must be non-negative")
        if self.induction_priority < 1:
            raise ValueError(f"{self.train_id}: induction_priority must be >= 1")
        if self.maintenance_required and self.status is TrainStatus.AVAILABLE:
            self.status = TrainStatus.MAINTENANCE

    # ------------------------------------------------------------------ state
    def transition(self, new_status: TrainStatus, now: int,
                   location: TrainLocation | None = None) -> None:
        """Move to ``new_status``; raises IllegalStateTransitionError if not allowed."""
        new_status = TrainStatus(new_status)
        if new_status is self.status:
            if location is not None:
                self.location = location
            return
        if new_status not in ALLOWED_TRANSITIONS[self.status]:
            raise IllegalStateTransitionError(
                f"{self.train_id}: {self.status.value} -> {new_status.value} is not allowed"
            )
        logger.debug("t=%s %s: %s -> %s", now, self.train_id, self.status.value, new_status.value)
        self.history.append((int(now), self.status.value, new_status.value))
        self.status = new_status
        if location is not None:
            self.location = location

    # -------------------------------------------------------------- readiness
    def readiness_issues(self, now: int) -> list[str]:
        """Return every reason the train cannot be inducted at ``now`` (empty = ready)."""
        issues: list[str] = []
        if self.maintenance_required or self.status is TrainStatus.MAINTENANCE:
            issues.append("maintenance pending")
        if self.location is not TrainLocation.DEPOT:
            issues.append(f"not in depot (location={self.location.value})")
        if self.status not in (TrainStatus.AVAILABLE, TrainStatus.DELAYED):
            issues.append(f"status {self.status.value} cannot be inducted")
        if now < self.readiness_time:
            issues.append(f"preparation incomplete until t={self.readiness_time}")
        return issues

    def is_ready(self, now: int) -> bool:
        return not self.readiness_issues(now)

    def validate_ready(self, now: int) -> None:
        """Raise TrainNotReadyError unless the train can be inducted at ``now``."""
        issues = self.readiness_issues(now)
        if issues:
            raise TrainNotReadyError(f"{self.train_id} not ready at t={now}: " + "; ".join(issues))

    def to_record(self) -> dict:
        return {
            "train_id": self.train_id,
            "status": self.status.value,
            "location": self.location.value,
            "depot_location": self.depot_location,
            "readiness_time_s": self.readiness_time,
            "min_prep_time_s": self.min_prep_time_s,
            "maintenance_required": self.maintenance_required,
            "induction_priority": self.induction_priority,
            "scheduled_induction_time_s": self.scheduled_induction_time,
            "km_since_maintenance": round(self.km_since_maintenance, 1),
        }
