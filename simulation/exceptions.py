"""Exception hierarchy for the MetroSched simulation.

Hard safety constraints are enforced by *raising* these errors whenever an
operation would violate them. Schedulers (FCFS, heuristic, RL) can therefore
never put the simulated railway in an unsafe state: an unsafe request is
rejected, not merely penalised.
"""


class MetroSchedError(Exception):
    """Base class for all MetroSched errors."""


class ConfigError(MetroSchedError):
    """Configuration file is missing, malformed or out of allowed bounds."""


class IllegalStateTransitionError(MetroSchedError):
    """A train was asked to move between two states that are not connected."""


class SafetyViolationError(MetroSchedError):
    """Base class for any rejected movement that would breach safety rules."""


class TrainNotReadyError(SafetyViolationError):
    """Induction requested for a train that is not ready (prep, maintenance, location)."""


class TrackConflictError(SafetyViolationError):
    """A track section is already occupied by another train."""


class HeadwayViolationError(SafetyViolationError):
    """Minimum safety separation since the previous train has not elapsed."""


class SectionClosedError(SafetyViolationError):
    """The track section or platform is closed (block / possession) at this time."""


class PlatformCapacityError(SafetyViolationError):
    """No platform is free to receive a train."""


class InductionCapacityError(SafetyViolationError):
    """Induction window closed or induction capacity exhausted."""
