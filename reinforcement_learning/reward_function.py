"""Multi-objective reward for the induction-scheduling agent.

The reward is computed once per decision interval (60 s of simulated time)
from the simulation state; it never decides what is *allowed* – safety is
enforced by the environment (rejection / waiting), so the reward only shapes
*preferences* among safe behaviours.

    r = + w_useful   * useful inductions completed            (successful induction)
        + w_band     * [required <= in_service <= required+1]  (right service headway)
        - w_short    * max(0, required - in_service)           (insufficient coverage)
        - w_surplus  * max(0, deployed - need_30min - 1)       (unnecessary deployment)
        - w_wait     * trains queued on the induction route    (waiting time)
        - w_queue    * new queueing events                     (track contention / conflicts)
        - w_missed   * demand slots that expired unfilled      (missed opportunities)
        - w_delay    * delayed trains in service               (operational delays)
        - w_failure  * new in-service failures                 (worn train inducted)
        - w_invalid  * rejected actions                        (constraint-violating requests)

Rate terms are per train-minute (dt = 1 min); event terms are per event.
See docs/TECHNICAL_REPORT.md for the reasoning behind the default weights.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from evaluation.metrics import SlotTracker
from simulation.metro_environment import MetroEnvironment
from simulation.train import TrainLocation, TrainStatus


@dataclass(frozen=True)
class RewardWeights:
    shortfall_per_train_min: float = 0.30
    surplus_per_train_min: float = 0.03
    waiting_per_train_min: float = 0.05
    queue_event: float = 0.10
    useful_induction: float = 0.50
    missed_opportunity: float = 1.00
    coverage_band_bonus: float = 0.05
    delayed_train_per_min: float = 0.02
    in_service_failure: float = 1.00
    invalid_action: float = 0.10

    @classmethod
    def from_config(cls, d: dict | None) -> "RewardWeights":
        d = dict(d or {})
        unknown = set(d) - set(cls.__dataclass_fields__)
        if unknown:
            raise KeyError(f"Unknown reward weights: {sorted(unknown)}")
        w = cls(**{k: float(v) for k, v in d.items()})
        if any(v < 0 for v in asdict(w).values()):
            raise ValueError("Reward weights must be non-negative (signs are fixed by the formula)")
        return w


def combine(components: dict[str, float], w: RewardWeights, dt_min: float = 1.0) -> float:
    """Pure function: weighted sum of raw reward components (unit-testable)."""
    return (
        + w.useful_induction * components["useful_inductions"]
        + w.coverage_band_bonus * components["in_band"] * dt_min
        - w.shortfall_per_train_min * components["shortfall"] * dt_min
        - w.surplus_per_train_min * components["surplus"] * dt_min
        - w.waiting_per_train_min * components["waiting"] * dt_min
        - w.queue_event * components["queue_events"]
        - w.missed_opportunity * components["missed"]
        - w.delayed_train_per_min * components["delayed"] * dt_min
        - w.in_service_failure * components["failures"]
        - w.invalid_action * components["invalid"]
    )


class RewardCalculator:
    """Stateful per-episode reward computation (call :meth:`step` after each time advance)."""

    def __init__(self, weights: RewardWeights, env: MetroEnvironment, surplus_horizon_s: int = 1800):
        self.w = weights
        self.env = env
        self.horizon = int(surplus_horizon_s)
        self.dt_min = env.decision_interval / 60.0
        self.slots = SlotTracker()
        self._n_inductions = 0
        self._n_failures = 0
        self._waiting_prev: list[str] = []
        self.totals: dict[str, float] = {}

    def components(self, invalid: int = 0) -> dict[str, float]:
        env = self.env
        t = env.now()
        in_service = env.in_service_count()
        started = t >= env.service_start
        required = env.required_at(t) if started else 0
        ahead_t = max(t, env.service_start)
        need = max(env.required_at(x) for x in range(ahead_t, min(ahead_t + self.horizon, env.service_end), 300)) \
            if ahead_t < env.service_end else required
        deployed = in_service + len(env.inducting)
        new_ind = len(env.induction_log) - self._n_inductions
        self._n_inductions = len(env.induction_log)
        # useful = fills a gap against the requirement now or within the next 10 min
        soon = max(env.required_at(x) for x in range(ahead_t, min(ahead_t + 600, env.service_end - 1) + 1, 60)) \
            if ahead_t < env.service_end else required
        useful = max(0, min(new_ind, max(required, soon) - (in_service - new_ind)))
        fails = sum(1 for m in env.maintenance_log if m["event_type"] == "in_service_failure")
        new_fail = fails - self._n_failures
        self._n_failures = fails
        queue_events = sum(1 for tid in env.waiting if tid not in self._waiting_prev)
        self._waiting_prev = list(env.waiting)
        missed = 0
        if started:
            _, missed = self.slots.update(t, required, in_service)
        delayed = sum(1 for tr in env.trains.values()
                      if tr.status is TrainStatus.DELAYED and tr.location is TrainLocation.MAINLINE)
        return {
            "useful_inductions": float(useful),
            "in_band": float(started and required <= in_service <= required + 1),
            "shortfall": float(max(0, required - in_service)),
            "surplus": float(max(0, deployed - need - 1)),
            "waiting": float(len(env.waiting)),
            "queue_events": float(queue_events),
            "missed": float(missed),
            "delayed": float(delayed),
            "failures": float(new_fail),
            "invalid": float(invalid),
        }

    def step(self, invalid: int = 0) -> tuple[float, dict[str, float]]:
        c = self.components(invalid)
        r = combine(c, self.w, self.dt_min)
        for k, v in c.items():
            self.totals[k] = self.totals.get(k, 0.0) + v
        return r, c

    def terminal(self) -> float:
        """Penalty for slots still open when service ends."""
        n = self.slots.close()
        self.totals["missed"] = self.totals.get("missed", 0.0) + n
        return -self.w.missed_opportunity * n
