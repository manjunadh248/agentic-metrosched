"""Common scheduler interface and the episode runner shared by every scheduler.

Decision protocol (identical for FCFS, heuristic and PPO):

* the simulation advances in ``decision_interval_s`` steps (60 s);
* at each decision time the scheduler is asked ``decide(env)`` repeatedly; it
  returns a :class:`Decision` (induct this train, optionally via this entry
  point) or ``None`` (wait), up to ``max_inductions_per_tick`` times;
* every decision goes through ``MetroEnvironment.apply_induction`` which
  rejects anything unsafe or infeasible; track/platform rules are enforced by
  the route itself. A scheduler therefore *cannot* create an unsafe movement.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from simulation.config import PROJECT_ROOT, seconds_to_clock
from simulation.data_generator import audit_result, build_environment
from simulation.metro_environment import MetroEnvironment, SimulationResult
from simulation.scenario_generator import Scenario

logger = logging.getLogger(__name__)

DEFAULT_RL_CONFIG = PROJECT_ROOT / "data" / "config" / "rl_config.yaml"


def load_rl_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else DEFAULT_RL_CONFIG
    if not path.is_file():
        raise FileNotFoundError(f"RL/scheduler config not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    for key in ("scheduling", "heuristic", "gym", "reward", "ppo", "training", "evaluation"):
        if key not in cfg:
            raise KeyError(f"rl_config missing section '{key}'")
    return cfg


@dataclass
class Decision:
    train_id: str
    entry_point: str | None = None
    reason: str = ""


@dataclass
class EpisodeResult:
    scheduler: str
    scenario: Scenario
    env: MetroEnvironment
    result: SimulationResult
    demand: object
    explanations: list[dict] = field(default_factory=list)
    safety_violations: list[dict] = field(default_factory=list)


class Scheduler(ABC):
    """Base class. Subclasses implement :meth:`decide`."""

    name = "base"

    def reset(self, env: MetroEnvironment) -> None:  # noqa: B027 - optional hook
        """Called once before the episode starts."""

    @abstractmethod
    def decide(self, env: MetroEnvironment) -> Decision | None:
        """Return the next induction decision at the current time, or None to wait."""


def ready_by_readiness(env: MetroEnvironment):
    return sorted(env.ready_candidates(), key=lambda t: (t.readiness_time, t.train_id))


def run_episode(cfg: dict, scenario: Scenario, scheduler: Scheduler,
                max_inductions_per_tick: int = 3) -> EpisodeResult:
    """Run one full operating day under ``scheduler`` with external control."""
    env, demand, *_ = build_environment(cfg, scenario, control="external", policy_name=scheduler.name)
    scheduler.reset(env)
    explanations: list[dict] = []
    for t in env.decision_times():
        env.advance_to(t)
        for _ in range(max_inductions_per_tick):
            try:
                decision = scheduler.decide(env)
            except Exception:  # never let a scheduler bug crash the simulation
                logger.exception("Scheduler %s raised at t=%s", scheduler.name, t)
                decision = None
            if decision is None:
                break
            ok, why = env.apply_induction(decision.train_id, decision.entry_point)
            explanations.append({"time_s": t, "time": seconds_to_clock(t), "train_id": decision.train_id,
                                 "entry_point": decision.entry_point or "auto", "accepted": ok,
                                 "explanation": decision.reason if ok else f"REJECTED: {why}"})
            if not ok:
                break
    result = env.finish()
    violations = audit_result(env)
    if violations:
        logger.error("Safety audit found %d violations for %s", len(violations), scheduler.name)
    return EpisodeResult(scheduler.name, scenario, env, result, demand, explanations, violations)
