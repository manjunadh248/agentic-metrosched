"""Run several schedulers on identical scenarios and collect per-episode metrics."""

from __future__ import annotations

import logging
import time
from typing import Callable

import pandas as pd

from scheduling.base import EpisodeResult, Scheduler, run_episode
from simulation.config import SCENARIO_TYPES
from simulation.scenario_generator import Scenario, ScenarioGenerator

from .metrics import episode_metrics

logger = logging.getLogger(__name__)


def evaluation_scenarios(cfg: dict, seed_base: int, per_type: int,
                         types: list[str] | None = None) -> list[Scenario]:
    """Deterministic unseen scenarios: seed_base + i, stratified by scenario type."""
    types = types or SCENARIO_TYPES
    out = []
    for i in range(per_type):
        for j, st in enumerate(types):
            out.append(ScenarioGenerator(cfg, seed_base + i * len(types) + j).single(st))
    return out


def run_comparison(cfg: dict, schedulers: list[Scheduler], scenarios: list[Scenario],
                   max_inductions_per_tick: int = 3, keep_examples: int = 1,
                   progress: Callable[[int, int], None] | None = None) -> tuple[pd.DataFrame, dict]:
    """Every scheduler runs every scenario (identical conditions).

    Returns (per-episode metrics DataFrame, examples) where examples maps
    (scheduler, scenario_type) -> EpisodeResult for the first ``keep_examples``
    scenarios of each type (used for Gantt charts and the dashboard).
    """
    rows, examples, seen = [], {}, {}
    total, done = len(scenarios) * len(schedulers), 0
    t0 = time.time()
    for sc in scenarios:
        seen[sc.scenario_type] = seen.get(sc.scenario_type, 0) + 1
        for sched in schedulers:
            ep: EpisodeResult = run_episode(cfg, sc, sched, max_inductions_per_tick)
            m = episode_metrics(ep, cfg)
            m["weekday"] = sc.weekday
            rows.append(m)
            if seen[sc.scenario_type] <= keep_examples:
                examples[(sched.name, sc.scenario_type)] = ep
            done += 1
            if progress:
                progress(done, total)
        if done % (10 * len(schedulers)) == 0:
            logger.info("evaluated %d/%d episodes (%.0fs)", done, total, time.time() - t0)
    return pd.DataFrame(rows), examples
