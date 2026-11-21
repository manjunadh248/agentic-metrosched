"""Operating-day scenarios and per-day fleet generation.

Three day types are produced:

* ``normal``      – baseline demand and low disruption probabilities.
* ``peak_demand`` – demand scaled up (e.g. festivals / events), slightly more
                    preparation delays and in-service delays.
* ``disruption``  – frequent preparation overruns and depot faults, an exit
                    track closed for part of the day, temporary blocks on
                    shared sections, a platform closure, extra safety headway,
                    reduced induction capacity and heavier through traffic.

Every random draw comes from a ``numpy.random.Generator`` derived from one
master seed, so identical seeds give identical scenarios.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from .config import SCENARIO_TYPES
from .depot import Depot
from .train import Train

logger = logging.getLogger(__name__)


@dataclass
class Scenario:
    day: int
    scenario_type: str
    weekday: int
    seed: int
    demand_multiplier: float
    prep_delay_prob: float
    prep_delay_mean_s: float
    depot_fault_prob: float
    trip_delay_factor: float
    through_traffic_factor: float
    extra_headway_s: int
    induction_capacity_factor: float
    # (section_id, start_s, end_s) closures for exit tracks / shared sections
    exit_track_closures: list[tuple[str, int, int]] = field(default_factory=list)
    section_blocks: list[tuple[str, int, int]] = field(default_factory=list)
    platform_closures: list[tuple[str, int, int]] = field(default_factory=list)

    def to_record(self) -> dict:
        rec = asdict(self)
        for key in ("exit_track_closures", "section_blocks", "platform_closures"):
            rec[key] = ";".join(f"{sid}@{s}-{e}" for sid, s, e in getattr(self, key)) or ""
        return rec


class ScenarioGenerator:
    """Builds a reproducible sequence of operating-day scenarios."""

    def __init__(self, cfg: dict, seed: int):
        self.cfg = cfg
        self.seed = int(seed)

    def scenario_types(self, n_days: int, rng: np.random.Generator) -> list[str]:
        """Draw day types from the configured mix, ensuring each type appears once."""
        mix = self.cfg["scenarios"]["mix"]
        probs = np.array([mix[t] for t in SCENARIO_TYPES], dtype=float)
        types = list(rng.choice(SCENARIO_TYPES, size=n_days, p=probs / probs.sum()))
        if n_days >= len(SCENARIO_TYPES):
            for st in SCENARIO_TYPES:
                if st not in types:
                    # overwrite the most common type at a random position
                    common = max(SCENARIO_TYPES, key=types.count)
                    idxs = [i for i, x in enumerate(types) if x == common]
                    types[idxs[int(rng.integers(len(idxs)))]] = st
        return [str(t) for t in types]

    def _closures(self, rng: np.random.Generator, ids: list[str], count: int,
                  duration: tuple[int, int]) -> list[tuple[str, int, int]]:
        out = []
        if count <= 0 or not ids:
            return out
        chosen = rng.choice(len(ids), size=min(count, len(ids)), replace=False)
        for idx in sorted(int(i) for i in chosen):
            start = int(rng.integers(6 * 3600, 20 * 3600))
            length = int(rng.integers(duration[0], duration[1] + 1))
            out.append((ids[idx], start, start + length))
        return out

    def generate(self, n_days: int, start_weekday: int = 0) -> list[Scenario]:
        if n_days < 1:
            raise ValueError("n_days must be >= 1")
        root = np.random.SeedSequence(self.seed)
        type_rng = np.random.default_rng(root.spawn(1)[0])
        types = self.scenario_types(n_days, type_rng)
        day_seqs = np.random.SeedSequence([self.seed, 1]).spawn(n_days)
        scenarios = []
        for day, (stype, seq) in enumerate(zip(types, day_seqs)):
            day_seed = int(seq.generate_state(1)[0])
            scenarios.append(self.build(day, stype, (start_weekday + day) % 7, day_seed))
        logger.info("Generated %d scenarios: %s", n_days,
                    {t: types.count(t) for t in SCENARIO_TYPES})
        return scenarios

    def build(self, day: int, stype: str, weekday: int, day_seed: int) -> Scenario:
        """Build one scenario of a given type from its own seed (deterministic)."""
        if stype not in SCENARIO_TYPES:
            raise ValueError(f"Unknown scenario type {stype!r}")
        cfg = self.cfg
        exit_ids = [t["id"] for t in cfg["depot"]["exit_tracks"]]
        shared_ids = [s["id"] for s in cfg["track"]["shared_sections"]]
        platform_ids = [f"PF-{i + 1}" for i in range(cfg["platforms"]["count"])]
        if True:
            p = cfg["scenarios"][stype]
            rng = np.random.default_rng(day_seed)
            dur = tuple(p.get("closure_duration_s", [1800, 7200]))
            sc = Scenario(
                day=day, scenario_type=stype, weekday=weekday, seed=int(day_seed),
                demand_multiplier=float(p["demand_multiplier"]),
                prep_delay_prob=float(p["prep_delay_prob"]),
                prep_delay_mean_s=float(p["prep_delay_mean_s"]),
                depot_fault_prob=float(p["depot_fault_prob"]),
                trip_delay_factor=float(p["trip_delay_factor"]),
                through_traffic_factor=float(p["through_traffic_factor"]),
                extra_headway_s=int(p["extra_headway_s"]),
                induction_capacity_factor=float(p["induction_capacity_factor"]),
                exit_track_closures=self._closures(rng, exit_ids, int(p["exit_track_closures"]), dur),
                section_blocks=self._closures(rng, shared_ids, int(p["section_blocks"]), dur),
                platform_closures=self._closures(rng, platform_ids, int(p["platform_closures"]), dur),
            )
        return sc

    def single(self, scenario_type: str | None = None) -> Scenario:
        """One scenario derived from ``self.seed`` (used for training / evaluation episodes).

        If ``scenario_type`` is None the type is drawn from the configured mix.
        Weekday is drawn uniformly. Identical seed -> identical scenario.
        """
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, 99]))
        mix = self.cfg["scenarios"]["mix"]
        probs = np.array([mix[t] for t in SCENARIO_TYPES], dtype=float)
        drawn = str(rng.choice(SCENARIO_TYPES, p=probs / probs.sum()))
        weekday = int(rng.integers(0, 7))
        day_seed = int(rng.integers(0, 2 ** 31 - 1))
        return self.build(0, scenario_type or drawn, weekday, day_seed)


def generate_fleet(cfg: dict, scenario: Scenario, rng: np.random.Generator) -> tuple[list[Train], list[dict]]:
    """Create the fleet for one day plus the list of preparation delays applied.

    Readiness time = depot_open + stagger + min_prep_time + preparation_delay.
    Trains over the maintenance odometer threshold start in Maintenance.
    """
    fcfg = cfg["fleet"]
    size = int(fcfg["size"])
    depot_open = cfg["simulation"]["depot_open"]
    berths = Depot(None, cfg).berth_names()
    order = rng.permutation(len(berths))[:size]
    stagger_order = rng.permutation(size)
    threshold = fcfg["maintenance_km_threshold"]
    levels = int(fcfg["priority_levels"])
    trains, delays = [], []
    for i in range(size):
        tid = f"{fcfg['id_prefix']}{i + 1:02d}"
        min_prep = int(rng.integers(fcfg["min_prep_time_s"][0], fcfg["min_prep_time_s"][1] + 1))
        km = float(rng.uniform(*fcfg["km_since_maintenance"]))
        prep_delay = 0
        if rng.random() < scenario.prep_delay_prob:
            prep_delay = int(round(rng.exponential(scenario.prep_delay_mean_s)))
        start = depot_open + int(stagger_order[i]) * int(fcfg["prep_stagger_s"])
        readiness = start + min_prep + prep_delay
        priority = int(min(levels, 1 + km // (threshold / levels)))
        tr = Train(train_id=tid, depot_location=berths[int(order[i])], readiness_time=int(readiness),
                   min_prep_time_s=min_prep, maintenance_required=bool(km > threshold),
                   induction_priority=priority, km_since_maintenance=km)
        trains.append(tr)
        if prep_delay > 0:
            delays.append({"train_id": tid, "prep_start_s": start, "delay_s": prep_delay})
    return trains, delays


def assign_scheduled_inductions(trains: list[Train], demand_df: pd.DataFrame, cfg: dict) -> list[dict]:
    """Derive today's *planned* induction slots from demand and assign trains.

    Each increase in ``required_trains`` between consecutive intervals creates
    that many slots at the interval start. Slots are filled in time order by
    the best-priority eligible train that is ready in time (else the earliest
    ready). The scheduled induction time is the slot time minus the induction
    look-ahead so that the train reaches the mainline when it is needed.
    Returns the slot list (useful later as the "induction plan").
    """
    lookahead = int(cfg["service"]["induction_lookahead_s"])
    slots, prev = [], 0
    for _, row in demand_df.iterrows():
        req = int(row["required_trains"])
        for _ in range(max(0, req - prev)):
            slots.append(int(row["interval_start_s"]))
        prev = req
    pool = sorted([t for t in trains if not t.maintenance_required],
                  key=lambda t: (t.induction_priority, t.readiness_time, t.train_id))
    plan = []
    for k, slot in enumerate(slots):
        if not pool:
            break
        start = max(cfg["simulation"]["depot_open"], slot - lookahead)
        ready = [t for t in pool if t.readiness_time <= start]
        chosen = ready[0] if ready else min(pool, key=lambda t: (t.readiness_time, t.train_id))
        pool.remove(chosen)
        chosen.scheduled_induction_time = int(start)
        plan.append({"slot": k, "slot_time_s": slot, "train_id": chosen.train_id,
                     "scheduled_induction_time_s": int(start)})
    return plan
