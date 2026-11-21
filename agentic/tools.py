"""Tools the agentic AI layer can call.

Each tool is a plain, deterministic Python function over the existing
MetroSched system (SimPy simulator, FCFS / heuristic / PPO schedulers, metrics
and safety audit). The LLM agent never touches trains directly: it can only
*observe* (overview, demand, fleet, saved results) and *experiment* (build a
what-if day, run a scheduler on it, compare, audit). Every induction still goes
through ``MetroEnvironment.apply_induction`` so the hard safety rules hold no
matter what the agent asks for.

``TOOL_SPECS`` holds the JSON schemas sent to the model; ``Toolbox.execute``
dispatches a tool call by name and always returns a JSON-serialisable dict
(errors are returned as ``{"error": ...}`` so the agent can recover).
"""

from __future__ import annotations

import dataclasses
import logging
import re
from pathlib import Path

import pandas as pd

from evaluation.metrics import METRICS, episode_metrics
from scheduling.base import EpisodeResult, load_rl_config, run_episode
from scheduling.baseline_fcfs import FCFSScheduler
from scheduling.heuristic_scheduler import HeuristicScheduler
from simulation.config import PROJECT_ROOT, SCENARIO_TYPES, load_config, seconds_to_clock
from simulation.data_generator import build_environment
from simulation.scenario_generator import Scenario, ScenarioGenerator

logger = logging.getLogger(__name__)

MODEL_PATH = PROJECT_ROOT / "models" / "ppo_metrosched.zip"
EVAL_DIR = PROJECT_ROOT / "results" / "evaluation"
SCHEDULERS = ["FCFS", "Heuristic", "PPO"]
KEY_METRICS = ["demand_coverage_pct", "avg_induction_delay_min", "missed_opportunities",
               "shortfall_train_hours", "avg_waiting_time_min", "induction_conflicts",
               "successful_inductions", "in_service_failures", "surplus_train_hours", "safety_violations"]


def parse_clock(value: str | int) -> int:
    """'HH:MM' (or seconds since midnight) -> seconds since midnight."""
    if isinstance(value, (int, float)):
        return int(value)
    m = re.fullmatch(r"\s*(\d{1,2}):(\d{2})\s*", str(value))
    if not m:
        raise ValueError(f"Time must be 'HH:MM', got {value!r}")
    h, mi = int(m.group(1)), int(m.group(2))
    if not (0 <= h <= 24 and 0 <= mi < 60):
        raise ValueError(f"Invalid time {value!r}")
    return h * 3600 + mi * 60


# --------------------------------------------------------------------- schemas
_CLOSURE = {
    "type": "object",
    "properties": {
        "section_id": {"type": "string", "description": "e.g. EXIT-1, EXIT-2, TRANSFER-LINE, DEPOT-JUNCTION, PF-1, PF-2"},
        "start": {"type": "string", "description": "start time HH:MM"},
        "end": {"type": "string", "description": "end time HH:MM"},
    },
    "required": ["section_id", "start", "end"],
    "additionalProperties": False,
}

TOOL_SPECS: list[dict] = [
    {
        "name": "get_system_overview",
        "description": "Describe the metro depot and line: fleet size, depot exit tracks, shared sections, "
                       "entry points, platforms, headways, induction capacity/windows, the available schedulers "
                       "and valid section ids. Call this first when you need to know the infrastructure.",
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "get_saved_evaluation",
        "description": "Return the stored benchmark of FCFS vs Heuristic vs PPO over 120 unseen simulated days "
                       "(mean ± std per metric and % improvement vs FCFS), overall or per scenario type.",
        "input_schema": {
            "type": "object",
            "properties": {"scenario_type": {"type": "string", "enum": ["all"] + SCENARIO_TYPES}},
            "additionalProperties": False,
        },
    },
    {
        "name": "create_scenario",
        "description": "Create a (what-if) operating day to experiment on. Starts from a reproducible base day "
                       "of the given type and seed, then applies optional changes: demand multiplier, fleet size, "
                       "extra safety headway, and extra closures of exit tracks / shared sections / platforms. "
                       "Returns a scenario_id used by the other tools.",
        "input_schema": {
            "type": "object",
            "properties": {
                "scenario_type": {"type": "string", "enum": SCENARIO_TYPES},
                "seed": {"type": "integer", "description": "reproducible day seed (default 2026)"},
                "demand_multiplier": {"type": "number", "description": "scale passenger demand, 0.5-2.0"},
                "fleet_size": {"type": "integer", "description": "number of trainsets, 15-30 (default from config)"},
                "extra_headway_s": {"type": "integer", "description": "additional safety headway in seconds, 0-180"},
                "closures": {"type": "array", "items": _CLOSURE,
                             "description": "extra temporary closures to add to the day"},
            },
            "required": ["scenario_type"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_demand_profile",
        "description": "Hourly passenger demand (pphpd) and required in-service trains for a scenario, with the "
                       "peak hours and the closures active that day.",
        "input_schema": {
            "type": "object",
            "properties": {"scenario_id": {"type": "string"}},
            "required": ["scenario_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_fleet_status",
        "description": "Morning fleet state for a scenario: each train's readiness time, induction priority, km "
                       "since maintenance and whether maintenance is required (cannot be inducted).",
        "input_schema": {
            "type": "object",
            "properties": {"scenario_id": {"type": "string"}},
            "required": ["scenario_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "run_scheduler",
        "description": "Simulate the full operating day of a scenario with one scheduler (FCFS, Heuristic or the "
                       "trained PPO reinforcement-learning agent) and return its KPIs and the safety audit result.",
        "input_schema": {
            "type": "object",
            "properties": {
                "scenario_id": {"type": "string"},
                "scheduler": {"type": "string", "enum": SCHEDULERS},
            },
            "required": ["scenario_id", "scheduler"],
            "additionalProperties": False,
        },
    },
    {
        "name": "compare_schedulers",
        "description": "Run several schedulers on the same scenario (identical faults/delays) and return a KPI "
                       "table side by side plus which scheduler is best on each metric.",
        "input_schema": {
            "type": "object",
            "properties": {
                "scenario_id": {"type": "string"},
                "schedulers": {"type": "array", "items": {"type": "string", "enum": SCHEDULERS}},
            },
            "required": ["scenario_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_induction_plan",
        "description": "The induction sequence a scheduler produced for a scenario: train, induction start, depot "
                       "exit, in-service time, exit track, entry point, platform, waits, and the scheduler's own "
                       "explanation for each decision. Optionally limited to a time window.",
        "input_schema": {
            "type": "object",
            "properties": {
                "scenario_id": {"type": "string"},
                "scheduler": {"type": "string", "enum": SCHEDULERS},
                "from_time": {"type": "string", "description": "HH:MM, optional"},
                "to_time": {"type": "string", "description": "HH:MM, optional"},
                "limit": {"type": "integer", "description": "max rows (default 15)"},
            },
            "required": ["scenario_id", "scheduler"],
            "additionalProperties": False,
        },
    },
]


# --------------------------------------------------------------------- toolbox
class Toolbox:
    """Holds the scenarios and simulation runs created during one agent session."""

    def __init__(self, cfg: dict | None = None, rl_cfg: dict | None = None, allow_ppo: bool = True):
        self.base_cfg = cfg or load_config()
        self.rl_cfg = rl_cfg or load_rl_config()
        self.allow_ppo = allow_ppo and MODEL_PATH.exists()
        self.scenarios: dict[str, tuple[Scenario, dict, dict]] = {}   # id -> (scenario, cfg, description)
        self.runs: dict[tuple[str, str], tuple[EpisodeResult, dict]] = {}
        self._ppo = None
        self._handlers = {
            "get_system_overview": self.get_system_overview,
            "get_saved_evaluation": self.get_saved_evaluation,
            "create_scenario": self.create_scenario,
            "get_demand_profile": self.get_demand_profile,
            "get_fleet_status": self.get_fleet_status,
            "run_scheduler": self.run_scheduler,
            "compare_schedulers": self.compare_schedulers,
            "get_induction_plan": self.get_induction_plan,
        }

    @property
    def specs(self) -> list[dict]:
        return TOOL_SPECS

    def execute(self, name: str, args: dict | None) -> dict:
        fn = self._handlers.get(name)
        if fn is None:
            return {"error": f"Unknown tool {name!r}. Available: {sorted(self._handlers)}"}
        try:
            return fn(**(args or {}))
        except TypeError as exc:
            return {"error": f"Bad arguments for {name}: {exc}"}
        except (ValueError, KeyError, FileNotFoundError) as exc:
            return {"error": str(exc)}
        except Exception as exc:  # noqa: BLE001 - surface to the agent, don't crash the loop
            logger.exception("Tool %s failed", name)
            return {"error": f"{type(exc).__name__}: {exc}"}

    # ----------------------------------------------------------- helpers
    def _scenario(self, scenario_id: str) -> tuple[Scenario, dict, dict]:
        if scenario_id not in self.scenarios:
            raise KeyError(f"Unknown scenario_id {scenario_id!r}. Create one with create_scenario first. "
                           f"Existing: {list(self.scenarios)}")
        return self.scenarios[scenario_id]

    def _make_scheduler(self, name: str, cfg: dict):
        if name == "FCFS":
            return FCFSScheduler()
        if name == "Heuristic":
            return HeuristicScheduler.from_config(self.rl_cfg)
        if name == "PPO":
            if not self.allow_ppo:
                raise FileNotFoundError("PPO model not available (models/ppo_metrosched.zip missing or disabled).")
            from scheduling.rl_scheduler import RLScheduler
            if self._ppo is None:
                self._ppo = RLScheduler(MODEL_PATH, cfg=self.base_cfg, rl_cfg=self.rl_cfg)
            from reinforcement_learning.gym_environment import ObservationEncoder
            sched = RLScheduler.__new__(RLScheduler)
            sched.__dict__.update(self._ppo.__dict__)
            sched.cfg = cfg
            sched.encoder = ObservationEncoder(cfg, self.rl_cfg["gym"]["candidate_slots"])
            return sched
        raise ValueError(f"Unknown scheduler {name!r}; choose from {SCHEDULERS}")

    def _run(self, scenario_id: str, scheduler: str) -> tuple[EpisodeResult, dict]:
        key = (scenario_id, scheduler)
        if key not in self.runs:
            sc, cfg, _ = self._scenario(scenario_id)
            ep = run_episode(cfg, sc, self._make_scheduler(scheduler, cfg),
                             int(self.rl_cfg["scheduling"]["max_inductions_per_tick"]))
            self.runs[key] = (ep, episode_metrics(ep, cfg))
        return self.runs[key]

    @staticmethod
    def _kpis(m: dict) -> dict:
        return {METRICS[k][0] + f" [{METRICS[k][1]}]": round(float(m[k]), 2) for k in KEY_METRICS if k in m}

    # ----------------------------------------------------------- tools
    def get_system_overview(self) -> dict:
        c = self.base_cfg
        d, t, p = c["depot"], c["track"], c["platforms"]
        return {
            "data_notice": "All data is SYNTHETIC simulation data, not real KMRL operational data.",
            "operating_day": {"depot_open": seconds_to_clock(c["simulation"]["depot_open"]),
                              "service_start": seconds_to_clock(c["simulation"]["service_start"]),
                              "service_end": seconds_to_clock(c["simulation"]["service_end"]),
                              "decision_interval_s": c["simulation"]["decision_interval_s"]},
            "fleet_size": c["fleet"]["size"],
            "exit_tracks": [{"id": e["id"], "headway_s": e["headway_s"]} for e in d["exit_tracks"]],
            "shared_sections": [{"id": s["id"], "headway_s": s["headway_s"]} for s in t["shared_sections"]],
            "entry_points": [{"id": e["id"], "headway_s": e["headway_s"]} for e in t["entry_points"]],
            "platforms": [f"PF-{i + 1}" for i in range(p["count"])],
            "max_inductions_per_hour": d["max_inductions_per_hour"],
            "induction_windows": [{"id": w["id"], "start": seconds_to_clock(w["start"]),
                                   "end": seconds_to_clock(w["end"]), "max_inductions": w["max_inductions"]}
                                  for w in d["induction_windows"]],
            "schedulers": {
                "FCFS": "baseline: earliest-ready train when service is short now, no look-ahead",
                "Heuristic": "rules: 12-min demand look-ahead, avoids bunching, prefers healthy trains",
                "PPO": ("reinforcement-learning agent (MaskablePPO, behaviour-cloning warm start), "
                        + ("available" if self.allow_ppo else "NOT available")),
            },
            "hard_safety_rules": "one train per block/platform, minimum headways, closures, readiness, induction "
                                 "capacity; enforced by the simulator for every scheduler",
            "scenarios_created_so_far": list(self.scenarios),
        }

    def get_saved_evaluation(self, scenario_type: str = "all") -> dict:
        f = EVAL_DIR / "episode_metrics.csv"
        if not f.exists():
            raise FileNotFoundError("No saved evaluation found. Run: python main.py evaluate")
        df = pd.read_csv(f)
        if scenario_type != "all":
            df = df[df["scenario_type"] == scenario_type]
            if df.empty:
                raise ValueError(f"No episodes for scenario type {scenario_type!r}")
        out = {}
        fcfs = df[df["scheduler"] == "FCFS"]
        for k in KEY_METRICS:
            if k not in df:
                continue
            row = {}
            base = fcfs[k].mean() if len(fcfs) else None
            for s, g in df.groupby("scheduler"):
                txt = f"{g[k].mean():.2f} ± {g[k].std():.2f}"
                direction = METRICS[k][2]
                if s != "FCFS" and base and direction in ("lower", "higher") and abs(base) > 1e-9:
                    imp = (base - g[k].mean()) / abs(base) * 100 * (1 if direction == "lower" else -1)
                    txt += f" ({imp:+.1f}% vs FCFS)"
                row[s] = txt
            out[f"{METRICS[k][0]} [{METRICS[k][1]}, {METRICS[k][2]} is better]"] = row
        return {"episodes": int(df.groupby("scheduler").size().max()), "scenario_type": scenario_type,
                "metrics": out}

    def create_scenario(self, scenario_type: str, seed: int = 2026, demand_multiplier: float | None = None,
                        fleet_size: int | None = None, extra_headway_s: int | None = None,
                        closures: list[dict] | None = None) -> dict:
        if scenario_type not in SCENARIO_TYPES:
            raise ValueError(f"scenario_type must be one of {SCENARIO_TYPES}")
        overrides: dict = {}
        if fleet_size is not None:
            if not 15 <= int(fleet_size) <= 30:
                raise ValueError("fleet_size must be between 15 and 30")
            overrides["fleet"] = {"size": int(fleet_size)}
        cfg = load_config(overrides=overrides) if overrides else self.base_cfg
        sc = ScenarioGenerator(cfg, int(seed)).single(scenario_type)
        changes = []
        if demand_multiplier is not None:
            if not 0.5 <= float(demand_multiplier) <= 2.0:
                raise ValueError("demand_multiplier must be between 0.5 and 2.0")
            sc = dataclasses.replace(sc, demand_multiplier=round(sc.demand_multiplier * float(demand_multiplier), 3))
            changes.append(f"demand x{demand_multiplier}")
        if extra_headway_s is not None:
            if not 0 <= int(extra_headway_s) <= 180:
                raise ValueError("extra_headway_s must be between 0 and 180")
            sc = dataclasses.replace(sc, extra_headway_s=sc.extra_headway_s + int(extra_headway_s))
            changes.append(f"+{extra_headway_s}s headway")
        if fleet_size is not None:
            changes.append(f"fleet {fleet_size}")
        exits = {e["id"] for e in cfg["depot"]["exit_tracks"]}
        shared = {s["id"] for s in cfg["track"]["shared_sections"]}
        plats = {f"PF-{i + 1}" for i in range(cfg["platforms"]["count"])}
        ex, sb, pc = list(sc.exit_track_closures), list(sc.section_blocks), list(sc.platform_closures)
        for c in closures or []:
            sid, s, e = c["section_id"].strip().upper(), parse_clock(c["start"]), parse_clock(c["end"])
            if e <= s:
                raise ValueError(f"Closure of {sid}: end must be after start")
            target = ex if sid in exits else sb if sid in shared else pc if sid in plats else None
            if target is None:
                raise ValueError(f"Unknown section {sid!r}. Valid: {sorted(exits | shared | plats)}")
            if target is ex and len({x[0] for x in ex} | {sid}) >= len(exits) and \
                    any(x[0] != sid and x[1] < e and s < x[2] for x in ex):
                raise ValueError("Closing every depot exit track at the same time would stop all inductions; "
                                 "the simulator does not allow that scenario.")
            target.append((sid, s, e))
            changes.append(f"close {sid} {seconds_to_clock(s)}-{seconds_to_clock(e)}")
        sc = dataclasses.replace(sc, exit_track_closures=ex, section_blocks=sb, platform_closures=pc)
        sid = f"S{len(self.scenarios) + 1}"
        desc = {"scenario_id": sid, "scenario_type": scenario_type, "seed": int(seed),
                "changes": changes or ["none (base day)"]}
        self.scenarios[sid] = (sc, cfg, desc)
        return {**desc, **self._scenario_summary(sc, cfg)}

    @staticmethod
    def _scenario_summary(sc: Scenario, cfg: dict) -> dict:
        fmt = lambda lst: [f"{i} {seconds_to_clock(s)}-{seconds_to_clock(e)}" for i, s, e in lst]  # noqa: E731
        return {"demand_multiplier": sc.demand_multiplier, "extra_headway_s": sc.extra_headway_s,
                "induction_capacity_factor": sc.induction_capacity_factor, "fleet_size": cfg["fleet"]["size"],
                "prep_delay_prob": sc.prep_delay_prob, "depot_fault_prob": sc.depot_fault_prob,
                "closures": fmt(sc.exit_track_closures) + fmt(sc.section_blocks) + fmt(sc.platform_closures)}

    def get_demand_profile(self, scenario_id: str) -> dict:
        sc, cfg, desc = self._scenario(scenario_id)
        _, demand, *_ = build_environment(cfg, sc, control="external", policy_name="probe")
        demand = demand.copy()
        demand["hour"] = demand["interval_start_s"] // 3600
        hourly = demand.groupby("hour").agg(pphpd=("demand_pphpd", "max"), required_trains=("required_trains", "max"))
        rows = [{"hour": f"{int(h):02d}:00", "peak_pphpd": int(r.pphpd), "required_trains": int(r.required_trains)}
                for h, r in hourly.iterrows()]
        top = hourly.sort_values("required_trains", ascending=False).head(3)
        return {"scenario_id": scenario_id, "hourly": rows,
                "peak_hours": [f"{int(h):02d}:00" for h in top.index],
                "max_required_trains": int(hourly["required_trains"].max()),
                "closures": self._scenario_summary(sc, cfg)["closures"]}

    def get_fleet_status(self, scenario_id: str) -> dict:
        sc, cfg, _ = self._scenario(scenario_id)
        env, *_ = build_environment(cfg, sc, control="external", policy_name="probe")
        trains = sorted(env.trains.values() if isinstance(env.trains, dict) else env.trains,
                        key=lambda t: (t.readiness_time, t.train_id))
        rows = [{"train": t.train_id, "ready_at": seconds_to_clock(t.readiness_time),
                 "priority": int(t.induction_priority), "km_since_maintenance": round(float(t.km_since_maintenance)),
                 "maintenance_required": bool(t.maintenance_required)} for t in trains]
        return {"scenario_id": scenario_id, "fleet_size": len(rows),
                "available_for_induction": sum(not r["maintenance_required"] for r in rows),
                "in_maintenance": [r["train"] for r in rows if r["maintenance_required"]],
                "trains": rows}

    def run_scheduler(self, scenario_id: str, scheduler: str) -> dict:
        ep, m = self._run(scenario_id, scheduler)
        return {"scenario_id": scenario_id, "scheduler": scheduler, "kpis": self._kpis(m),
                "safety_audit": "PASSED: 0 violations" if not ep.safety_violations
                else f"FAILED: {len(ep.safety_violations)} violations",
                "rejected_actions": int(m["rejected_actions"])}

    def compare_schedulers(self, scenario_id: str, schedulers: list[str] | None = None) -> dict:
        names = schedulers or [s for s in SCHEDULERS if s != "PPO" or self.allow_ppo]
        results = {n: self._run(scenario_id, n)[1] for n in names}
        table, best = {}, {}
        for k in KEY_METRICS:
            label = f"{METRICS[k][0]} [{METRICS[k][1]}]"
            table[label] = {n: round(float(results[n][k]), 2) for n in names}
            direction = METRICS[k][2]
            if direction in ("lower", "higher") and len(names) > 1:
                vals = table[label]
                pick = min(vals, key=vals.get) if direction == "lower" else max(vals, key=vals.get)
                ties = [n for n, v in vals.items() if v == vals[pick]]
                best[label] = "tie: " + ", ".join(ties) if len(ties) > 1 else pick
        return {"scenario_id": scenario_id, "table": table, "best_per_metric": best,
                "safety_violations": {n: int(results[n]["safety_violations"]) for n in names}}

    def get_induction_plan(self, scenario_id: str, scheduler: str, from_time: str | None = None,
                           to_time: str | None = None, limit: int = 15) -> dict:
        ep, _ = self._run(scenario_id, scheduler)
        ind = ep.result.inductions
        if ind.empty:
            return {"scenario_id": scenario_id, "scheduler": scheduler, "inductions": [], "total": 0}
        lo = parse_clock(from_time) if from_time else 0
        hi = parse_clock(to_time) if to_time else 10 ** 9
        expl = {(e["train_id"], e["time_s"]): e["explanation"] for e in ep.explanations if e["accepted"]}
        rows = []
        for _, r in ind.iterrows():
            t = int(r["authorised_time_s"])
            if not lo <= t <= hi:
                continue
            # the decision for this induction is the latest accepted request for this train at or before it
            match = [(ts, v) for (tid, ts), v in expl.items() if tid == r["train_id"] and ts <= t]
            why = max(match)[1] if match else ""
            rows.append({"train": r["train_id"], "induction_start": seconds_to_clock(t),
                         "depot_exit": seconds_to_clock(r["depot_exit_time_s"]),
                         "in_service": seconds_to_clock(r["service_start_time_s"]),
                         "exit_track": r["exit_track"], "entry_point": r["entry_point"], "platform": r["platform"],
                         "track_wait_s": int(r["track_wait_s"]), "capacity_wait_s": int(r["capacity_wait_s"]),
                         "why": why[:300]})
        limit = max(1, min(int(limit), 40))
        return {"scenario_id": scenario_id, "scheduler": scheduler, "total": len(rows),
                "showing": min(limit, len(rows)), "inductions": rows[:limit]}


def tool_names() -> list[str]:
    return [t["name"] for t in TOOL_SPECS]


def project_root() -> Path:
    return PROJECT_ROOT
