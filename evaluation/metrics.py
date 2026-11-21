"""Evaluation metrics computed from actual simulation logs (never hardcoded).

All metrics are scheduler-independent definitions applied to the same logs:

* **Demand slots.** Whenever the demand-derived requirement rises from ``r``
  to ``r'`` (and at the start of service for the first ``r`` trains), the
  levels ``r+1 .. r'`` become *slots* that must be filled by in-service trains.
  A slot is filled when the in-service count reaches its level. It is
  *missed* if the requirement falls back below its level (or service ends)
  before it was filled. :class:`SlotTracker` implements this online so the RL
  reward and the offline metrics use exactly the same logic.
* **Average induction delay** – mean time from slot creation to slot fill.
* **Missed induction opportunities** – number of missed slots.
* **Train induction conflicts** – inductions that had to queue behind another
  train at a block/platform (resource contention, resolved safely). Hard
  safety violations from the independent audit are reported separately and
  must always be zero.
* **Average train waiting time** – capacity wait + track wait per induction.
* **Passenger demand coverage** – Σ min(supplied capacity, demand) / Σ demand,
  with supplied pphpd = in_service × 3600 × capacity × target_load / cycle.
* **Track utilization** – occupied / available time on the depot induction
  route (exit tracks + shared sections) during service hours.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from simulation.demand_generator import cycle_time_s

# metric name -> (label, unit, direction) ; direction: "lower", "higher", "neutral"
METRICS: dict[str, tuple[str, str, str]] = {
    "induction_conflicts": ("Train induction conflicts", "count/day", "lower"),
    "avg_induction_delay_min": ("Average induction delay", "min", "lower"),
    "avg_waiting_time_min": ("Average train waiting time", "min", "lower"),
    "demand_coverage_pct": ("Passenger demand coverage", "%", "higher"),
    "track_utilization_pct": ("Track utilization (depot route)", "%", "neutral"),
    "successful_inductions": ("Successful train inductions", "count/day", "neutral"),
    "missed_opportunities": ("Missed induction opportunities", "count/day", "lower"),
    "shortfall_train_hours": ("Service shortfall", "train-h/day", "lower"),
    "surplus_train_hours": ("Unnecessary deployment (above need+1)", "train-h/day", "lower"),
    "fleet_efficiency_pct": ("Fleet utilization efficiency", "%", "higher"),
    "in_service_failures": ("In-service failures", "count/day", "lower"),
    "in_service_delay_min": ("In-service delay", "min/day", "lower"),
    "rejected_actions": ("Rejected (unsafe/infeasible) actions", "count/day", "lower"),
    "safety_violations": ("Hard safety violations (audit)", "count/day", "lower"),
}


@dataclass
class SlotTracker:
    """Online demand-slot bookkeeping (see module docstring)."""

    open_slots: list[tuple[int, int]] = field(default_factory=list)   # (level, created_at)
    filled: list[int] = field(default_factory=list)                   # delays in seconds
    missed: int = 0
    prev_required: int = 0

    def update(self, t: int, required: int, in_service: int) -> tuple[int, int]:
        """Advance to time ``t``. Returns (newly_filled, newly_missed)."""
        new_f = new_m = 0
        for level in range(self.prev_required + 1, required + 1):
            self.open_slots.append((level, t))
        still = []
        for level, created in self.open_slots:
            if in_service >= level:
                self.filled.append(t - created)
                new_f += 1
            elif required < level:
                self.missed += 1
                new_m += 1
            else:
                still.append((level, created))
        self.open_slots = still
        self.prev_required = required
        return new_f, new_m

    def close(self) -> int:
        n = len(self.open_slots)
        self.missed += n
        self.open_slots = []
        return n


def slot_metrics(monitor: pd.DataFrame, service_start: int) -> dict:
    tracker = SlotTracker()
    m = monitor[monitor["time_s"] >= service_start]
    for t, req, ins in zip(m["time_s"], m["required_now"], m["in_service"]):
        tracker.update(int(t), int(req), int(ins))
    tracker.close()
    delays = np.array(tracker.filled, dtype=float)
    return {"avg_induction_delay_min": float(delays.mean() / 60) if len(delays) else 0.0,
            "missed_opportunities": int(tracker.missed), "slots_filled": int(len(delays))}


def demand_coverage(monitor: pd.DataFrame, demand_df: pd.DataFrame, cfg: dict) -> float:
    m = monitor[monitor["time_s"] >= cfg["simulation"]["service_start"]]
    if m.empty:
        return 0.0
    starts = demand_df["interval_start_s"].to_numpy()
    idx = np.clip(np.searchsorted(starts, m["time_s"].to_numpy(), side="right") - 1, 0, len(starts) - 1)
    demand = demand_df["demand_pphpd"].to_numpy(dtype=float)[idx]
    per_train = 3600.0 * cfg["fleet"]["train_capacity"] * cfg["service"]["target_load_factor"] / cycle_time_s(cfg)
    supply = m["in_service"].to_numpy(dtype=float) * per_train
    return float(np.minimum(supply, demand).sum() / max(demand.sum(), 1e-9) * 100)


def track_utilization(occupancy: pd.DataFrame, closures: pd.DataFrame, cfg: dict) -> float:
    start, end = cfg["simulation"]["service_start"], cfg["simulation"]["service_end"]
    ids = [t["id"] for t in cfg["depot"]["exit_tracks"]] + [s["id"] for s in cfg["track"]["shared_sections"]]
    total_avail = total_occ = 0.0
    for sid in ids:
        occ = occupancy[occupancy["section_id"] == sid] if not occupancy.empty else occupancy
        busy = sum(max(0, min(e, end) - max(s, start)) for s, e in zip(occ.get("start_s", []), occ.get("end_s", [])))
        cl = closures[closures["section_id"] == sid] if not closures.empty else closures
        closed = sum(max(0, min(e, end) - max(s, start)) for s, e in zip(cl.get("start_s", []), cl.get("end_s", [])))
        total_avail += (end - start) - closed
        total_occ += busy
    return float(total_occ / max(total_avail, 1) * 100)


def episode_metrics(ep, cfg: dict) -> dict:
    """Compute every metric for one :class:`scheduling.base.EpisodeResult`."""
    res = ep.result
    mon = res.monitor
    start = cfg["simulation"]["service_start"]
    ms = mon[mon["time_s"] >= start]
    dt_h = cfg["simulation"]["decision_interval_s"] / 3600
    ih = res.inductions
    n_ind = int(len(ih))
    waits = (ih["capacity_wait_s"] + ih["track_wait_s"]) if n_ind else pd.Series(dtype=float)
    delays = res.delays
    me = res.maintenance_events
    req, ins = ms["required_now"].to_numpy(), ms["in_service"].to_numpy()
    out = {
        "scheduler": ep.scheduler, "scenario_type": ep.scenario.scenario_type, "seed": int(ep.scenario.seed),
        "induction_conflicts": int((ih["track_wait_s"] > 0).sum()) if n_ind else 0,
        "avg_waiting_time_min": float(waits.mean() / 60) if n_ind else 0.0,
        "demand_coverage_pct": demand_coverage(mon, ep.demand, cfg),
        "track_utilization_pct": track_utilization(res.occupancy, res.closures, cfg),
        "successful_inductions": n_ind,
        "shortfall_train_hours": float(np.clip(req - ins, 0, None).sum() * dt_h),
        "surplus_train_hours": float(np.clip(ins - req - 1, 0, None).sum() * dt_h),
        "fleet_efficiency_pct": float(np.minimum(ins, req).sum() / max(ins.sum(), 1) * 100),
        "in_service_failures": int((me["event_type"] == "in_service_failure").sum()) if len(me) else 0,
        "in_service_delay_min": float(delays.loc[delays["delay_type"] == "in_service", "delay_s"].sum() / 60)
        if len(delays) else 0.0,
        "rejected_actions": int(res.rejected_actions),
        "safety_violations": int(len(ep.safety_violations)),
    }
    out.update(slot_metrics(mon, start))
    return out
