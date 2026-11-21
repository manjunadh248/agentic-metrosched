"""Depot model: stabling, exit/reception tracks and induction capacity.

Induction capacity is limited in two independent ways (both hard rules):

* **Induction windows** – inductions may start only inside a configured
  window, and each window has its own maximum count.
* **Rolling-hour cap** – no more than ``max_inductions_per_hour`` inductions
  may start within any 3600 s span.

``authorise_induction()`` checks readiness and capacity and raises on any
violation; ``acquire_induction_slot()`` is the SimPy path that waits until an
induction is permitted.
"""

from __future__ import annotations

import logging

import simpy

from .exceptions import InductionCapacityError
from .track import TrackSection
from .train import Train, TrainLocation

logger = logging.getLogger(__name__)

HOUR = 3600


class Depot:
    def __init__(self, env: simpy.Environment | None, cfg: dict, extra_headway_s: int = 0,
                 induction_capacity_factor: float = 1.0):
        dcfg = cfg["depot"]
        self.env = env
        self.stabling_lines = int(dcfg["stabling_lines"])
        self.berths_per_line = int(dcfg["berths_per_line"])
        self.exit_tracks = [
            TrackSection(t["id"], "exit", int(t["traversal_s"]), int(t["headway_s"]) + extra_headway_s, env)
            for t in dcfg["exit_tracks"]
        ]
        rt = dcfg["reception_track"]
        self.reception_track = TrackSection(rt["id"], "reception", int(rt["traversal_s"]),
                                            int(rt["headway_s"]) + extra_headway_s, env)
        factor = max(0.0, float(induction_capacity_factor))
        self.max_per_hour = max(1, int(round(dcfg["max_inductions_per_hour"] * factor)))
        self.windows = [
            {"id": w["id"], "start": int(w["start"]), "end": int(w["end"]),
             "max_inductions": int(round(w["max_inductions"] * factor))}
            for w in sorted(dcfg["induction_windows"], key=lambda w: w["start"])
        ]
        self.induction_starts: list[int] = []
        self.window_counts: dict[str, int] = {w["id"]: 0 for w in self.windows}
        self._capacity_lock = simpy.Resource(env, capacity=1) if env is not None else None

    # ------------------------------------------------------------- stabling
    def berth_names(self) -> list[str]:
        letters = "ABCDEFGHIJ"
        return [f"SL{line + 1:02d}-{letters[b]}"
                for line in range(self.stabling_lines) for b in range(self.berths_per_line)]

    # ------------------------------------------------------------- capacity
    def window_at(self, t: int) -> dict | None:
        for w in self.windows:
            if w["start"] <= t < w["end"]:
                return w
        return None

    def inductions_last_hour(self, t: int) -> int:
        return sum(1 for s in self.induction_starts if t - HOUR < s <= t)

    def capacity_issues(self, t: int) -> list[str]:
        issues = []
        w = self.window_at(t)
        if w is None:
            issues.append("outside induction windows")
        elif self.window_counts[w["id"]] >= w["max_inductions"]:
            issues.append(f"window {w['id']} capacity {w['max_inductions']} exhausted")
        if self.inductions_last_hour(t) >= self.max_per_hour:
            issues.append(f"rolling-hour cap {self.max_per_hour} reached")
        return issues

    def remaining_capacity(self, t: int) -> int:
        """Inductions still permitted right now (min of window and hourly limits)."""
        w = self.window_at(t)
        if w is None:
            return 0
        return max(0, min(w["max_inductions"] - self.window_counts[w["id"]],
                          self.max_per_hour - self.inductions_last_hour(t)))

    def next_capacity_time(self, t: int) -> int | None:
        """Earliest time >= t at which capacity_issues() is empty, or None today."""
        cand = int(t)
        for _ in range(10_000):
            w = self.window_at(cand)
            if w is None:
                later = [x for x in self.windows if x["start"] > cand]
                if not later:
                    return None
                cand = later[0]["start"]
                continue
            if self.window_counts[w["id"]] >= w["max_inductions"]:
                later = [x for x in self.windows if x["start"] >= w["end"]]
                if not later:
                    return None
                cand = later[0]["start"]
                continue
            recent = sorted(s for s in self.induction_starts if cand - HOUR < s <= cand)
            if len(recent) >= self.max_per_hour:
                # the oldest counted induction leaves the (t-1h, t] window at s + 1h
                cand = recent[len(recent) - self.max_per_hour] + HOUR
                continue
            return cand
        return None

    def authorise_induction(self, train: Train, t: int) -> None:
        """Hard check: train ready AND capacity available, else raise. Records the start."""
        train.validate_ready(t)
        issues = self.capacity_issues(t)
        if issues:
            raise InductionCapacityError(f"Induction of {train.train_id} refused at t={t}: "
                                         + "; ".join(issues))
        w = self.window_at(t)
        self.window_counts[w["id"]] += 1
        self.induction_starts.append(int(t))

    def acquire_induction_slot(self, train: Train):
        """SimPy process: wait until capacity exists, then authorise. Returns wait_s or None.

        Returns None if no further capacity exists today (caller abandons the
        induction and the train stays in the depot).
        """
        if self.env is None or self._capacity_lock is None:
            raise RuntimeError("Depot has no SimPy environment")
        t0 = self.env.now
        with self._capacity_lock.request() as req:
            yield req
            while True:
                nxt = self.next_capacity_time(int(self.env.now))
                if nxt is None:
                    logger.debug("t=%s no induction capacity left today for %s", self.env.now,
                                train.train_id)
                    return None
                if nxt > self.env.now:
                    yield self.env.timeout(nxt - self.env.now)
                    continue
                self.authorise_induction(train, int(self.env.now))
                return int(self.env.now - t0)

    def select_exit_track(self, t: int) -> TrackSection:
        """Exit track that can be entered soonest (ties broken by id)."""
        def key(tr: TrackSection):
            busy = 1 if tr.occupant is not None else 0
            return (busy + tr.queue_length(), tr.earliest_entry(t), tr.section_id)
        return min(self.exit_tracks, key=key)

    @staticmethod
    def trains_in_depot(trains: list[Train]) -> list[Train]:
        return [tr for tr in trains if tr.location is TrainLocation.DEPOT]
