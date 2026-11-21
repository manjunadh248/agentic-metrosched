"""SimPy discrete-event model of one metro operating day.

One :class:`MetroEnvironment` instance == one episode == one operating day
from ``depot_open`` to ``service_end``. Simulation time is integer seconds
from midnight.

Induction route (every inducting train, in order)::

    depot berth -> EXIT track (one of N) -> shared sections (in sequence)
                -> mainline ENTRY junction (UP or DN) -> platform -> revenue service

Block-signalling semantics: a train acquires the next section *before*
releasing the current one, so it is always protected by at least one block.
Each acquisition goes through ``TrackSection.reserve``, which enforces
occupancy, minimum headway and closures; induction start goes through
``Depot.acquire_induction_slot`` (readiness, windows, rolling-hour cap).
Withdrawals use a separate reception track, so there are no opposing moves on
the induction route and no deadlock.

Revenue service: a running train alternates UP/DN legs of half a cycle each
and passes the corresponding entry junction once per leg, so the more trains
are in service, the busier the junctions are for later inductions. Random
trip delays and in-service failures are drawn from the environment's own
seeded generator.

Dispatch control: every ``decision_interval_s`` the controller compares the
trains committed to service with the demand-derived requirement
``induction_lookahead_s`` ahead, opens/cancels induction *requests*, and asks a
pluggable ``dispatch_policy`` which ready trains to induct. The policy can only
pick from the list of trains that are ready right now; anything else is
rejected. Phase 1 ships ``readiness_order_policy`` purely so that the data
generator can produce an induction history; the benchmark FCFS baseline,
heuristic and PPO schedulers arrive in later phases and plug in here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
import simpy

from .config import seconds_to_clock
from .demand_generator import required_trains_at
from .depot import Depot
from .exceptions import SafetyViolationError, TrainNotReadyError
from .scenario_generator import Scenario
from .track import PlatformGroup, TrackSection
from .train import Train, TrainLocation, TrainStatus

logger = logging.getLogger(__name__)

DispatchPolicy = Callable[[list[Train], int, dict], list[str]]


def readiness_order_policy(candidates: list[Train], n: int, state: dict) -> list[str]:
    """Data-generation dispatch rule: earliest readiness first, then priority, then id."""
    ordered = sorted(candidates, key=lambda t: (t.readiness_time, t.induction_priority, t.train_id))
    return [t.train_id for t in ordered[:n]]


@dataclass
class SimulationResult:
    day: int
    scenario_type: str
    inductions: pd.DataFrame
    maintenance_events: pd.DataFrame
    delays: pd.DataFrame
    snapshots: pd.DataFrame
    occupancy: pd.DataFrame
    closures: pd.DataFrame
    unfulfilled_requests: int
    rejected_actions: int
    monitor: pd.DataFrame | None = None
    decisions: pd.DataFrame | None = None
    policy: str = ""


class MetroEnvironment:
    def __init__(self, cfg: dict, scenario: Scenario, demand_df: pd.DataFrame, trains: list[Train],
                 rng: np.random.Generator, dispatch_policy: DispatchPolicy | None = None,
                 policy_name: str = "readiness_order", control: str = "internal"):
        if not trains:
            raise ValueError("MetroEnvironment needs at least one train")
        if demand_df.empty:
            raise ValueError("demand_df is empty")
        self.cfg = cfg
        self.scenario = scenario
        self.demand_df = demand_df
        self.rng = rng
        self.policy = dispatch_policy or readiness_order_policy
        self.policy_name = policy_name
        if control not in ("internal", "external"):
            raise ValueError("control must be 'internal' or 'external'")
        # internal: the built-in controller opens requests and calls dispatch_policy.
        # external: a scheduler (FCFS / heuristic / PPO) calls apply_induction() at
        #           each decision time; the environment still enforces every rule.
        self.control = control
        self._started = False
        sim = cfg["simulation"]
        self.depot_open = int(sim["depot_open"])
        self.service_start = int(sim["service_start"])
        self.service_end = int(sim["service_end"])
        self.decision_interval = int(sim["decision_interval_s"])
        self.interval_s = int(sim["interval_s"])
        self.env = simpy.Environment(initial_time=self.depot_open)

        extra = int(scenario.extra_headway_s)
        self.depot = Depot(self.env, cfg, extra_headway_s=extra,
                           induction_capacity_factor=scenario.induction_capacity_factor)
        self.shared_sections = [
            TrackSection(s["id"], "shared", int(s["traversal_s"]), int(s["headway_s"]) + extra, self.env)
            for s in cfg["track"]["shared_sections"]
        ]
        self.entry_points = [
            TrackSection(e["id"], "entry", int(e["traversal_s"]), int(e["headway_s"]) + extra, self.env,
                         direction=e["direction"])
            for e in cfg["track"]["entry_points"]
        ]
        pcfg = cfg["platforms"]
        self.platforms = PlatformGroup(self.env, int(pcfg["count"]), int(pcfg["induction_dwell_s"]),
                                       int(pcfg["clearance_s"]) + extra)
        self.sections: dict[str, TrackSection] = {}
        for sec in (self.depot.exit_tracks + [self.depot.reception_track] + self.shared_sections
                    + self.entry_points + self.platforms.platforms):
            self.sections[sec.section_id] = sec
        for sid, s, e in (scenario.exit_track_closures + scenario.section_blocks
                          + scenario.platform_closures):
            if sid not in self.sections:
                raise ValueError(f"Scenario closes unknown section {sid}")
            self.sections[sid].add_closure(s, e)

        svc = cfg["service"]
        self.leg_time = int(svc["one_way_run_time_s"] + svc["terminal_turnaround_s"])
        self.lookahead = int(svc["induction_lookahead_s"])
        self.withdraw_lookahead = int(svc["withdrawal_lookahead_s"])
        self.hysteresis = int(svc["withdrawal_hysteresis"])
        self.through_occ = max(1, int(round(cfg["track"]["through_train_occupancy_s"]
                                            * scenario.through_traffic_factor)))
        ops = cfg["operations"]
        self.leg_delay_prob = min(1.0, ops["trip_delay_prob"] * scenario.trip_delay_factor / 2)
        self.trip_delay_mean = float(ops["trip_delay_mean_s"])
        self.leg_failure_prob = ops["in_service_failure_prob"] / 2
        self.km_threshold = float(cfg["fleet"]["maintenance_km_threshold"])
        self.repair_range = ops["repair_time_s"]
        self.depot_turnaround = int(cfg["fleet"]["depot_turnaround_s"])
        self.nominal_route_s = (max(t.traversal_s for t in self.depot.exit_tracks)
                                + sum(s.traversal_s for s in self.shared_sections)
                                + max(e.traversal_s for e in self.entry_points)
                                + self.platforms.dwell_s)

        self.trains: dict[str, Train] = {t.train_id: t for t in trains}
        self.inducting: list[str] = []
        self.withdrawing: list[str] = []
        self.open_requests: list[tuple[int, int]] = []  # (opened_at, needed_by)
        self.unfulfilled_requests = 0
        self.rejected_actions = 0
        self._induction_seq = 0
        self._entry_rr = 0
        self._first_induction_done: list[str] = []
        self.induction_log: list[dict] = []
        self.maintenance_log: list[dict] = []
        self.delay_log: list[dict] = []
        self.snapshot_log: list[dict] = []
        self.monitor_log: list[dict] = []
        self.waiting: list[str] = []          # trains currently queued for a block/platform
        self.decision_log: list[dict] = []    # external-control decisions (accepted + rejected)
        # Common random numbers: each train has its own stream for trip delays,
        # failures and repairs, so different schedulers face the same random events.
        train_seqs = np.random.SeedSequence(int(scenario.seed) + 7_919).spawn(len(trains))
        self.train_rngs = {tid: np.random.default_rng(sq)
                           for tid, sq in zip(sorted(t.train_id for t in trains), train_seqs)}

        # Pre-draw depot faults with the environment generator (deterministic order).
        self._depot_faults: list[tuple[str, int]] = []
        for tid in sorted(self.trains):
            if self.rng.random() < scenario.depot_fault_prob:
                self._depot_faults.append(
                    (tid, int(self.rng.integers(self.depot_open, self.service_end - 2 * 3600))))

    # ================================================================ helpers
    def now(self) -> int:
        return int(self.env.now)

    def required_at(self, t: int) -> int:
        return required_trains_at(self.demand_df, t)

    def ready_candidates(self) -> list[Train]:
        t = self.now()
        return [tr for tid, tr in sorted(self.trains.items())
                if tid not in self.inducting and tr.is_ready(t)]

    def committed_trains(self) -> int:
        on_line = sum(1 for tid, tr in self.trains.items()
                      if tr.location is TrainLocation.MAINLINE and tid not in self.withdrawing)
        return on_line + len(self.inducting)

    def state(self) -> dict:
        """Observation-style snapshot handed to dispatch policies."""
        t = self.now()
        return {
            "time_s": t,
            "required_now": self.required_at(t),
            "required_ahead": self.required_at(t + self.lookahead),
            "committed": self.committed_trains(),
            "open_requests": len(self.open_requests),
            "remaining_induction_capacity": self.depot.remaining_capacity(t),
            "section_occupancy": {sid: s.occupant for sid, s in self.sections.items()},
            "platforms_free": len(self.platforms.free_platforms(t)),
            "delayed_trains": sum(1 for tr in self.trains.values() if tr.status is TrainStatus.DELAYED),
        }

    def _log_delay(self, train_id: str, kind: str, start: int, delay_s: int, detail: str = "") -> None:
        self.delay_log.append({"train_id": train_id, "delay_type": kind, "start_time_s": int(start),
                               "delay_s": int(delay_s), "detail": detail})

    # ============================================================ maintenance
    def _repair(self, train: Train, reason: str, source: str):
        start = self.now()
        dur = int(self.train_rngs[train.train_id].integers(self.repair_range[0], self.repair_range[1] + 1))
        if train.status is not TrainStatus.MAINTENANCE:
            train.transition(TrainStatus.MAINTENANCE, start, TrainLocation.DEPOT)
        train.location = TrainLocation.DEPOT
        rec = {"train_id": train.train_id, "event_type": source, "reason": reason,
               "start_time_s": start, "planned_duration_s": dur, "end_time_s": None}
        self.maintenance_log.append(rec)
        yield self.env.timeout(dur)
        train.maintenance_required = False
        train.transition(TrainStatus.AVAILABLE, self.now(), TrainLocation.DEPOT)
        train.readiness_time = self.now() + train.min_prep_time_s
        rec["end_time_s"] = self.now()
        logger.debug("t=%s %s repaired (%s)", self.now(), train.train_id, reason)

    def _depot_fault(self, train_id: str, at: int):
        yield self.env.timeout(max(0, at - self.now()))
        tr = self.trains[train_id]
        if (tr.location is TrainLocation.DEPOT and train_id not in self.inducting
                and tr.status in (TrainStatus.AVAILABLE, TrainStatus.DELAYED)):
            yield from self._repair(tr, "unexpected depot fault", "depot_fault")

    def _prep_watch(self, train: Train):
        sched = train.scheduled_induction_time
        if sched is None or train.readiness_time <= sched:
            return
        yield self.env.timeout(max(0, sched - self.now()))
        if train.status is TrainStatus.AVAILABLE and train.location is TrainLocation.DEPOT \
                and not train.is_ready(self.now()):
            train.transition(TrainStatus.DELAYED, self.now(), TrainLocation.DEPOT)

    # ============================================================== induction
    def _induction(self, train: Train, request: tuple[int, int], entry: TrackSection | None = None,
                   pre_authorised: bool = False):
        request_time, needed_by = request
        tid = train.train_id
        dispatch_t = self.now()
        try:
            if pre_authorised:
                cap_wait = 0
            else:
                cap_wait = yield from self.depot.acquire_induction_slot(train)
        except TrainNotReadyError as exc:
            logger.warning("Induction aborted: %s", exc)
            self.inducting.remove(tid)
            self.open_requests.insert(0, request)
            return
        if cap_wait is None:
            self.inducting.remove(tid)
            self.unfulfilled_requests += 1
            return
        auth_t = self.now()
        train.transition(TrainStatus.INDUCTED, auth_t, TrainLocation.TRANSIT)
        exit_track = self.depot.select_exit_track(auth_t)
        if entry is None:
            entry = self._choose_entry(auth_t)
        route = [exit_track] + self.shared_sections + [entry]
        track_wait, current, exit_time, entry_time = 0, None, None, None
        for sec in route:
            blocked = (not sec.is_available(self.now())) or sec.queue_length() > 0
            if blocked:
                self.waiting.append(tid)
            track_wait += yield from sec.reserve(tid, "induction")
            if blocked:
                self.waiting.remove(tid)
            if current is not None:
                current.release(tid, self.now())
                if current is exit_track:
                    exit_time = self.now()
            yield self.env.timeout(sec.traversal_s)
            current = sec
        pf_blocked = not self.platforms.free_platforms(self.now())
        if pf_blocked:
            self.waiting.append(tid)
        platform, pf_wait = yield from self.platforms.reserve(tid)
        if pf_blocked:
            self.waiting.remove(tid)
        track_wait += pf_wait
        current.release(tid, self.now())
        if exit_time is None:
            exit_time = self.now()
        entry_time = self.now()
        yield self.env.timeout(self.platforms.dwell_s)
        platform.release(tid, self.now())
        service_t = self.now()
        train.transition(TrainStatus.RUNNING, service_t, TrainLocation.MAINLINE)
        self.inducting.remove(tid)
        first = tid not in self._first_induction_done
        if first:
            self._first_induction_done.append(tid)
        sched = train.scheduled_induction_time if first else None
        self._induction_seq += 1
        self.induction_log.append({
            "induction_id": self._induction_seq, "train_id": tid, "policy": self.policy_name,
            "request_time_s": request_time, "needed_by_time_s": needed_by, "dispatch_time_s": dispatch_t, "authorised_time_s": auth_t,
            "exit_track": exit_track.section_id, "depot_exit_time_s": exit_time,
            "entry_point": entry.section_id, "mainline_entry_time_s": entry_time,
            "platform": platform.section_id, "service_start_time_s": service_t,
            "nominal_route_s": self.nominal_route_s, "actual_route_s": service_t - auth_t,
            "capacity_wait_s": cap_wait, "track_wait_s": track_wait,
            "induction_delay_s": max(0, service_t - needed_by),
            "scheduled_induction_time_s": sched,
            "schedule_deviation_s": (auth_t - sched) if sched is not None else None,
            "first_induction_of_day": first,
        })
        if track_wait > 0:
            self._log_delay(tid, "induction_track_wait", auth_t, track_wait, entry.section_id)
        self.env.process(self._service(train, entry.direction))

    def _choose_entry(self, t: int) -> TrackSection:
        """Entry junction that clears soonest; round-robin breaks ties."""
        n = len(self.entry_points)
        order = [self.entry_points[(self._entry_rr + i) % n] for i in range(n)]
        best = min(order, key=lambda e: (e.earliest_entry(t), e.queue_length()))
        self._entry_rr = (self.entry_points.index(best) + 1) % n
        return best

    # ================================================================ service
    def _service(self, train: Train, direction: str):
        tid = train.train_id
        trng = self.train_rngs[tid]
        dirs = [e.direction for e in self.entry_points]
        # Worn trains (high km since maintenance) fail more often: 0.5x .. 1.5x+.
        fail_p = self.leg_failure_prob * (0.5 + train.km_since_maintenance / self.km_threshold)
        while self.now() < self.service_end:
            yield self.env.timeout(max(1, self.leg_time - self.through_occ))
            if trng.random() < self.leg_delay_prob:
                d = max(1, int(round(trng.exponential(self.trip_delay_mean))))
                train.transition(TrainStatus.DELAYED, self.now())
                self._log_delay(tid, "in_service", self.now(), d, f"leg {direction}")
                yield self.env.timeout(d)
                train.transition(TrainStatus.RUNNING, self.now())
            if trng.random() < fail_p:
                yield from self._return_to_depot(train, failure=True)
                return
            if tid in self.withdrawing:
                yield from self._return_to_depot(train, failure=False)
                return
            direction = dirs[(dirs.index(direction) + 1) % len(dirs)] if direction in dirs else dirs[0]
            junction = next(e for e in self.entry_points if e.direction == direction)
            wait = yield from junction.reserve(tid, "through")
            if wait > 0:
                self._log_delay(tid, "junction_congestion", self.now() - wait, wait, junction.section_id)
            yield self.env.timeout(self.through_occ)
            junction.release(tid, self.now())

    def _return_to_depot(self, train: Train, failure: bool):
        tid = train.train_id
        rt = self.depot.reception_track
        wait = yield from rt.reserve(tid, "failure" if failure else "withdrawal")
        if wait > 0:
            self._log_delay(tid, "reception_wait", self.now() - wait, wait, rt.section_id)
        train.location = TrainLocation.TRANSIT
        yield self.env.timeout(rt.traversal_s)
        rt.release(tid, self.now())
        if tid in self.withdrawing:
            self.withdrawing.remove(tid)
        if failure:
            yield from self._repair(train, "in-service failure", "in_service_failure")
        else:
            train.transition(TrainStatus.AVAILABLE, self.now(), TrainLocation.DEPOT)
            train.readiness_time = self.now() + self.depot_turnaround

    # ============================================================= controller
    def _controller(self):
        # Start at the first induction window so the opening service is built up
        # before service_start; before service the target is the 05:00 requirement.
        first_window = min(w["start"] for w in self.depot.windows)
        start = max(self.depot_open, min(first_window, self.service_start - self.lookahead))
        yield self.env.timeout(max(0, start - self.now()))
        while self.now() < self.service_end:
            t = self.now()
            if self.control == "external":
                if t >= self.service_start:
                    self._maybe_withdraw(t)
                yield self.env.timeout(self.decision_interval)
                continue
            target_t = max(t + self.lookahead, self.service_start)
            target = self.required_at(min(target_t, self.service_end - 1))
            outstanding = self.committed_trains() + len(self.open_requests)
            if target > outstanding:
                self.open_requests.extend([(t, int(target_t))] * (target - outstanding))
            while target < self.committed_trains() + len(self.open_requests) and self.open_requests:
                self.open_requests.pop()
            if self.open_requests:
                self._dispatch()
            elif t >= self.service_start:
                self._maybe_withdraw(t)
            yield self.env.timeout(self.decision_interval)

    def _dispatch(self) -> None:
        candidates = self.ready_candidates()
        if not candidates:
            return
        n = min(len(self.open_requests), len(candidates))
        try:
            picks = list(self.policy(candidates, n, self.state()))
        except Exception:  # a buggy policy must not crash the simulation
            logger.exception("Dispatch policy %s raised; skipping decision", self.policy_name)
            self.rejected_actions += 1
            return
        valid_ids = [c.train_id for c in candidates]
        chosen: list[str] = []
        for tid in picks[:n]:
            if tid not in valid_ids or tid in chosen:
                self.rejected_actions += 1
                logger.warning("Rejected invalid dispatch of %s at t=%s", tid, self.now())
                continue
            chosen.append(tid)
        for tid in chosen:
            req_t = self.open_requests.pop(0)
            self.inducting.append(tid)
            self.env.process(self._induction(self.trains[tid], req_t))

    def _maybe_withdraw(self, t: int) -> None:
        horizon = range(t, min(t + self.withdraw_lookahead, self.service_end), self.interval_s)
        need = max((self.required_at(x) for x in horizon), default=self.required_at(t))
        if self.committed_trains() <= need + self.hysteresis:
            return
        running = [tr for tid, tr in sorted(self.trains.items())
                   if tr.status is TrainStatus.RUNNING and tid not in self.withdrawing]
        if not running:
            return
        victim = max(running, key=lambda tr: (tr.induction_priority, tr.km_since_maintenance, tr.train_id))
        self.withdrawing.append(victim.train_id)

    # ============================================================== snapshots
    def _snapshots(self):
        yield self.env.timeout(max(0, self.service_start - self.now()))
        while self.now() < self.service_end:
            t = self.now()
            for tid, tr in sorted(self.trains.items()):
                self.snapshot_log.append({"time_s": t, "train_id": tid, "status": tr.status.value,
                                          "location": tr.location.value})
            yield self.env.timeout(self.interval_s)

    def in_service_count(self) -> int:
        return sum(1 for tr in self.trains.values() if tr.location is TrainLocation.MAINLINE)

    def _monitor(self):
        """Per-decision-tick service log used by the evaluation metrics."""
        yield self.env.timeout(max(0, self.controller_start() - self.now()))
        while self.now() < self.service_end:
            t = self.now()
            self.monitor_log.append({
                "time_s": t, "required_now": self.required_at(t), "in_service": self.in_service_count(),
                "committed": self.committed_trains(), "inducting": len(self.inducting),
                "waiting": len(self.waiting), "withdrawing": len(self.withdrawing),
                "delayed": sum(1 for tr in self.trains.values()
                               if tr.status is TrainStatus.DELAYED and tr.location is TrainLocation.MAINLINE),
            })
            yield self.env.timeout(self.decision_interval)

    def controller_start(self) -> int:
        first_window = min(w["start"] for w in self.depot.windows)
        return max(self.depot_open, min(first_window, self.service_start - self.lookahead))

    # ====================================================== external control
    def decision_times(self) -> list[int]:
        return list(range(self.controller_start(), self.service_end, self.decision_interval))

    def advance_to(self, t: int) -> None:
        """Run the simulation up to (not including events at) time ``t``."""
        if not self._started:
            self.start()
        if t > self.now():
            self.env.run(until=min(int(t), self.service_end))

    def apply_induction(self, train_id: str, entry_id: str | None = None) -> tuple[bool, str]:
        """External-control action: start inducting ``train_id`` now.

        Rejected (returns ``(False, reason)``, nothing happens) if the train is
        not ready, the entry point is unknown, or no induction capacity is left
        right now. Track/platform safety is then enforced by the route itself.
        """
        if self.control != "external":
            raise RuntimeError("apply_induction() requires control='external'")
        if not self._started:
            self.start()
        t = self.now()
        reason = ""
        tr = self.trains.get(train_id)
        entry = None
        if tr is None:
            reason = "unknown train"
        elif train_id in self.inducting:
            reason = "already being inducted"
        elif not tr.is_ready(t):
            reason = "; ".join(tr.readiness_issues(t))
        elif t >= self.service_end:
            reason = "service over"
        if not reason and entry_id is not None:
            entry = next((e for e in self.entry_points if e.section_id == entry_id), None)
            if entry is None:
                reason = f"unknown entry point {entry_id}"
        if not reason:
            # Hard check + atomic capacity booking (readiness, window, rolling hour).
            try:
                self.depot.authorise_induction(tr, t)
            except SafetyViolationError as exc:
                reason = str(exc)
        ok = not reason
        self.decision_log.append({"time_s": t, "train_id": train_id, "entry_point": entry_id,
                                  "accepted": ok, "reason": reason or "accepted"})
        if not ok:
            self.rejected_actions += 1
            return False, reason
        self.inducting.append(train_id)
        self.env.process(self._induction(tr, (t, t), entry, pre_authorised=True))
        return True, "accepted"

    # ==================================================================== run
    def start(self) -> None:
        if self._started:
            return
        self._started = True
        for tid, tr in sorted(self.trains.items()):
            if tr.maintenance_required:
                self.env.process(self._repair(tr, "scheduled maintenance (odometer)", "planned"))
            else:
                self.env.process(self._prep_watch(tr))
        for tid, at in self._depot_faults:
            self.env.process(self._depot_fault(tid, at))
        self.env.process(self._controller())
        self.env.process(self._snapshots())
        self.env.process(self._monitor())

    def run(self) -> SimulationResult:
        """Run the whole day (internal control) and return the result."""
        self.start()
        try:
            self.env.run(until=self.service_end)
        except SafetyViolationError:
            logger.exception("Safety violation escaped the simulation (this is a bug)")
            raise
        return self._result()

    def finish(self) -> SimulationResult:
        """Run any remaining time to service end and return the result (external control)."""
        self.advance_to(self.service_end)
        return self._result()

    def _result(self) -> SimulationResult:
        occ = [{"section_id": r.section_id, "kind": self.sections[r.section_id].kind,
                "train_id": r.train_id, "start_s": r.start,
                "end_s": r.end if r.end is not None else self.service_end, "purpose": r.purpose}
               for sec in self.sections.values() for r in sec.log]
        clo = [{"section_id": sid, "start_s": s, "end_s": e}
               for sid, sec in self.sections.items() for s, e in sec.closures]
        for rec in self.maintenance_log:
            if rec["end_time_s"] is None:
                rec["end_time_s"] = rec["start_time_s"] + rec["planned_duration_s"]

        def frame(rows: list[dict], cols: list[str]) -> pd.DataFrame:
            return pd.DataFrame(rows, columns=cols) if not rows else pd.DataFrame(rows)

        return SimulationResult(
            day=self.scenario.day, scenario_type=self.scenario.scenario_type,
            inductions=frame(self.induction_log, ["induction_id", "train_id"]),
            maintenance_events=frame(self.maintenance_log, ["train_id", "event_type"]),
            delays=frame(self.delay_log, ["train_id", "delay_type", "start_time_s", "delay_s", "detail"]),
            snapshots=frame(self.snapshot_log, ["time_s", "train_id", "status", "location"]),
            occupancy=frame(occ, ["section_id", "kind", "train_id", "start_s", "end_s", "purpose"]),
            closures=frame(clo, ["section_id", "start_s", "end_s"]),
            unfulfilled_requests=self.unfulfilled_requests + len(self.open_requests),
            rejected_actions=self.rejected_actions,
            monitor=pd.DataFrame(self.monitor_log),
            decisions=pd.DataFrame(self.decision_log, columns=["time_s", "train_id", "entry_point",
                                                               "accepted", "reason"]),
            policy=self.policy_name,
        )


def clock(t: int) -> str:
    return seconds_to_clock(t)
