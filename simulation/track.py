"""Track sections, platforms and safety-separation enforcement.

A :class:`TrackSection` is a block that can hold at most one train. Entry is
only permitted when ALL of the following hold (hard constraints):

1. the section is not occupied by another train;
2. at least ``headway_s`` seconds have passed since the previous train cleared
   it (minimum safety separation);
3. the section is not inside a closure window (possession / block).

``occupy()`` raises a :class:`SafetyViolationError` subclass if any rule is
broken, so an unsafe movement is *impossible*, not merely discouraged.
``reserve()`` is the SimPy-friendly path: it waits (queues) until entry is
legal and then occupies the section.

Every occupation is logged as an interval so that utilisation can be
aggregated and independently audited with :func:`audit_occupancy`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import simpy

from .exceptions import (
    HeadwayViolationError,
    PlatformCapacityError,
    SectionClosedError,
    TrackConflictError,
)

logger = logging.getLogger(__name__)


@dataclass
class OccupancyRecord:
    section_id: str
    train_id: str
    start: int
    end: int | None = None
    purpose: str = ""


@dataclass
class TrackSection:
    """A single-train block with safety headway and optional closure windows."""

    section_id: str
    kind: str
    traversal_s: int
    headway_s: int
    env: simpy.Environment | None = None
    direction: str | None = None
    closures: list[tuple[int, int]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.traversal_s <= 0 or self.headway_s < 0:
            raise ValueError(f"{self.section_id}: invalid traversal/headway")
        self.occupant: str | None = None
        self.last_release: int | None = None
        self.log: list[OccupancyRecord] = []
        self._current: OccupancyRecord | None = None
        self._resource = simpy.Resource(self.env, capacity=1) if self.env is not None else None
        self._request = None

    # ----------------------------------------------------------- inspection
    def add_closure(self, start: int, end: int) -> None:
        if end <= start:
            raise ValueError("closure end must be after start")
        self.closures.append((int(start), int(end)))
        self.closures.sort()

    def is_closed(self, t: int) -> bool:
        return any(s <= t < e for s, e in self.closures)

    def closure_end(self, t: int) -> int | None:
        for s, e in self.closures:
            if s <= t < e:
                return e
        return None

    def headway_clear_time(self) -> int:
        """Earliest time the next train may enter based on headway alone."""
        if self.last_release is None:
            return 0
        return self.last_release + self.headway_s

    def entry_issues(self, t: int) -> list[str]:
        issues = []
        if self.occupant is not None:
            issues.append(f"occupied by {self.occupant}")
        if t < self.headway_clear_time():
            issues.append(f"headway not elapsed until t={self.headway_clear_time()}")
        if self.is_closed(t):
            issues.append(f"closed until t={self.closure_end(t)}")
        return issues

    def is_available(self, t: int) -> bool:
        return not self.entry_issues(t)

    def earliest_entry(self, t: int) -> int:
        """Earliest time >= t at which an *unoccupied* section can be entered."""
        cand = max(int(t), self.headway_clear_time())
        while True:
            end = self.closure_end(cand)
            if end is None:
                return cand
            cand = end

    # ----------------------------------------------------------- hard rules
    def occupy(self, train_id: str, t: int, purpose: str = "") -> None:
        """Place ``train_id`` in the section at time ``t`` or raise."""
        if self.occupant is not None:
            raise TrackConflictError(
                f"{self.section_id} conflict at t={t}: {train_id} vs occupant {self.occupant}")
        if t < self.headway_clear_time():
            raise HeadwayViolationError(
                f"{self.section_id} headway violation at t={t}: {train_id} must wait until "
                f"t={self.headway_clear_time()} (headway {self.headway_s}s)")
        if self.is_closed(t):
            raise SectionClosedError(f"{self.section_id} closed at t={t}")
        self.occupant = train_id
        self._current = OccupancyRecord(self.section_id, train_id, int(t), None, purpose)
        self.log.append(self._current)

    def release(self, train_id: str, t: int) -> None:
        if self.occupant != train_id:
            raise TrackConflictError(
                f"{self.section_id}: {train_id} tried to release a section held by {self.occupant}")
        assert self._current is not None
        self._current.end = int(t)
        self.occupant = None
        self.last_release = int(t)
        self._current = None
        if self._resource is not None and self._request is not None:
            self._resource.release(self._request)
            self._request = None

    # ---------------------------------------------------------------- SimPy
    def reserve(self, train_id: str, purpose: str = ""):
        """SimPy process: queue for the section, wait for headway/closure, occupy.

        Usage inside a process: ``yield from section.reserve(train_id)``.
        Returns the number of seconds spent waiting.
        """
        if self.env is None or self._resource is None:
            raise RuntimeError(f"{self.section_id} has no SimPy environment")
        t0 = self.env.now
        req = self._resource.request()
        yield req
        self._request = req
        while True:
            entry = self.earliest_entry(self.env.now)
            if entry > self.env.now:
                yield self.env.timeout(entry - self.env.now)
                continue
            break
        self.occupy(train_id, int(self.env.now), purpose)
        return int(self.env.now - t0)

    def queue_length(self) -> int:
        return len(self._resource.queue) if self._resource is not None else 0


class Platform(TrackSection):
    """A platform track: same rules as a section (one train, clearance, closures)."""


class PlatformGroup:
    """The set of platforms at the induction (depot-adjacent) station."""

    def __init__(self, env: simpy.Environment | None, count: int, dwell_s: int, clearance_s: int):
        if count < 1:
            raise ValueError("Need at least one platform")
        self.env = env
        self.dwell_s = int(dwell_s)
        self.platforms = [
            Platform(f"PF-{i + 1}", "platform", traversal_s=max(1, int(dwell_s)),
                     headway_s=int(clearance_s), env=env)
            for i in range(count)
        ]

    def occupied_count(self) -> int:
        return sum(p.occupant is not None for p in self.platforms)

    def open_platforms(self, t: int) -> list[Platform]:
        return [p for p in self.platforms if not p.is_closed(t)]

    def free_platforms(self, t: int) -> list[Platform]:
        return [p for p in self.platforms if p.is_available(t)]

    def occupy_any(self, train_id: str, t: int) -> Platform:
        """Hard-constraint path: put the train on any free platform or raise."""
        free = self.free_platforms(t)
        if not free:
            raise PlatformCapacityError(
                f"No platform free at t={t} ({self.occupied_count()}/{len(self.platforms)} occupied)")
        free[0].occupy(train_id, t, "induction")
        return free[0]

    def choose(self, t: int) -> Platform:
        """Pick the platform that can accept a train soonest (ties: lowest id)."""
        def key(p: Platform):
            base = p.earliest_entry(t)
            busy = 1 if p.occupant is not None else 0
            return (busy + p.queue_length(), base, p.section_id)
        return min(self.platforms, key=key)

    def reserve(self, train_id: str):
        """SimPy process: wait for and occupy a platform; returns (platform, wait_s)."""
        platform = self.choose(int(self.env.now))
        wait = yield from platform.reserve(train_id, "induction")
        return platform, wait


# --------------------------------------------------------------------- audit
def audit_occupancy(records: list[OccupancyRecord], headway_s: int,
                    closures: list[tuple[int, int]] | None = None) -> list[dict]:
    """Independently check an occupancy log for safety violations.

    Detects (a) overlapping occupations = induction/track conflicts, (b) entries
    that start sooner than ``headway_s`` after the previous exit, and (c)
    entries inside a closure window. Returns a list of violation dicts (empty
    means the log is safe). Used by tests and later by the evaluation module to
    prove that no scheduler produced unsafe movements.
    """
    violations: list[dict] = []
    closed = closures or []
    recs = sorted(records, key=lambda r: (r.start, r.end if r.end is not None else 10 ** 9))
    for prev, cur in zip(recs, recs[1:]):
        prev_end = prev.end if prev.end is not None else 10 ** 9
        if cur.start < prev_end:
            violations.append({"type": "conflict", "section_id": cur.section_id,
                               "train_a": prev.train_id, "train_b": cur.train_id, "time": cur.start})
        elif cur.start - prev_end < headway_s:
            violations.append({"type": "headway", "section_id": cur.section_id,
                               "train_a": prev.train_id, "train_b": cur.train_id, "time": cur.start,
                               "gap_s": cur.start - prev_end})
    for r in recs:
        if any(s <= r.start < e for s, e in closed):
            violations.append({"type": "closed", "section_id": r.section_id,
                               "train_a": r.train_id, "train_b": None, "time": r.start})
    return violations
