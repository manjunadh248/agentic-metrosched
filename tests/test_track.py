"""Track availability, minimum headway enforcement and conflict detection."""

import pytest

from simulation.exceptions import HeadwayViolationError, SectionClosedError, TrackConflictError
from simulation.track import OccupancyRecord, TrackSection, audit_occupancy


def section(env=None, headway=120, traversal=60):
    return TrackSection("SEC", "shared", traversal, headway, env)


# ------------------------------------------------------------ availability
def test_fresh_section_is_available():
    s = section()
    assert s.is_available(0)
    assert s.earliest_entry(0) == 0


def test_occupied_section_unavailable_and_conflict_rejected():
    s = section()
    s.occupy("A", 100)
    assert not s.is_available(150)
    with pytest.raises(TrackConflictError):
        s.occupy("B", 150)


def test_release_by_wrong_train_rejected():
    s = section()
    s.occupy("A", 0)
    with pytest.raises(TrackConflictError):
        s.release("B", 10)


def test_closure_blocks_entry():
    s = section()
    s.add_closure(1000, 2000)
    assert not s.is_available(1500)
    assert s.earliest_entry(1500) == 2000
    with pytest.raises(SectionClosedError):
        s.occupy("A", 1500)
    s.occupy("A", 2000)


# ----------------------------------------------------------------- headway
def test_headway_enforced_as_hard_constraint():
    s = section(headway=120)
    s.occupy("A", 0)
    s.release("A", 60)
    assert not s.is_available(179)
    with pytest.raises(HeadwayViolationError):
        s.occupy("B", 179)
    assert s.earliest_entry(100) == 180
    s.occupy("B", 180)


def test_simpy_reserve_waits_for_headway(env):
    s = section(env, headway=120, traversal=60)
    entries = {}

    def train(tid, delay):
        yield env.timeout(delay)
        yield from s.reserve(tid)
        entries[tid] = env.now
        yield env.timeout(s.traversal_s)
        s.release(tid, env.now)

    env.process(train("A", 0))
    env.process(train("B", 10))   # requests while A is inside
    env.process(train("C", 20))
    env.run()
    assert entries["A"] == 0
    assert entries["B"] == 60 + 120          # A leaves at 60, headway 120
    assert entries["C"] == 180 + 60 + 120    # B leaves at 240
    assert audit_occupancy(s.log, s.headway_s) == []


def test_simpy_reserve_waits_for_closure(env):
    s = section(env, headway=0)
    s.add_closure(0, 500)
    out = {}

    def train():
        wait = yield from s.reserve("A")
        out["t"], out["wait"] = env.now, wait

    env.process(train())
    env.run()
    assert out == {"t": 500, "wait": 500}


# --------------------------------------------------------- conflict audit
def test_audit_detects_conflict_headway_and_closure():
    recs = [
        OccupancyRecord("SEC", "A", 0, 100),
        OccupancyRecord("SEC", "B", 50, 150),     # overlaps A  -> conflict
        OccupancyRecord("SEC", "C", 200, 260),    # 50 s after B -> headway (needs 120)
        OccupancyRecord("SEC", "D", 1000, 1060),  # inside closure
    ]
    v = audit_occupancy(recs, headway_s=120, closures=[(900, 1100)])
    assert {x["type"] for x in v} == {"conflict", "headway", "closed"}
    conflict = next(x for x in v if x["type"] == "conflict")
    assert (conflict["train_a"], conflict["train_b"]) == ("A", "B")


def test_audit_clean_log():
    recs = [OccupancyRecord("SEC", "A", 0, 100), OccupancyRecord("SEC", "B", 220, 300)]
    assert audit_occupancy(recs, headway_s=120) == []


def test_invalid_section_parameters():
    with pytest.raises(ValueError):
        TrackSection("X", "shared", 0, 10)
    with pytest.raises(ValueError):
        TrackSection("X", "shared", 10, -1)
