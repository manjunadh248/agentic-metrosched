"""Platform capacity and depot induction-capacity enforcement."""

import pytest
import simpy

from simulation.config import deep_merge
from simulation.depot import Depot
from simulation.exceptions import InductionCapacityError, PlatformCapacityError, TrainNotReadyError
from simulation.track import PlatformGroup
from simulation.train import Train


# ---------------------------------------------------------------- platforms
def test_platform_capacity_hard_limit():
    pg = PlatformGroup(None, count=2, dwell_s=60, clearance_s=30)
    pg.occupy_any("A", 0)
    pg.occupy_any("B", 0)
    assert pg.occupied_count() == 2
    with pytest.raises(PlatformCapacityError):
        pg.occupy_any("C", 0)


def test_closed_platform_not_used():
    pg = PlatformGroup(None, count=2, dwell_s=60, clearance_s=30)
    pg.platforms[0].add_closure(0, 1000)
    assert pg.occupy_any("A", 10).section_id == "PF-2"
    with pytest.raises(PlatformCapacityError):
        pg.occupy_any("B", 10)


def test_simpy_platform_occupancy_never_exceeds_count():
    env = simpy.Environment()
    pg = PlatformGroup(env, count=2, dwell_s=60, clearance_s=30)
    peak = {"max": 0}

    def train(tid):
        platform, _ = yield from pg.reserve(tid)
        peak["max"] = max(peak["max"], pg.occupied_count())
        yield env.timeout(pg.dwell_s)
        platform.release(tid, env.now)

    for i in range(6):
        env.process(train(f"T{i}"))
    env.run()
    assert peak["max"] <= 2
    for p in pg.platforms:
        recs = sorted(p.log, key=lambda r: r.start)
        for a, b in zip(recs, recs[1:]):
            assert b.start >= a.end + p.headway_s


# -------------------------------------------------------------------- depot
def _train(ready=0, maint=False):
    return Train("TS01", "SL01-A", readiness_time=ready, min_prep_time_s=0, maintenance_required=maint)


def small_depot_cfg(cfg):
    return deep_merge(cfg, {"depot": {"max_inductions_per_hour": 2, "induction_windows": [
        {"id": "W", "start": 1000, "end": 5000, "max_inductions": 3}]}})


def test_induction_refused_outside_window(cfg):
    d = Depot(None, small_depot_cfg(cfg))
    with pytest.raises(InductionCapacityError, match="outside induction windows"):
        d.authorise_induction(_train(), 500)


def test_rolling_hour_and_window_caps(cfg):
    d = Depot(None, small_depot_cfg(cfg))
    d.authorise_induction(_train(), 1000)
    d.authorise_induction(_train(), 1100)
    with pytest.raises(InductionCapacityError, match="rolling-hour"):
        d.authorise_induction(_train(), 1200)
    assert d.next_capacity_time(1200) == 1000 + 3600
    d.authorise_induction(_train(), 4600)
    assert d.remaining_capacity(4700) == 0
    with pytest.raises(InductionCapacityError, match="exhausted"):
        d.authorise_induction(_train(), 4900)
    assert d.next_capacity_time(4900) is None


def test_depot_rejects_unready_train(cfg):
    d = Depot(None, small_depot_cfg(cfg))
    with pytest.raises(TrainNotReadyError):
        d.authorise_induction(_train(ready=2000), 1500)
    with pytest.raises(TrainNotReadyError):
        d.authorise_induction(_train(maint=True), 1500)
    assert d.window_counts["W"] == 0  # rejected requests consume no capacity


def test_capacity_factor_reduces_limits(cfg):
    d = Depot(None, cfg, induction_capacity_factor=0.5)
    assert d.max_per_hour == round(cfg["depot"]["max_inductions_per_hour"] * 0.5)
