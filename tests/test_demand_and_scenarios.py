"""Demand generator, demand->trains mapping and scenario generator."""

import numpy as np
import pytest

from simulation.config import SCENARIO_TYPES
from simulation.demand_generator import DemandGenerator, cycle_time_s, required_trains_for_demand
from simulation.scenario_generator import ScenarioGenerator, assign_scheduled_inductions, generate_fleet


def test_peaks_exceed_off_peak(cfg):
    dg = DemandGenerator(cfg)
    df = dg.generate(np.random.default_rng(0), weekday=1)
    by_period = df.groupby("period")["demand_pphpd"].mean()
    assert by_period["morning"] > by_period["off_peak"]
    assert by_period["evening"] > by_period["off_peak"]
    assert by_period["afternoon"] > by_period["off_peak"]
    assert len(df) == (cfg["simulation"]["service_end"] - cfg["simulation"]["service_start"]) // cfg[
        "simulation"]["interval_s"]


def test_required_trains_monotonic_and_bounded(cfg):
    levels = [0, 500, 2000, 4000, 6000, 8000, 10_000, 50_000]
    req = [required_trains_for_demand(d, cfg)[0] for d in levels]
    assert req == sorted(req)
    assert min(req) >= 1
    assert max(req) <= cfg["fleet"]["size"] - cfg["service"]["reserve_trains"]
    # max-headway floor: low demand still needs cycle / max_headway trains
    assert req[0] == int(np.ceil(cycle_time_s(cfg) / cfg["service"]["max_service_headway_s"]))


def test_peak_demand_scenario_raises_demand(cfg):
    dg = DemandGenerator(cfg)
    a = dg.generate(np.random.default_rng(5), 1, 1.0)
    b = dg.generate(np.random.default_rng(5), 1, 1.3)
    assert b["demand_pphpd"].sum() > a["demand_pphpd"].sum()
    assert b["required_trains"].max() >= a["required_trains"].max()


def test_demand_reproducible(cfg):
    dg = DemandGenerator(cfg)
    a = dg.generate(np.random.default_rng(9), 2)
    b = dg.generate(np.random.default_rng(9), 2)
    assert a.equals(b)


def test_scenarios_reproducible_and_cover_all_types(cfg):
    a = ScenarioGenerator(cfg, 11).generate(30)
    b = ScenarioGenerator(cfg, 11).generate(30)
    assert [s.to_record() for s in a] == [s.to_record() for s in b]
    assert set(s.scenario_type for s in a) == set(SCENARIO_TYPES)


def test_disruption_scenarios_have_closures(scenarios, cfg):
    dis = [s for s in scenarios if s.scenario_type == "disruption"]
    assert dis
    p = cfg["scenarios"]["disruption"]
    for s in dis:
        assert len(s.exit_track_closures) == p["exit_track_closures"]
        assert len(s.section_blocks) == p["section_blocks"]
        assert len(s.platform_closures) == p["platform_closures"]
        assert s.extra_headway_s == p["extra_headway_s"]


def test_fleet_generation(cfg, scenarios):
    trains, delays = generate_fleet(cfg, scenarios[0], np.random.default_rng(1))
    assert len(trains) == cfg["fleet"]["size"]
    assert len({t.train_id for t in trains}) == len(trains)
    assert len({t.depot_location for t in trains}) == len(trains)
    for t in trains:
        assert t.readiness_time >= cfg["simulation"]["depot_open"] + t.min_prep_time_s
        assert 1 <= t.induction_priority <= cfg["fleet"]["priority_levels"]
        assert t.maintenance_required == (t.km_since_maintenance > cfg["fleet"]["maintenance_km_threshold"])


def test_schedule_assigns_only_serviceable_trains(cfg, scenarios):
    trains, _ = generate_fleet(cfg, scenarios[0], np.random.default_rng(2))
    demand = DemandGenerator(cfg).generate(np.random.default_rng(2), 0)
    plan = assign_scheduled_inductions(trains, demand, cfg)
    assert plan
    by_id = {t.train_id: t for t in trains}
    assert len({p["train_id"] for p in plan}) == len(plan)
    for p in plan:
        assert not by_id[p["train_id"]].maintenance_required


def test_invalid_inputs(cfg):
    with pytest.raises(ValueError):
        DemandGenerator(cfg).generate(np.random.default_rng(0), weekday=7)
    with pytest.raises(ValueError):
        ScenarioGenerator(cfg, 1).generate(0)
