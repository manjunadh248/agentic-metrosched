"""End-to-end safety checks on full simulated operating days."""

import pandas as pd
import pytest

from simulation.data_generator import audit_result, simulate_day
from tests.conftest import first_of_type

HOUR = 3600


@pytest.fixture(scope="module")
def days(cfg, scenarios):
    out = {}
    for st in ("normal", "peak_demand", "disruption"):
        out[st] = simulate_day(cfg, first_of_type(scenarios, st))
    return out


@pytest.mark.parametrize("st", ["normal", "peak_demand", "disruption"])
def test_no_conflicts_headway_or_closure_violations(days, st):
    *_, env, res = days[st]
    assert audit_result(env) == []
    assert len(res.inductions) > 0


@pytest.mark.parametrize("st", ["normal", "peak_demand", "disruption"])
def test_induction_capacity_respected(days, st):
    *_, env, res = days[st]
    starts = sorted(res.inductions["authorised_time_s"])
    for t in starts:
        in_hour = sum(1 for s in starts if t - HOUR < s <= t)
        assert in_hour <= env.depot.max_per_hour
    for w in env.depot.windows:
        n = sum(1 for s in starts if w["start"] <= s < w["end"])
        assert n <= w["max_inductions"]
        assert n == env.depot.window_counts[w["id"]]


def test_disruption_closures_are_applied(days):
    *_, env, res = days["disruption"]
    assert not res.closures.empty
    for _, c in res.closures.iterrows():
        sub = res.occupancy[res.occupancy["section_id"] == c["section_id"]]
        assert not ((sub["start_s"] >= c["start_s"]) & (sub["start_s"] < c["end_s"])).any()


def test_platform_occupancy_within_capacity(days, cfg):
    *_, env, res = days["peak_demand"]
    pf = res.occupancy[res.occupancy["kind"] == "platform"]
    events = sorted([(s, 1) for s in pf["start_s"]] + [(e, -1) for e in pf["end_s"]],
                    key=lambda x: (x[0], x[1]))
    level = peak = 0
    for _, d in events:
        level += d
        peak = max(peak, level)
    assert peak <= cfg["platforms"]["count"]


def test_every_inducted_train_followed_the_full_route(days, cfg):
    *_, env, res = days["normal"]
    ih = res.inductions
    assert (ih["depot_exit_time_s"] >= ih["authorised_time_s"]).all()
    assert (ih["mainline_entry_time_s"] >= ih["depot_exit_time_s"]).all()
    assert (ih["service_start_time_s"] >= ih["mainline_entry_time_s"]).all()
    assert (ih["actual_route_s"] >= ih["nominal_route_s"] - cfg["platforms"]["induction_dwell_s"]).all()


def test_invalid_policy_actions_are_rejected(cfg, scenarios):
    """A policy that proposes unready / unknown / duplicate trains cannot induct them."""
    sc = first_of_type(scenarios, "normal")
    proposed = []

    def bad_policy(candidates, n, state):
        ids = [c.train_id for c in candidates]
        picks = ["NOPE", "NOPE"] + ids[:1] + ids[:1]
        proposed.append((state["time_s"], ids))
        return picks

    *_, env, res = simulate_day(cfg, sc, dispatch_policy=bad_policy, policy_name="bad")
    assert res.rejected_actions > 0
    assert "NOPE" not in set(res.inductions["train_id"])
    assert audit_result(env) == []


def test_policy_exception_does_not_break_safety(cfg, scenarios):
    sc = first_of_type(scenarios, "normal")

    def crashing_policy(candidates, n, state):
        raise RuntimeError("boom")

    *_, env, res = simulate_day(cfg, sc, dispatch_policy=crashing_policy, policy_name="crash")
    assert res.rejected_actions > 0
    assert res.inductions.empty or "train_id" in res.inductions
    assert audit_result(env) == []


def test_status_snapshots_use_configured_states(days, cfg):
    *_, res = days["disruption"]
    assert set(res.snapshots["status"]).issubset(set(cfg["fleet"]["states"]))
    assert isinstance(res.snapshots, pd.DataFrame)
