"""Phase 2: scheduling correctness and safety of FCFS and heuristic schedulers."""

import pytest

from evaluation.metrics import SlotTracker, episode_metrics
from scheduling.base import Decision, Scheduler, load_rl_config, run_episode
from scheduling.baseline_fcfs import FCFSScheduler
from scheduling.heuristic_scheduler import HeuristicScheduler
from simulation.data_generator import build_environment
from simulation.scenario_generator import ScenarioGenerator


@pytest.fixture(scope="module")
def rl_cfg():
    return load_rl_config()


@pytest.fixture(scope="module")
def scen(cfg):
    return {st: ScenarioGenerator(cfg, 4242).single(st) for st in ("normal", "peak_demand", "disruption")}


@pytest.mark.parametrize("st", ["normal", "peak_demand", "disruption"])
@pytest.mark.parametrize("make", [FCFSScheduler, lambda: HeuristicScheduler()])
def test_schedulers_never_unsafe(cfg, scen, st, make):
    ep = run_episode(cfg, scen[st], make())
    assert ep.safety_violations == []
    assert len(ep.result.inductions) > 0
    assert ep.result.rejected_actions == 0


def test_fcfs_picks_earliest_ready_and_respects_deficit(cfg, scen):
    env, *_ = build_environment(cfg, scen["normal"], control="external")
    env.advance_to(env.service_start - 600)
    ready = env.ready_candidates()
    assert ready
    d = FCFSScheduler().decide(env)
    assert d.train_id == min(ready, key=lambda t: (t.readiness_time, t.train_id)).train_id
    # once committed trains meet the requirement, FCFS waits
    env.advance_to(env.service_start + 3 * 3600)
    if env.committed_trains() >= env.required_at(env.now()):
        assert FCFSScheduler().decide(env) is None


def test_heuristic_prefers_priority(cfg, scen):
    env, *_ = build_environment(cfg, scen["peak_demand"], control="external")
    env.advance_to(env.service_start - 300)
    d = HeuristicScheduler().decide(env)
    assert d is not None
    best = min(env.ready_candidates(), key=lambda t: (t.induction_priority, t.readiness_time, t.train_id))
    assert d.train_id == best.train_id
    assert d.entry_point in [e.section_id for e in env.entry_points]


def test_same_scenario_is_identical_for_all_schedulers(cfg, scen):
    a, *_ = build_environment(cfg, scen["disruption"], control="external")
    b, *_ = build_environment(cfg, scen["disruption"], control="external")
    assert [t.to_record() for t in a.trains.values()] == [t.to_record() for t in b.trains.values()]
    assert a.demand_df.equals(b.demand_df)
    assert a._depot_faults == b._depot_faults


class _BadScheduler(Scheduler):
    name = "bad"

    def __init__(self):
        self.calls = 0

    def decide(self, env):
        self.calls += 1
        # try an unknown train, then a train in maintenance / not ready
        if self.calls % 2:
            return Decision("NOT-A-TRAIN")
        not_ready = [t for t in env.trains.values() if not t.is_ready(env.now())]
        return Decision(not_ready[0].train_id) if not_ready else None


def test_invalid_decisions_are_rejected(cfg, scen):
    ep = run_episode(cfg, scen["normal"], _BadScheduler())
    assert ep.result.rejected_actions > 0
    assert len(ep.result.inductions) == 0
    assert ep.safety_violations == []
    assert all(not e["accepted"] for e in ep.explanations)


def test_apply_induction_rejects_unknown_entry(cfg, scen):
    env, *_ = build_environment(cfg, scen["normal"], control="external")
    env.advance_to(env.service_start)
    tid = env.ready_candidates()[0].train_id
    ok, why = env.apply_induction(tid, "ENTRY-NOWHERE")
    assert not ok and "unknown entry" in why


def test_slot_tracker():
    st = SlotTracker()
    assert st.update(0, 2, 0) == (0, 0)       # 2 slots open
    assert st.update(60, 2, 1) == (1, 0)      # one filled after 60 s
    assert st.update(120, 1, 1) == (0, 1)     # requirement drops -> level-2 slot missed
    assert st.filled == [60] and st.missed == 1
    st.update(180, 3, 1)
    assert st.close() == 2


def test_episode_metrics_are_consistent(cfg, scen):
    ep = run_episode(cfg, scen["normal"], FCFSScheduler())
    m = episode_metrics(ep, cfg)
    assert 0 <= m["demand_coverage_pct"] <= 100
    assert 0 <= m["track_utilization_pct"] <= 100
    assert m["successful_inductions"] == len(ep.result.inductions)
    assert m["safety_violations"] == 0
    assert m["induction_conflicts"] <= m["successful_inductions"]
