"""Phase 3/4: Gymnasium environment, invalid actions, reward function, RL scheduler safety."""

import numpy as np
import pytest

from reinforcement_learning.gym_environment import MetroInductionEnv, scheduler_to_action
from reinforcement_learning.reward_function import RewardWeights, combine
from scheduling.base import run_episode
from scheduling.heuristic_scheduler import HeuristicScheduler


@pytest.fixture(scope="module")
def genv():
    return MetroInductionEnv()


def zero_components(**kw):
    c = {k: 0.0 for k in ["useful_inductions", "in_band", "shortfall", "surplus", "waiting", "queue_events",
                          "missed", "delayed", "failures", "invalid"]}
    c.update(kw)
    return c


def test_spaces_and_reset(genv):
    obs, info = genv.reset(seed=3)
    assert genv.observation_space.contains(obs)
    assert genv.action_space.n == 1 + genv.encoder.K * genv.encoder.E
    assert info["scenario_type"] in ("normal", "peak_demand", "disruption")


def test_reset_is_reproducible(genv):
    a, _ = genv.reset(options={"episode_seed": 11, "scenario_type": "disruption"})
    b, _ = genv.reset(options={"episode_seed": 11, "scenario_type": "disruption"})
    assert np.array_equal(a, b)


def test_invalid_action_rejected_without_effect(genv):
    genv.reset(options={"episode_seed": 5, "scenario_type": "normal"})
    # before any train is ready, every induct action is infeasible
    mask = genv.action_masks()
    bad = int(np.flatnonzero(~mask)[0])
    n_ind = len(genv.sim.inducting)
    _, r, _, _, info = genv.step(bad)
    assert len(genv.sim.inducting) == n_ind
    assert info["rejected_actions"] == 1
    assert genv.sim.decision_log[-1]["accepted"] is False


def test_random_agent_cannot_violate_safety(genv):
    genv.reset(options={"episode_seed": 21, "scenario_type": "disruption"})
    rng = np.random.default_rng(0)
    done, info = False, {}
    while not done:
        _, _, done, _, info = genv.step(int(rng.integers(genv.action_space.n)))
    assert info["safety_violations"] == 0


def test_heuristic_actions_expressible(genv):
    genv.reset(options={"episode_seed": 8, "scenario_type": "peak_demand"})
    teacher = HeuristicScheduler()
    done, accepted = False, 0
    while not done:
        a = scheduler_to_action(genv, teacher)
        assert genv.action_masks()[a]
        _, _, done, _, _ = genv.step(a)
        accepted += a != 0
    assert accepted > 10


def test_reward_signs_and_weights():
    w = RewardWeights()
    assert combine(zero_components(), w) == 0
    assert combine(zero_components(shortfall=2), w) == pytest.approx(-0.6)
    assert combine(zero_components(useful_inductions=1), w) == pytest.approx(0.5)
    assert combine(zero_components(missed=1, invalid=1), w) == pytest.approx(-1.1)
    assert combine(zero_components(in_band=1, waiting=1), w) == pytest.approx(0.0)
    # shortfall dominates surplus: one train short costs 10x one surplus train
    assert w.shortfall_per_train_min / w.surplus_per_train_min == pytest.approx(10)


def test_reward_weights_validation():
    with pytest.raises(KeyError):
        RewardWeights.from_config({"nonsense": 1})
    with pytest.raises(ValueError):
        RewardWeights.from_config({"shortfall_per_train_min": -1})


def test_rl_scheduler_with_untrained_model_is_safe(cfg, tmp_path):
    from sb3_contrib import MaskablePPO

    from scheduling.rl_scheduler import RLScheduler
    from simulation.scenario_generator import ScenarioGenerator

    env = MetroInductionEnv()
    model = MaskablePPO("MlpPolicy", env, n_steps=64, batch_size=64, seed=0, device="cpu")
    path = tmp_path / "m.zip"
    model.save(path)
    sched = RLScheduler(path)
    ep = run_episode(cfg, ScenarioGenerator(cfg, 77).single("disruption"), sched)
    assert ep.safety_violations == []
    assert ep.result.rejected_actions == 0  # shield removes infeasible actions
    assert all(e["explanation"].startswith("PPO (") for e in ep.explanations if e["accepted"])


def test_rl_scheduler_missing_model(tmp_path):
    from scheduling.rl_scheduler import RLScheduler

    with pytest.raises(FileNotFoundError):
        RLScheduler(tmp_path / "none.zip")
