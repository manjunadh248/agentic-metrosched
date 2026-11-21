"""Custom Gymnasium environment for train induction scheduling.

Episode  = one operating day (a scenario built from an episode seed).
Step     = one decision at the current decision time. Time advances by
           ``decision_interval_s`` (60 s) when the agent chooses WAIT, when an
           action is rejected, or after ``max_inductions_per_tick`` inductions.

Action space – ``Discrete(1 + K * E)``:
    0                       wait for a better opportunity
    1 + k * E + e           induct the k-th candidate train via mainline entry
                            point e (candidates: first K/2 by best induction
                            priority, the rest by earliest readiness)
                            (the entry point is the "induction slot")

Invalid actions (empty slot, no induction capacity, …) are *rejected by the
simulation* – they cannot move a train – and are treated as WAIT with a small
penalty. :meth:`action_masks` exposes the feasible set for safety shielding at
deployment time.

Observation – a flat float32 vector (see :class:`ObservationEncoder`) covering:
current time, current and future demand / required trains, trains in service,
committed, being inducted, queued, delayed, withdrawing, induction capacity
remaining (window and rolling hour), recent induction history, occupancy /
clearance / closure of every track section, free platforms, depot pipeline
(trains about to become ready, in maintenance) and the K candidate trains
(priority, km since maintenance, time since ready).
"""

from __future__ import annotations

import logging

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from scheduling.base import load_rl_config
from simulation.config import SCENARIO_TYPES, load_config
from simulation.data_generator import audit_result, build_environment
from simulation.metro_environment import MetroEnvironment
from simulation.scenario_generator import Scenario, ScenarioGenerator
from simulation.train import Train, TrainLocation, TrainStatus

from .reward_function import RewardCalculator, RewardWeights

logger = logging.getLogger(__name__)


class ObservationEncoder:
    """Turns a MetroEnvironment into the agent's observation, mask and action meaning."""

    def __init__(self, cfg: dict, candidate_slots: int = 6):
        self.cfg = cfg
        self.K = int(candidate_slots)
        self.entry_ids = [e["id"] for e in cfg["track"]["entry_points"]]
        self.E = len(self.entry_ids)
        self.section_ids = ([t["id"] for t in cfg["depot"]["exit_tracks"]]
                            + [s["id"] for s in cfg["track"]["shared_sections"]]
                            + self.entry_ids + [cfg["depot"]["reception_track"]["id"]])
        self.max_service = cfg["fleet"]["size"] - cfg["service"]["reserve_trains"]
        self.n_actions = 1 + self.K * self.E
        self.obs_dim = 26 + 4 * len(self.section_ids) + 4 * self.K

    # ------------------------------------------------------------- actions
    def candidates(self, env: MetroEnvironment) -> list[Train]:
        """K choosable trains: first half best induction priority, then earliest-ready."""
        ready = env.ready_candidates()
        by_prio = sorted(ready, key=lambda t: (t.induction_priority, t.readiness_time, t.train_id))
        by_ready = sorted(ready, key=lambda t: (t.readiness_time, t.train_id))
        out = by_prio[: (self.K + 1) // 2]
        for t in by_ready:
            if len(out) >= self.K:
                break
            if t not in out:
                out.append(t)
        return out

    def action_mask(self, env: MetroEnvironment) -> np.ndarray:
        mask = np.zeros(self.n_actions, dtype=bool)
        mask[0] = True
        if env.now() >= env.service_end or env.depot.remaining_capacity(env.now()) <= 0:
            return mask
        n = len(self.candidates(env))
        for k in range(n):
            mask[1 + k * self.E: 1 + (k + 1) * self.E] = True
        return mask

    def decode(self, env: MetroEnvironment, action: int) -> tuple[str, str] | None:
        action = int(action)
        if not 0 <= action < self.n_actions:
            raise ValueError(f"action {action} out of range")
        if action == 0:
            return None
        k, e = divmod(action - 1, self.E)
        cands = self.candidates(env)
        if k >= len(cands):
            return ("<empty-slot>", self.entry_ids[e])
        return cands[k].train_id, self.entry_ids[e]

    # --------------------------------------------------------- observation
    def encode(self, env: MetroEnvironment) -> np.ndarray:
        t = env.now()
        day_span = env.service_end - env.depot_open
        phase = 2 * np.pi * (t / 86400.0)
        ms = float(self.max_service)
        req_t = max(t, env.service_start)

        def req(dt):
            return env.required_at(min(req_t + dt, env.service_end - 1)) / ms

        def demand(dt):
            d = env.demand_df
            starts = d["interval_start_s"].to_numpy()
            i = int(np.clip(np.searchsorted(starts, min(req_t + dt, env.service_end - 1), "right") - 1,
                            0, len(starts) - 1))
            return d["demand_pphpd"].iloc[i] / (4.0 * self.cfg["demand"]["base_pphpd"])

        in_service = env.in_service_count()
        committed = env.committed_trains()
        need_ahead = max(env.required_at(min(req_t + x, env.service_end - 1)) for x in range(0, 1801, 300))
        depot = env.depot
        win = depot.window_at(t)
        win_left = (win["max_inductions"] - depot.window_counts[win["id"]]) / max(win["max_inductions"], 1) \
            if win else 0.0
        last_ind = max(depot.induction_starts) if depot.induction_starts else env.depot_open - 3600
        fleet = float(len(env.trains))
        in_depot = [tr for tr in env.trains.values() if tr.location is TrainLocation.DEPOT]
        pending = [tr.readiness_time - t for tr in in_depot
                   if tr.status is not TrainStatus.MAINTENANCE and tr.readiness_time > t]
        delayed = sum(1 for tr in env.trains.values()
                      if tr.status is TrainStatus.DELAYED and tr.location is TrainLocation.MAINLINE)
        f = [
            (t - env.depot_open) / day_span, np.sin(phase), np.cos(phase), float(t < env.service_start),
            req(0), req(600), req(1200), req(1800), req(3600),
            demand(0), demand(1800),
            in_service / ms, committed / ms, len(env.inducting) / 5.0, len(env.waiting) / 5.0,
            len(env.withdrawing) / 5.0, delayed / 5.0,
            (env.required_at(req_t) - in_service) / 10.0, (need_ahead - committed) / 10.0,
            depot.remaining_capacity(t) / depot.max_per_hour, depot.inductions_last_hour(t) / depot.max_per_hour,
            win_left, min(1.0, (t - last_ind) / 3600.0),
            len(env.ready_candidates()) / fleet, (min(pending) / 3600.0) if pending else 1.0,
            sum(tr.status is TrainStatus.MAINTENANCE for tr in env.trains.values()) / fleet,
        ]
        for sid in self.section_ids:
            sec = env.sections[sid]
            f += [float(sec.occupant is not None), min(1.0, max(0, sec.earliest_entry(t) - t) / 600.0),
                  float(sec.is_closed(t)), min(1.0, sec.queue_length() / 3.0)]
        levels = float(self.cfg["fleet"]["priority_levels"])
        thr = float(self.cfg["fleet"]["maintenance_km_threshold"])
        cands = self.candidates(env)
        for k in range(self.K):
            if k < len(cands):
                c = cands[k]
                f += [1.0, c.induction_priority / levels, c.km_since_maintenance / thr,
                      min(1.0, (t - c.readiness_time) / 3600.0)]
            else:
                f += [0.0, 0.0, 0.0, 0.0]
        obs = np.asarray(f, dtype=np.float32)
        if obs.shape[0] != self.obs_dim:
            raise RuntimeError(f"observation size {obs.shape[0]} != {self.obs_dim}")
        return np.clip(obs, -5.0, 5.0)


class MetroInductionEnv(gym.Env):
    """Gymnasium environment wrapping :class:`MetroEnvironment` (external control)."""

    metadata = {"render_modes": []}

    def __init__(self, cfg: dict | None = None, rl_cfg: dict | None = None, seed_low: int | None = None,
                 seed_high: int | None = None, scenario_mix: dict | None = None):
        super().__init__()
        self.cfg = cfg or load_config()
        self.rl_cfg = rl_cfg or load_rl_config()
        tr = self.rl_cfg["training"]
        self.seed_low = int(tr["train_seed_low"] if seed_low is None else seed_low)
        self.seed_high = int(tr["train_seed_high"] if seed_high is None else seed_high)
        if self.seed_high <= self.seed_low:
            raise ValueError("seed_high must be > seed_low")
        mix = scenario_mix or self.rl_cfg["gym"]["train_scenario_mix"]
        self.mix_types = [t for t in SCENARIO_TYPES if mix.get(t, 0) > 0]
        p = np.array([mix[t] for t in self.mix_types], dtype=float)
        self.mix_p = p / p.sum()
        self.weights = RewardWeights.from_config(self.rl_cfg["reward"])
        self.max_per_tick = int(self.rl_cfg["scheduling"]["max_inductions_per_tick"])
        self.encoder = ObservationEncoder(self.cfg, self.rl_cfg["gym"]["candidate_slots"])
        self.action_space = spaces.Discrete(self.encoder.n_actions)
        self.observation_space = spaces.Box(-5.0, 5.0, shape=(self.encoder.obs_dim,), dtype=np.float32)
        self.sim: MetroEnvironment | None = None
        self.scenario: Scenario | None = None

    # ---------------------------------------------------------------- API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}
        if "scenario" in options:
            self.scenario = options["scenario"]
        else:
            ep_seed = int(options.get("episode_seed", self.np_random.integers(self.seed_low, self.seed_high)))
            stype = options.get("scenario_type") or str(self.np_random.choice(self.mix_types, p=self.mix_p))
            self.scenario = ScenarioGenerator(self.cfg, ep_seed).single(stype)
        self.sim, self.demand, *_ = build_environment(self.cfg, self.scenario, control="external",
                                                      policy_name="PPO")
        self.reward_calc = RewardCalculator(self.weights, self.sim)
        self.times = self.sim.decision_times()
        self.tick = 0
        self.sub = 0
        self.episode_return = 0.0
        self.sim.advance_to(self.times[0])
        return self.encoder.encode(self.sim), self._info()

    def action_masks(self) -> np.ndarray:
        return self.encoder.action_mask(self.sim)

    def step(self, action):
        if self.sim is None:
            raise RuntimeError("Call reset() before step()")
        decoded = self.encoder.decode(self.sim, int(action))
        invalid, advance = 0, True
        if decoded is not None:
            ok, _ = self.sim.apply_induction(*decoded)
            if ok:
                self.sub += 1
                advance = self.sub >= self.max_per_tick or not self.action_masks()[1:].any()
            else:
                invalid = 1
        reward, terminated = 0.0, False
        if advance:
            self.tick += 1
            self.sub = 0
            if self.tick >= len(self.times):
                self.sim.finish()
                reward, _ = self.reward_calc.step(invalid)
                reward += self.reward_calc.terminal()
                terminated = True
            else:
                self.sim.advance_to(self.times[self.tick])
                reward, _ = self.reward_calc.step(invalid)
        self.episode_return += reward
        obs = self.encoder.encode(self.sim)
        info = self._info()
        if terminated:
            info["safety_violations"] = len(audit_result(self.sim))
            info["reward_totals"] = dict(self.reward_calc.totals)
        return obs, float(reward), terminated, False, info

    def _info(self) -> dict:
        return {"time_s": self.sim.now(), "scenario_type": self.scenario.scenario_type,
                "episode_seed": int(self.scenario.seed), "rejected_actions": self.sim.rejected_actions}


def scheduler_to_action(env: MetroInductionEnv, scheduler) -> int:
    """Map a rule-based scheduler's decision to this env's action index (0 = wait)."""
    d = scheduler.decide(env.sim)
    if d is None:
        return 0
    cands = [c.train_id for c in env.encoder.candidates(env.sim)]
    if d.train_id not in cands:
        return 0
    k = cands.index(d.train_id)
    entry = d.entry_point or env.sim._choose_entry(env.sim.now()).section_id
    e = env.encoder.entry_ids.index(entry)
    return 1 + k * env.encoder.E + e


def rollout_return(env: MetroInductionEnv, act_fn, options: dict) -> tuple[float, dict]:
    """Run one episode with ``act_fn(env, obs) -> action``; returns (return, final info)."""
    obs, _ = env.reset(options=options)
    total, done, info = 0.0, False, {}
    while not done:
        obs, r, done, _, info = env.step(act_fn(env, obs))
        total += r
    return total, info


def masked_policy_probs(model, obs: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Action probabilities of an SB3 (Maskable)PPO policy with infeasible actions removed."""
    import torch

    with torch.no_grad():
        t = model.policy.obs_to_tensor(obs)[0]
        probs = model.policy.get_distribution(t).distribution.probs.cpu().numpy()[0]
    probs = np.where(mask, probs, 0.0)
    total = probs.sum()
    return probs / total if total > 0 else mask / mask.sum()


def select_action(probs: np.ndarray, mode: str, rng: np.random.Generator) -> int:
    """'sample': draw from the (masked) policy with a seeded RNG (reproducible); 'argmax': most probable."""
    if mode == "argmax":
        return int(np.argmax(probs))
    if mode == "sample":
        return int(rng.choice(len(probs), p=probs))
    raise ValueError(f"unknown action selection mode {mode!r}")
