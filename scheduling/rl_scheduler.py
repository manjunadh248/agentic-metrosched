"""PPO-based scheduler: wraps a trained Stable-Baselines3 model as a Scheduler.

At each decision the policy network scores every action. A *safety shield*
masks actions that the environment would reject (empty candidate slot, no
induction capacity). The action is then drawn from the masked policy with a
random generator seeded by the scenario seed ("sample" mode, default – the
same mode used for validation-based model selection; identical day ->
identical decisions), or the most probable action is taken ("argmax" mode).
Sampling is used because after imitation pre-training WAIT is the single most
probable action at almost every tick, so argmax systematically delays
inductions (measured on validation seeds: see docs/TECHNICAL_REPORT.md). The shield only removes infeasible actions; it never adds a
decision the policy did not rate, and all hard safety rules are still
enforced by the simulation itself.

Explanations: for every induction the scheduler reports the probability the
policy assigned to the chosen action, the runner-up alternative and the
features of the chosen train that the policy observed.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from simulation.config import PROJECT_ROOT, load_config, seconds_to_clock
from simulation.metro_environment import MetroEnvironment

from .base import Decision, Scheduler, load_rl_config

logger = logging.getLogger(__name__)

DEFAULT_MODEL = PROJECT_ROOT / "models" / "ppo_metrosched.zip"


def _load_model(path: Path):
    """Load a MaskablePPO or plain PPO checkpoint."""
    from sb3_contrib import MaskablePPO
    from stable_baselines3 import PPO

    try:
        return MaskablePPO.load(str(path), device="cpu")
    except Exception:  # noqa: BLE001 - fall back to plain PPO checkpoints
        return PPO.load(str(path), device="cpu")


class RLScheduler(Scheduler):
    name = "PPO"

    def __init__(self, model_path: str | Path | None = None, cfg: dict | None = None, rl_cfg: dict | None = None,
                 use_shield: bool = True):
        # Imported lazily so the rest of the project works without torch installed.
        from reinforcement_learning.gym_environment import ObservationEncoder

        path = Path(model_path) if model_path else DEFAULT_MODEL
        if not path.is_file():
            raise FileNotFoundError(f"Trained PPO model not found: {path}. Run: python main.py train")
        self.cfg = cfg or load_config()
        self.rl_cfg = rl_cfg or load_rl_config()
        self.encoder = ObservationEncoder(self.cfg, self.rl_cfg["gym"]["candidate_slots"])
        self.model = _load_model(path)
        if self.model.observation_space.shape[0] != self.encoder.obs_dim:
            raise ValueError("Model observation size does not match the current encoder/config")
        self.use_shield = use_shield
        self.model_path = path
        self.mode = self.rl_cfg["gym"].get("action_selection", "sample")
        self._rng = np.random.default_rng(0)

    def reset(self, env: MetroEnvironment) -> None:
        # Seeded per scenario: identical day -> identical decisions (reproducible).
        self._rng = np.random.default_rng(int(env.scenario.seed))

    def action_probabilities(self, obs: np.ndarray) -> np.ndarray:
        import torch

        with torch.no_grad():
            t = self.model.policy.obs_to_tensor(obs)[0]
            dist = self.model.policy.get_distribution(t)
            return dist.distribution.probs.cpu().numpy()[0]

    def decide(self, env: MetroEnvironment) -> Decision | None:
        from reinforcement_learning.gym_environment import select_action

        obs = self.encoder.encode(env)
        probs = self.action_probabilities(obs)
        mask = self.encoder.action_mask(env) if self.use_shield else np.ones_like(probs, dtype=bool)
        masked = np.where(mask, probs, 0.0)
        masked = masked / masked.sum() if masked.sum() > 0 else mask / mask.sum()
        scored = np.where(mask, probs, -1.0)
        action = select_action(masked, self.mode, self._rng)
        if action == 0:
            return None
        decoded = self.encoder.decode(env, action)
        train_id, entry_id = decoded
        cands = self.encoder.candidates(env)
        train = next((c for c in cands if c.train_id == train_id), None)
        order = np.argsort(-scored)
        others = [int(a) for a in order if int(a) != action and scored[int(a)] >= 0]
        runner = others[0] if others else 0
        runner_txt = "wait" if runner == 0 else "induct {} via {}".format(*self.encoder.decode(env, runner))
        t = env.now()
        req = env.required_at(max(t, env.service_start))
        detail = ""
        if train is not None:
            detail = (f"{train_id}: priority {train.induction_priority}, "
                      f"{train.km_since_maintenance:.0f} km since maintenance, ready since "
                      f"{seconds_to_clock(train.readiness_time)}; ")
        how = "sampled" if self.mode == "sample" else "most probable"
        reason = (f"PPO ({how}): {detail}policy probability {masked[action]:.2f} (wait {masked[0]:.2f}, best other "
                  f"'{runner_txt}'); in service {env.in_service_count()}/{req} required, "
                  f"{env.committed_trains()} committed, {len(env.waiting)} queued; entry {entry_id}.")
        return Decision(train_id, entry_id, reason)
