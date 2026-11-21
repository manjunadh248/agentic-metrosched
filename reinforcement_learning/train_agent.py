"""Train the PPO induction-scheduling agent (Stable-Baselines3).

* Training env(s): episode seeds drawn from [train_seed_low, train_seed_high)
  over a mix of normal / peak-demand / disruption scenarios.
* Validation env: a *separate* environment on seeds starting at
  ``validation_seed_base`` (disjoint from training and final evaluation seeds)
  used for model selection (best checkpoint) and convergence monitoring.
* Outputs: models/ppo_metrosched.zip (best validation model),
  models/ppo_metrosched_final.zip, models/checkpoints/*, results/training/
  (monitor CSVs, validation curve, reference returns of FCFS and heuristic).

Usage:  python -m reinforcement_learning.train_agent --timesteps 400000
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from scheduling.base import load_rl_config
from scheduling.baseline_fcfs import FCFSScheduler
from scheduling.heuristic_scheduler import HeuristicScheduler
from simulation.config import PROJECT_ROOT, SCENARIO_TYPES, load_config

from .gym_environment import (MetroInductionEnv, masked_policy_probs, rollout_return, scheduler_to_action,
                              select_action)
from .pretrain import behaviour_clone, collect_demonstrations

logger = logging.getLogger(__name__)

MODELS_DIR = PROJECT_ROOT / "models"
TRAIN_RESULTS = PROJECT_ROOT / "results" / "training"


def validation_options(rl_cfg: dict, n: int) -> list[dict]:
    """Fixed validation episodes: seeds base+i, scenario types cycled."""
    base = int(rl_cfg["training"]["validation_seed_base"])
    return [{"episode_seed": base + i, "scenario_type": SCENARIO_TYPES[i % 3]} for i in range(n)]


class ValidationCallback(BaseCallback):
    """Every ``eval_every`` steps: evaluate the deterministic policy on validation seeds."""

    def __init__(self, eval_env: MetroInductionEnv, options: list[dict], eval_every: int, out_dir: Path,
                 best_path: Path, mode: str = "sample"):
        super().__init__()
        self.mode = mode
        self.eval_env, self.options, self.eval_every = eval_env, options, eval_every
        self.out_dir, self.best_path = out_dir, best_path
        self.rows: list[dict] = []
        self.best = -np.inf
        self._next = eval_every

    def _evaluate(self) -> None:
        returns, viol = [], 0
        for opt in self.options:
            # Same action selection as deployment (RLScheduler): masked policy, seeded sampling.
            rng = np.random.default_rng(opt["episode_seed"])

            def act(env, obs, rng=rng):
                probs = masked_policy_probs(self.model, obs, env.action_masks())
                return select_action(probs, self.mode, rng)
            r, info = rollout_return(self.eval_env, act, opt)
            returns.append(r)
            viol += info.get("safety_violations", 0)
        mean = float(np.mean(returns))
        self.rows.append({"timesteps": int(self.num_timesteps), "val_mean_return": mean,
                          "val_std_return": float(np.std(returns)), "safety_violations": viol,
                          "wall_time_s": round(time.time() - self._t0, 1)})
        pd.DataFrame(self.rows).to_csv(self.out_dir / "validation_curve.csv", index=False)
        logger.info("t=%d validation mean return %.1f (best %.1f)", self.num_timesteps, mean, self.best)
        if mean > self.best:
            self.best = mean
            self.model.save(self.best_path)

    def _on_training_start(self) -> None:
        self._t0 = time.time()
        self._evaluate()

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next:
            self._next += self.eval_every
            self._evaluate()
        return True


def reference_returns(rl_cfg: dict, cfg: dict, options: list[dict]) -> dict:
    env = MetroInductionEnv(cfg, rl_cfg)
    out = {}
    for sched in (FCFSScheduler(), HeuristicScheduler.from_config(rl_cfg)):
        vals = [rollout_return(env, lambda e, o, s=sched: scheduler_to_action(e, s), opt)[0] for opt in options]
        out[sched.name] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return out


def train(timesteps: int | None = None, n_envs: int | None = None, seed: int | None = None,
          rl_config: str | None = None, config: str | None = None, subproc: bool = False) -> Path:
    cfg = load_config(config)
    rl_cfg = load_rl_config(rl_config)
    tr, hp = rl_cfg["training"], rl_cfg["ppo"]
    timesteps = int(timesteps or tr["total_timesteps"])
    n_envs = int(n_envs or tr["n_envs"])
    seed = int(tr["seed"] if seed is None else seed)
    torch.set_num_threads(1)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (MODELS_DIR / "checkpoints").mkdir(exist_ok=True)
    TRAIN_RESULTS.mkdir(parents=True, exist_ok=True)

    def make(i):
        def _f():
            env = MetroInductionEnv(cfg, rl_cfg)
            env.reset(seed=seed + 1000 * i)
            return Monitor(env, str(TRAIN_RESULTS / f"monitor_{i}"), info_keywords=("scenario_type",))
        return _f

    vec_cls = SubprocVecEnv if subproc else DummyVecEnv
    vec = vec_cls([make(i) for i in range(n_envs)])
    algo = str(hp.get("algorithm", "maskable_ppo")).lower()
    algo_cls = {"maskable_ppo": MaskablePPO, "ppo": PPO}[algo]
    model = algo_cls("MlpPolicy", vec, learning_rate=hp["learning_rate"], n_steps=hp["n_steps"],
                batch_size=hp["batch_size"], n_epochs=hp["n_epochs"], gamma=hp["gamma"],
                gae_lambda=hp["gae_lambda"], clip_range=hp["clip_range"], ent_coef=hp["ent_coef"],
                vf_coef=hp["vf_coef"], target_kl=hp.get("target_kl"), policy_kwargs={"net_arch": list(hp["net_arch"])},
                seed=seed, verbose=0, device="cpu")
    pre = rl_cfg.get("pretraining", {}) or {}
    pre_info = {"enabled": bool(pre.get("enabled", False))}
    if pre_info["enabled"]:
        if algo != "maskable_ppo":
            raise ValueError("Behaviour-cloning warm start requires algorithm: maskable_ppo")
        teacher = HeuristicScheduler.from_config(rl_cfg)
        demos = collect_demonstrations(MetroInductionEnv(cfg, rl_cfg), teacher, int(pre["episodes"]),
                                       np.random.default_rng(seed + 17), float(hp["gamma"]))
        bc = behaviour_clone(model, demos, int(pre["epochs"]), int(pre["batch_size"]), float(pre["lr"]), seed=seed)
        pd.DataFrame(bc["history"]).to_csv(TRAIN_RESULTS / "behaviour_cloning.csv", index=False)
        pre_info.update({"teacher": "Heuristic", "episodes": int(pre["episodes"]), "steps": int(len(demos["act"])),
                         "teacher_train_mean_return": demos["teacher_mean_return"],
                         "bc_accuracy": bc["final_accuracy"]})
        model.save(MODELS_DIR / "ppo_metrosched_bc_init")
    val_opts = validation_options(rl_cfg, int(tr["eval_episodes"]))
    refs = reference_returns(rl_cfg, cfg, val_opts)
    (TRAIN_RESULTS / "reference_returns.json").write_text(json.dumps(refs, indent=2))
    logger.info("Validation reference returns: %s", refs)
    best_path = MODELS_DIR / "ppo_metrosched_best"
    callbacks = CallbackList([
        ValidationCallback(MetroInductionEnv(cfg, rl_cfg), val_opts, int(tr["eval_every_steps"]),
                           TRAIN_RESULTS, best_path, rl_cfg["gym"].get("action_selection", "sample")),
        CheckpointCallback(max(1, int(tr["checkpoint_every_steps"]) // n_envs), str(MODELS_DIR / "checkpoints"),
                           name_prefix="ppo_metrosched"),
    ])
    t0 = time.time()
    logger.info("Training PPO for %d timesteps on %d envs", timesteps, n_envs)
    model.learn(total_timesteps=timesteps, callback=callbacks, progress_bar=False)
    model.save(MODELS_DIR / "ppo_metrosched_final")
    shutil.copy(str(best_path) + ".zip", MODELS_DIR / "ppo_metrosched.zip")
    meta = {"timesteps": timesteps, "n_envs": n_envs, "seed": seed, "training_time_s": round(time.time() - t0, 1),
            "algorithm": algo, "pretraining": pre_info, "ppo": hp, "reward": rl_cfg["reward"], "train_seed_range": [tr["train_seed_low"], tr["train_seed_high"]],
            "validation_seed_base": tr["validation_seed_base"], "obs_dim": int(vec.observation_space.shape[0]),
            "n_actions": int(vec.action_space.n), "selected_model": "best validation return",
            "action_selection": rl_cfg["gym"].get("action_selection", "sample")}
    (MODELS_DIR / "ppo_metrosched_meta.json").write_text(json.dumps(meta, indent=2))
    vec.close()
    logger.info("Saved model to %s (%.0fs)", MODELS_DIR / "ppo_metrosched.zip", time.time() - t0)
    return MODELS_DIR / "ppo_metrosched.zip"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Train the MetroSched PPO agent")
    ap.add_argument("--timesteps", type=int, default=None)
    ap.add_argument("--n-envs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--rl-config", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--subproc", action="store_true", help="use SubprocVecEnv (multi-core)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    train(args.timesteps, args.n_envs, args.seed, args.rl_config, args.config, args.subproc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
