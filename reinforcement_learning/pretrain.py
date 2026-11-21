"""Behaviour-cloning warm start for the PPO policy (optional, configurable).

Long episodes (~1 150 decisions) and delayed consequences (inducting too many
trains in the morning exhausts induction-window capacity hours later) make
PPO from a random policy very slow on a CPU budget. We therefore initialise
the policy by imitation of the rule-based heuristic on TRAINING seeds only:

1. roll out the teacher in the Gym environment, recording (obs, mask, action,
   reward); 2. compute discounted returns; 3. fit the actor with weighted
   cross-entropy on the masked action distribution and the critic with MSE on
   the (scaled) returns. PPO then optimises the actual multi-objective reward
   from this starting point. Validation-based model selection reports whether
   PPO improved on the warm start (see results/training/validation_curve.csv).
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn.functional as F

from .gym_environment import MetroInductionEnv, scheduler_to_action

logger = logging.getLogger(__name__)


def collect_demonstrations(env: MetroInductionEnv, teacher, episodes: int, seed_rng: np.random.Generator,
                           gamma: float) -> dict:
    obs_l, mask_l, act_l, ret_l = [], [], [], []
    returns = []
    for _ in range(episodes):
        opts = {"episode_seed": int(seed_rng.integers(env.seed_low, env.seed_high)),
                "scenario_type": str(seed_rng.choice(env.mix_types, p=env.mix_p))}
        obs, _ = env.reset(options=opts)
        done, rews = False, []
        while not done:
            mask = env.action_masks()
            a = scheduler_to_action(env, teacher)
            if not mask[a]:
                a = 0
            obs_l.append(obs)
            mask_l.append(mask)
            act_l.append(a)
            obs, r, done, _, _ = env.step(a)
            rews.append(r)
        g, disc = 0.0, []
        for r in reversed(rews):
            g = r + gamma * g
            disc.append(g)
        ret_l.extend(reversed(disc))
        returns.append(sum(rews))
    logger.info("Collected %d demonstration steps from %d episodes (teacher mean return %.1f)",
                len(act_l), episodes, float(np.mean(returns)))
    return {"obs": np.asarray(obs_l, np.float32), "mask": np.asarray(mask_l, bool),
            "act": np.asarray(act_l, np.int64), "ret": np.asarray(ret_l, np.float32),
            "teacher_mean_return": float(np.mean(returns))}


def behaviour_clone(model, demos: dict, epochs: int = 20, batch_size: int = 512, lr: float = 1e-3,
                    value_weight: float = 0.5, seed: int = 0) -> dict:
    policy = model.policy
    policy.set_training_mode(True)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    n = len(demos["act"])
    acts = demos["act"]
    counts = np.bincount(acts, minlength=model.action_space.n).astype(float)
    # Balance WAIT (very frequent) against INDUCT decisions.
    w_class = np.where(counts > 0, (n / np.maximum(counts, 1)) ** 0.5, 0.0)
    w = torch.as_tensor(w_class[acts] / w_class[acts].mean(), dtype=torch.float32)
    ret = torch.as_tensor(demos["ret"])
    rng = np.random.default_rng(seed)
    hist = []
    for ep in range(epochs):
        idx = rng.permutation(n)
        tot_pi = tot_v = correct = 0.0
        for s in range(0, n, batch_size):
            b = idx[s:s + batch_size]
            obs = torch.as_tensor(demos["obs"][b])
            dist = policy.get_distribution(obs, action_masks=demos["mask"][b])
            a = torch.as_tensor(acts[b])
            loss_pi = -(w[b] * dist.log_prob(a)).mean()
            values = policy.predict_values(obs).flatten()
            loss_v = F.mse_loss(values, ret[b])
            loss = loss_pi + value_weight * loss_v / max(float(ret.var()), 1.0)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            opt.step()
            tot_pi += float(loss_pi) * len(b)
            tot_v += float(loss_v) * len(b)
            correct += float((dist.distribution.probs.argmax(-1) == a).float().sum())
        hist.append({"epoch": ep, "policy_loss": tot_pi / n, "value_mse": tot_v / n, "accuracy": correct / n})
    policy.set_training_mode(False)
    logger.info("Behaviour cloning done: accuracy %.3f, value MSE %.1f", hist[-1]["accuracy"], hist[-1]["value_mse"])
    return {"history": hist, "final_accuracy": hist[-1]["accuracy"]}
