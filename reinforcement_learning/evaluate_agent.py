"""Evaluate FCFS, heuristic and PPO on UNSEEN seeds under identical conditions.

Seeds: ``evaluation.seed_base + i`` (default 900 000 000+), which never overlap
the training range [train_seed_low, train_seed_high) nor the validation seeds.
Writes to results/evaluation/:

    episode_metrics.csv     one row per (scheduler, episode)
    summary.csv             mean / std / CI / improvement / paired stats per metric
    comparison_table.csv    wide report table (also printed)
    scenario_summary.csv    means per scenario type and scheduler
    examples/               induction history, occupancy and monitor logs of example days
    evaluation_meta.json

Usage:  python -m reinforcement_learning.evaluate_agent [--episodes-per-scenario 40]
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import pandas as pd

from evaluation.comparison import evaluation_scenarios, run_comparison
from evaluation.statistical_analysis import comparison_matrix, summary_table
from scheduling.base import load_rl_config
from scheduling.baseline_fcfs import FCFSScheduler
from scheduling.heuristic_scheduler import HeuristicScheduler
from simulation.config import PROJECT_ROOT, load_config

logger = logging.getLogger(__name__)
EVAL_DIR = PROJECT_ROOT / "results" / "evaluation"


def check_seed_disjointness(rl_cfg: dict, n_episodes: int) -> None:
    tr, ev = rl_cfg["training"], rl_cfg["evaluation"]
    lo, hi = int(ev["seed_base"]), int(ev["seed_base"]) + n_episodes
    if lo < int(tr["train_seed_high"]) and hi > int(tr["train_seed_low"]):
        raise ValueError("Evaluation seeds overlap the training seed range")
    v_lo = int(tr["validation_seed_base"])
    if lo < v_lo + int(tr["eval_episodes"]) and hi > v_lo:
        raise ValueError("Evaluation seeds overlap the validation seeds")


def save_examples(examples: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for (sched, st), ep in examples.items():
        stem = f"{sched}_{st}"
        ep.result.inductions.to_csv(out / f"{stem}_inductions.csv", index=False)
        ep.result.occupancy.to_csv(out / f"{stem}_occupancy.csv", index=False)
        ep.result.monitor.to_csv(out / f"{stem}_monitor.csv", index=False)
        ep.result.closures.to_csv(out / f"{stem}_closures.csv", index=False)
        pd.DataFrame(ep.explanations).to_csv(out / f"{stem}_decisions.csv", index=False)


def evaluate(episodes_per_scenario: int | None = None, model_path: str | None = None, include_ppo: bool = True,
             config: str | None = None, rl_config: str | None = None, out_dir: Path = EVAL_DIR) -> dict:
    cfg = load_config(config)
    rl_cfg = load_rl_config(rl_config)
    per = int(episodes_per_scenario or rl_cfg["evaluation"]["episodes_per_scenario"])
    check_seed_disjointness(rl_cfg, per * 3)
    schedulers = [FCFSScheduler(), HeuristicScheduler.from_config(rl_cfg)]
    if include_ppo:
        from scheduling.rl_scheduler import RLScheduler
        schedulers.append(RLScheduler(model_path, cfg, rl_cfg))
    scenarios = evaluation_scenarios(cfg, int(rl_cfg["evaluation"]["seed_base"]), per)
    t0 = time.time()
    df, examples = run_comparison(cfg, schedulers, scenarios, int(rl_cfg["scheduling"]["max_inductions_per_tick"]))
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "episode_metrics.csv", index=False)
    summary = summary_table(df, "FCFS")
    summary.to_csv(out_dir / "summary.csv", index=False)
    table = comparison_matrix(summary, "FCFS")
    table.to_csv(out_dir / "comparison_table.csv", index=False)
    scen = df.groupby(["scenario_type", "scheduler"], sort=False).mean(numeric_only=True).drop(
        columns=["seed", "weekday"], errors="ignore").reset_index()
    scen.to_csv(out_dir / "scenario_summary.csv", index=False)
    save_examples(examples, out_dir / "examples")
    meta = {"episodes": len(scenarios), "episodes_per_scenario": per, "schedulers": [s.name for s in schedulers],
            "seed_base": int(rl_cfg["evaluation"]["seed_base"]),
            "seeds": sorted({int(s.seed) for s in scenarios})[:5] + ["..."],
            "total_safety_violations": int(df["safety_violations"].sum()),
            "runtime_s": round(time.time() - t0, 1),
            "note": "All metrics computed from simulation runs on synthetic scenarios; not KMRL data."}
    (out_dir / "evaluation_meta.json").write_text(json.dumps(meta, indent=2))
    logger.info("Evaluation finished: %d episodes x %d schedulers in %.0fs", len(scenarios), len(schedulers),
                time.time() - t0)
    return {"episodes": df, "summary": summary, "table": table, "scenario": scen, "meta": meta}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate schedulers on unseen seeds")
    ap.add_argument("--episodes-per-scenario", type=int, default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--no-ppo", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    res = evaluate(args.episodes_per_scenario, args.model, not args.no_ppo)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(res["table"].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
