"""Phase 8: end-to-end pipeline on small settings (writes only to temporary folders)."""

import json

import pandas as pd

import main as cli
from reinforcement_learning.evaluate_agent import evaluate
from visualization.performance_plots import generate_all


def test_cli_generate_data(tmp_path):
    rc = cli.main(["--log-level", "WARNING", "generate-data", "--seed", "3", "--days", "3",
                   "--output-dir", str(tmp_path)])
    assert rc == 0
    m = json.loads((tmp_path / "dataset_manifest.json").read_text())
    assert m["safety_violations"] == 0 and m["n_days"] == 3


def test_cli_rejects_bad_config(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("fleet: {size: 5}\n")
    assert cli.main(["--log-level", "ERROR", "generate-data", "--days", "1", "--config", str(bad),
                     "--output-dir", str(tmp_path / "o")]) == 2


def test_pipeline_evaluate_and_plot(tmp_path, cfg):
    """Evaluation (FCFS + heuristic, plus PPO if a trained model exists) -> plots."""
    from scheduling.rl_scheduler import DEFAULT_MODEL

    res = evaluate(episodes_per_scenario=1, include_ppo=DEFAULT_MODEL.exists(), out_dir=tmp_path / "evaluation")
    ep = pd.read_csv(tmp_path / "evaluation" / "episode_metrics.csv")
    assert len(ep) == 3 * len(res["meta"]["schedulers"])
    assert res["meta"]["total_safety_violations"] == 0
    paths = generate_all(tmp_path / "evaluation", tmp_path / "missing", tmp_path / "plots")
    assert len(paths) >= 9


def test_parser_has_all_commands():
    p = cli.build_parser()
    choices = p._subparsers._group_actions[0].choices
    assert {"generate-data", "run-baseline", "train", "evaluate", "plot", "dashboard", "all"} <= set(choices)
