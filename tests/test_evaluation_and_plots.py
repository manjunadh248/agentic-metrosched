"""Phase 5/6: statistics, evaluation pipeline, plots; Phase 7: dashboard smoke test."""

import math

import numpy as np
import pandas as pd
import pytest

from evaluation.comparison import evaluation_scenarios, run_comparison
from evaluation.statistical_analysis import (comparison_matrix, describe, paired_bootstrap_ci,
                                             percentage_improvement, sign_test_p, summary_table)
from scheduling.baseline_fcfs import FCFSScheduler
from scheduling.heuristic_scheduler import HeuristicScheduler
from tests.conftest import ROOT

APP = ROOT / "dashboard" / "app.py"


def test_percentage_improvement_directions():
    assert percentage_improvement(10, 8, "lower") == pytest.approx(20)
    assert percentage_improvement(90, 99, "higher") == pytest.approx(10)
    assert percentage_improvement(10, 12, "lower") == pytest.approx(-20)
    assert math.isnan(percentage_improvement(0, 5, "lower"))


def test_describe_and_ci():
    d = describe([1, 2, 3, 4])
    assert d["mean"] == 2.5 and d["n"] == 4
    assert d["std"] == pytest.approx(np.std([1, 2, 3, 4], ddof=1))
    assert d["ci95_low"] < 2.5 < d["ci95_high"]
    lo, hi = paired_bootstrap_ci([1.0] * 10)
    assert lo == hi == 1.0


def test_sign_test():
    assert sign_test_p([1] * 10) == pytest.approx(2 / 1024)
    assert sign_test_p([1, -1]) == 1.0
    assert sign_test_p([0, 0]) == 1.0


@pytest.fixture(scope="module")
def small_eval(cfg):
    scen = evaluation_scenarios(cfg, 123_000_000, 2)
    df, examples = run_comparison(cfg, [FCFSScheduler(), HeuristicScheduler()], scen)
    return df, examples, scen


def test_identical_conditions_and_unseen_seeds(small_eval, cfg):
    df, _, scen = small_eval
    assert len(scen) == 6 and {s.scenario_type for s in scen} == {"normal", "peak_demand", "disruption"}
    by = df.groupby("scheduler")["seed"].apply(sorted)
    assert by["FCFS"] == by["Heuristic"]
    assert df["safety_violations"].sum() == 0


def test_summary_table(small_eval):
    df, _, _ = small_eval
    s = summary_table(df)
    row = s[(s.metric == "demand_coverage_pct") & (s.scheduler == "Heuristic")].iloc[0]
    base = df[df.scheduler == "FCFS"]["demand_coverage_pct"].mean()
    assert row["improvement_vs_baseline_pct"] == pytest.approx((row["mean"] - base) / base * 100)
    assert "sign_test_p" in s.columns
    t = comparison_matrix(s)
    assert {"Metric", "FCFS", "Heuristic"} <= set(t.columns)


def test_plots_generated(small_eval, tmp_path):
    from reinforcement_learning.evaluate_agent import save_examples
    from evaluation.statistical_analysis import summary_table as st
    from visualization.performance_plots import generate_all

    df, examples, _ = small_eval
    ev = tmp_path / "evaluation"
    ev.mkdir()
    df.to_csv(ev / "episode_metrics.csv", index=False)
    st(df).to_csv(ev / "summary.csv", index=False)
    save_examples(examples, ev / "examples")
    paths = generate_all(ev, tmp_path / "no_training", tmp_path / "plots")
    names = {p.name for p in paths}
    for expected in ["01_induction_conflicts.png", "02_avg_induction_delay.png", "03_train_waiting_time.png",
                     "04_demand_coverage.png", "05_track_utilization.png", "07_induction_gantt_disruption.png",
                     "08_track_occupancy_disruption.png", "09_scenario_comparison.png"]:
        assert expected in names
    assert all(p.stat().st_size > 10_000 for p in paths)


def test_dashboard_pages_run_without_errors():
    from streamlit.testing.v1 import AppTest

    pages = ["1 · Operations overview", "4 · Performance analytics", "5 · System configuration"]
    for page in pages:
        at = AppTest.from_file(str(APP), default_timeout=120)
        at.run()
        at.sidebar.radio[0].set_value(page).run()
        assert not at.exception, (page, [e.value for e in at.exception])


def test_dashboard_simulation_button_runs_simulation():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=300)
    at.run()
    at.sidebar.radio[0].set_value("2 · Metro simulation").run()
    at.selectbox(key="sim_sched").set_value("FCFS").run()
    at.button[0].click().run()
    assert not at.exception
    labels = {m.label: m.value for m in at.metric}
    assert labels["Safety violations"] == "0"
    assert int(labels["Inductions"]) > 0
