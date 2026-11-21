"""Before-vs-after comparison plots, scenario comparison and PPO reward convergence.

Every plot is drawn from CSV files produced by real simulation runs
(results/evaluation/*.csv, results/training/*.csv); nothing is hardcoded.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.metrics import METRICS
from simulation.config import PROJECT_ROOT

from .style import SCHEDULER_COLORS, TEXT, TEXT_2, apply_style, color_for, footnote, plt

logger = logging.getLogger(__name__)
PLOTS_DIR = PROJECT_ROOT / "results" / "plots"

BEFORE_AFTER = [
    ("induction_conflicts", "01_induction_conflicts.png"),
    ("avg_induction_delay_min", "02_avg_induction_delay.png"),
    ("avg_waiting_time_min", "03_train_waiting_time.png"),
    ("demand_coverage_pct", "04_demand_coverage.png"),
    ("track_utilization_pct", "05_track_utilization.png"),
]


def _order(names) -> list[str]:
    pref = ["FCFS", "Heuristic", "PPO"]
    return [n for n in pref if n in names] + [n for n in names if n not in pref]


def metric_bar(df: pd.DataFrame, metric: str, path: Path, baseline: str = "FCFS") -> Path:
    """Mean ± std bar per scheduler with value labels and improvement vs baseline."""
    apply_style()
    label, unit, direction = METRICS[metric]
    names = _order(df["scheduler"].unique())
    stats = df.groupby("scheduler")[metric].agg(["mean", "std", "count"]).reindex(names)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    x = np.arange(len(names))
    bars = ax.bar(x, stats["mean"], width=0.55, color=[color_for(n, i) for i, n in enumerate(names)],
                  yerr=stats["std"], capsize=4, error_kw={"ecolor": TEXT_2, "elinewidth": 1})
    base = stats.loc[baseline, "mean"] if baseline in stats.index else np.nan
    top = float((stats["mean"] + stats["std"].fillna(0)).max())
    pad = max(top, 1e-6) * 0.03
    for i, (b, n) in enumerate(zip(bars, names)):
        m = stats.loc[n, "mean"]
        txt = f"{m:.2f}"
        if n != baseline and np.isfinite(base) and base != 0:
            imp = ((base - m) if direction == "lower" else (m - base)) / abs(base) * 100
            txt += f"\n({imp:+.1f}% {'change' if direction == 'neutral' else 'vs ' + baseline})"
        sd = stats.loc[n, "std"] if np.isfinite(stats.loc[n, "std"]) else 0.0
        ax.text(b.get_x() + b.get_width() / 2, m + sd + pad, txt,
                ha="center", va="bottom", fontsize=9, color=TEXT)
    ax.set_xticks(x, names)
    ax.set_ylabel(f"{label} ({unit})")
    better = {"lower": "lower is better", "higher": "higher is better", "neutral": "informational"}[direction]
    n_ep = int(stats["count"].min())
    ax.set_title(f"{label}: before (FCFS) vs after\nmean ± std over {n_ep} unseen episodes – {better}")
    ax.set_ylim(0, max(1e-6, top) * 1.35)
    footnote(fig)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def reward_convergence(train_dir: Path, path: Path, window: int = 20) -> Path | None:
    """Training episode returns (rolling mean) + validation return vs timesteps, with references."""
    apply_style()
    monitors = sorted(train_dir.glob("monitor_*.monitor.csv"))
    val = train_dir / "validation_curve.csv"
    if not monitors and not val.exists():
        logger.warning("No training logs in %s – skipping reward convergence plot", train_dir)
        return None
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    if monitors:
        frames = []
        for f in monitors:
            m = pd.read_csv(f, skiprows=1)
            m["timesteps"] = m["l"].cumsum() * len(monitors)  # approx. global steps (envs run in lockstep)
            frames.append(m)
        mon = pd.concat(frames).sort_values("timesteps")
        ax.plot(mon["timesteps"], mon["r"], color="#86b6ef", linewidth=0, marker="o", markersize=2.5,
                alpha=0.45, label="training episode return")
        ax.plot(mon["timesteps"], mon["r"].rolling(window, min_periods=1).mean(), color="#1c5cab",
                label=f"training return ({window}-episode rolling mean)")
    if val.exists():
        v = pd.read_csv(val)
        ax.plot(v["timesteps"], v["val_mean_return"], color=SCHEDULER_COLORS["PPO"], marker="o", markersize=8,
                label="validation return (held-out validation days)")
    ref = train_dir / "reference_returns.json"
    if ref.exists():
        refs = json.loads(ref.read_text())
        for name, r in refs.items():
            ax.axhline(r["mean"], color=color_for(name), linestyle="--", linewidth=1.5,
                       label=f"{name} on validation seeds ({r['mean']:.0f})")
    ax.set_xlabel("Training timesteps (agent decisions)")
    ax.set_ylabel("Episode return (reward units)")
    ax.set_title("PPO reward convergence")
    # y-range: keep every validation point and reference line, clip only extreme early training outliers
    lows = [np.nanpercentile(mon["r"], 3)] if monitors else []
    if val.exists():
        lows.append(float(v["val_mean_return"].min()))
    if lows:
        lo = min(lows)
        ax.set_ylim(bottom=lo - 0.08 * abs(lo) - 5)
    ax.legend(loc="lower right", fontsize=8)
    footnote(fig)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def scenario_comparison(df: pd.DataFrame, path: Path) -> Path:
    """Small multiples: key metrics per scenario type, one bar per scheduler."""
    apply_style()
    metrics = ["avg_induction_delay_min", "avg_waiting_time_min", "demand_coverage_pct",
               "shortfall_train_hours", "missed_opportunities", "induction_conflicts"]
    names = _order(df["scheduler"].unique())
    types = [t for t in ["normal", "peak_demand", "disruption"] if t in df["scenario_type"].unique()]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.2))
    width = 0.8 / len(names)
    for ax, metric in zip(axes.flat, metrics):
        label, unit, direction = METRICS[metric]
        g = df.groupby(["scenario_type", "scheduler"])[metric].mean()
        for i, n in enumerate(names):
            vals = [g.get((t, n), np.nan) for t in types]
            xs = np.arange(len(types)) + (i - (len(names) - 1) / 2) * width
            ax.bar(xs, vals, width=width * 0.92, color=color_for(n, i), label=n)
            for xx, vv in zip(xs, vals):
                ax.text(xx, vv, f"{vv:.1f}", ha="center", va="bottom", fontsize=7, color=TEXT)
        ax.set_xticks(np.arange(len(types)), [t.replace("_", " ") for t in types])
        ax.set_title(f"{label} ({unit})", fontsize=10)
        ax.text(0.99, 0.98, {"lower": "↓ better", "higher": "↑ better", "neutral": ""}[direction],
                transform=ax.transAxes, ha="right", va="top", fontsize=8, color=TEXT_2)
        if metric == "demand_coverage_pct":
            lo = float(df[metric].min())
            ax.set_ylim(max(0, lo - 2), 100.5)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(names), bbox_to_anchor=(0.5, 0.995))
    fig.suptitle("Performance by operational scenario (mean over unseen episodes)", y=0.955, fontweight="bold")
    footnote(fig)
    fig.tight_layout(rect=(0, 0.01, 1, 0.93))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def improvement_overview(summary: pd.DataFrame, path: Path, baseline: str = "FCFS") -> Path:
    """Horizontal bars of % improvement vs baseline for every directional metric."""
    apply_style()
    s = summary[(summary["scheduler"] != baseline) & (summary["direction"] != "neutral")].copy()
    s = s[np.isfinite(s["improvement_vs_baseline_pct"])]
    labels = list(dict.fromkeys(s["label"]))
    names = _order(s["scheduler"].unique())
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(labels) + 1.6))
    h = 0.8 / max(1, len(names))
    for i, n in enumerate(names):
        sub = s[s["scheduler"] == n].set_index("label").reindex(labels)
        ys = np.arange(len(labels)) + (i - (len(names) - 1) / 2) * h
        ax.barh(ys, sub["improvement_vs_baseline_pct"], height=h * 0.9, color=color_for(n, i), label=n)
        for y, v in zip(ys, sub["improvement_vs_baseline_pct"]):
            if np.isfinite(v):
                ax.text(v, y, f" {v:+.1f}% ", va="center", ha="left" if v >= 0 else "right", fontsize=8, color=TEXT)
    ax.axvline(0, color=TEXT_2, linewidth=1)
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.invert_yaxis()
    ax.set_xlabel(f"Improvement vs {baseline} (%) – positive is better")
    ax.set_title(f"Improvement over the {baseline} baseline (unseen evaluation episodes)")
    ax.legend(loc="lower right")
    footnote(fig)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def generate_all(eval_dir: Path | None = None, train_dir: Path | None = None, out_dir: Path | None = None) -> list[Path]:
    """Generate every required plot from saved evaluation/training results."""
    from .gantt_chart import gantt_chart, track_occupancy_plot

    eval_dir = eval_dir or PROJECT_ROOT / "results" / "evaluation"
    train_dir = train_dir or PROJECT_ROOT / "results" / "training"
    out = out_dir or PLOTS_DIR
    out.mkdir(parents=True, exist_ok=True)
    ep_file = eval_dir / "episode_metrics.csv"
    if not ep_file.exists():
        raise FileNotFoundError(f"{ep_file} not found – run the evaluation first (python main.py evaluate)")
    df = pd.read_csv(ep_file)
    paths = [metric_bar(df, m, out / f) for m, f in BEFORE_AFTER]
    p = reward_convergence(train_dir, out / "06_reward_convergence.png")
    if p:
        paths.append(p)
    ex = eval_dir / "examples"
    names = _order(df["scheduler"].unique())
    for st in ("peak_demand", "disruption"):
        if all((ex / f"{n}_{st}_occupancy.csv").exists() for n in names):
            paths.append(gantt_chart(ex, names, st, out / f"07_induction_gantt_{st}.png"))
            paths.append(track_occupancy_plot(ex, names, st, out / f"08_track_occupancy_{st}.png"))
    paths.append(scenario_comparison(df, out / "09_scenario_comparison.png"))
    summ = eval_dir / "summary.csv"
    if summ.exists():
        paths.append(improvement_overview(pd.read_csv(summ), out / "10_improvement_overview.png"))
    logger.info("Saved %d plots to %s", len(paths), out)
    return paths
