"""Statistics for scheduler comparison (NumPy only, no SciPy dependency).

* mean / std / 95 % confidence interval of each metric,
* percentage improvement vs the FCFS baseline:
      lower-is-better : (baseline - optimized) / baseline * 100
      higher-is-better: (optimized - baseline) / baseline * 100
      neutral metrics : plain relative change (optimized - baseline) / baseline * 100
* paired bootstrap 95 % CI of the per-episode difference (same seeds),
* exact two-sided sign test p-value on paired episodes.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .metrics import METRICS


def describe(values) -> dict:
    v = np.asarray(values, dtype=float)
    n = len(v)
    mean = float(v.mean()) if n else float("nan")
    std = float(v.std(ddof=1)) if n > 1 else 0.0
    half = 1.96 * std / math.sqrt(n) if n > 1 else 0.0
    return {"n": n, "mean": mean, "std": std, "ci95_low": mean - half, "ci95_high": mean + half}


def percentage_improvement(baseline: float, optimized: float, direction: str) -> float:
    """Improvement in %, positive = better (for 'neutral' metrics: relative change)."""
    if baseline == 0 or not np.isfinite(baseline):
        return float("nan")
    if direction == "lower":
        return (baseline - optimized) / abs(baseline) * 100.0
    return (optimized - baseline) / abs(baseline) * 100.0


def paired_bootstrap_ci(diff, n_boot: int = 5000, seed: int = 0) -> tuple[float, float]:
    d = np.asarray(diff, dtype=float)
    if len(d) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(d, size=(n_boot, len(d)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def sign_test_p(diff) -> float:
    """Two-sided exact sign test on non-zero paired differences."""
    d = np.asarray(diff, dtype=float)
    d = d[d != 0]
    n = len(d)
    if n == 0:
        return 1.0
    k = int(min((d > 0).sum(), (d < 0).sum()))
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n * 2
    return float(min(1.0, p))


def summary_table(df: pd.DataFrame, baseline: str = "FCFS", metrics: list[str] | None = None) -> pd.DataFrame:
    """One row per (metric, scheduler) with stats and improvement vs baseline."""
    metrics = metrics or [m for m in METRICS if m in df.columns]
    rows = []
    base = df[df["scheduler"] == baseline].set_index(["scenario_type", "seed"])
    for metric in metrics:
        label, unit, direction = METRICS[metric]
        b_mean = float(base[metric].mean()) if len(base) else float("nan")
        for sched, sub in df.groupby("scheduler", sort=False):
            st = describe(sub[metric])
            row = {"metric": metric, "label": label, "unit": unit, "direction": direction, "scheduler": sched,
                   **st, "baseline_mean": b_mean,
                   "improvement_vs_baseline_pct": percentage_improvement(b_mean, st["mean"], direction)}
            if sched != baseline and len(base):
                paired = sub.set_index(["scenario_type", "seed"])[metric].reindex(base.index)
                diff = (paired - base[metric]).dropna().to_numpy()
                lo, hi = paired_bootstrap_ci(diff)
                row.update({"paired_diff_mean": float(diff.mean()) if len(diff) else float("nan"),
                            "paired_diff_ci95_low": lo, "paired_diff_ci95_high": hi,
                            "sign_test_p": sign_test_p(diff)})
            rows.append(row)
    return pd.DataFrame(rows)


def comparison_matrix(summary: pd.DataFrame, baseline: str = "FCFS") -> pd.DataFrame:
    """Wide 'Metric | FCFS | X | Y | improvement' table for reports."""
    out = []
    for metric, sub in summary.groupby("metric", sort=False):
        row = {"Metric": sub["label"].iloc[0], "Unit": sub["unit"].iloc[0], "Better": sub["direction"].iloc[0]}
        for _, r in sub.iterrows():
            row[r["scheduler"]] = f"{r['mean']:.2f} ± {r['std']:.2f}"
            if r["scheduler"] != baseline:
                imp = r["improvement_vs_baseline_pct"]
                row[f"{r['scheduler']} vs {baseline} (%)"] = "n/a" if not np.isfinite(imp) else f"{imp:+.1f}"
        out.append(row)
    return pd.DataFrame(out)
