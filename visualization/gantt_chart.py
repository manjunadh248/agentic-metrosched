"""Train induction timeline (Gantt) and track occupancy over time."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .style import SECTION_COLORS, TEXT_2, apply_style, footnote, plt

ROUTE_KINDS = ["exit", "shared", "entry", "platform"]
KIND_LABEL = {"exit": "depot exit track", "shared": "shared section", "entry": "mainline entry junction",
              "platform": "platform dwell"}


def _hours(s):
    return np.asarray(s, dtype=float) / 3600.0


def _kind(section_id: str) -> str:
    if section_id.startswith("EXIT"):
        return "exit"
    if section_id.startswith("ENTRY"):
        return "entry"
    if section_id.startswith("PF"):
        return "platform"
    if section_id.startswith("RECEPTION"):
        return "reception"
    return "shared"


def gantt_chart(examples_dir: Path, schedulers: list[str], scenario_type: str, path: Path,
                until_h: float = 11.0) -> Path:
    """One panel per scheduler: each induction's route blocks over time (same day, same seed)."""
    apply_style()
    fig, axes = plt.subplots(len(schedulers), 1, figsize=(12, 3.3 * len(schedulers)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, schedulers):
        occ = pd.read_csv(examples_dir / f"{name}_{scenario_type}_occupancy.csv")
        ind = pd.read_csv(examples_dir / f"{name}_{scenario_type}_inductions.csv")
        occ = occ[occ["purpose"] == "induction"].copy()
        occ["kind"] = occ["section_id"].map(_kind)
        order = ind.sort_values("authorised_time_s")
        rows = {(r.train_id, r.induction_id): i for i, r in enumerate(order.itertuples())}
        occ = occ[occ["start_s"] / 3600 <= until_h]
        for r in order.itertuples():
            y = rows[(r.train_id, r.induction_id)]
            if r.authorised_time_s / 3600 > until_h:
                continue
            sub = occ[(occ["train_id"] == r.train_id) & (occ["start_s"] >= r.authorised_time_s)
                      & (occ["end_s"] <= r.service_start_time_s)]
            # waiting before the first block (in depot, authorised)
            first = sub["start_s"].min() if len(sub) else r.authorised_time_s
            if first > r.authorised_time_s:
                ax.barh(y, (first - r.authorised_time_s) / 3600, left=r.authorised_time_s / 3600, height=0.6,
                        color="#d9d8d3")
            for s in sub.itertuples():
                ax.barh(y, (s.end_s - s.start_s) / 3600, left=s.start_s / 3600, height=0.6,
                        color=SECTION_COLORS[s.kind], edgecolor="#fcfcfb", linewidth=0.6)
        labels = [f"{r.train_id}" for r in order.itertuples() if r.authorised_time_s / 3600 <= until_h]
        ax.set_yticks(range(len(labels)), labels, fontsize=7)
        ax.invert_yaxis()
        mean_wait = (ind["track_wait_s"].mean() / 60) if len(ind) else 0
        ax.set_title(f"{name}: {len(ind)} inductions, mean track wait {mean_wait:.1f} min", fontsize=10)
        ax.set_ylabel("Induction (train ID)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=SECTION_COLORS[k]) for k in ROUTE_KINDS]
    handles.append(plt.Rectangle((0, 0), 1, 1, color="#d9d8d3"))
    fig.legend(handles, [KIND_LABEL[k] for k in ROUTE_KINDS] + ["queued in depot (authorised)"],
               loc="upper center", ncol=5, bbox_to_anchor=(0.5, 0.995), fontsize=8)
    axes[-1].set_xlabel("Time of day (hours)")
    start = 4.0
    axes[-1].set_xlim(start, until_h)
    fig.suptitle(f"Train induction timeline – same {scenario_type.replace('_', ' ')} day (identical seed)",
                 y=0.965, fontweight="bold")
    footnote(fig)
    fig.tight_layout(rect=(0, 0.01, 1, 0.95))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def occupancy_series(occ: pd.DataFrame, section_ids: list[str], start: int, end: int, step: int = 300) -> pd.DataFrame:
    """Fraction of each ``step`` window each section was occupied (0..1)."""
    bins = np.arange(start, end, step)
    out = {}
    for sid in section_ids:
        sub = occ[occ["section_id"] == sid]
        busy = np.zeros(len(bins))
        for a, b in zip(sub["start_s"], sub["end_s"]):
            busy += np.clip(np.minimum(bins + step, b) - np.maximum(bins, a), 0, None)
        out[sid] = busy / step
    return pd.DataFrame(out, index=bins)


def track_occupancy_plot(examples_dir: Path, schedulers: list[str], scenario_type: str, path: Path) -> Path:
    """Heat-strip per section: share of each 5-min window occupied, per scheduler (same day)."""
    apply_style()
    frames = {n: pd.read_csv(examples_dir / f"{n}_{scenario_type}_occupancy.csv") for n in schedulers}
    sections = sorted({s for f in frames.values() for s in f["section_id"].unique()},
                      key=lambda s: (["exit", "shared", "entry", "platform", "reception"].index(_kind(s)), s))
    start, end = 4 * 3600, 23 * 3600
    fig, axes = plt.subplots(len(schedulers), 1, figsize=(12, 0.32 * len(sections) * len(schedulers) + 2.0),
                             sharex=True)
    axes = np.atleast_1d(axes)
    im = None
    for ax, name in zip(axes, schedulers):
        series = occupancy_series(frames[name], sections, start, end)
        im = ax.imshow(series.T.to_numpy(), aspect="auto", cmap="Blues", vmin=0, vmax=1,
                       extent=(start / 3600, end / 3600, len(sections) - 0.5, -0.5), interpolation="nearest")
        cl = examples_dir / f"{name}_{scenario_type}_closures.csv"
        if cl.exists():
            c = pd.read_csv(cl)
            for r in c.itertuples():
                if r.section_id in sections:
                    y = sections.index(r.section_id)
                    ax.plot([r.start_s / 3600, r.end_s / 3600], [y, y], color="#e34948", linewidth=3,
                            solid_capstyle="butt")
        ax.set_yticks(range(len(sections)), sections, fontsize=7)
        ax.set_title(f"{name}", fontsize=10, loc="left")
        ax.grid(False)
    axes[-1].set_xlabel("Time of day (hours)")
    cbar = fig.colorbar(im, ax=list(axes), shrink=0.8, pad=0.01)
    cbar.set_label("Share of 5-min window occupied")
    fig.suptitle(f"Track occupancy over time – {scenario_type.replace('_', ' ')} day "
                 "(red line = closure / block)", fontweight="bold", x=0.45)
    fig.text(0.01, 0.005, "Synthetic simulation data – not KMRL operational data", fontsize=7, color=TEXT_2)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
