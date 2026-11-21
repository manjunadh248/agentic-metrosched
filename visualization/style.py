"""Shared plot styling: one fixed, validated categorical palette (never cycled)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless / Windows-safe; figures are saved, not shown
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
# Fixed scheduler -> colour mapping (colour follows the entity, never its rank).
SCHEDULER_COLORS = {"FCFS": "#2a78d6", "Heuristic": "#eb6834", "PPO": "#1baf7a"}
EXTRA = ["#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SECTION_COLORS = {"exit": "#2a78d6", "shared": "#eb6834", "entry": "#1baf7a", "platform": "#eda100",
                  "reception": "#6250d6"}


def color_for(name: str, i: int = 0) -> str:
    return SCHEDULER_COLORS.get(name, EXTRA[i % len(EXTRA)])


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2, "axes.titlecolor": TEXT, "axes.titleweight": "bold",
        "axes.titlesize": 12, "axes.labelsize": 10, "xtick.color": TEXT_2, "ytick.color": TEXT_2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "font.size": 10, "lines.linewidth": 2.0,
    })


def footnote(fig, text: str = "Synthetic simulation data – not KMRL operational data") -> None:
    fig.text(0.99, 0.005, text, ha="right", va="bottom", fontsize=7, color=TEXT_2)
