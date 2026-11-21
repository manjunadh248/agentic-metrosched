"""Interactive Plotly figures for the Streamlit dashboard (built from real simulation logs)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from simulation.config import seconds_to_clock

from .style import SCHEDULER_COLORS, SECTION_COLORS

STATUS_COLORS = {"Available": "#2a78d6", "Maintenance": "#e34948", "Inducted": "#eda100",
                 "Running": "#1baf7a", "Delayed": "#e87ba4"}
LAYOUT = dict(template="plotly_white", margin=dict(l=40, r=20, t=50, b=40), font=dict(size=12),
              hoverlabel=dict(bgcolor="white"))


def _h(s):
    return np.asarray(s, dtype=float) / 3600.0


def demand_figure(demand: pd.DataFrame, title: str = "Passenger demand and required trains") -> go.Figure:
    fig = go.Figure()
    x = _h(demand["interval_start_s"])
    fig.add_bar(x=x, y=demand["demand_pphpd"], name="Demand (pphpd)", marker_color="#86b6ef",
                hovertemplate="%{customdata}<br>%{y:.0f} passengers/h/dir<extra></extra>",
                customdata=demand["interval_start"])
    fig.add_scatter(x=x, y=demand["required_trains"], name="Required trains", yaxis="y2",
                    line=dict(color="#eb6834", width=2, shape="hv"),
                    hovertemplate="%{y} trains required<extra></extra>")
    fig.update_layout(**LAYOUT, title=title, xaxis_title="Time of day (h)", yaxis_title="Passengers per hour per direction",
                      yaxis2=dict(title="Required trains", overlaying="y", side="right", showgrid=False,
                                  rangemode="tozero", dtick=2, tickformat="d"),
                      legend=dict(orientation="h", y=1.08), bargap=0.05)
    return fig


def service_figure(monitor: pd.DataFrame, title: str = "Trains in service vs requirement") -> go.Figure:
    m = monitor
    fig = go.Figure()
    fig.add_scatter(x=_h(m["time_s"]), y=m["required_now"], name="Required", line=dict(color="#eb6834", shape="hv"))
    fig.add_scatter(x=_h(m["time_s"]), y=m["in_service"], name="In service", line=dict(color="#1baf7a", shape="hv"))
    fig.add_scatter(x=_h(m["time_s"]), y=m["waiting"], name="Queued on induction route",
                    line=dict(color="#6250d6", shape="hv", width=1.5))
    fig.update_layout(**LAYOUT, title=title, xaxis_title="Time of day (h)", yaxis_title="Trains",
                      hovermode="x unified", legend=dict(orientation="h", y=1.08))
    return fig


def fleet_status_figure(snapshots: pd.DataFrame) -> go.Figure:
    c = snapshots.groupby(["time_s", "status"]).size().unstack(fill_value=0).reset_index()
    fig = go.Figure()
    for st, col in STATUS_COLORS.items():
        if st in c:
            fig.add_scatter(x=_h(c["time_s"]), y=c[st], name=st, stackgroup="one", line=dict(color=col, width=0.5))
    fig.update_layout(**LAYOUT, title="Fleet status over the day", xaxis_title="Time of day (h)",
                      yaxis_title="Trains", hovermode="x unified", legend=dict(orientation="h", y=1.08))
    return fig


def train_movement_animation(snapshots: pd.DataFrame) -> go.Figure:
    """Animated board: each train is a dot in its zone (depot / transit / mainline), coloured by status."""
    zones = {"DEPOT": 0, "TRANSIT": 1, "MAINLINE": 2}
    s = snapshots.copy()
    trains = sorted(s["train_id"].unique())
    s["x"] = s["train_id"].map({t: i for i, t in enumerate(trains)})
    s["y"] = s["location"].map(zones)
    s["clock"] = s["time_s"].map(seconds_to_clock)
    fig = px.scatter(s, x="x", y="y", color="status", animation_frame="clock", hover_name="train_id",
                     color_discrete_map=STATUS_COLORS, category_orders={"status": list(STATUS_COLORS)},
                     range_x=[-1, len(trains)], range_y=[-0.6, 2.6])
    fig.update_traces(marker=dict(size=16, line=dict(width=2, color="white")))
    fig.update_layout(**LAYOUT, title="Train movements (15-min snapshots – press ▶)", height=420,
                      xaxis=dict(tickmode="array", tickvals=list(range(len(trains))), ticktext=trains, title="Train"),
                      yaxis=dict(tickmode="array", tickvals=[0, 1, 2], ticktext=["Depot", "Induction route", "Mainline"],
                                 title=""))
    return fig


def gantt_figure(occupancy: pd.DataFrame, inductions: pd.DataFrame, title: str = "Induction timeline") -> go.Figure:
    occ = occupancy[occupancy["purpose"] == "induction"].copy()
    if occ.empty:
        return go.Figure().update_layout(**LAYOUT, title=f"{title} – no inductions")
    occ["kind"] = occ["kind"].where(occ["kind"].isin(SECTION_COLORS), "shared")
    occ["dur_h"] = (occ["end_s"] - occ["start_s"]) / 3600
    occ["start_h"] = occ["start_s"] / 3600
    order = inductions.sort_values("authorised_time_s")["train_id"].drop_duplicates().tolist()
    fig = go.Figure()
    for kind, sub in occ.groupby("kind"):
        fig.add_bar(y=sub["train_id"], x=sub["dur_h"], base=sub["start_h"], orientation="h", name=kind,
                    marker_color=SECTION_COLORS.get(kind, "#999"),
                    customdata=np.stack([sub["section_id"], sub["start_s"].map(seconds_to_clock),
                                         sub["end_s"].map(seconds_to_clock)], axis=1),
                    hovertemplate="%{y} on %{customdata[0]}<br>%{customdata[1]} – %{customdata[2]}<extra></extra>")
    fig.update_layout(**LAYOUT, title=title, barmode="overlay", xaxis_title="Time of day (h)",
                      yaxis=dict(categoryorder="array", categoryarray=order[::-1], title="Train"),
                      height=max(350, 18 * len(order) + 120), legend=dict(orientation="h", y=1.05))
    return fig


def occupancy_heatmap(occupancy: pd.DataFrame, start: int = 4 * 3600, end: int = 23 * 3600, step: int = 300) -> go.Figure:
    from .gantt_chart import occupancy_series

    sections = sorted(occupancy["section_id"].unique()) if not occupancy.empty else []
    if not sections:
        return go.Figure().update_layout(**LAYOUT, title="Track occupancy – no data")
    s = occupancy_series(occupancy, sections, start, end, step)
    fig = go.Figure(go.Heatmap(z=s.T.to_numpy(), x=_h(s.index), y=sections, colorscale="Blues", zmin=0, zmax=1,
                               colorbar=dict(title="occupied"),
                               hovertemplate="%{y} at %{x:.2f} h<br>occupied %{z:.0%}<extra></extra>"))
    fig.update_layout(**LAYOUT, title="Track occupancy over time (share of each 5-min window)",
                      xaxis_title="Time of day (h)", height=360)
    return fig


def metric_box(df: pd.DataFrame, metric: str, label: str) -> go.Figure:
    order = [n for n in ["FCFS", "Heuristic", "PPO"] if n in df["scheduler"].unique()]
    fig = px.box(df, x="scheduler", y=metric, color="scheduler", points="all", category_orders={"scheduler": order},
                 color_discrete_map=SCHEDULER_COLORS, hover_data=["scenario_type", "seed"])
    fig.update_layout(**LAYOUT, title=f"{label} – distribution over evaluation episodes", showlegend=False,
                      yaxis_title=label, xaxis_title="")
    return fig


def scenario_bars(df: pd.DataFrame, metric: str, label: str) -> go.Figure:
    g = df.groupby(["scenario_type", "scheduler"])[metric].agg(["mean", "std"]).reset_index()
    order = [n for n in ["FCFS", "Heuristic", "PPO"] if n in g["scheduler"].unique()]
    fig = px.bar(g, x="scenario_type", y="mean", error_y="std", color="scheduler", barmode="group",
                 category_orders={"scheduler": order, "scenario_type": ["normal", "peak_demand", "disruption"]},
                 color_discrete_map=SCHEDULER_COLORS, text_auto=".2f")
    fig.update_layout(**LAYOUT, title=f"{label} by scenario (mean ± std)", yaxis_title=label, xaxis_title="")
    return fig
