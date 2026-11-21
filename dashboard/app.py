"""MetroSched – interactive Streamlit dashboard.

Run from the project root:   streamlit run dashboard/app.py

Every button triggers a real operation (simulation run, scheduler run,
dataset generation, evaluation, config validation / save). Nothing shown here
is precomputed or hardcoded: figures come from simulation logs or from the
CSV results written by the evaluation pipeline.

ALL DATA IS SYNTHETIC – NOT KMRL OPERATIONAL DATA.
"""

from __future__ import annotations

import copy
import io
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
import yaml  # noqa: E402

from evaluation.comparison import evaluation_scenarios, run_comparison  # noqa: E402
from evaluation.metrics import METRICS, episode_metrics  # noqa: E402
from evaluation.statistical_analysis import comparison_matrix, summary_table  # noqa: E402
from scheduling.base import load_rl_config, run_episode  # noqa: E402
from scheduling.baseline_fcfs import FCFSScheduler  # noqa: E402
from scheduling.heuristic_scheduler import HeuristicScheduler  # noqa: E402
from simulation.config import DEFAULT_CONFIG_PATH, SCENARIO_TYPES, load_config, seconds_to_clock  # noqa: E402
from simulation.data_generator import SYNTHETIC_NOTICE, generate_dataset  # noqa: E402
from simulation.exceptions import ConfigError  # noqa: E402
from simulation.scenario_generator import ScenarioGenerator  # noqa: E402
from visualization import simulation_visualizer as viz  # noqa: E402

logging.basicConfig(level=logging.WARNING)
log = logging.getLogger("dashboard")

GEN_DIR = ROOT / "data" / "generated"
EVAL_DIR = ROOT / "results" / "evaluation"
MODEL = ROOT / "models" / "ppo_metrosched.zip"
USER_CFG = ROOT / "data" / "config" / "user_config.yaml"
PAGES = ["1 · Operations overview", "2 · Metro simulation", "3 · AI scheduling agent",
         "4 · Performance analytics", "5 · System configuration", "6 · Agentic AI assistant"]
EXAMPLE_QUESTIONS = [
    "Festival tomorrow: demand +25% and platform 2 closed 08:00-09:30. Which scheduler should we use and what "
    "are the first inductions?",
    "Exit track 1 is closed from 17:00 to 18:30 on a normal weekday. How much does that hurt the evening peak?",
    "We only have 21 trainsets today on a peak demand day. Is the PPO agent still better than FCFS?",
    "Across all tested days, how does the reinforcement-learning agent compare with the heuristic?",
]
AGENT_ARCH_DOT = """digraph G { rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#eef3fb", fontsize=11];
 op [label="Operator\\n(natural-language question)", fillcolor="#fff4e0"];
 llm [label="LLM agent – Claude\\nplan · call tools · reflect · answer", fillcolor="#e3f0e3"];
 tools [label="Toolbox\\nwhat-if scenario · run / compare\\ndemand · fleet · induction plan"];
 rl [label="Schedulers\\nPPO (RL agent) · Heuristic · FCFS"];
 sim [label="SimPy simulator\\n+ hard safety rules + audit"];
 op -> llm [label="question"]; llm -> tools [label="tool call"]; tools -> rl; rl -> sim [label="inductions"];
 sim -> tools [label="KPIs, logs", style=dashed]; tools -> llm [label="observation", style=dashed];
 llm -> op [label="recommendation", style=dashed]; }"""

st.set_page_config(page_title="MetroSched", page_icon="🚇", layout="wide")


# ------------------------------------------------------------------ state
def _init_state() -> None:
    ss = st.session_state
    ss.setdefault("overrides", {})
    ss.setdefault("sim_episode", None)
    ss.setdefault("agent_episodes", {})
    ss.setdefault("eval_live", None)


def raw_config() -> dict:
    with DEFAULT_CONFIG_PATH.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def active_config(extra: dict | None = None) -> dict:
    """Default YAML + dashboard overrides (+ page-local overrides), validated."""
    from simulation.config import deep_merge
    ov = deep_merge(st.session_state.get("overrides", {}), extra or {})
    return load_config(overrides=ov)


def exit_tracks_override(n: int, headway: int) -> list[dict]:
    base = raw_config()["depot"]["exit_tracks"]
    out = []
    for i in range(n):
        src = copy.deepcopy(base[i % len(base)])
        src["id"] = f"EXIT-{i + 1}"
        src["headway_s"] = int(headway)
        out.append(src)
    return out


@st.cache_resource(show_spinner=False)
def load_ppo(path: str, mtime: float):
    from scheduling.rl_scheduler import RLScheduler
    return RLScheduler(path)


def make_scheduler(name: str, cfg: dict):
    rl_cfg = load_rl_config()
    if name == "FCFS":
        return FCFSScheduler()
    if name == "Heuristic":
        return HeuristicScheduler.from_config(rl_cfg)
    if not MODEL.exists():
        raise FileNotFoundError("No trained PPO model found (models/ppo_metrosched.zip). Run: python main.py train")
    from scheduling.rl_scheduler import RLScheduler
    base = load_ppo(str(MODEL), MODEL.stat().st_mtime)
    sched = RLScheduler.__new__(RLScheduler)
    sched.__dict__.update(base.__dict__)
    from reinforcement_learning.gym_environment import ObservationEncoder
    sched.cfg = cfg
    sched.encoder = ObservationEncoder(cfg, rl_cfg["gym"]["candidate_slots"])
    if sched.encoder.obs_dim != base.model.observation_space.shape[0]:
        raise ValueError("The PPO model was trained on a different infrastructure layout (number of track "
                         "sections). Use the default number of exit tracks or retrain the agent.")
    return sched


def scheduler_options() -> list[str]:
    return ["FCFS", "Heuristic", "PPO"] if MODEL.exists() else ["FCFS", "Heuristic"]


def kpi_row(items: list[tuple[str, str]]) -> None:
    cols = st.columns(len(items))
    for c, (label, value) in zip(cols, items):
        c.metric(label, value)


# ================================================================ pages
def page_overview() -> None:
    st.header("Metro operations overview")
    st.caption("From the generated synthetic dataset in data/generated/ (regenerate below).")
    if not (GEN_DIR / "scenarios.csv").exists():
        st.warning("No dataset found. Generate one below.")
    else:
        scen = pd.read_csv(GEN_DIR / "scenarios.csv")
        fleet = pd.read_csv(GEN_DIR / "fleet_status_by_interval.csv")
        demand = pd.read_csv(GEN_DIR / "passenger_demand.csv")
        c1, c2 = st.columns([1, 2])
        day = c1.selectbox("Simulated day", scen["day"].tolist(),
                           format_func=lambda d: f"Day {d:02d} – {scen.loc[scen.day == d, 'scenario_type'].iloc[0]}")
        f = fleet[fleet["day"] == day]
        times = f["time_s"].tolist()
        t = c2.select_slider("Time of day", options=times, value=times[min(14, len(times) - 1)],
                             format_func=seconds_to_clock)
        row = f[f["time_s"] == t].iloc[0]
        d = demand[demand["day"] == day]
        total = int(row[[c for c in row.index if c.startswith("n_")]].sum())
        kpi_row([("Total trains", str(total)), ("Available", str(int(row["n_available"]))),
                 ("Inducted / running", f"{int(row['n_inducted'])} / {int(row['n_running'])}"),
                 ("Under maintenance", str(int(row["n_maintenance"]))),
                 ("Scenario", scen.loc[scen.day == day, "scenario_type"].iloc[0].replace("_", " ")),
                 ("Daily demand (pax-h·dir)", f"{int(d['demand_pphpd'].sum() / 4):,}")])
        st.plotly_chart(viz.demand_figure(d, f"Day {day}: passenger demand and required trains"),
                        width="stretch")
        avail = pd.read_csv(GEN_DIR / "train_availability.csv")
        st.plotly_chart(viz.fleet_status_figure(avail[avail["day"] == day]), width="stretch")
    with st.expander("Regenerate synthetic dataset"):
        c1, c2 = st.columns(2)
        seed = c1.number_input("Seed", 0, 10 ** 9, 42, key="gen_seed")
        days = c2.number_input("Days", 30, 365, 30, key="gen_days")
        if st.button("Generate dataset", type="primary"):
            with st.spinner("Simulating days…"):
                data = generate_dataset(active_config(), int(seed), int(days), GEN_DIR)
            st.success(f"Generated {days} days, safety violations: {data['manifest']['safety_violations']}")
            st.rerun()


def _sim_controls(prefix: str) -> tuple[dict, str, int, str]:
    c1, c2, c3, c4 = st.columns(4)
    stype = c1.selectbox("Operational scenario", SCENARIO_TYPES, key=f"{prefix}_type")
    seed = int(c2.number_input("Scenario seed", 0, 10 ** 9, 2026, key=f"{prefix}_seed"))
    base_cfg = active_config()
    fleet = c3.slider("Fleet size", 20, 30, int(base_cfg["fleet"]["size"]), key=f"{prefix}_fleet")
    sched = c4.selectbox("Scheduler", scheduler_options(), index=len(scheduler_options()) - 1, key=f"{prefix}_sched")
    c5, c6, c7 = st.columns(3)
    exits = c5.slider("Depot exit tracks (track capacity)", 2, 3, len(base_cfg["depot"]["exit_tracks"]),
                      key=f"{prefix}_exits")
    hw = c6.slider("Safety headway on depot/shared track (s)", 60, 240,
                   int(base_cfg["track"]["shared_sections"][0]["headway_s"]), step=10, key=f"{prefix}_hw")
    dm = c7.slider("Passenger demand (× default)", 0.6, 1.6, 1.0, step=0.05, key=f"{prefix}_dm")
    raw = raw_config()
    extra = {"fleet": {"size": int(fleet)},
             "depot": {"exit_tracks": exit_tracks_override(exits, hw)},
             "track": {"shared_sections": [dict(s, headway_s=int(hw)) for s in raw["track"]["shared_sections"]]},
             "demand": {"base_pphpd": float(active_config()["demand"]["base_pphpd"]) * dm}}
    return extra, stype, seed, sched


def run_selected(extra: dict, stype: str, seed: int, sched_name: str):
    cfg = active_config(extra)
    scheduler = make_scheduler(sched_name, cfg)
    scenario = ScenarioGenerator(cfg, seed).single(stype)
    ep = run_episode(cfg, scenario, scheduler, int(load_rl_config()["scheduling"]["max_inductions_per_tick"]))
    return ep, episode_metrics(ep, cfg), cfg


def page_simulation() -> None:
    st.header("Metro simulation")
    extra, stype, seed, sched = _sim_controls("sim")
    b1, b2, _ = st.columns([1, 1, 4])
    if b1.button("▶ Start simulation", type="primary"):
        try:
            with st.spinner("Running SimPy simulation of the full operating day…"):
                st.session_state.sim_episode = run_selected(extra, stype, seed, sched)
        except (ConfigError, ValueError, FileNotFoundError) as exc:
            st.error(str(exc))
    if b2.button("⟲ Reset simulation"):
        st.session_state.sim_episode = None
        st.rerun()
    if st.session_state.sim_episode is None:
        st.info("Choose parameters and press **Start simulation**.")
        return
    ep, m, cfg = st.session_state.sim_episode
    st.subheader(f"{ep.scheduler} on a {ep.scenario.scenario_type.replace('_', ' ')} day (seed {seed})")
    kpi_row([("Inductions", str(m["successful_inductions"])), ("Demand coverage", f"{m['demand_coverage_pct']:.1f}%"),
             ("Avg induction delay", f"{m['avg_induction_delay_min']:.1f} min"),
             ("Avg waiting", f"{m['avg_waiting_time_min']:.1f} min"),
             ("Missed opportunities", str(m["missed_opportunities"])),
             ("Safety violations", str(m["safety_violations"]))])
    st.plotly_chart(viz.train_movement_animation(ep.result.snapshots), width="stretch")
    st.plotly_chart(viz.service_figure(ep.result.monitor), width="stretch")
    c1, c2 = st.columns(2)
    c1.plotly_chart(viz.gantt_figure(ep.result.occupancy, ep.result.inductions, "Induction timeline"),
                    width="stretch")
    c2.plotly_chart(viz.occupancy_heatmap(ep.result.occupancy), width="stretch")
    st.subheader("Induction decisions")
    st.dataframe(pd.DataFrame(ep.explanations), width="stretch", hide_index=True)


def page_agent() -> None:
    st.header("AI scheduling agent")
    if not MODEL.exists():
        st.warning("No trained PPO model yet – only FCFS and the heuristic are available. Run `python main.py train`.")
    extra, stype, seed, sched = _sim_controls("agent")
    if st.button("Run scheduling optimization", type="primary"):
        try:
            with st.spinner(f"Scheduling the day with {sched} (and FCFS for reference)…"):
                eps = {sched: run_selected(extra, stype, seed, sched)}
                if sched != "FCFS":
                    eps["FCFS"] = run_selected(extra, stype, seed, "FCFS")
                st.session_state.agent_episodes = eps
        except (ConfigError, ValueError, FileNotFoundError) as exc:
            st.error(str(exc))
    eps = st.session_state.agent_episodes
    if not eps:
        st.info("Press **Run scheduling optimization** to compute an induction plan.")
        _constraints(active_config(extra))
        return
    name = next(iter(eps))
    ep, m, cfg = eps[name]
    rows = []
    for k in METRICS:
        if k in m:
            rows.append({"Metric": METRICS[k][0], name: round(m[k], 2),
                         **({"FCFS (same day)": round(eps["FCFS"][1][k], 2)} if "FCFS" in eps and name != "FCFS" else {})})
    c1, c2 = st.columns([3, 2])
    with c1:
        st.subheader("Recommended induction sequence")
        ind = ep.result.inductions
        if ind.empty:
            st.write("No inductions were made.")
        else:
            seq = pd.DataFrame({
                "#": range(1, len(ind) + 1), "Train": ind["train_id"],
                "Induction start": ind["authorised_time_s"].map(seconds_to_clock),
                "Depot exit": ind["depot_exit_time_s"].map(seconds_to_clock),
                "In service": ind["service_start_time_s"].map(seconds_to_clock),
                "Exit track": ind["exit_track"], "Entry point": ind["entry_point"], "Platform": ind["platform"],
                "Track wait (s)": ind["track_wait_s"]})
            st.dataframe(seq, width="stretch", hide_index=True)
    with c2:
        st.subheader("Outcome on this day")
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.subheader("Why did the scheduler pick this train?")
    acc = [e for e in ep.explanations if e["accepted"]]
    if acc:
        labels = [f"{e['time']} – {e['train_id']} via {e['entry_point']}" for e in acc]
        i = st.selectbox("Decision", range(len(acc)), format_func=lambda j: labels[j])
        st.info(acc[i]["explanation"])
    _constraints(cfg)


def _constraints(cfg: dict) -> None:
    with st.expander("Scheduling constraints enforced by the environment (hard rules)", expanded=False):
        d, t, p = cfg["depot"], cfg["track"], cfg["platforms"]
        rows = [{"Constraint": f"Exit track {e['id']}", "Value": f"1 train, headway {e['headway_s']} s"} for e in d["exit_tracks"]]
        rows += [{"Constraint": f"Shared section {s['id']}", "Value": f"1 train, headway {s['headway_s']} s"} for s in t["shared_sections"]]
        rows += [{"Constraint": f"Entry point {e['id']}", "Value": f"1 train, headway {e['headway_s']} s"} for e in t["entry_points"]]
        rows += [{"Constraint": "Platforms", "Value": f"{p['count']} (clearance {p['clearance_s']} s)"},
                 {"Constraint": "Induction cap", "Value": f"{d['max_inductions_per_hour']} per rolling hour"},
                 {"Constraint": "Readiness", "Value": "prep complete, no maintenance pending, in depot"}]
        rows += [{"Constraint": f"Window {w['id']}", "Value": f"{seconds_to_clock(w['start'])}–{seconds_to_clock(w['end'])}, "
                  f"max {w['max_inductions']}"} for w in d["induction_windows"]]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def page_analytics() -> None:
    st.header("Performance analytics")
    src = st.radio("Results source", ["Saved evaluation (results/evaluation)", "Quick live evaluation"],
                   horizontal=True, key="analytics_src")
    df = None
    if src.startswith("Saved"):
        f = EVAL_DIR / "episode_metrics.csv"
        if f.exists():
            df = pd.read_csv(f)
        else:
            st.warning("No saved evaluation found – run `python main.py evaluate` or use a quick live evaluation.")
    else:
        c1, c2 = st.columns(2)
        per = c1.slider("Episodes per scenario type", 1, 10, 3)
        base = int(c2.number_input("Seed base (unseen)", 0, 2 ** 31 - 1, 910_000_000))
        if st.button("Run live evaluation", type="primary"):
            cfg = active_config()
            scheds = [make_scheduler(n, cfg) for n in scheduler_options()]
            bar = st.progress(0.0)
            df_live, _ = run_comparison(cfg, scheds, evaluation_scenarios(cfg, base, per),
                                        progress=lambda d, t: bar.progress(d / t))
            st.session_state.eval_live = df_live
        df = st.session_state.eval_live
    if df is None or df.empty:
        return
    summary = summary_table(df, "FCFS")
    table = comparison_matrix(summary, "FCFS")
    n = int(df.groupby("scheduler").size().min())
    st.caption(f"{n} episodes per scheduler, identical seeds. Improvement = (FCFS − X)/FCFS for lower-is-better metrics, "
               "(X − FCFS)/FCFS for higher-is-better ones.")
    st.dataframe(table, width="stretch", hide_index=True)
    others = [s for s in summary["scheduler"].unique() if s != "FCFS"]
    if others:
        tgt = st.selectbox("Improvement of", others, index=len(others) - 1)
        s = summary[(summary["scheduler"] == tgt) & (summary["direction"] != "neutral")]
        cols = st.columns(4)
        for i, r in enumerate(s.head(8).itertuples()):
            cols[i % 4].metric(r.label, f"{r.mean:.2f} {r.unit}",
                               None if pd.isna(r.improvement_vs_baseline_pct) else f"{r.improvement_vs_baseline_pct:+.1f}% vs FCFS",
                               delta_color="normal")
    metric = st.selectbox("Metric", [m for m in METRICS if m in df.columns], format_func=lambda k: METRICS[k][0])
    c1, c2 = st.columns(2)
    c1.plotly_chart(viz.metric_box(df, metric, METRICS[metric][0]), width="stretch")
    c2.plotly_chart(viz.scenario_bars(df, metric, METRICS[metric][0]), width="stretch")
    st.subheader("Export")
    c1, c2, c3 = st.columns(3)
    c1.download_button("Episode metrics (CSV)", df.to_csv(index=False), "episode_metrics.csv", "text/csv")
    c2.download_button("Summary statistics (CSV)", summary.to_csv(index=False), "summary.csv", "text/csv")
    c3.download_button("Comparison table (CSV)", table.to_csv(index=False), "comparison_table.csv", "text/csv")
    plots = ROOT / "results" / "plots"
    if plots.exists() and any(plots.glob("*.png")):
        with st.expander("Saved report plots (results/plots)"):
            for p in sorted(plots.glob("*.png")):
                st.image(str(p), caption=p.name)


def page_config() -> None:
    st.header("System configuration")
    st.caption("Changes apply to pages 2–4 in this session after **Apply**. **Save** writes data/config/user_config.yaml.")
    cfg = active_config()
    raw = raw_config()
    with st.form("cfg"):
        st.subheader("Fleet")
        c1, c2, c3 = st.columns(3)
        size = c1.number_input("Fleet size", 20, 30, int(cfg["fleet"]["size"]))
        reserve = c2.number_input("Reserve trains", 0, 10, int(cfg["service"]["reserve_trains"]))
        cap = c3.number_input("Train capacity (passengers)", 300, 3000, int(cfg["fleet"]["train_capacity"]))
        st.subheader("Track capacity & headways")
        c1, c2, c3, c4 = st.columns(4)
        exits = c1.number_input("Depot exit tracks", 2, 3, len(cfg["depot"]["exit_tracks"]))
        hw_exit = c2.number_input("Exit-track headway (s)", 30, 600, int(raw["depot"]["exit_tracks"][0]["headway_s"]))
        hw_entry = c3.number_input("Mainline entry headway (s)", 30, 600, int(raw["track"]["entry_points"][0]["headway_s"]))
        per_hour = c4.number_input("Max inductions per hour", 1, 30, int(cfg["depot"]["max_inductions_per_hour"]))
        c1, c2 = st.columns(2)
        platforms = c1.number_input("Platforms", 2, 4, int(cfg["platforms"]["count"]))
        min_hw = c2.number_input("Minimum service headway (s)", 120, 900, int(cfg["service"]["min_service_headway_s"]))
        st.subheader("Maintenance probabilities")
        c1, c2, c3 = st.columns(3)
        fail = c1.number_input("In-service failure prob. per round trip", 0.0, 0.2,
                               float(cfg["operations"]["in_service_failure_prob"]), format="%.4f")
        fault_n = c2.number_input("Depot fault prob. (normal day)", 0.0, 1.0,
                                  float(cfg["scenarios"]["normal"]["depot_fault_prob"]), format="%.3f")
        fault_d = c3.number_input("Depot fault prob. (disruption day)", 0.0, 1.0,
                                  float(cfg["scenarios"]["disruption"]["depot_fault_prob"]), format="%.3f")
        st.subheader("Passenger demand")
        c1, c2, c3, c4 = st.columns(4)
        base = c1.number_input("Base demand (pphpd)", 500, 20000, int(cfg["demand"]["base_pphpd"]))
        peaks = raw["demand"]["peaks"]
        amps = [c.number_input(f"{p['name'].title()} peak amplitude", 0.0, 5.0, float(p["amplitude"]))
                for c, p in zip((c2, c3, c4), peaks)]
        applied = st.form_submit_button("Apply", type="primary")
    ov = {"fleet": {"size": int(size), "train_capacity": int(cap)},
          "service": {"reserve_trains": int(reserve), "min_service_headway_s": int(min_hw)},
          "depot": {"exit_tracks": exit_tracks_override(int(exits), int(hw_exit)), "max_inductions_per_hour": int(per_hour)},
          "track": {"entry_points": [dict(e, headway_s=int(hw_entry)) for e in raw["track"]["entry_points"]]},
          "platforms": {"count": int(platforms)},
          "operations": {"in_service_failure_prob": float(fail)},
          "scenarios": {"normal": {"depot_fault_prob": float(fault_n)}, "disruption": {"depot_fault_prob": float(fault_d)}},
          "demand": {"base_pphpd": float(base), "peaks": [dict(p, amplitude=float(a)) for p, a in zip(peaks, amps)]}}
    if applied:
        try:
            load_config(overrides=ov)
            st.session_state.overrides = ov
            st.success("Configuration validated and applied to this session.")
        except ConfigError as exc:
            st.error(f"Invalid configuration: {exc}")
    c1, c2 = st.columns(2)
    if c1.button("Save applied configuration to data/config/user_config.yaml"):
        from simulation.config import deep_merge
        merged = deep_merge(raw, st.session_state.overrides)
        USER_CFG.write_text("# Saved from the MetroSched dashboard (synthetic parameters)\n"
                            + yaml.safe_dump(merged, sort_keys=False), encoding="utf-8")
        st.success(f"Saved to {USER_CFG.relative_to(ROOT)} – use with: python main.py generate-data --config {USER_CFG.relative_to(ROOT)}")
    if c2.button("Reset to defaults"):
        st.session_state.overrides = {}
        st.success("Defaults restored.")
        st.rerun()
    if st.session_state.overrides:
        buf = io.StringIO()
        yaml.safe_dump(st.session_state.overrides, buf, sort_keys=False)
        st.code(buf.getvalue(), language="yaml")


def _render_run(run) -> None:
    st.markdown(run.answer)
    meta = f"{run.tool_calls} tool calls · {run.model}"
    if run.mode == "llm":
        meta += f" · {run.input_tokens:,} input / {run.output_tokens:,} output tokens"
    with st.expander(f"Agent trace – reason → act → observe ({meta})"):
        st.dataframe(pd.DataFrame(run.trace_table()), width="stretch", hide_index=True)


def page_assistant() -> None:
    import os

    from agentic.agent import DEFAULT_MODEL, MetroOpsAgent, OfflineAgent

    st.header("Agentic AI assistant")
    st.caption("An LLM agent (Claude) that understands the operator's question, plans what to test, runs the "
               "simulator and the PPO / heuristic / FCFS schedulers through tools, checks the results and writes "
               "a recommendation. The reinforcement-learning agent is one of its tools.")
    with st.expander("How the agent works", expanded=False):
        st.graphviz_chart(AGENT_ARCH_DOT)
    ss = st.session_state
    ss.setdefault("assistant_chat", [])
    has_key = bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))
    c1, c2, c3 = st.columns([2, 2, 1])
    modes = ["LLM agent (Claude API)", "Offline planner (no API key, no LLM)"]
    mode = c1.radio("Agent mode", modes, index=0 if has_key else 1, key="assistant_mode")
    model = c2.selectbox("Claude model", [DEFAULT_MODEL, "claude-sonnet-5-5", "claude-haiku-5-5"],
                         key="assistant_model", disabled=mode != modes[0])
    if c3.button("New conversation"):
        ss.assistant_chat = []
        ss.pop("assistant_agent", None)
        st.rerun()
    if mode == modes[0] and not has_key:
        st.warning("No ANTHROPIC_API_KEY found in the environment. Set it (or run `ant auth login`) and restart, "
                   "or use the offline planner.")
    sig = (mode, model)
    if ss.get("assistant_sig") != sig:
        ss.assistant_sig = sig
        ss.pop("assistant_agent", None)

    for role, content, run in ss.assistant_chat:
        with st.chat_message(role):
            _render_run(run) if run is not None else st.markdown(content)

    st.caption("Try an example:")
    cols = st.columns(len(EXAMPLE_QUESTIONS))
    picked = None
    for i, (c, q) in enumerate(zip(cols, EXAMPLE_QUESTIONS)):
        if c.button(q[:60] + "…", key=f"ex_{i}", help=q):
            picked = q
    question = st.chat_input("Ask about today's induction plan, a closure, a demand surge…") or picked
    if not question:
        return
    ss.assistant_chat.append(("user", question, None))
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            if "assistant_agent" not in ss:
                ss.assistant_agent = MetroOpsAgent(model=model) if mode == modes[0] else OfflineAgent()
            agent = ss.assistant_agent
            with st.status("Agent is working…", expanded=True) as status:
                def on_step(step):
                    if step.kind == "tool_call":
                        args = ", ".join(f"{k}={v!r}" for k, v in (step.tool_input or {}).items())
                        status.write(f"🔧 **{step.tool}**({args})")
                    elif step.kind == "tool_result" and step.tool_output and "error" in step.tool_output:
                        status.write(f"⚠️ {step.tool_output['error']}")
                    elif step.kind == "thought":
                        status.write(f"💭 {step.content[:300]}")
                run = agent.ask(question, on_step=on_step)
                status.update(label=f"Done – {run.tool_calls} tool calls", state="complete", expanded=False)
            _render_run(run)
            ss.assistant_chat.append(("assistant", run.answer, run))
        except Exception as exc:  # noqa: BLE001 - credentials, network, rate limit …
            st.error(f"Agent failed: {exc}. Check your API key / network, or switch to the offline planner.")
            ss.pop("assistant_agent", None)


# ================================================================== main
def main() -> None:
    _init_state()
    with st.sidebar:
        st.title("🚇 MetroSched")
        st.caption("Intelligent train induction scheduling")
        page = st.radio("Page", PAGES, label_visibility="collapsed")
        st.divider()
        st.caption("⚠️ " + SYNTHETIC_NOTICE)
        if st.session_state.overrides:
            st.caption("Custom configuration active (page 5).")
    {PAGES[0]: page_overview, PAGES[1]: page_simulation, PAGES[2]: page_agent,
     PAGES[3]: page_analytics, PAGES[4]: page_config, PAGES[5]: page_assistant}[page]()


main()
