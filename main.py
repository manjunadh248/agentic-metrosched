"""MetroSched command-line entry point.

    python main.py generate-data --seed 42      # synthetic dataset (CSV)
    python main.py run-baseline --episodes 10   # FCFS (+ heuristic) on unseen seeds
    python main.py train                        # BC warm start + PPO training
    python main.py evaluate                     # FCFS vs heuristic vs PPO, 120 unseen episodes
    python main.py plot                         # all comparison graphs -> results/plots
    python main.py dashboard                    # launch the Streamlit dashboard
    python main.py agent "question"             # agentic AI assistant (Claude LLM agent, tools)
    python main.py agent --offline "question"   # same tools, rule-based planner, no API key
    python main.py all                          # everything above, in order

All generated data is SYNTHETIC, not actual KMRL operational data.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from simulation.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, load_config
from simulation.data_generator import SYNTHETIC_NOTICE, generate_dataset
from simulation.exceptions import MetroSchedError

logger = logging.getLogger("metrosched")


def setup_logging(level: str, log_file: Path | None = None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode="w", encoding="utf-8"))
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                        datefmt="%H:%M:%S", handlers=handlers, force=True)


def cmd_generate_data(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    out = Path(args.output_dir)
    print(SYNTHETIC_NOTICE)
    t0 = time.perf_counter()
    data = generate_dataset(cfg, seed=args.seed, n_days=args.days, output_dir=out,
                            start_weekday=args.start_weekday, config_path=args.config)
    elapsed = time.perf_counter() - t0
    m = data["manifest"]
    print(f"\nGenerated {m['n_days']} days (seed={m['seed']}, fleet={m['fleet_size']}) "
          f"in {elapsed:.1f}s -> {out}")
    print(f"Scenario mix: {m['scenario_counts']}")
    print(f"Safety audit violations: {m['safety_violations']}")
    for name, rows in m["files"].items():
        print(f"  {name:<30} {rows:>8} rows")
    return 0


def cmd_run_baseline(args: argparse.Namespace) -> int:
    import pandas as pd

    from evaluation.comparison import evaluation_scenarios, run_comparison
    from evaluation.statistical_analysis import comparison_matrix, summary_table
    from scheduling.base import load_rl_config
    from scheduling.baseline_fcfs import FCFSScheduler
    from scheduling.heuristic_scheduler import HeuristicScheduler

    cfg = load_config(args.config)
    rl_cfg = load_rl_config()
    scheds = [FCFSScheduler()] + ([] if args.fcfs_only else [HeuristicScheduler.from_config(rl_cfg)])
    scen = evaluation_scenarios(cfg, args.seed_base, args.episodes)
    df, examples = run_comparison(cfg, scheds, scen)
    out = PROJECT_ROOT / "results" / "baseline"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "baseline_episode_metrics.csv", index=False)
    ep = examples[("FCFS", scen[0].scenario_type)]
    ep.result.inductions.to_csv(out / "fcfs_example_induction_schedule.csv", index=False)
    table = comparison_matrix(summary_table(df, "FCFS"), "FCFS")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(table.to_string(index=False))
    print(f"\nSafety violations: {int(df['safety_violations'].sum())}  -> results saved in {out}")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from reinforcement_learning.train_agent import train

    path = train(args.timesteps, args.n_envs, args.seed, subproc=args.subproc)
    print(f"Trained model saved to {path}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    import pandas as pd

    from reinforcement_learning.evaluate_agent import evaluate

    res = evaluate(args.episodes_per_scenario, args.model, not args.no_ppo)
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print(res["table"].to_string(index=False))
    print(f"\n{res['meta']['episodes']} unseen episodes, safety violations: {res['meta']['total_safety_violations']}")
    return 0


def cmd_plot(args: argparse.Namespace) -> int:
    from visualization.performance_plots import generate_all

    for p in generate_all():
        print(f"  saved {p.relative_to(PROJECT_ROOT)}")
    return 0


def cmd_dashboard(args: argparse.Namespace) -> int:
    import subprocess

    app = PROJECT_ROOT / "dashboard" / "app.py"
    return subprocess.call([sys.executable, "-m", "streamlit", "run", str(app), "--server.port", str(args.port)])


def cmd_agent(args: argparse.Namespace) -> int:
    """Agentic AI assistant: an LLM agent (Claude) that plans and runs simulations through tools."""
    from agentic.agent import AgentStep, make_agent

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    agent = make_agent(offline=args.offline, model=args.model)
    mode = "offline rule-based planner (no LLM)" if args.offline else f"LLM agent ({args.model})"
    print(f"MetroOps agent – {mode}. Synthetic data, not KMRL data.")

    def show(step: AgentStep) -> None:
        if step.kind == "tool_call":
            print(f"  → {step.tool}({', '.join(f'{k}={v!r}' for k, v in (step.tool_input or {}).items())})")
        elif step.kind == "tool_result" and step.tool_output and "error" in step.tool_output:
            print(f"    ! {step.tool_output['error']}")
        elif step.kind == "thought" and args.verbose:
            print(f"  · {step.content}")

    def answer(question: str) -> None:
        try:
            run = agent.ask(question, on_step=show)
        except Exception as exc:  # noqa: BLE001 - e.g. missing credentials / network
            logger.error("Agent failed: %s. Use --offline to run without the Claude API.", exc)
            return
        print("\n" + run.answer)
        if run.mode == "llm":
            print(f"\n[{run.tool_calls} tool calls, {run.input_tokens} input / {run.output_tokens} output tokens]")

    if args.question:
        answer(" ".join(args.question))
        return 0
    print("Ask a question (empty line or 'exit' to quit). Follow-up questions keep the context.")
    while True:
        try:
            q = input("\nyou> ").strip()
        except EOFError:
            break
        if not q or q.lower() in ("exit", "quit"):
            break
        answer(q)
    return 0


def cmd_all(args: argparse.Namespace) -> int:
    ns = argparse.Namespace
    steps = [
        ("generate-data", cmd_generate_data, ns(config=args.config, seed=args.seed, days=30, start_weekday=0,
                                                output_dir=PROJECT_ROOT / "data" / "generated")),
        ("run-baseline", cmd_run_baseline, ns(config=args.config, episodes=5, seed_base=800_000_000, fcfs_only=False)),
        ("train", cmd_train, ns(timesteps=args.timesteps, n_envs=None, seed=None, subproc=False)),
        ("evaluate", cmd_evaluate, ns(episodes_per_scenario=None, model=None, no_ppo=False)),
        ("plot", cmd_plot, ns()),
    ]
    for name, fn, a in steps:
        print(f"\n=== {name} ===")
        rc = fn(a)
        if rc:
            return rc
    print("\nDone. Launch the dashboard with: python main.py dashboard")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="metrosched", description="MetroSched – synthetic metro induction scheduling")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    sub = p.add_subparsers(dest="command", required=True)
    g = sub.add_parser("generate-data", help="Generate the synthetic operational dataset (CSV)")
    g.add_argument("--seed", type=int, default=42, help="master random seed (default 42)")
    g.add_argument("--days", type=int, default=30, help="number of simulated days (default 30)")
    g.add_argument("--start-weekday", type=int, default=0, choices=range(7), help="0=Monday … 6=Sunday")
    g.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="YAML config path")
    g.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "generated")
    g.set_defaults(func=cmd_generate_data)

    b = sub.add_parser("run-baseline", help="Run FCFS (and heuristic) on unseen seeds")
    b.add_argument("--episodes", type=int, default=10, help="episodes per scenario type")
    b.add_argument("--seed-base", type=int, default=800_000_000)
    b.add_argument("--fcfs-only", action="store_true")
    b.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    b.set_defaults(func=cmd_run_baseline)

    t = sub.add_parser("train", help="Train the PPO agent (settings in data/config/rl_config.yaml)")
    t.add_argument("--timesteps", type=int, default=None)
    t.add_argument("--n-envs", type=int, default=None)
    t.add_argument("--seed", type=int, default=None)
    t.add_argument("--subproc", action="store_true")
    t.set_defaults(func=cmd_train)

    e = sub.add_parser("evaluate", help="Compare FCFS, heuristic and PPO on unseen seeds")
    e.add_argument("--episodes-per-scenario", type=int, default=None)
    e.add_argument("--model", default=None)
    e.add_argument("--no-ppo", action="store_true")
    e.set_defaults(func=cmd_evaluate)

    sub.add_parser("plot", help="Generate all graphs into results/plots").set_defaults(func=cmd_plot)

    d = sub.add_parser("dashboard", help="Launch the Streamlit dashboard")
    d.add_argument("--port", type=int, default=8501)
    d.set_defaults(func=cmd_dashboard)

    ag = sub.add_parser("agent", help="Agentic AI assistant (Claude LLM agent using the simulator as tools)")
    ag.add_argument("question", nargs="*", help="question; omit for an interactive session")
    ag.add_argument("--offline", action="store_true", help="rule-based planner, no API key / LLM needed")
    ag.add_argument("--model", default="claude-opus-5-5", help="Claude model id")
    ag.add_argument("--verbose", action="store_true", help="also print the agent's intermediate reasoning")
    ag.set_defaults(func=cmd_agent)

    a = sub.add_parser("all", help="Run the whole pipeline end to end")
    a.add_argument("--seed", type=int, default=42)
    a.add_argument("--timesteps", type=int, default=None)
    a.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    a.set_defaults(func=cmd_all)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level, PROJECT_ROOT / "results" / "logs" / f"{args.command}.log")
    try:
        return args.func(args)
    except (MetroSchedError, FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 2
    except KeyboardInterrupt:
        logger.error("Interrupted")
        return 130


if __name__ == "__main__":
    sys.exit(main())
