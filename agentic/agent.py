"""MetroOps agent – the agentic / generative AI layer of MetroSched.

Architecture (hierarchical, three levels of autonomy):

    Operator question (natural language)
            │
            ▼
    ┌──────────────────────────────┐   LLM agent (Claude, generative AI)
    │  MetroOpsAgent  – planner    │   reasons, plans, calls tools, reflects
    └──────────────┬───────────────┘   on the results, writes the answer
                   │ tool calls
                   ▼
    ┌──────────────────────────────┐   agentic/tools.py
    │  Toolbox                     │   what-if scenarios, run / compare
    └──────────────┬───────────────┘   schedulers, demand, fleet, plans
                   │ runs
                   ▼
    ┌──────────────────────────────┐   PPO reinforcement-learning agent,
    │  Schedulers (PPO, heuristic, │   rule-based heuristic, FCFS baseline
    │  FCFS) + SimPy simulator     │   → safety rules enforced here, always
    └──────────────────────────────┘

The agent loop is the classic perceive → plan → act → observe → reflect cycle:
Claude reads the question, decides which tool to call, gets the tool result,
decides whether it needs more evidence (another scenario, another scheduler),
and stops when it can give a recommendation backed by simulation numbers.

Two modes:

* ``MetroOpsAgent`` – the real LLM agent (Anthropic Claude API, tool use).
  Needs credentials (``ANTHROPIC_API_KEY`` or an ``ant auth login`` profile).
* ``OfflineAgent`` – a rule-based planner with the same tools and the same
  trace format, for demos without network / API key. It is clearly labelled
  as *not* generative AI.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Callable

from simulation.config import SCENARIO_TYPES

from .tools import Toolbox

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-opus-5-5"
FALLBACK_BETA = "server-side-fallback-2026-07-01"

SYSTEM_PROMPT = """You are MetroOps, an AI operations assistant for a metro depot. You help the control-room \
operator decide how to induct trains (release them from the depot onto the mainline) for an operating day.

You act through tools connected to a SimPy simulation of the depot and line and to three schedulers:
- FCFS: a baseline that sends the earliest-ready train when service is short now.
- Heuristic: rules with 12-minute demand look-ahead, avoids bunching, prefers healthy trains.
- PPO: a reinforcement-learning agent (MaskablePPO, warm-started by behaviour cloning) trained on simulated days.

How to work:
1. Understand the operator's situation. Turn it into a concrete what-if day with create_scenario (scenario \
type, demand change, fleet size, extra headway, closures with HH:MM times). If the question is general, use the \
saved evaluation and/or a representative base day.
2. Gather evidence: simulate the day with the relevant schedulers (compare_schedulers runs them on identical \
conditions). Look at the demand profile, fleet status or induction plan when they help explain the result.
3. Reflect. If the result is surprising or the decision is close, test one more variant (another seed or a \
milder/harsher version of the change) before concluding. Keep the number of tool calls proportionate: a simple \
question needs two or three calls, a hard what-if may need more.
4. Answer: recommend which scheduler / induction approach to use and why, quoting the simulated numbers. \
Include the first few recommended inductions when the operator asks for a plan.

Rules:
- Every number you state must come from a tool result in this conversation. Do not invent figures.
- Safety rules (one train per block and platform, minimum headways, closures, readiness, induction capacity) are \
enforced by the simulator for every scheduler. Report the safety audit; never suggest bypassing a safety rule.
- All data is synthetic simulation data, not real KMRL operational data. Say so once when you give results.
- If a request is outside what the tools can simulate, say what you can and cannot test.

Answer format (Markdown, concise): **Situation** (one or two lines), **What I tested**, **Results** (a small \
table), **Recommendation**, **Caveats**."""


@dataclass
class AgentStep:
    kind: str                     # "thought" | "tool_call" | "tool_result" | "answer" | "error"
    content: str = ""
    tool: str | None = None
    tool_input: dict | None = None
    tool_output: dict | None = None
    elapsed_s: float = 0.0


@dataclass
class AgentRun:
    question: str
    answer: str
    steps: list[AgentStep] = field(default_factory=list)
    mode: str = "llm"
    model: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    tool_calls: int = 0

    def trace_table(self) -> list[dict]:
        rows = []
        for i, s in enumerate(self.steps, 1):
            if s.kind == "tool_call":
                rows.append({"step": i, "type": "act", "detail": f"{s.tool}({json.dumps(s.tool_input)})"})
            elif s.kind == "tool_result":
                txt = json.dumps(s.tool_output)
                rows.append({"step": i, "type": "observe", "detail": txt[:200] + ("…" if len(txt) > 200 else "")})
            elif s.kind == "thought":
                rows.append({"step": i, "type": "reason", "detail": s.content[:200]})
            elif s.kind == "error":
                rows.append({"step": i, "type": "error", "detail": s.content})
        return rows


StepCallback = Callable[[AgentStep], None]


def _as_tool_param(spec: dict) -> dict:
    return {"name": spec["name"], "description": spec["description"], "input_schema": spec["input_schema"]}


class MetroOpsAgent:
    """LLM agent (Claude) that plans and acts through the MetroSched toolbox.

    The conversation is kept across ``ask`` calls, so the operator can ask
    follow-up questions ("now close platform 2 as well").
    """

    def __init__(self, client=None, toolbox: Toolbox | None = None, model: str = DEFAULT_MODEL,
                 max_turns: int = 15, effort: str = "medium", use_fallbacks: bool = True):
        if client is None:
            import anthropic
            client = anthropic.Anthropic()
        self.client = client
        self.toolbox = toolbox or Toolbox()
        self.model = model
        self.max_turns = int(max_turns)
        self.effort = effort
        self.use_fallbacks = use_fallbacks
        self.messages: list[dict] = []
        self.tools = [_as_tool_param(s) for s in self.toolbox.specs]

    def reset(self) -> None:
        self.messages = []

    def _create(self):
        kwargs = dict(model=self.model, max_tokens=16000, system=SYSTEM_PROMPT, tools=self.tools,
                      messages=self.messages, output_config={"effort": self.effort})
        if self.use_fallbacks:
            # On a safety-classifier decline the API re-runs the request on a fallback model.
            kwargs.update(betas=[FALLBACK_BETA], fallbacks="default")
        return self.client.beta.messages.create(**kwargs)

    def ask(self, question: str, on_step: StepCallback | None = None) -> AgentRun:
        run = AgentRun(question=question, answer="", model=self.model)
        emit = (lambda s: (run.steps.append(s), on_step(s) if on_step else None))  # noqa: E731
        self.messages.append({"role": "user", "content": question})
        for _ in range(self.max_turns):
            t0 = time.perf_counter()
            response = self._create()
            usage = getattr(response, "usage", None)
            if usage is not None:
                run.input_tokens += int(getattr(usage, "input_tokens", 0) or 0)
                run.output_tokens += int(getattr(usage, "output_tokens", 0) or 0)
            # Keep the full content (including thinking blocks) so the next request is valid.
            self.messages.append({"role": "assistant", "content": response.content})
            texts = [b.text for b in response.content if getattr(b, "type", None) == "text" and b.text.strip()]
            tool_uses = [b for b in response.content if getattr(b, "type", None) == "tool_use"]
            stop = response.stop_reason

            if stop == "refusal":
                emit(AgentStep("error", "The model declined this request."))
                run.answer = "I can't help with that request. Please ask about metro induction scheduling."
                return run
            if stop == "pause_turn":
                continue
            if not tool_uses:
                run.answer = "\n\n".join(texts) or "(no answer returned)"
                if stop == "max_tokens":
                    run.answer += "\n\n_(answer truncated: output limit reached)_"
                emit(AgentStep("answer", run.answer, elapsed_s=time.perf_counter() - t0))
                return run

            for txt in texts:   # reasoning the model shares before acting
                emit(AgentStep("thought", txt))
            results = []
            for tu in tool_uses:
                args = tu.input if isinstance(tu.input, dict) else {}
                emit(AgentStep("tool_call", tool=tu.name, tool_input=args))
                t1 = time.perf_counter()
                out = self.toolbox.execute(tu.name, args)
                run.tool_calls += 1
                emit(AgentStep("tool_result", tool=tu.name, tool_input=args, tool_output=out,
                               elapsed_s=time.perf_counter() - t1))
                results.append({"type": "tool_result", "tool_use_id": tu.id,
                                "content": json.dumps(out, default=str), "is_error": "error" in out})
            # all results of one turn go back in a single user message
            self.messages.append({"role": "user", "content": results})

        run.answer = ("I stopped after the maximum number of reasoning steps without a final answer. "
                      "Try a more specific question.")
        emit(AgentStep("error", run.answer))
        return run


# ============================================================ offline planner
_SECTION_ALIASES = [
    (r"\bexit[\s-]*(?:track)?[\s-]*([12])\b", "EXIT-{}"),
    (r"\b(?:platform|pf)[\s-]*([12])\b", "PF-{}"),
    (r"\btransfer[\s-]*line\b", "TRANSFER-LINE"),
    (r"\bdepot[\s-]*junction\b", "DEPOT-JUNCTION"),
]
_TIME_RANGE = r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\s*(?:-|–|to|until|till)\s*(\d{1,2})(?::(\d{2}))?\s*(am|pm)?"


def _hhmm(h: str, m: str | None, ampm: str | None) -> str:
    hh = int(h)
    if ampm == "pm" and hh < 12:
        hh += 12
    if ampm == "am" and hh == 12:
        hh = 0
    return f"{hh:02d}:{int(m or 0):02d}"


def parse_request(question: str) -> dict:
    """Very small rule-based parser: question -> create_scenario arguments."""
    q = question.lower()
    # demand events win over closure words: a closure is added to whatever day type is described
    if any(w in q for w in ("peak", "festival", "event", "rush", "crowd", "match", "concert")):
        stype = "peak_demand"
    elif any(w in q for w in ("disrupt", "fault", "failure", "breakdown", "closed", "closure", "block")):
        stype = "disruption"
    else:
        stype = "normal"
    for st in SCENARIO_TYPES:
        if st.replace("_", " ") in q or st in q:
            stype = st
    args: dict = {"scenario_type": stype}
    m = re.search(r"seed\s*(\d+)", q)
    if m:
        args["seed"] = int(m.group(1))
    m = re.search(r"demand[^.\d]*?([+-]?\d+)\s*%", q) or re.search(r"([+-]?\d+)\s*%\s*(?:more|higher|extra)", q)
    if m:
        pct = int(m.group(1))
        if "less" in q or "lower" in q or "drop" in q:
            pct = -abs(pct)
        args["demand_multiplier"] = round(min(2.0, max(0.5, 1 + pct / 100)), 2)
    m = re.search(r"(\d{2})\s*(?:trains|trainsets)", q)
    if m and 15 <= int(m.group(1)) <= 30:
        args["fleet_size"] = int(m.group(1))
    closures = []
    for pattern, fmt in _SECTION_ALIASES:
        for sm in re.finditer(pattern, q):
            sid = fmt.format(*sm.groups()) if sm.groups() else fmt
            tail = q[sm.end(): sm.end() + 60]
            tm = re.search(_TIME_RANGE, tail)
            if tm:
                h1, m1, a1, h2, m2, a2 = tm.groups()
                closures.append({"section_id": sid, "start": _hhmm(h1, m1, a1 or a2), "end": _hhmm(h2, m2, a2)})
    if closures:
        args["closures"] = closures
    return args


class OfflineAgent:
    """Same tools, same trace, but a fixed rule-based plan instead of an LLM.

    Use for demos without an API key. This mode is *agentic* (it plans, acts
    through tools and decides from the observations) but it is *not*
    generative AI: the answer is a template filled with simulated numbers.
    """

    def __init__(self, toolbox: Toolbox | None = None):
        self.toolbox = toolbox or Toolbox()

    def reset(self) -> None:
        self.toolbox.scenarios.clear()
        self.toolbox.runs.clear()

    def ask(self, question: str, on_step: StepCallback | None = None) -> AgentRun:
        run = AgentRun(question=question, answer="", mode="offline", model="rule-based planner")

        def emit(step: AgentStep) -> None:
            run.steps.append(step)
            if on_step:
                on_step(step)

        def act(tool: str, args: dict) -> dict:
            emit(AgentStep("tool_call", tool=tool, tool_input=args))
            t0 = time.perf_counter()
            out = self.toolbox.execute(tool, args)
            run.tool_calls += 1
            emit(AgentStep("tool_result", tool=tool, tool_input=args, tool_output=out,
                           elapsed_s=time.perf_counter() - t0))
            return out

        args = parse_request(question)
        emit(AgentStep("thought", f"Plan: build the operating day described ({args}), simulate it with every "
                                  "scheduler on identical conditions, pick the best on service-critical metrics, "
                                  "then list its first inductions."))
        sc = act("create_scenario", args)
        if "error" in sc:
            emit(AgentStep("thought", f"Scenario rejected ({sc['error']}); falling back to the base day."))
            args = {"scenario_type": args["scenario_type"]}
            sc = act("create_scenario", args)
        sid = sc["scenario_id"]
        demand = act("get_demand_profile", {"scenario_id": sid})
        cmp_ = act("compare_schedulers", {"scenario_id": sid})
        table = cmp_["table"]
        # decision rule: shortfall first, then missed opportunities, then induction delay
        crit = ["Service shortfall [train-h/day]", "Missed induction opportunities [count/day]",
                "Average induction delay [min]"]
        names = list(next(iter(table.values())).keys())
        best = min(names, key=lambda n: tuple(table[c][n] for c in crit))
        emit(AgentStep("thought", f"Reflect: lowest service shortfall / missed slots / delay → {best}."))
        plan = act("get_induction_plan", {"scenario_id": sid, "scheduler": best, "limit": 6})

        lines = [f"**Situation**: {args['scenario_type'].replace('_', ' ')} day; changes: "
                 f"{', '.join(sc['changes'])}. Peak need {demand['max_required_trains']} trains "
                 f"around {', '.join(demand['peak_hours'])}.",
                 f"**What I tested**: {', '.join(names)} on the same simulated day (scenario {sid}).",
                 "**Results**", "", "| Metric | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
        for label, vals in table.items():
            lines.append(f"| {label} | " + " | ".join(str(vals[n]) for n in names) + " |")
        lines += ["", f"**Recommendation**: use **{best}** for this day. First inductions:"]
        for r in plan["inductions"]:
            lines.append(f"- {r['induction_start']} {r['train']} via {r['exit_track']} → {r['entry_point']} "
                         f"(in service {r['in_service']})")
        sv = cmp_["safety_violations"]
        safety = ("0 for every scheduler" if not any(sv.values())
                  else ", ".join(f"{k}: {v}" for k, v in sv.items()))
        lines += ["", "**Caveats**: offline rule-based planner (no LLM); synthetic simulation data, not real KMRL "
                      f"data. Safety audit violations: {safety}."]
        run.answer = "\n".join(lines)
        emit(AgentStep("answer", run.answer))
        return run


def make_agent(offline: bool = False, model: str = DEFAULT_MODEL, toolbox: Toolbox | None = None):
    """LLM agent if requested and credentials work, otherwise the offline planner."""
    if offline:
        return OfflineAgent(toolbox)
    return MetroOpsAgent(toolbox=toolbox, model=model)
