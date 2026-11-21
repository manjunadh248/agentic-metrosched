"""Tests for the agentic AI layer (agentic/): tools, LLM agent loop (with a fake client), offline planner."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from agentic.agent import FALLBACK_BETA, MetroOpsAgent, OfflineAgent, parse_request
from agentic.tools import TOOL_SPECS, Toolbox, parse_clock


@pytest.fixture(scope="module")
def toolbox():
    return Toolbox(allow_ppo=False)   # FCFS + heuristic only: fast and torch-free


# ------------------------------------------------------------------ tools
def test_tool_specs_are_valid_json_schemas():
    names = [t["name"] for t in TOOL_SPECS]
    assert len(names) == len(set(names))
    for t in TOOL_SPECS:
        assert t["description"] and t["input_schema"]["type"] == "object"
        for req in t["input_schema"].get("required", []):
            assert req in t["input_schema"]["properties"]


def test_parse_clock():
    assert parse_clock("08:30") == 8 * 3600 + 1800
    assert parse_clock(60) == 60
    with pytest.raises(ValueError):
        parse_clock("8h30")


def test_create_scenario_applies_what_if_changes(toolbox):
    out = toolbox.execute("create_scenario", {
        "scenario_type": "normal", "seed": 7, "demand_multiplier": 1.2, "extra_headway_s": 30,
        "closures": [{"section_id": "pf-2", "start": "08:00", "end": "09:00"}]})
    assert "error" not in out
    sc, _, _ = toolbox.scenarios[out["scenario_id"]]
    base = toolbox.execute("create_scenario", {"scenario_type": "normal", "seed": 7})
    base_sc, _, _ = toolbox.scenarios[base["scenario_id"]]
    assert sc.demand_multiplier == pytest.approx(base_sc.demand_multiplier * 1.2, abs=1e-3)
    assert sc.extra_headway_s == base_sc.extra_headway_s + 30
    assert ("PF-2", 8 * 3600, 9 * 3600) in sc.platform_closures


@pytest.mark.parametrize("args", [
    {"scenario_type": "holiday"},
    {"scenario_type": "normal", "demand_multiplier": 5},
    {"scenario_type": "normal", "fleet_size": 3},
    {"scenario_type": "normal", "closures": [{"section_id": "NOPE", "start": "08:00", "end": "09:00"}]},
    {"scenario_type": "normal", "closures": [{"section_id": "EXIT-1", "start": "09:00", "end": "08:00"}]},
    {"scenario_type": "normal", "closures": [{"section_id": "EXIT-1", "start": "08:00", "end": "09:00"},
                                             {"section_id": "EXIT-2", "start": "08:30", "end": "09:30"}]},
])
def test_create_scenario_rejects_invalid_requests(toolbox, args):
    assert "error" in toolbox.execute("create_scenario", args)


def test_unknown_tool_and_scenario_return_errors(toolbox):
    assert "error" in toolbox.execute("launch_rocket", {})
    assert "error" in toolbox.execute("run_scheduler", {"scenario_id": "S999", "scheduler": "FCFS"})
    assert "error" in toolbox.execute("run_scheduler", {"scenario_id": "S1", "bogus": 1})


def test_compare_and_plan_come_from_real_simulation(toolbox):
    sid = toolbox.execute("create_scenario", {"scenario_type": "peak_demand", "seed": 11})["scenario_id"]
    cmp_ = toolbox.execute("compare_schedulers", {"scenario_id": sid, "schedulers": ["FCFS", "Heuristic"]})
    assert set(cmp_["safety_violations"].values()) == {0}
    cov = cmp_["table"]["Passenger demand coverage [%]"]
    assert 0 < cov["FCFS"] <= 100 and 0 < cov["Heuristic"] <= 100
    plan = toolbox.execute("get_induction_plan", {"scenario_id": sid, "scheduler": "Heuristic", "limit": 5})
    assert plan["showing"] == 5 and plan["total"] >= 5
    assert all(r["why"].startswith("Heuristic") for r in plan["inductions"])
    # cached: second call re-uses the simulation run
    n_runs = len(toolbox.runs)
    toolbox.execute("run_scheduler", {"scenario_id": sid, "scheduler": "FCFS"})
    assert len(toolbox.runs) == n_runs


def test_demand_fleet_overview_and_saved_eval(toolbox):
    sid = toolbox.execute("create_scenario", {"scenario_type": "normal"})["scenario_id"]
    d = toolbox.execute("get_demand_profile", {"scenario_id": sid})
    assert d["max_required_trains"] > 0 and len(d["hourly"]) >= 18
    f = toolbox.execute("get_fleet_status", {"scenario_id": sid})
    assert f["fleet_size"] == toolbox.base_cfg["fleet"]["size"]
    ov = toolbox.execute("get_system_overview", {})
    assert "SYNTHETIC" in ov["data_notice"]
    ev = toolbox.execute("get_saved_evaluation", {"scenario_type": "all"})
    assert "error" in ev or ev["episodes"] > 0


def test_ppo_unavailable_is_reported_not_crashing(toolbox):
    sid = toolbox.execute("create_scenario", {"scenario_type": "normal"})["scenario_id"]
    assert "error" in toolbox.execute("run_scheduler", {"scenario_id": sid, "scheduler": "PPO"})


# ------------------------------------------------------------------ LLM agent loop (fake client)
def _text(t):
    return SimpleNamespace(type="text", text=t)


def _tool(i, name, args):
    return SimpleNamespace(type="tool_use", id=i, name=name, input=args)


def _resp(content, stop):
    return SimpleNamespace(content=content, stop_reason=stop, usage=SimpleNamespace(input_tokens=10, output_tokens=5))


class FakeClient:
    """Mimics client.beta.messages.create and records every request."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append({**kwargs, "messages": list(kwargs["messages"])})
        return self.responses.pop(0)


def test_llm_agent_runs_tools_and_answers(toolbox):
    client = FakeClient([
        _resp([_text("I'll build the day first."),
               _tool("t1", "create_scenario", {"scenario_type": "normal", "seed": 3})], "tool_use"),
        _resp([_tool("t2", "run_scheduler", {"scenario_id": "S1", "scheduler": "FCFS"}),
               _tool("t3", "run_scheduler", {"scenario_id": "S1", "scheduler": "Heuristic"})], "tool_use"),
        _resp([_text("**Recommendation**: Heuristic.")], "end_turn"),
    ])
    agent = MetroOpsAgent(client=client, toolbox=Toolbox(allow_ppo=False))
    steps = []
    run = agent.ask("Which scheduler on a normal day?", on_step=steps.append)
    assert run.answer == "**Recommendation**: Heuristic."
    assert run.tool_calls == 3 and run.input_tokens == 30
    assert [s.kind for s in steps].count("tool_call") == 3
    # request shape: tools, effort, server-side fallbacks
    first = client.calls[0]
    assert first["model"] == "claude-opus-5-5"
    assert {t["name"] for t in first["tools"]} == {t["name"] for t in TOOL_SPECS}
    assert first["betas"] == [FALLBACK_BETA] and first["fallbacks"] == "default"
    # both parallel tool results returned in ONE user message, ids matching
    third = client.calls[2]["messages"]
    results = third[-1]["content"]
    assert third[-1]["role"] == "user" and [r["tool_use_id"] for r in results] == ["t2", "t3"]
    assert all(not r["is_error"] for r in results)
    assert "kpis" in json.loads(results[0]["content"])


def test_llm_agent_reports_tool_errors_to_model():
    client = FakeClient([
        _resp([_tool("t1", "run_scheduler", {"scenario_id": "S42", "scheduler": "FCFS"})], "tool_use"),
        _resp([_text("Sorry, I need a scenario first.")], "end_turn"),
    ])
    agent = MetroOpsAgent(client=client, toolbox=Toolbox(allow_ppo=False))
    agent.ask("run it")
    result = client.calls[1]["messages"][-1]["content"][0]
    assert result["is_error"] is True and "Unknown scenario_id" in result["content"]


def test_llm_agent_handles_refusal_and_turn_limit():
    agent = MetroOpsAgent(client=FakeClient([_resp([], "refusal")]), toolbox=Toolbox(allow_ppo=False))
    assert "can't help" in agent.ask("something odd").answer
    loop = [_resp([_tool(f"t{i}", "get_system_overview", {})], "tool_use") for i in range(3)]
    agent = MetroOpsAgent(client=FakeClient(loop), toolbox=Toolbox(allow_ppo=False), max_turns=3)
    run = agent.ask("loop forever")
    assert run.tool_calls == 3 and "maximum number" in run.answer


def test_llm_agent_keeps_conversation_for_follow_ups():
    client = FakeClient([_resp([_text("first")], "end_turn"), _resp([_text("second")], "end_turn")])
    agent = MetroOpsAgent(client=client, toolbox=Toolbox(allow_ppo=False))
    agent.ask("q1")
    agent.ask("q2")
    assert [m["role"] for m in client.calls[1]["messages"]] == ["user", "assistant", "user"]


# ------------------------------------------------------------------ offline planner
def test_parse_request_extracts_what_if():
    a = parse_request("Festival: demand +25%, platform 2 closed 8:00-9:30 and exit track 1 from 5pm to 6pm")
    assert a["scenario_type"] == "peak_demand" and a["demand_multiplier"] == 1.25
    assert {"section_id": "PF-2", "start": "08:00", "end": "09:30"} in a["closures"]
    assert {"section_id": "EXIT-1", "start": "17:00", "end": "18:00"} in a["closures"]
    assert parse_request("a breakdown with 22 trains") == {"scenario_type": "disruption", "fleet_size": 22}
    assert parse_request("normal day")["scenario_type"] == "normal"


def test_offline_agent_produces_recommendation():
    run = OfflineAgent(Toolbox(allow_ppo=False)).ask("Normal day, demand +10%")
    assert run.mode == "offline" and run.tool_calls == 4
    assert "**Recommendation**" in run.answer and "| Metric |" in run.answer
    kinds = [s.kind for s in run.steps]
    assert kinds[0] == "thought" and kinds[-1] == "answer"


# ------------------------------------------------------------------ dashboard page 6
def test_dashboard_assistant_page_offline(monkeypatch):
    from pathlib import Path

    from streamlit.testing.v1 import AppTest

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    app = Path(__file__).resolve().parent.parent / "dashboard" / "app.py"
    at = AppTest.from_file(str(app), default_timeout=300)
    at.run()
    at.sidebar.radio[0].set_value("6 · Agentic AI assistant").run()
    assert not at.exception
    assert at.radio(key="assistant_mode").value.startswith("Offline")
    at.chat_input[0].set_value("Normal day, demand +10%").run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("Recommendation" in m.value for m in at.markdown)
