# MetroSched – Agentic AI layer

> All data is synthetic simulation data, not KMRL operational data.

## 1. What it adds

The schedulers (FCFS, heuristic and the PPO reinforcement-learning agent) answer one fixed question: *which
train should leave the depot next?* A control-room operator asks broader questions, for example:

* "There's a festival tomorrow with demand +25%, and platform 2 is closed from 08:00 to 09:30. What should we do?"
* "Exit track 1 is closed from 17:00 to 18:30. How much does that hurt the evening peak?"
* "We only have 21 trainsets today. Is the RL agent still better than FCFS?"

The agentic layer answers such questions on its own. An **LLM agent (Claude)** reads the question, decides
which experiments to run, runs them through **tools** connected to the simulator and the schedulers, looks
at the results, runs more experiments if needed, and then writes a recommendation that quotes the simulated
numbers.

## 2. Architecture

```
Operator question (natural language)
        │
        ▼
┌───────────────────────────────┐  Generative + agentic AI
│ MetroOpsAgent (Claude LLM)    │  plan → act (tool call) → observe → reflect → answer
└──────────────┬────────────────┘
               │ tool calls (JSON)
               ▼
┌───────────────────────────────┐  agentic/tools.py
│ Toolbox: 8 tools              │  what-if scenario, run/compare schedulers,
└──────────────┬────────────────┘  demand, fleet, induction plan, saved results
               ▼
┌───────────────────────────────┐  Reinforcement-learning agent + baselines
│ PPO agent · Heuristic · FCFS  │
└──────────────┬────────────────┘
               ▼
┌───────────────────────────────┐  Hard safety rules + independent audit
│ SimPy metro simulator         │  (enforced for every scheduler and every tool call)
└───────────────────────────────┘
```

This is a **hierarchical multi-agent system**:

| Level | Agent | Type of AI | Decides |
|---|---|---|---|
| 1 | MetroOpsAgent | LLM agent (generative AI, tool use) | *what to test* and *what to recommend* |
| 2 | PPO scheduler | Reinforcement-learning agent | *which train to send, and when*, every 60 s |
| 3 | Simulator safety layer | Rule-based guard | *whether an action is allowed* (it can always say no) |

## 3. Agent properties and where they are in the code

| Property | Implementation |
|---|---|
| **Perception** | Tool results: KPIs, demand profile, fleet state, induction logs ([agentic/tools.py](../agentic/tools.py)) |
| **Planning / reasoning** | Claude chooses the next tool from the question and from earlier observations ([agentic/agent.py](../agentic/agent.py), `MetroOpsAgent.ask`) |
| **Action** | 8 tools defined with JSON schemas (`TOOL_SPECS`): `get_system_overview`, `get_saved_evaluation`, `create_scenario`, `get_demand_profile`, `get_fleet_status`, `run_scheduler`, `compare_schedulers`, `get_induction_plan` |
| **Reflection** | The system prompt tells the agent to test another variant when a result is close or surprising. Tool errors are sent back to the model so it can correct itself. |
| **Memory** | The conversation is kept between questions, so follow-ups work ("now close platform 1 as well"). Simulation runs are cached per scenario and scheduler. |
| **Autonomy with limits** | At most 15 reasoning turns. Safety rules live in the simulator, not in the LLM, so the agent cannot cause an unsafe movement. |
| **Generation** | The final answer is generated in natural language: situation, what was tested, a results table, a recommendation and caveats. |
| **Grounding** | Every number must come from a tool result. The agent may not invent figures. |

### The agent loop (simplified)

```python
messages = [{"role": "user", "content": question}]
while True:
    response = client.beta.messages.create(model="claude-opus-5-5", system=SYSTEM_PROMPT,
                                           tools=TOOLS, messages=messages, ...)
    messages.append({"role": "assistant", "content": response.content})
    tool_uses = [b for b in response.content if b.type == "tool_use"]
    if not tool_uses:                      # the agent has decided it has enough evidence
        return final_text(response)
    results = [run_tool(t.name, t.input) for t in tool_uses]   # act on the simulator
    messages.append({"role": "user", "content": results})      # observe
```

## 4. Example run (offline mode, real output)

Question: *"Festival tomorrow, demand +25%, platform 2 closed 8:00-9:30"*

| Step | Type | Detail |
|---|---|---|
| 1 | reason | plan: build the day, simulate every scheduler, pick the best, list the first inductions |
| 2–3 | act / observe | `create_scenario(peak_demand, demand ×1.25, close PF-2 08:00–09:30)` |
| 4–5 | act / observe | `get_demand_profile` → peak need 18 trains around 07:00–08:00 and 17:00 |
| 6–7 | act / observe | `compare_schedulers` → FCFS / Heuristic / PPO on identical conditions |
| 8 | reflect | lowest service shortfall → Heuristic (0.52 train-h vs FCFS 3.92) |
| 9–10 | act / observe | `get_induction_plan(Heuristic)` → first 6 inductions |
| 11 | answer | table + recommendation; 0 safety violations for every scheduler |

In LLM mode, Claude picks these steps itself. For example, it may test a second seed or a milder closure
before it concludes.

## 5. How to run

```powershell
pip install -r requirements.txt          # includes the anthropic SDK
$env:ANTHROPIC_API_KEY = "sk-ant-..."    # your key (never commit it)
python main.py agent "Festival tomorrow, demand +25%, platform 2 closed 08:00-09:30. What should we do?"
python main.py agent                     # interactive chat, follow-up questions keep context
python main.py agent --offline "..."     # no key: same tools, rule-based planner (not generative AI)
python main.py dashboard                 # page 6 · Agentic AI assistant
```

Options: `--model claude-sonnet-5-5` (cheaper), `--verbose` (prints the agent's intermediate reasoning).

## 6. Two modes, stated honestly

| | LLM agent (`MetroOpsAgent`) | Offline planner (`OfflineAgent`) |
|---|---|---|
| Understands free text | yes (Claude) | keywords and regular expressions only |
| Chooses its own steps | yes | fixed plan |
| Generative AI | yes | no (template answer) |
| Needs an API key or network | yes | no |
| Same tools, same safety | yes | yes |

The offline mode is there so the demo runs without internet. The agentic AI and generative AI part of the
project is the LLM mode.

## 7. Tests

[tests/test_agentic.py](../tests/test_agentic.py) covers:

* every tool, including invalid requests (unknown section, closing every exit track at once, out-of-range values);
* the LLM loop with a fake client: tool calls are executed, parallel results go back in one message, tool errors are returned to the model, refusals and the turn limit are handled, follow-up questions keep the context;
* the offline planner;
* dashboard page 6.

The tests make no real API calls.
