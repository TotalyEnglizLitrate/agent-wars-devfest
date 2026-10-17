# Autonomous Supply Coordinator

**Track:** Logistics Handlers · DevFest Agent Wars 2026

An AI agent that runs a regional sugar distributor's supply chain. It buys from 5 mills at the best total landed cost, delivers by truck to 6 factories, and when disruptions hit (a mill short-ships, a factory cancels) it re-plans deliveries, negotiates with customers and keeps the books straight.

> **The solver finds the best plan; the agent makes it real.**

Optimization is deterministic: a PuLP + HiGHS LP for procurement and a MIP for delivery planning. The agent is built with Google ADK and Gemini, and does what a solver can't:
- **Reads messy events:** *"truck 2 had a loading problem at the mill, only 12t went on"*.
- **Negotiates:** asks a factory to accept a delay and handles its counter-offer.
- **Runs the loop:** solve → ask → re-solve → commit → reconcile.
- **Explains its decisions** and escalates anything above ₹15 lakh to a human.
- **Learns across runs:** which mills really are flaky, and what discount each factory wants.

The LLM never does arithmetic. Solvers and the ledger are tools it calls. Design details are in [`docs/DESIGN.md`](docs/DESIGN.md).

---

## Setup

You need [uv](https://docs.astral.sh/uv/getting-started/installation/). It installs Python 3.14 and all dependencies for you.

```bash
git clone https://github.com/TotalyEnglizLitrate/agent-wars-devfest.git
cd agent-wars-devfest
uv sync
```

For the live LLM agent, add a Gemini API key (from [Google AI Studio](https://aistudio.google.com/apikey); the free tier works):

```bash
echo "GOOGLE_API_KEY=your-key-here" > .env     # .env is gitignored
```

Without a key, everything still runs: the demo switches to the scripted agent automatically.

---

## Running the demo

### 1. The full demo (recommended)

```bash
uv run python run_demo.py
```

This runs all three acts. Gemini drives every day that has an event or an open issue. At the end it prints the decision log, a comparison against the first-come-first-served baseline, and the lessons learned. It takes a few minutes, depending on Gemini's response times.

### 2. Scripted mode: no API key, no network, deterministic

```bash
uv run python run_demo.py --scripted
```

The same tools are called in a fixed order, so you get identical numbers every time. Use it for a reliable live pitch.

### 3. One act, showing every tool call

```bash
uv run python run_demo.py --acts act2 --verbose
```

Act 2 is the showpiece: a supplier short-ships, the agent asks Factory B to accept a delay, B counters at 5%, and the agent accepts and re-plans. Sample output (scripted):

```
━━━ act2 (starts day 1) ━━━
  day 1: committed 1 truck move(s) (MIP: 1 warehouse dispatch(es))
  day 2: event: S1 dispatch here - truck 2 had a loading problem at the mill, only 12t went on. ...;
         bought 8t from S3 (LP: forecast ₹39,829/t, reliability 0.90);
         asked C2 to take O2 on day 4: countered at 5%: accept;
         reconciled: PO1: recorded 8t short-shipped by S1; reconciled: PO1: booked credit ₹-302,858
  ...
━━━ Agent vs FCFS baseline (same seed) ━━━
  act2 FCFS : fulfilled 100% | on time 99% | wasted 0t | total ₹3,245,149
  act2 agent: fulfilled 100% | on time 91% | wasted 0t | total ₹3,019,485  (saves ₹225,664)
```

### 4. The learning demo

```bash
uv run python run_demo.py --scripted --reset-memory --acts act1 act2 --runs 2
```

Each act runs twice from a blank memory. Watch for:
- **Run 2 of act 2** opens with *"offering 5% (learned)"*, and Factory B accepts in one round instead of two.
- **Act 1 purchase reasons** show mill S1's reliability falling below the listed 0.70 as its misses are counted, and more of the buying goes to reliable S3.

Lessons are stored in `.agent_memory/lessons.json` and printed at the end. `--reset-memory` wipes them.

### 5. Against a live World server

```bash
# terminal 1: the simulated world
uv run uvicorn agent_wars_devfest.world.app:app --reload --port 8000

# terminal 2: the agent, talking to it over HTTP
uv run python run_demo.py --url http://localhost:8000
```

You can also poke the world directly, or open the interactive API docs at <http://localhost:8000/docs>:

```bash
curl -X POST localhost:8000/scenario/act2          # load act 2 (day 1)
curl -X POST localhost:8000/clock/advance -H 'content-type: application/json' -d '{"days": 1}'
curl localhost:8000/allocation-state               # the short-ship shows up under "issues"
curl -X POST localhost:8000/customers/C2/reschedule-request \
     -H 'content-type: application/json' -d '{"order_id": "O2", "new_day": 4}'   # → counter, 5%
curl localhost:8000/scorecard
```

### 6. Baseline and tests

```bash
uv run python -m agent_wars_devfest.tools.baseline   # FCFS numbers for every act
uv run pytest                                        # 64 tests, ~6 seconds
```

### All options

| Flag | What it does |
|---|---|
| `--scripted` | Deterministic tool sequence, no LLM. Also the automatic per-turn fallback if Gemini fails |
| `--acts act1 act2 act3` | Which acts to run (default: all) |
| `--runs N` | Repeat each act N times, to show learning |
| `--reset-memory` | Start from a blank lessons memory |
| `--url URL` | Use a running World server instead of the in-process one |
| `--verbose` | Print every tool call as it happens |
| `--llm-every-day` | Use the LLM on quiet days too (more requests) |

| Environment variable (`.env` works) | Default | Purpose |
|---|---|---|
| `GOOGLE_API_KEY` | none | Gemini key; without it, scripted mode |
| `AGENT_MODEL` | `gemini-3.5-flash-lite` | Any Gemini model with tool calling |
| `AGENT_MEMORY_PATH` | `.agent_memory/lessons.json` | Where lessons are stored |
| `WORLD_URL` | `http://localhost:8000` | World URL used by the agent tools outside `run_demo.py` |

---

## The three acts

| Act | Day | What happens | What the agent does |
|---|---|---|---|
| **1. Procurement** | 0 | 6 factories need 150t over a week; the warehouse holds 20t | Weighs forecast prices against each mill's reliability, buys just in time, escalates the big buy for approval |
| **2. Shortage** (showpiece) | 2 | Mill S1 short-ships truck 2: *"only 12t went on"* | Covers Factory A from stock, asks Factory B to take the rest on day 4, B **counters at 5%**, agent accepts, re-plans, commits; reconcile books an 8t credit against S1 |
| **3. Cancellation** | 3 | A factory cancels while 20t is on the road to it | Re-routes the truck to a factory that still needs sugar, cancels an unshipped PO, books the 10% cancellation fee |

## Results vs first-come-first-served

Same seed, from `GET /scorecard`. **Live** is Gemini 3.5 flash-lite; **scripted** is the deterministic mode.

| Act | FCFS | Agent (scripted) | Agent (live) | Takeaway |
|---|---|---|---|---|
| act1 | ₹49.42L | ₹49.40L | ₹50.12L | About even. Hedging with reliable mills costs a little; on time 63–69% vs FCFS 93% |
| act2 | ₹32.45L | ₹30.19L | ₹30.19L | **Saves ₹2.3L** |
| act3 | ₹26.99L | ₹15.54L | ₹15.54L | **Saves ₹11.5L; 0t wasted vs FCFS's 25t** sent to the cancelled factory |

FCFS fills 100% of orders in every act, because mill S1's 1-day lead time lets it re-buy after every miss. So the comparison that matters is **cost and waste**, not fulfillment rate.

---

## How it works

```
┌────────────────────────┐   tool calls   ┌──────────────────────────────┐
│  ADK agent (Gemini)    │ ─────────────▶ │  World API (FastAPI)         │
│  instruction = prompt  │ ◀───────────── │  suppliers, prices, orders,  │
│    + learned lessons   │                │  trucks, clock, events,      │
│  23 function tools     │                │  negotiations, ledger,       │
└──────────┬─────────────┘                │  scorecard                   │
           │                              └──────────────────────────────┘
           ▼
┌─────────────────────────────────────────────────────────────┐
│ Deterministic tools                                         │
│  procurement LP        — price + holding + shortfall risk   │
│  reallocation MIP      — warehouse → trucks → factories     │
│  solver adapter        — re-route in-transit loads, delay   │
│                          asks, just-in-time buying          │
│  reconciliation ledger — POST /reconcile, invariants        │
│  FCFS baseline         — the comparison                     │
└─────────────────────────────────────────────────────────────┘
```

**Each simulated day, the agent:**
1. Reads the day's events.
2. Runs the procurement LP and places today's buys.
3. Runs the reallocation planner.
4. Negotiates any delays it needs (max 3 rounds), re-solving after each reply.
5. Commits the plan.
6. Reconciles the books and checks the invariants.
7. Logs its reasoning in plain English.

**Guardrails, enforced in code:**
- Purchases or plans above ₹15 lakh are refused until `escalate_to_human` is called.
- Max 3 negotiation rounds per order.
- No committing while customer asks are unresolved.
- No direct ledger writes; the agent only calls `reconcile`.
- The world validates every input, so the agent can't invent stock, prices or money.

**The world** is seeded (`seed=42`) with Faker and numpy: 5 mills, 6 factories, 8 trucks and 90 days of prices. Mill S1 is cheapest but misses about 30% of deliveries, and purchase orders the agent places really do short-ship (seeded per order, so every run sees the same misses).

**Learning:** after each run, lessons are computed in code, not by the LLM, so their numbers are always true. They're added to the agent's instructions on every turn.

---

## Repo layout

```
├── run_demo.py              # entry point: runs acts, decision log, comparison, lessons
├── docs/DESIGN.md           # how and why it works
├── src/agent_wars_devfest/
│   ├── models.py            # shared pydantic schemas
│   ├── world/               # datagen.py (seeded data), app.py (FastAPI), scenarios.py (acts)
│   ├── tools/               # procurement.py (LP), reallocation.py (MIP), ledger.py, baseline.py
│   └── agent/               # agent.py (ADK), tools.py, prompts.py, memory.py, solvers.py (adapter)
└── tests/                   # 64 tests: world, ledger, baseline, agent
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `No GOOGLE_API_KEY set: running the scripted agent` | Add the key to `.env` in the repo root |
| `429 RESOURCE_EXHAUSTED` | Free-tier rate limit. The agent retries and falls back to the scripted turn for that day. Keep the default `gemini-3.5-flash-lite` (other flash models allow 5 requests/min) |
| Results differ between runs | Leftover lessons. Add `--reset-memory` |
| `uv sync` is slow the first time | It downloads Python 3.14 and the HiGHS solver once |

## Known limitations

- The reallocation MIP prices every unmet tonne the same and only plans free trucks from the warehouse. The adapter compensates: it solves one priority/due-day tier at a time, and handles re-routing and delay asks itself.
- In act 1, hedging with reliable mills costs slightly more than FCFS's "buy cheapest, re-buy on a miss", because this world makes misses cheap to recover from.
- `escalate_to_human` auto-approves in demo mode; a real deployment would wait for a person.
