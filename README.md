# Autonomous Supply Coordinator

**Track:** Logistics Handlers · DevFest Agent Wars 2026

An autonomous supply coordinator for a regional sugar distributor. It buys raw material at the best total landed cost, and when disruptions hit it re-plans deliveries and negotiates with customers to keep every factory supplied.

> **The solver finds the best plan; the agent makes it real.**

The optimization is deterministic (PuLP + HiGHS). The LLM agent (Google ADK + Gemini) handles what a solver can't:
- **Reads messy events:** e.g. *"truck 2 had a loading problem at the mill, only 12t went on"*.
- **Negotiates soft constraints:** asks a factory to accept a delay, and handles a counter-offer.
- **Runs the loop:** solve → ask → re-solve → commit → reconcile.
- **Explains its trade-offs**, and escalates to a human above ₹15 lakh.

The LLM never does arithmetic. Solvers and the ledger are tools it calls.

---

## Quick start

```bash
uv sync
echo "GOOGLE_API_KEY=your-key" > .env      # gitignored; only needed for the LLM agent

uv run python run_demo.py --scripted       # all three acts, deterministic, no LLM
uv run python run_demo.py                  # Gemini drives the eventful days
uv run pytest                              # 64 tests
```

Useful flags:

| Flag | What it does |
|---|---|
| `--scripted` | Same tools in a fixed order; no LLM needed. Also the automatic fallback if Gemini fails |
| `--acts act2` | Run only some acts |
| `--runs 2 --reset-memory` | Learning demo: run each act twice from a blank memory |
| `--url http://localhost:8000` | Use a live World server instead of the in-process one |
| `--verbose` | Print every tool call |
| `--llm-every-day` | Use the LLM on quiet days too (slow on the free tier) |

Live World API, for curl or a separate terminal:
```bash
uv run uvicorn agent_wars_devfest.world.app:app --reload --port 8000
curl localhost:8000/allocation-state
```

---

## The demo (three acts, one seed)

| Act | Day | What happens | What the agent does |
|---|---|---|---|
| **1. Procurement** | 0 | 6 factories need 150t over a week; the warehouse holds 20t | Weighs 5 suppliers' forecast prices against their reliability; buys just in time; escalates the big buy |
| **2. Shortage** (showpiece) | 2 | Supplier S1 short-ships truck 2: *"only 12t went on"* | Re-plans, asks Factory B to accept day 4, B **counters at 5%**, agent accepts, re-solves, commits; reconcile books an 8t credit against S1 |
| **3. Cancellation** | 3 | A factory cancels while 20t is on the road to it | Re-routes the truck to a waiting order, cancels an unshipped PO, books the 10% cancellation fee |

### Results vs first-come-first-served baseline

Scripted agent with B's solvers, same seed, from `GET /scorecard`:

| Act | FCFS total | Agent total | Difference | Notes |
|---|---|---|---|---|
| act1 | ₹49.42L | ₹49.40L | ≈ even | Agent on time 63% vs FCFS 93% (learns to 87% on run 2, at higher cost) |
| act2 | ₹32.45L | ₹30.19L | **saves ₹2.3L** | Agent on time 91% vs FCFS 99% |
| act3 | ₹26.99L | ₹15.54L | **saves ₹11.5L** | **0t wasted vs FCFS's 25t** delivered to the cancelled factory |

FCFS fills 100% of orders in every act, because a supplier's 1-day lead time lets it re-buy after every miss. So the comparison is **cost and waste**, not fulfillment rate.

---

## Architecture

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
│  procurement LP (B)      — price + holding + shortfall risk │
│  reallocation MIP (B)    — warehouse → trucks → factories   │
│  adapter (agent/solvers) — re-route in-transit loads,       │
│                            delay asks, JIT buying           │
│  reconciliation ledger   — POST /reconcile, invariants      │
│  FCFS baseline           — the comparison                   │
└─────────────────────────────────────────────────────────────┘
```

**Stack:** Python 3.14 · uv · google-adk · FastAPI · PuLP + HiGHS · Faker · numpy · pydantic · pytest

### The world
- Seeded with Faker and numpy (`seed=42`): 5 suppliers, 6 factories, 8 trucks, 90 days of prices.
- **Built-in trade-off:** S1 is cheapest but misses about 30% of deliveries; S2 is pricier but reliable. Purchase orders the agent places really do short-ship, seeded per PO so every run sees the same misses.
- Events physically change the world (a short-ship trims the truck). Each also carries a free-text message the agent has to interpret.
- Input is validated, so the agent can't invent stock, prices or money (no negative quantities, no caller-set prices).

### Guardrails, enforced in code
- **Big spends need approval:** anything above ₹15 lakh is refused until `escalate_to_human` is called, which auto-approves in demo mode and is logged loudly.
- **Negotiation limit:** at most 3 rounds per order.
- **No direct ledger writes:** the agent only calls `reconcile`.
- **No half-agreed plans:** `commit_plan` refuses while customer asks are unresolved.

### Learning loop: the agent gets better across runs
After each act, `agent/memory.py` turns the run into lessons. They're computed in code, not by the LLM, so the numbers are always true:
- **Supplier reliability seen in practice**, built up across runs and fed into the procurement LP. Example: *"S1 short-shipped 4 of our 10 POs; observed 0.63 vs listed 0.70"*.
- **Customer haggling:** *"C2 countered at 5%: offer that up front"*. Run 2 offers 5% straight away and saves a negotiation round.
- **Waste, lateness and best cost so far.**

Lessons are re-read into the ADK instruction on every turn. They're stored in `.agent_memory/` (gitignored); `--reset-memory` clears them.

---

## World API

| Endpoint | Purpose |
|---|---|
| `POST /scenario/{act1,act2,act3}` | Load a demo act |
| `GET /clock` · `POST /clock/advance` | Simulation clock; advancing fires events and deliveries |
| `GET /suppliers` · `GET /suppliers/{id}/prices` | Suppliers with today's price; price history (never future days) |
| `GET /orders` · `GET /inventory` · `GET /trucks` · `GET /travel-days` | State |
| `GET /allocation-state` | Everything the solver needs: remaining demand, reachable days per truck, issues |
| `POST /purchase-orders` · `POST /purchase-orders/{id}/cancel` | Buy at today's price; cancel if not shipped |
| `POST /shipments` · `POST /shipments/{id}/reassign` | Dispatch from the warehouse; re-route a truck |
| `POST /customers/{id}/reschedule-request` | Negotiate a delay → accept / reject / counter |
| `GET /events` · `GET /negotiations` · `GET /ledger` | History |
| `POST /reconcile` · `GET /ledger/invariants` | Keep POs and books consistent |
| `GET /scorecard` | Fulfillment, on-time rate, spend, delay, discounts, holding, waste, total |

---

## Repo layout

```
├── run_demo.py                    # runs the acts, decision log, comparison, lessons
├── PLAN.md                        # hackathon plan and team split
├── src/agent_wars_devfest/
│   ├── models.py                  # shared pydantic schemas
│   ├── world/                     # datagen.py, app.py (FastAPI), scenarios.py
│   ├── tools/                     # procurement.py, reallocation.py (B), ledger.py, baseline.py
│   └── agent/                     # tools.py, agent.py, prompts.py, memory.py, solvers.py
└── tests/                         # world, ledger, baseline, agent (64 tests)
```

---

## Known limitations

- **The LLM mode is wired up but hasn't been run end to end on a paid key.**
  - On the free tier (5 requests/min), every tool call is a request, so a single act takes many minutes.
  - The agent retries on 429 and uses the LLM only on days with events or issues.
  - If a turn still fails, that day falls back to the scripted turn, so the demo always completes.
- **The adapter works around gaps in B's reallocation MIP:**
  - Its objective charges every unmet tonne the same and only charges delay when a delivery happens, so in one big solve it starves overdue orders. The adapter solves it one priority and due-day tier at a time.
  - It only plans free trucks from the warehouse, so re-routing in-transit loads and asking for delays happen in the adapter.
- **Hedging isn't free:** in act 1, buying from reliable suppliers costs more than FCFS's "buy cheapest and re-buy on a miss", because the world makes misses cheap to recover from.

## Team

Narendra S · Niranjhan S U · Dharaneesh
