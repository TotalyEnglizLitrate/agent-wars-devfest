# Autonomous Supply Coordinator: Hackathon Plan

> **Track:** Logistics Handlers
> **Team:** 3 people (Person A: World, Person B: Solvers, Person C: Agent)
> **Build deadline:** **1:15 PM IST, Sat 17 Oct 2026** (about 40 minutes of build time from 12:35 PM)
> **Pitch:** *An autonomous supply coordinator for a regional distributor. It buys raw materials at the best total landed cost and, when disruptions hit, re-plans deliveries and negotiates with customers to keep everyone supplied.*
> **One-liner for judges:** *The solver finds the best plan; the agent makes it real.*

---

## 1. Core idea

Three connected capabilities, all running in one event loop:

1. **Price-aware procurement:** order from multiple suppliers using current and historical prices, supplier reliability, lead time, storage capacity and holding cost.
2. **Dynamic delivery reallocation:** when supply or demand changes, reassign trucks and loads across customer factories. Multi-drop and delays are allowed only where the customer agrees.
3. **Reconciliation on cancellation:** keep a ledger of POs, shipments, receipts, credits and fees consistent when buyers or suppliers cancel, including goods already in transit.

### Why an agent and not just an algorithm

The optimization is deterministic (PuLP). The LLM agent does what a solver can't:

- **Interprets messy events**, e.g. "Supplier B's truck broke down, can do 15t Thursday", and turns them into structured constraint changes.
- **Negotiates soft constraints**, e.g. asking the day-3 factory whether it can accept a one-day delay.
- **Runs the loop**: solve → identify needed asks → ask → update constraints from the reply → re-solve.
- **Explains trade-offs** and escalates to a human above a cost threshold.

The LLM never does the arithmetic. Solvers and the ledger are tools it calls.

---

## 2. Motivating example (sugar)

- A sugar supplier dispatches two 20t trucks, on day 1 and day 2. They arrive on day 2 and day 3.
- Factory A needs **30t on day 2**. Factory B needs sugar on **day 3**.
- Truck 1 alone (20t) can't cover Factory A.
- **Options the solver evaluates:**
  - **Split delivery + delay:** A gets truck 1 on day 2. Truck 2 is diverted to finish A's order on day 3. B is asked to accept day 4 from the next truck.
  - **Multi-drop:** truck 2 drops 10t at A and 10t at B, if B's quantity and time window allow.
- Model **divisible loads with a per-stop cost**, so the solver can choose either option.

---

## 3. Architecture

```
┌────────────────────────┐      HTTP       ┌──────────────────────────────┐
│  ADK Agent (Gemini)    │ ──────────────▶ │  Dummy World API (FastAPI)   │
│  - reads /events       │ ◀────────────── │  suppliers, prices, orders,  │
│  - calls tools         │                 │  trucks, clock, events,      │
│  - negotiates          │                 │  counterparty responses,     │
│  - escalates / logs    │                 │  ledger                      │
└──────────┬─────────────┘                 └──────────────────────────────┘
           │ function tools
           ▼
┌────────────────────────────────────────────┐
│ Deterministic tools                        │
│  - procurement_optimizer (PuLP)            │
│  - reallocation_optimizer (PuLP)           │
│  - reconciliation ledger (state machine)   │
│  - baseline_fcfs (for comparison)          │
└────────────────────────────────────────────┘
```

**Stack (matches the repo):** Python 3.14, `uv`, **google-adk** for the agent (Gemini, plain Python functions as tools), FastAPI + uvicorn, Faker (entities), numpy (prices), **PuLP** (bundled CBC solver, pure-Python wheel, so it works on 3.14), pydantic, httpx. State lives in in-memory dicts.

> **Why PuLP instead of OR-Tools:** OR-Tools ships native wheels and may not have one for Python 3.14 yet. PuLP installs everywhere and is enough for a model this size.

**Add the remaining deps once (Person A, 12:35).** faker, google-adk, fastapi and uvicorn are already in `pyproject.toml`:
```bash
uv add numpy pulp httpx pydantic
```

---

## 4. Data generation (seeded, reproducible)

- **Faker** for names, company names and cities only. Use `Faker.seed(42)`.
- **numpy** for prices: `np.random.default_rng(42)`, random walk + weekly seasonality + a supplier-specific offset.
- **Scale:** 1 commodity (sugar), **5 suppliers**, **6 customer factories**, **8 trucks**, **90 days** of price history, **7-day** planning horizon.
- **Built-in trade-off:** one supplier is cheapest but unreliable (misses about 30% of deliveries). Another is pricier but reliable.

### Schemas (the contract between A, B and C; agree on these at 12:35 and don't change them)

```python
Supplier:  id, name, city, capacity_t_per_day, lead_time_days,
           reliability (0-1), base_price_per_t
PricePoint: supplier_id, day, price_per_t
Customer:  id, name, city
Order:     id, customer_id, qty_t, requested_day,
           earliest_day, latest_acceptable_day,
           delay_penalty_per_day, accepts_split (bool),
           priority (1-3), status
Truck:     id, capacity_t (20), current_location, available_day
Shipment:  id, truck_id, supplier_id, stops[{customer_id, qty_t, day}],
           status (planned|in_transit|delivered|cancelled)
PurchaseOrder: id, supplier_id, qty_t, price_per_t, order_day,
               expected_day, status (open|shipped|received|cancelled)
LedgerEntry: id, ref_type, ref_id, kind (po|receipt|credit|fee|refund),
             amount, day, note
Inventory: warehouse_id, qty_t, capacity_t, holding_cost_per_t_day
TravelDays: matrix[location][location] -> int   # no road routing
```

Put these in `src/agent_wars_devfest/models.py` as pydantic models first, so everyone imports the same definitions.

---

## 5. Dummy API (FastAPI + uvicorn)

```
GET  /clock                                  POST /clock/advance   {days: 1}
GET  /suppliers                              GET  /suppliers/{id}/prices?days=90
GET  /inventory                              GET  /orders?status=open
POST /purchase-orders                        POST /purchase-orders/{id}/cancel
GET  /shipments                              POST /shipments/{id}/reassign
GET  /events?since=<day>                     # cancellations, delays, short-ships, price spikes
POST /customers/{id}/reschedule-request      # -> {accept | reject | counter{discount_pct}}
GET  /ledger
POST /scenario/{name}                        # load demo scenario: act1 | act2 | act3
```

- **Simulation clock:** `/clock/advance` fires the scheduled events for that day.
- **Counterparty replies:** deterministic per scenario. Act 2 is scripted so Factory B **counters** first.
- Run with `uv run uvicorn agent_wars_devfest.world.app:app --reload --port 8000`.

---

## 6. Deterministic tools

### 6.1 Procurement optimizer
- **Decision:** `x[s, d]` = tons bought from supplier `s` on day `d`.
- **Minimize:** price + holding cost + expected shortfall penalty, where expected shortfall = (1 − reliability) × qty × penalty.
- **Subject to:** supplier daily capacity, lead time, warehouse capacity, inventory balance meeting demand.
- **Price signal:** forecast = 7-day moving average + trend from the last 14 days.

### 6.2 Reallocation optimizer
- **Decision:** `y[t, c, d]` = tons truck `t` delivers to customer `c` on day `d` (divisible), plus binary `stop[t, c]`.
- **Minimize:** unmet demand × large penalty + delay days × `delay_penalty_per_day` + stops × per-stop cost + transport cost.
- **Subject to:** truck capacity (20t), arrival day ≥ dispatch day + travel days, delivery within `[earliest_day, latest_acceptable_day]`, delays allowed only if `customer_flex_confirmed`.
- **Output:** the plan **plus a list of `asks`**, i.e. delays or splits that need customer consent.

### 6.3 Reconciliation ledger (state machine)
- **Buyer cancels:**
  - Surplus goes back to open demand (re-run the reallocation optimizer).
  - Cancel or shrink the upstream PO if it hasn't shipped.
  - Record the cancellation fee or credit.
  - If goods are in transit, reroute or hold at the warehouse.
- **Supplier cancels or short-ships:**
  - Re-source from an alternate supplier at today's price (procurement optimizer).
  - Trigger reallocation and negotiation.
  - Record credit or penalty against the supplier.
- **Invariant check:** received qty + in-transit qty + cancelled qty = ordered qty for each PO. Expose this as a tool so the agent can verify.

### 6.4 Baseline
- **First-come-first-served:** no reallocation, no negotiation, buy from the cheapest supplier at today's price.
- Run it on the same seed and report the comparison.

---

## 7. Agent (Google ADK)

One `LlmAgent` with Gemini. Each tool is a plain Python function with type hints and a docstring; ADK builds the tool schema from those. Each function calls the World API over httpx or calls a solver directly.

**Tools → what each one calls.** The World API is built and tested, so these are real endpoints:

| Agent tool | Calls | Notes |
|---|---|---|
| `get_events` | `GET /events?since=<day>` | Each event has a messy `message` (for the LLM) plus a structured `payload` |
| `get_orders` | `GET /orders?status=open` | |
| `get_inventory` | `GET /inventory` | |
| `get_price_history` | `GET /suppliers`, `GET /suppliers/{id}/prices?days=90` | Never returns future days |
| `get_allocation_state` | `GET /allocation-state` | Remaining demand, every truck's reachable days per customer, warehouse stock, and an `issues` list. **Feed this to the reallocation solver** |
| `run_procurement_optimizer` | `tools/procurement.py` (B) | Pass learned reliability from §7.1 |
| `run_reallocation_optimizer` | `tools/reallocation.py` (B) | Input is `/allocation-state` |
| `create_purchase_order` | `POST /purchase-orders {supplier_id, qty_t}` | The price is always that day's supplier price; the caller can't set it |
| `cancel_purchase_order` | `POST /purchase-orders/{id}/cancel` | 409 once the PO has shipped |
| `dispatch_from_warehouse` | `POST /shipments {truck_id, stops}` | Free truck + warehouse stock |
| `reassign_shipment` | `POST /shipments/{id}/reassign {stops, truck_id?}` | Can't carry more than is on board; leftover goes back to the warehouse; stop day must be after today |
| `request_reschedule` | `POST /customers/{id}/reschedule-request {order_id, new_day, discount_pct}` | `discount_pct` is 0–50. Act 2's Factory B counters at 5% first |
| `reconcile` | `POST /reconcile` | Returns `actions`, `followups` (what to do next), `net_spend`, `invariants` |
| `check_ledger_invariants` | `GET /ledger/invariants` | `{ok, violations[]}` |
| `escalate_to_human` | local | |
| `advance_clock` | `POST /clock/advance {days}` | `days` is 1–30 |
| `get_scorecard` | `GET /scorecard` | Same yardstick as the FCFS baseline |
| `recall_lessons`, `save_lesson` | `agent/memory.py` | See §7.1 |

**Loop (described in the system prompt; `run_demo.py` drives one tick per act):**
```
for each tick:
    events = get_events()
    for event in events:
        interpret -> constraint changes
    plan, asks = run_reallocation_optimizer(constraints)
    for ask in asks:
        reply = request_reschedule(ask)
        if reply in (reject, counter): update constraints; re-solve (max 3 rounds)
    if plan.cost_delta > THRESHOLD: escalate_to_human(plan, explanation)
    commit plan (reassign_shipment / create_po / cancel_po)
    reconcile(); check_ledger_invariants()
    log reasoning to decision log
```

**Guardrails (enforce in code, not only in the prompt):**
- Human approval above a ₹ threshold: `escalate_to_human` returns "approved" in demo mode and logs loudly.
- At most 3 negotiation rounds per ask: `request_reschedule` keeps a counter and refuses a 4th round.
- The agent never edits the ledger directly; it only calls `reconcile`.

**Fallback if the LLM is flaky during the demo:** `run_demo.py --scripted` calls the same tools in a fixed order, so the solver and ledger story still works.

### 7.1 Learning loop: the agent gets better across runs (Person C)

Within one run the agent already gets feedback: counters force a re-solve, `reconcile` reports what's off, and `/scorecard` grades the run. What's missing is carry-over between runs. Add a small **lessons memory**:

- **File:** `agent/memory.py`. It stores to `.agent_memory/lessons.json` (gitignored), which survives restarts and scenario reloads.
- **After each act, `record_run(act)`** reads `/scorecard`, `/events`, `/purchase-orders` and `/negotiations`, then derives lessons **in code, not with the LLM**, so the numbers are always correct:
  - **Supplier reliability seen in practice:** count API POs vs `short_ship` events per supplier, summed across runs. Blend with the listed figure: `(on_time + 5 × listed) / (n + 5)`. Pass it to `run_procurement_optimizer` as a `reliability_override: dict[str, float]` (**B: accept this argument**).
  - **Customer haggling:** each customer's last counter discount, e.g. "C2 countered a delay at 5%: offer 5% up front." This saves a negotiation round.
  - **From the scorecard:** wasted tonnes ("re-route in-transit loads after a `buyer_cancel`; follow `reconcile`'s followups"), and late deliveries if `on_time_rate` < 95%.
  - **Run history:** act, total cost, wasted tonnes, e.g. "last act1 run cost ₹49.4L: beat it".
  - Key each lesson by (kind, subject) so it's **updated, not duplicated**. Put at most 10 in the prompt.
- **`save_lesson(text)` tool:** the agent can add its own free-text lessons (plain text, max 20 kept).
- **Prompt injection:** pass a function as the instruction: `LlmAgent(instruction=build_instruction)`, where `def build_instruction(ctx: ReadonlyContext) -> str: return BASE_PROMPT + lessons_block()`. *Checked against the installed google-adk:* `instruction` accepts a sync or async callable taking `ReadonlyContext`. A callable skips ADK's `{state}` templating, so don't put `{var}` placeholders in `BASE_PROMPT`.
- **Guardrails:** lessons are advice only and never override §7 guardrails. Run `rm -rf .agent_memory` before a clean demo.
- **Test:** make `extract_lessons(report) -> list[Lesson]` a pure function and unit-test it with the World `TestClient`, the same way `tests/test_baseline.py` does.
- **Demo moment:** wipe memory, run act 1 twice. Run 1 buys mostly from cheap S1 and gets short-shipped. Run 2's prompt shows "S1 observed reliability 0.44", the purchase split shifts toward other suppliers, and the scorecard cost drops. Show the lessons block on screen.

---

## 8. Demo script (≈5 min, three acts)

1. **Day 0: procurement.** The agent compares five suppliers' price trends and reliability, then splits the buy. Example output: *"Bought 60% from the cheapest supplier and 40% from the reliable one, because the cheap one missed 3 of its last 10 deliveries."*
2. **Day 2: shortage (showpiece).** A supplier short-ships. The solver proposes diverting truck 2 and delaying Factory B. The agent asks, B **counters** with a discount, the agent re-solves with the new constraint, confirms, and commits.
3. **Day 3: cancellation.** A customer cancels. The agent reconciles the ledger, reallocates surplus to a waiting order, cancels an unshipped upstream PO, and shows the net cost impact.

**Closing slide:** **total ₹ cost and wasted tonnes compared with the FCFS baseline** on the same seeded data, from `GET /scorecard`, plus **run 2 vs run 1 cost** from §7.1.

> Fulfillment rate is not the headline: FCFS already hits 100% in every act, because S1's 1-day lead time lets it re-buy after each miss. Baseline today (`uv run python -m agent_wars_devfest.tools.baseline`): act1 ₹49.4L, act2 ₹32.5L, act3 ₹26.9L with **25t wasted** on the cancelled order.

---

## 9. Team split (3 people, all start at 12:35)

### Person A: World ✅ done (all tested, `uv run pytest`)
Owns everything the agent talks *to*.
- `models.py` (shared pydantic schemas). **Push this first, by 12:40**, so B and C can import it.
- `world/datagen.py`: seeded Faker + numpy data.
- `world/app.py`: FastAPI server, all endpoints, sim clock.
- `world/scenarios.py`: act1/act2/act3 event scripts and scripted counterparty replies (B counters in act 2).
- Also built: `/allocation-state`, `/scorecard`, seeded supplier misses, warehouse dispatch, plus `tools/ledger.py` and `tools/baseline.py` (taken over from B).

### Person B: Solvers
Owns the deterministic math. **The agent's interface is fixed in the `agent/solvers.py` docstring**: expose `solve(...)` in `tools/reallocation.py` / `tools/procurement.py` with those signatures and the agent picks them up automatically. Until then, a greedy fallback runs and its output carries `"solver": "greedy-fallback"`.
- `tools/procurement.py` and `tools/reallocation.py`: PuLP models. Develop against mock dicts shaped like `models.py` until the API is live.
- ~~`tools/ledger.py`~~ and ~~`tools/baseline.py`~~: done by A. Focus on the two solvers.
- `run_procurement_optimizer` takes `reliability_override: dict[str, float]` from §7.1.

### Person C: Agent ✅ built (taken over by A; `uv run python run_demo.py --scripted`)
Owns everything the LLM does. LLM mode needs `GOOGLE_API_KEY` and hasn't been run live yet. Without a key, `run_demo.py` falls back to the scripted agent automatically.
- `agent/tools.py`: ADK function tools (httpx calls to the World API + solver calls). Stub return values first and swap in real calls at 12:55.
- `agent/agent.py` + `agent/prompts.py`: ADK `LlmAgent`, system prompt, negotiation round counter, escalation, decision log.
- `agent/memory.py`: lessons memory and learning loop (§7.1).
- `run_demo.py`: runs acts 1–3, prints the decision log and the comparison, with a `--scripted` fallback.

### Timeline

| Time (IST) | Person A (World) | Person B (Solvers) | Person C (Agent) | Checkpoint |
|---|---|---|---|---|
| 12:35 – 12:45 | `uv add` deps, `models.py`, datagen, FastAPI with all endpoints + clock | Reallocation optimizer on the sugar example (mock data) | ADK agent skeleton, all 14 tools stubbed, system prompt | **12:45:** `curl /suppliers /orders /events` return data |
| 12:45 – 12:55 | act2 scenario + scripted counterparty replies | Ledger state machine + invariants; procurement optimizer | Negotiation loop (max 3 rounds), escalation, decision log | **12:55:** sugar example solves with an `asks` list; invariants pass |
| 12:55 – 13:05 | act1 + act3 scenarios | FCFS baseline | Swap stubs for real API + solver calls | **13:05:** Act 2 runs end to end from `/scenario/act2` |
| 13:05 – 13:12 | Help debug integration | Baseline comparison numbers, closing slide | `run_demo.py` for all 3 acts + `--scripted` fallback | All three acts run on one seed |
| 13:12 – 13:15 | Freeze, rehearse pitch | Freeze, rehearse pitch | Freeze, rehearse pitch | Code frozen at **1:15 PM** |

**Working agreement:**
- Each person commits only their own files, so there are no merge conflicts. `models.py` belongs to A; ask before changing it.
- Pull (`git pull --rebase`) and push at every checkpoint. **No force-pushes to master.**
- If someone is blocked more than 3 minutes, say so out loud.

### If you fall behind, protect in this order
1. Act 2 (shortage → solve → negotiate → re-solve), the showpiece
2. Act 3 (cancellation reconciliation)
3. Act 1 (procurement)
4. Baseline comparison number

---

## 10. Repo layout (uv src layout)

```
.
├── PLAN.md
├── pyproject.toml              # uv; deps: faker google-adk fastapi uvicorn numpy pulp httpx pydantic
├── run_demo.py                 # C: runs acts 1-3, prints decision log + comparison
└── src/agent_wars_devfest/
    ├── models.py               # A: shared pydantic schemas (§4)
    ├── world/
    │   ├── datagen.py          # A: Faker + numpy, seeded
    │   ├── app.py              # A: FastAPI server
    │   └── scenarios.py        # A: act1/act2/act3 event scripts + counterparty replies
    ├── tools/
    │   ├── procurement.py      # B
    │   ├── reallocation.py     # B
    │   ├── ledger.py           # A (done)
    │   └── baseline.py         # A (done)
    └── agent/
        ├── tools.py            # C: ADK function tools
        ├── agent.py            # C: LlmAgent + loop
        ├── memory.py           # C: lessons memory (§7.1)
        └── prompts.py          # C
```

**Run:**
```bash
uv run uvicorn agent_wars_devfest.world.app:app --reload --port 8000   # terminal 1
uv run python run_demo.py                                              # terminal 2
```
Set `GOOGLE_API_KEY` in your environment for Gemini.
