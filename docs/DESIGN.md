# Design: Autonomous Supply Coordinator

This document explains how the system works and why it's built this way. For setup and running, see the [README](../README.md).

---

## 1. Problem

A regional distributor buys sugar from 5 mills and delivers it by truck to 6 food factories. Every day, someone has to:

- **Buy** at the best *landed* cost: price, plus storage, plus the risk that a cheap mill fails to deliver.
- **Re-plan deliveries** when a mill short-ships or a factory cancels, including trucks already on the road.
- **Negotiate** with factories when a delivery has to be late.
- **Keep the books consistent:** purchase orders, receipts, supplier credits, cancellation fees.

### Why an agent, and not just an optimizer

The arithmetic is deterministic, so it belongs in a solver. A solver can't do the rest of the job:

| Solver can't… | The agent does it |
|---|---|
| Read *"truck 2 had a loading problem, only 12t went on"* | Interprets free-text events into what changed |
| Ask a factory to accept a delay, or handle *"only for 5% off"* | Negotiates, then re-solves around the agreed terms |
| Decide what to ask, in what order, and when to commit | Runs the solve → ask → re-solve → commit → reconcile loop |
| Explain itself or know when to stop | Logs its reasoning and escalates big spends to a human |

**The LLM never does arithmetic.** Solvers and the ledger are tools it calls. *The solver finds the best plan; the agent makes it real.*

---

## 2. Motivating example (act 2)

- Mill S1 dispatches two 20t trucks: truck 1 arrives day 2, truck 2 arrives day 3.
- Factory A needs **30t on day 2**; Factory B needs **20t on day 3**, with no slack.
- On day 2, S1 reports *"only 12t went on"* truck 2.

That leaves 8t short by day 3 whatever we do. The plan:
1. Cover Factory A from warehouse stock (A has the higher priority).
2. Ask Factory B to accept the missing 8t on day 4, from a replenishment arriving day 3.
3. B **counters: "5% off"**. The agent accepts, re-solves, commits, and reconcile books an 8t credit against S1.

---

## 3. System overview

```
┌────────────────────────┐   tool calls   ┌──────────────────────────────┐
│  ADK agent (Gemini)    │ ─────────────▶ │  World API (FastAPI)         │
│  instruction = prompt  │ ◀───────────── │  the simulated "reality"     │
│    + learned lessons   │                └──────────────────────────────┘
│  23 function tools     │
└──────────┬─────────────┘
           ▼
┌─────────────────────────────────────────────────────────────┐
│ Deterministic tools                                         │
│  procurement LP · reallocation MIP · solver adapter ·       │
│  reconciliation ledger · FCFS baseline                      │
└─────────────────────────────────────────────────────────────┘
```

**One turn = one simulated day.** `run_demo.py` loads an act, gives the agent a turn, advances the clock, and repeats until day 9. Then it scores the run and records lessons. Each turn:

1. Read today's events and work out what physically changed.
2. **Procure:** the LP decides what to buy today, or which unshipped POs to cancel.
3. **Reallocate:** the solver plans truck moves and returns `asks`, i.e. delays that need customer consent.
4. Negotiate each ask (max 3 rounds), re-solving after every reply.
5. Commit the plan once no asks remain.
6. Reconcile the books, act on the follow-ups, and check the invariants.
7. Log one or two sentences of reasoning for the decision log.

---

## 4. The world (`world/`)

A FastAPI server that plays the role of reality. It can run in-process (the default for demos and tests) or as a real server.

**Data (`datagen.py`, seed 42).** Faker supplies names and cities. Prices are numpy-generated: random walk + weekly seasonality + a per-supplier offset, over 90 days of history plus the planning week. The setup is 5 suppliers, 6 factories, 8 trucks of 20t, and a symmetric travel-days matrix.

| Supplier | Price | Reliability | Lead time | Role |
|---|---|---|---|---|
| S1 | cheapest | 0.70 | 1 day | cheap but flaky (always "3 misses in the last 10") |
| S2 | most expensive | 0.98 | 1 day | reliable |
| S3–S5 | in between | 0.85–0.93 | 2–3 days | alternatives |

**Scenarios (`scenarios.py`).** Each act starts one day before its key event, so the first clock advance fires it. Travel days that the scripted stops rely on are pinned, so the scripts and the travel matrix always agree; this is tested.

**Physics:**
- Events change the world: a short-ship trims the truck, a cancellation cancels the order.
- POs placed through the API short-ship (60% delivered) with probability 1 − reliability. This is seeded per PO, so every run sees the same misses.
- Scripted POs never miss at random, so the demo script stays exact.
- Deliveries happen when the clock advances, so a new stop must be dated after today.

**Negotiation.**
1. Scripted replies come first. In act 2, Factory B counters at 5%, unless our offer already meets the counter.
2. After that, a default policy applies:
   - **Reject** if the new day is far too late.
   - **Counter again** if we offer less than the customer's last counter.
   - **Accept** otherwise.
3. Acceptance widens the order's window and records the discount.

**Input validation.** The agent is an untrusted client:
- Quantities must be positive, discounts 0–50%, clock advances 1–30 days.
- PO prices are always the supplier's price that day; the caller can't set them.
- A re-routed truck can't carry more than is on board.

**`/allocation-state`** precomputes everything a solver needs: remaining and uncovered demand per order, each truck's load, its earliest arrival at every customer, warehouse stock and incoming POs, free trucks, and a short plain-English `issues` list limited to orders due within 2 days.

**`/scorecard`** is the one yardstick for the agent and the baseline alike:
`total = procurement spend + delay penalties + discounts given + holding cost + unmet × ₹10,000/t`,
plus fulfillment rate, on-time rate and **wasted tonnes** (delivered to customers with no open order).

---

## 5. Deterministic tools

### Procurement LP (`tools/procurement.py`)
- **Decision:** tonnes bought from each supplier on each day of the horizon.
- **Minimize:** forecast price + holding cost + (1 − reliability) × shortfall penalty.
- **Constraints:** daily supplier capacity, lead time, warehouse capacity, inventory balance against demand.
- **Price forecast:** 7-day moving average + 14-day linear trend.

### Reallocation MIP (`tools/reallocation.py`)
- **Decision:** tonnes each free truck delivers to each order on each day (continuous), plus a binary stop variable.
- **Minimize:** unmet × ₹50,000 + delay days × the order's penalty + ₹500 per stop.
- **Constraints:** truck capacity, warehouse stock, and each order's delivery window (widened once the customer has agreed to a delay).
- Solved with HiGHS through PuLP.

### Solver adapter (`agent/solvers.py`)
The agent calls the solvers through an adapter. It adds what the agent needs around them:

**Reallocation, in three phases:**
1. **Re-route in-transit loads.** Loads heading to customers who cancelled go to any open order they can still reach in time, otherwise back to the warehouse. The MIP only plans free trucks.
2. **MIP for warehouse dispatch**, solved one *(priority, due day)* tier at a time. The MIP's objective prices every unmet tonne the same and charges delay only when a delivery happens. In a single solve it would therefore starve an overdue or high-priority order in favour of one that isn't due yet.
3. **Delay asks.** Whatever is still short becomes an ask against incoming POs. The MIP only plans inside customer windows, so it never asks.

**Procurement:**
- The LP gets the *real remaining need*: uncovered tonnes, net of stock, incoming POs and re-routable loads.
- Each order's due day is shifted earlier by the truck leg, since the LP balances stock at the warehouse.
- Only today's buys are executed; later days are re-planned tomorrow with fresh prices.
- PO cancellation (when there's a surplus against *all* remaining demand) stays greedy, because the LP doesn't cancel.

If a solver is missing or a solve fails, a greedy fallback runs on its own. Every result carries `"solver"`, so the decision log shows which path ran.

### Reconciliation ledger (`tools/ledger.py`)
It runs inside the world as `POST /reconcile` and `GET /ledger/invariants`. The agent has no way to write ledger entries directly.

**Invariants:**
- **Physical (direct POs):** tonnes on linked trucks + cancelled tonnes = tonnes ordered.
- **Physical (received POs):** received = ordered − cancelled.
- **Money:** the sum of a PO's ledger entries = (ordered − cancelled) × price.
- **Cancellations:** every cancelled customer order has a cancellation fee booked.

**What reconcile does:**
- Records unrecorded short-ships as cancelled tonnes.
- Posts the money difference as a credit or charge.
- Books cancellation fees.
- Returns follow-ups, e.g. *"SH2 still carries 20t to C3: reroute it"* or *"unshipped POs you could cancel: PO3"*.

It's idempotent: a second run books nothing new.

### FCFS baseline (`tools/baseline.py`)
First-come-first-served:
- No reallocation and no negotiation.
- Buys the shortfall from the cheapest supplier at today's price.
- Dispatches warehouse stock in order of arrival.

It drives the same World API in-process, so it faces the same seeded misses and is scored by the same `/scorecard`.

---

## 6. Agent (`agent/`)

**ADK setup:**
- One `LlmAgent` with 23 plain-Python function tools; ADK builds each tool's schema from its type hints and docstring.
- `instruction` is a callable that returns the base prompt plus the current lessons, so new lessons apply on the very next turn.
- Default model: `gemini-3.5-flash-lite` (see §8). Override it with `AGENT_MODEL`.
- Retries on 429/503 with backoff.

**Tools:**

| Group | Tools |
|---|---|
| Read | `get_clock`, `get_events`, `get_orders`, `get_inventory`, `get_suppliers` (with observed reliability), `get_price_history`, `get_allocation_state`, `get_scorecard` |
| Solve | `run_procurement_optimizer`, `run_reallocation_optimizer` |
| Act | `commit_plan`, `create_purchase_order`, `cancel_purchase_order`, `dispatch_from_warehouse`, `reassign_shipment`, `request_reschedule`, `advance_clock` |
| Books | `reconcile`, `check_ledger_invariants` |
| Oversight & memory | `escalate_to_human`, `log_decision`, `recall_lessons`, `save_lesson` |

**Guardrails, enforced in code rather than only in the prompt:**
- Anything above **₹15 lakh** (a single PO, or a plan's delay cost) is refused until `escalate_to_human`, which auto-approves in demo mode and is logged.
- **Max 3 negotiation rounds** per order.
- `commit_plan` **refuses while asks are unresolved**.
- **No direct ledger writes**.
- Tool errors come back as data, e.g. `{"error": "...", "status": 409}`, so the LLM can read them and correct itself.

**Scripted turn.** `scripted_tick()` calls the same tools in the order the prompt describes. It serves as the `--scripted` demo mode, and as an automatic per-turn fallback if a Gemini call fails, so a demo always completes.

---

## 7. Learning loop (`agent/memory.py`)

After each act, `record_run` reads `/scorecard`, `/events`, `/purchase-orders` and `/negotiations`. `extract_lessons` then derives lessons **in code, not by the LLM**, so the numbers in them are always true:

- **Observed supplier reliability**, built up across runs from the agent's own POs: `(on_time + 5 × listed) / (n + 5)`. The procurement LP receives the raw observed rate and blends it with the listed figure itself.
- **Customer haggling profiles:** each customer's last counter. The next run opens with that discount and saves a round.
- **Waste and lateness lessons** from the scorecard, plus **best cost so far** per act.

Lessons are keyed by (kind, subject), so they're updated rather than duplicated, and stored in `.agent_memory/lessons.json`. The agent can also add its own via `save_lesson`.

**What changes on run 2 (measured, scripted):**
- In act 2, the agent offers Factory B 5% up front and B accepts in one round instead of two.
- In act 1, observed S1 reliability falls from 0.70 to about 0.6, and buying shifts toward reliable S3. On-time delivery improves from 63% to 87%, but total cost rises by about ₹2.5L, because reliability costs money in this world.

---

## 8. Design decisions and trade-offs

- **PuLP + HiGHS rather than OR-Tools:** pure-Python install on Python 3.14; plenty for models this size.
- **Lessons computed in code:** if the LLM summarized its own runs it could invent "S1 missed 7 times". Counting in code keeps the learning claim checkable on the scorecard.
- **LLM only on eventful days:** turns with new events or open issues go to Gemini; quiet days run the scripted turn. This saves requests and spends the LLM where judgment matters.
- **`gemini-3.5-flash-lite` as the default model:** on the free tier, `gemini-3.5-flash`/`3.6-flash` allow 5 requests/minute, and every tool call is a request. Flash-lite answered 10/10 tool-calling requests in 11s without hitting a limit. All three acts were run live on it end to end.
- **Holding cost is charged:** without it, buying everything up front was free and just-in-time buying couldn't win.
- **Act 2's Factory B has no slack**, so its delay genuinely needs consent; otherwise the showpiece negotiation never happens.
- **Buying horizon of 4 days:** the shortest window that matched FCFS on act 1 lateness. It also preserves the act 2 negotiation, a demo-motivated choice.
- **Honest results:** FCFS fills 100% of orders because S1's 1-day lead time lets it re-buy after every miss, so the comparison is cost and waste. In act 1 the agent's hedging costs slightly more than FCFS.

---

## 9. Testing

`uv run pytest` runs 64 tests:

| File | Covers |
|---|---|
| `test_world.py` | Seeded reproducibility, no future prices, every act's story, input validation, dispatch rules, scorecard |
| `test_ledger.py` | Short-ship credits, cancellation fees, idempotency, re-routing that can't create sugar or erase history |
| `test_baseline.py` | FCFS is deterministic, finishes every act, and wastes the cancelled load |
| `test_agent.py` | ADK builds every tool schema, guardrails, the act 2 negotiation, act 3 zero waste, learning (learned discount saves a round, observed reliability drops), pure lesson extraction |

Several tests were confirmed to fail when the fix they protect is reverted (scenario travel days, delivered-stop history).
