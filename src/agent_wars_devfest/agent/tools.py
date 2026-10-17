"""ADK function tools (PLAN.md §7). Each is a plain function; ADK builds the schema from hints + docstring.

Guardrails live here, in code: max 3 negotiation rounds per order, human approval above a ₹
threshold, and no direct ledger writes (only `reconcile`).
"""

import functools
import os

import httpx

from agent_wars_devfest.agent import memory, solvers

WORLD_URL = os.environ.get("WORLD_URL", "http://localhost:8000")
MAX_ROUNDS = 3
APPROVAL_THRESHOLD_INR = 1_500_000  # single PO value, or a plan's delay cost, above this needs a human

_client: httpx.Client = httpx.Client(base_url=WORLD_URL, timeout=30)
_rounds: dict[str, int] = {}
_approved_inr: float = 0.0
_last_plan: dict | None = None
decision_log: list[dict] = []


def use_client(client) -> None:
    """Point the tools at another world, e.g. a FastAPI TestClient for in-process runs."""
    global _client
    _client = client


def reset_run_state() -> None:
    global _approved_inr, _last_plan
    _rounds.clear()
    _approved_inr, _last_plan = 0.0, None
    decision_log.clear()


def _call(method: str, path: str, **kw):
    r = _client.request(method, path, **kw)
    if r.status_code >= 400:
        try:
            detail = r.json().get("detail", r.text)
        except ValueError:
            detail = r.text
        return {"error": str(detail), "status": r.status_code}
    return r.json()


def _today() -> int:
    return _client.get("/clock").json()["day"]


def _log(kind: str, text: str) -> None:
    decision_log.append({"day": _today(), "kind": kind, "text": text})
    if os.environ.get("AGENT_VERBOSE"):
        print(f"  [day {decision_log[-1]['day']}] {kind}: {text}")


def logged(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        out = fn(*args, **kwargs)
        shown = ", ".join([*map(repr, args), *(f"{k}={v!r}" for k, v in kwargs.items())])
        err = f" -> ERROR {out['error']}" if isinstance(out, dict) and "error" in out else ""
        _log("tool", f"{fn.__name__}({shown}){err}")
        return out
    return wrapper


def load_scenario(act: str) -> dict:
    """Not an LLM tool: the demo runner loads acts."""
    reset_run_state()
    return _call("POST", f"/scenario/{act}")


# ---------- read ----------

@logged
def get_clock() -> dict:
    """Current simulation day and scenario name."""
    return _call("GET", "/clock")


@logged
def get_events(since_day: int = 0) -> list | dict:
    """Events that have happened since `since_day`: short-ships, delays, cancellations, price spikes.

    Each has a free-text `message` written by a human (interpret it) and a structured `payload`.
    """
    return _call("GET", "/events", params={"since": since_day})


@logged
def get_orders(status: str = "open") -> list | dict:
    """Customer orders. status: open | fulfilled | cancelled."""
    return _call("GET", "/orders", params={"status": status})


@logged
def get_inventory() -> dict:
    """Warehouse stock, capacity and holding cost."""
    return _call("GET", "/inventory")


@logged
def get_suppliers() -> list | dict:
    """All suppliers with today's price, lead time, capacity, listed reliability and last 10 deliveries.

    Also includes `observed_reliability` where we have our own delivery history.
    """
    sups = _call("GET", "/suppliers")
    if isinstance(sups, dict):
        return sups
    obs = memory.observed_reliability(memory.load(), sups)
    return [s | ({"observed_reliability": obs[s["id"]]} if s["id"] in obs else {}) for s in sups]


@logged
def get_price_history(supplier_id: str, days: int = 14) -> dict:
    """Recent daily prices (INR per tonne) for one supplier, plus a 7-day-MA + trend forecast."""
    hist = _call("GET", f"/suppliers/{supplier_id}/prices", params={"days": days})
    if isinstance(hist, dict):
        return hist
    return {"prices": [round(p["price_per_t"]) for p in hist],
            "forecast_next_days": round(solvers.forecast_price(hist, 2))}


@logged
def get_allocation_state() -> dict:
    """Remaining demand per order, every truck's load and reachable days, warehouse stock, and an `issues` list."""
    return _call("GET", "/allocation-state")


@logged
def get_scorecard() -> dict:
    """How this run is doing: fulfillment, on-time rate, spend, delay/discount cost, waste, total cost."""
    return _call("GET", "/scorecard")


# ---------- solve ----------

def _state_with_open_pos() -> dict:
    state = _client.get("/allocation-state").json()
    state["open_pos"] = {p["id"]: p["qty_t"] for p in _client.get("/purchase-orders").json() if p["status"] == "open"}
    return state


@logged
def run_procurement_optimizer() -> dict:
    """Decide what to buy (or which unshipped POs to cancel) at the best total landed cost.

    Weighs forecast price against expected shortfall from unreliable suppliers, using reliability
    we've actually observed where we have it. Returns `purchases` and `cancel_pos`; does not buy.
    """
    sups = _client.get("/suppliers").json()
    hist = {s["id"]: _client.get(f"/suppliers/{s['id']}/prices", params={"days": 14}).json() for s in sups}
    override = memory.observed_reliability(memory.load(), sups)
    plan = solvers.procure(sups, hist, _state_with_open_pos(), override)
    return plan | ({"reliability_used": override} if override else {})


@logged
def run_reallocation_optimizer() -> dict:
    """Plan truck moves for open orders. Returns `actions` (commit with commit_plan), `asks` (delays that
    need customer consent via request_reschedule), `deferred`, `unmet`, and a cost estimate.

    Re-run it after every customer reply: an accepted delay widens that order's window.
    """
    global _last_plan
    travel = _client.get("/travel-days").json()
    _last_plan = solvers.reallocate(_client.get("/allocation-state").json(), travel)
    return _last_plan


# ---------- act ----------

def _needs_approval(amount: float) -> dict | None:
    if amount > APPROVAL_THRESHOLD_INR and amount > _approved_inr:
        return {"error": f"₹{amount:,.0f} is above the ₹{APPROVAL_THRESHOLD_INR:,.0f} approval threshold. "
                         f"Call escalate_to_human first."}
    return None


@logged
def commit_plan() -> dict:
    """Execute the actions of the last reallocation plan. Refuses if it still has unresolved asks."""
    global _last_plan
    if not _last_plan:
        return {"error": "no plan: call run_reallocation_optimizer first"}
    if _last_plan["asks"]:
        return {"error": "plan has asks awaiting consent: request_reschedule, then re-run the optimizer"}
    if blocked := _needs_approval(_last_plan["cost_estimate"]):
        return blocked
    results = []
    for a in _last_plan["actions"]:
        if a["tool"] == "reassign_shipment":
            results.append(_call("POST", f"/shipments/{a['shipment_id']}/reassign", json={"stops": a["stops"]}))
        else:
            results.append(_call("POST", "/shipments", json={"truck_id": a["truck_id"], "stops": a["stops"]}))
    _last_plan = None
    errors = [r["error"] for r in results if isinstance(r, dict) and "error" in r]
    return {"committed": len(results) - len(errors), "errors": errors}


@logged
def create_purchase_order(supplier_id: str, qty_t: float) -> dict:
    """Buy `qty_t` tonnes from a supplier at today's price. Large orders need escalate_to_human first."""
    sup = next((s for s in _client.get("/suppliers").json() if s["id"] == supplier_id), None)
    if sup and (blocked := _needs_approval(qty_t * sup["price_today"])):
        return blocked
    return _call("POST", "/purchase-orders", json={"supplier_id": supplier_id, "qty_t": qty_t})


@logged
def cancel_purchase_order(po_id: str) -> dict:
    """Cancel a PO that hasn't shipped yet (refunded in full). Fails once it has shipped."""
    return _call("POST", f"/purchase-orders/{po_id}/cancel")


@logged
def dispatch_from_warehouse(truck_id: str, customer_id: str, qty_t: float, day: int) -> dict:
    """Send a free truck from the warehouse with `qty_t` tonnes to arrive at `customer_id` on `day`."""
    return _call("POST", "/shipments", json={"truck_id": truck_id,
                                             "stops": [{"customer_id": customer_id, "qty_t": qty_t, "day": day}]})


@logged
def reassign_shipment(shipment_id: str, customer_id: str, qty_t: float, day: int) -> dict:
    """Re-route a truck so it drops `qty_t` at `customer_id` on `day` (single drop; use commit_plan for
    multi-drop). Anything left on board goes back to the warehouse."""
    return _call("POST", f"/shipments/{shipment_id}/reassign",
                 json={"stops": [{"customer_id": customer_id, "qty_t": qty_t, "day": day}]})


@logged
def request_reschedule(customer_id: str, order_id: str, new_day: int, discount_pct: float = 0.0) -> dict:
    """Ask a customer to accept delivery on `new_day`, optionally offering a discount (0-50%).

    Reply is accept | reject | counter (with the discount they want). At most 3 rounds per order.
    """
    _rounds[order_id] = _rounds.get(order_id, 0) + 1
    if _rounds[order_id] > MAX_ROUNDS:
        return {"error": f"negotiation limit ({MAX_ROUNDS} rounds) reached for {order_id}: escalate_to_human"}
    reply = _call("POST", f"/customers/{customer_id}/reschedule-request",
                  json={"order_id": order_id, "new_day": new_day, "discount_pct": discount_pct})
    if isinstance(reply, dict) and "decision" in reply:
        _log("negotiation", f"{customer_id}/{order_id} day {new_day} @ {discount_pct:g}% -> {reply['decision']}"
                            + (f" ({reply['discount_pct']:g}%)" if reply["decision"] == "counter" else ""))
        reply["round"] = _rounds[order_id]
    return reply


@logged
def reconcile() -> dict:
    """Bring POs and the ledger in line with what physically happened. Returns actions taken,
    `followups` you should act on (e.g. reroute a load), net spend and the invariant check."""
    return _call("POST", "/reconcile")


@logged
def check_ledger_invariants() -> dict:
    """Verify every PO's tonnes and money add up and every cancellation has its fee booked."""
    return _call("GET", "/ledger/invariants")


@logged
def escalate_to_human(summary: str, amount_inr: float) -> dict:
    """Ask a human to approve a costly decision. In demo mode this auto-approves and logs loudly."""
    global _approved_inr
    _approved_inr = max(_approved_inr, amount_inr)
    _log("ESCALATION", f"₹{amount_inr:,.0f}: {summary} -> auto-approved (demo mode)")
    return {"approved": True, "amount_inr": amount_inr}


@logged
def advance_clock(days: int = 1) -> dict:
    """Move the simulation forward; deliveries happen and scheduled events fire."""
    return _call("POST", "/clock/advance", json={"days": days})


@logged
def log_decision(summary: str) -> dict:
    """Record your reasoning for the decision log shown to the judges. Keep it to one or two sentences."""
    _log("decision", summary)
    return {"ok": True}


@logged
def recall_lessons() -> str:
    """Lessons learned from previous runs (also already in your instructions)."""
    return memory.lessons_block() or "No lessons yet."


@logged
def save_lesson(text: str) -> dict:
    """Save a short lesson for future runs, e.g. a pattern you noticed. Plain text, one sentence."""
    memory.save_agent_lesson(text)
    return {"saved": True}


ALL_TOOLS = [
    get_clock, get_events, get_orders, get_inventory, get_suppliers, get_price_history,
    get_allocation_state, get_scorecard, run_procurement_optimizer, run_reallocation_optimizer,
    commit_plan, create_purchase_order, cancel_purchase_order, dispatch_from_warehouse,
    reassign_shipment, request_reschedule, reconcile, check_ledger_invariants, escalate_to_human,
    advance_clock, log_decision, recall_lessons, save_lesson,
]
