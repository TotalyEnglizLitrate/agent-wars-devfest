"""Reallocation optimizer (PLAN.md §6.2).

MIP: assign free trucks and warehouse stock to uncovered customer demand.
Input is the precomputed /allocation-state response — no separate API calls needed.

Decision variables:
  y[truck_id, customer_id, day] = tonnes delivered  (continuous ≥ 0)
  stop[truck_id, customer_id]   = 1 if truck visits customer  (binary)

Objective (minimise):
  unmet tonnes × 50,000   (largest penalty — never leave demand uncovered if avoidable)
  + delay days × order.delay_penalty_per_day
  + stops × 500           (per-stop routing cost)

Outputs a ready-to-POST dispatch plan plus a list of `asks` (customer reschedule requests
the agent must make before committing delayed deliveries).

Run self-check: uv run python -m agent_wars_devfest.tools.reallocation
"""

from __future__ import annotations

import pulp

_SOLVER = pulp.getSolver("HiGHS", msg=False)
_UNMET_PENALTY = 50_000   # INR per unmet tonne
_STOP_COST = 500          # INR per truck stop


def run_reallocation_optimizer(
    allocation_state: dict,
    customer_flex: dict[str, bool] | None = None,
) -> dict:
    """MIP over free trucks and uncovered demand from /allocation-state.

    Args:
        allocation_state: response from GET /allocation-state.
        customer_flex:    customer_id -> True if the customer has already accepted a delay.

    Returns dict with keys: status, dispatches, asks, unmet, total_penalty.
      dispatches: list of POST /shipments bodies {truck_id, stops: [{customer_id, qty_t, day}]}
      asks:       list of {customer_id, order_id, new_day, reason} — agent must negotiate first
      unmet:      {customer_id: tonnes} for gaps the solver couldn't fill
    """
    customer_flex = customer_flex or {}
    try:
        return _solve(allocation_state, customer_flex)
    except Exception as exc:
        return {"status": "error", "error": str(exc), "dispatches": [], "asks": [], "unmet": {},
                "total_penalty": 0}


def _solve(state: dict, flex: dict[str, bool]) -> dict:
    today = state["day"]
    demand = [o for o in state["demand"] if o["uncovered_t"] > 1e-6]
    free_trucks = state["free_trucks"]
    warehouse_qty = state["warehouse"]["qty_t"]

    if not demand or not free_trucks:
        return {"status": "optimal", "dispatches": [], "asks": [], "unmet": {}, "total_penalty": 0}

    # index structures
    tids = [t["id"] for t in free_trucks]
    t_map = {t["id"]: t for t in free_trucks}
    # one demand entry per order (allocation-state may have multiple orders per customer)
    orders = demand  # use as-is; each has order_id, customer_id, uncovered_t, etc.
    oids = [o["id"] for o in orders]
    o_map = {o["id"]: o for o in orders}

    # candidate (truck, order, day) triples — only feasible delivery days
    candidates: list[tuple[str, str, int]] = []
    for t in free_trucks:
        for o in orders:
            cid = o["customer_id"]
            earliest_arrive = t["arrival_day_by_customer"].get(cid, today + 2)
            earliest = max(o["earliest_day"], earliest_arrive, today + 1)
            flex_confirmed = flex.get(cid, o.get("customer_flex_confirmed", False))
            latest = o["latest_acceptable_day"] + (2 if flex_confirmed else 0)
            for day in range(earliest, latest + 1):
                candidates.append((t["id"], o["id"], day))

    if not candidates:
        unmet_all = {o["customer_id"]: o["uncovered_t"] for o in orders}
        return {"status": "infeasible", "dispatches": [], "asks": [],
                "unmet": unmet_all, "total_penalty": sum(unmet_all.values()) * _UNMET_PENALTY}

    prob = pulp.LpProblem("reallocation", pulp.LpMinimize)

    # y[(tid, oid, day)] = tonnes delivered
    y = prob.add_variable_dict("y_%s_%s_%s", candidates, lowBound=0)
    # set capacity upper bounds
    for (tid, oid, day), var in y.items():
        prob += var <= t_map[tid]["capacity_t"]

    # stop[(tid, oid)] = 1 if truck visits this order's customer
    stop_keys = list({(tid, oid) for tid, oid, _ in candidates})
    stop = prob.add_variable_dict("s_%s_%s", stop_keys, cat="Binary")

    # unmet[oid] slack variable
    unmet = prob.add_variable_dict("u_%s", oids, lowBound=0)

    # delay cost per order: we model the latest planned day minus requested_day
    # (binary variable for "any delay" per order-truck pair would complicate; use continuous approx)

    # objective
    obj = []
    for oid in oids:
        o = o_map[oid]
        obj.append(_UNMET_PENALTY * unmet[oid])
        # delay cost: weighted by fraction of order, penalty per day late
        for (tid, oid2, day) in candidates:
            if oid2 != oid:
                continue
            delay_days = max(0, day - o["requested_day"])
            if delay_days > 0:
                obj.append(o["delay_penalty_per_day"] * delay_days / o["uncovered_t"] * y[(tid, oid, day)])
    for (tid, oid) in stop_keys:
        obj.append(_STOP_COST * stop[(tid, oid)])
    prob += pulp.lpSum(obj)

    # demand satisfaction: sum of delivered + unmet >= uncovered_t
    for oid in oids:
        o = o_map[oid]
        delivered = pulp.lpSum(y[(tid, oid2, day)] for tid, oid2, day in candidates if oid2 == oid)
        prob += delivered + unmet[oid] >= o["uncovered_t"]

    # truck capacity per day: each truck can carry at most capacity_t tonnes per day
    for tid in tids:
        cap = t_map[tid]["capacity_t"]
        day_set = {day for t2, _, day in candidates if t2 == tid}
        for day in day_set:
            load = pulp.lpSum(y[(tid, oid, day)] for _, oid2, d2 in candidates
                              if _ == tid and d2 == day for oid in [oid2])
            prob += load <= cap

    # linking: y[t,o,d] > 0 → stop[t,o] = 1
    for (tid, oid, day), var in y.items():
        if (tid, oid) in stop:
            prob += var <= t_map[tid]["capacity_t"] * stop[(tid, oid)]

    # warehouse supply: total dispatched <= warehouse stock
    total_dispatched = pulp.lpSum(y.values())
    prob += total_dispatched <= warehouse_qty

    result = prob.solve(_SOLVER)

    if not result.has_solution:
        unmet_all = {o["customer_id"]: o["uncovered_t"] for o in orders}
        return {"status": "infeasible", "dispatches": [], "asks": [],
                "unmet": unmet_all, "total_penalty": sum(unmet_all.values()) * _UNMET_PENALTY}

    # extract dispatch plan: group by truck
    truck_stops: dict[str, list[dict]] = {tid: [] for tid in tids}
    for (tid, oid, day), var in y.items():
        qty = round(pulp.value(var) or 0.0, 2)
        if qty > 0.01:
            truck_stops[tid].append({"customer_id": o_map[oid]["customer_id"], "qty_t": qty, "day": day})

    dispatches = []
    for tid, stops in truck_stops.items():
        if stops:
            stops_sorted = sorted(stops, key=lambda s: s["day"])
            dispatches.append({"truck_id": tid, "stops": stops_sorted})

    # asks: delays past requested_day where flex not yet confirmed
    asks = []
    seen_orders: set[str] = set()
    for (tid, oid, day), var in y.items():
        if (pulp.value(var) or 0.0) < 0.01:
            continue
        o = o_map[oid]
        cid = o["customer_id"]
        if day > o["requested_day"] and not flex.get(cid, o.get("customer_flex_confirmed", False)):
            if oid not in seen_orders:
                seen_orders.add(oid)
                asks.append({
                    "customer_id": cid,
                    "order_id": oid,
                    "new_day": day,
                    "reason": (
                        f"Earliest available truck ({tid}) reaches {cid} on day {day}; "
                        f"order requested day {o['requested_day']}."
                    ),
                })

    unmet_out = {o_map[oid]["customer_id"]: round(pulp.value(unmet[oid]) or 0.0, 2)
                 for oid in oids if (pulp.value(unmet[oid]) or 0.0) > 0.01}

    return {
        "status": "optimal",
        "dispatches": dispatches,
        "asks": asks,
        "unmet": unmet_out,
        "total_penalty": round(pulp.value(prob.objective) or 0.0, 2),
    }


# ---------- self-check: act2 sugar example ----------

if __name__ == "__main__":
    from fastapi.testclient import TestClient
    from agent_wars_devfest.world.app import app

    c = TestClient(app)
    c.post("/scenario/act2")
    # advance to day 2: short-ship event fires, SH2 trimmed to 12t
    c.post("/clock/advance", json={"days": 1})

    state = c.get("/allocation-state").json()
    print(f"day: {state['day']}")
    print(f"issues: {state['issues']}")
    print(f"demand: {[(o['id'], o['customer_id'], o['uncovered_t']) for o in state['demand']]}")
    print(f"free trucks: {[t['id'] for t in state['free_trucks']]}")

    result = run_reallocation_optimizer(state)
    print(f"\nstatus: {result['status']}")
    print(f"dispatches: {result['dispatches']}")
    print(f"asks: {result['asks']}")
    print(f"unmet: {result['unmet']}")
    print(f"total_penalty: ₹{result['total_penalty']:,.0f}")

    assert result["status"] in ("optimal", "feasible"), f"expected solution, got {result['status']}"
    print("\nreallocation self-check passed ✓")
