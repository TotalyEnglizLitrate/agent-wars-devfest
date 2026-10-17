"""Adapter between the agent and B's solvers, with greedy fallbacks until they land.

Contract for B (the agent only depends on this):

    tools/reallocation.py:  solve(state: dict, travel_days: dict) -> dict
        state = GET /allocation-state JSON
        returns {"actions": [...], "asks": [...], "deferred": [...], "unmet": [...],
                 "cost_estimate": float, "explanation": str}
        action = {"tool": "reassign_shipment", "shipment_id": str, "stops": [{customer_id, qty_t, day}]}
               | {"tool": "dispatch_from_warehouse", "truck_id": str, "stops": [...]}
        ask    = {"customer_id": str, "order_id": str, "new_day": int, "qty_t": float, "reason": str}

    tools/procurement.py:   solve(suppliers: list, price_history: dict[str, list], state: dict,
                                  reliability_override: dict[str, float]) -> dict
        returns {"purchases": [{"supplier_id", "qty_t", "reason"}], "cancel_pos": [po_id, ...],
                 "net_need_t": float, "explanation": str}
"""

import numpy as np

URGENT_WINDOW_DAYS = 2  # plan deliveries for orders that can start receiving within this many days
PROCURE_HORIZON_DAYS = 4  # buy just in time: only for orders due within this many days (holding costs money)
MIN_SPLIT_T = 10  # below this, buy from one supplier instead of splitting
SHORT_FRACTION = 0.4  # a missed delivery ships 60% (matches the world)
SHORTFALL_COST_PER_T = 10_000  # emergency re-buy premium + delay, INR per tonne short
MAX_SHARE = 0.6  # never put more than this share of a buy on one supplier


def reallocate(state: dict, travel_days: dict) -> dict:
    try:
        from agent_wars_devfest.tools.reallocation import solve
    except ImportError:
        return greedy_reallocate(state, travel_days) | {"solver": "greedy-fallback"}
    return solve(state, travel_days) | {"solver": "pulp"}


def procure(suppliers: list, price_history: dict, state: dict, reliability_override: dict) -> dict:
    try:
        from agent_wars_devfest.tools.procurement import solve
    except ImportError:
        return greedy_procure(suppliers, price_history, state, reliability_override) | {"solver": "greedy-fallback"}
    return solve(suppliers, price_history, state, reliability_override) | {"solver": "pulp"}


# ---------- fallbacks ----------

def greedy_reallocate(state: dict, travel_days: dict) -> dict:
    d = state["day"]
    wh = state["warehouse"]
    open_customers = {o["customer_id"] for o in state["demand"]}
    urgent = sorted((o for o in state["demand"] if o["uncovered_t"] > 1e-6 and o["earliest_day"] <= d + URGENT_WINDOW_DAYS),
                    key=lambda o: (o["priority"], o["requested_day"]))

    # Loads already on the road to customers who no longer want them.
    orphans = []
    for sh in state["shipments"]:
        movable = sum(s["qty_t"] for s in sh["stops"] if s["customer_id"] not in open_customers)
        if movable > 1e-6:
            kept = [s for s in sh["stops"] if s["customer_id"] in open_customers]
            orphans.append({"sh": sh, "qty": movable, "kept": kept})
    stock = wh["qty_t"]
    trucks = sorted(state["free_trucks"], key=lambda t: t["id"])
    incoming = sorted(wh["incoming"], key=lambda i: i["day"])
    incoming_left = [dict(i) for i in incoming]

    actions, asks, deferred, unmet, cost = [], [], [], [], 0.0
    for o in urgent:
        need, c = o["uncovered_t"], o["customer_id"]
        for orph in orphans:
            arrive = orph["sh"]["arrival_day_by_customer"][c]
            if need > 1e-6 and orph["qty"] > 1e-6 and arrive <= o["latest_acceptable_day"]:
                qty = min(need, orph["qty"])
                orph["kept"].append({"customer_id": c, "qty_t": qty, "day": max(arrive, o["earliest_day"])})
                orph["qty"] -= qty
                need -= qty
        while need > 1e-6 and stock > 1e-6 and trucks:
            arrive = trucks[0]["arrival_day_by_customer"][c]
            if arrive > o["latest_acceptable_day"]:
                break
            t = trucks.pop(0)
            qty = min(need, stock, t["capacity_t"])
            actions.append({"tool": "dispatch_from_warehouse", "truck_id": t["id"],
                            "stops": [{"customer_id": c, "qty_t": qty, "day": max(arrive, o["earliest_day"])}]})
            stock -= qty
            need -= qty
        for inc in incoming_left:
            if need <= 1e-6:
                break
            if inc["qty_t"] <= 1e-6:
                continue
            qty = min(need, inc["qty_t"])
            arrive = max(d + 1, inc["day"] + travel_days[wh["warehouse_id"]][c])
            inc["qty_t"] -= qty
            need -= qty
            if arrive <= o["latest_acceptable_day"]:
                deferred.append(f"{o['id']}: send {qty:g}t from {inc['po_id']} when it lands on day {inc['day']} "
                                f"(arrives day {arrive})")
            else:
                late = arrive - o["latest_acceptable_day"]
                asks.append({"customer_id": c, "order_id": o["id"], "new_day": arrive, "qty_t": qty,
                             "reason": f"{qty:g}t can only arrive on day {arrive} (from {inc['po_id']})"})
                cost += late * o["delay_penalty_per_day"]
        if need > 1e-6:
            unmet.append({"order_id": o["id"], "customer_id": c, "qty_t": need})

    # Leftover orphaned tonnes never keep heading to a cancelled customer: give them to any open order
    # they can reach in time, else bring them back to the warehouse.
    claimed = {o["id"] for o in urgent}
    for orph in orphans:
        for o in sorted(state["demand"], key=lambda o: o["requested_day"]):
            arrive = orph["sh"]["arrival_day_by_customer"][o["customer_id"]]
            if orph["qty"] > 1e-6 and o["id"] not in claimed and o["uncovered_t"] > 1e-6 \
                    and arrive <= o["latest_acceptable_day"]:
                qty = min(orph["qty"], o["uncovered_t"])
                orph["kept"].append({"customer_id": o["customer_id"], "qty_t": qty,
                                     "day": max(arrive, o["earliest_day"])})
                orph["qty"] -= qty
                claimed.add(o["id"])

    for orph in orphans:  # whatever is still unassigned returns to the warehouse on reassign
        if orph["kept"] != [s for s in orph["sh"]["stops"] if s["customer_id"] in open_customers] or orph["qty"] > 1e-6:
            actions.append({"tool": "reassign_shipment", "shipment_id": orph["sh"]["shipment_id"],
                            "stops": orph["kept"]})

    parts = []
    if asks:
        parts.append(f"{len(asks)} delay(s) need customer consent")
    if unmet:
        parts.append(f"{sum(u['qty_t'] for u in unmet):g}t still unmet: buy more")
    return {"actions": actions, "asks": asks, "deferred": deferred, "unmet": unmet,
            "cost_estimate": round(cost, 2), "explanation": "; ".join(parts) or "deliveries on track"}


def forecast_price(history: list[dict], days_ahead: int) -> float:
    """7-day moving average + trend from the last 14 days (PLAN.md §6.1)."""
    prices = np.array([p["price_per_t"] for p in history[-14:]])
    ma7 = prices[-7:].mean()
    slope = np.polyfit(np.arange(len(prices)), prices, 1)[0] if len(prices) > 1 else 0.0
    return float(ma7 + slope * days_ahead)


def greedy_procure(suppliers: list, price_history: dict, state: dict, reliability_override: dict) -> dict:
    d = state["day"]
    wh = state["warehouse"]
    uncovered = sum(o["uncovered_t"] for o in state["demand"] if o["requested_day"] <= d + PROCURE_HORIZON_DAYS)
    incoming = sum(i["qty_t"] for i in wh["incoming"])
    open_customers = {o["customer_id"] for o in state["demand"]}
    orphaned = sum(s["qty_t"] for sh in state["shipments"] for s in sh["stops"]
                   if s["customer_id"] not in open_customers)  # reroutable, so it counts as supply
    net = uncovered - wh["qty_t"] - incoming - orphaned

    if net <= 1e-6:
        # Cancel only against ALL remaining demand: stock for later orders isn't surplus.
        all_uncovered = sum(o["uncovered_t"] for o in state["demand"])
        surplus = max(0.0, wh["qty_t"] + incoming + orphaned - all_uncovered)
        cancel = []
        for po_id, qty in sorted(state.get("open_pos", {}).items(), key=lambda kv: -kv[1]):
            if qty <= surplus + 1e-6:
                cancel.append(po_id)
                surplus -= qty
        return {"purchases": [], "cancel_pos": cancel, "net_need_t": round(net, 2),
                "explanation": f"next {PROCURE_HORIZON_DAYS} days covered" + (f"; {surplus:g}t surplus overall, cancel {', '.join(cancel)}" if cancel else "")}

    last_day = max((o["latest_acceptable_day"] for o in state["demand"]), default=d + 7)
    ranked = []
    for s in suppliers:
        if d + s["lead_time_days"] + 1 > last_day:  # +1 day truck leg from the warehouse
            continue
        rel = reliability_override.get(s["id"], s["reliability"])
        price = forecast_price(price_history[s["id"]], s["lead_time_days"])
        risk = (1 - rel) * SHORT_FRACTION * SHORTFALL_COST_PER_T
        ranked.append((price + risk, price, rel, s))
    ranked.sort(key=lambda r: r[0])

    purchases, left = [], net
    for score, price, rel, s in ranked:
        if left <= 1e-6:
            break
        last = s is ranked[-1][3] or net < MIN_SPLIT_T
        share = left if last else min(left, net * MAX_SHARE)
        qty = round(min(share, s["capacity_t_per_day"] * 3), 1)
        if qty <= 0:
            continue
        purchases.append({"supplier_id": s["id"], "qty_t": qty,
                          "reason": f"forecast ₹{price:,.0f}/t + risk ₹{score - price:,.0f}/t "
                                    f"(reliability {rel:.2f}) = ₹{score:,.0f}/t effective"})
        left -= qty
    return {"purchases": purchases, "cancel_pos": [], "net_need_t": round(net, 2),
            "explanation": f"need {net:g}t; ranked by price + expected shortfall cost; max {MAX_SHARE:.0%} per supplier"}
