"""Adapter between the agent and B's solvers (tools/reallocation.py, tools/procurement.py).

B's MIP/LP handle the core optimisation; this module adds what the agent needs around them:

  reallocate:  1. greedy: re-route loads already on the road to cancelled customers (B's MIP only
                  plans free trucks from the warehouse)
               2. B's MIP: warehouse stock -> free trucks for the remaining urgent demand
               3. greedy: whatever is still short becomes delay `asks` against incoming POs (B's MIP
                  only plans inside each customer's window, so it never asks)
  procure:     B's LP on the real remaining need (uncovered tonnes, net of stock, incoming POs and
               re-routable loads); only today's buys are executed, the rest is re-planned tomorrow.
               PO cancellation stays greedy (the LP doesn't cancel).

If B's modules are missing or a solve fails, the greedy fallbacks below run on their own.
Every result carries "solver" so the decision log shows which path ran.
"""

import copy

import numpy as np

URGENT_WINDOW_DAYS = 2  # plan deliveries for orders that can start receiving within this many days
PROCURE_HORIZON_DAYS = 4  # buy just in time: only for orders due within this many days (holding costs money)
MIN_SPLIT_T = 10  # below this, buy from one supplier instead of splitting
SHORT_FRACTION = 0.4  # a missed delivery ships 60% (matches the world)
SHORTFALL_COST_PER_T = 10_000  # emergency re-buy premium + delay, INR per tonne short
MAX_SHARE = 0.6  # never put more than this share of a buy on one supplier


def reallocate(state: dict, travel_days: dict) -> dict:
    try:
        from agent_wars_devfest.tools.reallocation import run_reallocation_optimizer as mip
    except ImportError:
        return greedy_reallocate(state, travel_days) | {"solver": "greedy-fallback"}

    # 1. loads on the road to cancelled customers (warehouse and incoming switched off)
    s1 = copy.deepcopy(state)
    s1["warehouse"] |= {"qty_t": 0, "incoming": []}
    s1["free_trucks"] = []
    g1 = greedy_reallocate(s1, travel_days)
    residual = {u["order_id"]: u["qty_t"] for u in g1["unmet"]}

    # 2. B's MIP over the remaining urgent demand, one (priority, requested_day) tier at a time.
    #    B's objective prices every unmet tonne the same but charges delay only when delivered, so in
    #    one big solve it starves overdue or high-priority orders in favour of ones not yet due.
    dispatches, mip_note = [], "MIP: nothing to dispatch"
    stock = state["warehouse"]["qty_t"]
    trucks = list(state["free_trucks"])
    for tier in sorted({(o["priority"], o["requested_day"]) for o in state["demand"] if o["id"] in residual}):
        tier_orders = [o for o in state["demand"]
                       if o["id"] in residual and (o["priority"], o["requested_day"]) == tier]
        if not tier_orders or stock <= 0.01 or not trucks:
            continue
        s2 = copy.deepcopy(state)
        s2["demand"] = [o | {"uncovered_t": residual[o["id"]]} for o in tier_orders]
        s2["warehouse"]["qty_t"], s2["free_trucks"] = stock, trucks
        b = mip(s2)
        if b.get("status") not in ("optimal", "feasible"):
            return greedy_reallocate(state, travel_days) | {"solver": f"greedy-fallback (MIP {b.get('status')})"}
        caps = {t["id"]: t["capacity_t"] for t in trucks}
        tier_disp = []
        for disp in b["dispatches"]:  # MIP caps load per day; a truck makes one trip, so cap the total
            room, stops = min(caps[disp["truck_id"]], stock), []
            for stop in disp["stops"]:
                qty = round(min(stop["qty_t"], room), 2)
                if qty > 0.01:
                    stops.append(stop | {"qty_t": qty})
                    room -= qty
            if stops:
                tier_disp.append({"tool": "dispatch_from_warehouse", "truck_id": disp["truck_id"], "stops": stops})
                stock -= sum(st["qty_t"] for st in stops)
        used = {disp["truck_id"] for disp in tier_disp}
        trucks = [t for t in trucks if t["id"] not in used]
        delivered: dict[str, float] = {}
        for disp in tier_disp:
            for stop in disp["stops"]:
                delivered[stop["customer_id"]] = delivered.get(stop["customer_id"], 0) + stop["qty_t"]
        for o in sorted(tier_orders, key=lambda o: o["requested_day"]):
            take = min(residual[o["id"]], delivered.get(o["customer_id"], 0))
            residual[o["id"]] -= take
            delivered[o["customer_id"]] = delivered.get(o["customer_id"], 0) - take
        dispatches += tier_disp
    residual = {k: v for k, v in residual.items() if v > 0.01}
    if dispatches:
        mip_note = f"MIP: {len(dispatches)} warehouse dispatch(es)"

    # 3. still short: delay asks against incoming POs
    s3 = copy.deepcopy(state)
    s3["demand"] = [o | {"uncovered_t": residual[o["id"]]} for o in state["demand"] if o["id"] in residual]
    s3["warehouse"]["qty_t"] = 0
    s3["free_trucks"], s3["shipments"] = [], []
    g3 = greedy_reallocate(s3, travel_days)

    parts = [p for p in (g1["explanation"] if g1["actions"] else "", mip_note,
                         g3["explanation"] if g3["asks"] or g3["unmet"] else "") if p]
    return {"actions": g1["actions"] + dispatches, "asks": g3["asks"], "deferred": g3["deferred"],
            "unmet": g3["unmet"], "cost_estimate": g3["cost_estimate"], "explanation": "; ".join(parts),
            "solver": "greedy re-route + MIP dispatch + greedy asks"}


def procure(suppliers: list, price_history: dict, state: dict, reliability_override: dict,
            raw_reliability: dict | None = None, travel_days: dict | None = None) -> dict:
    g = greedy_procure(suppliers, price_history, state, reliability_override)
    try:
        from agent_wars_devfest.tools.procurement import run_procurement_optimizer as lp
    except ImportError:
        return g | {"solver": "greedy-fallback"}
    if g["net_need_t"] <= 1e-6:
        return g | {"solver": "greedy (nothing to buy; LP not needed)"}

    d, wh = state["day"], state["warehouse"]
    open_customers = {o["customer_id"] for o in state["demand"]}
    orphaned = sum(s["qty_t"] for sh in state["shipments"] for s in sh["stops"] if s["customer_id"] not in open_customers)
    on_hand = wh["qty_t"] + sum(i["qty_t"] for i in wh["incoming"]) + orphaned
    # B's LP balances stock at the warehouse, so each order is due there a truck leg before the customer needs it.
    leg = (travel_days or {}).get(wh["warehouse_id"], {})
    orders = [o | {"qty_t": o["uncovered_t"], "status": "open",
                   "requested_day": max(o["requested_day"] - leg.get(o["customer_id"], 1), d + 1)}
              for o in state["demand"] if o["uncovered_t"] > 1e-6 and o["requested_day"] <= d + PROCURE_HORIZON_DAYS]
    inventory = {"qty_t": on_hand, "capacity_t": max(wh["capacity_t"], on_hand + 200),
                 "holding_cost_per_t_day": wh["holding_cost_per_t_day"]}
    b = lp(suppliers, price_history, orders, inventory, today=d, horizon=PROCURE_HORIZON_DAYS,
           reliability_override=raw_reliability or {})
    if b.get("status") != "optimal":
        return g | {"solver": f"greedy-fallback (LP {b.get('status')})"}

    today = {}
    for row in b["plan"]:
        if row["order_day"] == d:  # later days get re-planned tomorrow with fresh prices
            today.setdefault(row["supplier_id"], []).append(row)
    purchases = [{"supplier_id": sid, "qty_t": round(sum(r["qty_t"] for r in rows), 1),
                  "reason": f"LP: forecast ₹{rows[0]['price_per_t']:,.0f}/t, reliability {rows[0]['effective_reliability']:.2f}"}
                 for sid, rows in today.items()]
    later = sum(r["qty_t"] for r in b["plan"] if r["order_day"] > d)
    return {"purchases": purchases, "cancel_pos": [], "net_need_t": g["net_need_t"],
            "explanation": f"LP over {PROCURE_HORIZON_DAYS} days: buy {sum(p['qty_t'] for p in purchases):g}t today"
                           + (f", {later:g}t planned for later days" if later else ""),
            "solver": "LP (B)"}


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
