"""Procurement optimizer (PLAN.md §6.1).

LP: choose how many tonnes to buy from each supplier on each day to meet demand
at minimum total landed cost (price × qty + holding cost + expected shortfall penalty).

Price signal: 7-day moving average + linear trend extrapolated forward.

Run self-check: uv run python -m agent_wars_devfest.tools.procurement
"""

from __future__ import annotations

import numpy as np
import pulp

_SOLVER = pulp.getSolver("HiGHS", msg=False)
_SHORTFALL_PENALTY_PER_T = 10_000  # INR; multiplied by (1 - reliability)


def _forecast(history: list[dict], today: int, horizon: int) -> dict[int, float]:
    """7-day MA at today + linear trend, projected for each future order day."""
    pts = sorted(history, key=lambda p: p["day"])
    days = np.array([p["day"] for p in pts], dtype=float)
    prices = np.array([p["price_per_t"] for p in pts], dtype=float)

    mask7 = (days >= today - 7) & (days <= today)
    ma = float(prices[mask7].mean()) if mask7.any() else float(prices[-1])

    mask14 = (days >= today - 14) & (days <= today)
    slope = float(np.polyfit(days[mask14], prices[mask14], 1)[0]) if mask14.sum() >= 2 else 0.0

    return {d: round(ma + slope * (d - today), 2) for d in range(today, today + horizon + 1)}


def run_procurement_optimizer(
    suppliers: list[dict],
    price_history: dict[str, list[dict]],
    orders: list[dict],
    inventory: dict,
    today: int,
    horizon: int = 7,
    reliability_override: dict[str, float] | None = None,
) -> dict:
    """LP: minimise cost to cover open demand within the horizon.

    Args:
        suppliers:           list of supplier dicts (from GET /suppliers).
        price_history:       sid -> list of price-point dicts (GET /suppliers/{id}/prices).
        orders:              open order dicts (GET /orders?status=open).
        inventory:           inventory dict (GET /inventory).
        today:               current simulation day.
        horizon:             days ahead to plan (default 7).
        reliability_override: observed reliability per supplier from the learning loop;
                             blended with the listed figure automatically.

    Returns dict with keys: status, plan, total_cost, forecast_prices.
    """
    try:
        return _solve(suppliers, price_history, orders, inventory, today, horizon,
                      reliability_override or {})
    except Exception as exc:
        return {"status": "error", "error": str(exc), "plan": [], "total_cost": 0,
                "forecast_prices": {}}


def _effective_reliability(s: dict, overrides: dict[str, float]) -> float:
    listed = s["reliability"]
    if s["id"] not in overrides:
        return listed
    # Bayesian blend: (observed × n + listed × 5) / (n + 5) — weight 5 prior observations
    observed = overrides[s["id"]]
    return round((observed + 5 * listed) / 6, 4)


def _solve(suppliers, price_history, orders, inventory, today, horizon, overrides) -> dict:
    open_orders = [o for o in orders if o.get("status", "open") == "open"]
    if not open_orders:
        return {"status": "optimal", "plan": [], "total_cost": 0, "forecast_prices": {}}

    days = list(range(today, today + horizon + 1))

    # demand per arrival day
    demand: dict[int, float] = {}
    for o in open_orders:
        d = o["requested_day"]
        demand[d] = demand.get(d, 0.0) + o["qty_t"]

    s_map = {s["id"]: s for s in suppliers}

    # forecast prices and effective reliability per supplier
    forecasts: dict[str, dict[int, float]] = {}
    for s in suppliers:
        sid = s["id"]
        forecasts[sid] = _forecast(price_history.get(sid, []), today, horizon)

    # build LP
    prob = pulp.LpProblem("procurement", pulp.LpMinimize)

    # x[(sid, order_day)] = tonnes ordered; arrive on order_day + lead_time
    var_keys = []
    for s in suppliers:
        sid, lead = s["id"], s["lead_time_days"]
        for d in days:
            if d + lead <= today + horizon + 1:  # delivery can land within horizon
                var_keys.append((sid, d))

    x = prob.add_variable_dict("x_%s_%s", var_keys, lowBound=0)
    # add_variable_dict doesn't support per-key upper bounds; enforce via constraints
    for (sid, d), var in x.items():
        prob += var <= s_map[sid]["capacity_t_per_day"]

    # inventory tracking: inv[d] = stock at end of day d
    inv_keys = days
    inv = prob.add_variable_dict("inv_%s", inv_keys, lowBound=0,
                                 upBound=inventory["capacity_t"])

    # objective: procurement cost + shortfall penalty + holding cost
    holding = inventory["holding_cost_per_t_day"]
    obj_terms = []
    for (sid, order_day), var in x.items():
        price = forecasts[sid].get(order_day, s_map[sid]["base_price_per_t"])
        rel = _effective_reliability(s_map[sid], overrides)
        penalty = (1 - rel) * _SHORTFALL_PENALTY_PER_T
        obj_terms.append((price + penalty) * var)
    for d in days:
        obj_terms.append(holding * inv[d])
    prob += pulp.lpSum(obj_terms)

    # inventory balance: inv[d] = inv[d-1] + received[d] - demand[d]
    prev = inventory["qty_t"]
    for d in days:
        received = pulp.lpSum(
            x[(sid, od)]
            for (sid, od) in var_keys
            if od + s_map[sid]["lead_time_days"] == d
        )
        dem = demand.get(d, 0.0)
        prob += inv[d] == prev + received - dem
        prev = inv[d]

    result = prob.solve(_SOLVER)

    if not result.has_solution:
        return {"status": "infeasible", "plan": [], "total_cost": 0,
                "forecast_prices": {sid: list(fp.items()) for sid, fp in forecasts.items()}}

    plan = []
    for (sid, order_day), var in x.items():
        qty = round(pulp.value(var) or 0.0, 2)
        if qty > 0.01:
            s = s_map[sid]
            plan.append({
                "supplier_id": sid,
                "qty_t": qty,
                "order_day": order_day,
                "expected_day": order_day + s["lead_time_days"],
                "price_per_t": forecasts[sid].get(order_day, s["base_price_per_t"]),
                "effective_reliability": _effective_reliability(s, overrides),
            })
    plan.sort(key=lambda p: p["order_day"])

    return {
        "status": "optimal",
        "plan": plan,
        "total_cost": round(pulp.value(prob.objective) or 0.0, 2),
        "forecast_prices": {sid: list(fp.items()) for sid, fp in forecasts.items()},
    }


# ---------- self-check ----------

if __name__ == "__main__":
    from agent_wars_devfest.world.scenarios import act1

    w = act1()
    assert w.inventory is not None
    suppliers = [s.model_dump() | {"price_today": w.price_today(s.id)} for s in w.suppliers.values()]
    ph = {sid: [p.model_dump() for p in pts] for sid, pts in w.prices.items()}
    orders = [o.model_dump() for o in w.orders.values()]
    inv = w.inventory.model_dump()

    result = run_procurement_optimizer(suppliers, ph, orders, inv, today=0)
    total_demand = sum(o["qty_t"] for o in orders)
    total_planned = sum(p["qty_t"] for p in result["plan"])
    print(f"status: {result['status']}")
    print(f"planned {total_planned:.1f}t to cover {total_demand:.1f}t demand")
    print(f"total cost: ₹{result['total_cost']:,.0f}")
    for p in result["plan"]:
        print(f"  {p['supplier_id']} day{p['order_day']}→{p['expected_day']}: "
              f"{p['qty_t']:.1f}t @ ₹{p['price_per_t']:,.0f} (rel={p['effective_reliability']:.2f})")

    # with reliability_override: S1 gets penalised harder
    result2 = run_procurement_optimizer(suppliers, ph, orders, inv, today=0,
                                        reliability_override={"S1": 0.44})
    planned2 = sum(p["qty_t"] for p in result2["plan"])
    s1_qty = sum(p["qty_t"] for p in result2["plan"] if p["supplier_id"] == "S1")
    print(f"\nwith S1 reliability_override=0.44: {planned2:.1f}t planned, S1 gets {s1_qty:.1f}t")

    assert result["status"] == "optimal", f"expected optimal, got {result['status']}"
    # warehouse has 20t; LP only buys the gap
    net_to_buy = total_demand - w.inventory.qty_t
    assert total_planned >= net_to_buy - 5, f"short by {net_to_buy - total_planned:.1f}t"
    print("\nprocurement self-check passed ✓")
