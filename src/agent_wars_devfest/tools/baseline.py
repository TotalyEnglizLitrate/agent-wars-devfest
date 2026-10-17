"""FCFS baseline (PLAN.md §6.4): no reallocation, no negotiation, buy cheapest at today's price.

Drives the World API in-process, so it needs no running server and uses the same
/scorecard the agent is judged by. It reloads scenarios, so don't call it inside the
process that's serving the live demo.

Run: uv run python -m agent_wars_devfest.tools.baseline
"""

from fastapi.testclient import TestClient

from agent_wars_devfest.world.app import app

END_DAY = 9  # every scenario's orders have closed by then


def _step(c: TestClient) -> None:
    state = c.get("/allocation-state").json()
    suppliers = c.get("/suppliers").json()
    cheapest = min(suppliers, key=lambda s: s["price_today"])
    pool = state["warehouse"]["qty_t"]
    incoming = sum(i["qty_t"] for i in state["warehouse"]["incoming"])
    free = sorted(state["free_trucks"], key=lambda t: t["id"])

    for o in sorted(state["demand"], key=lambda o: (o["requested_day"], o["id"])):  # first come, first served
        need = o["uncovered_t"]
        while need > 1e-6 and pool > 1e-6 and free:
            truck = free.pop(0)
            qty = min(need, pool, truck["capacity_t"])
            day = max(o["earliest_day"], truck["arrival_day_by_customer"][o["customer_id"]])
            c.post("/shipments", json={"truck_id": truck["id"],
                                       "stops": [{"customer_id": o["customer_id"], "qty_t": qty, "day": day}]})
            need, pool = need - qty, pool - qty
        covered = min(need, incoming)
        need, incoming = need - covered, incoming - covered
        if need > 1e-6:
            qty = min(need, cheapest["capacity_t_per_day"] * 3)
            c.post("/purchase-orders", json={"supplier_id": cheapest["id"], "qty_t": qty})


def run_fcfs(act: str) -> dict:
    c = TestClient(app)
    c.post(f"/scenario/{act}")
    while c.get("/clock").json()["day"] < END_DAY:
        _step(c)
        c.post("/clock/advance", json={"days": 1})
    return c.get("/scorecard").json()


if __name__ == "__main__":
    for act in ("act1", "act2", "act3"):
        s = run_fcfs(act)
        print(f"{act}: fulfilled {s['fulfillment_rate']:.0%}, on time {s['on_time_rate']:.0%}, "
              f"unmet {s['unmet_t']:g}t, wasted {s['wasted_t']:g}t, total cost ₹{s['total_cost']:,.0f} "
              f"(spend ₹{s['procurement_spend']:,.0f}, delay ₹{s['delay_cost']:,.0f})")
