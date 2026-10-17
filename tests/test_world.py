"""World API tests: seeded data, the three demo acts, and /allocation-state.

Run: uv run pytest
"""

import pytest
from fastapi.testclient import TestClient

from agent_wars_devfest.world.app import app
from agent_wars_devfest.world.datagen import generate_base_world

client = TestClient(app)


def ok(resp):
    assert resp.status_code < 400, (resp.status_code, resp.text)
    return resp.json()


def load(act: str) -> dict:
    return ok(client.post(f"/scenario/{act}"))


def advance(days: int = 1) -> dict:
    return ok(client.post("/clock/advance", json={"days": days}))


# ---------- seeded data ----------

def test_world_is_reproducible():
    a, b = generate_base_world(), generate_base_world()
    assert a.suppliers == b.suppliers
    assert a.prices == b.prices
    assert a.travel_days == b.travel_days


def test_supplier_tradeoff():
    sup = {s["id"]: s for s in ok(client.get("/suppliers"))}
    assert len(sup) == 5
    assert sup["S1"]["base_price_per_t"] == min(s["base_price_per_t"] for s in sup.values())
    assert sup["S2"]["reliability"] == max(s["reliability"] for s in sup.values())
    assert sup["S1"]["recent_deliveries"].count(False) == 3  # "missed 3 of its last 10"


def test_prices_never_leak_the_future():
    load("act1")
    hist = ok(client.get("/suppliers/S1/prices?days=90"))
    assert len(hist) == 90
    assert max(p["day"] for p in hist) == 0
    advance()
    assert max(p["day"] for p in ok(client.get("/suppliers/S1/prices"))) == 1


def test_travel_days_symmetric():
    t = ok(client.get("/travel-days"))
    assert all(t[a][b] == t[b][a] for a in t for b in t)


def test_unknown_scenario_404():
    assert client.post("/scenario/nope").status_code == 404


# ---------- act 1: procurement ----------

def test_act1_demand_and_po():
    fired = load("act1")["events_fired"]
    assert fired[0]["type"] == "new_order"
    assert sum(o["qty_t"] for o in ok(client.get("/orders?status=open"))) == 150

    po = ok(client.post("/purchase-orders", json={"supplier_id": "S1", "qty_t": 60}))
    sup = {s["id"]: s for s in ok(client.get("/suppliers"))}
    assert po["price_per_t"] == sup["S1"]["price_today"]
    assert po["expected_day"] == sup["S1"]["lead_time_days"]
    entry = ok(client.get("/ledger"))[-1]
    assert (entry["kind"], entry["ref_id"]) == ("po", po["id"])
    assert entry["amount"] == pytest.approx(60 * po["price_per_t"])


def test_po_over_capacity_rejected():
    load("act1")
    assert client.post("/purchase-orders", json={"supplier_id": "S1", "qty_t": 10_000}).status_code == 400


# ---------- act 2: shortage + negotiation ----------

def test_act2_short_ship():
    load("act2")
    fired = advance()["events_fired"]
    assert [e["type"] for e in fired] == ["short_ship"]
    sh = {s["id"]: s for s in ok(client.get("/shipments"))}
    assert sh["SH1"]["status"] == "delivered"
    assert sum(s["qty_t"] for s in sh["SH2"]["stops"]) == 12


def test_act2_factory_b_counters_then_accepts():
    load("act2")
    advance()
    first = ok(client.post("/customers/C2/reschedule-request", json={"order_id": "O2", "new_day": 4}))
    assert (first["decision"], first["discount_pct"]) == ("counter", 5)
    lowball = ok(client.post("/customers/C2/reschedule-request",
                             json={"order_id": "O2", "new_day": 4, "discount_pct": 2}))
    assert lowball["decision"] == "counter"
    deal = ok(client.post("/customers/C2/reschedule-request",
                          json={"order_id": "O2", "new_day": 4, "discount_pct": 5}))
    assert deal["decision"] == "accept"
    o2 = next(o for o in ok(client.get("/orders")) if o["id"] == "O2")
    assert o2["customer_flex_confirmed"] and o2["discount_pct"] == 5
    assert len(ok(client.get("/negotiations"))) == 3


def test_reschedule_far_too_late_rejected():
    load("act2")
    r = ok(client.post("/customers/C3/reschedule-request", json={"order_id": "O3", "new_day": 30}))
    assert r["decision"] == "reject"


def test_reschedule_wrong_customer_400():
    load("act2")
    assert client.post("/customers/C1/reschedule-request",
                       json={"order_id": "O2", "new_day": 4}).status_code == 400


def test_reassign_respects_capacity_and_time():
    load("act2")
    advance()
    stops = [{"customer_id": "C1", "qty_t": 10, "day": 3}, {"customer_id": "C2", "qty_t": 2, "day": 3}]
    assert ok(client.post("/shipments/SH2/reassign", json={"stops": stops}))["stops"][0]["customer_id"] == "C1"
    too_heavy = [{"customer_id": "C1", "qty_t": 25, "day": 3}]
    assert client.post("/shipments/SH2/reassign", json={"stops": too_heavy}).status_code == 400
    past = [{"customer_id": "C1", "qty_t": 5, "day": 1}]
    assert client.post("/shipments/SH2/reassign", json={"stops": past}).status_code == 400


def test_act2_diverted_truck_delivers_to_factory_a():
    load("act2")
    advance()
    stops = [{"customer_id": "C1", "qty_t": 10, "day": 3}, {"customer_id": "C2", "qty_t": 2, "day": 3}]
    ok(client.post("/shipments/SH2/reassign", json={"stops": stops}))
    advance()
    o1 = next(o for o in ok(client.get("/orders")) if o["id"] == "O1")
    assert o1["status"] == "fulfilled" and o1["delivered_qty_t"] == 30


def test_act2_allocation_state_shows_the_showpiece():
    load("act2")
    advance()
    s = ok(client.get("/allocation-state"))
    demand = {o["id"]: o for o in s["demand"]}
    assert demand["O1"]["uncovered_t"] == 10  # Factory A got 20t of 30t
    assert demand["O2"]["committed_t"] == 12  # truck 2 short-shipped
    sh2 = next(sh for sh in s["shipments"] if sh["shipment_id"] == "SH2")
    assert sh2["arrival_day_by_customer"]["C1"] == 3  # divertible to A on day 3
    assert s["warehouse"]["qty_t"] == 10
    assert len(s["issues"]) == 2 and all(i.startswith(("O1", "O2")) for i in s["issues"])


# ---------- act 3: cancellation ----------

def test_act3_cancel_and_po_rules():
    load("act3")
    assert [e["type"] for e in advance()["events_fired"]] == ["buyer_cancel"]
    orders = {o["customer_id"]: o for o in ok(client.get("/orders"))}
    assert orders["C3"]["status"] == "cancelled"

    pos = {p["id"]: p for p in ok(client.get("/purchase-orders"))}
    assert pos["PO3"]["status"] == "open"  # top-up still cancellable on day 3
    cancelled = ok(client.post("/purchase-orders/PO3/cancel"))
    assert cancelled["status"] == "cancelled" and cancelled["cancelled_qty_t"] == 25
    refund = ok(client.get("/ledger"))[-1]
    assert refund["kind"] == "refund" and refund["amount"] == pytest.approx(-25 * cancelled["price_per_t"])
    assert client.post("/purchase-orders/PO1/cancel").status_code == 409  # already received


def test_act3_allocation_state_flags_orphan_and_waiting_order():
    load("act3")
    advance()
    s = ok(client.get("/allocation-state"))
    assert any("heading to C3 has no open order" in i for i in s["issues"])
    assert any(i.startswith("O3 (C4) is short 20t") for i in s["issues"])
    orphan = next(sh for sh in s["shipments"] if sh["stops"][0]["customer_id"] == "C3")
    assert orphan["arrival_day_by_customer"]["C4"] <= 6  # can be rerouted inside C4's window


# ---------- scenario consistency ----------

@pytest.mark.parametrize("act", ["act1", "act2", "act3"])
def test_scripted_stops_match_travel_days(act):
    load(act)
    t = ok(client.get("/travel-days"))
    for sh in ok(client.get("/shipments")):
        first = sh["stops"][0]
        assert first["day"] >= sh["dispatch_day"] + t[sh["supplier_id"]][first["customer_id"]], sh["id"]


@pytest.mark.parametrize("act", ["act1", "act2", "act3"])
def test_scenario_reload_resets_state(act):
    load(act)
    client.post("/purchase-orders", json={"supplier_id": "S2", "qty_t": 10})
    advance()
    first = load(act)
    assert first["day"] == ok(client.get("/clock"))["day"]
    assert ok(client.get("/negotiations")) == []
    assert all(p["supplier_id"] != "S2" or p["qty_t"] != 10 for p in ok(client.get("/purchase-orders")))


# ---------- warehouse dispatch, supplier reliability, scorecard ----------

def test_dispatch_from_warehouse():
    load("act2")  # 10t in the warehouse, WH1->C1 is 1 day
    sh = ok(client.post("/shipments", json={"truck_id": "T5", "stops": [{"customer_id": "C1", "qty_t": 10, "day": 2}]}))
    assert sh["supplier_id"] == "WH1"
    assert ok(client.get("/inventory"))["qty_t"] == 0
    advance()
    o1 = next(o for o in ok(client.get("/orders")) if o["id"] == "O1")
    assert o1["delivered_qty_t"] == 30  # 20t from T1 + 10t from the warehouse


def test_dispatch_rules():
    load("act2")
    stop = {"customer_id": "C1", "qty_t": 5, "day": 2}
    assert client.post("/shipments", json={"truck_id": "T2", "stops": [stop]}).status_code == 409  # busy truck
    assert client.post("/shipments", json={"truck_id": "T5", "stops": [stop | {"qty_t": 15}]}).status_code == 400  # stock
    assert client.post("/shipments", json={"truck_id": "T5", "stops": [stop | {"day": 1}]}).status_code == 400  # too soon


def test_supplier_misses_are_seeded_and_only_hit_api_pos():
    def misses():
        load("act1")
        for _ in range(4):
            client.post("/purchase-orders", json={"supplier_id": "S1", "qty_t": 20})
        advance(3)
        return [e["payload"]["po_id"] for e in ok(client.get("/events")) if e["type"] == "short_ship"]
    first = misses()
    assert first and first == misses()
    pos = {p["id"]: p for p in ok(client.get("/purchase-orders"))}
    for po_id in first:
        assert pos[po_id]["cancelled_qty_t"] == 8 and pos[po_id]["received_qty_t"] == 12

    load("act2")  # scripted POs never miss at random, so the demo script stays exact
    advance(3)
    assert [e["type"] for e in ok(client.get("/events"))] == ["short_ship"]
    assert all(p["cancelled_qty_t"] == 0 for p in ok(client.get("/purchase-orders")))


def test_scorecard_counts_waste_and_delay():
    load("act3")
    advance(3)  # the in-transit 20t lands at C3 after its order was cancelled
    s = ok(client.get("/scorecard"))
    assert s["wasted_t"] == 20
    assert s["total_cost"] == pytest.approx(s["procurement_spend"] + s["delay_cost"] + s["holding_cost"]
                                            + s["discount_cost"] + s["unmet_penalty"])


# ---------- input validation (the agent must not be able to invent stock, prices or money) ----------

@pytest.mark.parametrize("method,url,body", [
    ("post", "/shipments", {"truck_id": "T5", "stops": [{"customer_id": "C1", "qty_t": -500, "day": 2}]}),
    ("post", "/shipments", {"truck_id": "T5", "stops": []}),
    ("post", "/shipments/SH2/reassign", {"stops": [{"customer_id": "C1", "qty_t": -5, "day": 3}]}),
    ("post", "/purchase-orders", {"supplier_id": "S2", "qty_t": -10}),
    ("post", "/customers/C2/reschedule-request", {"order_id": "O2", "new_day": 4, "discount_pct": -50}),
    ("post", "/clock/advance", {"days": -5}),
])
def test_rejects_invalid_input(method, url, body):
    load("act2")
    assert getattr(client, method)(url, json=body).status_code == 422
    assert ok(client.get("/inventory"))["qty_t"] == 10


def test_po_price_cannot_be_chosen_by_caller():
    load("act2")
    po = ok(client.post("/purchase-orders", json={"supplier_id": "S2", "qty_t": 10, "price_per_t": 1}))
    today = next(s["price_today"] for s in ok(client.get("/suppliers")) if s["id"] == "S2")
    assert po["price_per_t"] == today
