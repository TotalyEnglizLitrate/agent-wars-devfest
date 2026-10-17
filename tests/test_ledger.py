"""Reconciliation ledger (tools/ledger.py) via POST /reconcile and GET /ledger/invariants."""

import pytest
from fastapi.testclient import TestClient

from agent_wars_devfest.tools.baseline import run_fcfs
from agent_wars_devfest.world.app import app

client = TestClient(app)


def ok(resp):
    assert resp.status_code < 400, (resp.status_code, resp.text)
    return resp.json()


def load(act):
    ok(client.post(f"/scenario/{act}"))


def advance(days=1):
    return ok(client.post("/clock/advance", json={"days": days}))


@pytest.mark.parametrize("act", ["act1", "act2", "act3"])
def test_scenarios_start_balanced(act):
    load(act)
    assert ok(client.get("/ledger/invariants")) == {"ok": True, "violations": []}


def test_act2_short_ship_becomes_supplier_credit():
    load("act2")
    advance()  # S1 loads only 12t of truck 2's 20t
    inv = ok(client.get("/ledger/invariants"))
    assert not inv["ok"] and "unrecorded short-ship" in inv["violations"][0]

    r = ok(client.post("/reconcile"))
    assert r["invariants"]["ok"]
    po1 = next(p for p in ok(client.get("/purchase-orders")) if p["id"] == "PO1")
    assert po1["cancelled_qty_t"] == 8
    credit = ok(client.get("/ledger"))[-1]
    assert credit["kind"] == "credit" and credit["amount"] == pytest.approx(-8 * po1["price_per_t"])


def test_act3_cancellation_fee_and_followups():
    load("act3")
    advance()  # C3 cancels, 10% fee
    inv = ok(client.get("/ledger/invariants"))
    assert any("no cancellation fee" in v for v in inv["violations"])

    r = ok(client.post("/reconcile"))
    assert r["invariants"]["ok"]
    fee = next(e for e in ok(client.get("/ledger")) if e["ref_type"] == "order")
    assert fee["kind"] == "fee" and fee["amount"] < 0  # money coming in
    assert any("still carries 20t to C3" in f for f in r["followups"])
    assert any("PO3" in f for f in r["followups"])


def test_reconcile_is_idempotent():
    load("act3")
    advance()
    ok(client.post("/reconcile"))
    before = ok(client.get("/ledger"))
    again = ok(client.post("/reconcile"))
    assert again["actions"] == [] and ok(client.get("/ledger")) == before


def test_cancelling_po_keeps_books_balanced():
    load("act3")
    advance()
    ok(client.post("/reconcile"))
    ok(client.post("/purchase-orders/PO3/cancel"))
    assert ok(client.get("/ledger/invariants"))["ok"]


@pytest.mark.parametrize("act", ["act1", "act2", "act3"])
def test_fcfs_run_reconciles_clean(act):
    run_fcfs(act)
    assert ok(client.post("/reconcile"))["invariants"]["ok"]


# ---------- reassign can't conjure sugar or erase history ----------

def test_reassign_cannot_exceed_whats_on_board():
    load("act2")
    advance()  # SH2 now carries 12t
    too_much = [{"customer_id": "C1", "qty_t": 20, "day": 3}]
    assert client.post("/shipments/SH2/reassign", json={"stops": too_much}).status_code == 400


def test_reassign_returns_leftover_to_warehouse():
    load("act2")
    advance()
    r = ok(client.post("/shipments/SH2/reassign", json={"stops": [{"customer_id": "C1", "qty_t": 10, "day": 3}]}))
    assert r["returned_to_warehouse_t"] == 2
    assert ok(client.get("/inventory"))["qty_t"] == 12
    assert ok(client.post("/reconcile"))["invariants"]["ok"]


def test_reassign_keeps_delivered_stops():
    load("act3")
    advance()  # SH1 delivered to C1 on day 3; SH2 still heading to C3 for day 4
    ok(client.post("/shipments/SH2/reassign", json={"stops": [{"customer_id": "C4", "qty_t": 20, "day": 4}]}))
    advance()
    c4 = next(o for o in ok(client.get("/orders")) if o["customer_id"] == "C4")
    assert c4["status"] == "fulfilled"
    assert ok(client.post("/reconcile"))["invariants"]["ok"]


def test_multidrop_reroute_after_first_drop_keeps_books_balanced():
    load("act2")
    advance()  # day 2, SH2 carries 12t
    ok(client.post("/reconcile"))  # book the 8t short-ship first
    drops = [{"customer_id": "C2", "qty_t": 6, "day": 3}, {"customer_id": "C1", "qty_t": 6, "day": 4}]
    ok(client.post("/shipments/SH2/reassign", json={"stops": drops}))
    advance()  # day 3: C2 drop delivered, 6t still on board
    r = ok(client.post("/shipments/SH2/reassign", json={"stops": [{"customer_id": "C3", "qty_t": 6, "day": 5}]}))
    assert [s["customer_id"] for s in r["stops"]] == ["C2", "C3"]  # delivered C2 drop survives
    assert ok(client.get("/ledger/invariants"))["ok"]
