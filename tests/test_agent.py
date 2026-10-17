"""Agent tools, guardrails, scripted turns, greedy solver fallbacks and lessons memory."""

import pytest
from fastapi.testclient import TestClient

from agent_wars_devfest.agent import memory, solvers, tools
from agent_wars_devfest.agent.agent import build_agent, scripted_tick
from agent_wars_devfest.tools.baseline import END_DAY
from agent_wars_devfest.world.app import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def fresh(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_MEMORY_PATH", str(tmp_path / "lessons.json"))
    tools.use_client(client)


def run_scripted(act: str) -> dict:
    day = tools.load_scenario(act)["day"]
    while True:
        scripted_tick()
        if day >= END_DAY:
            break
        day = tools.advance_clock(1)["day"]
    return client.get("/scorecard").json()


# ---------- ADK wiring ----------

def test_adk_builds_every_tool_schema():
    from google.adk.tools import FunctionTool

    for fn in tools.ALL_TOOLS:
        assert FunctionTool(fn)._get_declaration().name == fn.__name__


def test_instruction_includes_lessons():
    memory.save_agent_lesson("Factory B always haggles.")
    agent = build_agent()
    assert callable(agent.instruction)
    assert "Factory B always haggles." in agent.instruction(None)


# ---------- guardrails ----------

def test_big_po_needs_escalation():
    tools.load_scenario("act1")
    assert "error" in tools.create_purchase_order("S1", 60)  # ~₹23L
    tools.escalate_to_human("weekly buy", 3_000_000)
    assert "error" not in tools.create_purchase_order("S1", 60)


def test_negotiation_round_limit():
    tools.load_scenario("act2")
    for _ in range(tools.MAX_ROUNDS):
        assert "decision" in tools.request_reschedule("C2", "O2", 4, 0)
    assert "limit" in tools.request_reschedule("C2", "O2", 4, 0)["error"]


def test_commit_refuses_plan_with_open_asks():
    tools.load_scenario("act2")
    tools.advance_clock(1)
    plan = tools.run_reallocation_optimizer()
    assert plan["asks"]
    assert "asks" in tools.commit_plan()["error"]


def test_tool_errors_come_back_as_data():
    tools.load_scenario("act3")
    out = tools.cancel_purchase_order("PO1")  # already shipped
    assert out["status"] == 409 and "error" in out


# ---------- greedy fallbacks ----------

def test_act2_plan_asks_factory_b_for_day_4():
    tools.load_scenario("act2")
    tools.advance_clock(1)
    tools.run_reallocation_optimizer()  # day-2 short-ship leaves B 8t short with nothing arriving by day 3
    plan = solvers.greedy_reallocate(client.get("/allocation-state").json(), client.get("/travel-days").json())
    assert [(a["customer_id"], a["order_id"], a["new_day"]) for a in plan["asks"]] == [("C2", "O2", 4)]


def test_orphaned_load_is_rerouted_not_wasted():
    tools.load_scenario("act3")
    tools.advance_clock(1)  # C3 cancels with 20t on the road to it
    plan = tools.run_reallocation_optimizer()
    moved = [a for a in plan["actions"] if a["tool"] == "reassign_shipment"]
    assert moved and all(s["customer_id"] != "C3" for a in moved for s in a["stops"])


def test_procurement_prefers_reliable_supplier_once_cheap_one_is_flaky():
    tools.load_scenario("act1")
    state = client.get("/allocation-state").json() | {"open_pos": {}}
    sups = client.get("/suppliers").json()
    hist = {s["id"]: client.get(f"/suppliers/{s['id']}/prices").json() for s in sups}
    trusting = solvers.greedy_procure(sups, hist, state, {})
    burned = solvers.greedy_procure(sups, hist, state, {"S1": 0.2})
    assert trusting["purchases"][0]["supplier_id"] == "S1"
    assert burned["purchases"][0]["supplier_id"] != "S1"


# ---------- scripted end to end ----------

def test_scripted_act2_negotiates_and_stays_balanced():
    score = run_scripted("act2")
    negs = client.get("/negotiations").json()
    assert [n["reply"]["decision"] for n in negs] == ["counter", "accept"]
    assert score["fulfillment_rate"] == 1
    assert client.get("/ledger/invariants").json()["ok"]


def test_scripted_act3_wastes_nothing_and_beats_fcfs():
    from agent_wars_devfest.tools.baseline import run_fcfs

    score = run_scripted("act3")
    assert score["wasted_t"] == 0 and score["fulfillment_rate"] == 1
    assert client.get("/ledger/invariants").json()["ok"]
    assert score["total_cost"] < run_fcfs("act3")["total_cost"]


# ---------- learning loop ----------

def test_second_run_uses_learned_discount_and_saves_a_round():
    run_scripted("act2")
    memory.record_run(client, "act2")
    assert memory.load()["customer_profiles"]["C2"]["counter_discount_pct"] == 5

    run_scripted("act2")
    negs = client.get("/negotiations").json()
    assert [(n["request"]["discount_pct"], n["reply"]["decision"]) for n in negs] == [(5, "accept")]


def test_observed_reliability_drops_for_a_flaky_supplier():
    run_scripted("act1")
    mem = memory.record_run(client, "act1")
    obs = memory.observed_reliability(mem, client.get("/suppliers").json())
    assert obs["S1"] < 0.70
    assert "S1" in mem["lessons"]["supplier:S1"]


def test_extract_lessons_is_pure_and_counts_only_our_pos():
    report = {
        "act": "actX",
        "scorecard": {"total_cost": 100.0, "fulfillment_rate": 1, "on_time_rate": 0.5, "wasted_t": 3, "unmet_t": 0},
        "events": [{"type": "short_ship", "payload": {"po_id": "PO2"}}],
        "purchase_orders": [
            {"id": "PO1", "supplier_id": "S1", "status": "received", "placed_by": "api"},
            {"id": "PO2", "supplier_id": "S1", "status": "received", "placed_by": "api"},
            {"id": "PO3", "supplier_id": "S1", "status": "received", "placed_by": "scenario"},
            {"id": "PO4", "supplier_id": "S1", "status": "open", "placed_by": "api"},
        ],
        "negotiations": [{"customer_id": "C9", "reply": {"decision": "counter", "discount_pct": 7}}],
        "suppliers": [{"id": "S1", "name": "Acme", "reliability": 0.7}],
    }
    empty = memory.load()
    mem = memory.extract_lessons(report, empty)
    assert empty["supplier_stats"] == {}  # input untouched
    assert mem["supplier_stats"]["S1"] == {"n": 2, "misses": 1}
    assert mem["customer_profiles"]["C9"]["counter_discount_pct"] == 7
    assert {"waste", "late", "runs:actX", "supplier:S1", "customer:C9"} <= set(mem["lessons"])
