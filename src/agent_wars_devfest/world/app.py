"""Dummy World API (PLAN.md §5).

Run: uv run uvicorn agent_wars_devfest.world.app:app --reload --port 8000
"""

import random

from fastapi import FastAPI, HTTPException

from agent_wars_devfest.models import (
    AdvanceRequest,
    CreatePORequest,
    DispatchRequest,
    Event,
    LedgerEntry,
    NewLedgerEntry,
    Order,
    PurchaseOrder,
    ReassignRequest,
    Shipment,
    RescheduleReply,
    RescheduleRequest,
)
from agent_wars_devfest.world.datagen import SEED, World
from agent_wars_devfest.world.scenarios import SCENARIOS

app = FastAPI(title="Supply World")
world: World = SCENARIOS["act2"]()
negotiations: list[dict] = []


def _load(name: str) -> None:
    global world
    world = SCENARIOS[name]()
    negotiations.clear()
    _fire_events_through(world.day)


def _get(d: dict, key: str, what: str):
    if key not in d:
        raise HTTPException(404, f"{what} {key} not found")
    return d[key]


def _ledger(kind: str, ref_type: str, ref_id: str, amount: float, note: str) -> LedgerEntry:
    e = LedgerEntry(id=world.next_id("L"), ref_type=ref_type, ref_id=ref_id, kind=kind,
                    amount=round(amount, 2), day=world.day, note=note)
    world.ledger.append(e)
    return e


# ---------- simulation ----------

def _apply_event(e: Event) -> None:
    p = e.payload
    match e.type:
        case "short_ship":
            sh = world.shipments[p["shipment_id"]]
            short = sum(s.qty_t for s in sh.stops) - p["new_qty_t"]
            for stop in reversed(sh.stops):  # trim the last stops first
                cut = min(stop.qty_t, short)
                stop.qty_t -= cut
                short -= cut
            sh.stops = [s for s in sh.stops if s.qty_t > 0]
        case "delay":
            sh = world.shipments[p["shipment_id"]]
            for stop in sh.stops:
                stop.day += p["delay_days"]
        case "buyer_cancel":
            world.orders[p["order_id"]].status = "cancelled"
        case "supplier_cancel":
            po = world.purchase_orders[p["po_id"]]
            if po.status in ("open", "shipped"):
                po.cancelled_qty_t = po.qty_t - po.received_qty_t
                po.status = "cancelled"
        case "price_spike":
            for pt in world.prices[p["supplier_id"]]:
                if pt.day > world.day:
                    pt.price_per_t = round(pt.price_per_t * (1 + p["pct"] / 100), 2)
        case "new_order":
            for o in p.get("orders", []):
                order = Order(**o)
                world.orders[order.id] = order


def _fire_events_through(day: int) -> None:
    for e in [e for e in world.scheduled_events if e.day <= day]:
        world.scheduled_events.remove(e)
        _apply_event(e)
        world.fired_events.append(e)


def _deliver(customer_id: str, qty: float) -> None:
    for o in sorted(world.orders.values(), key=lambda o: o.requested_day):
        if qty <= 0:
            break
        if o.customer_id == customer_id and o.status == "open":
            take = min(qty, o.qty_t - o.delivered_qty_t)
            o.delivered_qty_t += take
            world.deliveries.append({"order_id": o.id, "qty_t": take, "day": world.day})
            qty -= take
            if o.delivered_qty_t >= o.qty_t - 1e-6:
                o.status = "fulfilled"
    if qty > 1e-6:  # nobody ordered it: refused at the gate and written off
        world.deliveries.append({"order_id": None, "customer_id": customer_id, "qty_t": qty, "day": world.day})


def _maybe_short_ship(po: PurchaseOrder) -> None:
    """Unreliable suppliers sometimes ship 60%. Seeded per PO, so every run sees the same misses."""
    sup = world.suppliers[po.supplier_id]
    if random.Random(f"{SEED}-{po.id}-{po.supplier_id}").random() <= sup.reliability:
        return
    short = round(po.qty_t * 0.4, 1)
    po.cancelled_qty_t = short
    _ledger("credit", "po", po.id, -short * po.price_per_t, f"{sup.id} short-shipped {short}t")
    world.fired_events.append(Event(
        id=world.next_id("E"), day=world.day, type="short_ship",
        message=f"{sup.name}: sorry, only {po.qty_t - short:g}t of your {po.qty_t:g}t could be loaded today.",
        payload={"po_id": po.id, "supplier_id": sup.id, "new_qty_t": po.qty_t - short, "short_t": short},
    ))


def _progress_day() -> None:
    d = world.day
    for po in world.purchase_orders.values():
        if po.status == "open" and po.expected_day - 1 <= d:  # 1 day in transit; lead time covers prep
            po.status = "shipped"
            if po.id in world.live_pos:
                _maybe_short_ship(po)
        if po.status == "shipped" and po.expected_day <= d:
            po.status = "received"
            po.received_qty_t = po.qty_t - po.cancelled_qty_t
            if not po.shipment_ids:  # warehouse-bound
                world.inventory.qty_t += po.received_qty_t
            _ledger("receipt", "po", po.id, 0, f"Received {po.received_qty_t}t from {po.supplier_id}")
    for sh in world.shipments.values():
        if sh.status == "planned" and sh.dispatch_day <= d:
            sh.status = "in_transit"
        if sh.status == "in_transit":
            for stop in sh.stops:
                if stop.day == d:
                    _deliver(stop.customer_id, stop.qty_t)
            if all(s.day <= d for s in sh.stops):
                sh.status = "delivered"
                last = sh.stops[-1].customer_id if sh.stops else sh.supplier_id
                world.trucks[sh.truck_id].current_location = last


# ---------- endpoints ----------

@app.get("/clock")
def get_clock():
    return {"day": world.day, "scenario": world.scenario}


@app.post("/clock/advance")
def advance(req: AdvanceRequest):
    fired = []
    for _ in range(req.days):
        world.day += 1
        before = len(world.fired_events)
        _fire_events_through(world.day)
        fired += world.fired_events[before:]
        _progress_day()
    return {"day": world.day, "events_fired": fired}


@app.post("/scenario/{name}")
def load_scenario(name: str):
    if name not in SCENARIOS:
        raise HTTPException(404, f"unknown scenario {name}; try {list(SCENARIOS)}")
    _load(name)
    return {"scenario": name, "day": world.day, "events_fired": world.fired_events}


@app.get("/suppliers")
def suppliers():
    return [s.model_dump() | {"price_today": world.price_today(s.id)} for s in world.suppliers.values()]


@app.get("/suppliers/{sid}/prices")
def prices(sid: str, days: int = 90):
    series = _get(world.prices, sid, "supplier")
    return [p for p in series if world.day - days < p.day <= world.day]  # no peeking at the future


@app.get("/customers")
def customers():
    return list(world.customers.values())


@app.get("/trucks")
def trucks():
    return list(world.trucks.values())


@app.get("/travel-days")
def travel_days():
    return world.travel_days


@app.get("/inventory")
def inventory():
    return world.inventory


@app.get("/orders")
def orders(status: str | None = None):
    return [o for o in world.orders.values() if status is None or o.status == status]


@app.get("/purchase-orders")
def purchase_orders():
    return list(world.purchase_orders.values())


@app.post("/purchase-orders")
def create_po(req: CreatePORequest):
    s = _get(world.suppliers, req.supplier_id, "supplier")
    if req.qty_t <= 0 or req.qty_t > s.capacity_t_per_day * 3:
        raise HTTPException(400, f"qty must be in (0, {s.capacity_t_per_day * 3}]")
    price = req.price_per_t or world.price_today(s.id)
    po = PurchaseOrder(id=world.next_id("PO"), supplier_id=s.id, qty_t=req.qty_t, price_per_t=price,
                       order_day=world.day, expected_day=world.day + s.lead_time_days)
    world.purchase_orders[po.id] = po
    world.live_pos.add(po.id)
    _ledger("po", "po", po.id, req.qty_t * price, f"PO {req.qty_t}t from {s.id} @ {price}/t")
    return po


@app.post("/purchase-orders/{po_id}/cancel")
def cancel_po(po_id: str):
    po = _get(world.purchase_orders, po_id, "purchase order")
    if po.status != "open":
        raise HTTPException(409, f"{po_id} is {po.status}; only open (unshipped) POs can be cancelled")
    po.status, po.cancelled_qty_t = "cancelled", po.qty_t
    _ledger("refund", "po", po.id, -po.qty_t * po.price_per_t, f"Cancelled unshipped PO {po.id}")
    return po


@app.get("/shipments")
def shipments():
    return list(world.shipments.values())


@app.post("/shipments")
def dispatch(req: DispatchRequest):
    truck = _get(world.trucks, req.truck_id, "truck")
    if any(sh.truck_id == truck.id and sh.status in ("planned", "in_transit") for sh in world.shipments.values()):
        raise HTTPException(409, f"{truck.id} is already on a shipment")
    load = sum(s.qty_t for s in req.stops)
    if load > truck.capacity_t + 1e-6:
        raise HTTPException(400, f"{load}t exceeds {truck.id} capacity {truck.capacity_t}t")
    inv = world.inventory
    if load > inv.qty_t + 1e-6:
        raise HTTPException(400, f"only {inv.qty_t}t in the warehouse")
    depart = max(world.day, truck.available_day)
    for s in req.stops:
        _get(world.customers, s.customer_id, "customer")
        earliest = max(world.day + 1, depart + world.travel_days[inv.warehouse_id][s.customer_id])
        if s.day < earliest:
            raise HTTPException(400, f"{s.customer_id} can't be reached before day {earliest}")
    inv.qty_t -= load
    sh = Shipment(id=world.next_id("SH"), truck_id=truck.id, supplier_id=inv.warehouse_id,
                  dispatch_day=depart, stops=sorted(req.stops, key=lambda s: s.day))
    world.shipments[sh.id] = sh
    truck.available_day = sh.stops[-1].day if sh.stops else depart
    return sh


@app.post("/shipments/{sh_id}/reassign")
def reassign(sh_id: str, req: ReassignRequest):
    sh = _get(world.shipments, sh_id, "shipment")
    if sh.status in ("delivered", "cancelled"):
        raise HTTPException(409, f"{sh_id} is {sh.status}")
    truck = _get(world.trucks, req.truck_id or sh.truck_id, "truck")
    load = sum(s.qty_t for s in req.stops)
    if load > truck.capacity_t + 1e-6:
        raise HTTPException(400, f"{load}t exceeds {truck.id} capacity {truck.capacity_t}t")
    for s in req.stops:
        _get(world.customers, s.customer_id, "customer")
        if s.day < world.day:
            raise HTTPException(400, f"stop day {s.day} is in the past (today is {world.day})")
    sh.truck_id, sh.stops = truck.id, sorted(req.stops, key=lambda s: s.day)
    truck.available_day = max((s.day for s in sh.stops), default=world.day)
    return sh


@app.get("/allocation-state")
def allocation_state():
    """Everything the reallocation optimizer needs, precomputed.

    Stops must land on day >= today + 1: deliveries are processed when the clock advances.
    """
    d, t, wh = world.day, world.travel_days, world.inventory.warehouse_id
    active = [sh for sh in world.shipments.values() if sh.status in ("planned", "in_transit")]
    pending = {sh.id: [s for s in sh.stops if sh.status == "planned" or s.day > d] for sh in active}

    # Pending stop qty per customer, allocated to that customer's open orders earliest-first.
    incoming: dict[str, list[tuple[float, int]]] = {}
    for sh in active:
        for s in pending[sh.id]:
            incoming.setdefault(s.customer_id, []).append((s.qty_t, s.day))
    demand, issues = [], []
    open_orders = sorted((o for o in world.orders.values() if o.status == "open"), key=lambda o: o.requested_day)
    for o in open_orders:
        remaining = o.qty_t - o.delivered_qty_t
        committed, late_t = 0.0, 0.0
        queue = incoming.get(o.customer_id, [])
        while queue and committed < remaining - 1e-6:
            qty, day = queue.pop(0)
            take = min(qty, remaining - committed)
            committed += take
            if day > o.latest_acceptable_day:
                late_t += take
            if qty > take:
                queue.insert(0, (qty - take, day))
        demand.append(o.model_dump() | {"remaining_t": remaining, "committed_t": committed,
                                        "uncovered_t": remaining - committed})
        if remaining - committed > 1e-6 and o.earliest_day <= d + 2:  # only urgent gaps; later ones get bought for
            issues.append(f"{o.id} ({o.customer_id}) is short {remaining - committed:g}t "
                          f"(needs {remaining:g}t by day {o.latest_acceptable_day})")
        if late_t:
            issues.append(f"{o.id} ({o.customer_id}): {late_t:g}t arrives after day {o.latest_acceptable_day}")
    for cid, queue in incoming.items():
        if queue:
            surplus = sum(q for q, _ in queue)
            issues.append(f"{surplus:g}t heading to {cid} has no open order to fill (cancelled or over-delivered)")

    def reach(origin: str, ready: int) -> dict[str, int]:
        return {c: max(d + 1, ready + t[origin][c]) for c in world.customers}

    shipments_out = []
    for sh in active:
        arrival = reach(sh.supplier_id, sh.dispatch_day)
        for s in pending[sh.id]:
            arrival[s.customer_id] = s.day  # already-planned legs keep their day
        shipments_out.append({"shipment_id": sh.id, "truck_id": sh.truck_id, "supplier_id": sh.supplier_id,
                              "status": sh.status, "qty_t": sum(s.qty_t for s in pending[sh.id]),
                              "capacity_t": world.trucks[sh.truck_id].capacity_t,
                              "stops": pending[sh.id], "arrival_day_by_customer": arrival})
    busy = {sh.truck_id for sh in active}
    free_trucks = [tr.model_dump() | {"arrival_day_by_customer": reach(wh, max(d, tr.available_day))}
                   for tr in world.trucks.values() if tr.id not in busy]
    warehouse_incoming = [{"po_id": po.id, "qty_t": po.qty_t - po.cancelled_qty_t, "day": po.expected_day}
                          for po in world.purchase_orders.values()
                          if po.status in ("open", "shipped") and not po.shipment_ids]
    return {
        "day": d,
        "demand": demand,
        "shipments": shipments_out,
        "warehouse": world.inventory.model_dump() | {"incoming": warehouse_incoming},
        "free_trucks": free_trucks,
        "issues": issues,
    }


@app.get("/events")
def events(since: int = 0):
    return [e for e in world.fired_events if e.day >= since]


@app.post("/customers/{cid}/reschedule-request")
def reschedule(cid: str, req: RescheduleRequest) -> RescheduleReply:
    _get(world.customers, cid, "customer")
    order = _get(world.orders, req.order_id, "order")
    if order.customer_id != cid:
        raise HTTPException(400, f"{req.order_id} belongs to {order.customer_id}, not {cid}")

    queue = world.scripted_replies.get(cid, [])
    last_counter = next((n["reply"]["discount_pct"] for n in reversed(negotiations)
                         if n["customer_id"] == cid and n["reply"]["decision"] == "counter"), 0)
    if queue:
        reply = queue.pop(0)
    elif req.new_day > order.latest_acceptable_day + 2:
        reply = RescheduleReply(decision="reject", message="That's far too late for our production run.")
    elif req.discount_pct + 1e-6 < last_counter:
        reply = RescheduleReply(decision="counter", discount_pct=last_counter,
                                message=f"We still need {last_counter}% off to make day {req.new_day} work.")
    else:
        reply = RescheduleReply(decision="accept", message=f"OK, day {req.new_day} works for us.")

    if reply.decision == "accept":
        order.latest_acceptable_day = max(order.latest_acceptable_day, req.new_day)
        order.customer_flex_confirmed = True
        order.discount_pct = req.discount_pct
    negotiations.append({"day": world.day, "customer_id": cid, "request": req.model_dump(),
                         "reply": reply.model_dump()})
    return reply


@app.get("/negotiations")
def get_negotiations():
    return negotiations


UNMET_PENALTY_PER_T = 10_000  # INR per tonne never delivered by an order's latest day


@app.get("/scorecard")
def scorecard():
    """One yardstick for both the agent and the FCFS baseline (closing slide)."""
    live = [o for o in world.orders.values() if o.status != "cancelled"]
    total = sum(o.qty_t for o in live)
    delivered = sum(o.delivered_qty_t for o in live)
    by_id = world.orders
    filled = [(dl, by_id[dl["order_id"]]) for dl in world.deliveries if dl["order_id"]]
    wasted = sum(dl["qty_t"] for dl in world.deliveries if not dl["order_id"])
    on_time = sum(dl["qty_t"] for dl, o in filled if dl["day"] <= o.requested_day and o.status != "cancelled")
    delay_cost = sum(dl["qty_t"] / o.qty_t * o.delay_penalty_per_day * max(0, dl["day"] - o.requested_day)
                     for dl, o in filled)
    avg_price = sum(world.price_today(s) for s in world.suppliers) / len(world.suppliers)
    discount_cost = sum(o.discount_pct / 100 * o.delivered_qty_t * avg_price for o in live)
    unmet = sum(o.qty_t - o.delivered_qty_t for o in live if o.latest_acceptable_day < world.day)
    spend = sum(e.amount for e in world.ledger)
    return {
        "day": world.day,
        "fulfillment_rate": round(delivered / total, 3) if total else 1.0,
        "on_time_rate": round(on_time / total, 3) if total else 1.0,
        "delivered_t": delivered, "ordered_t": total, "unmet_t": unmet, "wasted_t": wasted,
        "procurement_spend": round(spend, 2), "delay_cost": round(delay_cost, 2),
        "discount_cost": round(discount_cost, 2), "unmet_penalty": unmet * UNMET_PENALTY_PER_T,
        "total_cost": round(spend + delay_cost + discount_cost + unmet * UNMET_PENALTY_PER_T, 2),
    }


@app.get("/ledger")
def ledger():
    return world.ledger


@app.post("/ledger")
def add_ledger_entry(req: NewLedgerEntry):
    """For tools/ledger.py `reconcile` only. The agent never calls this directly."""
    return _ledger(req.kind, req.ref_type, req.ref_id, req.amount, req.note)


_load("act2")  # default scenario on startup
