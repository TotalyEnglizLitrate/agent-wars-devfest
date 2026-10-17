"""Dummy World API (PLAN.md §5).

Run: uv run uvicorn agent_wars_devfest.world.app:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException

from agent_wars_devfest.models import (
    AdvanceRequest,
    CreatePORequest,
    Event,
    LedgerEntry,
    NewLedgerEntry,
    Order,
    PurchaseOrder,
    ReassignRequest,
    RescheduleReply,
    RescheduleRequest,
)
from agent_wars_devfest.world.datagen import World
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
            qty -= take
            if o.delivered_qty_t >= o.qty_t - 1e-6:
                o.status = "fulfilled"


def _progress_day() -> None:
    d = world.day
    for po in world.purchase_orders.values():
        if po.status == "open" and po.expected_day - 1 <= d:  # 1 day in transit; lead time covers prep
            po.status = "shipped"
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


@app.get("/ledger")
def ledger():
    return world.ledger


@app.post("/ledger")
def add_ledger_entry(req: NewLedgerEntry):
    """For tools/ledger.py `reconcile` only. The agent never calls this directly."""
    return _ledger(req.kind, req.ref_type, req.ref_id, req.amount, req.note)


_load("act2")  # default scenario on startup
