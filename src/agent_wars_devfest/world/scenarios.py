"""Demo scenarios act1 / act2 / act3 (PLAN.md §8) layered on the seeded base world.

Each scenario starts the clock one day before its key event, so the agent's
first `advance_clock` fires it. Events with day <= start day fire on load.
"""

from agent_wars_devfest.models import (
    Event,
    LedgerEntry,
    Order,
    PurchaseOrder,
    RescheduleReply,
    Shipment,
    Stop,
)
from agent_wars_devfest.world.datagen import World, generate_base_world


def _order(w: World, cid: str, qty: float, day: int, *, slack: int = 2, penalty: float = 2_000,
           split: bool = True, priority: int = 2) -> Order:
    o = Order(
        id=w.next_id("O"), customer_id=cid, qty_t=qty, requested_day=day,
        earliest_day=day, latest_acceptable_day=day + slack,
        delay_penalty_per_day=penalty, accepts_split=split, priority=priority,
    )
    w.orders[o.id] = o
    return o


def _po(w: World, sid: str, qty: float, order_day: int, *, status: str = "open",
        shipment_ids: list[str] | None = None) -> PurchaseOrder:
    s = w.suppliers[sid]
    price = next(p.price_per_t for p in w.prices[sid] if p.day == order_day)
    po = PurchaseOrder(
        id=w.next_id("PO"), supplier_id=sid, qty_t=qty, price_per_t=price, order_day=order_day,
        expected_day=order_day + s.lead_time_days, status=status, shipment_ids=shipment_ids or [],
    )
    w.purchase_orders[po.id] = po
    w.ledger.append(LedgerEntry(
        id=w.next_id("L"), ref_type="po", ref_id=po.id, kind="po",
        amount=round(qty * price, 2), day=order_day, note=f"PO {qty}t from {sid} @ {price}/t",
    ))
    return po


def _shipment(w: World, truck: str, sid: str, dispatch: int, stops: list[tuple[str, float, int]],
              status: str = "planned") -> Shipment:
    sh = Shipment(
        id=w.next_id("SH"), truck_id=truck, supplier_id=sid, dispatch_day=dispatch,
        stops=[Stop(customer_id=c, qty_t=q, day=d) for c, q, d in stops], status=status,
    )
    w.shipments[sh.id] = sh
    w.trucks[truck].available_day = max(d for _, _, d in stops)
    return sh


def _set_travel(w: World, a: str, b: str, days: int) -> None:
    """Pin a leg so hand-written stop days agree with the travel matrix."""
    w.travel_days[a][b] = w.travel_days[b][a] = days


def _event(w: World, day: int, type_: str, message: str, **payload) -> Event:
    e = Event(id=w.next_id("E"), day=day, type=type_, message=message, payload=payload)
    w.scheduled_events.append(e)
    return e


def act1() -> World:
    """Day 0: procurement. A week of demand, almost no stock, no POs yet."""
    w = generate_base_world()
    w.scenario, w.day = "act1", 0
    for cid, qty, day in [("C1", 30, 2), ("C2", 20, 3), ("C3", 25, 4),
                          ("C4", 20, 5), ("C5", 30, 6), ("C6", 25, 7)]:
        _order(w, cid, qty, day)
    _event(w, 0, "new_order",
           "Weekly demand is in: 6 factories need 150t of sugar between day 2 and day 7. "
           "Warehouse has 20t. Nothing on order.",
           order_ids=list(w.orders))
    _event(w, 1, "price_spike", "Heads up, S3 is raising prices ~8% from tomorrow, cane shortage.",
           supplier_id="S3", pct=8)
    return w


def act2() -> World:
    """Day 1 -> 2: shortage. The sugar example from PLAN.md §2, plus S1 short-ships truck 2."""
    w = generate_base_world()
    w.scenario, w.day = "act2", 1
    w.inventory.qty_t = 10
    for a, b in [("S1", "C1"), ("S1", "C2"), ("WH1", "C1"), ("WH1", "C2")]:
        _set_travel(w, a, b, 1)  # truck 2 must be divertible to Factory A on day 3
    a = _order(w, "C1", 30, 2, slack=1, penalty=5_000, priority=1)  # Factory A: 30t on day 2
    b = _order(w, "C2", 20, 3, slack=0, penalty=2_000)              # Factory B: 20t on day 3, no slack: a delay needs consent
    _order(w, "C3", 20, 5)
    _order(w, "C4", 15, 6)

    # S1 dispatches two 20t trucks on day 1 and day 2, arriving day 2 and day 3.
    sh1 = _shipment(w, "T1", "S1", 1, [(a.customer_id, 20, 2)], status="in_transit")
    sh2 = _shipment(w, "T2", "S1", 2, [(b.customer_id, 20, 3)])
    _po(w, "S1", 40, 0, status="shipped", shipment_ids=[sh1.id, sh2.id])
    _po(w, "S2", 20, 2)  # regular replenishment, arrives at the warehouse on day 3

    _event(w, 2, "short_ship",
           "S1 dispatch here - truck 2 had a loading problem at the mill, only 12t went on. "
           "Sorry, rest maybe next week.",
           shipment_id=sh2.id, supplier_id="S1", new_qty_t=12, short_t=8)

    # Factory B haggles first: accepts a delay only for a 5% discount.
    w.scripted_replies["C2"] = [RescheduleReply(
        decision="counter", discount_pct=5,
        message="Day 4 hurts our line schedule. We can do it for 5% off the invoice.",
    )]
    return w


def act3() -> World:
    """Day 2 -> 3: a customer cancels with goods in transit and an upstream PO still open."""
    w = generate_base_world()
    w.scenario, w.day = "act3", 2
    w.inventory.qty_t = 30
    _set_travel(w, "S2", "C1", 1)
    _order(w, "C1", 20, 3)
    c3 = _order(w, "C3", 25, 4, penalty=1_500)
    _order(w, "C4", 20, 4, slack=2, penalty=3_000, priority=1)  # no truck assigned yet
    _order(w, "C5", 20, 6)

    sh_c1 = _shipment(w, "T1", "S2", 2, [("C1", 20, 3)], status="in_transit")
    sh_c3 = _shipment(w, "T3", "S3", 2, [("C3", 20, 4)], status="in_transit")
    _po(w, "S2", 20, 1, status="shipped", shipment_ids=[sh_c1.id])
    _po(w, "S3", 20, 0, status="shipped", shipment_ids=[sh_c3.id])
    _po(w, "S5", 25, 2)  # top-up for C3's order: ships day 4, so still cancellable on day 3

    _event(w, 3, "buyer_cancel",
           f"Email from {w.customers['C3'].name}: plant shutdown for maintenance, "
           "please cancel our sugar order this week. We'll pay the 10% fee.",
           order_id=c3.id, customer_id="C3", cancellation_fee_pct=10)
    return w


SCENARIOS = {"act1": act1, "act2": act2, "act3": act3}
