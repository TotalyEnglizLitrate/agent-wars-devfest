"""Reconciliation ledger (PLAN.md §6.3).

Runs inside the world via POST /reconcile and GET /ledger/invariants. The agent only ever
calls those two endpoints; it never writes ledger entries itself.

Invariants, per PO:
  physical: linked-shipment qty + cancelled qty == ordered qty   (direct-to-customer POs)
            received qty == ordered - cancelled                  (once received)
  money:    sum of its ledger entries == (ordered - cancelled) * price
and every cancelled customer order carries a cancellation fee entry.
"""

from agent_wars_devfest.models import LedgerEntry, PurchaseOrder
from agent_wars_devfest.world.datagen import World

EPS = 0.01
DEFAULT_CANCEL_FEE_PCT = 10


def _shipped_qty(w: World, po: PurchaseOrder) -> float:
    return sum(s.qty_t for sid in po.shipment_ids for s in w.shipments[sid].stops)


def _po_money(w: World, po_id: str) -> float:
    return sum(e.amount for e in w.ledger if e.ref_type == "po" and e.ref_id == po_id)


def _has_fee(w: World, order_id: str) -> bool:
    return any(e.ref_type == "order" and e.ref_id == order_id and e.kind == "fee" for e in w.ledger)


def _avg_price(w: World) -> float:
    return sum(w.price_today(s) for s in w.suppliers) / len(w.suppliers)


def _fee_pct(w: World, order_id: str) -> float:
    return next((e.payload.get("cancellation_fee_pct", DEFAULT_CANCEL_FEE_PCT) for e in w.fired_events
                 if e.type == "buyer_cancel" and e.payload.get("order_id") == order_id), DEFAULT_CANCEL_FEE_PCT)


def _post(w: World, kind: str, ref_type: str, ref_id: str, amount: float, note: str) -> LedgerEntry:
    e = LedgerEntry(id=w.next_id("L"), ref_type=ref_type, ref_id=ref_id, kind=kind,
                    amount=round(amount, 2), day=w.day, note=note)
    w.ledger.append(e)
    return e


def check_invariants(w: World) -> dict:
    violations = []
    for po in w.purchase_orders.values():
        if po.shipment_ids and po.status != "cancelled":
            shipped = _shipped_qty(w, po)
            if abs(shipped + po.cancelled_qty_t - po.qty_t) > EPS:
                violations.append(f"{po.id}: {shipped:g}t on trucks + {po.cancelled_qty_t:g}t cancelled "
                                  f"!= {po.qty_t:g}t ordered (unrecorded short-ship?)")
        if po.status == "received" and abs(po.received_qty_t - (po.qty_t - po.cancelled_qty_t)) > EPS:
            violations.append(f"{po.id}: received {po.received_qty_t:g}t but owed "
                              f"{po.qty_t - po.cancelled_qty_t:g}t")
        owed = (po.qty_t - po.cancelled_qty_t) * po.price_per_t
        booked = _po_money(w, po.id)
        if abs(booked - owed) > 1:
            violations.append(f"{po.id}: ledger says ₹{booked:,.0f} but {po.qty_t - po.cancelled_qty_t:g}t "
                              f"@ ₹{po.price_per_t:,.0f} = ₹{owed:,.0f}")
    for o in w.orders.values():
        if o.status == "cancelled" and not _has_fee(w, o.id):
            violations.append(f"{o.id} ({o.customer_id}) cancelled with no cancellation fee booked")
    if w.inventory.qty_t < -EPS:
        violations.append(f"warehouse is negative: {w.inventory.qty_t:g}t")
    return {"ok": not violations, "violations": violations}


def reconcile(w: World) -> dict:
    """Bring POs and the ledger in line with what physically happened. Safe to call repeatedly."""
    actions, followups = [], []

    for po in w.purchase_orders.values():
        # Supplier short-shipped onto a direct truck: record the missing tonnes as cancelled.
        if po.shipment_ids and po.status != "cancelled":
            missing = po.qty_t - po.cancelled_qty_t - _shipped_qty(w, po)
            if missing > EPS:
                po.cancelled_qty_t += missing
                if po.status == "received":
                    po.received_qty_t = po.qty_t - po.cancelled_qty_t
                actions.append(f"{po.id}: recorded {missing:g}t short-shipped by {po.supplier_id}")
        # Money follows tonnes: credit or charge the difference.
        diff = (po.qty_t - po.cancelled_qty_t) * po.price_per_t - _po_money(w, po.id)
        if abs(diff) > 1:
            kind = "credit" if diff < 0 else "fee"
            _post(w, kind, "po", po.id, diff, f"reconcile: {po.supplier_id} owes us" if diff < 0
                  else f"reconcile: we owe {po.supplier_id}")
            actions.append(f"{po.id}: booked {kind} ₹{diff:,.0f}")

    live_customers = {o.customer_id for o in w.orders.values() if o.status == "open"}
    for o in w.orders.values():
        if o.status != "cancelled":
            continue
        if not _has_fee(w, o.id):
            pct = _fee_pct(w, o.id)
            fee = pct / 100 * o.qty_t * _avg_price(w)
            _post(w, "fee", "order", o.id, -fee, f"{pct:g}% cancellation fee from {o.customer_id}")
            actions.append(f"{o.id}: booked {pct:g}% cancellation fee ₹{fee:,.0f} receivable")
        if o.customer_id in live_customers:
            continue
        for sh in w.shipments.values():
            if sh.status not in ("planned", "in_transit"):
                continue
            stuck = sum(s.qty_t for s in sh.stops if s.customer_id == o.customer_id and s.day > w.day)
            if stuck > EPS:
                followups.append(f"{sh.id} ({sh.truck_id}) still carries {stuck:g}t to {o.customer_id}: "
                                 f"reroute it with run_reallocation_optimizer + reassign_shipment")
        open_pos = [p for p in w.purchase_orders.values() if p.status == "open"]
        if open_pos:
            followups.append("unshipped POs you could cancel or shrink: "
                             + ", ".join(f"{p.id} ({p.qty_t:g}t from {p.supplier_id})" for p in open_pos))

    net = sum(e.amount for e in w.ledger)
    return {"actions": actions, "followups": followups, "net_spend": round(net, 2),
            "invariants": check_invariants(w)}
