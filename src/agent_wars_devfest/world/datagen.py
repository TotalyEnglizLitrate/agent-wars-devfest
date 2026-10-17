"""Seeded base world: suppliers, prices, customers, trucks, travel days (PLAN.md §4).

Orders, shipments, POs and events are layered on top by scenarios.py.
"""

from dataclasses import dataclass, field

import numpy as np
from faker import Faker

from agent_wars_devfest.models import (
    Customer,
    Event,
    Inventory,
    LedgerEntry,
    Order,
    PricePoint,
    PurchaseOrder,
    RescheduleReply,
    Shipment,
    Supplier,
    Truck,
)

SEED = 42
HISTORY_DAYS = 90
HORIZON_DAYS = 7
WAREHOUSE = "WH1"

# (base price INR/t, reliability, capacity t/day, lead time days)
# S1 is the cheap-but-flaky one, S2 the pricey-but-reliable one: the built-in trade-off.
SUPPLIER_PROFILES = [
    (38_000, 0.70, 60, 1),
    (42_000, 0.98, 50, 1),
    (40_000, 0.90, 40, 2),
    (39_500, 0.85, 45, 2),
    (41_000, 0.93, 55, 3),
]


@dataclass
class World:
    day: int = 0
    scenario: str = "base"
    suppliers: dict[str, Supplier] = field(default_factory=dict)
    prices: dict[str, list[PricePoint]] = field(default_factory=dict)
    customers: dict[str, Customer] = field(default_factory=dict)
    trucks: dict[str, Truck] = field(default_factory=dict)
    travel_days: dict[str, dict[str, int]] = field(default_factory=dict)
    inventory: Inventory | None = None
    orders: dict[str, Order] = field(default_factory=dict)
    shipments: dict[str, Shipment] = field(default_factory=dict)
    purchase_orders: dict[str, PurchaseOrder] = field(default_factory=dict)
    ledger: list[LedgerEntry] = field(default_factory=list)
    scheduled_events: list[Event] = field(default_factory=list)
    fired_events: list[Event] = field(default_factory=list)
    # customer_id -> queue of scripted replies; consumed in order, then default policy applies
    scripted_replies: dict[str, list[RescheduleReply]] = field(default_factory=dict)
    counters: dict[str, int] = field(default_factory=dict)

    def next_id(self, prefix: str) -> str:
        self.counters[prefix] = self.counters.get(prefix, 0) + 1
        return f"{prefix}{self.counters[prefix]}"

    def price_today(self, supplier_id: str) -> float:
        return next(p.price_per_t for p in reversed(self.prices[supplier_id]) if p.day <= self.day)


def generate_base_world(seed: int = SEED) -> World:
    Faker.seed(seed)
    fake = Faker("en_IN")
    rng = np.random.default_rng(seed)
    w = World()

    for i, (base, rel, cap, lead) in enumerate(SUPPLIER_PROFILES, start=1):
        sid = f"S{i}"
        if sid == "S1":
            # Deterministic so the demo line "missed 3 of its last 10" is always true.
            recent = [True, False, True, True, False, True, True, True, False, True]
        else:
            recent = [bool(x) for x in rng.random(10) < rel]
        w.suppliers[sid] = Supplier(
            id=sid,
            name=f"{fake.last_name()} Sugars Pvt Ltd",
            city=fake.city(),
            capacity_t_per_day=cap,
            lead_time_days=lead,
            reliability=rel,
            base_price_per_t=base,
            recent_deliveries=recent,
        )

        # Random walk + weekly seasonality + supplier-specific offset.
        days = np.arange(-HISTORY_DAYS + 1, HORIZON_DAYS + 1)
        walk = np.cumsum(rng.normal(0, 180, len(days)))
        weekly = 500 * np.sin(2 * np.pi * days / 7)
        offset = rng.normal(0, 300)
        series = base + walk + weekly + offset
        w.prices[sid] = [
            PricePoint(supplier_id=sid, day=int(d), price_per_t=round(float(p), 2))
            for d, p in zip(days, series)
        ]

    for i in range(1, 7):
        cid = f"C{i}"
        w.customers[cid] = Customer(id=cid, name=f"{fake.company()} Foods", city=fake.city())

    for i in range(1, 9):
        w.trucks[f"T{i}"] = Truck(id=f"T{i}", current_location=WAREHOUSE, available_day=0)

    locations = [WAREHOUSE, *w.suppliers, *w.customers]
    for a in locations:
        w.travel_days[a] = {}
    for i, a in enumerate(locations):
        for b in locations[i:]:
            d = 0 if a == b else int(rng.integers(1, 3))  # 1-2 days, symmetric
            w.travel_days[a][b] = w.travel_days[b][a] = d

    w.inventory = Inventory(warehouse_id=WAREHOUSE, qty_t=20, capacity_t=200, holding_cost_per_t_day=150)
    return w
