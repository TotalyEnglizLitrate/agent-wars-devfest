"""Shared schemas (PLAN.md §4). Owned by Person A; ask before changing."""

from typing import Literal

from pydantic import BaseModel, Field


class Supplier(BaseModel):
    id: str
    name: str
    city: str
    capacity_t_per_day: float
    lead_time_days: int
    reliability: float  # 0-1, probability a delivery arrives in full and on time
    base_price_per_t: float
    recent_deliveries: list[bool] = []  # last 10 deliveries, True = on time and in full


class PricePoint(BaseModel):
    supplier_id: str
    day: int  # history uses days <= 0; day 0 is "today" at scenario start
    price_per_t: float


class Customer(BaseModel):
    id: str
    name: str
    city: str


class Order(BaseModel):
    id: str
    customer_id: str
    qty_t: float
    requested_day: int
    earliest_day: int
    latest_acceptable_day: int
    delay_penalty_per_day: float
    accepts_split: bool
    priority: int  # 1 (highest) - 3
    status: Literal["open", "fulfilled", "cancelled"] = "open"
    customer_flex_confirmed: bool = False  # set when a reschedule is accepted
    discount_pct: float = 0.0  # agreed discount from a counter-offer
    delivered_qty_t: float = 0.0


class Truck(BaseModel):
    id: str
    capacity_t: float = 20.0
    current_location: str
    available_day: int


class Stop(BaseModel):
    customer_id: str
    qty_t: float = Field(gt=0)
    day: int = Field(ge=0)


class Shipment(BaseModel):
    id: str
    truck_id: str
    supplier_id: str
    dispatch_day: int
    stops: list[Stop]
    status: Literal["planned", "in_transit", "delivered", "cancelled"] = "planned"


class PurchaseOrder(BaseModel):
    id: str
    supplier_id: str
    qty_t: float
    price_per_t: float
    order_day: int
    expected_day: int
    status: Literal["open", "shipped", "received", "cancelled"] = "open"
    received_qty_t: float = 0.0
    cancelled_qty_t: float = 0.0
    shipment_ids: list[str] = []  # non-empty = shipped direct to customers, bypassing the warehouse
    placed_by: Literal["scenario", "api"] = "scenario"


class LedgerEntry(BaseModel):
    id: str
    ref_type: str  # "po" | "shipment" | "order"
    ref_id: str
    kind: Literal["po", "receipt", "credit", "fee", "refund"]
    amount: float  # INR; positive = money out, negative = money in
    day: int
    note: str = ""


class Inventory(BaseModel):
    warehouse_id: str
    qty_t: float
    capacity_t: float
    holding_cost_per_t_day: float


class Event(BaseModel):
    id: str
    day: int
    type: Literal["short_ship", "delay", "buyer_cancel", "supplier_cancel", "price_spike", "new_order"]
    message: str  # messy human-readable text the agent has to interpret
    payload: dict  # structured details, e.g. {"shipment_id": "SH2", "new_qty_t": 10}


# ---- request / response bodies ----

class AdvanceRequest(BaseModel):
    days: int = Field(default=1, ge=1, le=30)


class CreatePORequest(BaseModel):
    supplier_id: str
    qty_t: float = Field(gt=0)  # always bought at the supplier's price today


class DispatchRequest(BaseModel):
    """Send a free truck out from the warehouse with stock on hand."""
    truck_id: str
    stops: list[Stop] = Field(min_length=1)


class ReassignRequest(BaseModel):
    stops: list[Stop]
    truck_id: str | None = None


class RescheduleRequest(BaseModel):
    order_id: str
    new_day: int = Field(ge=0)
    discount_pct: float = Field(default=0.0, ge=0, le=50)  # discount we offer in exchange for the delay


class RescheduleReply(BaseModel):
    decision: Literal["accept", "reject", "counter"]
    discount_pct: float = 0.0  # set when decision == "counter"
    message: str = ""
