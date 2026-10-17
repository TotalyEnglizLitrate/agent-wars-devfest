"""The ADK agent (Gemini) plus a scripted fallback that calls the same tools in a fixed order."""

import os

from agent_wars_devfest.agent import memory, tools
from agent_wars_devfest.agent.prompts import BASE_PROMPT


def build_instruction(ctx=None) -> str:
    """ADK InstructionProvider: re-read lessons on every turn so new ones apply immediately."""
    return BASE_PROMPT + memory.lessons_block()


def build_agent():
    from google.adk.agents import LlmAgent

    kwargs = {"model": os.environ["AGENT_MODEL"]} if os.environ.get("AGENT_MODEL") else {}  # else ADK default
    return LlmAgent(name="supply_coordinator", instruction=build_instruction, tools=tools.ALL_TOOLS,
                    description="Buys sugar, re-plans deliveries, negotiates delays, reconciles the books.",
                    **kwargs)


class LlmRunner:
    """One ADK session per act, so the agent remembers earlier turns within the act."""

    def __init__(self, act: str):
        from google.adk.runners import InMemoryRunner

        self.runner = InMemoryRunner(agent=build_agent(), app_name="supply_coordinator")
        self.session_id = f"{act}-session"

    async def tick(self, day: int) -> str:
        events = await self.runner.run_debug(
            f"It is now day {day}. Run your turn.", session_id=self.session_id, quiet=True)
        texts = [p.text for e in events if e.content and e.content.parts for p in e.content.parts
                 if getattr(p, "text", None) and e.author != "user"]
        return texts[-1] if texts else ""


def scripted_tick() -> str:
    """Deterministic version of the turn in BASE_PROMPT, for when the LLM is unavailable or flaky."""
    day = tools.get_clock()["day"]
    events = [e for e in tools.get_events(since_day=day) if e["day"] == day]
    notes = [f"event: {e['message']}" for e in events]

    proc = tools.run_procurement_optimizer()
    for buy in proc["purchases"]:
        price = next(s["price_today"] for s in tools.get_suppliers() if s["id"] == buy["supplier_id"])
        if buy["qty_t"] * price > tools.APPROVAL_THRESHOLD_INR:
            tools.escalate_to_human(f"buy {buy['qty_t']:g}t from {buy['supplier_id']}: {buy['reason']}",
                                    buy["qty_t"] * price)
        po = tools.create_purchase_order(buy["supplier_id"], buy["qty_t"])
        if "error" not in po:
            notes.append(f"bought {buy['qty_t']:g}t from {buy['supplier_id']} ({buy['reason']})")
    for po_id in proc["cancel_pos"]:
        if "error" not in tools.cancel_purchase_order(po_id):
            notes.append(f"cancelled unneeded {po_id}")

    profiles = memory.load()["customer_profiles"]
    plan = tools.run_reallocation_optimizer()
    for _ in range(tools.MAX_ROUNDS):
        if not plan["asks"]:
            break
        for ask in plan["asks"]:
            offer = profiles.get(ask["customer_id"], {}).get("counter_discount_pct", 0.0)  # learned
            reply = tools.request_reschedule(ask["customer_id"], ask["order_id"], ask["new_day"], offer)
            story = f"asked {ask['customer_id']} to take {ask['order_id']} on day {ask['new_day']}"
            story += f" offering {offer:g}% (learned)" if offer else ""
            if reply.get("decision") == "counter":
                story += f": countered at {reply['discount_pct']:g}%"
                reply = tools.request_reschedule(ask["customer_id"], ask["order_id"], ask["new_day"],
                                                 reply["discount_pct"])
            notes.append(f"{story}: {reply.get('decision', reply.get('error'))}")
        plan = tools.run_reallocation_optimizer()
    if plan["asks"]:
        tools.escalate_to_human(f"unresolved delays: {plan['asks']}", plan["cost_estimate"])
    elif plan["actions"]:
        result = tools.commit_plan()
        notes.append(f"committed {result.get('committed', 0)} truck move(s) ({plan['explanation']})")

    rec = tools.reconcile()
    notes += [f"reconciled: {a}" for a in rec["actions"]]
    if not tools.check_ledger_invariants()["ok"]:
        notes.append("ledger invariants FAILED")
    summary = "; ".join(notes) or "quiet day, nothing to do"
    tools.log_decision(summary)
    return summary
