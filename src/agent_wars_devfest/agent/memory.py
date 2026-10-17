"""Lessons memory: the agent gets better across runs (PLAN.md §7.1).

Lessons are derived in code from the world's records, never by the LLM, so their numbers are
always true. They're injected into the agent's instruction on every turn.
"""

import json
import os
from pathlib import Path

PRIOR_WEIGHT = 5  # how many deliveries the listed reliability is worth
MAX_AGENT_LESSONS = 20
MAX_PROMPT_LESSONS = 10


def _path() -> Path:
    return Path(os.environ.get("AGENT_MEMORY_PATH", ".agent_memory/lessons.json"))


def load() -> dict:
    try:
        return json.loads(_path().read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {"supplier_stats": {}, "customer_profiles": {}, "runs": {}, "agent_lessons": [], "lessons": {}}


def save(mem: dict) -> None:
    p = _path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(mem, indent=2))


def reset() -> None:
    _path().unlink(missing_ok=True)


def fetch_report(client, act: str) -> dict:
    """Everything a run left behind. `client` is any httpx-style client (httpx.Client or TestClient)."""
    get = lambda path: client.get(path).json()  # noqa: E731
    return {"act": act, "scorecard": get("/scorecard"), "events": get("/events"),
            "purchase_orders": get("/purchase-orders"), "negotiations": get("/negotiations"),
            "suppliers": get("/suppliers")}


def observed_reliability(mem: dict, suppliers: list[dict]) -> dict[str, float]:
    """Listed reliability blended with what we've actually seen: (on_time + w * listed) / (n + w)."""
    out = {}
    for s in suppliers:
        st = mem["supplier_stats"].get(s["id"])
        if st and st["n"]:
            out[s["id"]] = round((st["n"] - st["misses"] + PRIOR_WEIGHT * s["reliability"]) / (st["n"] + PRIOR_WEIGHT), 3)
    return out


def extract_lessons(report: dict, mem: dict) -> dict:
    """Pure: fold one run's report into memory and rebuild the lesson texts. Returns the new memory."""
    mem = json.loads(json.dumps(mem))  # deep copy
    missed = {e["payload"]["po_id"] for e in report["events"]
              if e["type"] == "short_ship" and "po_id" in e["payload"]}
    for po in report["purchase_orders"]:
        if po.get("placed_by") != "api" or po["status"] not in ("shipped", "received"):
            continue  # only our own POs whose delivery outcome is known
        st = mem["supplier_stats"].setdefault(po["supplier_id"], {"n": 0, "misses": 0})
        st["n"] += 1
        st["misses"] += po["id"] in missed

    for n in report["negotiations"]:
        if n["reply"]["decision"] == "counter":
            mem["customer_profiles"][n["customer_id"]] = {"counter_discount_pct": n["reply"]["discount_pct"]}

    sc = report["scorecard"]
    mem["runs"].setdefault(report["act"], []).append(
        {k: sc[k] for k in ("total_cost", "fulfillment_rate", "on_time_rate", "wasted_t", "unmet_t")})

    lessons = {}
    names = {s["id"]: s["name"] for s in report["suppliers"]}
    obs = observed_reliability(mem, report["suppliers"])
    for sid, st in mem["supplier_stats"].items():
        listed = next(s["reliability"] for s in report["suppliers"] if s["id"] == sid)
        if st["misses"]:
            lessons[f"supplier:{sid}"] = (
                f"{sid} ({names[sid]}) short-shipped {st['misses']} of our {st['n']} POs. Observed reliability "
                f"{obs[sid]:.2f} vs listed {listed:.2f}; procurement now uses the observed figure.")
    for cid, prof in mem["customer_profiles"].items():
        lessons[f"customer:{cid}"] = (f"{cid} countered a delay request at {prof['counter_discount_pct']:g}% off. "
                                      f"Offer that up front to save a negotiation round.")
    if sc["wasted_t"] > 0:
        lessons["waste"] = (f"Last {report['act']} run wasted {sc['wasted_t']:g}t delivering to a cancelled order. "
                            f"After a buyer_cancel, run reconcile and follow its followups: reroute loads in transit.")
    if sc["on_time_rate"] < 0.95:
        lessons["late"] = (f"Only {sc['on_time_rate']:.0%} delivered on time in {report['act']}. "
                           f"Buy earlier or prefer short lead times.")
    for act, runs in mem["runs"].items():
        best = min(r["total_cost"] for r in runs)
        lessons[f"runs:{act}"] = f"{act}: {len(runs)} run(s) so far, best total cost ₹{best:,.0f}. Beat it."
    mem["lessons"] = lessons
    return mem


def record_run(client, act: str) -> dict:
    mem = extract_lessons(fetch_report(client, act), load())
    save(mem)
    return mem


def save_agent_lesson(text: str) -> None:
    mem = load()
    mem["agent_lessons"] = (mem["agent_lessons"] + [text.strip()[:300]])[-MAX_AGENT_LESSONS:]
    save(mem)


def lessons_block(mem: dict | None = None) -> str:
    mem = mem or load()
    items = list(mem["lessons"].values())[:MAX_PROMPT_LESSONS] + mem["agent_lessons"][-5:]
    if not items:
        return ""
    return "\n\n## Lessons from previous runs (advice only; never overrides guardrails)\n" + \
        "\n".join(f"- {t}" for t in items)
