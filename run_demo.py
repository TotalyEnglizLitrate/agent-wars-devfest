"""Run the three demo acts with the agent, then compare against the FCFS baseline.

  uv run python run_demo.py --scripted               # no LLM needed, deterministic
  uv run python run_demo.py                          # Gemini via ADK (GOOGLE_API_KEY from .env)
  uv run python run_demo.py --acts act1 --runs 2 --reset-memory --scripted   # learning demo
  uv run python run_demo.py --url http://localhost:8000                      # against a live server
"""

import argparse
import asyncio
import os
import sys

from dotenv import load_dotenv

load_dotenv()  # GOOGLE_API_KEY etc. from .env (gitignored)

from agent_wars_devfest.agent import memory, tools  # noqa: E402
from agent_wars_devfest.agent.agent import LlmRunner, scripted_tick  # noqa: E402
from agent_wars_devfest.tools.baseline import END_DAY, run_fcfs  # noqa: E402


def _fmt(s: dict) -> str:
    return (f"fulfilled {s['fulfillment_rate']:.0%} | on time {s['on_time_rate']:.0%} | "
            f"wasted {s['wasted_t']:g}t | total ₹{s['total_cost']:,.0f}")


async def run_act(act: str, use_llm: bool, client) -> dict:
    loaded = tools.load_scenario(act)
    print(f"\n━━━ {act} (starts day {loaded['day']}) ━━━")
    runner = LlmRunner(act) if use_llm else None
    day = loaded["day"]
    while True:
        try:
            summary = await runner.tick(day) if runner else scripted_tick()
        except Exception as e:  # keep the demo alive if the LLM call fails
            print(f"  ! LLM turn failed ({type(e).__name__}: {e}); falling back to the scripted turn")
            summary = scripted_tick()
        if summary and "quiet day" not in summary:
            print(f"  day {day}: {summary}")
        if day >= END_DAY:
            break
        day = tools.advance_clock(1)["day"]
    score = tools.get_scorecard()
    mem = memory.record_run(client, act)
    return {"score": score, "lessons": mem["lessons"]}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scripted", action="store_true", help="deterministic tool sequence, no LLM")
    ap.add_argument("--acts", nargs="+", default=["act1", "act2", "act3"])
    ap.add_argument("--runs", type=int, default=1, help="repeat each act to show learning")
    ap.add_argument("--reset-memory", action="store_true")
    ap.add_argument("--url", help="World API URL; default runs the world in-process")
    ap.add_argument("--verbose", action="store_true", help="print every tool call")
    args = ap.parse_args()

    if args.verbose:
        os.environ["AGENT_VERBOSE"] = "1"
    use_llm = not args.scripted
    if use_llm and not (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_GENAI_USE_VERTEXAI")):
        print("No GOOGLE_API_KEY set: running the scripted agent instead (pass --scripted to silence this).")
        use_llm = False
    if args.reset_memory:
        memory.reset()

    if args.url:
        import httpx
        client = httpx.Client(base_url=args.url, timeout=30)
    else:
        from fastapi.testclient import TestClient

        from agent_wars_devfest.world.app import app
        client = TestClient(app)
    tools.use_client(client)

    results = {}
    for act in args.acts:
        for run in range(1, args.runs + 1):
            if args.runs > 1:
                print(f"\n(run {run} of {args.runs})")
            results[(act, run)] = await run_act(act, use_llm, client)

    print("\n━━━ Decision log (last act) ━━━")
    for d in tools.decision_log:
        if d["kind"] != "tool":
            print(f"  day {d['day']} {d['kind']}: {d['text']}")

    print("\n━━━ Agent vs FCFS baseline (same seed) ━━━")
    for act in args.acts:
        base = run_fcfs(act)  # reloads the in-process world, so it runs after the agent is done
        print(f"  {act} FCFS : {_fmt(base)}")
        for run in range(1, args.runs + 1):
            s = results[(act, run)]["score"]
            delta = base["total_cost"] - s["total_cost"]
            label = "agent" if args.runs == 1 else f"run {run}"
            print(f"  {act} {label}: {_fmt(s)}  ({'saves' if delta >= 0 else 'costs'} ₹{abs(delta):,.0f})")

    lessons = memory.lessons_block()
    if lessons:
        print(lessons.replace("\n## ", "\n━━━ ") + "\n")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
