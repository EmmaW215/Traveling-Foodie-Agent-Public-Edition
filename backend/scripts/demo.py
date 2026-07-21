#!/usr/bin/env python3
"""Run the Tier 1 pipeline from the command line.

    python -m scripts.demo                 # S1 scenario, mock mode (no keys)
    python -m scripts.demo --live          # use the real free LLM chain
    python -m scripts.demo --days 1 --budget 250 --cuisines thai,indian

The Tier-0-copilot equivalent of a working demo: it prints the streamed trace
exactly as the UI will render it, then the final itinerary. Mock mode is the
default so it runs anywhere, including CI, with zero setup.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models import Preferences  # noqa: E402
from src.orchestrator import run_tier1  # noqa: E402

# S1 — the standard scenario carried from the hackathon: 2 days, $500, peanut
# allergy, party of two, international food.
S1 = Preferences(
    city="Calgary",
    days=2,
    budget_total=500,
    party_size=2,
    cuisines=["japanese", "italian", "thai"],
    allergies=["peanut"],
    notes="We love international food and want a good mix.",
)


def _print_event(event: dict) -> None:
    kind = event["event"]
    if kind == "planner_done":
        print(f"\n  PLANNER  {event['summary']}")
        if event["cuisines_priority"]:
            print(f"           cuisines: {', '.join(event['cuisines_priority'])}")
    elif kind == "executor_result":
        tag = "AI" if event["picked_by"] == "model" else "fallback"
        print(
            f"  PICK     {event['slot']:<17} {event['name']:<32} "
            f"${event['spent']:>6.0f} spent  [{tag}]"
        )
    elif kind == "validation":
        status = "PASS" if event["ok"] else f"FAIL {event['issues']}"
        print(f"\n  VALIDATE {status}")
    elif kind == "route_ready":
        print(f"  ROUTE    total {event['total_km']} km")
    elif kind == "notice":
        print(f"  NOTE     {event['message']}")
    elif kind == "error":
        print(f"  ERROR    {event['detail']}")


def _print_final(event: dict) -> None:
    print("\n" + "=" * 68)
    print(f"  {event['title']}")
    print("=" * 68)
    print(f"  {event['intro']}\n")
    for day in event["days"]:
        print(f"  Day {day['day']}")
        print(f"    {day['summary']}")
        for stop in day["stops"]:
            cost = f"${stop['cost_per_person']:.0f}pp" if stop["cost_per_person"] else "free"
            print(f"      - {stop['name']} ({stop['category']}, {cost}) — {stop['neighbourhood']}")
        print(f"    route: {day['route']['total_km']} km, {day['route']['total_travel_minutes']} min travel\n")
    b = event["budget"]
    print(f"  Budget: ${b['spent']:.0f} of ${b['budget_total']:.0f} "
          f"({b['utilisation'] * 100:.0f}% used) for {b['party_size']} people")
    served = "mock (offline)" if event["mock"] else "live free LLMs"
    print(f"  Served by: {served}  ·  {event['elapsed_ms']} ms  ·  data {event['data_version']}")
    if not event["validation"]["ok"]:
        print(f"  WARNING: validation issues: {event['validation']['issues']}")


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="Use real free LLM providers.")
    parser.add_argument("--days", type=int, default=None)
    parser.add_argument("--budget", type=float, default=None)
    parser.add_argument("--party", type=int, default=None)
    parser.add_argument("--cuisines", type=str, default=None, help="comma-separated")
    parser.add_argument("--allergies", type=str, default=None, help="comma-separated")
    args = parser.parse_args()

    prefs = S1.model_copy()
    if args.days:
        prefs.days = args.days
    if args.budget:
        prefs.budget_total = args.budget
    if args.party:
        prefs.party_size = args.party
    if args.cuisines is not None:
        prefs.cuisines = [c.strip() for c in args.cuisines.split(",") if c.strip()]
    if args.allergies is not None:
        prefs.allergies = [a.strip() for a in args.allergies.split(",") if a.strip()]

    mode = "live free LLM chain" if args.live else "mock mode (no keys needed)"
    print(f"Traveling Foodie Agent — Tier 1 demo ({mode})")
    print(f"Request: {prefs.days} days, ${prefs.budget_total:.0f}, party of {prefs.party_size}, "
          f"cuisines={prefs.cuisines or 'any'}, allergies={prefs.allergies or 'none'}")

    final = None
    async for event in run_tier1(prefs, mock=not args.live):
        _print_event(event)
        if event["event"] == "final":
            final = event
        if event["event"] == "error":
            return 1

    if final:
        _print_final(final)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
