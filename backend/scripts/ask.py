#!/usr/bin/env python3
"""Ask the Tier 0 copilot a question from the command line.

    python -m scripts.ask "where can I get ramen under $30?"
    python -m scripts.ask --live "vegan options in Kensington?"

Mock model + local retriever by default, so it runs with no keys. --live uses
the real free LLM chain (and Upstash, if configured). Prints the grounded
answer, its citations, and which retriever served it.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agents.base import default_model  # noqa: E402
from src.agents.mock import MockLLM  # noqa: E402
from src.rag import copilot  # noqa: E402
from src.rag.retriever import LocalRetriever, build_retriever  # noqa: E402

# The three standard copilot scenarios, carried from the hackathon.
STANDARD = [
    "Where can I get good ramen for lunch, and roughly what does it cost?",
    "I have a peanut allergy — which restaurants are safe for dinner?",
    "What free attractions are there near downtown?",
]


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="*", help="Your question (omit to run the 3 standards).")
    parser.add_argument("--live", action="store_true", help="Use the real free LLM chain.")
    args = parser.parse_args()

    model = default_model() if args.live else MockLLM()
    retriever = build_retriever() if args.live else LocalRetriever()
    questions = [" ".join(args.question)] if args.question else STANDARD

    for question in questions:
        result = await copilot.answer_question(question, model=model, retriever=retriever)
        print(f"\nQ: {question}")
        print(f"A: {result['answer']}")
        if result["citations"]:
            cites = ", ".join(f"{c['name']} ({c['venue_id']})" for c in result["citations"])
            print(f"   cited: {cites}")
        flag = "REFUSED" if result["refused"] else ("grounded" if result["grounded"] else "ungrounded")
        print(f"   [{flag}] via {result['retriever']} retriever")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
