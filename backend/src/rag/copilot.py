"""Tier 0 copilot — retrieve, answer grounded, guard against invented venues.

The flow:
  1. Retrieve the top-k venue chunks for the question.
  2. If nothing relevant comes back, refuse — do not answer from a weak match.
  3. Ask the model for a grounded answer that cites venue ids from the context.
  4. Guard, in code: every cited id must (a) exist in the catalogue and (b) have
     been in the retrieved context. A cited id that fails either is a
     hallucination. On failure we re-ask once with the valid ids spelled out;
     if it still cites something unverifiable, we refuse rather than pass an
     invented venue to the user.

This is the code-level half of "answers only from its knowledge base". The
prompt asks the model to behave; this module makes it impossible not to.
"""
from __future__ import annotations

import logging

from ..agents.base import AgentError, ChatModel, run_structured
from ..guards import venue_has_allergen
from ..tools import catalog
from .prompts import COPILOT_SYSTEM
from .retriever import Hit, Retriever
from .schemas import CopilotAnswer

log = logging.getLogger(__name__)

# The allergens the catalogue tracks. If the question mentions one, we must not
# surface a venue that uses it — a peanut-allergy question retrieving the peanut
# restaurant (which lexical search will happily do) is the one answer that is
# genuinely unsafe, not just unhelpful.
_KNOWN_ALLERGENS = (
    "peanut", "tree_nut", "tree nut", "dairy", "gluten", "egg",
    "soy", "shellfish", "fish", "sesame",
)


def _allergens_in_question(question: str) -> list[str]:
    q = question.lower()
    found = set()
    for term in _KNOWN_ALLERGENS:
        if term in q:
            found.add(term.replace(" ", "_"))
    return sorted(found)


def _drop_unsafe(hits: list[Hit], allergies: list[str], known: dict) -> list[Hit]:
    """Remove retrieved venues that use an allergen named in the question."""
    if not allergies:
        return hits
    safe = []
    for h in hits:
        venue = known.get(h.id)
        if venue and venue_has_allergen(venue, allergies):
            continue
        safe.append(h)
    return safe


_REFUSAL = (
    "I can only answer from this Calgary food guide, and I couldn't find "
    "anything in it that covers that. Try asking about restaurants, cuisines, "
    "neighbourhoods, budgets, dietary needs, or attractions."
)


def _context_block(hits: list[Hit]) -> str:
    return "\n".join(f"id={h.id} | {h.text}" for h in hits)


def _user_prompt(question: str, hits: list[Hit], correction: list[str] | None = None) -> str:
    parts = [
        "CONTEXT (the only venues you may use):",
        _context_block(hits),
        "",
        f"QUESTION: {question}",
    ]
    if correction:
        parts.append(
            "\nYour previous answer cited ids that are not in the CONTEXT: "
            f"{', '.join(correction)}. Only cite ids shown above, or set "
            "answerable to false."
        )
    return "\n".join(parts)


def _refusal(message: str, hits: list[Hit], retriever_mode: str) -> dict:
    return {
        "answer": message,
        "grounded": False,
        "refused": True,
        "citations": [],
        "sources": [h.id for h in hits],
        "retriever": retriever_mode,
    }


async def answer_question(
    question: str, *, model: ChatModel, retriever: Retriever, k: int = 8
) -> dict:
    known = catalog.all_venues()
    hits = await retriever.retrieve(question, k=k)

    # Safety filter BEFORE the model sees the context: if the question states an
    # allergy, no venue using that allergen may be offered as a candidate. This
    # is the allergen guard from Tier 1, applied to retrieval.
    allergies = _allergens_in_question(question)
    hits = _drop_unsafe(hits, allergies, known)
    retrieved_ids = {h.id for h in hits}

    if not hits:
        message = (
            "I couldn't find a place in the guide that avoids that allergen for "
            "what you asked. I'd rather say so than risk suggesting somewhere unsafe."
            if allergies
            else _REFUSAL
        )
        return _refusal(message, hits, retriever.mode)

    async def _ask(correction: list[str] | None = None) -> CopilotAnswer:
        result = await run_structured(
            model,
            system=COPILOT_SYSTEM,
            user=_user_prompt(question, hits, correction),
            schema=CopilotAnswer,
            temperature=0.3,
            max_tokens=500,
        )
        assert isinstance(result, CopilotAnswer)
        return result

    def _invalid(ids: list[str]) -> list[str]:
        # A cited id is invalid if it isn't a real venue OR wasn't retrieved.
        return [vid for vid in ids if vid not in known or vid not in retrieved_ids]

    try:
        result = await _ask()
        invalid = _invalid(result.cited_venue_ids)
        if invalid:
            # One bounded correction pass — same discipline as the agent repair.
            result = await _ask(correction=invalid)
            invalid = _invalid(result.cited_venue_ids)
    except AgentError:
        return _refusal(_REFUSAL, hits, retriever.mode)

    if not result.answerable:
        message = result.answer.strip() or _REFUSAL
        return _refusal(message, hits, retriever.mode)

    if invalid:
        # The guard fired and couldn't be repaired: the model insisted on a
        # venue we cannot verify. Refuse rather than surface an invented place.
        log.warning("copilot refused: unverifiable citations %s", invalid)
        return _refusal(
            "I started to mention a place I can't verify in this guide, so I'm "
            "holding back rather than risk inventing somewhere. Could you rephrase?",
            hits,
            retriever.mode,
        )

    citations = [
        {
            "venue_id": vid,
            "name": known[vid]["name"],
            "category": known[vid]["category"],
            "neighbourhood": known[vid]["neighbourhood"],
            "kind": known[vid]["kind"],
        }
        for vid in result.cited_venue_ids
    ]
    return {
        "answer": result.answer.strip(),
        "grounded": True,
        "refused": False,
        "citations": citations,
        "sources": [h.id for h in hits],
        "retriever": retriever.mode,
    }
