"""Budget arithmetic — pure Python, deliberately.

The single most transferable lesson from the hackathon design: never let an LLM
do the arithmetic. A model that adds $47 + $118 + $26 and gets $181 produces a
plan that looks right and is wrong, and it is the first thing a judge checks.

Money is handled in cents internally so repeated addition cannot drift.
"""
from __future__ import annotations

from dataclasses import dataclass, field


def _to_cents(amount: float) -> int:
    return int(round(amount * 100))


def _to_dollars(cents: int) -> float:
    return round(cents / 100, 2)


@dataclass
class LineItem:
    slot: str
    name: str
    cost_per_person: float
    party_size: int

    @property
    def total(self) -> float:
        return _to_dollars(_to_cents(self.cost_per_person) * self.party_size)


@dataclass
class BudgetReport:
    budget_total: float
    spent: float
    remaining: float
    party_size: int
    line_items: list[dict] = field(default_factory=list)

    @property
    def over_budget(self) -> bool:
        # Compare in cents; 0.1 + 0.2 style drift must not decide this.
        return _to_cents(self.spent) > _to_cents(self.budget_total)

    @property
    def utilisation(self) -> float:
        """Fraction of the budget used, for the UI progress bar."""
        if self.budget_total <= 0:
            return 0.0
        return round(self.spent / self.budget_total, 4)

    def as_dict(self) -> dict:
        return {
            "budget_total": self.budget_total,
            "spent": self.spent,
            "remaining": self.remaining,
            "party_size": self.party_size,
            "over_budget": self.over_budget,
            "utilisation": self.utilisation,
            "line_items": self.line_items,
        }


class BudgetTracker:
    """Running total across an itinerary.

    Usage mirrors how the Tier 1/2 orchestrator builds a plan slot by slot:

        tracker = BudgetTracker(budget_total=500, party_size=2)
        tracker.add("d1_lunch", venue)
        tracker.can_afford(other_venue)   # before committing
    """

    def __init__(self, budget_total: float, party_size: int = 1) -> None:
        if budget_total <= 0:
            raise ValueError("budget_total must be positive")
        if party_size < 1:
            raise ValueError("party_size must be at least 1")
        self.budget_cents = _to_cents(budget_total)
        self.party_size = party_size
        self._items: list[LineItem] = []

    @property
    def spent_cents(self) -> int:
        return sum(_to_cents(i.cost_per_person) * i.party_size for i in self._items)

    def add(self, slot: str, venue: dict) -> LineItem:
        item = LineItem(
            slot=slot,
            name=venue["name"],
            cost_per_person=float(venue["cost_per_person"]),
            party_size=self.party_size,
        )
        self._items.append(item)
        return item

    def remove(self, slot: str) -> None:
        self._items = [i for i in self._items if i.slot != slot]

    def can_afford(self, venue: dict) -> bool:
        """Would adding this venue stay within budget?"""
        cost = _to_cents(float(venue["cost_per_person"])) * self.party_size
        return self.spent_cents + cost <= self.budget_cents

    def remaining_cents(self) -> int:
        return self.budget_cents - self.spent_cents

    def max_affordable_per_person(self, slots_left: int) -> float:
        """Even split of what is left — the executor's price ceiling hint.

        Returns 0 when nothing is left, so a caller can short-circuit instead
        of searching for a venue that cannot exist.
        """
        if slots_left <= 0:
            return 0.0
        remaining = max(self.remaining_cents(), 0)
        return _to_dollars(remaining // (slots_left * self.party_size))

    def report(self) -> BudgetReport:
        spent = _to_dollars(self.spent_cents)
        return BudgetReport(
            budget_total=_to_dollars(self.budget_cents),
            spent=spent,
            remaining=_to_dollars(self.remaining_cents()),
            party_size=self.party_size,
            line_items=[
                {
                    "slot": i.slot,
                    "name": i.name,
                    "cost_per_person": i.cost_per_person,
                    "total": i.total,
                }
                for i in self._items
            ],
        )


def price_plan(plan: dict[str, dict], budget_total: float, party_size: int) -> BudgetReport:
    """Convenience: price a whole assembled plan in one call."""
    tracker = BudgetTracker(budget_total=budget_total, party_size=party_size)
    for slot, venue in plan.items():
        if venue and "cost_per_person" in venue:
            tracker.add(slot, venue)
    return tracker.report()
