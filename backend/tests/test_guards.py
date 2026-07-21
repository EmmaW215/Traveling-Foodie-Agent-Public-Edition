"""Guards — the constraints that must hold regardless of what an LLM says.

The M1 exit criterion lives here: the planted allergen trap must be filtered.
"""
import pytest

from src.guards import (
    SLOT_IDS,
    Issue,
    SlotValidationError,
    filter_allergens,
    filter_open_on,
    is_open_on,
    is_valid_slot,
    resolve_venue,
    slots_for_days,
    validate_plan,
    validate_slot,
    venue_exists,
    venue_has_allergen,
)
from src.tools import catalog


# ---------------------------------------------------------------------------
# Slot vocabulary — the Critic-loop failure mode
# ---------------------------------------------------------------------------
def test_slot_vocabulary_is_exactly_ten_slots():
    assert len(SLOT_IDS) == 10


@pytest.mark.parametrize("slot", ["d1_breakfast", "d2_dinner", "d1_am_attraction"])
def test_valid_slots_accepted(slot):
    assert validate_slot(slot) == slot


@pytest.mark.parametrize(
    "slot",
    ["day1_lunch", "lunch_day_1", "D1_LUNCH", "d3_lunch", "lunch", "", "d1-lunch"],
)
def test_off_vocabulary_slots_rejected(slot):
    """These are the exact shapes an LLM drifts into. Every one must raise."""
    assert not is_valid_slot(slot)
    with pytest.raises(SlotValidationError):
        validate_slot(slot)


def test_one_day_itinerary_uses_first_five_slots():
    assert slots_for_days(1) == [
        "d1_breakfast",
        "d1_am_attraction",
        "d1_lunch",
        "d1_pm_attraction",
        "d1_dinner",
    ]
    assert len(slots_for_days(2)) == 10


# ---------------------------------------------------------------------------
# Allergen exclusion — the M1 exit criterion
# ---------------------------------------------------------------------------
def test_peanut_trap_is_excluded_for_peanut_allergy(peanut_trap):
    assert venue_has_allergen(peanut_trap, ["peanut"])
    assert filter_allergens([peanut_trap], ["peanut"]) == []


def test_peanut_trap_survives_when_not_allergic(peanut_trap):
    """Exclusion must be precise, not blanket: an unrelated allergy keeps it.

    'dairy' is deliberately chosen — the trap venue also lists shellfish and
    tree_nut, so testing with either of those would pass for the wrong reason.
    """
    assert "dairy" not in peanut_trap["allergens_present"]
    assert filter_allergens([peanut_trap], ["dairy"]) == [peanut_trap]
    assert filter_allergens([peanut_trap], []) == [peanut_trap]


def test_peanut_trap_also_excluded_by_its_other_allergens(peanut_trap):
    """It lists several allergens; any one of them must exclude it."""
    for allergen in ("peanut", "shellfish", "tree_nut", "soy", "fish"):
        assert filter_allergens([peanut_trap], [allergen]) == [], allergen


def test_allergen_matching_is_case_insensitive(peanut_trap):
    assert venue_has_allergen(peanut_trap, ["PEANUT"])
    assert venue_has_allergen(peanut_trap, ["  Peanut  "])


def test_catalog_search_never_returns_the_peanut_trap(peanut_trap):
    """The end-to-end guarantee: an allergic traveller cannot even see it."""
    results = catalog.search(kind="restaurant", allergies=["peanut"], limit=100)
    assert peanut_trap["venue_id"] not in {r["venue_id"] for r in results}


def test_catalog_search_can_return_it_without_the_allergy(peanut_trap):
    results = catalog.search(kind="restaurant", cuisines=["thai"], limit=100)
    assert peanut_trap["venue_id"] in {r["venue_id"] for r in results}


def test_multiple_allergies_are_all_applied():
    results = catalog.search(
        kind="restaurant", allergies=["peanut", "shellfish", "sesame"], limit=100
    )
    for venue in results:
        present = set(venue["allergens_present"].split(";"))
        assert not present & {"peanut", "shellfish", "sesame"}


# ---------------------------------------------------------------------------
# Opening hours
# ---------------------------------------------------------------------------
def test_monday_closed_venues_are_filtered_on_monday(monday_closed_venues):
    assert monday_closed_venues, "dataset must contain a closed_monday trap"
    for venue in monday_closed_venues:
        assert not is_open_on(venue, "mon")
        assert is_open_on(venue, "tue")
    assert filter_open_on(monday_closed_venues, "mon") == []


def test_venue_with_no_closed_days_is_always_open(venues):
    always_open = next(v for v in venues.values() if v["closed_days"] == "")
    assert all(is_open_on(always_open, day) for day in ("mon", "sat", "sun"))


def test_invalid_weekday_raises(venues):
    venue = next(iter(venues.values()))
    with pytest.raises(ValueError):
        is_open_on(venue, "someday")


# ---------------------------------------------------------------------------
# Anti-hallucination
# ---------------------------------------------------------------------------
def test_known_venue_resolves_by_id_and_by_name(venues):
    venue = venues["r001"]
    assert venue_exists("r001", venues)
    assert venue_exists(venue["name"], venues)
    assert venue_exists(venue["name"].upper(), venues)
    assert resolve_venue("r001", venues)["name"] == venue["name"]


def test_invented_venue_is_rejected(venues):
    """The failure this prevents: a plausible-sounding restaurant that
    does not exist, confidently recommended with an address."""
    assert not venue_exists("The Gilded Bison Chophouse", venues)
    assert not venue_exists("r999", venues)
    assert not venue_exists("", venues)
    assert resolve_venue("Totally Real Cafe", venues) is None


# ---------------------------------------------------------------------------
# Whole-plan validation
# ---------------------------------------------------------------------------
def _clean_plan(venues):
    return {
        "d1_breakfast": next(
            v for v in venues.values() if v["kind"] == "restaurant" and "breakfast" in v["slot_types"]
        ),
        "d1_lunch": next(
            v
            for v in venues.values()
            if v["kind"] == "restaurant" and "lunch" in v["slot_types"] and v["cost_per_person"] < 25
        ),
        "d1_am_attraction": next(
            v for v in venues.values() if v["kind"] == "attraction" and "am" in v["slot_types"]
        ),
    }


def test_clean_plan_passes(venues):
    report = validate_plan(
        _clean_plan(venues), allergies=[], budget_total=500, party_size=2
    )
    assert report.ok, report.as_dict()


def test_allergy_violation_is_reported(venues, peanut_trap):
    plan = {"d1_dinner": peanut_trap}
    report = validate_plan(plan, allergies=["peanut"], budget_total=500, party_size=2)
    assert not report.ok
    assert any(i.issue == "allergy_violation" for i in report.issues)


def test_budget_overrun_is_reported(venues, budget_busters):
    plan = {"d1_dinner": budget_busters[0], "d2_dinner": budget_busters[-1]}
    report = validate_plan(plan, allergies=[], budget_total=100, party_size=2)
    assert not report.ok
    assert any(i.issue == "budget_exceeded" for i in report.issues)


def test_closed_venue_is_reported_for_that_day(monday_closed_venues):
    restaurant = next(v for v in monday_closed_venues if v["kind"] == "restaurant")
    plan = {"d1_lunch": restaurant}
    report = validate_plan(
        plan, allergies=[], budget_total=500, party_size=2, day_names={1: "mon", 2: "tue"}
    )
    assert any(i.issue == "closed" for i in report.issues)

    ok_report = validate_plan(
        plan, allergies=[], budget_total=500, party_size=2, day_names={1: "wed", 2: "thu"}
    )
    assert not any(i.issue == "closed" for i in ok_report.issues)


def test_wrong_kind_in_slot_is_reported(venues):
    attraction = next(v for v in venues.values() if v["kind"] == "attraction")
    report = validate_plan(
        {"d1_lunch": attraction}, allergies=[], budget_total=500, party_size=2
    )
    assert any(i.issue == "wrong_kind" for i in report.issues)


def test_invalid_slot_in_plan_is_reported(venues):
    venue = next(iter(venues.values()))
    report = validate_plan({"lunch_day_1": venue}, allergies=[], budget_total=500, party_size=2)
    assert any(i.issue == "invalid_slot" for i in report.issues)


def test_repeated_venue_is_reported(venues):
    """Found by an M1 dry run: every hard constraint passed while the plan
    booked the same restaurant for lunch and dinner."""
    same = next(
        v
        for v in venues.values()
        if v["kind"] == "restaurant" and {"lunch", "dinner"} <= set(v["slot_types"].split(";"))
    )
    report = validate_plan(
        {"d1_lunch": same, "d1_dinner": same},
        allergies=[],
        budget_total=500,
        party_size=2,
    )
    assert not report.ok
    dupes = [i for i in report.issues if i.issue == "duplicate_venue"]
    assert len(dupes) == 1
    assert "d1_lunch" in dupes[0].detail


def test_distinct_venues_are_not_flagged_as_duplicates(venues):
    plan = _clean_plan(venues)
    report = validate_plan(plan, allergies=[], budget_total=500, party_size=2)
    assert not any(i.issue == "duplicate_venue" for i in report.issues)


def test_issue_serialises_for_the_trace_stream():
    assert Issue("d1_lunch", "closed", "shut on Monday").as_dict() == {
        "slot": "d1_lunch",
        "issue": "closed",
        "detail": "shut on Monday",
    }
