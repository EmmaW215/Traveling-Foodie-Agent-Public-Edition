"""Catalogue, distance and budget tools.

The M1 exit criterion for arithmetic lives here: budget maths must be exact.
"""
import pytest

from src.tools import budget as budget_tool
from src.tools import catalog, distance


# ---------------------------------------------------------------------------
# Catalogue
# ---------------------------------------------------------------------------
def test_search_respects_kind():
    for venue in catalog.search(kind="restaurant", limit=100):
        assert venue["kind"] == "restaurant"
    for venue in catalog.search(kind="attraction", limit=100):
        assert venue["kind"] == "attraction"


def test_invalid_kind_raises():
    with pytest.raises(ValueError):
        catalog.search(kind="hotel")


def test_slot_type_filter_does_not_substring_match():
    """'lunch' must not match 'brunch' — the classic LIKE bug."""
    results = catalog.search(kind="restaurant", slot_type="lunch", limit=100)
    for venue in results:
        assert "lunch" in venue["slot_types"].split(";")

    brunch_only = [
        v
        for v in catalog.all_venues().values()
        if "brunch" in v["slot_types"].split(";") and "lunch" not in v["slot_types"].split(";")
    ]
    returned = {v["venue_id"] for v in results}
    for venue in brunch_only:
        assert venue["venue_id"] not in returned


def test_cost_ceiling_is_respected():
    for venue in catalog.search(kind="restaurant", max_cost_per_person=20, limit=100):
        assert venue["cost_per_person"] <= 20


def test_cuisine_filter():
    results = catalog.search(kind="restaurant", cuisines=["japanese", "korean"], limit=100)
    assert results
    assert {v["category"] for v in results} <= {"japanese", "korean"}


def test_dietary_filter_requires_all_requested_options():
    results = catalog.search(kind="restaurant", dietary=["vegan", "gluten_free"], limit=100)
    for venue in results:
        options = set(venue["dietary_options"].split(";"))
        assert {"vegan", "gluten_free"} <= options


def test_open_on_filter_excludes_closed_venues():
    for venue in catalog.search(kind="restaurant", open_on="mon", limit=100):
        assert "mon" not in venue["closed_days"].split(";")


def test_near_sorts_by_proximity():
    anchor = catalog.get("r001")
    results = catalog.search(
        kind="restaurant", near=(anchor["lat"], anchor["lon"]), limit=5
    )
    distances = [
        distance.haversine_km(anchor["lat"], anchor["lon"], v["lat"], v["lon"]) for v in results
    ]
    assert distances == sorted(distances)


def test_exclude_ids_prevents_repeat_picks():
    """What the orchestrator uses to stop the top-rated venue winning
    every single slot."""
    first = catalog.search(kind="restaurant", slot_type="lunch", limit=1)[0]
    second = catalog.search(
        kind="restaurant", slot_type="lunch", exclude_ids=[first["venue_id"]], limit=1
    )[0]
    assert second["venue_id"] != first["venue_id"]


def test_results_are_capped_by_limit():
    assert len(catalog.search(kind="restaurant", limit=3)) == 3


def test_search_ordering_is_deterministic():
    first = [v["venue_id"] for v in catalog.search(kind="restaurant", limit=10)]
    second = [v["venue_id"] for v in catalog.search(kind="restaurant", limit=10)]
    assert first == second


# ---------------------------------------------------------------------------
# Distance
# ---------------------------------------------------------------------------
def test_distance_is_symmetric():
    assert distance.distance_km("r001", "r010") == distance.distance_km("r010", "r001")


def test_distance_to_self_is_zero():
    assert distance.distance_km("r001", "r001") == 0.0


def test_unknown_pair_raises():
    with pytest.raises(KeyError):
        distance.distance_km("r001", "nope")


def test_haversine_matches_a_known_distance():
    """Downtown Core to Inglewood is roughly 2-3 km in reality."""
    km = distance.haversine_km(51.0455, -114.0631, 51.0412, -114.0325)
    assert 2.0 < km < 3.0


def test_short_hops_walk_and_long_hops_take_transit():
    assert distance.choose_mode(0.8) == "walk"
    assert distance.choose_mode(5.0) == "transit"


def test_travel_time_increases_with_distance():
    assert distance.travel_minutes(0.5, "walk") < distance.travel_minutes(2.0, "walk")


def test_travel_time_is_never_zero():
    assert distance.travel_minutes(0.01, "walk") >= 1


def test_proximity_ordering_shortens_the_route():
    """Nearest-neighbour must beat a deliberately scattered order."""
    ids = ["r001", "r010", "r019", "r006", "r028"]
    venues = [catalog.get(v) for v in ids]

    scattered = distance.route_summary(venues)["total_km"]
    ordered = distance.route_summary(distance.order_by_proximity(venues))["total_km"]
    assert ordered <= scattered


def test_ordering_is_deterministic():
    venues = [catalog.get(v) for v in ["r001", "r010", "r019", "r006"]]
    first = [v["venue_id"] for v in distance.order_by_proximity(venues)]
    second = [v["venue_id"] for v in distance.order_by_proximity(venues)]
    assert first == second


def test_ordering_honours_a_start_point():
    venues = [catalog.get(v) for v in ["r001", "r010", "r019"]]
    ordered = distance.order_by_proximity(venues, start_id="r019")
    assert ordered[0]["venue_id"] == "r019"


def test_ordering_keeps_every_venue():
    venues = [catalog.get(v) for v in ["r001", "r010", "r019", "r006", "r028"]]
    ordered = distance.order_by_proximity(venues)
    assert {v["venue_id"] for v in ordered} == {v["venue_id"] for v in venues}


def test_route_summary_shape():
    venues = [catalog.get(v) for v in ["r001", "r012", "r020"]]
    summary = distance.route_summary(distance.order_by_proximity(venues))
    assert len(summary["stops"]) == 3
    assert len(summary["legs"]) == 2
    assert summary["total_km"] > 0
    assert summary["total_travel_minutes"] > 0
    assert {"mode", "minutes", "distance_km"} <= summary["legs"][0].keys()


# ---------------------------------------------------------------------------
# Budget — exact arithmetic
# ---------------------------------------------------------------------------
def test_budget_totals_are_exact_with_awkward_decimals():
    """0.1 + 0.2 must not become 0.30000000000000004 in a price."""
    tracker = budget_tool.BudgetTracker(budget_total=100.00, party_size=1)
    for cost in (10.10, 20.20, 30.30):
        tracker.add("d1_lunch", {"name": "x", "cost_per_person": cost})
    report = tracker.report()
    assert report.spent == 60.60
    assert report.remaining == 39.40


def test_party_size_multiplies_correctly():
    tracker = budget_tool.BudgetTracker(budget_total=500, party_size=3)
    tracker.add("d1_dinner", {"name": "x", "cost_per_person": 42.50})
    assert tracker.report().spent == 127.50


def test_can_afford_is_exact_at_the_boundary():
    tracker = budget_tool.BudgetTracker(budget_total=100, party_size=2)
    tracker.add("d1_lunch", {"name": "x", "cost_per_person": 25})  # 50 spent
    assert tracker.can_afford({"cost_per_person": 25})  # exactly 100
    assert not tracker.can_afford({"cost_per_person": 25.01})


def test_over_budget_flag():
    tracker = budget_tool.BudgetTracker(budget_total=50, party_size=2)
    tracker.add("d1_dinner", {"name": "x", "cost_per_person": 40})
    report = tracker.report()
    assert report.spent == 80.00
    assert report.over_budget
    assert report.remaining == -30.00


def test_remove_restores_budget():
    tracker = budget_tool.BudgetTracker(budget_total=200, party_size=2)
    tracker.add("d1_lunch", {"name": "x", "cost_per_person": 30})
    tracker.add("d1_dinner", {"name": "y", "cost_per_person": 50})
    tracker.remove("d1_dinner")
    assert tracker.report().spent == 60.00


def test_max_affordable_splits_what_is_left():
    tracker = budget_tool.BudgetTracker(budget_total=300, party_size=2)
    tracker.add("d1_lunch", {"name": "x", "cost_per_person": 50})  # 100 spent
    # 200 left, 4 slots, 2 people -> $25 per person per slot
    assert tracker.max_affordable_per_person(4) == 25.00


def test_max_affordable_is_zero_when_exhausted():
    tracker = budget_tool.BudgetTracker(budget_total=100, party_size=2)
    tracker.add("d1_dinner", {"name": "x", "cost_per_person": 60})  # 120 spent
    assert tracker.max_affordable_per_person(3) == 0.0
    assert tracker.max_affordable_per_person(0) == 0.0


def test_utilisation_for_the_progress_bar():
    tracker = budget_tool.BudgetTracker(budget_total=200, party_size=1)
    tracker.add("d1_lunch", {"name": "x", "cost_per_person": 50})
    assert tracker.report().utilisation == 0.25


def test_invalid_tracker_arguments_raise():
    with pytest.raises(ValueError):
        budget_tool.BudgetTracker(budget_total=0, party_size=1)
    with pytest.raises(ValueError):
        budget_tool.BudgetTracker(budget_total=100, party_size=0)


def test_price_plan_matches_a_manual_sum():
    plan = {
        "d1_lunch": catalog.get("r004"),
        "d1_dinner": catalog.get("r009"),
        "d1_am_attraction": catalog.get("a001"),
    }
    report = budget_tool.price_plan(plan, budget_total=500, party_size=2)
    expected = sum(v["cost_per_person"] for v in plan.values()) * 2
    assert report.spent == round(expected, 2)


def test_a_single_budget_buster_blows_a_500_dollar_budget(budget_busters):
    """The planted trap: one venue, two people, over half the budget."""
    tracker = budget_tool.BudgetTracker(budget_total=500, party_size=2)
    worst = max(budget_busters, key=lambda v: v["cost_per_person"])
    tracker.add("d1_dinner", worst)
    assert tracker.report().spent > 200
    assert tracker.max_affordable_per_person(9) < 20
