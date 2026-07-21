"""Distance and route ordering — this is what replaces a paid routing API.

Every pairwise distance is precomputed at build time (see scripts/seed.py), so
route planning at runtime is dictionary lookups and a nearest-neighbour walk.
No Google Routes, no Mapbox, no API key, no per-request cost.

Travel times use fixed speeds rather than live traffic. For a walkable
downtown itinerary that is accurate enough to order stops sensibly, and it is
honest about what it is: `mode` and `speed_kmh` are returned with every leg.
"""
from __future__ import annotations

import json
import math
from functools import lru_cache

from ..paths import DISTANCE_MATRIX_PATH

EARTH_RADIUS_KM = 6371.0088

# Straight-line distance underestimates real walking distance on a street grid.
# 1.35 is the usual circuity factor for a gridded downtown.
GRID_CIRCUITY = 1.35

SPEEDS_KMH = {
    "walk": 4.8,
    "transit": 18.0,
    "drive": 30.0,
}

# Below this, walking beats waiting for anything.
WALKABLE_KM = 1.6


class DistanceMatrixUnavailableError(RuntimeError):
    """The precomputed matrix is missing — the seed step did not run."""


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km. Mirrors the seed script exactly."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


@lru_cache(maxsize=1)
def _matrix() -> dict[str, dict[str, float]]:
    if not DISTANCE_MATRIX_PATH.exists():
        raise DistanceMatrixUnavailableError(
            f"{DISTANCE_MATRIX_PATH} not found. Run `python -m scripts.seed`."
        )
    return json.loads(DISTANCE_MATRIX_PATH.read_text(encoding="utf-8"))


def distance_km(venue_id_a: str, venue_id_b: str) -> float:
    """Precomputed distance between two venues.

    The matrix is stored triangular (a<b only), so try both orderings.
    """
    if venue_id_a == venue_id_b:
        return 0.0
    matrix = _matrix()
    forward = matrix.get(venue_id_a, {}).get(venue_id_b)
    if forward is not None:
        return forward
    backward = matrix.get(venue_id_b, {}).get(venue_id_a)
    if backward is not None:
        return backward
    raise KeyError(f"No precomputed distance for {venue_id_a} <-> {venue_id_b}")


def choose_mode(km: float) -> str:
    return "walk" if km <= WALKABLE_KM else "transit"


def travel_minutes(km: float, mode: str | None = None) -> int:
    """Door-to-door minutes, rounded up, including a fixed transit wait."""
    mode = mode or choose_mode(km)
    if mode not in SPEEDS_KMH:
        raise ValueError(f"mode must be one of {sorted(SPEEDS_KMH)}")
    road_km = km * GRID_CIRCUITY
    minutes = (road_km / SPEEDS_KMH[mode]) * 60
    if mode == "transit":
        minutes += 6  # average wait; without it transit beats walking absurdly
    return max(1, math.ceil(minutes))


def leg(venue_a: dict, venue_b: dict) -> dict:
    """A single hop between two venues, ready for the UI and the Critic."""
    km = distance_km(venue_a["venue_id"], venue_b["venue_id"])
    mode = choose_mode(km)
    return {
        "from_id": venue_a["venue_id"],
        "from_name": venue_a["name"],
        "to_id": venue_b["venue_id"],
        "to_name": venue_b["name"],
        "distance_km": round(km, 2),
        "mode": mode,
        "speed_kmh": SPEEDS_KMH[mode],
        "minutes": travel_minutes(km, mode),
    }


def order_by_proximity(venues: list[dict], start_id: str | None = None) -> list[dict]:
    """Nearest-neighbour ordering — the Route agent's core move.

    Not the optimal tour, and deliberately so: for 5 stops the difference from
    optimal is negligible, while the result is deterministic, instant, and easy
    to explain to a user ("we kept each hop short"). Ties break on venue_id so
    the same input always produces the same itinerary.
    """
    if len(venues) <= 2:
        return list(venues)

    remaining = {v["venue_id"]: v for v in venues}
    if start_id and start_id in remaining:
        current = remaining.pop(start_id)
    else:
        first_id = sorted(remaining)[0]
        current = remaining.pop(first_id)

    ordered = [current]
    while remaining:
        nearest_id = min(
            remaining,
            key=lambda vid: (distance_km(current["venue_id"], vid), vid),
        )
        current = remaining.pop(nearest_id)
        ordered.append(current)
    return ordered


def route_summary(ordered_venues: list[dict]) -> dict:
    """Legs plus totals for an ordered day, for the Formatter and the map."""
    legs = [
        leg(ordered_venues[i], ordered_venues[i + 1]) for i in range(len(ordered_venues) - 1)
    ]
    return {
        "stops": [
            {
                "venue_id": v["venue_id"],
                "name": v["name"],
                "lat": v["lat"],
                "lon": v["lon"],
                "kind": v["kind"],
            }
            for v in ordered_venues
        ],
        "legs": legs,
        "total_km": round(sum(leg_["distance_km"] for leg_ in legs), 2),
        "total_travel_minutes": sum(leg_["minutes"] for leg_ in legs),
    }
