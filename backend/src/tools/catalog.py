"""Venue catalogue — the only place the agents get facts about the world.

Read-only SQLite. The executors are allowed to *choose* from what this returns;
they are never allowed to invent something that isn't in here (see guards.py).
"""
from __future__ import annotations

import sqlite3
from functools import lru_cache

from ..paths import SQLITE_PATH

VALID_KINDS = {"restaurant", "attraction"}


class CatalogUnavailableError(RuntimeError):
    """The SQLite artefact is missing — the seed step did not run."""


def _connect() -> sqlite3.Connection:
    if not SQLITE_PATH.exists():
        raise CatalogUnavailableError(
            f"{SQLITE_PATH} not found. Run `python -m scripts.seed` "
            f"(it runs automatically in CI and on deploy)."
        )
    conn = sqlite3.connect(f"file:{SQLITE_PATH}?mode=ro", uri=True, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _rows_to_dicts(rows: list[sqlite3.Row]) -> list[dict]:
    return [dict(row) for row in rows]


@lru_cache(maxsize=1)
def all_venues() -> dict[str, dict]:
    """Every venue keyed by id. Cached — the catalogue is immutable at runtime.

    Small enough (~85 rows) to hold in memory on a 512 MB instance, and having
    it resident makes the anti-hallucination guard free.
    """
    with _connect() as conn:
        rows = conn.execute("SELECT * FROM venues ORDER BY venue_id").fetchall()
    return {row["venue_id"]: dict(row) for row in rows}


@lru_cache(maxsize=1)
def dataset_meta() -> dict:
    with _connect() as conn:
        rows = conn.execute("SELECT key, value FROM dataset_meta").fetchall()
    meta = {row["key"]: row["value"] for row in rows}
    for numeric in ("restaurants", "attractions"):
        if numeric in meta:
            meta[numeric] = int(meta[numeric])
    return meta


def get(venue_id: str) -> dict | None:
    return all_venues().get(venue_id)


def search(
    *,
    kind: str,
    slot_type: str | None = None,
    cuisines: list[str] | None = None,
    max_cost_per_person: float | None = None,
    allergies: list[str] | None = None,
    dietary: list[str] | None = None,
    open_on: str | None = None,
    neighbourhood: str | None = None,
    near: tuple[float, float] | None = None,
    exclude_ids: list[str] | None = None,
    limit: int = 10,
) -> list[dict]:
    """Filtered venue search — the tool the Restaurant/Attraction agents call.

    Filtering happens in SQL and in code rather than in the prompt, so the
    executor LLM only ever sees candidates that already satisfy every hard
    constraint. It picks on taste; it cannot pick something unsafe.

    `near` sorts by distance from a (lat, lon) anchor — used to keep a day's
    itinerary geographically tight.

    `exclude_ids` drops venues already used elsewhere in the plan. Without it
    the highest-rated venue wins every slot and the traveller eats at the same
    place three times — which every hard constraint happily permits.
    """
    if kind not in VALID_KINDS:
        raise ValueError(f"kind must be one of {sorted(VALID_KINDS)}")

    sql = ["SELECT * FROM venues WHERE kind = ?"]
    params: list = [kind]

    if slot_type:
        # slot_types is a ';' list; match on a delimited substring so that
        # 'lunch' cannot accidentally match 'brunch'.
        sql.append("AND (';' || slot_types || ';') LIKE ?")
        params.append(f"%;{slot_type.strip().lower()};%")

    if cuisines:
        placeholders = ",".join("?" for _ in cuisines)
        sql.append(f"AND category IN ({placeholders})")
        params.extend(c.strip().lower() for c in cuisines)

    if max_cost_per_person is not None:
        sql.append("AND cost_per_person <= ?")
        params.append(float(max_cost_per_person))

    if neighbourhood:
        sql.append("AND neighbourhood = ?")
        params.append(neighbourhood)

    if open_on:
        day = open_on.strip().lower()
        sql.append("AND (';' || closed_days || ';') NOT LIKE ?")
        params.append(f"%;{day};%")

    if exclude_ids:
        placeholders = ",".join("?" for _ in exclude_ids)
        sql.append(f"AND venue_id NOT IN ({placeholders})")
        params.extend(exclude_ids)

    sql.append("ORDER BY rating DESC, venue_id ASC")

    with _connect() as conn:
        rows = _rows_to_dicts(conn.execute(" ".join(sql), params).fetchall())

    # Allergen exclusion is applied in code, not SQL: it is a safety filter and
    # belongs next to its tests, in one auditable place.
    if allergies:
        from ..guards import filter_allergens

        rows = filter_allergens(rows, allergies)

    if dietary:
        wanted = {d.strip().lower() for d in dietary if d.strip()}
        rows = [
            r
            for r in rows
            if wanted <= {d for d in r["dietary_options"].split(";") if d}
        ]

    if near:
        from .distance import haversine_km

        lat, lon = near
        rows.sort(key=lambda r: haversine_km(lat, lon, r["lat"], r["lon"]))

    return rows[:limit]


def cuisines_available() -> list[str]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT DISTINCT category FROM venues WHERE kind = 'restaurant' ORDER BY category"
        ).fetchall()
    return [row["category"] for row in rows]


def neighbourhoods_available() -> list[str]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT DISTINCT neighbourhood FROM venues ORDER BY neighbourhood"
        ).fetchall()
    return [row["neighbourhood"] for row in rows]
