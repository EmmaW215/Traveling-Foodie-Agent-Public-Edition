#!/usr/bin/env python3
"""Build the runtime data artefacts from the source CSVs.

    python -m scripts.seed

Produces (all gitignored — the CSVs are the only source of truth):

    data/build/foodie.sqlite        venue catalogue queried by the tools
    data/build/distance_matrix.json every pairwise distance, precomputed
    data/build/chunks.jsonl         one text chunk per venue, for M3 RAG

Deterministic: same CSVs in, byte-identical artefacts out. That matters because
the build runs on every Render deploy and in CI, and a nondeterministic seed
would make "did the data change?" unanswerable.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.paths import (  # noqa: E402
    ATTRACTIONS_CSV,
    BUILD_DIR,
    CHUNKS_PATH,
    DISTANCE_MATRIX_PATH,
    RESTAURANTS_CSV,
    SQLITE_PATH,
)

EARTH_RADIUS_KM = 6371.0088

SCHEMA = """
DROP TABLE IF EXISTS venues;
CREATE TABLE venues (
    venue_id         TEXT PRIMARY KEY,
    kind             TEXT NOT NULL,           -- 'restaurant' | 'attraction'
    name             TEXT NOT NULL,
    neighbourhood    TEXT NOT NULL,
    lat              REAL NOT NULL,
    lon              REAL NOT NULL,
    category         TEXT NOT NULL,           -- cuisine, or attraction category
    slot_types       TEXT NOT NULL,           -- ';' list: breakfast/lunch/dinner or am/pm
    price_band       TEXT,
    cost_per_person  REAL NOT NULL,
    duration_min     INTEGER,
    closed_days      TEXT NOT NULL DEFAULT '',
    open_time        TEXT NOT NULL,
    close_time       TEXT NOT NULL,
    allergens_present TEXT NOT NULL DEFAULT '',
    dietary_options  TEXT NOT NULL DEFAULT '',
    rating           REAL NOT NULL,
    is_trap          TEXT NOT NULL DEFAULT '',
    description      TEXT NOT NULL DEFAULT ''
);
CREATE INDEX idx_venues_kind      ON venues(kind);
CREATE INDEX idx_venues_category  ON venues(category);
CREATE INDEX idx_venues_cost      ON venues(cost_per_person);

DROP TABLE IF EXISTS dataset_meta;
CREATE TABLE dataset_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance. Good enough for a walkable city grid."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def _norm_list(raw: str) -> str:
    """Normalise a ';' separated field: lowercase, trimmed, sorted, deduped."""
    items = sorted({item.strip().lower() for item in (raw or "").split(";") if item.strip()})
    return ";".join(items)


def load_restaurants() -> list[dict]:
    rows = []
    with RESTAURANTS_CSV.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows.append(
                {
                    "venue_id": row["venue_id"].strip(),
                    "kind": "restaurant",
                    "name": row["name"].strip(),
                    "neighbourhood": row["neighbourhood"].strip(),
                    "lat": float(row["lat"]),
                    "lon": float(row["lon"]),
                    "category": row["cuisine"].strip().lower(),
                    "slot_types": _norm_list(row["meal_types"]),
                    "price_band": row["price_band"].strip(),
                    "cost_per_person": float(row["cost_per_person"]),
                    "duration_min": 75,  # a sit-down meal, used by the route agent
                    "closed_days": _norm_list(row["closed_days"]),
                    "open_time": row["open_time"].strip(),
                    "close_time": row["close_time"].strip(),
                    "allergens_present": _norm_list(row["allergens_present"]),
                    "dietary_options": _norm_list(row["dietary_options"]),
                    "rating": float(row["rating"]),
                    "is_trap": row["is_trap"].strip().lower(),
                    "description": row["description"].strip(),
                }
            )
    return rows


def load_attractions() -> list[dict]:
    rows = []
    with ATTRACTIONS_CSV.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows.append(
                {
                    "venue_id": row["venue_id"].strip(),
                    "kind": "attraction",
                    "name": row["name"].strip(),
                    "neighbourhood": row["neighbourhood"].strip(),
                    "lat": float(row["lat"]),
                    "lon": float(row["lon"]),
                    "category": row["category"].strip().lower(),
                    "slot_types": _norm_list(row["slot_types"]),
                    "price_band": None,
                    "cost_per_person": float(row["cost_per_person"]),
                    "duration_min": int(row["duration_min"]),
                    "closed_days": _norm_list(row["closed_days"]),
                    "open_time": row["open_time"].strip(),
                    "close_time": row["close_time"].strip(),
                    "allergens_present": "",
                    "dietary_options": "",
                    "rating": float(row["rating"]),
                    "is_trap": row["is_trap"].strip().lower(),
                    "description": row["description"].strip(),
                }
            )
    return rows


def validate(venues: list[dict]) -> list[str]:
    """Catch data problems at build time, not at 2am in a demo."""
    errors: list[str] = []
    seen_ids: set[str] = set()
    valid_days = {"mon", "tue", "wed", "thu", "fri", "sat", "sun"}

    for v in venues:
        vid = v["venue_id"]
        if vid in seen_ids:
            errors.append(f"{vid}: duplicate venue_id")
        seen_ids.add(vid)

        # Calgary sits near 51.0 N, -114.1 W. A typo in a coordinate would
        # silently produce nonsense routes, so fence the whole metro area.
        if not (50.80 <= v["lat"] <= 51.25):
            errors.append(f"{vid}: latitude {v['lat']} outside Calgary")
        if not (-114.40 <= v["lon"] <= -113.85):
            errors.append(f"{vid}: longitude {v['lon']} outside Calgary")

        if v["cost_per_person"] < 0:
            errors.append(f"{vid}: negative cost")
        if not 0 <= v["rating"] <= 5:
            errors.append(f"{vid}: rating {v['rating']} out of range")

        for day in filter(None, v["closed_days"].split(";")):
            if day not in valid_days:
                errors.append(f"{vid}: unknown closed_day '{day}'")

        for field in ("open_time", "close_time"):
            value = v[field]
            if len(value) != 5 or value[2] != ":":
                errors.append(f"{vid}: {field} '{value}' is not HH:MM")

        if not v["slot_types"]:
            errors.append(f"{vid}: no slot_types")

    # The guards test suite depends on these existing.
    traps = {v["is_trap"] for v in venues if v["is_trap"]}
    for required in ("peanut_risk", "budget_buster", "closed_monday"):
        if required not in traps:
            errors.append(f"dataset is missing a '{required}' trap venue")

    return errors


def build_distance_matrix(venues: list[dict]) -> dict:
    """Every pairwise distance, precomputed. ~85 venues -> ~3.6k pairs: trivial.

    This is what replaces a paid routing API. Storing it symmetrically halves
    the file and makes the lookup helper the only place that needs to know.
    """
    matrix: dict[str, dict[str, float]] = {}
    for i, a in enumerate(venues):
        for b in venues[i + 1 :]:
            km = round(haversine_km(a["lat"], a["lon"], b["lat"], b["lon"]), 4)
            matrix.setdefault(a["venue_id"], {})[b["venue_id"]] = km
    return matrix


def build_chunks(venues: list[dict]) -> list[dict]:
    """One retrieval chunk per venue, for the M3 RAG index.

    Written now so the corpus is versioned alongside the data it came from;
    embedding and upserting happens in a separate CI job (M3).
    """
    chunks = []
    for v in venues:
        closed = v["closed_days"].replace(";", ", ") or "open daily"
        cost = (
            "Free to visit — no admission cost."
            if v["cost_per_person"] == 0
            else f"Typical cost per person: ${v['cost_per_person']:.0f} CAD."
        )
        parts = [
            f"{v['name']} is a {v['category']} {v['kind']} in {v['neighbourhood']}, Calgary.",
            v["description"],
            cost,
            f"Hours: {v['open_time']} to {v['close_time']}. Closed: {closed}.",
            f"Suitable for: {v['slot_types'].replace(';', ', ')}.",
        ]
        if v["allergens_present"]:
            parts.append(f"Allergens present in the kitchen: {v['allergens_present'].replace(';', ', ')}.")
        if v["dietary_options"]:
            parts.append(f"Dietary options: {v['dietary_options'].replace(';', ', ')}.")
        parts.append(f"Rating: {v['rating']}/5.")

        chunks.append(
            {
                "id": v["venue_id"],
                "text": " ".join(p for p in parts if p),
                "metadata": {
                    "venue_id": v["venue_id"],
                    "name": v["name"],
                    "kind": v["kind"],
                    "category": v["category"],
                    "neighbourhood": v["neighbourhood"],
                    "cost_per_person": v["cost_per_person"],
                },
            }
        )
    return chunks


def write_sqlite(venues: list[dict], data_version: str) -> None:
    SQLITE_PATH.unlink(missing_ok=True)
    conn = sqlite3.connect(SQLITE_PATH)
    try:
        conn.executescript(SCHEMA)
        conn.executemany(
            """INSERT INTO venues (
                venue_id, kind, name, neighbourhood, lat, lon, category, slot_types,
                price_band, cost_per_person, duration_min, closed_days, open_time,
                close_time, allergens_present, dietary_options, rating, is_trap, description
            ) VALUES (
                :venue_id, :kind, :name, :neighbourhood, :lat, :lon, :category, :slot_types,
                :price_band, :cost_per_person, :duration_min, :closed_days, :open_time,
                :close_time, :allergens_present, :dietary_options, :rating, :is_trap, :description
            )""",
            venues,
        )
        restaurants = sum(1 for v in venues if v["kind"] == "restaurant")
        conn.executemany(
            "INSERT INTO dataset_meta (key, value) VALUES (?, ?)",
            [
                ("data_version", data_version),
                ("cities", "Calgary"),
                ("restaurants", str(restaurants)),
                ("attractions", str(len(venues) - restaurants)),
                (
                    "data_disclaimer",
                    "Venues are fictional; Calgary geography is real. Demo data — do not "
                    "rely on hours, prices or allergen information. See backend/data/README.md.",
                ),
            ],
        )
        conn.commit()
    finally:
        conn.close()


def main() -> int:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)

    venues = load_restaurants() + load_attractions()
    venues.sort(key=lambda v: v["venue_id"])  # determinism

    errors = validate(venues)
    if errors:
        print("SEED FAILED — dataset validation errors:")
        for err in errors:
            print(f"  - {err}")
        return 1

    # Version = hash of the source CSVs, so a changed dataset is visible in the
    # API without anyone having to remember to bump a number.
    digest = hashlib.sha256()
    for path in (RESTAURANTS_CSV, ATTRACTIONS_CSV):
        digest.update(path.read_bytes())
    data_version = digest.hexdigest()[:12]

    write_sqlite(venues, data_version)

    matrix = build_distance_matrix(venues)
    DISTANCE_MATRIX_PATH.write_text(
        json.dumps(matrix, sort_keys=True, separators=(",", ":")), encoding="utf-8"
    )

    chunks = build_chunks(venues)
    with CHUNKS_PATH.open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(json.dumps(chunk, sort_keys=True, ensure_ascii=False) + "\n")

    pairs = sum(len(v) for v in matrix.values())
    restaurants = sum(1 for v in venues if v["kind"] == "restaurant")
    print(
        f"seed ok: {len(venues)} venues "
        f"({restaurants} restaurants, {len(venues) - restaurants} attractions), "
        f"{pairs} distance pairs, {len(chunks)} chunks, data_version={data_version}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
