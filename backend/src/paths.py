"""Filesystem layout. One place to change if the deployment shape changes.

Everything the running service reads lives under `backend/`, because Render's
rootDir is `backend` — data outside it would only work by accident of the full
checkout and would break the moment we containerise.
"""
from __future__ import annotations

from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = BACKEND_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
BUILD_DIR = DATA_DIR / "build"

RESTAURANTS_CSV = RAW_DIR / "calgary_restaurants.csv"
ATTRACTIONS_CSV = RAW_DIR / "calgary_attractions.csv"

SQLITE_PATH = BUILD_DIR / "foodie.sqlite"
DISTANCE_MATRIX_PATH = BUILD_DIR / "distance_matrix.json"
CHUNKS_PATH = BUILD_DIR / "chunks.jsonl"
