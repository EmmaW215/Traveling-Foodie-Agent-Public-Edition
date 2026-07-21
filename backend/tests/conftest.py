"""Shared fixtures. The seed runs once per session so tests exercise the real
artefacts rather than a hand-built fake that could drift from the schema.
"""
import subprocess
import sys
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session", autouse=True)
def built_dataset():
    """Build data/build/* before any test that touches the catalogue."""
    result = subprocess.run(
        [sys.executable, "-m", "scripts.seed"],
        cwd=BACKEND_ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.fail(f"seed failed:\n{result.stdout}\n{result.stderr}")
    return result.stdout


@pytest.fixture()
def venues():
    from src.tools import catalog

    return catalog.all_venues()


@pytest.fixture()
def peanut_trap(venues):
    """The venue planted specifically to defeat prompt-only allergen handling."""
    return next(v for v in venues.values() if v["is_trap"] == "peanut_risk")


@pytest.fixture()
def monday_closed_venues(venues):
    return [v for v in venues.values() if v["is_trap"] == "closed_monday"]


@pytest.fixture()
def budget_busters(venues):
    return [v for v in venues.values() if v["is_trap"] == "budget_buster"]
