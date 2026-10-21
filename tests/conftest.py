"""Shared test setup: put ``src/`` on the import path and expose the data directory."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture(scope="session")
def data_dir() -> Path:
    return ROOT / "assignment"
