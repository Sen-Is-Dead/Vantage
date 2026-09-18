"""Shared test fixtures. A fake FPL client serves the JSON under tests/fixtures so transforms and
the loader are exercised without network. DB tests need TEST_DATABASE_URL (skipped otherwise)."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


class FakeFPLClient:
    def __init__(self, root: Path = FIXTURES):
        self.root = root
        self.calls: list[str] = []

    def _load(self, name: str):
        self.calls.append(name)
        p = self.root / f"{name}.json"
        if not p.exists():
            raise FileNotFoundError(name)
        return json.loads(p.read_text())

    def bootstrap(self):
        return self._load("bootstrap_static")

    def fixtures(self):
        return self._load("fixtures")

    def element_summary(self, player_id: int):
        return self._load(f"element_summary_{player_id}")

    def entry(self, team_id: int):
        return self._load("entry")

    def entry_history(self, team_id: int):
        return self._load("entry_history")

    def entry_picks(self, team_id: int, gw: int):
        return self._load(f"entry_picks_{gw}")


@pytest.fixture
def fake_client() -> FakeFPLClient:
    return FakeFPLClient()


@pytest.fixture
def bootstrap():
    return json.loads((FIXTURES / "bootstrap_static.json").read_text())


@pytest.fixture
def db_conn():
    url = os.getenv("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL not set")
    import psycopg
    from psycopg.rows import dict_row

    with psycopg.connect(url, row_factory=dict_row) as conn:
        conn.execute("DROP SCHEMA public CASCADE; CREATE SCHEMA public;")
        conn.commit()
        yield conn
