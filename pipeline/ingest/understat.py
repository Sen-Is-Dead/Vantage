"""Optional Understat scrape (public pages, no auth). Disabled by default (UNDERSTAT_ENABLED=false).

Understat embeds its data as `var playersData = JSON.parse('<hex-escaped json>')` inside the HTML.
That has been stable for years but it is an unofficial source with no API contract, so:
  * every failure is caught and logged; the pipeline NEVER fails because Understat is down,
  * v1 features use FPL's own expected_goals / expected_assists (in element-summary since 2022/23),
    so this source is purely supplementary (used later for a team-strength Elo, section 4 of the spec).
"""
from __future__ import annotations

import json
import logging
import re
from datetime import date, datetime
from typing import Any

import requests

log = logging.getLogger(__name__)

_LEAGUE_URL = "https://understat.com/league/EPL/{year}"
_PLAYER_URL = "https://understat.com/player/{pid}"
_HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"}


def _extract_var(html: str, name: str) -> Any:
    m = re.search(rf"var\s+{name}\s*=\s*JSON\.parse\('((?:\\.|[^'])*)'\)", html)
    if not m:
        raise ValueError(f"{name} not found in page (Understat markup may have changed)")
    raw = m.group(1).encode("utf-8").decode("unicode_escape")
    return json.loads(raw)


def fetch_league_players(season_start_year: int, session: requests.Session | None = None) -> list[dict[str, Any]]:
    s = session or requests.Session()
    r = s.get(_LEAGUE_URL.format(year=season_start_year), headers=_HEADERS, timeout=30)
    r.raise_for_status()
    return _extract_var(r.text, "playersData")


def fetch_player_matches(understat_player_id: int, session: requests.Session | None = None) -> list[dict[str, Any]]:
    s = session or requests.Session()
    r = s.get(_PLAYER_URL.format(pid=understat_player_id), headers=_HEADERS, timeout=30)
    r.raise_for_status()
    return _extract_var(r.text, "matchesData")


def transform_player_matches(understat_player_id: int, player_name: str, matches: list[dict[str, Any]],
                             since: date | None = None) -> list[dict[str, Any]]:
    rows = []
    for m in matches:
        d = datetime.strptime(m["date"], "%Y-%m-%d").date()
        if since and d < since:
            continue
        home = m.get("h_a") == "h"
        rows.append({
            "understat_player_id": understat_player_id, "player_name": player_name, "match_date": d,
            "team": m.get("h_team") if home else m.get("a_team"),
            "opponent": m.get("a_team") if home else m.get("h_team"),
            "was_home": home, "minutes": int(m.get("time") or 0),
            "xg": float(m.get("xG") or 0), "xa": float(m.get("xA") or 0), "npxg": float(m.get("npxG") or 0),
            "shots": int(m.get("shots") or 0), "key_passes": int(m.get("key_passes") or 0),
            "fpl_player_code": None,
        })
    return rows


def probe() -> tuple[bool, str]:
    """Cheap viability check used by run_weekly: can we fetch and parse the league page?"""
    try:
        year = date.today().year if date.today().month >= 7 else date.today().year - 1
        players = fetch_league_players(year)
        return True, f"understat ok: {len(players)} players on EPL/{year}"
    except Exception as exc:  # noqa: BLE001 - we want to report any failure, never raise
        return False, f"understat unavailable: {exc}"
