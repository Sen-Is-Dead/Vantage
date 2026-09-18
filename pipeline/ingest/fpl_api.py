"""FPL public API client + pure transform functions.

Endpoints used (all public, read-only, no auth):
  bootstrap-static/                 players, teams, events (gameweeks), chips, game_settings
  fixtures/                         all fixtures for the season
  element-summary/{player_id}/      per-player per-fixture history for the current season
  entry/{team_id}/                  the user's entry summary (rank, bank, value)
  entry/{team_id}/history/          per-gw points/transfers + chips used
  entry/{team_id}/event/{gw}/picks/ the user's 15 picks for a gw

SAFETY: this module only ever performs GET requests. There is deliberately no code here that
authenticates or writes to FPL (transfers, chips, lineups). See Safety Rules in fpl-ai-spec.md.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Callable, Iterable

import requests

from pipeline.config import (
    FPL_BASE_URL,
    FREE_TRANSFERS_PER_GW,
    MAX_FREE_TRANSFERS,
    POSITIONS,
)

log = logging.getLogger(__name__)

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; VantageFPL/0.1; +https://github.com/Sen-Is-Dead/Vantage)",
    "Accept": "application/json",
}


class FPLClient:
    """Small requests wrapper with retries and polite pacing."""

    def __init__(self, base_url: str = FPL_BASE_URL, session: requests.Session | None = None,
                 retries: int = 4, backoff: float = 1.5, pause: float = 0.05):
        self.base_url = base_url.rstrip("/")
        self.session = session or requests.Session()
        self.session.headers.update(_HEADERS)
        self.retries = retries
        self.backoff = backoff
        self.pause = pause

    def get(self, path: str) -> Any:
        url = f"{self.base_url}/{path.strip('/')}/"
        last_exc: Exception | None = None
        for attempt in range(self.retries):
            try:
                r = self.session.get(url, timeout=30)
                if r.status_code == 404:
                    raise FileNotFoundError(url)
                if r.status_code in (429, 500, 502, 503, 504):
                    raise requests.HTTPError(f"{r.status_code} from {url}")
                r.raise_for_status()
                time.sleep(self.pause)
                return r.json()
            except (requests.RequestException, ValueError) as exc:  # ValueError = bad JSON
                last_exc = exc
                sleep = self.backoff ** attempt
                log.warning("GET %s failed (%s); retry %d/%d in %.1fs", url, exc, attempt + 1, self.retries, sleep)
                time.sleep(sleep)
        raise RuntimeError(f"FPL API request failed after {self.retries} attempts: {url}") from last_exc

    # --- endpoints ---
    def bootstrap(self) -> dict[str, Any]:
        return self.get("bootstrap-static")

    def fixtures(self) -> list[dict[str, Any]]:
        return self.get("fixtures")

    def element_summary(self, player_id: int) -> dict[str, Any]:
        return self.get(f"element-summary/{player_id}")

    def entry(self, team_id: int) -> dict[str, Any]:
        return self.get(f"entry/{team_id}")

    def entry_history(self, team_id: int) -> dict[str, Any]:
        return self.get(f"entry/{team_id}/history")

    def entry_picks(self, team_id: int, gw: int) -> dict[str, Any]:
        return self.get(f"entry/{team_id}/event/{gw}/picks")


# ----------------------------------------------------------------------------------------------
# Pure transforms: API JSON -> rows matching pipeline/db/schema.sql. Unit-tested against fixtures.
# ----------------------------------------------------------------------------------------------

def _num(v: Any) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _ts(v: str | None) -> datetime | None:
    if not v:
        return None
    return datetime.fromisoformat(v.replace("Z", "+00:00")).astimezone(timezone.utc)


def season_label(bootstrap: dict[str, Any]) -> str:
    """'2026/27' derived from the first event's deadline (season starts in August)."""
    first = min(bootstrap["events"], key=lambda e: e["id"])
    dt = _ts(first["deadline_time"])
    start_year = dt.year if dt.month >= 7 else dt.year - 1
    return f"{start_year}/{str(start_year + 1)[-2:]}"


def gameweek_state(bootstrap: dict[str, Any]) -> dict[str, int | None]:
    events = bootstrap["events"]
    cur = next((e["id"] for e in events if e.get("is_current")), None)
    nxt = next((e["id"] for e in events if e.get("is_next")), None)
    finished = [e["id"] for e in events if e.get("finished")]
    return {"current": cur, "next": nxt, "last_finished": max(finished) if finished else None}


def transform_teams(bootstrap: dict[str, Any], season: str) -> list[dict[str, Any]]:
    return [
        {
            "id": t["id"], "season": season, "code": t.get("code"), "name": t["name"],
            "short_name": t.get("short_name"), "strength": t.get("strength"),
            "strength_overall_home": t.get("strength_overall_home"),
            "strength_overall_away": t.get("strength_overall_away"),
            "strength_attack_home": t.get("strength_attack_home"),
            "strength_attack_away": t.get("strength_attack_away"),
            "strength_defence_home": t.get("strength_defence_home"),
            "strength_defence_away": t.get("strength_defence_away"),
        }
        for t in bootstrap["teams"]
    ]


def transform_players(bootstrap: dict[str, Any], season: str) -> list[dict[str, Any]]:
    rows = []
    for e in bootstrap["elements"]:
        rows.append({
            "id": e["id"], "season": season, "code": e.get("code"),
            "web_name": e["web_name"], "first_name": e.get("first_name"), "second_name": e.get("second_name"),
            "team_id": e["team"], "position": POSITIONS[e["element_type"]],
            "price": e["now_cost"] / 10.0,
            "cost_change_start": (e.get("cost_change_start") or 0) / 10.0,
            "status": e.get("status"), "news": e.get("news") or None, "news_added": _ts(e.get("news_added")),
            "chance_of_playing_next_round": e.get("chance_of_playing_next_round"),
            "chance_of_playing_this_round": e.get("chance_of_playing_this_round"),
            "selected_by_percent": _num(e.get("selected_by_percent")),
            "total_points": e.get("total_points"), "form": _num(e.get("form")),
        })
    return rows


def transform_fixtures(fixtures: list[dict[str, Any]], season: str) -> list[dict[str, Any]]:
    return [
        {
            "id": f["id"], "season": season, "code": f.get("code"), "gw": f.get("event"),
            "home_team_id": f["team_h"], "away_team_id": f["team_a"], "kickoff_time": _ts(f.get("kickoff_time")),
            "difficulty_home": f.get("team_h_difficulty"), "difficulty_away": f.get("team_a_difficulty"),
            "finished": bool(f.get("finished")), "started": bool(f.get("started")),
            "home_score": f.get("team_h_score"), "away_score": f.get("team_a_score"),
        }
        for f in fixtures
    ]


def transform_player_history(player_id: int, summary: dict[str, Any], season: str) -> list[dict[str, Any]]:
    rows = []
    for h in summary.get("history", []):
        rows.append({
            "player_id": player_id, "season": season, "gw": h["round"], "fixture_id": h["fixture"],
            "opponent_team_id": h.get("opponent_team"), "was_home": h.get("was_home"),
            "kickoff_time": _ts(h.get("kickoff_time")),
            "minutes": h.get("minutes", 0), "points": h.get("total_points", 0),
            "goals": h.get("goals_scored", 0), "assists": h.get("assists", 0),
            "clean_sheet": h.get("clean_sheets", 0), "goals_conceded": h.get("goals_conceded", 0),
            "own_goals": h.get("own_goals", 0), "penalties_saved": h.get("penalties_saved", 0),
            "penalties_missed": h.get("penalties_missed", 0), "yellow_cards": h.get("yellow_cards", 0),
            "red_cards": h.get("red_cards", 0), "saves": h.get("saves", 0), "bonus": h.get("bonus", 0),
            "bps": h.get("bps", 0), "defensive_contribution": h.get("defensive_contribution", 0),
            "starts": h.get("starts", 0),
            "influence": _num(h.get("influence")), "creativity": _num(h.get("creativity")),
            "threat": _num(h.get("threat")), "ict_index": _num(h.get("ict_index")),
            "xg": _num(h.get("expected_goals")), "xa": _num(h.get("expected_assists")),
            "xgi": _num(h.get("expected_goal_involvements")), "xgc": _num(h.get("expected_goals_conceded")),
            "value": (h.get("value") or 0) / 10.0, "selected": h.get("selected"),
            "transfers_in": h.get("transfers_in"), "transfers_out": h.get("transfers_out"),
            "team_h_score": h.get("team_h_score"), "team_a_score": h.get("team_a_score"),
        })
    return rows


def transform_picks(picks_payload: dict[str, Any], season: str, gw: int) -> list[dict[str, Any]]:
    rows = []
    for p in picks_payload.get("picks", []):
        pos = p["position"]
        starting = pos <= 11
        rows.append({
            "season": season, "gw": gw, "player_id": p["element"], "is_starting": starting,
            "is_captain": bool(p.get("is_captain")), "is_vice_captain": bool(p.get("is_vice_captain")),
            "bench_order": None if starting else pos - 11, "squad_position": pos,
            "purchase_price": None,
        })
    return rows


def derive_free_transfers(history_current: list[dict[str, Any]], chips: list[dict[str, Any]]) -> dict[int, int]:
    """Return {gw: free transfers available going INTO gw+1}, replayed from the entry history.

    Rules (2024/25 onwards, confirmed for 2026/27 via game_settings.max_extra_free_transfers=4):
      - 1 FT is added after every gameweek, banked up to MAX_FREE_TRANSFERS (5).
      - Transfers made in a gw consume banked FTs first; the excess are paid hits (-4 each).
      - On a Wildcard / Free Hit gw, transfers made are free AND banked FTs are retained.
      - Going into GW2 you always have exactly 1 FT (GW1 transfers are unlimited).
    The public API never exposes the banked FT count directly, so this replay is the best
    available estimate; run_weekly cross-checks it against `event_transfers_cost` and logs a
    warning if the replay disagrees with the hits FPL actually charged.
    """
    chip_by_gw = {c["event"]: c["name"] for c in chips if c.get("name") in ("wildcard", "freehit")}
    out: dict[int, int] = {}
    ft = 1
    for row in sorted(history_current, key=lambda r: r["event"]):
        gw = row["event"]
        if gw == 1:
            ft = 1
        elif gw in chip_by_gw:
            ft = min(MAX_FREE_TRANSFERS, ft + FREE_TRANSFERS_PER_GW)
        else:
            used = row.get("event_transfers", 0) or 0
            ft = min(MAX_FREE_TRANSFERS, max(ft - used, 0) + FREE_TRANSFERS_PER_GW)
        out[gw] = ft
    return out


def transform_entry_state(history: dict[str, Any], picks_by_gw: dict[int, dict[str, Any]], season: str) -> list[dict[str, Any]]:
    chips = history.get("chips", []) or []
    ft_after = derive_free_transfers(history.get("current", []), chips)
    rows = []
    for h in history.get("current", []):
        gw = h["event"]
        picks = picks_by_gw.get(gw, {})
        rows.append({
            "season": season, "gw": gw,
            "bank": (h.get("bank") or 0) / 10.0, "squad_value": (h.get("value") or 0) / 10.0,
            "event_transfers": h.get("event_transfers"), "event_transfers_cost": h.get("event_transfers_cost"),
            "free_transfers_after": ft_after.get(gw),
            "active_chip": picks.get("active_chip"),
            "chips_used_json": [{"name": c["name"], "event": c["event"]} for c in chips if c.get("event") is not None and c["event"] <= gw],
            "points": h.get("points"), "total_points": h.get("total_points"),
            "overall_rank": h.get("overall_rank"), "gw_rank": h.get("rank"),
            "points_on_bench": h.get("points_on_bench"),
        })
    return rows


def fetch_all_player_histories(client: FPLClient, player_ids: Iterable[int], season: str,
                               progress: Callable[[int, int], None] | None = None) -> list[dict[str, Any]]:
    ids = list(player_ids)
    rows: list[dict[str, Any]] = []
    for i, pid in enumerate(ids, 1):
        try:
            rows.extend(transform_player_history(pid, client.element_summary(pid), season))
        except FileNotFoundError:
            log.warning("element-summary 404 for player %s (removed?)", pid)
        if progress and (i % 100 == 0 or i == len(ids)):
            progress(i, len(ids))
    return rows
