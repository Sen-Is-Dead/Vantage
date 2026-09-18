"""Historical seasons from the vaastav/Fantasy-Premier-League archive (public GitHub repo, MIT-ish
community dataset built from the same FPL API). Needed because the FPL API only serves the current
season's per-gameweek history, and a model trained on 4 gameweeks is not a model.

Loads, per season: teams.csv -> teams, fixtures.csv -> fixtures, players_raw.csv (for stable player
`code` + position) and gws/merged_gw.csv -> player_gw_stats (source='vaastav').

    python -m pipeline.run_weekly history --seasons 2022-23 2023-24 2024-25 2025-26
"""
from __future__ import annotations

import io
import logging
from typing import Any

import pandas as pd
import psycopg
import requests

from pipeline.db.conn import upsert

log = logging.getLogger(__name__)

RAW_BASE = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
DEFAULT_SEASONS = ("2022-23", "2023-24", "2024-25", "2025-26")
_POS = {"GK": "GKP", "GKP": "GKP", "DEF": "DEF", "MID": "MID", "FWD": "FWD"}


def season_label_from_dir(s: str) -> str:
    """'2025-26' -> '2025/26' (matches season_label() used for the live API)."""
    a, b = s.split("-")
    return f"{a}/{b}"


def _fetch_csv(path: str, session: requests.Session) -> pd.DataFrame:
    r = session.get(f"{RAW_BASE}/{path}", timeout=60)
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text))


def _none(v: Any) -> Any:
    return None if pd.isna(v) else v


def transform_history_season(season_dir: str, teams: pd.DataFrame, fixtures: pd.DataFrame,
                             players_raw: pd.DataFrame, merged_gw: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    """Pure transform of the four archive CSVs into rows for teams / fixtures / player_gw_stats."""
    season = season_label_from_dir(season_dir)

    team_rows = [{
        "season": season, "id": int(t.id), "code": _none(t.code), "name": t.name, "short_name": _none(t.short_name),
        "strength": _none(t.strength),
        "strength_overall_home": _none(t.strength_overall_home), "strength_overall_away": _none(t.strength_overall_away),
        "strength_attack_home": _none(t.strength_attack_home), "strength_attack_away": _none(t.strength_attack_away),
        "strength_defence_home": _none(t.strength_defence_home), "strength_defence_away": _none(t.strength_defence_away),
    } for t in teams.itertuples()]
    team_id_by_name = {t.name: int(t.id) for t in teams.itertuples()}

    fx = fixtures.copy()
    fixture_rows = [{
        "season": season, "id": int(f.id), "code": _none(f.code), "gw": None if pd.isna(f.event) else int(f.event),
        "home_team_id": int(f.team_h), "away_team_id": int(f.team_a),
        "kickoff_time": None if pd.isna(f.kickoff_time) else pd.Timestamp(f.kickoff_time).to_pydatetime(),
        "difficulty_home": _none(f.team_h_difficulty), "difficulty_away": _none(f.team_a_difficulty),
        "finished": bool(f.finished), "started": bool(getattr(f, "started", f.finished)),
        "home_score": None if pd.isna(f.team_h_score) else int(f.team_h_score),
        "away_score": None if pd.isna(f.team_a_score) else int(f.team_a_score),
    } for f in fx.itertuples()]

    code_by_id = dict(zip(players_raw["id"].astype(int), players_raw["code"].astype(int)))

    g = merged_gw.drop_duplicates(["element", "fixture"], keep="last")
    num = lambda col: pd.to_numeric(g[col], errors="coerce") if col in g.columns else pd.Series(0, index=g.index)  # noqa: E731
    out = pd.DataFrame({
        "season": season,
        "player_id": g["element"].astype(int),
        "player_code": g["element"].astype(int).map(code_by_id),
        "position": g["position"].map(_POS),
        "team_id": g["team"].map(team_id_by_name),
        "gw": g["GW"].astype(int),
        "fixture_id": g["fixture"].astype(int),
        "opponent_team_id": num("opponent_team"),
        "was_home": g["was_home"].astype(bool),
        "kickoff_time": pd.to_datetime(g["kickoff_time"], utc=True),
        "minutes": num("minutes").fillna(0).astype(int),
        "points": num("total_points").fillna(0).astype(int),
        "goals": num("goals_scored").fillna(0).astype(int),
        "assists": num("assists").fillna(0).astype(int),
        "clean_sheet": num("clean_sheets").fillna(0).astype(int),
        "goals_conceded": num("goals_conceded").fillna(0).astype(int),
        "own_goals": num("own_goals").fillna(0).astype(int),
        "penalties_saved": num("penalties_saved").fillna(0).astype(int),
        "penalties_missed": num("penalties_missed").fillna(0).astype(int),
        "yellow_cards": num("yellow_cards").fillna(0).astype(int),
        "red_cards": num("red_cards").fillna(0).astype(int),
        "saves": num("saves").fillna(0).astype(int),
        "bonus": num("bonus").fillna(0).astype(int),
        "bps": num("bps").fillna(0).astype(int),
        "defensive_contribution": num("defensive_contribution").fillna(0).astype(int),
        "starts": num("starts").fillna(0).astype(int),
        "influence": num("influence"), "creativity": num("creativity"), "threat": num("threat"),
        "ict_index": num("ict_index"),
        "xg": num("expected_goals"), "xa": num("expected_assists"),
        "xgi": num("expected_goal_involvements"), "xgc": num("expected_goals_conceded"),
        "value": num("value") / 10.0,
        "selected": num("selected"), "transfers_in": num("transfers_in"), "transfers_out": num("transfers_out"),
        "team_h_score": num("team_h_score"), "team_a_score": num("team_a_score"),
        "source": "vaastav",
    })
    # NaN -> None for psycopg; ints stay ints
    out = out.astype(object).where(pd.notna(out), None)
    for c in ("opponent_team_id", "team_id", "player_code", "selected", "transfers_in", "transfers_out",
              "team_h_score", "team_a_score"):
        out[c] = out[c].map(lambda v: None if v is None else int(v))
    stat_rows = out.to_dict("records")
    return {"teams": team_rows, "fixtures": fixture_rows, "player_gw_stats": stat_rows}


def load_history(conn: psycopg.Connection, seasons: tuple[str, ...] = DEFAULT_SEASONS,
                 session: requests.Session | None = None) -> dict[str, int]:
    session = session or requests.Session()
    counts: dict[str, int] = {}
    for sd in seasons:
        log.info("history: fetching %s", sd)
        teams = _fetch_csv(f"{sd}/teams.csv", session)
        fixtures = _fetch_csv(f"{sd}/fixtures.csv", session)
        players_raw = _fetch_csv(f"{sd}/players_raw.csv", session)
        merged = _fetch_csv(f"{sd}/gws/merged_gw.csv", session)
        rows = transform_history_season(sd, teams, fixtures, players_raw, merged)
        upsert(conn, "teams", rows["teams"], ["season", "id"])
        upsert(conn, "fixtures", rows["fixtures"], ["season", "id"])
        n = upsert(conn, "player_gw_stats", rows["player_gw_stats"], ["season", "player_id", "fixture_id"])
        conn.commit()
        counts[season_label_from_dir(sd)] = n
        log.info("history: %s -> %d player_gw_stats rows", sd, n)
    return counts
