"""Orchestrates one ingestion run: FPL API -> Postgres. Idempotent (upserts everywhere)."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import psycopg
from psycopg.types.json import Jsonb

from pipeline import config
from pipeline.db.conn import apply_schema, scalar, upsert
from pipeline.ingest import understat
from pipeline.ingest.fpl_api import (
    FPLClient,
    fetch_all_player_histories,
    gameweek_state,
    season_label,
    transform_chip_windows,
    transform_entry_state,
    transform_fixtures,
    transform_picks,
    transform_players,
    transform_teams,
)

log = logging.getLogger(__name__)


@dataclass
class IngestResult:
    season: str
    current_gw: int | None
    next_gw: int | None
    n_teams: int = 0
    n_players: int = 0
    n_fixtures: int = 0
    n_gw_stats: int = 0
    n_user_squad_rows: int = 0
    n_entry_state_rows: int = 0
    n_understat: int = 0
    free_transfers_now: int | None = None
    ft_replay_warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"season={self.season} current_gw={self.current_gw} next_gw={self.next_gw} | "
            f"teams={self.n_teams} players={self.n_players} fixtures={self.n_fixtures} "
            f"player_gw_stats={self.n_gw_stats} user_squad={self.n_user_squad_rows} "
            f"entry_state={self.n_entry_state_rows} understat={self.n_understat} "
            f"free_transfers_now={self.free_transfers_now}"
        )


def _check_ft_replay(history_current: list[dict[str, Any]], ft_after: dict[int, int]) -> list[str]:
    """Compare the replayed FT bank against hits FPL actually charged; disagreement = warning."""
    warnings = []
    prev = 1
    for h in sorted(history_current, key=lambda r: r["event"]):
        gw = h["event"]
        if gw > 1:
            used, cost = h.get("event_transfers", 0) or 0, h.get("event_transfers_cost", 0) or 0
            expected_cost = max(used - prev, 0) * config.TRANSFER_HIT_COST
            if cost != expected_cost and cost != 0:
                warnings.append(f"GW{gw}: replay expected hit cost {expected_cost}, FPL charged {cost} (chip week?)")
        prev = ft_after.get(gw, prev)
    return warnings


def run_ingest(conn: psycopg.Connection, client: FPLClient | None = None, team_id: int | None = None,
               include_histories: bool = True, include_understat: bool | None = None) -> IngestResult:
    client = client or FPLClient()
    team_id = team_id or config.FPL_TEAM_ID
    include_understat = config.UNDERSTAT_ENABLED if include_understat is None else include_understat

    apply_schema(conn)

    bootstrap = client.bootstrap()
    season = season_label(bootstrap)
    gws = gameweek_state(bootstrap)
    res = IngestResult(season=season, current_gw=gws["current"], next_gw=gws["next"])
    log.info("bootstrap ok: %s", res.summary())

    res.n_teams = upsert(conn, "teams", transform_teams(bootstrap, season), ["season", "id"])
    # players is a current-season snapshot: clear rows from any other season first
    conn.execute("DELETE FROM players WHERE season <> %s", (season,))
    res.n_players = upsert(conn, "players", transform_players(bootstrap, season), ["id"])
    res.n_fixtures = upsert(conn, "fixtures", transform_fixtures(client.fixtures(), season), ["season", "id"])
    upsert(conn, "chip_windows", transform_chip_windows(bootstrap, season), ["season", "name", "start_event"])
    conn.commit()

    if include_histories:
        rows = fetch_all_player_histories(
            client, bootstrap["elements"], season, progress=lambda i, n: log.info("element-summary %d/%d", i, n)
        )
        res.n_gw_stats = upsert(conn, "player_gw_stats", rows, ["season", "player_id", "fixture_id"])
        conn.commit()

    # --- the user's entry (read-only) ---
    try:
        history = client.entry_history(team_id)
        played = [h["event"] for h in history.get("current", [])]
        picks_by_gw: dict[int, dict[str, Any]] = {}
        for gw in played:
            try:
                picks_by_gw[gw] = client.entry_picks(team_id, gw)
            except FileNotFoundError:
                log.warning("no picks for GW%d", gw)
        squad_rows = [r for gw, p in picks_by_gw.items() for r in transform_picks(p, season, gw)]
        res.n_user_squad_rows = upsert(conn, "user_squad", squad_rows, ["season", "gw", "player_id"])

        state_rows = transform_entry_state(history, picks_by_gw, season, bootstrap.get("total_players"))
        for r in state_rows:
            r["chips_used_json"] = Jsonb(r["chips_used_json"])
        res.n_entry_state_rows = upsert(conn, "user_entry_state", state_rows, ["season", "gw"])
        if state_rows:
            latest = max(state_rows, key=lambda r: r["gw"])
            res.free_transfers_now = latest["free_transfers_after"]
        ft_after = {r["gw"]: r["free_transfers_after"] for r in state_rows}
        res.ft_replay_warnings = _check_ft_replay(history.get("current", []), ft_after)
        conn.commit()
    except FileNotFoundError:
        res.notes.append(f"entry {team_id} not found on FPL API")
        log.error("entry %s not found", team_id)

    # --- optional Understat ---
    if include_understat:
        ok, msg = understat.probe()
        res.notes.append(msg)
        if ok:
            try:
                year = int(season.split("/")[0])
                players = understat.fetch_league_players(year)
                rows: list[dict[str, Any]] = []
                for p in players:
                    rows.extend(understat.transform_player_matches(int(p["id"]), p["player_name"],
                                                                    understat.fetch_player_matches(int(p["id"]))))
                res.n_understat = upsert(conn, "understat_player_match", rows, ["understat_player_id", "match_date"])
                conn.commit()
            except Exception as exc:  # noqa: BLE001
                res.notes.append(f"understat load failed: {exc}")
                log.warning("understat load failed: %s", exc)

    conn.execute(
        "INSERT INTO ingest_log (season, current_gw, n_players, n_teams, n_fixtures, n_gw_stats, n_understat, notes) "
        "VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
        (season, res.current_gw, res.n_players, res.n_teams, res.n_fixtures, res.n_gw_stats, res.n_understat,
         "; ".join(res.notes + res.ft_replay_warnings) or None),
    )
    conn.commit()
    return res


def db_counts(conn: psycopg.Connection) -> dict[str, int]:
    tables = ["teams", "players", "fixtures", "player_gw_stats", "user_squad", "user_entry_state",
              "understat_player_match", "predictions", "recommendations", "model_runs"]
    return {t: scalar(conn, f"SELECT count(*) FROM {t}") for t in tables}
