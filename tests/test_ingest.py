from __future__ import annotations

from pipeline.ingest.fpl_api import (
    derive_free_transfers,
    gameweek_state,
    season_label,
    transform_entry_state,
    transform_fixtures,
    transform_picks,
    transform_player_history,
    transform_players,
    transform_teams,
)
from pipeline.ingest.load import db_counts, run_ingest


def test_season_and_gw_state(bootstrap):
    assert season_label(bootstrap) == "2026/27"
    assert gameweek_state(bootstrap) == {"current": 4, "next": 5, "last_finished": 4}


def test_transform_players_maps_positions_prices_and_status(bootstrap):
    rows = {r["id"]: r for r in transform_players(bootstrap, "2026/27")}
    assert rows[1]["position"] == "GKP" and rows[1]["price"] == 6.0 and rows[1]["cost_change_start"] == 0.5
    assert rows[165]["web_name"] == "Mbeumo" and rows[165]["position"] == "MID"
    inj = rows[200]
    assert inj["status"] == "i" and inj["chance_of_playing_next_round"] == 0
    assert inj["news_added"].year == 2026 and inj["news"].startswith("Hamstring")
    assert rows[1]["news"] is None  # empty string normalised to NULL


def test_transform_teams(bootstrap):
    rows = transform_teams(bootstrap, "2026/27")
    assert len(rows) == 3 and rows[0]["strength_attack_home"] == 1360


def test_transform_fixtures_handles_unscheduled(fake_client):
    rows = transform_fixtures(fake_client.fixtures(), "2026/27")
    by_id = {r["id"]: r for r in rows}
    assert by_id[1]["gw"] == 1 and by_id[1]["finished"] is True and by_id[1]["home_score"] == 2
    assert by_id[99]["gw"] is None and by_id[99]["kickoff_time"] is None


def test_transform_player_history(fake_client):
    rows = transform_player_history(165, fake_client.element_summary(165), "2026/27")
    assert len(rows) == 3  # MUN plays in GW2, GW3, GW4 of the fixture set
    r = rows[0]
    assert r["gw"] == 2 and r["xg"] == 0.45 and r["value"] == 8.0 and r["minutes"] == 90
    assert all(k in r for k in ("points", "bonus", "bps", "defensive_contribution", "xgc"))


def test_transform_picks_bench_order(fake_client):
    rows = transform_picks(fake_client.entry_picks(1, 4), "2026/27", 4)
    by_id = {r["player_id"]: r for r in rows}
    assert by_id[165]["is_captain"] and by_id[165]["is_starting"] and by_id[165]["bench_order"] is None
    assert by_id[464]["is_vice_captain"]
    assert by_id[200]["is_starting"] is False and by_id[200]["bench_order"] == 1


def test_derive_free_transfers_replay():
    # GW1 -> 1 FT. GW2 no transfers -> 2. GW3 uses 3 (1 hit) -> 0+1 = 1. GW4 uses 2 (1 hit) -> 1. Cap at 5.
    hist = [
        {"event": 1, "event_transfers": 0}, {"event": 2, "event_transfers": 0},
        {"event": 3, "event_transfers": 3}, {"event": 4, "event_transfers": 2},
    ]
    assert derive_free_transfers(hist, []) == {1: 1, 2: 2, 3: 1, 4: 1}
    quiet = [{"event": i, "event_transfers": 0} for i in range(1, 9)]
    assert derive_free_transfers(quiet, [])[8] == 5  # banked, capped at 5
    # Wildcard week: transfers are free and the bank is retained (+1).
    wc = [{"event": 1, "event_transfers": 0}, {"event": 2, "event_transfers": 0}, {"event": 3, "event_transfers": 11}]
    assert derive_free_transfers(wc, [{"name": "wildcard", "event": 3}])[3] == 3


def test_transform_entry_state(fake_client):
    hist = fake_client.entry_history(1)
    picks = {gw: fake_client.entry_picks(1, gw) for gw in range(1, 5)}
    rows = transform_entry_state(hist, picks, "2026/27")
    last = rows[-1]
    assert last["gw"] == 4 and last["bank"] == 0.1 and last["squad_value"] == 100.1
    assert last["free_transfers_after"] == 1 and last["chips_used_json"] == []


def test_full_ingest_into_postgres(db_conn, fake_client):
    res = run_ingest(db_conn, client=fake_client, team_id=3964630, include_histories=True, include_understat=False)
    assert res.season == "2026/27" and res.current_gw == 4 and res.next_gw == 5
    counts = db_counts(db_conn)
    assert counts["teams"] == 3 and counts["players"] == 5 and counts["fixtures"] == 6
    assert counts["player_gw_stats"] == sum(
        len(fake_client.element_summary(pid)["history"]) for pid in (1, 8, 165, 200, 464)
    )
    assert counts["user_squad"] == 4 * 5 and counts["user_entry_state"] == 4
    assert res.free_transfers_now == 1
    # idempotent: a second run changes nothing
    res2 = run_ingest(db_conn, client=fake_client, team_id=3964630, include_histories=True, include_understat=False)
    assert db_counts(db_conn) == counts and res2.n_players == 5
    # protected player is readable with the status the safety rule keys off
    row = db_conn.execute("SELECT status FROM players WHERE lower(web_name) = 'mbeumo'").fetchone()
    assert row["status"] == "a"
    n_log = db_conn.execute("SELECT count(*) AS n FROM ingest_log").fetchone()["n"]
    assert n_log == 2
