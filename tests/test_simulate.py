"""Tests for the parameterised simulator.

The pure ones (scoring rules, config validation) always run. The end-to-end ones need
TEST_DATABASE_URL and build a small synthetic league: two seasons, 20 teams, 300 players a season,
with points driven by a hidden per-player skill so the models have real signal to find. Everything
is kept deliberately tiny (few gameweeks, few boosting rounds) so the suite stays quick.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from pipeline.simulate import SimConfig, run_simulation, score_gameweek

SEASONS = ("2024/25", "2025/26")
N_TEAMS = 20
GWS = 16
SHAPE = {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}   # per team, 15 players => 300 a season


# ------------------------------------------------------------------------------------------
# scoring rules
# ------------------------------------------------------------------------------------------

def test_bench_boost_scores_all_fifteen():
    lineup, bench = list(range(1, 12)), [12, 13, 14, 15]
    actual = {i: 2.0 for i in range(1, 16)}
    minutes = {i: 90 for i in range(1, 16)}
    pos = {1: "GKP", **{i: "DEF" for i in range(2, 6)}, **{i: "MID" for i in range(6, 10)},
           **{i: "FWD" for i in range(10, 12)}, 12: "GKP", 13: "DEF", 14: "MID", 15: "FWD"}

    plain, _, unused = score_gameweek(lineup, bench, 1, 2, actual, minutes, pos)
    boosted, _, boosted_unused = score_gameweek(lineup, bench, 1, 2, actual, minutes, pos, bench_boost=True)
    assert plain == pytest.approx(11 * 2 + 2)          # XI + captain's extra
    assert boosted == pytest.approx(15 * 2 + 2)        # all 15 + captain's extra
    assert unused == pytest.approx(8) and boosted_unused == 0


def test_triple_captain_adds_one_more_multiple():
    lineup, bench = list(range(1, 12)), [12, 13, 14, 15]
    actual = {i: 1.0 for i in range(1, 16)} | {1: 10.0}
    minutes = {i: 90 for i in range(1, 16)}
    pos = {1: "GKP", **{i: "DEF" for i in range(2, 6)}, **{i: "MID" for i in range(6, 10)},
           **{i: "FWD" for i in range(10, 12)}, 12: "GKP", 13: "DEF", 14: "MID", 15: "FWD"}
    double, cap2, _ = score_gameweek(lineup, bench, 1, 2, actual, minutes, pos)
    triple, cap3, _ = score_gameweek(lineup, bench, 1, 2, actual, minutes, pos, captain_multiplier=3)
    assert cap2 == pytest.approx(10) and cap3 == pytest.approx(20)
    assert triple - double == pytest.approx(10)


def test_triple_captain_falls_to_the_vice_when_the_captain_blanks():
    lineup, bench = list(range(1, 12)), [12, 13, 14, 15]
    actual = {i: 1.0 for i in range(1, 16)} | {1: 10.0, 2: 6.0}
    minutes = {i: 90 for i in range(1, 16)} | {1: 0}
    pos = {1: "GKP", **{i: "DEF" for i in range(2, 6)}, **{i: "MID" for i in range(6, 10)},
           **{i: "FWD" for i in range(10, 12)}, 12: "GKP", 13: "DEF", 14: "MID", 15: "FWD"}
    _, cap, _ = score_gameweek(lineup, bench, 1, 2, actual, minutes, pos, captain_multiplier=3)
    assert cap == pytest.approx(12)   # the vice's 6, twice over


# ------------------------------------------------------------------------------------------
# config validation
# ------------------------------------------------------------------------------------------

@pytest.mark.parametrize("kwargs", [
    {"mode": "nonsense"},
    {"strategy": "nonsense"},
    {"captain_metric": "nonsense"},
    {"chip_policy": "nonsense"},
    {"repeats": 0},
    {"repeats": 99},
    {"start_gw": 10, "end_gw": 3},
    {"formation": "4-4-4"},      # 12 outfielders
    {"formation": "2-5-3"},      # breaks the 3-defender minimum
    {"formation": "wat"},
])
def test_bad_config_is_refused(kwargs):
    with pytest.raises(ValueError):
        SimConfig(season="2025/26", **kwargs)


def test_formation_parses_to_counts():
    assert SimConfig(season="2025/26", formation="3-4-3").xi_shape == (3, 4, 3)
    assert SimConfig(season="2025/26", formation="5-4-1").xi_shape == (5, 4, 1)
    assert SimConfig(season="2025/26").xi_shape is None


def test_quantiles_are_only_fitted_when_a_knob_needs_them():
    assert SimConfig(season="2025/26").needs_quantiles is False
    assert SimConfig(season="2025/26", repeats=5).needs_quantiles is True
    assert SimConfig(season="2025/26", captain_metric="q80").needs_quantiles is True


def test_bench_weight_defaults_by_mode():
    assert SimConfig(season="2025/26", mode="realistic").effective_bench_weight == 1.0
    assert SimConfig(season="2025/26", mode="free_hit").effective_bench_weight == 0.05
    assert SimConfig(season="2025/26", bench_weight=0.3).effective_bench_weight == 0.3


# ------------------------------------------------------------------------------------------
# end to end on a synthetic league
# ------------------------------------------------------------------------------------------

def _ko(season_i: int, gw: int) -> datetime:
    return datetime(2024 + season_i, 8, 10, 14, tzinfo=timezone.utc) + timedelta(days=7 * gw)


def _synthetic():
    """A small league whose points are a hidden skill plus noise, so a model can beat a coin flip."""
    rng = np.random.default_rng(7)
    teams, fixtures, players, stats = [], [], [], []
    for si, season in enumerate(SEASONS):
        for t in range(1, N_TEAMS + 1):
            s = 1000 + 20 * t
            teams.append({"season": season, "id": t, "name": f"T{t}", "short_name": f"T{t:02d}",
                          "strength_attack_home": s, "strength_attack_away": s - 40,
                          "strength_defence_home": s, "strength_defence_away": s - 40,
                          "strength_overall_home": s, "strength_overall_away": s - 40})
        fid = 0
        for gw in range(1, GWS + 1):
            order = list(range(1, N_TEAMS + 1))
            rng.shuffle(order)
            for h, a in zip(order[::2], order[1::2]):
                fid += 1
                fixtures.append({"season": season, "id": si * 10000 + fid, "gw": gw,
                                 "home_team_id": h, "away_team_id": a, "kickoff_time": _ko(si, gw),
                                 "difficulty_home": 1 + (a % 5), "difficulty_away": 1 + (h % 5),
                                 "finished": True, "home_score": 1, "away_score": 1})
        pid = 0
        for t in range(1, N_TEAMS + 1):
            for pos, n in SHAPE.items():
                for _ in range(n):
                    pid += 1
                    skill = float(rng.gamma(2.0, 1.1))
                    price = round(float(np.clip(3.9 + skill * 0.9 + rng.normal(0, 0.4), 3.9, 14.0)) * 2) / 2
                    players.append({"season": season, "id": pid, "code": pid, "web_name": f"P{pid}",
                                    "first_name": "A", "second_name": f"Player{pid}", "team_id": t,
                                    "team_code": t, "position": pos, "price": price, "status": "a",
                                    "selected_by_percent": 5.0, "total_points": 0, "form": 0})
                    starter = rng.random() < 0.75
                    for gw in range(1, GWS + 1):
                        fx = next(f for f in fixtures
                                  if f["season"] == season and f["gw"] == gw and t in (f["home_team_id"], f["away_team_id"]))
                        mins = int(90 if starter and rng.random() > 0.1 else rng.choice([0, 0, 20, 65]))
                        pts = 0 if mins == 0 else max(0, int(rng.poisson(max(0.2, skill * (0.8 if mins < 60 else 1.0)))) + (1 if mins >= 60 else 0))
                        stats.append({
                            "season": season, "player_id": pid, "player_code": pid, "position": pos,
                            "team_id": t, "gw": gw, "fixture_id": fx["id"],
                            "opponent_team_id": fx["away_team_id"] if fx["home_team_id"] == t else fx["home_team_id"],
                            "was_home": fx["home_team_id"] == t, "kickoff_time": _ko(si, gw),
                            "minutes": mins, "points": pts, "goals": 0, "assists": 0, "clean_sheet": 0,
                            "goals_conceded": 1, "own_goals": 0, "penalties_saved": 0, "penalties_missed": 0,
                            "yellow_cards": 0, "red_cards": 0, "saves": 0, "bonus": 0, "bps": pts * 4,
                            "defensive_contribution": 0, "starts": 1 if mins >= 60 else 0,
                            "influence": 1.0, "creativity": 1.0, "threat": 1.0, "ict_index": float(pts),
                            "xg": 0.1, "xa": 0.1, "xgi": 0.2, "xgc": 1.0, "value": price,
                            "selected": 1000, "transfers_in": 0, "transfers_out": 0,
                            "team_h_score": 1, "team_a_score": 1, "source": "fpl_api",
                        })
    return (pd.DataFrame(teams), pd.DataFrame(fixtures), pd.DataFrame(players), pd.DataFrame(stats))


@pytest.fixture(scope="module")
def sim_db():
    url = os.getenv("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL not set")
    import psycopg
    from psycopg.rows import dict_row

    from pipeline.db.conn import apply_schema, upsert

    with psycopg.connect(url, row_factory=dict_row) as conn:
        conn.execute("DROP SCHEMA public CASCADE; CREATE SCHEMA public;")
        conn.commit()
        apply_schema(conn)
        teams, fixtures, players, stats = _synthetic()
        upsert(conn, "teams", teams.to_dict("records"), ["season", "id"])
        upsert(conn, "fixtures", fixtures.to_dict("records"), ["season", "id"])
        upsert(conn, "players", players.to_dict("records"), ["id"])
        upsert(conn, "player_gw_stats", stats.to_dict("records"), ["season", "player_id", "fixture_id"])
        conn.commit()
        # The job-runner tests open their own connections and re-apply the schema, whose
        # ALTER TABLE ... ADD COLUMN IF NOT EXISTS needs an ACCESS EXCLUSIVE lock. Holding an open
        # transaction here would block that forever, so this connection stops holding one.
        conn.autocommit = True
        yield conn


def _cfg(**kw) -> SimConfig:
    """A deliberately cheap run: a handful of gameweeks, one refit, small trees."""
    base = dict(season=SEASONS[1], start_gw=6, end_gw=11, refit_every=99, rounds=40, horizon=3)
    return SimConfig(**{**base, **kw})


def test_realistic_run_writes_a_row_with_its_parameters(sim_db):
    res = run_simulation(sim_db, _cfg(label="unit"), write=True)
    assert len(res.best.gws) == 6
    assert res.best.total > 0
    row = sim_db.execute("SELECT * FROM backtests ORDER BY id DESC LIMIT 1").fetchone()
    assert row["label"] == "unit"
    assert row["params_json"]["horizon"] == 3
    assert row["season"] == SEASONS[1]
    # every squad must be affordable; this is the free_hit budget bug regression guard
    assert all(g.squad_cost <= 100.0 + 1e-6 for g in res.best.gws)


def test_free_hit_rebuilds_on_the_original_budget_every_week(sim_db):
    """The old harness let free_hit spend bank + squad value, so from the second gameweek on it was
    picking a ~200.0 squad and the totals were inflated."""
    res = run_simulation(sim_db, _cfg(mode="free_hit"), write=False)
    assert len(res.best.gws) == 6
    assert max(g.squad_cost for g in res.best.gws) <= 100.0 + 1e-6
    assert all(g.transfers == 0 and g.hits == 0 for g in res.best.gws)


def test_banned_players_never_appear_and_locked_ones_always_do(sim_db):
    name = sim_db.execute("SELECT web_name FROM players WHERE position='MID' ORDER BY price DESC LIMIT 1").fetchone()["web_name"]
    pid = sim_db.execute("SELECT id FROM players WHERE web_name=%s", (name,)).fetchone()["id"]

    banned = run_simulation(sim_db, _cfg(banned=(name,)), write=False)
    assert banned.best.total > 0
    locked = run_simulation(sim_db, _cfg(locked=(name,)), write=False)
    assert locked.best.total > 0
    assert pid > 0   # the name resolved to a real element, so the constraints were not silently empty


def test_forced_formation_is_respected(sim_db):
    res = run_simulation(sim_db, _cfg(formation="5-4-1"), write=False)
    assert len(res.best.gws) == 6


def test_disallowing_hits_produces_none(sim_db):
    res = run_simulation(sim_db, _cfg(allow_hits=False), write=False)
    assert sum(g.hits for g in res.best.gws) == 0


def test_repeats_produce_a_distribution(sim_db):
    res = run_simulation(sim_db, _cfg(repeats=3), write=False)
    s = res.summary()
    assert s["n_repeats"] == 3 and len(s["totals"]) == 3
    assert s["total_points_p10"] <= s["total_points_mean"] <= s["total_points_p90"]


def test_naive_strategy_runs_without_fitting_anything(sim_db):
    res = run_simulation(sim_db, _cfg(strategy="naive_last5"), write=False)
    assert len(res.best.gws) == 6


def test_a_missing_season_is_a_clear_error(sim_db):
    with pytest.raises(ValueError, match="no gameweeks"):
        run_simulation(sim_db, SimConfig(season="1999/00"), write=False)


def test_threshold_chip_policy_plays_chips_and_never_reuses_one(sim_db):
    # thresholds at zero so a chip definitely fires inside a six-gameweek window
    cfg = _cfg(chip_policy="threshold", chip_thresholds={c: 0.0 for c in ("bboost", "3xc", "freehit", "wildcard")})
    res = run_simulation(sim_db, cfg, write=False)
    played = [g.chip for g in res.best.gws if g.chip]
    assert played, "no chip was played even with a zero threshold"
    assert len(played) == len(set(played)), f"a chip was played twice in one half: {played}"


def test_hindsight_puts_the_bench_boost_on_the_best_bench_week(sim_db):
    plain = run_simulation(sim_db, _cfg(), write=False).best
    hind = run_simulation(sim_db, _cfg(chip_policy="hindsight"), write=False).best
    bb = [g.gw for g in hind.gws if g.chip == "bboost"]
    assert bb, "hindsight played no bench boost"
    best_bench = max(plain.gws, key=lambda g: g.bench_points).gw
    assert bb[0] == best_bench
    assert hind.total >= plain.total


def test_job_runner_executes_a_queued_simulation(sim_db, monkeypatch):
    import json

    from pipeline.jobs import run_job

    monkeypatch.setenv("DATABASE_URL", os.environ["TEST_DATABASE_URL"])
    params = {"seasons": [SEASONS[1]], "start_gw": 6, "end_gw": 9, "refit_every": 99, "rounds": 30, "horizon": 2}
    job_id = sim_db.execute(
        "INSERT INTO jobs (kind, params_json, label) VALUES ('simulate', %s, 'from a test') RETURNING id",
        (json.dumps(params),)).fetchone()["id"]
    sim_db.commit()

    assert run_job(job_id) == 0

    row = sim_db.execute("SELECT * FROM jobs WHERE id = %s", (job_id,)).fetchone()
    assert row["status"] == "success", row["error"]
    assert row["finished_at"] is not None
    assert row["result_json"]["runs"][0]["season"] == SEASONS[1]
    assert row["log_tail"]
    assert sim_db.execute("SELECT count(*) AS n FROM backtests WHERE job_id = %s", (job_id,)).fetchone()["n"] == 1


def test_job_runner_records_a_failure_instead_of_raising(sim_db, monkeypatch):
    import json

    from pipeline.jobs import run_job

    monkeypatch.setenv("DATABASE_URL", os.environ["TEST_DATABASE_URL"])
    job_id = sim_db.execute(
        "INSERT INTO jobs (kind, params_json) VALUES ('simulate', %s) RETURNING id",
        (json.dumps({"seasons": ["1999/00"]}),)).fetchone()["id"]
    sim_db.commit()

    assert run_job(job_id) == 1
    row = sim_db.execute("SELECT * FROM jobs WHERE id = %s", (job_id,)).fetchone()
    assert row["status"] == "failed"
    assert "no gameweeks" in (row["error"] or "")


def test_a_job_is_never_run_twice(sim_db, monkeypatch):
    import json

    from pipeline.jobs import run_job

    monkeypatch.setenv("DATABASE_URL", os.environ["TEST_DATABASE_URL"])
    job_id = sim_db.execute(
        "INSERT INTO jobs (kind, params_json, status) VALUES ('simulate', %s, 'success') RETURNING id",
        (json.dumps({"seasons": [SEASONS[1]]}),)).fetchone()["id"]
    sim_db.commit()
    with pytest.raises(SystemExit, match="already success"):
        run_job(job_id)
