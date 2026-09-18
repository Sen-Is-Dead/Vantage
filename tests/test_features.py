from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from pipeline.features.build_features import (
    aggregate_per_gw,
    build_prediction_frame,
    build_training_frame,
    rolling_state,
    team_fixture_features,
)
from pipeline.models.predict import availability_multiplier


def _ko(gw: int) -> str:
    return (datetime(2025, 8, 10, 14, tzinfo=timezone.utc) + timedelta(days=7 * gw)).strftime("%Y-%m-%dT%H:%M:%SZ")


def _stats(season="2025/26", pid=1, code=100, team=1, points=(2, 6, 1, 9), minutes=(90, 90, 60, 90)):
    rows = []
    for i, (p, m) in enumerate(zip(points, minutes), start=1):
        rows.append({"season": season, "player_id": pid, "player_code": code, "position": "MID", "team_id": team,
                     "gw": i, "fixture_id": i * 10 + pid, "opponent_team_id": 2, "was_home": i % 2 == 1,
                     "kickoff_time": _ko(i), "minutes": m, "points": p,
                     "xg": 0.3, "xa": 0.1, "xgi": 0.4, "bps": 20, "ict_index": 5.0, "bonus": 0, "goals": 0,
                     "assists": 0, "clean_sheet": 0, "goals_conceded": 1, "xgc": 1.0, "saves": 0,
                     "defensive_contribution": 0, "starts": 1, "value": 5.0})
    return pd.DataFrame(rows)


def _fixtures(season="2025/26", gws=6):
    rows = []
    for g in range(1, gws + 1):
        home, away = (1, 2) if g % 2 == 1 else (2, 1)
        done = g <= 4
        rows.append({"season": season, "id": g * 10, "gw": g, "home_team_id": home, "away_team_id": away,
                     "kickoff_time": _ko(g), "difficulty_home": 2, "difficulty_away": 4,
                     "finished": done, "home_score": 2 if done else None, "away_score": 1 if done else None})
    return pd.DataFrame(rows)


def _teams(season="2025/26"):
    return pd.DataFrame([
        {"season": season, "id": 1, "strength_attack_home": 1300, "strength_attack_away": 1250,
         "strength_defence_home": 1280, "strength_defence_away": 1230, "strength_overall_home": 1290, "strength_overall_away": 1240},
        {"season": season, "id": 2, "strength_attack_home": 1100, "strength_attack_away": 1050,
         "strength_defence_home": 1080, "strength_defence_away": 1030, "strength_overall_home": 1090, "strength_overall_away": 1040},
    ])


def test_rolling_state_windows_and_ewm():
    st = rolling_state(aggregate_per_gw(_stats()))
    g3 = st[st.gw == 3].iloc[0]
    assert g3["points_l3"] == pytest.approx((2 + 6 + 1) / 3)
    assert g3["points_l5"] == pytest.approx((2 + 6 + 1) / 3)  # min_periods=1
    assert g3["gws_played_season"] == 3
    # EWM with half-life 2 weights the latest gw most: after [2,6,1] the ewm sits below the plain mean
    assert g3["points_ewm"] < g3["points_l3"]


def test_training_rows_use_only_prior_gameweeks():
    ff = build_training_frame(_stats(), _fixtures(), _teams())
    r = ff.rows.set_index("gw")
    assert list(r.index) == [2, 3, 4]  # gw1 has no prior form and is dropped
    assert r.loc[2, "target"] == 6 and r.loc[2, "points_l3"] == pytest.approx(2.0)      # only gw1 known
    assert r.loc[4, "points_l3"] == pytest.approx((2 + 6 + 1) / 3)                      # gw1..3, never gw4 itself
    assert r.loc[4, "is_home"] == 0.0 and r.loc[4, "fdr"] == 4                          # gw4: team 1 away
    assert r.loc[4, "opp_attack"] == 1100 and r.loc[4, "own_attack"] == 1250  # opponent (team 2) is at home in gw4
    assert r.loc[4, "days_since_last_match"] == pytest.approx(7.0)
    assert r.loc[4, "team_gf_l5"] == pytest.approx((2 + 1 + 2) / 3)  # team 1 scored 2 (h), 1 (a), 2 (h) before gw4
    assert set(ff.feature_cols) <= set(ff.rows.columns)


def test_prediction_frame_for_future_gameweeks():
    players = pd.DataFrame([{"id": 1, "code": 100, "position": "MID", "team_id": 1, "price": 5.2, "cost_change_start": 0.2,
                             "status": "a", "chance_of_playing_next_round": None, "web_name": "Test"}])
    pf = build_prediction_frame(_stats(), _fixtures(), _teams(), players, "2025/26", as_of_gw=4, target_gws=[5, 6])
    r = pf.rows.set_index("gw")
    assert list(r.index) == [5, 6]
    assert r.loc[5, "points_l3"] == pytest.approx((6 + 1 + 9) / 3) and r.loc[6, "points_l3"] == r.loc[5, "points_l3"]
    assert r.loc[5, "price"] == 5.2 and r.loc[5, "n_fixtures"] == 1 and r.loc[5, "is_home"] == 1.0
    assert r.loc[6, "days_since_last_match"] > r.loc[5, "days_since_last_match"]


def test_double_gameweek_aggregates_and_counts_two_fixtures():
    s = _stats()
    extra = s.iloc[[3]].copy()
    extra["fixture_id"] = 999
    extra["points"] = 4
    per = aggregate_per_gw(pd.concat([s, extra]))
    assert per[per.gw == 4]["points"].iloc[0] == 13 and per[per.gw == 4]["n_fixtures_played_in"].iloc[0] == 2
    fx = _fixtures()
    fx = pd.concat([fx, pd.DataFrame([{**fx.iloc[3].to_dict(), "id": 999}])])
    tf = team_fixture_features(fx, _teams())
    assert tf[(tf.team_id == 1) & (tf.gw == 4)]["n_fixtures"].iloc[0] == 2


def test_availability_multiplier():
    p = pd.DataFrame({"status": ["a", "i", "d", "s", "a"], "chance_of_playing_next_round": [None, None, 75, 0, 100]})
    assert availability_multiplier(p).tolist() == [1.0, 0.0, 0.75, 0.0, 1.0]
