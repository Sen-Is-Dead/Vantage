from __future__ import annotations

import pandas as pd
import pytest

from pipeline.optimize.recommend import build_pool, protected_ids, risk_appetite


def test_risk_appetite_maps_rank_percentile():
    assert risk_appetite(None, None) == 0.0
    assert risk_appetite(100_000, 11_000_000) == -1.0          # top 1%: protect
    assert risk_appetite(6_000_000, 11_000_000) == 1.0         # bottom half: chase
    mid = risk_appetite(0.275 * 11_000_000, 11_000_000)        # midpoint of the ramp
    assert mid == pytest.approx(0.0, abs=1e-9)


def test_protected_player_lock_and_injury_override():
    players = pd.DataFrame([
        {"id": 1, "web_name": "Mbeumo", "second_name": "Mbeumo", "status": "a", "news": None},
        {"id": 2, "web_name": "Saka", "second_name": "Saka", "status": "a", "news": None},
        {"id": 3, "web_name": "B.Mbeumo", "second_name": "Mbeumo", "status": "i", "news": "Knee injury"},
    ])
    ids, reasons = protected_ids(players)
    assert ids == {1}                          # 3 matches the name but is injured -> lock lifted
    assert "locked" in reasons[1] and "lock is lifted" in reasons[3] and 2 not in reasons
    # 'd' (doubtful) does NOT lift the lock
    players.loc[players.id == 1, "status"] = "d"
    ids, _ = protected_ids(players)
    assert ids == {1}


def test_build_pool_horizon_and_captain_shading():
    pred = pd.DataFrame([
        {"player_id": 1, "gw": g, "predicted_points": 4.0, "predicted_points_low": 2.0, "predicted_points_high": 8.0, "as_of_gw": 4}
        for g in (5, 6, 7, 8, 9)
    ] + [
        {"player_id": 2, "gw": g, "predicted_points": 4.5, "predicted_points_low": 3.5, "predicted_points_high": 5.5, "as_of_gw": 4}
        for g in (5, 6, 7, 8, 9)
    ])
    players = pd.DataFrame([
        {"id": 1, "web_name": "Ceiling", "position": "MID", "team_id": 1, "price": 8.0, "status": "a", "chance_of_playing_next_round": None, "news": None, "selected_by_percent": 5.0},
        {"id": 2, "web_name": "Floor", "position": "MID", "team_id": 2, "price": 8.0, "status": "a", "chance_of_playing_next_round": None, "news": None, "selected_by_percent": 50.0},
        {"id": 3, "web_name": "Crocked", "position": "FWD", "team_id": 3, "price": 6.0, "status": "i", "chance_of_playing_next_round": 0, "news": "out", "selected_by_percent": 1.0},
    ])
    teams = pd.DataFrame([{"id": 1, "short_name": "AAA"}, {"id": 2, "short_name": "BBB"}, {"id": 3, "short_name": "CCC"}])
    inp = {"pred": pred, "players": players, "teams": teams}
    chase = build_pool(inp, next_gw=5, horizon=5, r=1.0).set_index("player_id")
    protect = build_pool(inp, next_gw=5, horizon=5, r=-1.0).set_index("player_id")
    # horizon value = sum of weights for GW+2..+5 (0.7+0.5+0.4+0.3 = 1.9) x points
    assert chase.loc[1, "squad_value"] == pytest.approx(4.0 * 1.9) and chase.loc[1, "xi_value"] == 4.0
    # chasing rank prefers the high-ceiling captain; protecting rank prefers the high-floor one
    assert chase.loc[1, "captain_value"] > chase.loc[2, "captain_value"]
    assert protect.loc[2, "captain_value"] > protect.loc[1, "captain_value"]
    assert not chase.loc[3, "available"] and chase.loc[1, "available"]
    assert chase.loc[1, "team"] == "AAA"
