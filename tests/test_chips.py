from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.optimize.chips import THRESHOLDS, chip_availability, evaluate_chips, pick_chip_for_now
from pipeline.optimize.solver import solve_squad

WINDOWS = pd.DataFrame([{"season": "2026/27", "name": n, "start_event": a, "stop_event": b, "number": 1}
                        for n in ("wildcard", "freehit", "bboost", "3xc") for a, b in ((2, 19), (20, 38))])


def test_chip_availability_windows_and_usage():
    used = [{"name": "wildcard", "event": 8}, {"name": "bboost", "event": 25}]
    a = chip_availability(WINDOWS, used, gw=10)
    assert a["wildcard"][0] is False and "GW8" in a["wildcard"][1]
    assert a["bboost"][0] is True                      # used in the second half only
    assert a["3xc"][0] is True and a["freehit"][0] is True
    b = chip_availability(WINDOWS, used, gw=25)
    assert b["wildcard"][0] is True and b["bboost"][0] is False
    assert chip_availability(WINDOWS, [], gw=1)["wildcard"][0] is False   # no wildcard window at GW1


def _pool(seed=1):
    rng = np.random.default_rng(seed)
    rows, pid = [], 1
    for pos, n in (("GKP", 8), ("DEF", 20), ("MID", 20), ("FWD", 12)):
        for _ in range(n):
            rows.append({"id": pid, "web_name": f"P{pid}", "position": pos, "team_id": int(rng.integers(1, 9)),
                         "price": float(rng.choice([4.0, 4.5, 5.0, 5.5, 6.5, 7.5, 9.0, 12.5])), "base": float(rng.uniform(1, 6))})
            pid += 1
    players = pd.DataFrame(rows)
    pred = pd.DataFrame([{"player_id": r.id, "gw": g, "predicted_points": r.base * (1.6 if g == 7 else 1.0)}
                         for r in players.itertuples() for g in (5, 6, 7, 8, 9)])
    return players, pred


def test_evaluate_chips_prefers_the_double_gameweek():
    players, pred = _pool()
    frame = players.rename(columns={"id": "player_id"}).assign(xi_value=lambda d: d["base"], squad_value=lambda d: d["base"])
    squad = solve_squad(frame, budget=100.0).squad
    evals = evaluate_chips(pred, players, squad, budget=100.0, windows=WINDOWS, used=[], next_gw=5, horizon_gws=[5, 6, 7, 8, 9])
    assert set(evals) == {"bboost", "3xc", "freehit", "wildcard"}
    for chip in ("bboost", "3xc"):                    # GW7 predictions are 1.6x everything else
        assert evals[chip].best_gw == 7 and not evals[chip].recommend_now
    assert evals["3xc"].by_gw[7] > evals["3xc"].by_gw[5] and evals["bboost"].by_gw[7] > evals["bboost"].by_gw[5]
    assert all(v >= 0 for v in evals["bboost"].by_gw.values())
    assert all(evals[c].available for c in evals)
    # already-used chips are neither available nor evaluated in their window
    evals2 = evaluate_chips(pred, players, squad, 100.0, WINDOWS, [{"name": "3xc", "event": 3}], 5, [5, 6, 7, 8, 9])
    assert not evals2["3xc"].available and evals2["3xc"].by_gw == {}
    assert pick_chip_for_now(evals) is None


def test_pick_chip_for_now_uses_threshold_margin():
    from pipeline.optimize.chips import ChipEval
    a = ChipEval("bboost", True, "", {5: 20.0}, 5, 20.0, THRESHOLDS["bboost"], True)
    b = ChipEval("3xc", True, "", {5: 9.0}, 5, 9.0, THRESHOLDS["3xc"], True)
    assert pick_chip_for_now({"bboost": a, "3xc": b}) == "bboost"   # 20-12=8 beats 9-7=2
