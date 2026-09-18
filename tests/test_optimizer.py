from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.backtest import score_gameweek
from pipeline.config import MAX_PER_TEAM, SQUAD_COMPOSITION
from pipeline.optimize.solver import solve_squad


def _pool(seed=0, n_teams=10):
    """A synthetic player pool: 4 GKP, 12 DEF, 12 MID, 8 FWD per 'league' with varied prices."""
    rng = np.random.default_rng(seed)
    rows, pid = [], 1
    for pos, n in (("GKP", 12), ("DEF", 30), ("MID", 30), ("FWD", 20)):
        for _ in range(n):
            price = float(rng.choice([4.0, 4.5, 5.0, 5.5, 6.5, 7.5, 9.0, 12.5]))
            val = max(0.0, rng.normal(1.0 + price * 0.35, 1.0))
            rows.append({"player_id": pid, "position": pos, "team_id": int(rng.integers(1, n_teams + 1)),
                         "price": price, "xi_value": val, "squad_value": val * 1.5})
            pid += 1
    return pd.DataFrame(rows)


def _legal(p, sol):
    sq = p.set_index("player_id").loc[sol.squad]
    assert len(sol.squad) == 15 and len(sol.lineup) == 11
    assert sq["position"].value_counts().to_dict() == SQUAD_COMPOSITION
    assert sq["team_id"].value_counts().max() <= MAX_PER_TEAM
    xi = p.set_index("player_id").loc[sol.lineup]["position"].value_counts()
    assert xi.get("GKP", 0) == 1 and xi.get("DEF", 0) >= 3 and xi.get("MID", 0) >= 2 and xi.get("FWD", 0) >= 1
    assert sol.captain in sol.lineup and sol.vice_captain in sol.lineup and sol.captain != sol.vice_captain
    assert set(sol.lineup) <= set(sol.squad) and set(sol.bench) == set(sol.squad) - set(sol.lineup)


def test_fresh_squad_respects_all_fpl_rules_and_budget():
    p = _pool()
    sol = solve_squad(p, budget=100.0)
    assert sol.status == "Optimal"
    _legal(p, sol)
    assert sol.cost <= 100.0 + 1e-6
    # captain is the highest-value player in the XI
    best = p.set_index("player_id").loc[sol.lineup]["xi_value"].idxmax()
    assert sol.captain == best


def test_budget_binds():
    p = _pool()
    rich = solve_squad(p, budget=100.0)
    poor = solve_squad(p, budget=70.0)
    assert poor.status == "Optimal" and poor.cost <= 70.0 + 1e-6 and poor.xi_points <= rich.xi_points


def test_transfers_are_counted_and_hits_charged():
    p = _pool()
    base = solve_squad(p, budget=100.0)
    # make one non-squad player suddenly enormous: with 1 free transfer the solver should bring him in
    star = p[~p.player_id.isin(base.squad)].sort_values("price").iloc[0]["player_id"]
    p.loc[p.player_id == star, ["xi_value", "squad_value"]] = 30.0
    sell = p.set_index("player_id").loc[base.squad]["price"].sum()
    sol = solve_squad(p, budget=100.0 - base.cost + sell, current_squad=base.squad, free_transfers=1, max_transfers=1)
    assert sol.status == "Optimal" and star in sol.squad and len(sol.transfers_in) == 1 and sol.hits == 0
    _legal(p, sol)
    # with zero free transfers the same move costs a hit, and a huge hit cost prevents it
    sol2 = solve_squad(p, budget=100.0 - base.cost + sell, current_squad=base.squad, free_transfers=0, max_transfers=1)
    assert star in sol2.squad and sol2.hits == 1
    sol3 = solve_squad(p, budget=100.0 - base.cost + sell, current_squad=base.squad, free_transfers=0, hit_cost=1000)
    assert sol3.transfers_in == [] and sol3.hits == 0


def test_locked_and_banned_players():
    p = _pool()
    base = solve_squad(p, budget=100.0)
    worst = p.set_index("player_id").loc[base.squad]["squad_value"].idxmin()
    p.loc[p.player_id == worst, ["xi_value", "squad_value"]] = 0.0
    sell = p.set_index("player_id").loc[base.squad]["price"].sum()
    kwargs = dict(budget=100.0 - base.cost + sell, current_squad=base.squad, free_transfers=2)
    free = solve_squad(p, **kwargs)
    assert worst in free.transfers_out            # unlocked: the dead weight goes
    locked = solve_squad(p, locked={worst}, **kwargs)
    assert worst in locked.squad                  # protected player stays (Safety Rule 2 plumbing)
    best = p.sort_values("xi_value").iloc[-1]["player_id"]
    banned = solve_squad(p, banned={best}, budget=100.0)
    assert best not in banned.squad


def test_score_gameweek_captain_vice_and_autosub():
    pos = {1: "GKP", 2: "DEF", 3: "DEF", 4: "DEF", 5: "MID", 6: "MID", 7: "MID", 8: "MID", 9: "FWD", 10: "FWD", 11: "FWD",
           12: "GKP", 13: "DEF", 14: "MID", 15: "FWD"}
    lineup, bench = list(range(1, 12)), [12, 13, 14, 15]
    actual = {i: float(i) for i in range(1, 16)}
    minutes = {i: 90 for i in range(1, 16)}
    pts, cap, unused = score_gameweek(lineup, bench, captain=11, vice=10, actual=actual, minutes=minutes, pos_of=pos)
    assert pts == sum(range(1, 12)) + 11 and cap == 11 and unused == 12 + 13 + 14 + 15
    # captain 11 blanks (0 minutes) -> vice doubles and 11 is auto-subbed; starter 5 (MID) blanks too.
    # Bench order 13 (DEF) replaces 5, then 14 (MID) replaces 11; both keep the formation legal.
    minutes[11] = 0
    minutes[5] = 0
    pts, cap, unused = score_gameweek(lineup, bench, captain=11, vice=10, actual=actual, minutes=minutes, pos_of=pos)
    assert cap == 10 and pts == sum(range(1, 12)) - 11 - 5 + 13 + 14 + 10 and unused == 12 + 15
    # a second GK can only replace the GK
    minutes[1] = 0
    pts2, _, _ = score_gameweek(lineup, bench, captain=11, vice=10, actual=actual, minutes=minutes, pos_of=pos)
    assert pts2 == pts - 1 + 12
