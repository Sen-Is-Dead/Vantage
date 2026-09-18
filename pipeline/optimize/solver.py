"""FPL squad / transfer optimiser (PuLP + CBC). Spec section 5.

Decision variables per player i:
  s_i  in squad (15)      l_i  in starting XI (11, l <= s)      c_i captain (1, c <= l)
Constraints: 2 GKP / 5 DEF / 5 MID / 3 FWD in the squad; XI has 1 GKP, 3-5 DEF, 2-5 MID, 1-3 FWD;
max 3 per club; budget; and, when a current squad is given, transfer accounting:
  n_in = number of squad players not in the current squad; hits h >= n_in - free_transfers, h >= 0.
Objective: sum(xi_value * l) + sum(xi_value * c)  [captain counts double]
         + BENCH_WEIGHT * sum(squad_value * s)     [bench/holding value over the horizon]
         - hit_cost * h

Nothing here talks to FPL. It returns a recommendation; a human confirms (Safety Rule 1).
`locked` players cannot be transferred out (Safety Rule 2 plumbing; the status check lives in
pipeline/optimize/recommend.py so the solver stays a pure function).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import pulp

from pipeline.config import MAX_PER_TEAM, SQUAD_COMPOSITION, TRANSFER_HIT_COST, XI_MAX, XI_MIN

BENCH_WEIGHT = 0.10   # how much a squad (non-XI) slot's horizon value counts vs. XI points this gw
VICE_WEIGHT = 0.02    # tiny tie-break so the vice is the 2nd best captain option


@dataclass
class Solution:
    squad: list[int]
    lineup: list[int]
    captain: int
    vice_captain: int
    bench: list[int]                 # in bench order (GK first, then by value desc)
    transfers_in: list[int] = field(default_factory=list)
    transfers_out: list[int] = field(default_factory=list)
    hits: int = 0
    objective: float = 0.0
    xi_points: float = 0.0           # predicted XI points incl. captain double
    cost: float = 0.0
    status: str = "unknown"


def solve_squad(players: pd.DataFrame, budget: float, current_squad: list[int] | None = None,
                free_transfers: int = 1, max_transfers: int | None = None, hit_cost: int = TRANSFER_HIT_COST,
                locked: set[int] | None = None, banned: set[int] | None = None,
                xi_col: str = "xi_value", squad_col: str = "squad_value",
                captain_col: str | None = None, triple_captain: bool = False,
                bench_weight: float = BENCH_WEIGHT, time_limit: int = 60) -> Solution:
    """`players` needs columns: player_id, position, team_id, price, xi_col, squad_col.
    `budget` is the total spend allowed (bank + selling value of the current squad)."""
    p = players.drop_duplicates("player_id").reset_index(drop=True)
    p = p[p["position"].isin(SQUAD_COMPOSITION)]
    ids = p["player_id"].tolist()
    idx = {pid: i for i, pid in enumerate(ids)}
    cur = set(current_squad or [])
    locked = set(locked or [])
    banned = set(banned or [])
    cap_col = captain_col or xi_col
    cap_mult = 2.0 if triple_captain else 1.0   # extra multiple over the XI value (2x -> +1, 3x -> +2)

    prob = pulp.LpProblem("fpl", pulp.LpMaximize)
    s = pulp.LpVariable.dicts("s", ids, cat="Binary")
    l = pulp.LpVariable.dicts("l", ids, cat="Binary")
    c = pulp.LpVariable.dicts("c", ids, cat="Binary")
    v = pulp.LpVariable.dicts("v", ids, cat="Binary")
    h = pulp.LpVariable("hits", lowBound=0, cat="Integer")

    xi = dict(zip(ids, p[xi_col].fillna(0).astype(float)))
    sq = dict(zip(ids, p[squad_col].fillna(0).astype(float)))
    cp = dict(zip(ids, p[cap_col].fillna(0).astype(float)))
    price = dict(zip(ids, p["price"].astype(float)))

    prob += (pulp.lpSum(xi[i] * l[i] for i in ids) + cap_mult * pulp.lpSum(cp[i] * c[i] for i in ids)
             + VICE_WEIGHT * pulp.lpSum(cp[i] * v[i] for i in ids)
             + bench_weight * pulp.lpSum(sq[i] * s[i] for i in ids) - hit_cost * h)

    pos_of = dict(zip(ids, p["position"]))
    team_of = dict(zip(ids, p["team_id"]))
    for pos, n in SQUAD_COMPOSITION.items():
        prob += pulp.lpSum(s[i] for i in ids if pos_of[i] == pos) == n
        prob += pulp.lpSum(l[i] for i in ids if pos_of[i] == pos) >= XI_MIN[pos]
        prob += pulp.lpSum(l[i] for i in ids if pos_of[i] == pos) <= XI_MAX[pos]
    prob += pulp.lpSum(l[i] for i in ids) == 11
    prob += pulp.lpSum(c[i] for i in ids) == 1
    prob += pulp.lpSum(v[i] for i in ids) == 1
    for i in ids:
        prob += l[i] <= s[i]
        prob += c[i] <= l[i]
        prob += v[i] <= l[i]
        prob += c[i] + v[i] <= 1
        if i in banned:
            prob += s[i] == 0
        if i in locked and i in idx:
            prob += s[i] == 1
    for t in set(team_of.values()):
        prob += pulp.lpSum(s[i] for i in ids if team_of[i] == t) <= MAX_PER_TEAM
    prob += pulp.lpSum(price[i] * s[i] for i in ids) <= budget + 1e-6

    if current_squad is not None:
        n_in = pulp.lpSum(s[i] for i in ids if i not in cur)
        prob += h >= n_in - free_transfers
        if max_transfers is not None:
            prob += n_in <= max_transfers
    else:
        prob += h == 0

    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit)
    prob.solve(solver)
    status = pulp.LpStatus[prob.status]
    if status not in ("Optimal", "Not Solved"):
        return Solution([], [], -1, -1, [], status=status)

    squad = [i for i in ids if s[i].value() > 0.5]
    lineup = [i for i in ids if l[i].value() > 0.5]
    captain = next(i for i in ids if c[i].value() > 0.5)
    vice = next(i for i in ids if v[i].value() > 0.5)
    bench_ids = [i for i in squad if i not in lineup]
    bench = sorted(bench_ids, key=lambda i: (pos_of[i] != "GKP", -xi[i]))  # GK first, then best-first
    t_in = sorted(i for i in squad if i not in cur) if current_squad is not None else []
    t_out = sorted(i for i in cur if i not in squad) if current_squad is not None else []
    xi_pts = sum(xi[i] for i in lineup) + cp[captain] * (cap_mult)
    return Solution(squad=squad, lineup=lineup, captain=captain, vice_captain=vice, bench=bench,
                    transfers_in=t_in, transfers_out=t_out, hits=int(round(h.value() or 0)),
                    objective=float(pulp.value(prob.objective)), xi_points=float(xi_pts),
                    cost=float(sum(price[i] for i in squad)), status=status)
