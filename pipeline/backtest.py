"""Backtest harness (spec section 6): replay a past season gameweek by gameweek.

At each gameweek T the models are (re)fit ONLY on rows from earlier seasons and from gameweeks
< T of the replayed season, predictions are made for T from features that use only gameweeks
< T (the training frame already guarantees that), a squad is chosen with the optimiser, and the
squad is scored with the ACTUAL points of T (captain doubled, vice if the captain didn't play,
simple auto-subs). Two modes:

  free_hit   best legal 15 + XI from scratch every week (unlimited transfers): what the
             predictions are worth when the squad is unconstrained.
  realistic  pick a squad at the start, then 1 free transfer per week (banked up to 5, hits
             allowed if the solver thinks they pay), continuous budget with buy = sell = current
             price (the 50% sell-on fee is ignored in v1).

Compared against: a naive "last-5 mean" manager using the same optimiser, plus the user's real
season total when the FPL API knows it (entry/{id}/history/ -> past).
Known optimism/pessimism: historical injury flags are unavailable, so the model can pick an
injured player (pessimistic for the model); the free_hit mode ignores transfer friction (optimistic).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import psycopg

from pipeline.config import MAX_FREE_TRANSFERS, XI_MIN
from pipeline.features.build_features import HORIZON_WEIGHTS, build_prediction_frame, build_training_frame
from pipeline.models.predict import load_tables
from pipeline.models.train import MODEL_VERSION, POSITIONS, PositionModel, train_all

log = logging.getLogger(__name__)
START_BUDGET = 100.0
# Predicted gains are noisy and the solver picks the *largest* apparent gain (winner's curse), so a
# -4 hit is only worth taking when the predicted edge is clearly bigger than 4. We charge hits at
# 4 * HIT_RISK_PREMIUM inside the objective (the real -4 is still what's scored) and cap paid
# transfers per gameweek. Measured on 2025/26: premium 1.5 -> 23 hits, 1924 pts; premium 3 -> 3 hits, 1976 pts.
HIT_RISK_PREMIUM = float(__import__("os").getenv("VANTAGE_HIT_RISK_PREMIUM", "3.0"))
MAX_TRANSFERS_PER_GW = 3


@dataclass
class GWResult:
    gw: int
    points: float
    captain_points: float
    transfers: int
    hits: int
    bench_points: float
    predicted_xi: float


@dataclass
class BacktestResult:
    season: str
    mode: str
    strategy: str
    gws: list[GWResult] = field(default_factory=list)

    @property
    def total(self) -> float:
        return float(sum(g.points for g in self.gws))

    def summary(self) -> dict:
        return {"season": self.season, "mode": self.mode, "strategy": self.strategy, "n_gws": len(self.gws),
                "total_points": self.total, "hits_total": int(sum(g.hits for g in self.gws)),
                "transfers_total": int(sum(g.transfers for g in self.gws)),
                "avg_per_gw": round(self.total / max(len(self.gws), 1), 2),
                "captain_points": float(sum(g.captain_points for g in self.gws)),
                "bench_points": float(sum(g.bench_points for g in self.gws)),
                "per_gw": [g.__dict__ for g in self.gws]}


# ------------------------------------------------------------------------------------------
# scoring with FPL rules
# ------------------------------------------------------------------------------------------

def score_gameweek(lineup: list[int], bench: list[int], captain: int, vice: int,
                   actual: dict[int, float], minutes: dict[int, float], pos_of: dict[int, str]) -> tuple[float, float, float]:
    """Returns (points, captain_bonus, bench_points_unused). Auto-subs: a starter with 0 minutes is
    replaced by the first bench player (in order) that keeps the formation legal."""
    xi = list(lineup)
    played = lambda i: minutes.get(i, 0) > 0  # noqa: E731
    bench_left = list(bench)
    for starter in list(xi):
        if played(starter):
            continue
        for b in list(bench_left):
            if not played(b):
                continue
            trial = [i for i in xi if i != starter] + [b]
            counts = {p: sum(1 for i in trial if pos_of.get(i) == p) for p in XI_MIN}
            if counts["GKP"] == 1 and all(counts[p] >= XI_MIN[p] for p in XI_MIN):
                xi = trial
                bench_left.remove(b)
                break
    pts = sum(actual.get(i, 0.0) for i in xi)
    cap = captain if played(captain) else (vice if played(vice) else None)
    cap_bonus = actual.get(cap, 0.0) if cap is not None else 0.0
    bench_unused = sum(actual.get(i, 0.0) for i in bench_left)
    return pts + cap_bonus, cap_bonus, bench_unused


# ------------------------------------------------------------------------------------------
# replay
# ------------------------------------------------------------------------------------------

def _predict_rows(models: dict[str, PositionModel], rows: pd.DataFrame) -> pd.Series:
    out = pd.Series(np.nan, index=rows.index)
    for pos, m in models.items():
        mask = rows["position"] == pos
        if mask.any():
            out[mask] = m.predict(rows[mask])[0]
    return out.clip(lower=0)


def _future_value(models: dict[str, PositionModel], stats: pd.DataFrame, fixtures: pd.DataFrame, teams: pd.DataFrame,
                  season: str, as_of: int, horizon: int) -> pd.Series:
    """Horizon-weighted predicted points for gameweeks as_of+2 .. as_of+horizon (as_of+1 is the XI
    value handled separately), using only data up to as_of. Indexed by player_id."""
    if horizon < 2:
        return pd.Series(dtype=float)
    known = stats[(stats["season"] == season) & (stats["gw"] <= as_of)]
    last = known.sort_values("gw").groupby("player_id").tail(1)
    players = pd.DataFrame({
        "id": last["player_id"], "code": last["player_code"], "position": last["position"], "team_id": last["team_id"],
        "price": pd.to_numeric(last["value"], errors="coerce"), "cost_change_start": np.nan, "status": "a",
        "chance_of_playing_next_round": np.nan, "web_name": "",
    })
    targets = list(range(as_of + 2, as_of + 1 + horizon))
    pf = build_prediction_frame(stats, fixtures, teams, players, season, as_of, targets)
    rows = pf.rows
    pred = _predict_rows(models, rows)
    w = (rows["gw"] - as_of).map(HORIZON_WEIGHTS).fillna(0.3)
    pred = pred.where(rows["n_fixtures"] > 0, 0.0)
    return (pred * w).groupby(rows["player_id"]).sum()


def run_backtest(conn: psycopg.Connection, season: str, mode: str = "realistic", strategy: str = "model",
                 start_gw: int = 2, end_gw: int = 38, refit_every: int = 4, rounds: int = 300,
                 horizon: int = 5, write: bool = True) -> BacktestResult:
    t = load_tables(conn)
    ff = build_training_frame(t["stats"], t["fixtures"], t["teams"])
    frame, feature_cols = ff.rows[ff.rows["season"] <= season], ff.feature_cols
    prev_season = max([s for s in t["stats"]["season"].unique() if s < season], default=None)
    stats_h = t["stats"][t["stats"]["season"].isin([season, prev_season])]
    fixtures_h = t["fixtures"][t["fixtures"]["season"].isin([season, prev_season])]
    res = BacktestResult(season, mode, strategy)
    from pipeline.optimize.solver import solve_squad

    models: dict[str, PositionModel] | None = None
    squad: list[int] | None = None
    bank = START_BUDGET
    free_transfers = 1
    last_fit = None
    gws = sorted(frame[frame["season"] == season]["gw"].unique())
    gws = [g for g in gws if start_gw <= g <= end_gw]
    for T in gws:
        rows = frame[(frame["season"] == season) & (frame["gw"] == T)].copy()
        if rows.empty:
            continue
        if strategy == "model":
            if models is None or last_fit is None or T - last_fit >= refit_every:
                train_rows = frame[(frame["season"] < season) | (frame["gw"] < T)]
                models = train_all(train_rows, feature_cols, POSITIONS, {p: rounds for p in POSITIONS}, with_quantiles=False)
                last_fit = T
                log.info("backtest %s GW%d: refit on %d rows", season, T, len(train_rows))
            rows["pred"] = _predict_rows(models, rows)
        else:  # naive: last-5 mean scaled by fixtures
            rows["pred"] = rows["points_l5"].fillna(0) * rows["n_fixtures"].clip(lower=1) * rows["played_l5"].fillna(1)
        rows["xi_value"] = rows["pred"]
        if mode == "realistic" and strategy == "model" and horizon > 1:
            fut = _future_value(models, stats_h, fixtures_h, t["teams"], season, T - 1, horizon)
            rows["squad_value"] = rows["player_id"].map(fut).fillna(0.0)
        elif mode == "realistic":
            # naive manager: assume this week's estimate persists over the horizon
            rows["squad_value"] = rows["pred"] * sum(w for k, w in HORIZON_WEIGHTS.items() if 2 <= k <= horizon)
        else:
            rows["squad_value"] = 0.0
        bench_w = 1.0 if mode == "realistic" else 0.05

        actual = dict(zip(rows["player_id"], rows["target"].astype(float)))
        minutes = dict(zip(rows["player_id"], rows["target_minutes"]))  # minutes actually played in gw T
        pos_of = dict(zip(rows["player_id"], rows["position"]))
        price = dict(zip(rows["player_id"], rows["price"].astype(float)))

        if mode == "free_hit" or squad is None:
            sol = solve_squad(rows, budget=START_BUDGET if squad is None else bank + sum(price.get(i, 0) for i in squad),
                              bench_weight=bench_w)
            if sol.status != "Optimal":
                log.warning("GW%d: solver status %s", T, sol.status)
                continue
            transfers, hits = 0, 0
            if squad is None and mode == "realistic":
                bank = START_BUDGET - sol.cost
        else:
            # players in the current squad with no row this gw (blank / gone) still hold their slot
            missing = [i for i in squad if i not in set(rows["player_id"])]
            if missing:
                held = frame[(frame["season"] == season) & (frame["player_id"].isin(missing)) & (frame["gw"] < T)] \
                    .sort_values("gw").groupby("player_id").tail(1)
                held = held.assign(pred=0.0, xi_value=0.0, squad_value=0.0, target=0.0, target_minutes=0)
                rows = pd.concat([rows, held[rows.columns.intersection(held.columns)]], ignore_index=True)
                price.update(dict(zip(held["player_id"], held["price"].astype(float))))
                pos_of.update(dict(zip(held["player_id"], held["position"])))
            sell_value = sum(price.get(i, 0.0) for i in squad)
            sol = solve_squad(rows, budget=bank + sell_value, current_squad=squad, free_transfers=free_transfers,
                              max_transfers=max(free_transfers, MAX_TRANSFERS_PER_GW),
                              hit_cost=int(round(4 * HIT_RISK_PREMIUM)), bench_weight=bench_w)
            if sol.status != "Optimal":
                log.warning("GW%d: solver status %s", T, sol.status)
                continue
            transfers, hits = len(sol.transfers_in), sol.hits
            bank = bank + sell_value - sol.cost
            free_transfers = min(MAX_FREE_TRANSFERS, max(free_transfers - transfers, 0) + 1)
        squad = sol.squad
        pts, cap_pts, bench_pts = score_gameweek(sol.lineup, sol.bench, sol.captain, sol.vice_captain, actual, minutes, pos_of)
        pts -= 4 * hits
        res.gws.append(GWResult(int(T), float(pts), float(cap_pts), int(transfers), int(hits), float(bench_pts), float(sol.xi_points)))
        log.info("backtest %s %s/%s GW%d: %.0f pts (cap %.0f, %d transfers, %d hits, bank %.1f)",
                 season, mode, strategy, T, pts, cap_pts, transfers, hits, bank)

    if write and res.gws:
        s = res.summary()
        conn.execute(
            "INSERT INTO backtests (season, mode, strategy, model_version, start_gw, end_gw, total_points, per_gw_json, notes) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            (season, mode, strategy, MODEL_VERSION, gws[0], gws[-1], s["total_points"], json.dumps(s["per_gw"]),
             f"hits={s['hits_total']} transfers={s['transfers_total']} refit_every={refit_every} rounds={rounds} horizon={horizon}"),
        )
        conn.commit()
    return res
