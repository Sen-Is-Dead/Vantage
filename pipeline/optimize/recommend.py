"""Weekly recommendation: transfers, starting XI, bench order, captain / vice (spec section 5).

Reads predictions + the user's current squad/bank/free transfers from Postgres, runs the solver,
and writes ONE row to `recommendations` with status='recommended'. Nothing here submits anything
to FPL (Safety Rule 1): a human reads the recommendation (CLI / dashboard) and acts on it.

Objective (per spec): maximise predicted points over the look-ahead, nearer gameweeks weighted
more (GW+1 1.0, +2 0.7, +3 0.5, +4 0.4, +5 0.3), minus transfer hits.
  xi_value    = predicted points next GW (drives the XI + captain)
  squad_value = sum_{k=2..5} w_k * predicted points GW+k (drives who is worth holding / buying)

Safety Rule 2 (protected players): a protected player who is NOT injured/suspended/unavailable per
the FPL API status is LOCKED into the squad. We also solve without the lock; if that solution
would have sold him, the recommendation carries a prominent warning with the points it costs.

Captaincy risk (user input: "let the model decide from rank"): risk appetite r in [-1, 1] from the
overall-rank percentile (top 5% -> -1 protect the rank; outside the top 50% -> +1 chase). The
captain score is the point prediction shaded towards the q80 (r > 0) or q20 (r < 0) quantile.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import psycopg
from psycopg.types.json import Jsonb

from pipeline import config
from pipeline.features.build_features import HORIZON_WEIGHTS
from pipeline.optimize.chips import ChipEval, chip_report, evaluate_chips, pick_chip_for_now
from pipeline.optimize.solver import Solution, solve_squad

log = logging.getLogger(__name__)

ROLL_THRESHOLD = 1.5     # horizon-weighted points a transfer must gain, else recommend banking the FT
HIT_RISK_PREMIUM = 3.0   # see backtest.py: charge hits at 3x inside the objective


@dataclass
class Recommendation:
    season: str
    gw: int
    as_of_gw: int
    horizon_gws: list[int]
    squad: list[dict[str, Any]]
    transfers: list[dict[str, Any]]
    hits: int
    captain: dict[str, Any]
    vice_captain: dict[str, Any]
    expected_points_gw: float
    expected_points_gain: float
    roll_transfer: bool
    free_transfers: int
    bank_after: float
    risk_appetite: float
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    protected_locked: list[str] = field(default_factory=list)
    chips: dict[str, ChipEval] = field(default_factory=dict)
    chip_now: str | None = None

    def report(self) -> str:
        lines = [f"=== Recommendation for GW{self.gw} (predictions as of GW{self.as_of_gw}; horizon GW{self.horizon_gws[0]}-{self.horizon_gws[-1]}) ==="]
        for w in self.warnings:
            lines.append(f"!!! WARNING: {w}")
        if self.roll_transfer:
            lines.append(f"Transfers: NONE. Roll the free transfer ({self.free_transfers} banked). Best available move gains < {ROLL_THRESHOLD} pts over the horizon.")
        elif not self.transfers:
            lines.append("Transfers: none recommended.")
        else:
            lines.append(f"Transfers ({len(self.transfers)}, {self.hits} hit{'s' if self.hits != 1 else ''} = -{4 * self.hits} pts):")
            for t in self.transfers:
                lines.append(f"  OUT {t['out_name']:<16} ({t['out_price']:.1f})  ->  IN {t['in_name']:<16} ({t['in_price']:.1f})   horizon gain {t['gain']:+.1f}")
        lines.append(f"Expected XI points GW{self.gw}: {self.expected_points_gw:.1f}  (gain vs no transfers over horizon: {self.expected_points_gain:+.1f})")
        lines.append(f"Captain: {self.captain['name']} ({self.captain['pred']:.1f}, q20 {self.captain['low']:.1f} / q80 {self.captain['high']:.1f})   "
                     f"Vice: {self.vice_captain['name']} ({self.vice_captain['pred']:.1f})   risk appetite {self.risk_appetite:+.2f}")
        lines.append(f"Bank after: {self.bank_after:.1f}m   Free transfers left: {self.free_transfers - len(self.transfers) if not self.roll_transfer else self.free_transfers}")
        xi = [p for p in self.squad if p["is_starting"]]
        bench = sorted([p for p in self.squad if not p["is_starting"]], key=lambda p: p["bench_order"])
        by_pos = {pos: [p for p in xi if p["position"] == pos] for pos in ("GKP", "DEF", "MID", "FWD")}
        lines.append("Starting XI:")
        for pos in ("GKP", "DEF", "MID", "FWD"):
            lines.append(f"  {pos}: " + ", ".join(f"{p['name']} {p['pred']:.1f}{' (C)' if p['is_captain'] else ''}{' (V)' if p['is_vice_captain'] else ''}" for p in by_pos[pos]))
        lines.append("Bench: " + ", ".join(f"{p['bench_order']}. {p['name']} {p['pred']:.1f}" for p in bench))
        if self.chips:
            lines += chip_report(self.chips, self.gw)
            if self.chip_now:
                lines.append(f"  >>> Chip recommended THIS gameweek: {self.chips[self.chip_now].to_json()['label']}")
        for n in self.notes:
            lines.append(f"note: {n}")
        lines.append("Nothing has been submitted to FPL. Apply these yourself if you agree.")
        return "\n".join(lines)


# ------------------------------------------------------------------------------------------
# inputs
# ------------------------------------------------------------------------------------------

def load_inputs(conn: psycopg.Connection, season: str) -> dict[str, Any]:
    q = lambda sql, *a: pd.DataFrame(conn.execute(sql, a).fetchall())  # noqa: E731
    pred = q("SELECT * FROM predictions WHERE season=%s AND as_of_gw=(SELECT max(as_of_gw) FROM predictions WHERE season=%s)", season, season)
    if pred.empty:
        raise RuntimeError("no predictions: run `predict` first")
    latest_version = pred.sort_values("predicted_at")["model_version"].iloc[-1]
    pred = pred[pred["model_version"] == latest_version]
    players = q("SELECT * FROM players WHERE season=%s", season)
    teams = q("SELECT id, short_name FROM teams WHERE season=%s", season)
    state = q("SELECT * FROM user_entry_state WHERE season=%s ORDER BY gw DESC LIMIT 1", season)
    if state.empty:
        raise RuntimeError("no user_entry_state: run `ingest` first")
    st = state.iloc[0]
    squad = q("SELECT * FROM user_squad WHERE season=%s AND gw=%s", season, int(st["gw"]))
    windows = q("SELECT * FROM chip_windows WHERE season=%s", season)
    if windows.empty:  # fallback: the 2026/27 layout (two of each chip, one per half)
        windows = pd.DataFrame([{"season": season, "name": n, "start_event": a, "stop_event": b, "number": 1}
                                for n in ("wildcard", "freehit", "bboost", "3xc") for a, b in ((1, 19), (20, 38))])
    used = st.get("chips_used_json") or []
    return {"pred": pred, "players": players, "teams": teams, "state": st, "squad": squad, "chip_windows": windows, "chips_used": list(used)}


def risk_appetite(overall_rank: float | None, total_players: float | None) -> float:
    """-1 (protect a great rank) .. +1 (chase). Linear between the 5th and 50th percentile."""
    if not overall_rank or not total_players:
        return 0.0
    p = float(overall_rank) / float(total_players)
    return float(np.clip((p - 0.05) / 0.45 * 2 - 1, -1, 1))


def protected_ids(players: pd.DataFrame) -> tuple[set[int], dict[int, str]]:
    """Ids of protected players and, for each, why they are (or aren't) locked."""
    ids: set[int] = set()
    reason: dict[int, str] = {}
    for _, p in players.iterrows():
        names = {str(p.get("web_name", "")).lower(), str(p.get("second_name", "")).lower()}
        if any(any(k in n for n in names) for k in config.PROTECTED_PLAYERS):
            if str(p.get("status", "a")) in config.PROTECTED_OVERRIDE_STATUSES:
                reason[int(p["id"])] = f"{p['web_name']} is protected but the FPL API marks him status '{p['status']}' ({p.get('news') or 'no news'}), so the lock is lifted."
            else:
                ids.add(int(p["id"]))
                reason[int(p["id"])] = f"{p['web_name']} is protected (status '{p.get('status')}') and locked into the squad."
    return ids, reason


# ------------------------------------------------------------------------------------------
# core
# ------------------------------------------------------------------------------------------

def build_pool(inp: dict[str, Any], next_gw: int, horizon: int, r: float) -> pd.DataFrame:
    pred, players, teams = inp["pred"], inp["players"], inp["teams"]
    gws = [g for g in range(next_gw, next_gw + horizon)]
    p = pred[pred["gw"].isin(gws)].copy()
    for c in ("predicted_points", "predicted_points_low", "predicted_points_high"):
        p[c] = p[c].astype(float)
    p["w"] = (p["gw"] - next_gw + 1).map(HORIZON_WEIGHTS).fillna(0.3)
    nxt = p[p["gw"] == next_gw].set_index("player_id")
    later = p[p["gw"] > next_gw]
    squad_value = (later["predicted_points"] * later["w"]).groupby(later["player_id"]).sum()
    pool = players[["id", "web_name", "position", "team_id", "price", "status", "chance_of_playing_next_round", "news", "selected_by_percent"]].rename(columns={"id": "player_id"}).copy()
    pool["price"] = pool["price"].astype(float)
    pool = pool.merge(teams.rename(columns={"id": "team_id", "short_name": "team"}), on="team_id", how="left")
    pool["xi_value"] = pool["player_id"].map(nxt["predicted_points"]).fillna(0.0)
    pool["low"] = pool["player_id"].map(nxt["predicted_points_low"]).fillna(0.0)
    pool["high"] = pool["player_id"].map(nxt["predicted_points_high"]).fillna(0.0)
    pool["squad_value"] = pool["player_id"].map(squad_value).fillna(0.0)
    up, down = max(r, 0.0), max(-r, 0.0)
    pool["captain_value"] = pool["xi_value"] + 0.5 * up * (pool["high"] - pool["xi_value"]) - 0.5 * down * (pool["xi_value"] - pool["low"])
    # never buy someone the API says is out (status i/s/u/n with no chance of playing)
    out_now = pool["status"].isin(config.PROTECTED_OVERRIDE_STATUSES) & pool["chance_of_playing_next_round"].fillna(0).astype(float).le(0)
    pool["available"] = ~out_now
    return pool


def _squad_rows(sol: Solution, pool: pd.DataFrame) -> list[dict[str, Any]]:
    pi = pool.set_index("player_id")
    rows = []
    for pid in sol.squad:
        p = pi.loc[pid]
        starting = pid in sol.lineup
        rows.append({
            "player_id": int(pid), "name": str(p["web_name"]), "team": str(p.get("team") or ""), "position": str(p["position"]),
            "price": float(p["price"]), "pred": float(p["xi_value"]), "low": float(p["low"]), "high": float(p["high"]),
            "horizon_value": float(p["squad_value"]), "is_starting": starting,
            "is_captain": pid == sol.captain, "is_vice_captain": pid == sol.vice_captain,
            "bench_order": None if starting else sol.bench.index(pid) + 1,
            "status": str(p.get("status") or "a"), "chance": None if pd.isna(p.get("chance_of_playing_next_round")) else int(p["chance_of_playing_next_round"]),
            "news": None if not p.get("news") or pd.isna(p.get("news")) else str(p["news"]),
        })
    order = {"GKP": 0, "DEF": 1, "MID": 2, "FWD": 3}
    rows.sort(key=lambda x: (not x["is_starting"], x["bench_order"] or 0, order[x["position"]], -x["pred"]))
    return rows


def recommend(conn: psycopg.Connection, season: str, horizon: int = 5, write: bool = True,
              free_transfers_override: int | None = None, max_transfers: int = 3) -> Recommendation:
    inp = load_inputs(conn, season)
    st = inp["state"]
    as_of_gw = int(inp["pred"]["as_of_gw"].max())
    next_gw = as_of_gw + 1
    horizon_gws = list(range(next_gw, next_gw + horizon))
    r = risk_appetite(st.get("overall_rank"), st.get("total_players"))
    pool = build_pool(inp, next_gw, horizon, r)

    current = [int(x) for x in inp["squad"]["player_id"].tolist()]
    if len(current) != 15:
        raise RuntimeError(f"user_squad for GW{st['gw']} has {len(current)} players, expected 15")
    bank = float(st["bank"] or 0.0)
    squad_value = float(st["squad_value"] or 0.0)        # FPL's own selling value of the 15
    budget = bank + squad_value                            # conservative: kept players valued at sell price too
    fts = int(free_transfers_override if free_transfers_override is not None else (st.get("free_transfers_after") or 1))

    locked, reasons = protected_ids(inp["players"])
    banned = set(pool.loc[~pool["available"], "player_id"]) - set(current)
    hit_cost = int(round(config.TRANSFER_HIT_COST * HIT_RISK_PREMIUM))
    common = dict(budget=budget, current_squad=current, hit_cost=hit_cost, bench_weight=1.0,
                  captain_col="captain_value", banned=banned)

    base = solve_squad(pool, free_transfers=fts, max_transfers=0, locked=set(), **common)
    best = solve_squad(pool, free_transfers=fts, max_transfers=max(fts, max_transfers), locked=locked & set(current), **common)
    warnings, notes = [], []
    for pid, why in reasons.items():
        (notes if pid in locked else warnings).append(why)
    if locked & set(current):
        unlocked = solve_squad(pool, free_transfers=fts, max_transfers=max(fts, max_transfers), locked=set(), **common)
        sold = [pid for pid in unlocked.transfers_out if pid in locked]
        if sold:
            names = ", ".join(pool.set_index("player_id").loc[pid, "web_name"] for pid in sold)
            warnings.append(f"PROTECTED PLAYER: the optimiser wanted to SELL {names} (+{unlocked.objective - best.objective:.1f} pts over the horizon). "
                            f"Blocked because he is not injured/suspended per the FPL API. Override only if you are sure.")
    for pid in locked - set(current):
        notes.append(f"protected player {pool.set_index('player_id').loc[pid, 'web_name']} is not in your squad; nothing to lock.")

    gain = best.objective - base.objective
    roll = bool(best.transfers_in) and gain < ROLL_THRESHOLD and fts < config.MAX_FREE_TRANSFERS
    chosen = base if (roll or not best.transfers_in) else best
    pi = pool.set_index("player_id")

    transfers = []
    if chosen is best:
        outs = sorted(best.transfers_out, key=lambda i: pi.loc[i, "position"])
        ins = sorted(best.transfers_in, key=lambda i: pi.loc[i, "position"])
        for o, i in zip(outs, ins):  # pair by position where possible (FPL requires like-for-like)
            transfers.append({"out": int(o), "out_name": pi.loc[o, "web_name"], "out_price": float(pi.loc[o, "price"]),
                              "in": int(i), "in_name": pi.loc[i, "web_name"], "in_price": float(pi.loc[i, "price"]),
                              "gain": float((pi.loc[i, "xi_value"] + pi.loc[i, "squad_value"]) - (pi.loc[o, "xi_value"] + pi.loc[o, "squad_value"]))})
    squad_rows = _squad_rows(chosen, pool)
    cap = next(x for x in squad_rows if x["is_captain"])
    vice = next(x for x in squad_rows if x["is_vice_captain"])
    bank_after = budget - chosen.cost
    rec = Recommendation(
        season=season, gw=next_gw, as_of_gw=as_of_gw, horizon_gws=horizon_gws, squad=squad_rows, transfers=transfers,
        hits=chosen.hits if chosen is best else 0, captain=cap, vice_captain=vice,
        expected_points_gw=float(chosen.xi_points), expected_points_gain=float(gain if chosen is best else 0.0),
        roll_transfer=roll, free_transfers=fts, bank_after=bank_after, risk_appetite=r,
        warnings=warnings, notes=notes, protected_locked=[pi.loc[i, "web_name"] for i in locked & set(current)],
    )
    # --- chips: each evaluated alone, per candidate gameweek, against the no-chip baseline ---
    try:
        rec.chips = evaluate_chips(inp["pred"], inp["players"], chosen.squad, budget, inp["chip_windows"], inp["chips_used"],
                                   next_gw, horizon_gws, locked=locked & set(chosen.squad), banned=banned)
        rec.chip_now = pick_chip_for_now(rec.chips)
    except Exception as exc:  # noqa: BLE001 - chip planning must never block the core recommendation
        log.exception("chip evaluation failed")
        rec.notes.append(f"chip evaluation failed: {exc}")
    if r != 0 and st.get("total_players"):
        rec.notes.append(f"captaincy shaded towards the {'q80 ceiling (chasing rank)' if r > 0 else 'q20 floor (protecting rank)'}: "
                         f"overall rank {int(st['overall_rank']):,} of {int(st['total_players']):,}")
    if write:
        conn.execute(
            "INSERT INTO recommendations (season, gw, model_version, squad_json, transfers_json, captain_id, vice_captain_id, chip_used, "
            "chip_plan_json, expected_points_gain, expected_points_gw, warnings_json, horizon_weights, status) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,'recommended')",
            (season, next_gw, str(inp["pred"]["model_version"].iloc[0]), Jsonb(squad_rows),
             Jsonb({"transfers": transfers, "hits": rec.hits, "roll_transfer": roll, "free_transfers": fts, "bank_after": bank_after,
                    "budget": budget, "as_of_gw": as_of_gw}),
             cap["player_id"], vice["player_id"], rec.chip_now,
             Jsonb({k: v.to_json() for k, v in rec.chips.items()}) if rec.chips else None,
             rec.expected_points_gain, rec.expected_points_gw,
             Jsonb({"warnings": warnings, "notes": [n for n in rec.notes if n], "risk_appetite": r, "protected_locked": rec.protected_locked}),
             json.dumps(HORIZON_WEIGHTS)),
        )
        conn.commit()
    return rec
