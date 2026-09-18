"""Chip timing (spec section 5): evaluate each chip ON its own, for every candidate gameweek in the
look-ahead window, against the no-chip baseline, and report the week with the biggest edge.
No joint optimisation of several chips (v1).

Per candidate gameweek g (predictions exist for g), with S = the squad after this week's
recommended transfers and XI_g = best legal XI from S using GW g predictions:
  Bench Boost     gain = sum(pred_g over S) - sum(pred_g over XI_g)          (the bench counts)
  Triple Captain  gain = max(pred_g over XI_g)                               (captain 3x instead of 2x)
  Free Hit        gain = best fresh 15/XI for g only (budget = bank + value) - XI_g points
  Wildcard        gain = best fresh squad valued over g..end of horizon - S valued the same way,
                  minus what ordinary free transfers would plausibly capture anyway
A chip is recommended for THIS gameweek only if this gameweek is its best week within the window
AND the gain clears the chip's threshold. Availability: chip windows from bootstrap-static and the
chips already used this season (user_entry_state.chips_used_json). One chip per gameweek.

Nothing here activates a chip on FPL (Safety Rule 3): this is planning output only.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from pipeline.features.build_features import HORIZON_WEIGHTS
from pipeline.optimize.solver import solve_squad

log = logging.getLogger(__name__)

CHIPS = ("bboost", "3xc", "freehit", "wildcard")
CHIP_LABELS = {"bboost": "Bench Boost", "3xc": "Triple Captain", "freehit": "Free Hit", "wildcard": "Wildcard"}
# minimum predicted gain (points) before a chip is recommended now; deliberately demanding because
# predictions are shrunk towards the mean and a chip spent early cannot be spent on a double gameweek.
THRESHOLDS = {"bboost": 12.0, "3xc": 7.0, "freehit": 12.0, "wildcard": 15.0}
FT_ALLOWANCE_PER_GW = 1.5   # points/gw ordinary free transfers are assumed to capture (subtracted from the wildcard gain)


@dataclass
class ChipEval:
    chip: str
    available: bool
    reason: str
    by_gw: dict[int, float] = field(default_factory=dict)
    best_gw: int | None = None
    best_gain: float = 0.0
    threshold: float = 0.0
    recommend_now: bool = False
    detail: dict[int, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {"chip": self.chip, "label": CHIP_LABELS[self.chip], "available": self.available, "reason": self.reason,
                "by_gw": {str(k): round(v, 2) for k, v in self.by_gw.items()}, "best_gw": self.best_gw,
                "best_gain": round(self.best_gain, 2), "threshold": self.threshold, "recommend_now": self.recommend_now,
                "detail": {str(k): v for k, v in self.detail.items()}}


def chip_availability(windows: pd.DataFrame, used: list[dict[str, Any]], gw: int) -> dict[str, tuple[bool, str]]:
    """Is each chip usable in `gw`? A chip is usable if some window contains gw and no chip of that
    name has already been played inside that window."""
    out = {}
    for chip in CHIPS:
        w = windows[(windows["name"] == chip) & (windows["start_event"] <= gw) & (windows["stop_event"] >= gw)]
        if w.empty:
            out[chip] = (False, f"{CHIP_LABELS[chip]}: no window covers GW{gw}")
            continue
        win = w.iloc[0]
        played = [u for u in used if u.get("name") == chip and win["start_event"] <= int(u.get("event", 0)) <= win["stop_event"]]
        if played:
            out[chip] = (False, f"{CHIP_LABELS[chip]}: already used in GW{played[0]['event']} (window GW{win['start_event']}-{win['stop_event']})")
        else:
            out[chip] = (True, f"{CHIP_LABELS[chip]}: available until GW{win['stop_event']}")
    return out


def _frame_for_gw(pred: pd.DataFrame, players: pd.DataFrame, gw: int, weights_from: int | None = None,
                  last_gw: int | None = None) -> pd.DataFrame:
    """Pool with xi_value = pred at gw and (optionally) squad_value = weighted preds after gw."""
    p = players[["id", "position", "team_id", "price", "web_name"]].rename(columns={"id": "player_id"}).copy()
    p["price"] = p["price"].astype(float)
    at = pred[pred["gw"] == gw].set_index("player_id")["predicted_points"].astype(float)
    p["xi_value"] = p["player_id"].map(at).fillna(0.0)
    p["squad_value"] = 0.0
    if weights_from is not None and last_gw is not None:
        later = pred[(pred["gw"] > gw) & (pred["gw"] <= last_gw)].copy()
        later["w"] = (later["gw"] - weights_from + 1).map(HORIZON_WEIGHTS).fillna(0.3)
        sv = (later["predicted_points"].astype(float) * later["w"]).groupby(later["player_id"]).sum()
        p["squad_value"] = p["player_id"].map(sv).fillna(0.0)
    return p


def _best_xi_from_squad(frame: pd.DataFrame, squad: list[int]) -> tuple[list[int], float, int]:
    sub = frame[frame["player_id"].isin(squad)]
    sol = solve_squad(sub, budget=10_000.0, locked=set(squad), bench_weight=0.0)
    if sol.status != "Optimal":
        return [], 0.0, -1
    return sol.lineup, float(sum(sub.set_index("player_id").loc[sol.lineup, "xi_value"])), sol.captain


def evaluate_chips(pred: pd.DataFrame, players: pd.DataFrame, squad: list[int], budget: float,
                   windows: pd.DataFrame, used: list[dict[str, Any]], next_gw: int, horizon_gws: list[int],
                   locked: set[int] | None = None, banned: set[int] | None = None) -> dict[str, ChipEval]:
    evals: dict[str, ChipEval] = {}
    avail_now = chip_availability(windows, used, next_gw)
    last_gw = horizon_gws[-1]
    pi = players.set_index("id")

    for chip in CHIPS:
        ok, why = avail_now[chip]
        ev = ChipEval(chip=chip, available=ok, reason=why, threshold=THRESHOLDS[chip])
        evals[chip] = ev
        for g in horizon_gws:
            g_ok, _ = chip_availability(windows, used, g)[chip]
            if not g_ok:
                continue
            frame = _frame_for_gw(pred, players, g, weights_from=g if chip == "wildcard" else None, last_gw=last_gw)
            lineup, xi_pts, cap = _best_xi_from_squad(frame, squad)
            if not lineup:
                continue
            fi = frame.set_index("player_id")
            if chip == "bboost":
                bench = [i for i in squad if i not in lineup]
                gain = float(fi.loc[bench, "xi_value"].sum())
                ev.detail[g] = {"bench": [str(pi.loc[i, "web_name"]) for i in bench], "bench_points": round(gain, 2)}
            elif chip == "3xc":
                best = fi.loc[lineup, "xi_value"].idxmax()
                gain = float(fi.loc[best, "xi_value"])
                ev.detail[g] = {"captain": str(pi.loc[best, "web_name"]), "captain_points": round(gain, 2)}
            elif chip == "freehit":
                sol = solve_squad(frame, budget=budget, banned=banned, bench_weight=0.0)
                gain = float(sol.xi_points - (xi_pts + fi.loc[cap, "xi_value"])) if sol.status == "Optimal" else 0.0
                ev.detail[g] = {"free_hit_xi_points": round(sol.xi_points, 2), "held_xi_points": round(xi_pts + float(fi.loc[cap, "xi_value"]), 2),
                                "xi": [str(pi.loc[i, "web_name"]) for i in sol.lineup]}
            else:  # wildcard: value over g..last_gw, current squad vs fresh squad, minus FT allowance
                sol = solve_squad(frame, budget=budget, locked=locked, banned=banned, bench_weight=1.0)
                held = solve_squad(frame[frame["player_id"].isin(squad)], budget=10_000.0, locked=set(squad), bench_weight=1.0)
                gws_left = last_gw - g + 1
                raw = float(sol.objective - held.objective) if sol.status == "Optimal" and held.status == "Optimal" else 0.0
                gain = raw - FT_ALLOWANCE_PER_GW * gws_left
                ev.detail[g] = {"raw_uplift": round(raw, 2), "ft_allowance": round(FT_ALLOWANCE_PER_GW * gws_left, 2),
                                "squad": [str(pi.loc[i, "web_name"]) for i in sol.squad] if sol.status == "Optimal" else []}
            ev.by_gw[g] = gain
        if ev.by_gw:
            ev.best_gw = max(ev.by_gw, key=ev.by_gw.get)
            ev.best_gain = ev.by_gw[ev.best_gw]
            ev.recommend_now = bool(ok and ev.best_gw == next_gw and ev.best_gain >= ev.threshold)
    return evals


def pick_chip_for_now(evals: dict[str, ChipEval]) -> str | None:
    """At most one chip per gameweek: the recommended one with the largest margin over its threshold."""
    cands = [(e.best_gain - e.threshold, e.chip) for e in evals.values() if e.recommend_now]
    return max(cands)[1] if cands else None


def chip_report(evals: dict[str, ChipEval], next_gw: int) -> list[str]:
    lines = ["Chips (planning only; you activate them yourself):"]
    for chip in CHIPS:
        e = evals[chip]
        if not e.available and not e.by_gw:
            lines.append(f"  {CHIP_LABELS[chip]:<14} {e.reason}")
            continue
        by = ", ".join(f"GW{g}: {v:+.1f}" for g, v in e.by_gw.items())
        verdict = (f"USE NOW (+{e.best_gain:.1f} >= {e.threshold})" if e.recommend_now
                   else f"hold (best GW{e.best_gw} {e.best_gain:+.1f}, threshold {e.threshold})")
        lines.append(f"  {CHIP_LABELS[chip]:<14} {verdict}   [{by}]")
        d = e.detail.get(e.best_gw or -1, {})
        if chip == "wildcard" and d.get("squad"):
            lines.append(f"      wildcard squad for GW{e.best_gw}: " + ", ".join(d["squad"]) + f"  (raw uplift {d['raw_uplift']:+.1f}, minus FT allowance {d['ft_allowance']:.1f})")
        if chip == "freehit" and d.get("xi"):
            lines.append(f"      free-hit XI for GW{e.best_gw}: " + ", ".join(d["xi"]))
        if chip == "bboost" and d.get("bench"):
            lines.append(f"      bench in GW{e.best_gw}: " + ", ".join(d["bench"]))
        if chip == "3xc" and d.get("captain"):
            lines.append(f"      triple-captain pick for GW{e.best_gw}: {d['captain']} ({d['captain_points']} pts predicted)")
    return lines
