"""Parameterised season simulator: the backtest harness of spec section 6 with every policy knob
pulled out into a config the dashboard can post.

The engine runs in two passes, because the two halves cost wildly different amounts:

  1. prediction pass  refit the models at the chosen cadence and predict every gameweek of the
                      replayed season. Point-in-time throughout: at gameweek T the models see only
                      earlier seasons and gameweeks < T. This is the expensive pass (minutes).
  2. replay pass      walk the season applying a transfer / captaincy / chip policy to those cached
                      predictions and score each gameweek on the ACTUAL points. Cheap (seconds),
                      so Monte Carlo repeats and policy variants reuse pass 1 instead of refitting.

Two season modes:
  free_hit   best legal 15 + XI from scratch every gameweek on a fresh 100.0 budget ("perfect
             wildcard every week"): what the predictions are worth with no transfer friction.
  realistic  pick a squad, then 1 free transfer a week (banked to 5, hits allowed if the solver
             thinks they pay), continuous budget.

Known optimism/pessimism, unchanged from the original harness: historical injury flags are not
available, so the model can field a player who was actually injured (pessimistic for the model);
free_hit ignores transfer friction entirely (optimistic). Chip hindsight mode is a ceiling, not a
strategy. Nothing here ever talks to FPL.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Callable

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
# 4 * hit_risk_premium inside the objective (the real -4 is still what gets scored).
# Measured on 2025/26: premium 1.5 -> 23 hits, 1924 pts; premium 3.0 -> 3 hits, 1976 pts.
DEFAULT_HIT_RISK_PREMIUM = float(__import__("os").getenv("VANTAGE_HIT_RISK_PREMIUM", "3.0"))

CHIPS = ("bboost", "3xc", "freehit", "wildcard")
CHIP_LABELS = {"bboost": "Bench Boost", "3xc": "Triple Captain", "freehit": "Free Hit", "wildcard": "Wildcard"}
# Same thresholds the live recommender uses (predicted gain, in points, needed to burn a chip).
DEFAULT_CHIP_THRESHOLDS = {"bboost": 12.0, "3xc": 7.0, "freehit": 12.0, "wildcard": 15.0}
# 2026/27 gives two of each chip, one per half of the season.
FIRST_HALF_END = 19
# q20..q80 spans 1.683 standard deviations of a normal, which is how we turn the quantile models
# into a noise scale for the Monte Carlo repeats.
Q_SPAN_SIGMAS = 1.683

Progress = Callable[[str], None]


def _noop(_msg: str) -> None:
    pass


# ------------------------------------------------------------------------------------------
# config
# ------------------------------------------------------------------------------------------

@dataclass
class SimConfig:
    """Everything the dashboard form can set. Defaults reproduce the original backtest exactly."""
    season: str
    start_gw: int = 2
    end_gw: int = 38
    mode: str = "realistic"                 # realistic | free_hit
    strategy: str = "model"                 # model | naive_last5

    # --- model knobs: these invalidate the prediction pass ---
    refit_every: int = 4
    rounds: int = 300
    horizon: int = 5

    # --- transfer policy ---
    hit_risk_premium: float = DEFAULT_HIT_RISK_PREMIUM
    max_transfers_per_gw: int = 3
    allow_hits: bool = True
    roll_threshold: float = 0.0             # keep the squad unless the best move gains this much
    sell_fee: bool = False                  # charge the 50% sell-on fee on price rises

    # --- squad shape and captaincy ---
    bench_weight: float | None = None       # None = mode default (1.0 realistic, 0.05 free_hit)
    captain_metric: str = "mean"            # mean | q20 | q80 | risk
    risk_appetite: float = 0.0              # -1 floor .. +1 ceiling, only when captain_metric=risk
    locked: tuple[str, ...] = ()            # names that must stay in the squad
    banned: tuple[str, ...] = ()            # names that can never be picked
    formation: str | None = None            # "3-4-3" etc, forces the XI shape

    # --- chips ---
    chip_policy: str = "none"               # none | threshold | hindsight
    chip_thresholds: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_CHIP_THRESHOLDS))

    # --- monte carlo ---
    repeats: int = 1
    seed: int = 42

    label: str | None = None

    def __post_init__(self) -> None:
        if self.mode not in ("realistic", "free_hit"):
            raise ValueError(f"mode must be realistic or free_hit, got {self.mode!r}")
        if self.strategy not in ("model", "naive_last5"):
            raise ValueError(f"strategy must be model or naive_last5, got {self.strategy!r}")
        if self.captain_metric not in ("mean", "q20", "q80", "risk"):
            raise ValueError(f"captain_metric must be mean/q20/q80/risk, got {self.captain_metric!r}")
        if self.chip_policy not in ("none", "threshold", "hindsight"):
            raise ValueError(f"chip_policy must be none/threshold/hindsight, got {self.chip_policy!r}")
        if not 1 <= self.repeats <= 25:
            raise ValueError("repeats must be between 1 and 25")
        if not 1 <= self.start_gw <= 38 or not 1 <= self.end_gw <= 38 or self.start_gw > self.end_gw:
            raise ValueError("start_gw/end_gw must be 1..38 with start <= end")
        if self.formation is not None:
            self.xi_shape  # noqa: B018 - validate now, not mid-replay
        self.locked = tuple(self.locked or ())
        self.banned = tuple(self.banned or ())
        self.chip_thresholds = {**DEFAULT_CHIP_THRESHOLDS, **(self.chip_thresholds or {})}

    @property
    def xi_shape(self) -> tuple[int, int, int] | None:
        """(DEF, MID, FWD) counts for a forced formation. GKP is always 1."""
        if not self.formation:
            return None
        parts = [p for p in str(self.formation).replace("-", " ").split() if p]
        if len(parts) != 3 or not all(p.isdigit() for p in parts):
            raise ValueError(f"formation must look like 3-4-3, got {self.formation!r}")
        d, m, f = (int(p) for p in parts)
        if d + m + f != 10:
            raise ValueError(f"formation {self.formation} has {d + m + f} outfielders, needs 10")
        if d < XI_MIN["DEF"] or m < XI_MIN["MID"] or f < XI_MIN["FWD"]:
            raise ValueError(f"formation {self.formation} breaks the FPL minimums 3/2/1")
        return d, m, f

    @property
    def needs_quantiles(self) -> bool:
        """Quantile models roughly double the fit time, so only pay for them when a knob uses them."""
        return self.repeats > 1 or self.captain_metric != "mean"

    @property
    def effective_bench_weight(self) -> float:
        if self.bench_weight is not None:
            return float(self.bench_weight)
        return 1.0 if self.mode == "realistic" else 0.05

    def to_dict(self) -> dict:
        d = asdict(self)
        d["locked"] = list(self.locked)
        d["banned"] = list(self.banned)
        return d


@dataclass
class GWResult:
    gw: int
    points: float
    captain_points: float
    transfers: int
    hits: int
    bench_points: float
    predicted_xi: float
    chip: str | None = None
    squad_cost: float = 0.0     # what the 15 cost this week; must never exceed the budget


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
                "chips": {g.chip: g.gw for g in self.gws if g.chip},
                "per_gw": [g.__dict__ for g in self.gws]}


@dataclass
class SimResult:
    config: SimConfig
    runs: list[BacktestResult]
    seconds: float = 0.0

    @property
    def totals(self) -> list[float]:
        return [r.total for r in self.runs]

    @property
    def best(self) -> BacktestResult:
        """The run reported in detail: the median total, so the per-gw table matches a typical run."""
        return sorted(self.runs, key=lambda r: r.total)[len(self.runs) // 2]

    def summary(self) -> dict:
        t = np.array(self.totals, dtype=float)
        s = self.best.summary()
        s.update({
            "n_repeats": len(self.runs),
            "total_points_mean": float(t.mean()),
            "total_points_sd": float(t.std(ddof=1)) if len(t) > 1 else 0.0,
            "total_points_p10": float(np.percentile(t, 10)),
            "total_points_p90": float(np.percentile(t, 90)),
            "totals": [float(x) for x in t],
            "seconds": round(self.seconds, 1),
            "config": self.config.to_dict(),
        })
        return s


# ------------------------------------------------------------------------------------------
# scoring with FPL rules
# ------------------------------------------------------------------------------------------

def score_gameweek(lineup: list[int], bench: list[int], captain: int, vice: int,
                   actual: dict[int, float], minutes: dict[int, float], pos_of: dict[int, str],
                   bench_boost: bool = False, captain_multiplier: int = 2) -> tuple[float, float, float]:
    """Returns (points, captain_bonus, bench_points_unused). Auto-subs: a starter with 0 minutes is
    replaced by the first bench player (in order) that keeps the formation legal. Under a bench
    boost all 15 score and there is nothing to substitute."""
    played = lambda i: minutes.get(i, 0) > 0  # noqa: E731
    cap = captain if played(captain) else (vice if played(vice) else None)
    cap_bonus = actual.get(cap, 0.0) * (captain_multiplier - 1) if cap is not None else 0.0

    if bench_boost:
        pts = sum(actual.get(i, 0.0) for i in list(lineup) + list(bench))
        return pts + cap_bonus, cap_bonus, 0.0

    xi = list(lineup)
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
    bench_unused = sum(actual.get(i, 0.0) for i in bench_left)
    return pts + cap_bonus, cap_bonus, bench_unused


# ------------------------------------------------------------------------------------------
# pass 1: predictions
# ------------------------------------------------------------------------------------------

def _predict_rows(models: dict[str, PositionModel], rows: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    n = len(rows)
    mid = pd.Series(np.nan, index=rows.index)
    lo = pd.Series(np.nan, index=rows.index)
    hi = pd.Series(np.nan, index=rows.index)
    for pos, m in models.items():
        mask = rows["position"] == pos
        if mask.any():
            p, l, h = m.predict(rows[mask])
            mid[mask], lo[mask], hi[mask] = p, l, h
    assert len(mid) == n
    return mid.clip(lower=0), lo.clip(lower=0), hi.clip(lower=0)


def _future_value(models: dict[str, PositionModel], stats: pd.DataFrame, fixtures: pd.DataFrame, teams: pd.DataFrame,
                  season: str, as_of: int, horizon: int) -> pd.Series:
    """Horizon-weighted predicted points for gameweeks as_of+2 .. as_of+horizon (as_of+1 is the XI
    value, handled separately), using only data up to as_of. Indexed by player_id."""
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
    pred, _, _ = _predict_rows(models, rows)
    w = (rows["gw"] - as_of).map(HORIZON_WEIGHTS).fillna(0.3)
    pred = pred.where(rows["n_fixtures"] > 0, 0.0)
    return (pred * w).groupby(rows["player_id"]).sum()


@dataclass
class GwFrame:
    gw: int
    rows: pd.DataFrame          # player_id, position, team_id, price, pred, pred_low, pred_high,
                                # future, target, target_minutes
    model_version: str


def _prediction_pass(conn: psycopg.Connection, cfg: SimConfig, progress: Progress) -> list[GwFrame]:
    t = load_tables(conn)
    ff = build_training_frame(t["stats"], t["fixtures"], t["teams"])
    frame, feature_cols = ff.rows[ff.rows["season"] <= cfg.season], ff.feature_cols
    prev_season = max([s for s in t["stats"]["season"].unique() if s < cfg.season], default=None)
    stats_h = t["stats"][t["stats"]["season"].isin([cfg.season, prev_season])]
    fixtures_h = t["fixtures"][t["fixtures"]["season"].isin([cfg.season, prev_season])]

    gws = sorted(frame[frame["season"] == cfg.season]["gw"].unique())
    gws = [g for g in gws if cfg.start_gw <= g <= cfg.end_gw]
    if not gws:
        raise ValueError(f"no gameweeks in {cfg.season} between GW{cfg.start_gw} and GW{cfg.end_gw}; "
                         f"is that season loaded? (run `history`)")

    out: list[GwFrame] = []
    models: dict[str, PositionModel] | None = None
    last_fit: int | None = None
    keep = ["player_id", "position", "team_id", "price", "target", "target_minutes"]
    for n, T in enumerate(gws, 1):
        rows = frame[(frame["season"] == cfg.season) & (frame["gw"] == T)].copy()
        if rows.empty:
            continue
        if cfg.strategy == "model":
            if models is None or last_fit is None or T - last_fit >= cfg.refit_every:
                train_rows = frame[(frame["season"] < cfg.season) | (frame["gw"] < T)]
                progress(f"fitting models at GW{T} on {len(train_rows):,} rows ({n}/{len(gws)})")
                models = train_all(train_rows, feature_cols, POSITIONS, {p: cfg.rounds for p in POSITIONS},
                                   with_quantiles=cfg.needs_quantiles)
                last_fit = T
                log.info("sim %s GW%d: refit on %d rows", cfg.season, T, len(train_rows))
            pred, lo, hi = _predict_rows(models, rows)
            rows["pred"], rows["pred_low"], rows["pred_high"] = pred, lo, hi
        else:  # naive: last-5 mean scaled by fixtures
            rows["pred"] = rows["points_l5"].fillna(0) * rows["n_fixtures"].clip(lower=1) * rows["played_l5"].fillna(1)
            rows["pred_low"] = rows["pred"] * 0.5
            rows["pred_high"] = rows["pred"] * 1.5

        if cfg.mode == "realistic" and cfg.strategy == "model" and cfg.horizon > 1:
            fut = _future_value(models, stats_h, fixtures_h, t["teams"], cfg.season, T - 1, cfg.horizon)
            rows["future"] = rows["player_id"].map(fut).fillna(0.0)
        elif cfg.mode == "realistic":
            rows["future"] = rows["pred"] * sum(w for k, w in HORIZON_WEIGHTS.items() if 2 <= k <= cfg.horizon)
        else:
            rows["future"] = 0.0

        out.append(GwFrame(int(T), rows[keep + ["pred", "pred_low", "pred_high", "future"]].copy(),
                           MODEL_VERSION if cfg.strategy == "model" else "naive_last5"))
        progress(f"predicted GW{T} ({n}/{len(gws)})")
    return out


# ------------------------------------------------------------------------------------------
# pass 2: replay
# ------------------------------------------------------------------------------------------

def _resolve_names(conn: psycopg.Connection, season: str, names: tuple[str, ...]) -> tuple[set[int], list[str]]:
    """Map user-typed player names to that season's element ids. Returns (ids, names_not_found)."""
    if not names:
        return set(), []
    rows = conn.execute(
        "SELECT DISTINCT s.player_id, lower(coalesce(p.web_name,'')) AS web, lower(coalesce(p.second_name,'')) AS second "
        "FROM player_gw_stats s LEFT JOIN players p ON p.code = s.player_code WHERE s.season = %s",
        (season,)).fetchall()
    ids: set[int] = set()
    missing: list[str] = []
    for raw in names:
        needle = raw.strip().lower()
        if not needle:
            continue
        hit = {r["player_id"] for r in rows if needle in (r["web"] or "") or needle in (r["second"] or "")}
        if hit:
            ids |= hit
        else:
            missing.append(raw)
    return ids, missing


def _captain_column(rows: pd.DataFrame, cfg: SimConfig) -> pd.Series:
    """What the solver maximises when choosing the captain. q20 protects a rank, q80 chases one."""
    if cfg.captain_metric == "q20":
        return rows["pred_low"]
    if cfg.captain_metric == "q80":
        return rows["pred_high"]
    if cfg.captain_metric == "risk":
        a = float(np.clip(cfg.risk_appetite, -1.0, 1.0))
        other = rows["pred_high"] if a >= 0 else rows["pred_low"]
        return rows["pred"] * (1 - abs(a)) + other * abs(a)
    return rows["pred"]


def _perturb(rows: pd.DataFrame, rng: np.random.Generator) -> pd.Series:
    """Resample predictions inside their own q20/q80 interval. Answers 'how much of the season total
    is the policy and how much is prediction noise', by holding the actuals fixed and jittering only
    what the manager believed at the time."""
    sigma = ((rows["pred_high"] - rows["pred_low"]) / Q_SPAN_SIGMAS).clip(lower=0.0).fillna(0.0)
    return (rows["pred"] + rng.normal(0.0, sigma)).clip(lower=0.0)


def _replay(frames: list[GwFrame], cfg: SimConfig, rng: np.random.Generator | None,
            locked_ids: set[int], banned_ids: set[int], hindsight_chips: dict[int, str] | None,
            progress: Progress) -> BacktestResult:
    from pipeline.optimize.solver import solve_squad

    res = BacktestResult(cfg.season, cfg.mode, cfg.strategy)
    bench_w = cfg.effective_bench_weight
    hit_cost = int(round(4 * cfg.hit_risk_premium))
    xi_shape = cfg.xi_shape
    squad: list[int] | None = None
    bank = START_BUDGET
    free_transfers = 1
    buy_price: dict[int, float] = {}          # what we paid, for the 50% sell-on fee
    chips_left = {c: {1: 1, 2: 1} for c in CHIPS}   # two of each, one per half
    last_rows: dict[int, pd.Series] = {}      # last seen row per player, to hold blanked slots

    for f in frames:
        T = f.gw
        rows = f.rows.copy()
        if rng is not None:
            rows["pred"] = _perturb(rows, rng)
        rows["xi_value"] = rows["pred"]
        rows["squad_value"] = rows["future"]
        rows["cap_value"] = _captain_column(rows, cfg)

        for _, r in rows.iterrows():
            last_rows[int(r["player_id"])] = r

        actual = dict(zip(rows["player_id"], rows["target"].astype(float)))
        minutes = dict(zip(rows["player_id"], rows["target_minutes"]))
        pos_of = dict(zip(rows["player_id"], rows["position"]))
        price = dict(zip(rows["player_id"], rows["price"].astype(float)))

        # players in the current squad with no row this gameweek (blank / transferred out of the
        # game) still occupy their slot at their last known price and score nothing.
        if squad:
            missing = [i for i in squad if i not in set(rows["player_id"])]
            held = [last_rows[i] for i in missing if i in last_rows]
            if held:
                h = pd.DataFrame(held).assign(pred=0.0, xi_value=0.0, squad_value=0.0, cap_value=0.0,
                                              target=0.0, target_minutes=0)
                rows = pd.concat([rows, h[rows.columns.intersection(h.columns)]], ignore_index=True)
                price.update({int(r["player_id"]): float(r["price"]) for r in held})
                pos_of.update({int(r["player_id"]): r["position"] for r in held})
                actual.update({int(r["player_id"]): 0.0 for r in held})
                minutes.update({int(r["player_id"]): 0 for r in held})

        half = 1 if T <= FIRST_HALF_END else 2
        chip_used: str | None = None
        forced = (hindsight_chips or {}).get(T)

        def _solve(current, fts, max_t, budget, **kw):
            return solve_squad(rows, budget=budget, current_squad=current, free_transfers=fts,
                               max_transfers=max_t, hit_cost=hit_cost, bench_weight=bench_w,
                               locked=locked_ids or None, banned=banned_ids or None,
                               captain_col="cap_value", xi_shape=xi_shape, **kw)

        # ---------- choose a squad ----------
        if cfg.mode == "free_hit" or squad is None:
            # perfect wildcard every week: rebuild from scratch on the ORIGINAL budget, not on a
            # budget that compounds with the squad you happen to be holding.
            sol = _solve(None, 1, None, START_BUDGET)
            if sol.status != "Optimal":
                log.warning("GW%d: solver status %s", T, sol.status)
                continue
            transfers, hits = 0, 0
            if squad is None and cfg.mode == "realistic":
                bank = START_BUDGET - sol.cost
                buy_price = {i: price.get(i, 0.0) for i in sol.squad}
        else:
            sell_value = sum(_sell_value(i, price, buy_price, cfg.sell_fee) for i in squad)
            max_t = max(free_transfers, cfg.max_transfers_per_gw) if cfg.allow_hits else free_transfers
            sol = _solve(squad, free_transfers, max_t, bank + sell_value)
            if sol.status != "Optimal":
                log.warning("GW%d: solver status %s", T, sol.status)
                continue
            if cfg.roll_threshold > 0 and sol.transfers_in:
                stay = _solve(squad, free_transfers, 0, bank + sell_value)
                if stay.status == "Optimal" and (sol.objective - stay.objective) < cfg.roll_threshold:
                    sol = stay          # the best available move is not worth the free transfer
            transfers, hits = len(sol.transfers_in), sol.hits
            bank = bank + sell_value - sol.cost
            for i in sol.transfers_in:
                buy_price[i] = price.get(i, 0.0)
            free_transfers = min(MAX_FREE_TRANSFERS, max(free_transfers - transfers, 0) + 1)

        squad = sol.squad
        lineup, bench, captain, vice = sol.lineup, sol.bench, sol.captain, sol.vice_captain

        # ---------- chips ----------
        if cfg.chip_policy != "none":
            cand = _chip_candidate(rows, sol, cfg, chips_left, half, forced, _solve, squad, bank, price, buy_price)
            if cand:
                chip_used, cand_sol = cand
                chips_left[chip_used][half] = 0
                if chip_used in ("freehit", "wildcard") and cand_sol is not None:
                    lineup, bench = cand_sol.lineup, cand_sol.bench
                    captain, vice = cand_sol.captain, cand_sol.vice_captain
                    if chip_used == "wildcard":
                        squad = cand_sol.squad
                        bank = bank + sum(_sell_value(i, price, buy_price, cfg.sell_fee) for i in sol.squad) - cand_sol.cost
                        for i in cand_sol.squad:
                            buy_price.setdefault(i, price.get(i, 0.0))
                        transfers, hits = len(set(cand_sol.squad) - set(sol.squad)), 0
                        free_transfers = 1

        pts, cap_pts, bench_pts = score_gameweek(
            lineup, bench, captain, vice, actual, minutes, pos_of,
            bench_boost=(chip_used == "bboost"), captain_multiplier=3 if chip_used == "3xc" else 2)
        pts -= 4 * hits
        res.gws.append(GWResult(int(T), float(pts), float(cap_pts), int(transfers), int(hits),
                                float(bench_pts), float(sol.xi_points), chip_used,
                                float(sum(price.get(i, 0.0) for i in squad))))
        log.info("sim %s %s/%s GW%d: %.0f pts (cap %.0f, %d transfers, %d hits, bank %.1f%s)",
                 cfg.season, cfg.mode, cfg.strategy, T, pts, cap_pts, transfers, hits, bank,
                 f", {CHIP_LABELS[chip_used]}" if chip_used else "")
    return res


def _sell_value(pid: int, price: dict[int, float], buy_price: dict[int, float], sell_fee: bool) -> float:
    """FPL sells at the buy price plus half the rise, rounded down to 0.1. Off by default because
    the rest of the harness treats buy = sell = current price."""
    now = price.get(pid, 0.0)
    if not sell_fee:
        return now
    bought = buy_price.get(pid, now)
    rise = now - bought
    if rise <= 0:
        return now
    return bought + np.floor(rise * 10 / 2) / 10


def _chip_candidate(rows, sol, cfg: SimConfig, chips_left, half: int, forced: str | None,
                    solve, squad, bank, price, buy_price):
    """Returns (chip, solution_or_None) if a chip should be played this gameweek.

    threshold: the same predicted-gain thresholds the live recommender uses.
    hindsight: `forced` names the chip a prior pass proved was best here (bench boost and triple
               captain only, where the gain is exactly knowable after the fact).
    """
    def available(c):
        return chips_left.get(c, {}).get(half, 0) > 0

    if forced:
        if not available(forced):
            return None
        if forced in ("bboost", "3xc"):
            return forced, None
        return None

    if cfg.chip_policy != "threshold":
        return None

    pred = dict(zip(rows["player_id"], rows["pred"].astype(float)))
    cap = dict(zip(rows["player_id"], rows["cap_value"].astype(float)))
    gains: dict[str, tuple[float, object]] = {}

    if available("bboost"):
        gains["bboost"] = (sum(pred.get(i, 0.0) for i in sol.bench), None)
    if available("3xc"):
        gains["3xc"] = (cap.get(sol.captain, 0.0), None)
    if available("freehit"):
        fh = solve(None, 15, None, START_BUDGET)
        if fh.status == "Optimal":
            gains["freehit"] = (fh.xi_points - sol.xi_points, fh)
    if available("wildcard"):
        sell = sum(_sell_value(i, price, buy_price, cfg.sell_fee) for i in squad)
        wc = solve(squad, 15, None, bank + sell)
        if wc.status == "Optimal":
            gains["wildcard"] = (wc.objective - sol.objective, wc)

    best = max(((c, g, s) for c, (g, s) in gains.items() if g >= cfg.chip_thresholds.get(c, 1e9)),
               key=lambda x: x[1] - cfg.chip_thresholds.get(x[0], 0.0), default=None)
    return (best[0], best[2]) if best else None


def _hindsight_plan(base: BacktestResult, frames: list[GwFrame]) -> dict[int, str]:
    """Best possible gameweek for Bench Boost and Triple Captain, given what actually happened in a
    no-chip run: one of each per half. This is a ceiling, not a strategy you could have followed."""
    by_gw = {g.gw: g for g in base.gws}
    plan: dict[int, str] = {}
    for half, lo, hi in ((1, 1, FIRST_HALF_END), (2, FIRST_HALF_END + 1, 38)):
        pool = [g for g in base.gws if lo <= g.gw <= hi]
        if not pool:
            continue
        for chip, key in (("bboost", lambda g: g.bench_points), ("3xc", lambda g: g.captain_points)):
            cand = [g for g in pool if g.gw not in plan]
            if not cand:
                continue
            best = max(cand, key=key)
            if key(best) > 0:
                plan[best.gw] = chip
    assert len(by_gw) >= len(plan)
    return plan


# ------------------------------------------------------------------------------------------
# entry point
# ------------------------------------------------------------------------------------------

def run_simulation(conn: psycopg.Connection, cfg: SimConfig, progress: Progress | None = None,
                   write: bool = True, job_id: int | None = None) -> SimResult:
    progress = progress or _noop
    t0 = time.time()
    locked_ids, miss_l = _resolve_names(conn, cfg.season, cfg.locked)
    banned_ids, miss_b = _resolve_names(conn, cfg.season, cfg.banned)
    for name in miss_l + miss_b:
        log.warning("no player matching %r in %s; ignoring", name, cfg.season)
    if locked_ids & banned_ids:
        raise ValueError("the same player is both locked and banned")

    progress("building features and fitting models")
    frames = _prediction_pass(conn, cfg, progress)

    hindsight: dict[int, str] | None = None
    if cfg.chip_policy == "hindsight":
        progress("replaying once without chips to find the best chip weeks")
        base = _replay(frames, cfg, None, locked_ids, banned_ids, None, progress)
        hindsight = _hindsight_plan(base, frames)
        progress("best chip weeks: " + (", ".join(f"{CHIP_LABELS[c]} GW{g}" for g, c in sorted(hindsight.items())) or "none"))

    runs: list[BacktestResult] = []
    for r in range(cfg.repeats):
        rng = np.random.default_rng(cfg.seed + r) if cfg.repeats > 1 else None
        progress(f"replaying season ({r + 1}/{cfg.repeats})")
        runs.append(_replay(frames, cfg, rng, locked_ids, banned_ids, hindsight, progress))

    result = SimResult(cfg, runs, time.time() - t0)
    if write and runs:
        s = result.summary()
        conn.execute(
            "INSERT INTO backtests (season, mode, strategy, model_version, start_gw, end_gw, total_points, "
            "per_gw_json, notes, params_json, job_id, label, n_repeats, points_sd, points_p10, points_p90) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            (cfg.season, cfg.mode, cfg.strategy, frames[0].model_version, frames[0].gw, frames[-1].gw,
             s["total_points_mean"], json.dumps(s["per_gw"]),
             f"hits={s['hits_total']} transfers={s['transfers_total']} refit_every={cfg.refit_every} "
             f"rounds={cfg.rounds} horizon={cfg.horizon} chips={cfg.chip_policy} repeats={cfg.repeats}",
             json.dumps(cfg.to_dict()), job_id, cfg.label, cfg.repeats,
             s["total_points_sd"], s["total_points_p10"], s["total_points_p90"]),
        )
        conn.commit()
    return result


def run_backtest(conn: psycopg.Connection, season: str, mode: str = "realistic", strategy: str = "model",
                 start_gw: int = 2, end_gw: int = 38, refit_every: int = 4, rounds: int = 300,
                 horizon: int = 5, write: bool = True) -> BacktestResult:
    """Back-compatible entry point for `run_weekly backtest` and anything that called the old
    harness. Same defaults, same numbers, minus the free_hit budget bug."""
    cfg = SimConfig(season=season, mode=mode, strategy=strategy, start_gw=start_gw, end_gw=end_gw,
                    refit_every=refit_every, rounds=rounds, horizon=horizon)
    return run_simulation(conn, cfg, write=write).best
