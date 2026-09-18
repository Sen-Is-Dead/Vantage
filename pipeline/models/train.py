"""One LightGBM regressor per position (GKP/DEF/MID/FWD), target = FPL points in the gameweek.

Supervised retraining on a growing dataset (not "reinforcement learning"): every weekly run refits
from scratch on all history in player_gw_stats. Evaluation is strictly time-ordered: a validation
season (default: the most recent complete one) is held out, MAE/RMSE are compared with two naive
baselines, then the production model is refit on everything.

Point estimate: L2 regression. Interval: two extra quantile models (q20 / q80) give
predicted_points_low / high for the dashboard and for captaincy risk (Phase 4).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

POSITIONS = ("GKP", "DEF", "MID", "FWD")
MODEL_VERSION = "lgbm-v1"

LGB_PARAMS = dict(
    objective="regression", learning_rate=0.03, num_leaves=31, min_data_in_leaf=60,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1, seed=42,
)
QUANTILE_PARAMS = {**LGB_PARAMS, "objective": "quantile"}
NUM_ROUNDS = 600


@dataclass
class PositionModel:
    position: str
    feature_cols: list[str]
    booster: lgb.Booster
    q_low: lgb.Booster | None = None
    q_high: lgb.Booster | None = None
    version: str = MODEL_VERSION
    metrics: dict = field(default_factory=dict)

    def predict(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = X[self.feature_cols].astype(float)
        p = self.booster.predict(x)
        lo = self.q_low.predict(x) if self.q_low is not None else p - 2.0
        hi = self.q_high.predict(x) if self.q_high is not None else p + 3.0
        return p, np.minimum(lo, p), np.maximum(hi, p)

    def importance(self, top: int = 15) -> pd.Series:
        imp = pd.Series(self.booster.feature_importance("gain"), index=self.feature_cols)
        return (imp / imp.sum()).sort_values(ascending=False).head(top)


def _fit(X: pd.DataFrame, y: pd.Series, params: dict, rounds: int, valid: tuple | None = None) -> lgb.Booster:
    dtrain = lgb.Dataset(X.astype(float), y.astype(float))
    if valid is not None:
        dvalid = lgb.Dataset(valid[0].astype(float), valid[1].astype(float), reference=dtrain)
        return lgb.train(params, dtrain, rounds, valid_sets=[dvalid],
                         callbacks=[lgb.early_stopping(50, verbose=False)])
    return lgb.train(params, dtrain, rounds)


def _mae(y, p) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def _rmse(y, p) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def evaluate_position(rows: pd.DataFrame, feature_cols: list[str], position: str,
                      valid_season: str) -> dict:
    """Train on seasons < valid_season, evaluate on valid_season. Returns metrics incl. baselines."""
    d = rows[rows["position"] == position]
    seasons = sorted(d["season"].unique())
    train = d[d["season"] < valid_season]
    valid = d[d["season"] == valid_season]
    if train.empty or valid.empty:
        raise ValueError(f"{position}: need seasons before and including {valid_season}; have {seasons}")
    booster = _fit(train[feature_cols], train["target"], LGB_PARAMS, NUM_ROUNDS,
                   valid=(valid[feature_cols], valid["target"]))
    p = booster.predict(valid[feature_cols].astype(float))
    y = valid["target"].values
    # naive baselines: last-5 mean, season ppg, and (for reference) the constant 0
    b_l5 = valid["points_l5"].fillna(0).values
    b_ppg = valid["points_season_ppg"].fillna(0).values
    m = {
        "position": position, "valid_season": valid_season, "n_train": int(len(train)), "n_valid": int(len(valid)),
        "best_iteration": int(booster.best_iteration or NUM_ROUNDS),
        "mae": _mae(y, p), "rmse": _rmse(y, p),
        "mae_baseline_last5": _mae(y, b_l5), "mae_baseline_ppg": _mae(y, b_ppg), "mae_baseline_zero": _mae(y, 0 * y),
    }
    # the numbers that matter for picking players: the top of the predicted ranking
    vv = valid.assign(pred=p)
    played = vv[vv["minutes_l5"].fillna(0) > 45]  # regular starters only
    m["mae_starters"] = _mae(played["target"], played["pred"])
    m["mae_starters_baseline_last5"] = _mae(played["target"], played["points_l5"].fillna(0))
    top = vv.sort_values("pred", ascending=False).groupby("gw").head(10)
    m["top10_per_gw_actual_mean"] = float(top["target"].mean())
    top_b = vv.sort_values("points_l5", ascending=False).groupby("gw").head(10)
    m["top10_per_gw_actual_mean_baseline_last5"] = float(top_b["target"].mean())
    corr = float(np.corrcoef(y, p)[0, 1])
    m["spearman"] = float(pd.Series(y).rank().corr(pd.Series(p).rank()))
    m["pearson"] = corr
    imp = pd.Series(booster.feature_importance("gain"), index=feature_cols)
    imp = (imp / imp.sum()).sort_values(ascending=False)
    m["top_features"] = imp.head(12).round(4).to_dict()
    form_share = float(imp[[c for c in feature_cols if any(c.startswith(f"{b}_") for b in ("points", "xg", "xa", "xgi", "minutes", "bps", "ict_index", "bonus", "played", "started"))]].sum())
    m["form_feature_gain_share"] = form_share
    return m


def train_position(rows: pd.DataFrame, feature_cols: list[str], position: str, rounds: int | None = None,
                   with_quantiles: bool = True) -> PositionModel:
    d = rows[rows["position"] == position]
    if d.empty:
        raise ValueError(f"no training rows for {position}")
    n = rounds or NUM_ROUNDS
    booster = _fit(d[feature_cols], d["target"], LGB_PARAMS, n)
    q_lo = _fit(d[feature_cols], d["target"], {**QUANTILE_PARAMS, "alpha": 0.2}, max(n // 2, 100)) if with_quantiles else None
    q_hi = _fit(d[feature_cols], d["target"], {**QUANTILE_PARAMS, "alpha": 0.8}, max(n // 2, 100)) if with_quantiles else None
    return PositionModel(position, list(feature_cols), booster, q_lo, q_hi,
                         metrics={"n_train": int(len(d)), "rounds": n})


def train_all(rows: pd.DataFrame, feature_cols: list[str], positions=POSITIONS, rounds_by_pos: dict | None = None,
              with_quantiles: bool = True) -> dict[str, PositionModel]:
    return {p: train_position(rows, feature_cols, p, (rounds_by_pos or {}).get(p), with_quantiles) for p in positions}
