"""Train-on-everything then predict the next 1..5 gameweeks for every current player.

Availability is applied AFTER the model (the archive has no historical injury flags, so the model
can't learn it): expected points are scaled by chance_of_playing_next_round / 100, zeroed for
status injured/suspended/unavailable with no stated chance, and zeroed for blank gameweeks.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import psycopg

from pipeline.db.conn import upsert
from pipeline.features.build_features import HORIZON_WEIGHTS, build_prediction_frame, build_training_frame
from pipeline.models.train import MODEL_VERSION, POSITIONS, PositionModel, train_all

log = logging.getLogger(__name__)
UNAVAILABLE_STATUSES = {"i", "s", "u", "n"}


@dataclass
class PredictResult:
    season: str
    as_of_gw: int
    target_gws: list[int]
    n_rows: int
    model_version: str
    predictions: pd.DataFrame


def load_tables(conn: psycopg.Connection) -> dict[str, pd.DataFrame]:
    q = lambda sql: pd.DataFrame(conn.execute(sql).fetchall())  # noqa: E731
    return {
        "stats": q("SELECT * FROM player_gw_stats"),
        "fixtures": q("SELECT * FROM fixtures"),
        "teams": q("SELECT * FROM teams"),
        "players": q("SELECT * FROM players"),
    }


def availability_multiplier(players: pd.DataFrame) -> pd.Series:
    chance = pd.to_numeric(players["chance_of_playing_next_round"], errors="coerce")
    status = players["status"].fillna("a")
    mult = chance.fillna(100.0) / 100.0
    mult = mult.where(~(chance.isna() & status.isin(UNAVAILABLE_STATUSES)), 0.0)
    return mult.clip(0.0, 1.0)


def current_gw_state(fixtures: pd.DataFrame, season: str) -> tuple[int, int]:
    """(last finished gw, next gw) from the fixture list of the live season."""
    fx = fixtures[(fixtures["season"] == season) & fixtures["gw"].notna()]
    finished = fx[fx["finished"]]["gw"].astype(int)
    last_done = int(finished.max()) if not finished.empty else 0
    remaining = fx[~fx["finished"]]["gw"].astype(int)
    next_gw = int(remaining.min()) if not remaining.empty else last_done + 1
    return last_done, next_gw


def run_predict(conn: psycopg.Connection, season: str, positions=POSITIONS, horizon: int = 5,
                as_of_gw: int | None = None, write: bool = True,
                models: dict[str, PositionModel] | None = None) -> PredictResult:
    t = load_tables(conn)
    last_done, next_gw = current_gw_state(t["fixtures"], season)
    as_of = last_done if as_of_gw is None else as_of_gw
    target_gws = list(range(as_of + 1, as_of + 1 + horizon))
    log.info("predict: season=%s as_of_gw=%d targets=%s", season, as_of, target_gws)

    train = build_training_frame(t["stats"], t["fixtures"], t["teams"])
    train_rows = train.rows[(train.rows["season"] != season) | (train.rows["gw"] <= as_of)]
    models = models or train_all(train_rows, train.feature_cols, positions)

    players = t["players"][t["players"]["season"] == season].copy()
    pf = build_prediction_frame(t["stats"], t["fixtures"], t["teams"], players, season, as_of, target_gws)
    rows = pf.rows
    rows["pred"], rows["pred_low"], rows["pred_high"] = np.nan, np.nan, np.nan
    for pos, model in models.items():
        m = rows["position"] == pos
        if m.any():
            p, lo, hi = model.predict(rows.loc[m])
            rows.loc[m, "pred"], rows.loc[m, "pred_low"], rows.loc[m, "pred_high"] = p, lo, hi
    rows = rows.dropna(subset=["pred"])

    mult = availability_multiplier(rows)
    # availability only applies with certainty to the next gw; beyond that, decay the penalty
    steps = rows["gw"] - as_of
    mult = np.where(steps <= 1, mult, 1.0 - (1.0 - mult) * 0.5)
    mult = np.where(rows["n_fixtures"] <= 0, 0.0, mult)
    for c in ("pred", "pred_low", "pred_high"):
        rows[c] = (rows[c].clip(lower=0) * mult).round(2)
    rows["horizon_weight"] = (rows["gw"] - as_of).map(HORIZON_WEIGHTS).fillna(0.3)

    out = rows[["season", "player_id", "gw", "pred", "pred_low", "pred_high"]].rename(
        columns={"pred": "predicted_points", "pred_low": "predicted_points_low", "pred_high": "predicted_points_high"})
    out["as_of_gw"] = as_of
    out["model_version"] = MODEL_VERSION
    if write:
        upsert(conn, "predictions", out.to_dict("records"), ["season", "player_id", "gw", "model_version"])
        for pos, model in models.items():
            conn.execute(
                "INSERT INTO model_runs (model_version, position, n_train, notes) VALUES (%s,%s,%s,%s)",
                (MODEL_VERSION, pos, model.metrics.get("n_train"), f"refit for as_of_gw={as_of}; top gain: "
                 + ", ".join(f"{k}={v:.2f}" for k, v in model.importance(5).items())),
            )
        conn.commit()
    keep = ["season", "player_id", "web_name", "position", "team_id", "gw", "pred", "pred_low", "pred_high",
            "price", "status", "chance_of_playing_next_round", "n_fixtures", "fdr", "is_home", "horizon_weight"]
    return PredictResult(season, as_of, target_gws, len(out), MODEL_VERSION, rows[keep].reset_index(drop=True))
