"""Prediction accuracy log: once a gameweek has finished, compare what we predicted for it (made
as of the previous gameweek) with what actually happened, per position. Fills
`prediction_accuracy` so the dashboard can show a running MAE over the season."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import psycopg

from pipeline.db.conn import upsert

log = logging.getLogger(__name__)


def compute_accuracy(conn: psycopg.Connection, season: str) -> pd.DataFrame:
    q = lambda sql, *a: pd.DataFrame(conn.execute(sql, a).fetchall())  # noqa: E731
    pred = q("SELECT player_id, gw, as_of_gw, model_version, predicted_points FROM predictions WHERE season=%s", season)
    if pred.empty:
        return pd.DataFrame()
    # the "next-gw" prediction only: made as of gw-1
    pred = pred[pred["gw"] == pred["as_of_gw"] + 1]
    actual = q("SELECT player_id, gw, position, sum(points) AS points, sum(minutes) AS minutes FROM player_gw_stats "
               "WHERE season=%s GROUP BY player_id, gw, position", season)
    finished = q("SELECT gw FROM fixtures WHERE season=%s AND gw IS NOT NULL GROUP BY gw HAVING bool_and(finished)", season)
    if actual.empty or finished.empty:
        return pd.DataFrame()
    done = set(finished["gw"].astype(int))
    m = pred.merge(actual, on=["player_id", "gw"], how="inner")
    m = m[m["gw"].isin(done)]
    if m.empty:
        return pd.DataFrame()
    m["err"] = m["predicted_points"].astype(float) - m["points"].astype(float)
    rows = []
    for (gw, mv, pos), g in m.groupby(["gw", "model_version", "position"]):
        rows.append({"season": season, "gw": int(gw), "model_version": mv, "position": pos, "n": int(len(g)),
                     "mae": float(g["err"].abs().mean()), "rmse": float(np.sqrt((g["err"] ** 2).mean()))})
    for (gw, mv), g in m.groupby(["gw", "model_version"]):
        rows.append({"season": season, "gw": int(gw), "model_version": mv, "position": "ALL", "n": int(len(g)),
                     "mae": float(g["err"].abs().mean()), "rmse": float(np.sqrt((g["err"] ** 2).mean()))})
    out = pd.DataFrame(rows)
    upsert(conn, "prediction_accuracy", out.to_dict("records"), ["season", "gw", "model_version", "position"])
    conn.commit()
    return out
