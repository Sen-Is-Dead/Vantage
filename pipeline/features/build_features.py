"""Feature pipeline (spec section 4). Every feature for a (player, gameweek T) row uses ONLY data from
gameweeks < T: player form is the rolling state after the player's most recent gameweek before T,
fixture features come from the fixture list for T (known in advance), and price is the last
observed price. The same code path builds training rows (T in the past, target known) and
prediction rows (T in the future), so there is no train/serve skew.

Recent form is deliberately over-represented: plain rolling means over the last 3 / 5 / 10 gameweeks
AND an exponentially-weighted mean (half-life 2 GWs, so the last GW carries ~30% of the weight)
for points, xG, xA, xGI, minutes, BPS and ICT. `pipeline.models.train` prints feature importance so
we can confirm these actually carry the model (spec asks for that check).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

WINDOWS = (3, 5, 10)
EWM_HALFLIFE = 2.0
FORM_COLS = ("points", "xg", "xa", "xgi", "minutes", "bps", "ict_index", "bonus", "goals", "assists",
             "clean_sheet", "goals_conceded", "xgc", "saves", "defensive_contribution", "starts")
HORIZON_WEIGHTS = {1: 1.0, 2: 0.7, 3: 0.5, 4: 0.4, 5: 0.3}


@dataclass
class FeatureFrames:
    rows: pd.DataFrame          # one row per (season, player_id, gw) with features (+ target when known)
    feature_cols: list[str]


# --------------------------------------------------------------------------------------------
# 1. per-gameweek aggregation (double gameweeks -> one row, sums)
# --------------------------------------------------------------------------------------------

def aggregate_per_gw(stats: pd.DataFrame) -> pd.DataFrame:
    s = stats.copy()
    for c in FORM_COLS + ("value",):
        if c not in s.columns:
            s[c] = 0.0
        s[c] = pd.to_numeric(s[c], errors="coerce").fillna(0.0).astype(float)
    s["kickoff_time"] = pd.to_datetime(s["kickoff_time"], utc=True)
    agg = {c: "sum" for c in FORM_COLS}
    agg.update({"value": "last", "kickoff_time": "max", "team_id": "last", "position": "last",
                "player_code": "last", "fixture_id": "count"})
    g = (s.sort_values(["season", "player_id", "gw", "kickoff_time"])
          .groupby(["season", "player_id", "gw"], as_index=False, sort=True).agg(agg)
          .rename(columns={"fixture_id": "n_fixtures_played_in", "value": "price"}))
    g["played"] = (g["minutes"] > 0).astype(float)
    g["started"] = (g["starts"] > 0).astype(float)
    return g


# --------------------------------------------------------------------------------------------
# 2. rolling "state after gameweek g" per player-season
# --------------------------------------------------------------------------------------------

def rolling_state(per_gw: pd.DataFrame) -> pd.DataFrame:
    """Adds columns describing the player's form INCLUDING gameweek g (state after g)."""
    df = per_gw.sort_values(["season", "player_id", "gw"]).reset_index(drop=True)
    keys = ["season", "player_id"]
    cols = list(FORM_COLS + ("played", "started"))
    grp = df.groupby(keys, sort=False)
    parts = []
    for w in WINDOWS:  # vectorised grouped rolling (Cython), not a per-group Python lambda
        r = grp[cols].rolling(w, min_periods=1).mean().reset_index(level=keys, drop=True).sort_index()
        parts.append(r.add_suffix(f"_l{w}"))
    e = grp[cols].ewm(halflife=EWM_HALFLIFE, min_periods=1).mean().reset_index(level=keys, drop=True).sort_index()
    parts.append(e.add_suffix("_ewm"))
    x = grp[["points", "minutes"]].expanding().mean().reset_index(level=keys, drop=True).sort_index()
    parts.append(x.rename(columns={"points": "points_season_ppg", "minutes": "minutes_season_avg"}))
    extra = pd.DataFrame({
        "gws_played_season": grp.cumcount() + 1,
        "price_start": grp["price"].transform("first"),
        "last_kickoff": df["kickoff_time"],
        "last_gw": df["gw"],
    })
    state = pd.concat([df] + parts + [extra], axis=1)
    state["price_change_season"] = state["price"] - state["price_start"]
    return state


STATE_COLS_BASE = ["price", "price_change_season", "gws_played_season", "points_season_ppg", "minutes_season_avg"]


def state_feature_cols() -> list[str]:
    cols = list(STATE_COLS_BASE)
    for c in FORM_COLS + ("played", "started"):
        cols += [f"{c}_l{w}" for w in WINDOWS] + [f"{c}_ewm"]
    return cols


# --------------------------------------------------------------------------------------------
# 3. previous-season priors (helps GW1-3 when there is little current-season form)
# --------------------------------------------------------------------------------------------

def previous_season_priors(per_gw: pd.DataFrame) -> pd.DataFrame:
    seasons = sorted(per_gw["season"].unique())
    nxt = {s: seasons[i + 1] for i, s in enumerate(seasons[:-1])}
    # also allow mapping the last archive season to the following live season label
    last = seasons[-1]
    y = int(last.split("/")[0]) + 1
    nxt[last] = f"{y}/{str(y + 1)[-2:]}"
    p = (per_gw.groupby(["season", "player_code"], as_index=False)
              .agg(prev_ppg=("points", "mean"), prev_min_per_gw=("minutes", "mean"),
                   prev_xgi_per_gw=("xgi", "mean"), prev_gws=("gw", "count"), prev_total=("points", "sum")))
    p["season"] = p["season"].map(nxt)
    return p.dropna(subset=["season"])


PRIOR_COLS = ["prev_ppg", "prev_min_per_gw", "prev_xgi_per_gw", "prev_gws", "prev_total"]


# --------------------------------------------------------------------------------------------
# 4. fixture features for a target gameweek (known in advance)
# --------------------------------------------------------------------------------------------

def team_fixture_features(fixtures: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    """One row per (season, team_id, gw): n_fixtures, home share, mean FDR, opponent strengths,
    own strengths, first kickoff. Blank gameweeks simply have no row."""
    fx = fixtures.dropna(subset=["gw"]).copy()
    fx["gw"] = fx["gw"].astype(int)
    fx["kickoff_time"] = pd.to_datetime(fx["kickoff_time"], utc=True)
    home = fx.rename(columns={"home_team_id": "team_id", "away_team_id": "opp_id", "difficulty_home": "fdr"})
    home["is_home"] = 1.0
    away = fx.rename(columns={"away_team_id": "team_id", "home_team_id": "opp_id", "difficulty_away": "fdr"})
    away["is_home"] = 0.0
    cols = ["season", "gw", "team_id", "opp_id", "fdr", "is_home", "kickoff_time", "id"]
    both = pd.concat([home[cols], away[cols]], ignore_index=True)

    t = teams[["season", "id", "strength_attack_home", "strength_attack_away",
               "strength_defence_home", "strength_defence_away", "strength_overall_home", "strength_overall_away"]]
    both = both.merge(t.rename(columns={"id": "opp_id"}), on=["season", "opp_id"], how="left")
    # opponent plays away when we are home
    both["opp_attack"] = np.where(both["is_home"] == 1, both["strength_attack_away"], both["strength_attack_home"])
    both["opp_defence"] = np.where(both["is_home"] == 1, both["strength_defence_away"], both["strength_defence_home"])
    both["opp_overall"] = np.where(both["is_home"] == 1, both["strength_overall_away"], both["strength_overall_home"])
    both = both.drop(columns=[c for c in both.columns if c.startswith("strength_")])
    both = both.merge(t.rename(columns={"id": "team_id"}), on=["season", "team_id"], how="left")
    both["own_attack"] = np.where(both["is_home"] == 1, both["strength_attack_home"], both["strength_attack_away"])
    both["own_defence"] = np.where(both["is_home"] == 1, both["strength_defence_home"], both["strength_defence_away"])
    both["own_overall"] = np.where(both["is_home"] == 1, both["strength_overall_home"], both["strength_overall_away"])
    both = both.drop(columns=[c for c in both.columns if c.startswith("strength_")])

    tf = (both.groupby(["season", "team_id", "gw"], as_index=False)
              .agg(n_fixtures=("id", "count"), is_home=("is_home", "mean"), fdr=("fdr", "mean"),
                   opp_attack=("opp_attack", "mean"), opp_defence=("opp_defence", "mean"),
                   opp_overall=("opp_overall", "mean"), own_attack=("own_attack", "mean"),
                   own_defence=("own_defence", "mean"), own_overall=("own_overall", "mean"),
                   first_kickoff=("kickoff_time", "min")))
    tf["strength_diff"] = tf["own_overall"] - tf["opp_overall"]
    return tf


FIXTURE_COLS = ["n_fixtures", "is_home", "fdr", "opp_attack", "opp_defence", "opp_overall",
                "own_attack", "own_defence", "own_overall", "strength_diff"]


def team_form(fixtures: pd.DataFrame) -> pd.DataFrame:
    """Team goals for/against over the last 5 finished fixtures, as state after gameweek g."""
    fx = fixtures.dropna(subset=["gw", "home_score", "away_score"]).copy()
    fx["gw"] = fx["gw"].astype(int)
    h = fx.rename(columns={"home_team_id": "team_id", "home_score": "gf", "away_score": "ga"})[["season", "team_id", "gw", "gf", "ga", "kickoff_time"]]
    a = fx.rename(columns={"away_team_id": "team_id", "away_score": "gf", "home_score": "ga"})[["season", "team_id", "gw", "gf", "ga", "kickoff_time"]]
    both = pd.concat([h, a]).sort_values(["season", "team_id", "kickoff_time"])
    grp = both.groupby(["season", "team_id"], sort=False)
    both["team_gf_l5"] = grp["gf"].transform(lambda x: x.rolling(5, min_periods=1).mean())
    both["team_ga_l5"] = grp["ga"].transform(lambda x: x.rolling(5, min_periods=1).mean())
    return both.groupby(["season", "team_id", "gw"], as_index=False).agg(team_gf_l5=("team_gf_l5", "last"), team_ga_l5=("team_ga_l5", "last"))


TEAM_FORM_COLS = ["team_gf_l5", "team_ga_l5"]


# --------------------------------------------------------------------------------------------
# 5. assemble
# --------------------------------------------------------------------------------------------

def _attach_state(targets: pd.DataFrame, state: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """For each target (season, player_id, gw) attach the player's latest state with last_gw < gw."""
    st = state[["season", "player_id", "last_gw", "last_kickoff"] + cols].sort_values("last_gw")
    tg = targets.sort_values("gw").copy()
    tg["_gw_key"] = tg["gw"].astype(float) - 0.5
    st["_gw_key"] = st["last_gw"].astype(float)
    tg = tg.sort_values("_gw_key")
    st = st.sort_values("_gw_key")
    merged = pd.merge_asof(tg, st, on="_gw_key", by=["season", "player_id"], direction="backward")
    return merged.drop(columns=["_gw_key"])


def _attach_team_form(rows: pd.DataFrame, tform: pd.DataFrame) -> pd.DataFrame:
    tf = tform.rename(columns={"gw": "_tgw"}).sort_values("_tgw")
    tf["_gw_key"] = tf["_tgw"].astype(float)
    r = rows.copy()
    r["_gw_key"] = r["gw"].astype(float) - 0.5
    r = r.sort_values("_gw_key")
    m = pd.merge_asof(r, tf, on="_gw_key", by=["season", "team_id"], direction="backward")
    return m.drop(columns=["_gw_key", "_tgw"])


def all_feature_cols() -> list[str]:
    return state_feature_cols() + PRIOR_COLS + FIXTURE_COLS + TEAM_FORM_COLS + ["days_since_last_match", "gw_number"]


def build_training_frame(stats: pd.DataFrame, fixtures: pd.DataFrame, teams: pd.DataFrame,
                         min_prior_gws: int = 1) -> FeatureFrames:
    """Rows for every (season, player, gw) where the player has at least `min_prior_gws` earlier
    gameweeks that season. Target = points scored in that gw."""
    per_gw = aggregate_per_gw(stats)
    state = rolling_state(per_gw)
    priors = previous_season_priors(per_gw)
    tfx = team_fixture_features(fixtures, teams)
    tform = team_form(fixtures)

    targets = per_gw[["season", "player_id", "player_code", "position", "team_id", "gw", "points", "minutes"]].rename(
        columns={"points": "target", "minutes": "target_minutes"})
    rows = _attach_state(targets, state, state_feature_cols())
    rows = rows[rows["gws_played_season"].fillna(0) >= min_prior_gws]
    rows = rows.merge(priors, on=["season", "player_code"], how="left")
    rows = rows.merge(tfx, on=["season", "team_id", "gw"], how="left")
    rows = _attach_team_form(rows, tform)
    rows["days_since_last_match"] = (rows["first_kickoff"] - rows["last_kickoff"]).dt.total_seconds() / 86400.0
    rows["gw_number"] = rows["gw"].astype(float)
    rows = rows.dropna(subset=["n_fixtures"])  # team had a fixture that gw
    return FeatureFrames(rows.reset_index(drop=True), all_feature_cols())


def build_prediction_frame(stats: pd.DataFrame, fixtures: pd.DataFrame, teams: pd.DataFrame,
                           players: pd.DataFrame, season: str, as_of_gw: int, target_gws: list[int]) -> FeatureFrames:
    """Rows for every current player x target gw, using only stats with gw <= as_of_gw."""
    hist = stats[(stats["season"] != season) | (stats["gw"] <= as_of_gw)]
    per_gw = aggregate_per_gw(hist)
    state = rolling_state(per_gw)
    priors = previous_season_priors(per_gw)
    tfx = team_fixture_features(fixtures, teams)
    tform = team_form(fixtures[(fixtures["season"] != season) | (fixtures["gw"] <= as_of_gw)])

    base = players[["id", "code", "position", "team_id", "price", "cost_change_start",
                    "status", "chance_of_playing_next_round", "web_name"]].rename(
        columns={"id": "player_id", "code": "player_code", "price": "price_now"})
    targets = pd.concat([base.assign(gw=g, season=season) for g in target_gws], ignore_index=True)
    rows = _attach_state(targets, state, state_feature_cols())
    # the live snapshot is the freshest price; it also covers players with no rows yet this season
    rows["price"] = pd.to_numeric(rows["price_now"], errors="coerce").astype(float).fillna(rows["price"])
    rows["price_change_season"] = pd.to_numeric(rows["cost_change_start"], errors="coerce").astype(float).fillna(rows["price_change_season"])
    rows["gws_played_season"] = rows["gws_played_season"].fillna(0)
    rows = rows.merge(priors, on=["season", "player_code"], how="left")
    rows = rows.merge(tfx, on=["season", "team_id", "gw"], how="left")
    rows = _attach_team_form(rows, tform)
    rows["days_since_last_match"] = (rows["first_kickoff"] - rows["last_kickoff"]).dt.total_seconds() / 86400.0
    rows["gw_number"] = rows["gw"].astype(float)
    rows["n_fixtures"] = rows["n_fixtures"].fillna(0)  # blank gw -> 0 fixtures -> 0 points downstream
    return FeatureFrames(rows.reset_index(drop=True), all_feature_cols())
