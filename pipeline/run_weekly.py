"""Single entrypoint the GitHub Action (and you, locally) calls.

    python -m pipeline.run_weekly ingest [--quick]        # Phase 1: FPL API -> Postgres
    python -m pipeline.run_weekly history [--seasons ...] # Phase 2: past seasons (vaastav archive) -> Postgres
    python -m pipeline.run_weekly train [--positions MID] # evaluate on a held-out season, log MAE to model_runs
    python -m pipeline.run_weekly predict [--positions MID] [--horizon 5]   # refit on everything, write predictions
    python -m pipeline.run_weekly counts
    python -m pipeline.run_weekly reset-db --yes           # drop ALL tables (data is regenerable), re-apply schema

This script only ever READS from FPL and WRITES to our own Postgres. It never submits anything to
FPL (Safety Rule 1).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys

import pandas as pd

from pipeline.db.conn import apply_schema, get_conn
from pipeline.ingest.fpl_api import FPLClient
from pipeline.ingest.history import DEFAULT_SEASONS, load_history
from pipeline.ingest.load import db_counts, run_ingest
from pipeline.models.train import POSITIONS


def _live_season(conn) -> str:
    row = conn.execute("SELECT season FROM players ORDER BY updated_at DESC LIMIT 1").fetchone()
    if not row:
        raise SystemExit("players table is empty: run `ingest` first")
    return row["season"]


def cmd_train(conn, args) -> None:
    from pipeline.features.build_features import build_training_frame
    from pipeline.models.predict import load_tables
    from pipeline.models.train import MODEL_VERSION, evaluate_position

    t = load_tables(conn)
    ff = build_training_frame(t["stats"], t["fixtures"], t["teams"])
    seasons = sorted(ff.rows["season"].unique())
    valid = args.valid_season or [s for s in seasons if len(ff.rows[ff.rows.season == s]) > 5000][-1]
    print(f"training frame: {len(ff.rows)} rows, {len(ff.feature_cols)} features, seasons={seasons}, valid={valid}")
    for pos in args.positions:
        m = evaluate_position(ff.rows, ff.feature_cols, pos, valid)
        conn.execute(
            "INSERT INTO model_runs (model_version, position, mae, rmse, n_train, n_valid, notes) VALUES (%s,%s,%s,%s,%s,%s,%s)",
            (MODEL_VERSION, pos, m["mae"], m["rmse"], m["n_train"], m["n_valid"],
             json.dumps({k: v for k, v in m.items() if k not in ("position", "n_train", "n_valid", "mae", "rmse")})),
        )
        conn.commit()
        print(f"\n== {pos} (validated on {valid}) ==")
        print(f"  MAE {m['mae']:.3f}  (baselines: last-5 mean {m['mae_baseline_last5']:.3f}, season ppg {m['mae_baseline_ppg']:.3f}, zero {m['mae_baseline_zero']:.3f})")
        print(f"  MAE regular starters {m['mae_starters']:.3f} vs last-5 baseline {m['mae_starters_baseline_last5']:.3f}")
        print(f"  actual points of the model's top-10 each GW: {m['top10_per_gw_actual_mean']:.2f} vs {m['top10_per_gw_actual_mean_baseline_last5']:.2f} for the last-5 baseline")
        print(f"  spearman {m['spearman']:.3f}  best_iteration {m['best_iteration']}  form-feature gain share {m['form_feature_gain_share']:.0%}")
        print("  top features: " + ", ".join(f"{k}={v:.3f}" for k, v in m["top_features"].items()))


def cmd_predict(conn, args) -> None:
    from pipeline.models.predict import run_predict

    season = _live_season(conn)
    res = run_predict(conn, season, positions=tuple(args.positions), horizon=args.horizon, as_of_gw=args.as_of_gw)
    print(f"PREDICT OK: season={season} as_of_gw={res.as_of_gw} targets={res.target_gws} rows={res.n_rows} model={res.model_version}")
    p = res.predictions
    teams = pd.DataFrame(conn.execute("SELECT id, short_name FROM teams WHERE season=%s", (season,)).fetchall())
    p = p.merge(teams.rename(columns={"id": "team_id", "short_name": "team"}), on="team_id", how="left")
    nxt = res.target_gws[0]
    for pos in args.positions:
        top = p[(p.gw == nxt) & (p.position == pos)].sort_values("pred", ascending=False).head(args.top)
        print(f"\n== top {args.top} {pos} for GW{nxt} ==")
        print(top[["web_name", "team", "price", "pred", "pred_low", "pred_high", "fdr", "is_home", "n_fixtures", "status", "chance_of_playing_next_round"]]
              .to_string(index=False))
    # 5-gw horizon leaders (weighted)
    p["wpts"] = p["pred"] * p["horizon_weight"]
    agg = p.groupby(["web_name", "position", "team"], as_index=False).agg(pts_5gw=("pred", "sum"), weighted=("wpts", "sum"), price=("price", "first"))
    for pos in args.positions:
        print(f"\n== {pos}: top {args.top} over GW{res.target_gws[0]}-{res.target_gws[-1]} (sum of predicted points) ==")
        print(agg[agg.position == pos].sort_values("pts_5gw", ascending=False).head(args.top).round(2).to_string(index=False))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vantage")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ing = sub.add_parser("ingest", help="pull FPL API data into Postgres")
    p_ing.add_argument("--quick", action="store_true", help="skip element-summary histories")
    p_ing.add_argument("--understat", action="store_true", help="force the optional Understat scrape on")

    p_hist = sub.add_parser("history", help="load past seasons from the vaastav archive")
    p_hist.add_argument("--seasons", nargs="+", default=list(DEFAULT_SEASONS))

    p_tr = sub.add_parser("train", help="evaluate models on a held-out season and log MAE")
    p_tr.add_argument("--positions", nargs="+", default=list(POSITIONS), choices=POSITIONS)
    p_tr.add_argument("--valid-season", default=None, help="e.g. 2025/26 (default: latest full season)")

    p_pr = sub.add_parser("predict", help="refit on all data and write predictions for the next gameweeks")
    p_pr.add_argument("--positions", nargs="+", default=list(POSITIONS), choices=POSITIONS)
    p_pr.add_argument("--horizon", type=int, default=5)
    p_pr.add_argument("--as-of-gw", type=int, default=None, help="pretend only gameweeks <= this are known")
    p_pr.add_argument("--top", type=int, default=15)

    sub.add_parser("counts", help="print row counts")
    p_reset = sub.add_parser("reset-db", help="DROP every table and re-apply the schema")
    p_reset.add_argument("--yes", action="store_true")

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    with get_conn() as conn:
        if args.cmd == "reset-db":
            if not args.yes:
                raise SystemExit("refusing without --yes")
            conn.execute("DROP SCHEMA public CASCADE; CREATE SCHEMA public;")
            apply_schema(conn)
            print("schema reset")
        elif args.cmd == "ingest":
            res = run_ingest(conn, FPLClient(), include_histories=not args.quick,
                             include_understat=True if args.understat else None)
            print("INGEST OK:", res.summary())
            for w in res.ft_replay_warnings:
                print("WARN free-transfer replay:", w)
            for n in res.notes:
                print("NOTE:", n)
        elif args.cmd == "history":
            apply_schema(conn)
            counts = load_history(conn, tuple(args.seasons))
            print("HISTORY OK:", counts)
        elif args.cmd == "train":
            cmd_train(conn, args)
        elif args.cmd == "predict":
            cmd_predict(conn, args)
        if args.cmd in ("ingest", "history", "counts", "reset-db"):
            for table, n in db_counts(conn).items():
                print(f"{table:24s} {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
