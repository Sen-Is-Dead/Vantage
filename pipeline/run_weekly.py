"""Single entrypoint the GitHub Action (and you, locally) calls.

    python -m pipeline.run_weekly ingest [--quick]        # Phase 1: FPL API -> Postgres
    python -m pipeline.run_weekly history [--seasons ...] # Phase 2: past seasons (vaastav archive) -> Postgres
    python -m pipeline.run_weekly train [--positions MID] # evaluate on a held-out season, log MAE to model_runs
    python -m pipeline.run_weekly predict [--positions MID] [--horizon 5]   # refit on everything, write predictions
    python -m pipeline.run_weekly recommend [--horizon 5]  # transfers / XI / captain for the next GW -> recommendations
    python -m pipeline.run_weekly backtest --season 2025/26 [--mode realistic|free_hit] [--strategy model|naive_last5]
    python -m pipeline.run_weekly all                      # Phase 6: the weekly job (ingest, history if empty,
                                                           #   accuracy, train-eval, predict, recommend)
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


def cmd_recommend(conn, args) -> None:
    from pipeline.optimize.recommend import recommend

    season = _live_season(conn)
    rec = recommend(conn, season, horizon=args.horizon, free_transfers_override=args.free_transfers)
    print(rec.report())


def cmd_backtest(conn, args) -> None:
    from pipeline.backtest import run_backtest

    results = []
    for strategy in args.strategies:
        r = run_backtest(conn, args.season, mode=args.mode, strategy=strategy, start_gw=args.start_gw,
                         end_gw=args.end_gw, refit_every=args.refit_every, rounds=args.rounds, horizon=args.horizon)
        s = r.summary()
        results.append(s)
        print(f"\nBACKTEST {args.season} mode={args.mode} strategy={strategy}: {s['total_points']:.0f} pts over GW{args.start_gw}-{args.end_gw} "
              f"({s['avg_per_gw']}/gw), captain {s['captain_points']:.0f}, {s['transfers_total']} transfers, {s['hits_total']} hits, "
              f"{s['bench_points']:.0f} left on bench")
    # the user's real total for that season, if the FPL API knows it
    try:
        from pipeline import config
        from pipeline.ingest.fpl_api import FPLClient
        past = FPLClient().entry_history(config.FPL_TEAM_ID).get("past", [])
        mine = next((p for p in past if p.get("season_name") == args.season), None)
        if mine:
            print(f"Your real {args.season}: {mine['total_points']} pts (rank {mine['rank']:,}) over all 38 GWs")
    except Exception as exc:  # noqa: BLE001
        print(f"(could not fetch your real season total: {exc})")
    return results


def cmd_all(conn, args) -> int:
    """The weekly job. Each step logs; a failure in accuracy/train-eval is reported but does not stop
    predict + recommend, because a stale recommendation is worse than a missing MAE line."""
    from types import SimpleNamespace

    from pipeline.db.conn import scalar
    from pipeline.models.accuracy import compute_accuracy

    failures: list[str] = []
    print("== 1/6 ingest ==")
    res = run_ingest(conn, FPLClient(), include_histories=not args.quick)
    print("INGEST OK:", res.summary())
    for w in res.ft_replay_warnings:
        print("WARN free-transfer replay:", w)
    season = res.season

    print("== 2/6 history ==")
    n_hist = scalar(conn, "SELECT count(*) FROM player_gw_stats WHERE source='vaastav'")
    if n_hist == 0:
        print("HISTORY:", load_history(conn, tuple(DEFAULT_SEASONS)))
    else:
        print(f"history already loaded ({n_hist} archive rows); skipping")

    print("== 3/6 accuracy of past predictions ==")
    try:
        acc = compute_accuracy(conn, season)
        if acc.empty:
            print("no finished gameweeks with predictions yet")
        else:
            print(acc[acc.position == "ALL"][["gw", "model_version", "n", "mae", "rmse"]].round(3).to_string(index=False))
    except Exception as exc:  # noqa: BLE001
        failures.append(f"accuracy: {exc}")
        print("accuracy failed:", exc)

    print("== 4/6 train (held-out evaluation, logged to model_runs) ==")
    try:
        cmd_train(conn, SimpleNamespace(positions=list(POSITIONS), valid_season=None))
    except Exception as exc:  # noqa: BLE001
        failures.append(f"train: {exc}")
        print("train-eval failed:", exc)

    print("== 5/6 predict ==")
    cmd_predict(conn, SimpleNamespace(positions=list(POSITIONS), horizon=5, as_of_gw=None, top=5))

    print("== 6/6 recommend ==")
    cmd_recommend(conn, SimpleNamespace(horizon=5, free_transfers=None))

    if failures:
        print("\nNON-FATAL FAILURES:", "; ".join(failures))
    print("\nWEEKLY RUN COMPLETE. Nothing was submitted to FPL.")
    return 0


def main(argv: list[str] | None = None) -> int:
    # Windows consoles default to cp1252, which cannot print names like "Kovačić"
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
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

    p_rc = sub.add_parser("recommend", help="transfers / XI / captain for the next gameweek")
    p_rc.add_argument("--horizon", type=int, default=5)
    p_rc.add_argument("--free-transfers", type=int, default=None, help="override the replayed free-transfer count")

    p_bt = sub.add_parser("backtest", help="replay a past season gameweek by gameweek")
    p_bt.add_argument("--season", required=True, help="e.g. 2025/26")
    p_bt.add_argument("--mode", default="realistic", choices=["realistic", "free_hit"])
    p_bt.add_argument("--strategies", nargs="+", default=["model", "naive_last5"])
    p_bt.add_argument("--start-gw", type=int, default=2)
    p_bt.add_argument("--end-gw", type=int, default=38)
    p_bt.add_argument("--refit-every", type=int, default=4)
    p_bt.add_argument("--rounds", type=int, default=300)
    p_bt.add_argument("--horizon", type=int, default=5, help="gameweeks of look-ahead used to value transfers")

    p_all = sub.add_parser("all", help="the weekly job: ingest, history, accuracy, train-eval, predict, recommend")
    p_all.add_argument("--quick", action="store_true", help="skip element-summary histories (smoke test only)")

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
        elif args.cmd == "recommend":
            apply_schema(conn)
            cmd_recommend(conn, args)
        elif args.cmd == "backtest":
            apply_schema(conn)
            cmd_backtest(conn, args)
        elif args.cmd == "all":
            apply_schema(conn)
            cmd_all(conn, args)
        if args.cmd in ("ingest", "history", "counts", "reset-db"):
            for table, n in db_counts(conn).items():
                print(f"{table:24s} {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
