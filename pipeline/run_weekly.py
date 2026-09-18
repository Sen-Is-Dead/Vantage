"""Single entrypoint the GitHub Action (and you, locally) calls.

    python -m pipeline.run_weekly ingest            # Phase 1: pull FPL data into Postgres
    python -m pipeline.run_weekly ingest --quick    # skip per-player histories (fast smoke test)
    python -m pipeline.run_weekly counts            # print row counts per table

Later phases add: features, train, predict, optimize, all. This script only ever READS from FPL
and WRITES to our own Postgres. It never submits anything to FPL (Safety Rule 1).
"""
from __future__ import annotations

import argparse
import logging
import sys

from pipeline.db.conn import get_conn
from pipeline.ingest.fpl_api import FPLClient
from pipeline.ingest.load import db_counts, run_ingest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vantage")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_ing = sub.add_parser("ingest", help="pull FPL API data into Postgres")
    p_ing.add_argument("--quick", action="store_true", help="skip element-summary histories")
    p_ing.add_argument("--understat", action="store_true", help="force the optional Understat scrape on")
    sub.add_parser("counts", help="print row counts")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    with get_conn() as conn:
        if args.cmd == "ingest":
            res = run_ingest(conn, FPLClient(), include_histories=not args.quick,
                             include_understat=True if args.understat else None)
            print("INGEST OK:", res.summary())
            for w in res.ft_replay_warnings:
                print("WARN free-transfer replay:", w)
            for n in res.notes:
                print("NOTE:", n)
        if args.cmd in ("ingest", "counts"):
            for table, n in db_counts(conn).items():
                print(f"{table:24s} {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
