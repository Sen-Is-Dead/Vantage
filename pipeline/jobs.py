"""Job runner: the dashboard inserts a row in `jobs` and fires a GitHub Actions workflow_dispatch;
that run calls `python -m pipeline.run_weekly job --id N` and this module executes it, streaming
status, progress and a log tail back into the row so the dashboard can show what is happening.

Three kinds, none of which can submit anything to FPL (Safety Rule 1 is unchanged):

    refresh    ingest -> predict -> recommend. The "don't wait until Tuesday" button.
    weekly     the full Tuesday job (adds history, accuracy and the held-out training evaluation).
    simulate   a parameterised season replay, see pipeline/simulate.py.

Status updates go over their OWN connection so that committing progress never commits half-finished
pipeline work on the main one.
"""
from __future__ import annotations

import contextlib
import io
import json
import logging
import os
import time
import traceback
from collections import deque
from typing import Any

import psycopg

from pipeline.db.conn import get_conn

log = logging.getLogger(__name__)

LOG_TAIL_LINES = 120
FLUSH_SECONDS = 5.0
VALID_KINDS = ("refresh", "weekly", "simulate")


def run_url() -> str | None:
    server = os.getenv("GITHUB_SERVER_URL")
    repo = os.getenv("GITHUB_REPOSITORY")
    rid = os.getenv("GITHUB_RUN_ID")
    return f"{server}/{repo}/actions/runs/{rid}" if server and repo and rid else None


class JobStatus:
    """Writes progress/log back to the jobs row on a throttle, over its own connection."""

    def __init__(self, job_id: int, url: str | None = None):
        self.job_id = job_id
        self.url = url
        self.lines: deque[str] = deque(maxlen=LOG_TAIL_LINES)
        self._progress = ""
        self._last_flush = 0.0

    def write(self, text: str) -> int:          # stdout tee
        for line in text.splitlines():
            if line.strip():
                self.lines.append(line.rstrip())
        self.flush_maybe()
        return len(text)

    def flush(self) -> None:                    # stdout tee
        pass

    def progress(self, msg: str) -> None:
        self._progress = msg[:500]
        self.lines.append(msg)
        self.flush_maybe(force=True)

    def flush_maybe(self, force: bool = False) -> None:
        now = time.time()
        if not force and now - self._last_flush < FLUSH_SECONDS:
            return
        self._last_flush = now
        with contextlib.suppress(Exception):    # status is best effort; never kill a job over it
            with get_conn() as c:
                c.execute("UPDATE jobs SET progress = %s, log_tail = %s, run_url = coalesce(run_url, %s) WHERE id = %s",
                          (self._progress, "\n".join(self.lines), self.url, self.job_id))


class _Tee(io.TextIOBase):
    def __init__(self, *targets):
        self.targets = targets

    def write(self, text: str) -> int:
        for t in self.targets:
            with contextlib.suppress(Exception):
                t.write(text)
        return len(text)

    def flush(self) -> None:
        for t in self.targets:
            with contextlib.suppress(Exception):
                t.flush()


# ------------------------------------------------------------------------------------------
# the work
# ------------------------------------------------------------------------------------------

def _do_refresh(conn: psycopg.Connection, params: dict, status: JobStatus) -> dict:
    from types import SimpleNamespace

    from pipeline.ingest.fpl_api import FPLClient
    from pipeline.ingest.load import run_ingest
    from pipeline.models.train import POSITIONS
    from pipeline.run_weekly import cmd_predict, cmd_recommend

    horizon = int(params.get("horizon", 5))
    status.progress("ingesting the latest FPL data")
    res = run_ingest(conn, FPLClient(), include_histories=not params.get("quick", False))
    print("INGEST OK:", res.summary())

    status.progress("predicting the next gameweeks")
    cmd_predict(conn, SimpleNamespace(positions=list(POSITIONS), horizon=horizon, as_of_gw=None, top=5))

    status.progress("optimising transfers, XI and captain")
    cmd_recommend(conn, SimpleNamespace(horizon=horizon, free_transfers=params.get("free_transfers")))

    row = conn.execute("SELECT id, gw, expected_points_gw, expected_points_gain FROM recommendations "
                       "ORDER BY run_date DESC LIMIT 1").fetchone()
    return {"kind": "refresh", "season": res.season, "recommendation": row}


def _do_weekly(conn: psycopg.Connection, params: dict, status: JobStatus) -> dict:
    from types import SimpleNamespace

    from pipeline.run_weekly import cmd_all

    status.progress("running the full weekly job")
    cmd_all(conn, SimpleNamespace(quick=bool(params.get("quick", False))))
    row = conn.execute("SELECT id, gw, expected_points_gw, expected_points_gain FROM recommendations "
                       "ORDER BY run_date DESC LIMIT 1").fetchone()
    return {"kind": "weekly", "recommendation": row}


def _do_simulate(conn: psycopg.Connection, params: dict, status: JobStatus, job_id: int) -> dict:
    from pipeline.simulate import SimConfig, run_simulation

    seasons = params.pop("seasons", None) or [params.get("season")]
    base = {k: v for k, v in params.items() if k != "season"}
    out = []
    for n, season in enumerate(seasons, 1):
        status.progress(f"simulating {season} ({n}/{len(seasons)})")
        cfg = SimConfig(season=season, **base)
        res = run_simulation(conn, cfg, progress=status.progress, job_id=job_id)
        s = res.summary()
        print(f"\nSIM {season} {cfg.mode}/{cfg.strategy}: {s['total_points_mean']:.0f} pts"
              + (f" (sd {s['total_points_sd']:.0f}, p10-p90 {s['total_points_p10']:.0f}-{s['total_points_p90']:.0f})"
                 if cfg.repeats > 1 else "")
              + f" over GW{cfg.start_gw}-{cfg.end_gw}, {s['transfers_total']} transfers, {s['hits_total']} hits, "
                f"{s['bench_points']:.0f} left on the bench")
        if s["chips"]:
            print("  chips: " + ", ".join(f"{c} GW{g}" for c, g in s["chips"].items()))
        out.append({k: s[k] for k in ("season", "mode", "strategy", "total_points", "total_points_mean",
                                      "total_points_sd", "total_points_p10", "total_points_p90", "avg_per_gw",
                                      "hits_total", "transfers_total", "captain_points", "bench_points",
                                      "chips", "n_repeats", "seconds")})
    return {"kind": "simulate", "runs": out}


HANDLERS = {"refresh": _do_refresh, "weekly": _do_weekly}


def run_job(job_id: int) -> int:
    """Claim, execute and close out one job row. Returns a process exit code."""
    url = run_url()
    status = JobStatus(job_id, url)
    with get_conn() as c:
        row = c.execute("SELECT * FROM jobs WHERE id = %s", (job_id,)).fetchone()
        if row is None:
            raise SystemExit(f"job {job_id} does not exist")
        if row["status"] not in ("queued", "running"):
            raise SystemExit(f"job {job_id} is already {row['status']}; refusing to run it twice")
        c.execute("UPDATE jobs SET status='running', started_at=now(), run_url=%s, progress=%s WHERE id=%s",
                  (url, "starting", job_id))

    kind = row["kind"]
    params: dict[str, Any] = dict(row["params_json"] or {})
    if kind not in VALID_KINDS:
        _finish(job_id, "failed", None, f"unknown job kind {kind!r}")
        return 1

    print(f"== job {job_id}: {kind} ==")
    print("params:", json.dumps(params, sort_keys=True))
    tee = _Tee(__import__("sys").stdout, status)
    try:
        from pipeline.db.conn import apply_schema

        # Apply the schema on a connection of its own that commits straight away. Doing it on the
        # long-lived work connection below would hold ALTER TABLE's ACCESS EXCLUSIVE lock on `jobs`
        # for the whole run, and the first progress update -- which uses a separate connection --
        # would block on it forever.
        with get_conn() as setup:
            apply_schema(setup)

        with contextlib.redirect_stdout(tee), get_conn() as conn:
            if kind == "simulate":
                result = _do_simulate(conn, params, status, job_id)
            else:
                result = HANDLERS[kind](conn, params, status)
    except Exception as exc:  # noqa: BLE001 - the row is the only place the user sees this
        status.lines.append(traceback.format_exc()[-2000:])
        _finish(job_id, "failed", None, f"{type(exc).__name__}: {exc}", status)
        log.exception("job %d failed", job_id)
        return 1
    _finish(job_id, "success", result, None, status)
    print(f"== job {job_id} complete ==")
    return 0


def _finish(job_id: int, state: str, result: dict | None, error: str | None,
            status: JobStatus | None = None) -> None:
    tail = "\n".join(status.lines) if status else None
    with contextlib.suppress(Exception):
        with get_conn() as c:
            c.execute(
                "UPDATE jobs SET status=%s, finished_at=now(), result_json=%s, error=%s, "
                "progress=%s, log_tail=coalesce(%s, log_tail) WHERE id=%s",
                (state, json.dumps(result, default=str) if result else None, (error or "")[:2000] or None,
                 "done" if state == "success" else "failed", tail, job_id),
            )
