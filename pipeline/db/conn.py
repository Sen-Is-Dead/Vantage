"""Thin psycopg3 helpers. One direct connection per pipeline run; no ORM."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import psycopg
from psycopg import sql
from psycopg.rows import dict_row

from pipeline.config import database_url

SCHEMA_PATH = Path(__file__).with_name("schema.sql")


@contextmanager
def get_conn(url: str | None = None) -> Iterator[psycopg.Connection]:
    url = url or database_url()
    try:
        conn = psycopg.connect(url, row_factory=dict_row, connect_timeout=20)
    except psycopg.OperationalError as exc:
        if "resolve host" in str(exc) and ".supabase.co" in url:
            raise RuntimeError(
                "Cannot resolve the Supabase direct host. Supabase's db.<ref>.supabase.co hosts are IPv6-only; "
                "on an IPv4-only network (most home ISPs, GitHub Actions runners) use the *Session pooler* string "
                "instead: Supabase -> Connect -> Session pooler "
                "(postgresql://postgres.<ref>:<password>@aws-0-<region>.pooler.supabase.com:5432/postgres)."
            ) from exc
        raise
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def apply_schema(conn: psycopg.Connection) -> None:
    conn.execute(SCHEMA_PATH.read_text())


def upsert(
    conn: psycopg.Connection,
    table: str,
    rows: Sequence[dict[str, Any]],
    key_cols: Iterable[str],
    update_cols: Iterable[str] | None = None,
) -> int:
    """INSERT ... ON CONFLICT (key) DO UPDATE for a list of dicts sharing the same keys.
    Returns the number of rows sent."""
    if not rows:
        return 0
    cols = list(rows[0].keys())
    key_cols = list(key_cols)
    update_cols = [c for c in (update_cols if update_cols is not None else cols) if c not in key_cols]
    if update_cols:
        action = sql.SQL("DO UPDATE SET {sets}").format(
            sets=sql.SQL(", ").join(sql.SQL("{c} = EXCLUDED.{c}").format(c=sql.Identifier(c)) for c in update_cols)
        )
    else:
        action = sql.SQL("DO NOTHING")
    col_list = sql.SQL(", ").join(map(sql.Identifier, cols))
    keys = sql.SQL(", ").join(map(sql.Identifier, key_cols))
    with conn.cursor() as cur:
        if len(rows) < COPY_THRESHOLD:
            stmt = sql.SQL("INSERT INTO {t} ({cols}) VALUES ({vals}) ON CONFLICT ({keys}) {action}").format(
                t=sql.Identifier(table), cols=col_list,
                vals=sql.SQL(", ").join(sql.Placeholder(c) for c in cols), keys=keys, action=action,
            )
            cur.executemany(stmt, rows)
        else:
            # big batches (historical seasons): COPY into a temp table, then one INSERT ... ON CONFLICT.
            # Orders of magnitude faster than executemany over a remote (Supabase) connection.
            tmp = sql.Identifier(f"_tmp_{table}")
            cur.execute(sql.SQL("CREATE TEMP TABLE {tmp} (LIKE {t} INCLUDING DEFAULTS) ON COMMIT DROP").format(tmp=tmp, t=sql.Identifier(table)))
            with cur.copy(sql.SQL("COPY {tmp} ({cols}) FROM STDIN").format(tmp=tmp, cols=col_list)) as cp:
                for r in rows:
                    cp.write_row([_copy_value(r[c]) for c in cols])
            cur.execute(sql.SQL("INSERT INTO {t} ({cols}) SELECT {cols} FROM {tmp} ON CONFLICT ({keys}) {action}").format(
                t=sql.Identifier(table), cols=col_list, tmp=tmp, keys=keys, action=action))
    return len(rows)


COPY_THRESHOLD = 2000


def _copy_value(v: Any) -> Any:
    """COPY needs plain Python types; numpy scalars and NaN become native / NULL."""
    if v is None:
        return None
    if hasattr(v, "item"):  # numpy scalar
        v = v.item()
    if isinstance(v, float) and v != v:  # NaN
        return None
    return v


def scalar(conn: psycopg.Connection, query: str, params: Sequence[Any] | None = None) -> Any:
    row = conn.execute(query, params).fetchone()
    return None if row is None else next(iter(row.values()))
