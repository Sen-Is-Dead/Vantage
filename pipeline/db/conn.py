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
    conn = psycopg.connect(url or database_url(), row_factory=dict_row, connect_timeout=20)
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
    stmt = sql.SQL("INSERT INTO {t} ({cols}) VALUES ({vals}) ON CONFLICT ({keys}) {action}").format(
        t=sql.Identifier(table),
        cols=sql.SQL(", ").join(map(sql.Identifier, cols)),
        vals=sql.SQL(", ").join(sql.Placeholder(c) for c in cols),
        keys=sql.SQL(", ").join(map(sql.Identifier, key_cols)),
        action=action,
    )
    with conn.cursor() as cur:
        cur.executemany(stmt, rows)
    return len(rows)


def scalar(conn: psycopg.Connection, query: str, params: Sequence[Any] | None = None) -> Any:
    row = conn.execute(query, params).fetchone()
    return None if row is None else next(iter(row.values()))
