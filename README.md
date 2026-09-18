# Vantage: FPL AI assistant

Pulls Fantasy Premier League data weekly, predicts expected points per player for the next 1/3/5
gameweeks, and recommends transfers, starting XI, captain and chip timing. Full spec: `fpl-ai-spec.md`.

**Safety rules (non-negotiable, see the spec):** the system only ever *recommends*. No code path
submits transfers, lineups or chips to FPL without an explicit user confirmation step, and protected
players (currently Bryan Mbeumo) are never recommended out unless the FPL API marks them
injured/suspended/unavailable.

## Layout

```
pipeline/            Python 3.11: ingest -> features -> models -> optimizer
  config.py          env-driven settings (team id, protected players, transfer rules)
  db/schema.sql      Postgres schema (raw tables separate from derived tables)
  ingest/fpl_api.py  FPL public API client + pure transforms (GET only)
  ingest/understat.py optional supplementary xG (off by default)
  ingest/load.py     one idempotent ingestion run
  run_weekly.py      CLI entrypoint used by GitHub Actions
dashboard/           Next.js on Vercel (Phase 7)
.github/workflows/   ingest.yml (manual), weekly.yml (Phase 6 cron)
tests/               pytest; fixtures mirror real FPL API shapes
```

## Setup

```bash
python -m venv .venv && . .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                 # then fill in DATABASE_URL (never commit .env)
python -m pipeline.run_weekly ingest --quick         # ~10s smoke test: teams, players, fixtures, your squad
python -m pipeline.run_weekly ingest                 # full: + per-player gameweek histories (~700 requests, 2-5 min)
python -m pipeline.run_weekly counts
```

Tests (unit tests need nothing; the DB test needs any throwaway Postgres):

```bash
TEST_DATABASE_URL=postgresql://postgres:pw@localhost:5432/vantage_test python -m pytest -q
```

Never point `TEST_DATABASE_URL` at Supabase: the DB test drops and recreates the `public` schema.

## Where secrets live

| Where | Variable | Used by |
|---|---|---|
| `.env` (gitignored) | `DATABASE_URL` (Supabase *Session pooler*, port 5432), `DATABASE_URL_POOLED` (transaction pooler, 6543), `FPL_TEAM_ID`, `PROTECTED_PLAYERS` | local runs |
| GitHub Actions secrets | `DATABASE_URL` | `ingest.yml`, `weekly.yml` |
| Vercel env vars | `DATABASE_URL_POOLED` | dashboard (Phase 7) |

`.env.example` lists the names only. Use the pooler hosts, not `db.<ref>.supabase.co`: that direct host is IPv6-only and fails DNS on IPv4-only networks (home ISPs, GitHub Actions). If the Supabase password is ever exposed, rotate it in
Supabase -> Project Settings -> Database and update the three places above.

## Free-transfer bookkeeping

The public FPL API does not expose the banked free-transfer count, so `derive_free_transfers`
replays it from your transfer history (1 FT per GW, bank up to 5, wildcard/free hit retain the
bank; confirmed for 2026/27 via `game_settings.max_extra_free_transfers = 4`). The ingest run
prints a warning if the replay disagrees with the hits FPL actually charged you.
