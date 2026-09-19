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
  ingest/history.py  past seasons from the vaastav/Fantasy-Premier-League archive
  features/build_features.py  point-in-time features (rolling 3/5/10 + EWM form, fixtures, price, priors)
  models/train.py    one LightGBM regressor per position (+ q20/q80 quantile models), time-ordered eval
  models/predict.py  refit on everything, predict next 1..5 GWs, apply availability
  optimize/solver.py PuLP/CBC squad + transfer optimiser (15/XI/captain/hits, budget, 3-per-club)
  optimize/recommend.py weekly recommendation: horizon-weighted transfers, XI, bench, captain; protected-player lock
  optimize/chips.py  chip timing: each chip alone, per candidate GW vs no-chip baseline, thresholds
  backtest.py        replay a past season GW by GW with point-in-time refits; model vs naive manager
  run_weekly.py      CLI entrypoint used by GitHub Actions
dashboard/           Next.js 16 (App Router, server components) on Vercel; reads Postgres via DATABASE_URL_POOLED, read-only
.github/workflows/   weekly.yml (Tuesday 03:00 UTC cron + manual), ingest.yml (manual ingest only)
tests/               pytest; fixtures mirror real FPL API shapes
```

## Setup

```bash
python -m venv .venv && . .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                 # then fill in DATABASE_URL (never commit .env)
python -m pipeline.run_weekly ingest --quick         # ~10s smoke test: teams, players, fixtures, your squad
python -m pipeline.run_weekly ingest                 # full: + per-player gameweek histories (~700 requests, 2-5 min)
python -m pipeline.run_weekly history                # 4 past seasons (~25 MB download), needed to train anything
python -m pipeline.run_weekly train --positions MID  # evaluate on held-out 2025/26, log MAE to model_runs
python -m pipeline.run_weekly predict --positions MID  # refit on all data, write predictions for the next 5 GWs
python -m pipeline.run_weekly recommend               # transfers / XI / captain for YOUR squad -> recommendations table
python -m pipeline.run_weekly backtest --season 2025/26  # replay a season: model vs naive last-5 manager
python -m pipeline.run_weekly all                     # the weekly job (what GitHub Actions runs every Tuesday 03:00 UTC)
python -m pipeline.run_weekly counts
```

`reset-db --yes` drops every table (all data is regenerable from the FPL API and the archive).

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

## Backtest results (v1, no chips, GW2-38, buy = sell price)

| season | model | naive last-5 manager | Manjit's real total (38 GWs, with chips) |
|---|---|---|---|
| 2024/25 | 2160 | 2090 | 2327 (top 11%) |
| 2025/26 | 1976 | 1709 | 2256 (top 3%) |

Free-hit-every-week upper bound for 2025/26 (unlimited transfers): 2054. The model beats a naive
manager comfortably but does not yet beat a strong human season; the gap is prediction quality,
not the optimiser. The backtest has no injury information (a live run does), no chips (Phase 5),
and ignores GW1.

## Weekly automation

`.github/workflows/weekly.yml` runs the test suite against a throwaway Postgres, then
`python -m pipeline.run_weekly all` against Supabase: ingest the latest gameweek, load the archive
if the table is empty, score last week's predictions into `prediction_accuracy`, evaluate the models
on a held-out season (`model_runs`), refit on everything and write `predictions` for the next five
gameweeks, and write one `recommendations` row. The recommendation is copied into the run's job
summary (Actions tab) and the full log is kept as an artifact for 30 days. Nothing is ever
submitted to FPL; you read the recommendation and act on it yourself.

## Dashboard (Vercel)

`dashboard/` is a Next.js app that reads the same Postgres tables (read-only) and shows the latest
recommendation (transfers, XI, bench, captain, chips, warnings), the prediction table, and the
accuracy / model-run / backtest history. It talks to Postgres directly with `pg` through
`DATABASE_URL_POOLED` (Supabase transaction pooler, port 6543); the Supabase client SDK and the
`NEXT_PUBLIC_*` keys are not used, so nothing database-related ever reaches the browser.

Deploy: in the Vercel project set **Root Directory = `dashboard`** (framework auto-detects Next.js),
add the environment variable `DATABASE_URL_POOLED`, and connect the GitHub repo; every push to
`main` redeploys. Pages are rendered on demand, so the Tuesday pipeline run shows up on the next
page load with no redeploy.

Local: `cd dashboard && npm install && DATABASE_URL_POOLED=... npm run dev` (or put it in the
repo-root `.env`, which `next dev` does not read; export it in the shell instead).
