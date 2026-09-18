# FPL AI Assistant — Build Spec

Hand this file to Claude Code (desktop app) with a prompt like:
"Read fpl-ai-spec.md in this repo and build Phase 1. Ask me only for the credentials marked [USER PROVIDES], proceed with everything else."

---

## User Inputs

**⚠️ Security note**: this file contains a live database password. Do NOT commit this file to the GitHub repo as-is. Keep it local and feed it to Claude Code from your machine, or strip the password out and replace it with `<see .env>` once your `.env` / Vercel / GitHub Actions secrets are set up. Rotate the Supabase password if this file (or the chat it came from) is ever exposed.

- **FPL team ID**: `3964630`
- **FPL login email** (only needed for Phase 8 auto-submit): `senpurple420@gmail.com` — password not stored here; Claude Code should prompt for it interactively only when Phase 8 is actually reached, and it should go straight into a secret store, never into a file.
- **GitHub repo URL**: `https://github.com/Sen-Is-Dead/Vantage`
- **Vercel project**: `https://vercel.com/sen-is-dead-s-projects/vantage`
- **Supabase project**: `https://pphyqxqqondnmwcorwsd.supabase.co`
  - Direct connection (use for local scripts / GitHub Actions weekly job): `<see .env: DATABASE_URL>`
  - Pooled/pgbouncer connection (use for the Vercel dashboard's serverless functions): grab this from Supabase → Project Settings → Database → Connection string → "Connection pooling" (port 6543). Not yet supplied, get this before Phase 7.
  - Public API credentials (safe to expose client-side, unlike the password/connection strings above, since these are permission-scoped by Supabase's Row Level Security rather than granting raw DB access):
    - `NEXT_PUBLIC_SUPABASE_URL=https://pphyqxqqondnmwcorwsd.supabase.co`
    - `NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY=<see .env>`
    - These are only meaningful if the dashboard reads via the Supabase client SDK rather than a raw Postgres connection; if Claude Code builds the dashboard against `DATABASE_URL_POOLED` directly instead, these two aren't needed. Decide one approach and don't mix both.
    - Even though these are "public-safe," still set them as env vars (Vercel Environment Variables, `.env` locally) rather than pasting into component code, so rotating them later doesn't mean hunting through the codebase.
  - Store the sensitive ones as `DATABASE_URL` (direct) and `DATABASE_URL_POOLED` (pooled) env vars, never hard-coded.
- **Captain risk preference**: let the model decide based on current league/rank position (more aggressive/differential picks when behind, safer floor picks when protecting a good rank)
- **Protected players** (never recommend selling unless injured/suspended; if the optimizer's math ever suggests it anyway, surface a prominent, unmissable warning rather than silently doing it): **Bryan Mbeumo**
- **Current squad state**: 0 chips used this season; 2 free transfers currently banked

---

## Safety Rules (non-negotiable, applies to every phase)

1. **No live submission without explicit confirmation.** The system only ever *recommends* transfers, captain, chips, and lineup. It must never call FPL's authenticated write-endpoints automatically. Every recommendation is displayed on the dashboard (or CLI) and requires the user to click/type an explicit final confirmation before anything is submitted to FPL's live servers. This applies even after Phase 8 is built, no "auto-pilot" mode without a confirmation step in front of every submission.
2. **Protected player list is a hard override.** Before finalizing any transfer-out recommendation, check it against the protected players list above. If a protected player is flagged for transfer out and isn't actually injured/suspended per the FPL API's own status field, don't just quietly recommend it, block it or wrap it in a clearly visible warning the user can't miss.
3. **Never spend a transfer or use a chip on the user's behalf.** The system tracks free transfers and chip state for planning purposes only; the actual transfer/chip usage on FPL's servers happens only when the user confirms (rule 1).

---

## 0. Goal

Build a system that:
1. Pulls FPL (Fantasy Premier League) data + supplementary stats weekly.
2. Predicts each player's expected points for the next 1, 3, and 5 gameweeks.
3. Runs an optimizer to recommend: transfers, starting XI, bench order, captain/vice-captain, and chip timing (Wildcard, Free Hit, Bench Boost, Triple Captain).
4. Retrains weekly as new results come in and logs prediction accuracy over time.
5. Displays recommendations on a small hosted dashboard.
6. (Later phase, optional) Submits transfers automatically.

Human-in-the-loop for the first month minimum: the system recommends, the user approves manually before any live transfer is submitted.

---

## 1. Architecture

```
fpl-ai/
├── pipeline/                  # Python: data + ML + optimizer
│   ├── ingest/
│   │   ├── fpl_api.py         # bootstrap-static, fixtures, element-summary, event/live
│   │   └── understat.py       # xG/xA supplementary data
│   ├── features/
│   │   └── build_features.py  # rolling form, fixture difficulty, minutes reliability
│   ├── models/
│   │   ├── train.py           # LightGBM, one model per position
│   │   └── predict.py         # outputs expected points per player per GW
│   ├── optimize/
│   │   └── solver.py          # PuLP/OR-Tools: transfers, XI, captain, chips
│   ├── db/
│   │   └── schema.sql         # Postgres schema (see section 3)
│   └── run_weekly.py          # single entrypoint the GitHub Action calls
├── dashboard/                 # Next.js app, deployed to Vercel
│   ├── app/
│   ├── lib/db.ts              # reads from hosted Postgres
│   └── ...
├── .github/workflows/
│   └── weekly.yml             # cron-triggered pipeline run
├── .env.example
└── README.md
```

**Stack**
- Data/ML: Python 3.11, pandas, requests, LightGBM, scikit-learn
- Optimization: PuLP (CBC solver, free) — switch to OR-Tools CP-SAT only if PuLP is too slow
- DB: Postgres, hosted on Supabase or Neon (free tier)
- Automation: GitHub Actions (weekly cron, e.g. Tuesday 3am UTC after Monday night deadline processing)
- Dashboard: Next.js, deployed on Vercel, reads DB read-only
- No SQLite in production (Vercel functions are stateless); SQLite is fine for local dev only

---

## 2. Credentials / accounts needed

| Item | Who provides | Notes |
|---|---|---|
| GitHub repo + `gh auth login` | [USER PROVIDES] | already created: `Sen-Is-Dead/Vantage` |
| Vercel account + `vercel login` | [USER PROVIDES] | already created: `sen-is-dead-s-projects/vantage` |
| Supabase Postgres connection string | [USER PROVIDES] | see User Inputs above; store as env var, not in code |
| FPL team ID | [USER PROVIDES] | `3964630` |
| FPL session cookie (only needed for auto-submit, Phase 4+) | [USER PROVIDES] | treat as a secret; store in GitHub Actions secrets, never commit |
| Understat access | none needed | public scrape, no auth |

**Where secrets actually live (not in the repo, not in this file long-term):**
- Local dev: a `.env` file at the repo root, listed in `.gitignore` from the very first commit.
- GitHub Actions (weekly job): repo → Settings → Secrets and variables → Actions → add `DATABASE_URL` there; the workflow reads it as `${{ secrets.DATABASE_URL }}`.
- Vercel (dashboard): Project → Settings → Environment Variables → add `DATABASE_URL_POOLED` there for the deployed app to read at runtime.
- Claude Code should create a `.env.example` with the variable *names* only (no real values) so the repo documents what's needed without exposing anything.

Claude Code should stop and ask for exactly these, nothing else, and store them as environment variables / GitHub secrets, never hard-coded.

---

## 3. Database schema (core tables)

```sql
players (id, web_name, team_id, position, price, status, news)
teams (id, name, strength_overall, strength_home, strength_away)
fixtures (id, gw, home_team_id, away_team_id, kickoff_time, difficulty_home, difficulty_away)
player_gw_stats (player_id, gw, minutes, points, xg, xa, goals, assists, clean_sheet, bonus, ...)
predictions (player_id, gw, predicted_points, predicted_points_low, predicted_points_high, model_version)
model_runs (id, run_date, model_version, mae, notes)
recommendations (id, gw, run_date, squad_json, transfers_json, captain_id, vice_captain_id, chip_used, expected_points_gain)
user_squad (gw, player_id, is_starting, is_captain, is_vice_captain, bench_order)
```

Keep raw ingested data (player_gw_stats, fixtures) separate from derived data (predictions, recommendations) so the model can always be retrained from scratch against history.

---

## 4. Feature set (v1)

Per player, per gameweek, at prediction time (i.e. only using data available *before* that GW):
- **Recent form is a priority signal, not just one feature among many.** Give the model multiple recency windows (last 3 / 5 / 10 GWs) rather than a single season-long average, and make the short windows (last 3) at least as prominent as longer ones, since a player's current run of form matters more than their season-long baseline. Consider adding an explicit recency-weighted average (e.g. exponential decay, most recent GW weighted highest) alongside the plain rolling averages, and check feature importance after training to confirm form-related features are actually pulling weight, not being drowned out by static features like price or team strength.
- Rolling points, last 3 / 5 / 10 GWs (recency-weighted variant included, per above)
- Rolling xG, xA, last 3 / 5 / 10 GWs (recency-weighted variant included, per above)
- Minutes played, last 5 GWs (rotation risk proxy)
- Fixture difficulty for the upcoming GW (start with FPL's own FDR, upgrade later to a custom Elo built from Understat team xG)
- Home/away
- Price and season-to-date price change (proxy for form/ownership momentum)
- Team's attacking/defensive strength rating
- Days since last match (fixture congestion)
- `chance_of_playing_next_round` from FPL API

Target: actual points scored that GW.

Train one LightGBM regressor per position (GKP, DEF, MID, FWD), since scoring rules differ by position.

---

## 5. Optimizer constraints (transfer + squad selection)

Standard FPL rules to encode:
- 15-player squad: 2 GKP, 5 DEF, 5 MID, 3 FWD
- Starting XI: 1 GKP, minimum 3 DEF, minimum 2 MID, minimum 1 FWD, exactly 11 total
- Max 3 players from any one real-world team
- Budget: total squad value ≤ current bank + squad value
- Free transfers: 1 per week (banks up to 5 from 2024/25 rules — confirm current season's rule), each extra transfer costs 4 points
- Captain: 2x points on one starting player; Vice-captain: 2x points if captain doesn't play
- Chips: Wildcard (unlimited free transfers, once per half-season), Free Hit (one-week unlimited transfers, squad reverts after), Bench Boost (bench points count that week), Triple Captain (captain scores 3x instead of 2x)

Objective: maximize sum of predicted points over the lookahead horizon (weight nearer gameweeks more heavily, e.g. GW1 full weight, GW2 at 0.7, GW3 at 0.5), minus transfer point costs, subject to the above.

For chip timing: run the optimizer with the chip forced "on" for each candidate week in the lookahead window vs "off", compare expected point gain, recommend the chip for whichever week gives the biggest edge. Don't try to jointly optimize all chips at once in v1, evaluate them one at a time against the baseline.

---

## 6. Weekly retrain + backtesting

- Every run: append newest GW's actual results to `player_gw_stats`, retrain models on all available history, log MAE per position to `model_runs`.
- Backtest harness (build this before trusting the system): replay the last 2-3 seasons GW by GW, feeding the model only data available at that point in time, and compare the resulting recommended squad's actual points against what the user's real historical squad scored. This is the sanity check before going live.
- Don't call the retraining "reinforcement learning" in code/comments, it's supervised retraining with a growing dataset. If a genuine bandit-style component is added later (e.g. tuning captain risk-appetite based on outcomes), keep it as a small separate module, not baked into the core predictor.

---

## 7. Build phases (do these in order, confirm before moving to next)

**Phase 1**: Repo scaffold, DB schema, FPL API ingestion working, raw data flowing into Postgres.
**Phase 2**: Feature pipeline + LightGBM baseline model for one position, sanity-check predictions against known good players.
**Phase 3**: Extend model to all positions, build backtest harness, report historical accuracy.
**Phase 4**: Optimizer (transfers, XI, captain) on top of predictions.
**Phase 5**: Chip timing logic.
**Phase 6**: GitHub Actions weekly automation writing to Postgres.
**Phase 7**: Next.js dashboard on Vercel, read-only display of recommendations.
**Phase 8 (optional, later)**: Authenticated FPL write access for auto-submitting transfers, only after Phase 1-7 have run successfully for several real gameweeks.

---

## 8. Open questions for Claude Code to flag, not guess

- Confirm current season's free-transfer banking rule (has changed between seasons).
- Confirm Understat's current scraping terms/robustness; if it's fragile, fall back to FPL data alone plus a simpler fixture-difficulty proxy.
- Ask the user for their risk preference on captaincy (safe floor vs high ceiling) before defaulting the optimizer's objective.
