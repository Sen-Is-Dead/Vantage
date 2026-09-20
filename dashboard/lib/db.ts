// Postgres access for the dashboard. Uses DATABASE_URL_POOLED (Supabase transaction pooler, port
// 6543) so serverless invocations share pooled connections. Falls back to DATABASE_URL for local
// development. Everything in this file reads; the only writes the dashboard makes anywhere are the
// job rows in lib/jobs.ts, which queue pipeline work. Nothing here can submit to FPL.
import { Pool } from "pg";

declare global {
  // eslint-disable-next-line no-var
  var __vantagePool: Pool | undefined;
}

export function getPool(): Pool {
  if (!global.__vantagePool) {
    const url = process.env.DATABASE_URL_POOLED || process.env.DATABASE_URL;
    if (!url) throw new Error("DATABASE_URL_POOLED is not set (Vercel: Project -> Settings -> Environment Variables)");
    global.__vantagePool = new Pool({
      connectionString: url,
      max: 3,
      idleTimeoutMillis: 10_000,
      ssl: url.includes("supabase") ? { rejectUnauthorized: false } : undefined,
      statement_timeout: 15_000,
    });
  }
  return global.__vantagePool;
}

export async function query<T = Record<string, unknown>>(sql: string, params: unknown[] = []): Promise<T[]> {
  const res = await getPool().query(sql, params);
  return res.rows as T[];
}

// The pipeline owns the schema: `apply_schema` creates tables and adds columns at the start of every
// run. So a dashboard deployed with new features ahead of the first pipeline run is talking to a
// database that does not have them yet. That is a normal, temporary state and must not 500 a page —
// these two codes let the callers below degrade into "run the pipeline once" instead.
const PG_UNDEFINED_TABLE = "42P01";
const PG_UNDEFINED_COLUMN = "42703";

export function isMissingSchema(e: unknown): boolean {
  const code = (e as { code?: string } | null | undefined)?.code;
  return code === PG_UNDEFINED_TABLE || code === PG_UNDEFINED_COLUMN;
}

/** Run a query, but treat "that table/column does not exist yet" as a soft miss. */
export async function queryIfMigrated<T>(fn: () => Promise<T>, fallback: T): Promise<{ data: T; migrated: boolean }> {
  try {
    return { data: await fn(), migrated: true };
  } catch (e) {
    if (isMissingSchema(e)) return { data: fallback, migrated: false };
    throw e;
  }
}

export type SquadPlayer = {
  player_id: number; name: string; team: string; position: "GKP" | "DEF" | "MID" | "FWD";
  price: number; pred: number; low: number; high: number; horizon_value: number;
  is_starting: boolean; is_captain: boolean; is_vice_captain: boolean; bench_order: number | null;
  status: string; chance: number | null; news: string | null;
};
export type Transfer = { out: number; out_name: string; out_price: number; in: number; in_name: string; in_price: number; gain: number };
export type ChipEval = {
  chip: string; label: string; available: boolean; reason: string; by_gw: Record<string, number>;
  best_gw: number | null; best_gain: number; threshold: number; recommend_now: boolean; detail: Record<string, Record<string, unknown>>;
};
export type Recommendation = {
  id: number; season: string; gw: number; run_date: string; model_version: string | null;
  squad_json: SquadPlayer[];
  transfers_json: { transfers: Transfer[]; hits: number; roll_transfer: boolean; free_transfers: number; bank_after: number; budget: number; as_of_gw: number };
  captain_id: number | null; vice_captain_id: number | null; chip_used: string | null;
  chip_plan_json: Record<string, ChipEval> | null;
  expected_points_gain: number | null; expected_points_gw: number | null;
  warnings_json: { warnings: string[]; notes: string[]; risk_appetite: number; protected_locked: string[] } | null;
  status: string;
};

export async function latestRecommendation(): Promise<Recommendation | null> {
  const rows = await query<Recommendation>(
    "SELECT * FROM recommendations ORDER BY run_date DESC LIMIT 1"
  );
  return rows[0] ?? null;
}

export async function recentRecommendations(limit = 10) {
  return query<Pick<Recommendation, "id" | "gw" | "run_date" | "expected_points_gw" | "expected_points_gain" | "chip_used" | "status">>(
    "SELECT id, gw, run_date, expected_points_gw, expected_points_gain, chip_used, status FROM recommendations ORDER BY run_date DESC LIMIT $1",
    [limit]
  );
}

export type EntryState = {
  season: string; gw: number; bank: number; squad_value: number; free_transfers_after: number | null;
  overall_rank: number | null; total_points: number | null; points: number | null; total_players: number | null;
  chips_used_json: { name: string; event: number }[] | null;
};
export async function latestEntryState(): Promise<EntryState | null> {
  const rows = await query<EntryState>("SELECT * FROM user_entry_state ORDER BY gw DESC LIMIT 1");
  return rows[0] ?? null;
}

export type PredictionRow = {
  player_id: number; web_name: string; team: string; position: string; price: number; status: string;
  chance_of_playing_next_round: number | null; gw: number; predicted_points: number; predicted_points_low: number; predicted_points_high: number;
};
export async function predictionsForNextGw(position?: string, limit = 40) {
  const meta = await query<{ season: string; as_of_gw: number; gw: number }>(
    "SELECT season, as_of_gw, min(gw) AS gw FROM predictions WHERE as_of_gw = (SELECT max(as_of_gw) FROM predictions) GROUP BY season, as_of_gw LIMIT 1"
  );
  if (!meta[0]) return { meta: null, rows: [] as PredictionRow[], horizon: [] as { player_id: number; total: number }[] };
  const { season, as_of_gw, gw } = meta[0];
  const rows = await query<PredictionRow>(
    `SELECT p.player_id, pl.web_name, t.short_name AS team, pl.position, pl.price::float AS price, pl.status,
            pl.chance_of_playing_next_round, p.gw, p.predicted_points::float AS predicted_points,
            p.predicted_points_low::float AS predicted_points_low, p.predicted_points_high::float AS predicted_points_high
       FROM predictions p JOIN players pl ON pl.id = p.player_id
       LEFT JOIN teams t ON t.id = pl.team_id AND t.season = p.season
      WHERE p.season = $1 AND p.as_of_gw = $2 AND p.gw = $3 AND ($4::text IS NULL OR pl.position = $4)
      ORDER BY p.predicted_points DESC LIMIT $5`,
    [season, as_of_gw, gw, position ?? null, limit]
  );
  const horizon = await query<{ player_id: number; total: number }>(
    "SELECT player_id, sum(predicted_points)::float AS total FROM predictions WHERE season=$1 AND as_of_gw=$2 GROUP BY player_id",
    [season, as_of_gw]
  );
  return { meta: meta[0], rows, horizon };
}

export async function modelRuns(limit = 12) {
  return query<{ id: number; run_date: string; model_version: string; position: string; mae: number | null; rmse: number | null; n_train: number | null; n_valid: number | null; notes: string | null }>(
    "SELECT id, run_date, model_version, position, mae::float AS mae, rmse::float AS rmse, n_train, n_valid, notes FROM model_runs WHERE mae IS NOT NULL ORDER BY run_date DESC LIMIT $1",
    [limit]
  );
}

export async function accuracyRows() {
  return query<{ season: string; gw: number; model_version: string; position: string; n: number; mae: number; rmse: number }>(
    "SELECT season, gw, model_version, position, n, mae::float AS mae, rmse::float AS rmse FROM prediction_accuracy ORDER BY gw, position"
  );
}

export async function backtestRows() {
  return query<{ id: number; run_date: string; season: string; mode: string; strategy: string; start_gw: number; end_gw: number; total_points: number; notes: string | null }>(
    "SELECT id, run_date, season, mode, strategy, start_gw, end_gw, total_points::float AS total_points, notes FROM backtests ORDER BY season DESC, run_date DESC LIMIT 20"
  );
}

export type SimRun = {
  id: number; run_date: string; season: string; mode: string; strategy: string; label: string | null;
  start_gw: number; end_gw: number; total_points: number; n_repeats: number | null;
  points_sd: number | null; points_p10: number | null; points_p90: number | null;
  params_json: Record<string, unknown> | null; notes: string | null; job_id: number | null;
};

/**
 * Every simulation ever run, newest first, with the parameters that produced it.
 * `migrated` is false when the pipeline has not yet added the simulation columns to `backtests`,
 * in which case the caller should say so rather than show an empty table.
 */
export async function simulationRuns(limit = 40): Promise<{ runs: SimRun[]; migrated: boolean }> {
  const { data, migrated } = await queryIfMigrated(
    () =>
      query<SimRun>(
        `SELECT id, run_date, season, mode, strategy, label, start_gw, end_gw,
                total_points::float AS total_points, n_repeats,
                points_sd::float AS points_sd, points_p10::float AS points_p10, points_p90::float AS points_p90,
                params_json, notes, job_id
           FROM backtests ORDER BY run_date DESC LIMIT $1`,
        [limit]
      ),
    [] as SimRun[]
  );
  return { runs: data, migrated };
}

/** Seasons with gameweek data loaded, so the form only offers ones that can actually be replayed. */
export async function availableSeasons(): Promise<string[]> {
  const { data } = await queryIfMigrated(
    () =>
      query<{ season: string; n: number }>(
        "SELECT season, count(*)::int AS n FROM player_gw_stats GROUP BY season HAVING count(*) > 1000 ORDER BY season DESC"
      ),
    [] as { season: string; n: number }[]
  );
  return data.map((r) => r.season);
}

export async function lastIngest() {
  const rows = await query<{ run_at: string; season: string; current_gw: number; n_players: number; n_gw_stats: number; notes: string | null }>(
    "SELECT run_at, season, current_gw, n_players, n_gw_stats, notes FROM ingest_log ORDER BY run_at DESC LIMIT 1"
  );
  return rows[0] ?? null;
}
