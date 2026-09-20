// Queueing side of the dashboard. Everything else here is read-only; this file is the one place
// that writes, and the only thing it writes is a row in `jobs` describing work for the pipeline to
// do. It cannot submit anything to FPL: the runner only accepts the three kinds below, all of which
// read the FPL API and write to our own Postgres.
import { isMissingSchema, query, queryIfMigrated } from "@/lib/db";

export type JobKind = "refresh" | "weekly" | "simulate";
export const JOB_KINDS: JobKind[] = ["refresh", "weekly", "simulate"];

export type Job = {
  id: number;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  kind: JobKind;
  status: "queued" | "running" | "success" | "failed";
  label: string | null;
  params_json: Record<string, unknown>;
  result_json: Record<string, unknown> | null;
  progress: string | null;
  log_tail: string | null;
  error: string | null;
  run_url: string | null;
};

// No login on this dashboard, so these two are what stop a stranger (or a stuck retry loop) from
// queueing season replays all afternoon. Set RUN_SECRET in Vercel to add a password on top.
export const MAX_JOBS_PER_DAY = 40;

// The `jobs` table is created by the pipeline's apply_schema, so it does not exist until the
// pipeline has run at least once since this version was deployed. Every read below treats that as
// "no jobs yet" rather than an error; only queueing one is a hard failure, with a message that says
// what to do about it.

export const NOT_MIGRATED =
  "The pipeline has not run since this version was deployed, so the jobs table does not exist yet. " +
  "Run the weekly workflow once from the repository's Actions tab (it applies the schema), then try again.";

export async function recentJobs(limit = 12): Promise<{ jobs: Job[]; ready: boolean }> {
  const { data, migrated } = await queryIfMigrated(
    () => query<Job>("SELECT * FROM jobs ORDER BY created_at DESC LIMIT $1", [limit]),
    [] as Job[]
  );
  return { jobs: data, ready: migrated };
}

export async function activeJob(): Promise<Job | null> {
  const { data } = await queryIfMigrated(
    () => query<Job>("SELECT * FROM jobs WHERE status IN ('queued','running') ORDER BY created_at LIMIT 1"),
    [] as Job[]
  );
  return data[0] ?? null;
}

export async function jobsToday(): Promise<number> {
  const { data } = await queryIfMigrated(
    () => query<{ n: string }>("SELECT count(*) AS n FROM jobs WHERE created_at > now() - interval '24 hours'"),
    [] as { n: string }[]
  );
  return Number(data[0]?.n ?? 0);
}

/** A job that claims to be running but whose workflow never reported back. */
export async function expireStaleJobs(): Promise<void> {
  await queryIfMigrated(
    () =>
      query(
        `UPDATE jobs SET status='failed', finished_at=now(),
                error=coalesce(error,'no heartbeat from the workflow for 6 hours; it was probably never picked up')
          WHERE status IN ('queued','running') AND created_at < now() - interval '6 hours'`
      ),
    []
  );
}

export async function createJob(kind: JobKind, params: Record<string, unknown>, label: string | null) {
  try {
    const rows = await query<{ id: number }>(
      "INSERT INTO jobs (kind, params_json, label) VALUES ($1, $2, $3) RETURNING id",
      [kind, JSON.stringify(params), label]
    );
    return rows[0].id;
  } catch (e) {
    if (isMissingSchema(e)) throw new Error(NOT_MIGRATED);
    throw e;
  }
}

export async function failJob(id: number, error: string) {
  await query("UPDATE jobs SET status='failed', finished_at=now(), error=$2 WHERE id=$1", [id, error]);
}

/** Fire the workflow that actually runs the job. Returns nothing on success, throws on failure. */
export async function dispatch(jobId: number, kind: JobKind): Promise<void> {
  const repo = process.env.GITHUB_REPO || "Sen-Is-Dead/Vantage";
  const ref = process.env.GITHUB_REF_NAME || "main";
  const token = process.env.GITHUB_DISPATCH_TOKEN;
  if (!token) {
    throw new Error(
      "GITHUB_DISPATCH_TOKEN is not set. Vercel -> Settings -> Environment Variables: a fine-grained " +
        "personal access token with Actions: read and write on this repo only."
    );
  }
  const res = await fetch(`https://api.github.com/repos/${repo}/actions/workflows/run.yml/dispatches`, {
    method: "POST",
    headers: {
      Accept: "application/vnd.github+json",
      Authorization: `Bearer ${token}`,
      "X-GitHub-Api-Version": "2022-11-28",
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ ref, inputs: { job_id: String(jobId), kind } }),
  });
  if (res.status !== 204) {
    const body = await res.text().catch(() => "");
    throw new Error(`GitHub refused the dispatch (${res.status}): ${body.slice(0, 300)}`);
  }
}

// ---------------------------------------------------------------------------------------------
// parameter validation
// ---------------------------------------------------------------------------------------------
// The POST route is open, so nothing from the client is trusted: unknown keys are dropped and every
// number is clamped to a range the pipeline can actually finish inside the workflow timeout. These
// bounds mirror SimConfig.__post_init__ in pipeline/simulate.py.

const num = (v: unknown, lo: number, hi: number, dflt: number) => {
  const n = typeof v === "number" ? v : parseFloat(String(v));
  return Number.isFinite(n) ? Math.min(hi, Math.max(lo, n)) : dflt;
};
const int = (v: unknown, lo: number, hi: number, dflt: number) => Math.round(num(v, lo, hi, dflt));
const pick = <T extends string>(v: unknown, allowed: readonly T[], dflt: T): T =>
  allowed.includes(v as T) ? (v as T) : dflt;
const names = (v: unknown, max = 8): string[] =>
  (Array.isArray(v) ? v : String(v ?? "").split(","))
    .map((s) => String(s).trim())
    .filter(Boolean)
    .slice(0, max)
    .map((s) => s.slice(0, 40));

export const SEASON_RE = /^\d{4}\/\d{2}$/;

export function cleanSimParams(raw: Record<string, unknown>): Record<string, unknown> {
  const seasons = (Array.isArray(raw.seasons) ? raw.seasons : [raw.season])
    .map((s) => String(s ?? "").trim())
    .filter((s) => SEASON_RE.test(s))
    .slice(0, 4);
  if (seasons.length === 0) throw new Error("pick at least one season, formatted like 2025/26");

  const start_gw = int(raw.start_gw, 1, 38, 2);
  const end_gw = Math.max(start_gw, int(raw.end_gw, 1, 38, 38));
  const repeats = int(raw.repeats, 1, 25, 1);

  // Fitting dominates the runtime and Monte Carlo replays it; keep the product inside the timeout.
  const span = end_gw - start_gw + 1;
  if (seasons.length * repeats * span > 38 * 12) {
    throw new Error(
      "that is too much work for one run: seasons x repeats x gameweeks is over the limit. " +
        "Drop the repeats, narrow the gameweek range, or run the seasons separately."
    );
  }

  const out: Record<string, unknown> = {
    seasons,
    start_gw,
    end_gw,
    mode: pick(raw.mode, ["realistic", "free_hit"] as const, "realistic"),
    strategy: pick(raw.strategy, ["model", "naive_last5"] as const, "model"),
    refit_every: int(raw.refit_every, 1, 19, 4),
    rounds: int(raw.rounds, 50, 1200, 300),
    horizon: int(raw.horizon, 1, 8, 5),
    hit_risk_premium: num(raw.hit_risk_premium, 1, 6, 3),
    max_transfers_per_gw: int(raw.max_transfers_per_gw, 1, 5, 3),
    allow_hits: raw.allow_hits !== false,
    roll_threshold: num(raw.roll_threshold, 0, 20, 0),
    sell_fee: raw.sell_fee === true,
    captain_metric: pick(raw.captain_metric, ["mean", "q20", "q80", "risk"] as const, "mean"),
    risk_appetite: num(raw.risk_appetite, -1, 1, 0),
    locked: names(raw.locked),
    banned: names(raw.banned),
    chip_policy: pick(raw.chip_policy, ["none", "threshold", "hindsight"] as const, "none"),
    repeats,
    seed: int(raw.seed, 0, 1e6, 42),
  };
  if (raw.bench_weight !== undefined && raw.bench_weight !== null && raw.bench_weight !== "") {
    out.bench_weight = num(raw.bench_weight, 0, 2, 1);
  }
  const formation = String(raw.formation ?? "").trim();
  if (formation) {
    if (!/^\d-\d-\d$/.test(formation)) throw new Error("formation must look like 3-4-3");
    const [d, m, f] = formation.split("-").map(Number);
    if (d + m + f !== 10) throw new Error(`formation ${formation} has ${d + m + f} outfielders, needs 10`);
    if (d < 3 || m < 2 || f < 1) throw new Error(`formation ${formation} breaks the FPL minimums 3/2/1`);
    out.formation = formation;
  }
  if (raw.label) out.label = String(raw.label).slice(0, 80);
  return out;
}

export function cleanRefreshParams(raw: Record<string, unknown>): Record<string, unknown> {
  return {
    horizon: int(raw.horizon, 1, 8, 5),
    quick: raw.quick === true,
    ...(raw.free_transfers == null || raw.free_transfers === ""
      ? {}
      : { free_transfers: int(raw.free_transfers, 0, 5, 1) }),
  };
}
