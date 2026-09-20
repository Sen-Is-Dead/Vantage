import { NextResponse } from "next/server";
import {
  activeJob,
  cleanRefreshParams,
  cleanSimParams,
  createJob,
  dispatch,
  expireStaleJobs,
  failJob,
  JOB_KINDS,
  jobsToday,
  MAX_JOBS_PER_DAY,
  NOT_MIGRATED,
  type JobKind,
} from "@/lib/jobs";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const bad = (message: string, status = 400) => NextResponse.json({ error: message }, { status });

export async function POST(req: Request) {
  let body: Record<string, unknown>;
  try {
    body = (await req.json()) as Record<string, unknown>;
  } catch {
    return bad("expected a JSON body");
  }

  // Optional password. Unset (the default) means the buttons are open to anyone with the URL;
  // set RUN_SECRET in Vercel and this starts being enforced with no code change.
  const secret = process.env.RUN_SECRET;
  if (secret) {
    const given = req.headers.get("x-run-secret") ?? String(body.secret ?? "");
    if (given !== secret) return bad("wrong or missing password", 401);
  }

  const kind = String(body.kind ?? "") as JobKind;
  if (!JOB_KINDS.includes(kind)) return bad(`kind must be one of ${JOB_KINDS.join(", ")}`);

  let params: Record<string, unknown>;
  try {
    params =
      kind === "simulate"
        ? cleanSimParams((body.params as Record<string, unknown>) ?? {})
        : cleanRefreshParams((body.params as Record<string, unknown>) ?? {});
  } catch (e) {
    return bad(e instanceof Error ? e.message : "bad parameters");
  }

  try {
    await expireStaleJobs();

    // One at a time. The pipeline writes to shared tables and the workflow serialises anyway, so a
    // second run would only queue behind the first while looking to the user like it had started.
    const busy = await activeJob();
    if (busy) {
      return NextResponse.json(
        { error: `job ${busy.id} (${busy.kind}) is already ${busy.status}`, job: busy },
        { status: 409 }
      );
    }
    const today = await jobsToday();
    if (today >= MAX_JOBS_PER_DAY) {
      return bad(`daily cap reached (${MAX_JOBS_PER_DAY} runs in 24 hours)`, 429);
    }

    const label = (params.label as string) ?? null;
    delete params.label;
    let id: number;
    try {
      id = await createJob(kind, params, label);
    } catch (e) {
      const message = e instanceof Error ? e.message : String(e);
      // 503 rather than 500: the request was fine, the database just is not ready yet
      return bad(message, message === NOT_MIGRATED ? 503 : 500);
    }
    try {
      await dispatch(id, kind);
    } catch (e) {
      const message = e instanceof Error ? e.message : String(e);
      await failJob(id, message);
      return NextResponse.json({ error: message, id }, { status: 502 });
    }
    return NextResponse.json({ id, kind, params }, { status: 202 });
  } catch (e) {
    return bad(e instanceof Error ? e.message : "could not queue the job", 500);
  }
}
