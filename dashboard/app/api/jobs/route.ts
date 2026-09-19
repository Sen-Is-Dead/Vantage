import { NextResponse } from "next/server";
import { expireStaleJobs, recentJobs } from "@/lib/jobs";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

// Polled by the job panel while something is running. The log tail can be long, so it is only sent
// for the newest row; older ones come back as headlines.
export async function GET(req: Request) {
  const limit = Math.min(25, Math.max(1, Number(new URL(req.url).searchParams.get("limit") ?? 8)));
  try {
    await expireStaleJobs();
    const jobs = await recentJobs(limit);
    return NextResponse.json(
      { jobs: jobs.map((j, i) => (i === 0 ? j : { ...j, log_tail: null })) },
      { headers: { "Cache-Control": "no-store" } }
    );
  } catch (e) {
    return NextResponse.json({ error: e instanceof Error ? e.message : "could not read jobs" }, { status: 500 });
  }
}
