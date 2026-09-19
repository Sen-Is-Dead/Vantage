"use client";

import { Fragment, useCallback, useEffect, useRef, useState } from "react";

export type Job = {
  id: number;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  kind: string;
  status: "queued" | "running" | "success" | "failed";
  label: string | null;
  params_json: Record<string, unknown>;
  result_json: Record<string, unknown> | null;
  progress: string | null;
  log_tail: string | null;
  error: string | null;
  run_url: string | null;
};

const ACTIVE = new Set(["queued", "running"]);
const when = (s: string | null) =>
  s ? new Date(s).toLocaleString("en-GB", { timeZone: "Europe/London", dateStyle: "medium", timeStyle: "short" }) : "–";

function elapsed(job: Job): string {
  const from = job.started_at ?? job.created_at;
  const to = job.finished_at ?? new Date().toISOString();
  const s = Math.max(0, Math.round((Date.parse(to) - Date.parse(from)) / 1000));
  return s < 60 ? `${s}s` : `${Math.floor(s / 60)}m ${s % 60}s`;
}

/** Polls /api/jobs: fast while something is in flight, slowly when idle. */
export function useJobs(pollMs = 4000) {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [loaded, setLoaded] = useState(false);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  // the polling loop reads the latest jobs without re-subscribing on every change
  const jobsRef = useRef<Job[]>([]);
  jobsRef.current = jobs;

  const refresh = useCallback(async () => {
    try {
      const res = await fetch("/api/jobs?limit=8", { cache: "no-store" });
      const body = await res.json();
      if (Array.isArray(body.jobs)) setJobs(body.jobs);
    } catch {
      /* a dropped poll is not worth showing; the next one will land */
    } finally {
      setLoaded(true);
    }
  }, []);

  useEffect(() => {
    let stop = false;
    const tick = async () => {
      if (stop) return;
      await refresh();
      if (stop) return;
      const busy = jobsRef.current.some((j) => ACTIVE.has(j.status));
      timer.current = setTimeout(tick, busy ? pollMs : pollMs * 8);
    };
    tick();
    return () => {
      stop = true;
      if (timer.current) clearTimeout(timer.current);
    };
  }, [refresh, pollMs]);

  return { jobs, loaded, refresh, busy: jobs.some((j) => ACTIVE.has(j.status)) };
}

export async function postRun(kind: string, params: Record<string, unknown>, secret?: string) {
  const res = await fetch("/api/run", {
    method: "POST",
    headers: { "Content-Type": "application/json", ...(secret ? { "x-run-secret": secret } : {}) },
    body: JSON.stringify({ kind, params }),
  });
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.error || `request failed (${res.status})`);
  return body as { id: number };
}

/** The "don't wait until Tuesday" button: ingest, predict, recommend, now. */
export function RunNowButton({ onQueued }: { onQueued?: () => void }) {
  const [state, setState] = useState<"idle" | "sending">("idle");
  const [error, setError] = useState<string | null>(null);
  const [full, setFull] = useState(false);

  const go = async () => {
    setState("sending");
    setError(null);
    try {
      await postRun(full ? "weekly" : "refresh", { horizon: 5 });
      onQueued?.();
    } catch (e) {
      setError(e instanceof Error ? e.message : "could not start the run");
    } finally {
      setState("idle");
    }
  };

  return (
    <div className="runbar">
      <button className="btn primary" onClick={go} disabled={state === "sending"}>
        {state === "sending" ? "starting…" : full ? "Run full weekly job" : "Run prediction now"}
      </button>
      <label className="small muted check">
        <input type="checkbox" checked={full} onChange={(e) => setFull(e.target.checked)} /> full job
        (adds accuracy and the held-out training evaluation, several minutes slower)
      </label>
      {error && <div className="banner danger" style={{ marginTop: 8 }}>{error}</div>}
    </div>
  );
}

export function JobPanel({ jobs, loaded }: { jobs: Job[]; loaded: boolean }) {
  const [open, setOpen] = useState<number | null>(null);
  if (!loaded) return <p className="small muted">checking for running jobs…</p>;
  if (jobs.length === 0) return <p className="small muted">No runs yet. Nothing has been queued from here.</p>;

  return (
    <table className="jobs">
      <thead>
        <tr>
          <th>#</th><th>Kind</th><th>Status</th><th>Progress</th><th className="num">Took</th><th>Started</th><th></th>
        </tr>
      </thead>
      <tbody>
        {jobs.map((j) => (
          <Fragment key={j.id}>
            <tr className={ACTIVE.has(j.status) ? "active" : ""}>
              <td className="muted">{j.id}</td>
              <td>{j.label ?? j.kind}</td>
              <td><span className={`pill ${j.status}`}>{j.status}</span></td>
              <td className="small">{j.error ? <span className="err">{j.error}</span> : j.progress ?? "–"}</td>
              <td className="num small">{elapsed(j)}</td>
              <td className="small muted">{when(j.started_at ?? j.created_at)}</td>
              <td className="small">
                <button className="btn link" onClick={() => setOpen(open === j.id ? null : j.id)}>
                  {open === j.id ? "hide" : "details"}
                </button>
                {j.run_url && (
                  <>
                    {" · "}
                    <a href={j.run_url} target="_blank" rel="noreferrer">log</a>
                  </>
                )}
              </td>
            </tr>
            {open === j.id && (
              <tr>
                <td colSpan={7}>
                  <div className="detail">
                    <div className="small muted">params</div>
                    <pre>{JSON.stringify(j.params_json, null, 2)}</pre>
                    {j.result_json && (
                      <>
                        <div className="small muted">result</div>
                        <pre>{JSON.stringify(j.result_json, null, 2)}</pre>
                      </>
                    )}
                    {j.log_tail && (
                      <>
                        <div className="small muted">log tail</div>
                        <pre className="log">{j.log_tail}</pre>
                      </>
                    )}
                  </div>
                </td>
              </tr>
            )}
          </Fragment>
        ))}
      </tbody>
    </table>
  );
}

/** Convenience wrapper: the panel plus its own polling, for pages that just want to drop it in. */
export function LiveJobs({ heading = "Runs" }: { heading?: string }) {
  const { jobs, loaded } = useJobs();
  return (
    <div className="panel" style={{ marginTop: 14 }}>
      <h2>{heading}</h2>
      <JobPanel jobs={jobs} loaded={loaded} />
    </div>
  );
}

/**
 * The home page island: the run button and the job list sharing one poll, so queueing a run makes
 * the list update immediately instead of waiting for the next tick. The page itself is a server
 * component rendered from the database, so it reloads once the job reports success.
 */
export function HomeRun() {
  const { jobs, loaded, refresh, busy } = useJobs();
  const wasBusy = useRef(false);

  useEffect(() => {
    // the newest recommendation is server-rendered, so pull the page again when a run finishes
    if (wasBusy.current && !busy) window.location.reload();
    wasBusy.current = busy;
  }, [busy]);

  return (
    <div className="panel" style={{ marginBottom: 14 }}>
      <h2>Run the pipeline</h2>
      <RunNowButton onQueued={refresh} />
      <p className="small muted" style={{ marginBottom: 10 }}>
        Ingests the latest FPL data, refits, predicts and re-optimises, rather than waiting for the Tuesday
        03:00 job. It still cannot submit anything to FPL.
      </p>
      <JobPanel jobs={jobs} loaded={loaded} />
    </div>
  );
}
