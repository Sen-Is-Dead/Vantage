import { availableSeasons, simulationRuns, type SimRun } from "@/lib/db";
import { NotMigrated } from "../run-controls";
import { SimForm } from "./sim-form";

export const dynamic = "force-dynamic";

const fmt = (n: number | null | undefined, d = 0) => (n == null ? "–" : Number(n).toFixed(d));
const when = (s: string) => new Date(s).toLocaleString("en-GB", { timeZone: "Europe/London", dateStyle: "short", timeStyle: "short" });

/** The handful of parameters that actually differ from the defaults, so the table stays readable. */
const DEFAULTS: Record<string, unknown> = {
  start_gw: 2, end_gw: 38, mode: "realistic", strategy: "model", refit_every: 4, rounds: 300, horizon: 5,
  hit_risk_premium: 3, max_transfers_per_gw: 3, allow_hits: true, roll_threshold: 0, sell_fee: false,
  captain_metric: "mean", risk_appetite: 0, chip_policy: "none", repeats: 1, seed: 42,
};

function diffs(params: Record<string, unknown> | null): string {
  if (!params) return "";
  const out: string[] = [];
  for (const [k, v] of Object.entries(params)) {
    if (k === "season" || k === "seasons" || k === "label") continue;
    if (Array.isArray(v) && v.length === 0) continue;
    if (k in DEFAULTS && JSON.stringify(DEFAULTS[k]) === JSON.stringify(v)) continue;
    out.push(`${k}=${Array.isArray(v) ? v.join("/") : String(v)}`);
  }
  return out.join(" · ");
}

function Total({ r }: { r: SimRun }) {
  if (!r.n_repeats || r.n_repeats <= 1) return <b>{fmt(r.total_points)}</b>;
  return (
    <>
      <b>{fmt(r.total_points)}</b>
      <span className="small muted"> ±{fmt(r.points_sd)} ({fmt(r.points_p10)}–{fmt(r.points_p90)})</span>
    </>
  );
}

export default async function Simulate() {
  const [seasons, { runs, migrated }] = await Promise.all([availableSeasons(), simulationRuns(40)]);

  return (
    <>
      <div className="panel" style={{ marginBottom: 14 }}>
        <h2>Simulate a season</h2>
        <p className="small">
          Replays a season gameweek by gameweek under a policy you choose, scoring each week on what actually
          happened. Point-in-time throughout: at gameweek T the models have only seen earlier seasons and
          gameweeks before T, so nothing here can peek at the future except where a control says it does.
        </p>
        <p className="small muted">
          Two caveats that apply to every run, unchanged from the original harness. Historical injury flags are
          not available, so a replay can field someone who was actually injured, which is pessimistic for the
          model. And free_hit ignores transfer friction entirely, which is optimistic. Read the totals against
          each other, not as a prediction of your real season.
        </p>
      </div>

      <SimForm seasons={seasons} />

      <div className="panel" style={{ marginTop: 14 }}>
        <h2>Past simulations</h2>
        {!migrated ? (
          <NotMigrated />
        ) : runs.length === 0 ? (
          <p className="muted">None yet.</p>
        ) : (
          <table>
            <thead>
              <tr>
                <th>Run</th><th>Label</th><th>Season</th><th>Mode</th><th>Strategy</th><th>GWs</th>
                <th className="num">Points</th><th>Differences from the defaults</th>
              </tr>
            </thead>
            <tbody>
              {runs.map((r) => (
                <tr key={r.id}>
                  <td className="small muted">{when(r.run_date)}</td>
                  <td>{r.label ?? "–"}</td>
                  <td>{r.season}</td>
                  <td>{r.mode}</td>
                  <td>{r.strategy}</td>
                  <td className="small">{r.start_gw}–{r.end_gw}</td>
                  <td className="num"><Total r={r} /></td>
                  <td className="small muted">{diffs(r.params_json) || "defaults"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </>
  );
}
