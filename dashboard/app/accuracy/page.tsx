import { accuracyRows, backtestRows, modelRuns } from "@/lib/db";

export const dynamic = "force-dynamic";
const fmt = (n: number | null | undefined, d = 3) => (n == null ? "–" : Number(n).toFixed(d));
const when = (s: string) => new Date(s).toLocaleString("en-GB", { timeZone: "Europe/London", dateStyle: "medium", timeStyle: "short" });

export default async function Accuracy() {
  const [acc, runs, bts] = await Promise.all([accuracyRows(), modelRuns(16), backtestRows()]);
  const gws = Array.from(new Set(acc.map((a) => a.gw))).sort((a, b) => a - b);
  const byKey = new Map(acc.map((a) => [`${a.gw}|${a.position}`, a]));
  return (
    <>
      <div className="panel">
        <h2>Live accuracy · MAE of next-gameweek predictions once the gameweek finished</h2>
        {gws.length === 0 ? <p className="muted">No finished gameweek with predictions yet. This fills in after the first Tuesday run following a completed gameweek.</p> : (
          <table><thead><tr><th>GW</th>{["ALL", "GKP", "DEF", "MID", "FWD"].map((p) => <th key={p} className="num">{p}</th>)}<th className="num">n</th></tr></thead>
            <tbody>{gws.map((g) => (
              <tr key={g}><td>{g}</td>{["ALL", "GKP", "DEF", "MID", "FWD"].map((p) => <td key={p} className="num">{fmt(byKey.get(`${g}|${p}`)?.mae)}</td>)}<td className="num">{byKey.get(`${g}|ALL`)?.n ?? "–"}</td></tr>
            ))}</tbody></table>
        )}
      </div>
      <div className="panel" style={{ marginTop: 14 }}>
        <h2>Model evaluations (held-out season)</h2>
        <table><thead><tr><th>Run</th><th>Model</th><th>Pos</th><th className="num">MAE</th><th className="num">RMSE</th><th className="num">train</th><th className="num">valid</th></tr></thead>
          <tbody>{runs.map((r) => <tr key={r.id}><td>{when(r.run_date)}</td><td>{r.model_version}</td><td>{r.position}</td><td className="num">{fmt(r.mae)}</td><td className="num">{fmt(r.rmse)}</td><td className="num">{r.n_train ?? "–"}</td><td className="num">{r.n_valid ?? "–"}</td></tr>)}</tbody></table>
        <p className="small muted">MAE over all players (mostly non-starters, so small). The evaluation notes in the database also hold the regular-starter MAE and the naive baselines.</p>
      </div>
      <div className="panel" style={{ marginTop: 14 }}>
        <h2>Backtests (season replays, GW-by-GW, point-in-time refits)</h2>
        {bts.length === 0 ? <p className="muted">none run yet</p> : (
          <table><thead><tr><th>Season</th><th>Mode</th><th>Strategy</th><th>GWs</th><th className="num">Points</th><th>Notes</th></tr></thead>
            <tbody>{bts.map((b) => <tr key={b.id}><td>{b.season}</td><td>{b.mode}</td><td>{b.strategy}</td><td>{b.start_gw}–{b.end_gw}</td><td className="num"><b>{fmt(b.total_points, 0)}</b></td><td className="small muted">{b.notes}</td></tr>)}</tbody></table>
        )}
      </div>
    </>
  );
}
