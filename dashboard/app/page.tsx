import { latestEntryState, latestRecommendation, lastIngest, recentRecommendations, type SquadPlayer } from "@/lib/db";
import { HomeRun } from "./run-controls";

export const dynamic = "force-dynamic";

const fmt = (n: number | null | undefined, d = 1) => (n == null ? "–" : Number(n).toFixed(d));
const when = (s: string) => new Date(s).toLocaleString("en-GB", { timeZone: "Europe/London", dateStyle: "medium", timeStyle: "short" });

function Player({ p }: { p: SquadPlayer }) {
  return (
    <div className="card">
      <div className="n">
        <span className={`pos ${p.position}`}>{p.position}</span> {p.name}
        {p.is_captain && <span className="tag c">C</span>}
        {p.is_vice_captain && <span className="tag v">V</span>}
      </div>
      <div className="m">
        {p.team} · £{fmt(p.price)}m · <b>{fmt(p.pred)}</b> pts <span className="small">({fmt(p.low)}–{fmt(p.high)})</span>
        {p.chance != null && p.chance < 100 && <span className="tag">{p.chance}%</span>}
        {p.status !== "a" && <span className="tag">{p.status}</span>}
      </div>
      {p.news && <div className="m small">{p.news}</div>}
    </div>
  );
}

export default async function Home() {
  const [rec, state, ingest, history] = await Promise.all([latestRecommendation(), latestEntryState(), lastIngest(), recentRecommendations(8)]);
  if (!rec) {
    return (
      <>
        <HomeRun />
        <div className="panel"><h2>No recommendation yet</h2><p>Use the button above, run <code>python -m pipeline.run_weekly all</code> locally, or wait for the Tuesday job.</p></div>
      </>
    );
  }
  const t = rec.transfers_json;
  const w = rec.warnings_json;
  const xi = rec.squad_json.filter((p) => p.is_starting);
  const bench = rec.squad_json.filter((p) => !p.is_starting).sort((a, b) => (a.bench_order ?? 0) - (b.bench_order ?? 0));
  const rows = (["GKP", "DEF", "MID", "FWD"] as const).map((pos) => xi.filter((p) => p.position === pos));
  const chips = rec.chip_plan_json ? Object.values(rec.chip_plan_json) : [];
  const gws = chips[0] ? Object.keys(chips[0].by_gw) : [];

  return (
    <>
      <HomeRun />
      {w?.warnings?.map((m, i) => <div key={i} className="banner danger">⚠ {m}</div>)}
      {rec.chip_used && <div className="banner warn">Chip suggested for GW{rec.gw}: <b>{rec.chip_plan_json?.[rec.chip_used]?.label ?? rec.chip_used}</b>. See the chip table below before deciding.</div>}

      <div className="panel" style={{ marginBottom: 14 }}>
        <h2>Gameweek {rec.gw} · recommended {when(rec.run_date)} · predictions as of GW{t.as_of_gw}</h2>
        <div className="kpis">
          <div className="kpi"><div className="v">{fmt(rec.expected_points_gw)}</div><div className="l">expected XI points (captain ×2)</div></div>
          <div className="kpi"><div className="v">{t.roll_transfer ? "roll" : `${t.transfers.length}${t.hits ? ` (−${4 * t.hits})` : ""}`}</div><div className="l">transfers{t.hits ? " · hits" : ""}</div></div>
          <div className="kpi"><div className="v">{fmt(rec.expected_points_gain)}</div><div className="l">gain vs no transfers (5 GW, weighted)</div></div>
          <div className="kpi"><div className="v">£{fmt(t.bank_after)}m</div><div className="l">bank after</div></div>
          <div className="kpi"><div className="v">{t.free_transfers}</div><div className="l">free transfers available</div></div>
          {state?.overall_rank && <div className="kpi"><div className="v">{Number(state.overall_rank).toLocaleString("en-GB")}</div><div className="l">overall rank after GW{state.gw} · {state.total_points} pts</div></div>}
        </div>
      </div>

      <div className="grid">
        <div className="panel">
          <h2>Transfers</h2>
          {t.roll_transfer && <p>None. Roll the free transfer: the best move gains less than 1.5 points over the horizon.</p>}
          {!t.roll_transfer && t.transfers.length === 0 && <p>None recommended.</p>}
          {t.transfers.length > 0 && (
            <table><thead><tr><th>Out</th><th className="num">£</th><th>In</th><th className="num">£</th><th className="num">Gain</th></tr></thead>
              <tbody>{t.transfers.map((x, i) => (
                <tr key={i}><td>{x.out_name}</td><td className="num">{fmt(x.out_price)}</td><td><b>{x.in_name}</b></td><td className="num">{fmt(x.in_price)}</td><td className="num">{x.gain >= 0 ? "+" : ""}{fmt(x.gain)}</td></tr>
              ))}</tbody></table>
          )}
          {w?.protected_locked?.length ? <p className="small muted">Protected and locked: {w.protected_locked.join(", ")}.</p> : null}
        </div>
        <div className="panel">
          <h2>Captaincy</h2>
          <p><b>Captain:</b> {xi.find((p) => p.is_captain)?.name} · <b>Vice:</b> {xi.find((p) => p.is_vice_captain)?.name}</p>
          <p className="small muted">Risk appetite {w?.risk_appetite != null ? Number(w.risk_appetite).toFixed(2) : "0"}: negative shades towards the safe floor (q20), positive towards the ceiling (q80), from your overall-rank percentile.</p>
        </div>
      </div>

      <div className="panel" style={{ marginTop: 14 }}>
        <h2>Starting XI</h2>
        <div className="pitch">{rows.map((r, i) => <div key={i} className="row">{r.map((p) => <Player key={p.player_id} p={p} />)}</div>)}</div>
        <h2 style={{ marginTop: 14 }}>Bench (in order)</h2>
        <div className="row">{bench.map((p) => <Player key={p.player_id} p={p} />)}</div>
      </div>

      {chips.length > 0 && (
        <div className="panel" style={{ marginTop: 14 }}>
          <h2>Chips · predicted gain by gameweek (each chip evaluated alone vs no chip)</h2>
          <table className="chips"><thead><tr><th>Chip</th>{gws.map((g) => <th key={g} className="num">GW{g}</th>)}<th className="num">Threshold</th><th>Verdict</th></tr></thead>
            <tbody>{chips.map((c) => (
              <tr key={c.chip}>
                <td>{c.label}</td>
                {gws.map((g) => <td key={g} className={"num" + (c.best_gw === Number(g) ? " ok" : "")}>{c.by_gw[g] == null ? "–" : fmt(c.by_gw[g])}</td>)}
                <td className="num">{c.threshold}</td>
                <td>{!c.available && !Object.keys(c.by_gw).length ? <span className="hold">{c.reason}</span> : c.recommend_now ? <span className="ok">use now (+{fmt(c.best_gain)})</span> : <span className="hold">hold · best GW{c.best_gw} ({fmt(c.best_gain)})</span>}</td>
              </tr>
            ))}</tbody></table>
          {chips.map((c) => {
            const d = c.best_gw != null ? c.detail[String(c.best_gw)] : undefined;
            if (!d) return null;
            const list = (d.squad ?? d.xi ?? d.bench) as string[] | undefined;
            return list ? <p key={c.chip} className="small muted"><b>{c.label}</b> GW{c.best_gw}: {list.join(", ")}</p>
              : d.captain ? <p key={c.chip} className="small muted"><b>{c.label}</b> GW{c.best_gw}: {String(d.captain)} ({String(d.captain_points)} pts)</p> : null;
          })}
        </div>
      )}

      {w?.notes?.length ? <div className="panel" style={{ marginTop: 14 }}><h2>Notes</h2><ul>{w.notes.map((n, i) => <li key={i} className="small">{n}</li>)}</ul></div> : null}

      <div className="grid" style={{ marginTop: 14 }}>
        <div className="panel">
          <h2>Recent runs</h2>
          <table><thead><tr><th>GW</th><th>Run</th><th className="num">XI pts</th><th className="num">Gain</th><th>Chip</th></tr></thead>
            <tbody>{history.map((h) => <tr key={h.id}><td>{h.gw}</td><td>{when(h.run_date)}</td><td className="num">{fmt(h.expected_points_gw)}</td><td className="num">{fmt(h.expected_points_gain)}</td><td>{h.chip_used ?? "–"}</td></tr>)}</tbody></table>
        </div>
        <div className="panel">
          <h2>Data</h2>
          {ingest ? <p className="small">Last ingest {when(ingest.run_at)} · season {ingest.season} · current GW{ingest.current_gw} · {ingest.n_players} players · {ingest.n_gw_stats} GW rows{ingest.notes ? ` · ${ingest.notes}` : ""}</p> : <p className="small">no ingest yet</p>}
          {state && <p className="small">Bank £{fmt(state.bank)}m · squad value £{fmt(state.squad_value)}m · chips used: {state.chips_used_json?.length ? state.chips_used_json.map((c) => `${c.name} (GW${c.event})`).join(", ") : "none"}</p>}
          <p className="small muted">Model {rec.model_version ?? "?"} · status: {rec.status}</p>
        </div>
      </div>
    </>
  );
}
