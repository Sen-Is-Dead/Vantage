import Link from "next/link";
import { predictionsForNextGw } from "@/lib/db";

export const dynamic = "force-dynamic";
const fmt = (n: number | null | undefined, d = 1) => (n == null ? "–" : Number(n).toFixed(d));
const POS = ["ALL", "GKP", "DEF", "MID", "FWD"];

export default async function Predictions({ searchParams }: { searchParams: Promise<{ pos?: string }> }) {
  const { pos } = await searchParams;
  const position = pos && pos !== "ALL" ? pos : undefined;
  const { meta, rows, horizon } = await predictionsForNextGw(position, 60);
  const h = new Map(horizon.map((x) => [x.player_id, x.total]));
  if (!meta) return <div className="panel"><h2>No predictions yet</h2></div>;
  return (
    <div className="panel">
      <h2>Predicted points · GW{meta.gw} (as of GW{meta.as_of_gw}) · top 60</h2>
      <p className="filters small">
        {POS.map((p) => <Link key={p} href={p === "ALL" ? "/predictions" : `/predictions?pos=${p}`} className={(pos ?? "ALL") === p ? "active" : ""}>{p}</Link>)}
        <span className="muted">· low/high are the q20/q80 quantile models · 5-GW = sum of predicted points over the horizon</span>
      </p>
      <table>
        <thead><tr><th>#</th><th>Player</th><th>Team</th><th>Pos</th><th className="num">£</th><th className="num">GW{meta.gw}</th><th className="num">low</th><th className="num">high</th><th className="num">5-GW</th><th>Availability</th></tr></thead>
        <tbody>{rows.map((r, i) => (
          <tr key={r.player_id}>
            <td className="muted">{i + 1}</td><td><b>{r.web_name}</b></td><td>{r.team}</td><td><span className={`pos ${r.position}`}>{r.position}</span></td>
            <td className="num">{fmt(r.price)}</td><td className="num"><b>{fmt(r.predicted_points)}</b></td><td className="num">{fmt(r.predicted_points_low)}</td><td className="num">{fmt(r.predicted_points_high)}</td>
            <td className="num">{fmt(h.get(r.player_id))}</td>
            <td className="small">{r.status === "a" && r.chance_of_playing_next_round == null ? "" : `${r.status}${r.chance_of_playing_next_round != null ? ` · ${r.chance_of_playing_next_round}%` : ""}`}</td>
          </tr>
        ))}</tbody>
      </table>
    </div>
  );
}
