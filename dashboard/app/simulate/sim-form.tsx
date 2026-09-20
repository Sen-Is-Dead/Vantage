"use client";

import { useState } from "react";
import { JobPanel, postRun, useJobs } from "../run-controls";

type Params = {
  seasons: string[];
  start_gw: number;
  end_gw: number;
  mode: "realistic" | "free_hit";
  strategy: "model" | "naive_last5";
  refit_every: number;
  rounds: number;
  horizon: number;
  hit_risk_premium: number;
  max_transfers_per_gw: number;
  allow_hits: boolean;
  roll_threshold: number;
  sell_fee: boolean;
  bench_weight: string;
  captain_metric: "mean" | "q20" | "q80" | "risk";
  risk_appetite: number;
  locked: string;
  banned: string;
  formation: string;
  chip_policy: "none" | "threshold" | "hindsight";
  repeats: number;
  seed: number;
  label: string;
};

const base = (season: string): Params => ({
  seasons: season ? [season] : [],
  start_gw: 2,
  end_gw: 38,
  mode: "realistic",
  strategy: "model",
  refit_every: 4,
  rounds: 300,
  horizon: 5,
  hit_risk_premium: 3,
  max_transfers_per_gw: 3,
  allow_hits: true,
  roll_threshold: 0,
  sell_fee: false,
  bench_weight: "",
  captain_metric: "mean",
  risk_appetite: 0,
  locked: "",
  banned: "",
  formation: "",
  chip_policy: "none",
  repeats: 1,
  seed: 42,
  label: "",
});

const PRESETS: { name: string; hint: string; patch: Partial<Params> }[] = [
  {
    name: "Perfect wildcard every week",
    hint: "Rebuild the best legal 15 from scratch every gameweek on a fresh 100.0 budget. The ceiling on what the predictions are worth with no transfer friction.",
    patch: { mode: "free_hit", strategy: "model", label: "perfect wildcard" },
  },
  {
    name: "Realistic, as configured",
    hint: "1 free transfer a week banked to 5, hits allowed when the solver thinks they pay. This is what the live recommender does.",
    patch: { mode: "realistic", strategy: "model", label: "" },
  },
  {
    name: "Naive last-5 baseline",
    hint: "Same optimiser, but players are valued by their last-5 mean. If the model cannot beat this, the model is not earning its keep.",
    patch: { strategy: "naive_last5", label: "naive baseline" },
  },
  {
    name: "Never take a hit",
    hint: "Free transfers only. Tests whether your hits actually paid over a season.",
    patch: { allow_hits: true, max_transfers_per_gw: 1, hit_risk_premium: 6, label: "no hits" },
  },
  {
    name: "Chase the ceiling",
    hint: "Captain the highest q80 rather than the highest mean: more variance, for when you need rank rather than safety.",
    patch: { captain_metric: "q80", label: "ceiling captain" },
  },
  {
    name: "Protect the rank",
    hint: "Captain the highest q20, the safest floor. The mirror image of chasing.",
    patch: { captain_metric: "q20", label: "floor captain" },
  },
  {
    name: "Chip ceiling",
    hint: "Bench Boost and Triple Captain placed on their best possible weeks in hindsight: how much you left on the table by mistiming chips.",
    patch: { chip_policy: "hindsight", label: "chip ceiling" },
  },
  {
    name: "Noise check",
    hint: "Ten replays with predictions resampled inside their own q20/q80 band. Turns one fragile number into a range.",
    patch: { repeats: 10, label: "noise check" },
  },
];

const Row = ({ label, hint, children }: { label: string; hint?: string; children: React.ReactNode }) => (
  <div className="field">
    <label>{label}</label>
    <div className="control">{children}</div>
    {hint && <div className="small muted hint">{hint}</div>}
  </div>
);

export function SimForm({ seasons }: { seasons: string[] }) {
  const [p, setP] = useState<Params>(base(seasons[0] ?? ""));
  const [error, setError] = useState<string | null>(null);
  const [note, setNote] = useState<string | null>(null);
  const [sending, setSending] = useState(false);
  const { jobs, loaded, ready, refresh } = useJobs();

  const set = <K extends keyof Params>(k: K, v: Params[K]) => setP((s) => ({ ...s, [k]: v }));
  const toggleSeason = (s: string) =>
    setP((st) => ({
      ...st,
      seasons: st.seasons.includes(s) ? st.seasons.filter((x) => x !== s) : [...st.seasons, s].slice(0, 4),
    }));

  const span = Math.max(0, p.end_gw - p.start_gw + 1);
  const work = p.seasons.length * p.repeats * span;
  const overBudget = work > 38 * 12;
  // Fitting dominates: roughly a refit per refit_every gameweeks, then a cheap replay per repeat.
  const mins = Math.round(
    (p.strategy === "model" ? (span / Math.max(1, p.refit_every)) * (p.rounds / 300) * 1.6 : 0.2) * p.seasons.length +
      (span * p.repeats * p.seasons.length) / 90
  );

  const submit = async () => {
    setError(null);
    setNote(null);
    setSending(true);
    try {
      const params: Record<string, unknown> = {
        ...p,
        locked: p.locked.split(",").map((s) => s.trim()).filter(Boolean),
        banned: p.banned.split(",").map((s) => s.trim()).filter(Boolean),
      };
      if (p.bench_weight === "") delete params.bench_weight;
      if (p.formation === "") delete params.formation;
      if (!p.label) delete params.label;
      const { id } = await postRun("simulate", params);
      setNote(`Queued as job ${id}. It will appear below within a few seconds and update as it runs.`);
      refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : "could not queue the simulation");
    } finally {
      setSending(false);
    }
  };

  return (
    <>
      <div className="panel">
        <h2>Presets</h2>
        <p className="small muted">A starting point. Everything stays editable underneath.</p>
        <div className="presets">
          {PRESETS.map((x) => (
            <button key={x.name} className="preset" onClick={() => setP((s) => ({ ...s, ...x.patch }))} title={x.hint}>
              <b>{x.name}</b>
              <span className="small muted">{x.hint}</span>
            </button>
          ))}
        </div>
      </div>

      <div className="panel" style={{ marginTop: 14 }}>
        <h2>Season and scope</h2>
        <Row label="Seasons" hint="Up to four. Several seasons in one run is the check that the model is not fitted to one year.">
          <div className="chips-row">
            {seasons.length === 0 && <span className="small muted">no seasons loaded yet; run the history step first</span>}
            {seasons.map((s) => (
              <button key={s} className={`chip ${p.seasons.includes(s) ? "on" : ""}`} onClick={() => toggleSeason(s)}>
                {s}
              </button>
            ))}
          </div>
        </Row>
        <div className="grid2">
          <Row label="First gameweek">
            <input type="number" min={1} max={38} value={p.start_gw} onChange={(e) => set("start_gw", +e.target.value)} />
          </Row>
          <Row label="Last gameweek">
            <input type="number" min={1} max={38} value={p.end_gw} onChange={(e) => set("end_gw", +e.target.value)} />
          </Row>
        </div>
        <div className="grid2">
          <Row label="Mode" hint="free_hit rebuilds from scratch every week on a fresh 100.0 budget.">
            <select value={p.mode} onChange={(e) => set("mode", e.target.value as Params["mode"])}>
              <option value="realistic">realistic (1 free transfer a week)</option>
              <option value="free_hit">free_hit (perfect wildcard every week)</option>
            </select>
          </Row>
          <Row label="Strategy" hint="What values a player: the trained models, or a last-5 mean baseline.">
            <select value={p.strategy} onChange={(e) => set("strategy", e.target.value as Params["strategy"])}>
              <option value="model">model (LightGBM)</option>
              <option value="naive_last5">naive_last5 (baseline)</option>
            </select>
          </Row>
        </div>
        <Row label="Label" hint="Shows up next to the result so you can tell runs apart later.">
          <input value={p.label} onChange={(e) => set("label", e.target.value)} placeholder="e.g. no hits, horizon 8" />
        </Row>
      </div>

      <div className="grid" style={{ marginTop: 14 }}>
        <div className="panel">
          <h2>Model</h2>
          <Row label={`Refit every ${p.refit_every} GW`} hint="How often the models are retrained during the replay. Lower is more faithful and much slower; this knob dominates the runtime.">
            <input type="range" min={1} max={19} value={p.refit_every} onChange={(e) => set("refit_every", +e.target.value)} />
          </Row>
          <Row label={`Boosting rounds: ${p.rounds}`} hint="LightGBM rounds per position. 300 is the backtest default; production predictions use 600.">
            <input type="range" min={50} max={1200} step={50} value={p.rounds} onChange={(e) => set("rounds", +e.target.value)} />
          </Row>
          <Row label={`Horizon: ${p.horizon} GW`} hint="How far ahead a transfer is valued. Short chases this week, long holds fixtures.">
            <input type="range" min={1} max={8} value={p.horizon} onChange={(e) => set("horizon", +e.target.value)} />
          </Row>
        </div>

        <div className="panel">
          <h2>Transfers and hits</h2>
          <Row label={`Hit risk premium: ${p.hit_risk_premium.toFixed(1)}`} hint="A -4 is only taken when the predicted edge beats 4 x this. You measured 1.5 -> 23 hits, 1924 pts and 3.0 -> 3 hits, 1976 pts on 2025/26.">
            <input type="range" min={1} max={6} step={0.25} value={p.hit_risk_premium} onChange={(e) => set("hit_risk_premium", +e.target.value)} />
          </Row>
          <Row label={`Max transfers per GW: ${p.max_transfers_per_gw}`}>
            <input type="range" min={1} max={5} value={p.max_transfers_per_gw} onChange={(e) => set("max_transfers_per_gw", +e.target.value)} />
          </Row>
          <Row label={`Roll threshold: ${p.roll_threshold.toFixed(1)} pts`} hint="Keep the free transfer unless the best available move gains at least this much. 0 means always take the best move the solver finds.">
            <input type="range" min={0} max={8} step={0.5} value={p.roll_threshold} onChange={(e) => set("roll_threshold", +e.target.value)} />
          </Row>
          <label className="check small">
            <input type="checkbox" checked={p.allow_hits} onChange={(e) => set("allow_hits", e.target.checked)} /> allow hits at all
          </label>
          <label className="check small">
            <input type="checkbox" checked={p.sell_fee} onChange={(e) => set("sell_fee", e.target.checked)} /> charge the 50% sell-on fee
            <span className="muted"> (off elsewhere in the harness, which treats buy = sell)</span>
          </label>
        </div>
      </div>

      <div className="grid" style={{ marginTop: 14 }}>
        <div className="panel">
          <h2>Squad shape and captaincy</h2>
          <Row label="Captain on" hint="q20 is the safest floor, q80 the highest ceiling, risk blends the two by appetite.">
            <select value={p.captain_metric} onChange={(e) => set("captain_metric", e.target.value as Params["captain_metric"])}>
              <option value="mean">mean prediction</option>
              <option value="q20">q20 (floor, protect a rank)</option>
              <option value="q80">q80 (ceiling, chase a rank)</option>
              <option value="risk">blended by risk appetite</option>
            </select>
          </Row>
          {p.captain_metric === "risk" && (
            <Row label={`Risk appetite: ${p.risk_appetite.toFixed(2)}`} hint="-1 all floor, +1 all ceiling. The live recommender derives this from your overall-rank percentile.">
              <input type="range" min={-1} max={1} step={0.05} value={p.risk_appetite} onChange={(e) => set("risk_appetite", +e.target.value)} />
            </Row>
          )}
          <Row label="Bench weight" hint="How much a bench slot's horizon value counts against XI points. Blank uses the mode default (1.0 realistic, 0.05 free_hit). Low values buy bench fodder.">
            <input type="number" step={0.05} min={0} max={2} value={p.bench_weight} placeholder="default"
                   onChange={(e) => set("bench_weight", e.target.value)} />
          </Row>
          <Row label="Force formation" hint="Blank leaves the XI free inside 3-5 / 2-5 / 1-3.">
            <input value={p.formation} placeholder="e.g. 3-4-3" onChange={(e) => set("formation", e.target.value)} />
          </Row>
          <Row label="Always own" hint="Comma-separated names, matched the way PROTECTED_PLAYERS is. These can never be transferred out.">
            <input value={p.locked} placeholder="Mbeumo" onChange={(e) => set("locked", e.target.value)} />
          </Row>
          <Row label="Never own" hint="Comma-separated. Useful for asking what a season looks like without one player.">
            <input value={p.banned} placeholder="Haaland" onChange={(e) => set("banned", e.target.value)} />
          </Row>
        </div>

        <div className="panel">
          <h2>Chips and noise</h2>
          <Row label="Chip policy" hint="threshold uses the live thresholds on predicted gain (BB 12, TC 7, FH 12, WC 15). hindsight places Bench Boost and Triple Captain on their best weeks after the fact: a ceiling, not a strategy.">
            <select value={p.chip_policy} onChange={(e) => set("chip_policy", e.target.value as Params["chip_policy"])}>
              <option value="none">no chips</option>
              <option value="threshold">threshold policy (what the recommender would do)</option>
              <option value="hindsight">perfect hindsight (BB and TC only)</option>
            </select>
          </Row>
          <Row label={`Monte Carlo repeats: ${p.repeats}`} hint="Each repeat resamples every prediction inside its own q20/q80 band and replays the season against the same real results. One number becomes a distribution. Repeats above 1 also force the quantile models to be fitted, which roughly doubles the fitting time.">
            <input type="range" min={1} max={25} value={p.repeats} onChange={(e) => set("repeats", +e.target.value)} />
          </Row>
          <Row label="Seed">
            <input type="number" value={p.seed} onChange={(e) => set("seed", +e.target.value)} />
          </Row>
        </div>
      </div>

      <div className="panel" style={{ marginTop: 14 }}>
        <div className="runbar">
          <button className="btn primary" onClick={submit} disabled={sending || p.seasons.length === 0 || overBudget}>
            {sending ? "queueing…" : "Run simulation"}
          </button>
          <span className="small muted">
            {p.seasons.length} season{p.seasons.length === 1 ? "" : "s"} x {p.repeats} repeat
            {p.repeats === 1 ? "" : "s"} x {span} gameweeks. Rough estimate: {mins < 1 ? "under a minute" : `${mins} min`} on
            a GitHub runner.
          </span>
        </div>
        {overBudget && (
          <div className="banner warn" style={{ marginTop: 8 }}>
            That is too much work for one run. Drop the repeats, narrow the gameweek range, or run the seasons separately.
          </div>
        )}
        {error && <div className="banner danger" style={{ marginTop: 8 }}>{error}</div>}
        {note && <div className="banner" style={{ marginTop: 8 }}>{note}</div>}
      </div>

      <div className="panel" style={{ marginTop: 14 }}>
        <h2>Runs</h2>
        <JobPanel jobs={jobs} loaded={loaded} ready={ready} />
      </div>
    </>
  );
}
