import { useState } from "react";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ErrorBar, Legend, LineChart, Line, Cell } from "recharts";
import { useLatestRun } from "@/hooks/useOulad";
import { PageHeader, Panel, NeedsData, ChartTip } from "@/components/common";
import { pm, fmt3, METRICS, MODEL_COLORS, SOURCE_COLORS } from "@/lib/api";

const MetricsTable = ({ res }) => {
  const best = Object.fromEntries(METRICS.map(([k]) => [k, Math.max(...res.models.map((m) => m.metrics[k].mean))]));
  return (
    <div className="overflow-x-auto">
      <table data-testid="results-metrics-table" className="w-full text-sm">
        <thead>
          <tr className="bg-[#F8F6F0] text-[11px] uppercase tracking-wider text-slate-500">
            <th className="text-left px-3 py-2">Model</th>
            {METRICS.map(([k, l]) => <th key={k} className="text-left px-3 py-2">{l}</th>)}
          </tr>
        </thead>
        <tbody className="font-mono text-[13px]">
          {res.models.map((m) => (
            <tr key={m.name} data-testid={`metrics-row-${m.name}`} className="border-b border-[#F0EBE1]">
              <td className="px-3 py-2.5 font-sans font-semibold flex items-center gap-2"><span className="w-2 h-2 rounded-full" style={{ background: MODEL_COLORS[m.name] }} />{m.name}</td>
              {METRICS.map(([k]) => (
                <td key={k} className={`px-3 py-2.5 whitespace-nowrap ${m.metrics[k].mean === best[k] ? "text-[#1E3A8A] font-bold" : ""}`}>
                  {pm(m.metrics[k])}
                  <span className="block text-[10px] text-slate-400">trial σ {fmt3(m.metrics[k].trial_std)}</span>
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      <p className="mt-2 text-xs text-slate-500">± = std across all {res.n_fits_per_model} folds; trial σ = std of the {res.n_repeats} per-shuffle means. Bold = best per metric. Positive class = at risk, threshold 0.5.</p>
    </div>
  );
};

const MetricBars = ({ res }) => {
  const data = METRICS.map(([k, l]) => ({
    metric: l,
    ...Object.fromEntries(res.models.flatMap((m) => [[m.name, m.metrics[k].mean], [`${m.name}_e`, m.metrics[k].std]])),
  }));
  return (
    <div className="h-[340px]" data-testid="chart-metric-bars">
      <ResponsiveContainer>
        <BarChart data={data} margin={{ left: 0 }}>
          <CartesianGrid stroke="#E2DCD2" strokeDasharray="3 3" vertical={false} />
          <XAxis dataKey="metric" tick={{ fontSize: 12 }} />
          <YAxis domain={[0.5, 1]} tick={{ fontSize: 11 }} allowDataOverflow />
          <Tooltip content={<ChartTip />} cursor={{ fill: "#F3EFE6" }} />
          <Legend wrapperStyle={{ fontSize: 12 }} />
          {res.models.map((m) => (
            <Bar key={m.name} dataKey={m.name} fill={MODEL_COLORS[m.name]} radius={[2, 2, 0, 0]}>
              <ErrorBar dataKey={`${m.name}_e`} width={3} stroke="#0F172A" strokeWidth={1} />
            </Bar>
          ))}
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};

const RocChart = ({ res }) => {
  const data = res.models[0].roc.fpr.map((x, i) => ({ fpr: x, chance: x, ...Object.fromEntries(res.models.map((m) => [m.name, m.roc.tpr[i]])) }));
  return (
    <div className="h-[340px]" data-testid="chart-roc">
      <ResponsiveContainer>
        <LineChart data={data}>
          <CartesianGrid stroke="#E2DCD2" strokeDasharray="3 3" />
          <XAxis dataKey="fpr" type="number" domain={[0, 1]} tick={{ fontSize: 11 }} label={{ value: "False positive rate", position: "insideBottom", offset: -2, fontSize: 11 }} />
          <YAxis domain={[0, 1]} tick={{ fontSize: 11 }} />
          <Tooltip content={<ChartTip labelPrefix="FPR " />} />
          <Legend wrapperStyle={{ fontSize: 11 }} />
          <Line dataKey="chance" name="Chance" stroke="#CBD5E1" strokeDasharray="4 4" dot={false} />
          {res.models.map((m) => <Line key={m.name} dataKey={m.name} stroke={MODEL_COLORS[m.name]} dot={false} strokeWidth={2} />)}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

const Confusion = ({ m }) => {
  const [[tn, fp], [fn, tp]] = m.confusion;
  const cell = (v, label, tone) => (
    <div className={`p-3 rounded ${tone}`}><p className="text-[10px] font-mono uppercase text-slate-500">{label}</p><p className="font-mono font-bold text-lg">{Math.round(v).toLocaleString()}</p></div>
  );
  return (
    <div data-testid={`confusion-${m.name}`} className="border border-[#E2DCD2] rounded-md p-3">
      <p className="text-xs font-semibold mb-2">{m.name}</p>
      <div className="grid grid-cols-2 gap-1.5">
        {cell(tn, "TN · ok→ok", "bg-[#F0FDF4]")}{cell(fp, "FP · ok→risk", "bg-[#FFFBEB]")}
        {cell(fn, "FN · risk→ok", "bg-[#FEF2F2]")}{cell(tp, "TP · risk→risk", "bg-[#EFF6FF]")}
      </div>
    </div>
  );
};

const Importance = ({ res }) => {
  const opts = res.models.filter((m) => m.importance);
  const [sel, setSel] = useState("XGBoost");
  const m = opts.find((o) => o.name === sel) || opts[0];
  const data = m.importance.slice(0, 15).map((d) => ({ ...d }));
  return (
    <div>
      <div className="flex gap-2 mb-3 flex-wrap">
        {opts.map((o) => (
          <button key={o.name} data-testid={`importance-tab-${o.name}`} onClick={() => setSel(o.name)}
            className={`text-xs px-3 py-1 rounded border transition-colors ${o.name === m.name ? "bg-[#1E3A8A] text-white border-[#1E3A8A]" : "border-[#E2DCD2] hover:bg-[#F3EFE6]"}`}>{o.name}</button>
        ))}
      </div>
      <div className="h-[420px]" data-testid="chart-importance">
        <ResponsiveContainer>
          <BarChart data={data} layout="vertical" margin={{ left: 40 }}>
            <CartesianGrid stroke="#E2DCD2" strokeDasharray="3 3" horizontal={false} />
            <XAxis type="number" tick={{ fontSize: 10 }} />
            <YAxis type="category" dataKey="feature" tick={{ fontSize: 10, fontFamily: "IBM Plex Mono" }} width={150} />
            <Tooltip content={<ChartTip />} cursor={{ fill: "#F3EFE6" }} />
            <Bar dataKey="value" name={m.name === "Logistic Regression" ? "Coefficient (std. units)" : "Importance"}>
              {data.map((d) => <Cell key={d.feature} fill={SOURCE_COLORS[d.source]} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      <p className="text-xs text-slate-500 flex gap-4 flex-wrap">
        {Object.entries(SOURCE_COLORS).map(([k, c]) => <span key={k} className="flex items-center gap-1"><span className="w-2.5 h-2.5 rounded-sm" style={{ background: c }} />{k}</span>)}
        <span>{m.name === "Logistic Regression" ? "Positive = raises at-risk odds." : "Mean impurity / gain importance over all folds."}</span>
      </p>
    </div>
  );
};

const Holdout = ({ rows }) => (
  <div className="overflow-x-auto">
    <table data-testid="holdout-table" className="w-full text-sm">
      <thead><tr className="bg-[#F8F6F0] text-[11px] uppercase tracking-wider text-slate-500"><th className="text-left px-3 py-2">Split</th><th className="text-left px-3 py-2">Model</th>{METRICS.map(([k, l]) => <th key={k} className="text-left px-3 py-2">{l}</th>)}</tr></thead>
      <tbody className="font-mono text-[12px]">
        {rows.map((r) => (
          <tr key={r.split + r.model} className="border-b border-[#F0EBE1]">
            <td className="px-3 py-1.5">{r.split}</td><td className="px-3 py-1.5 font-sans">{r.model}</td>
            {METRICS.map(([k]) => <td key={k} className="px-3 py-1.5 whitespace-nowrap">{pm(r[k])}</td>)}
          </tr>
        ))}
      </tbody>
    </table>
    <p className="mt-2 text-xs text-slate-500">Train/test, stratified, SMOTE on train only, 5 seeds per split.</p>
  </div>
);

const Body = () => {
  const { data } = useLatestRun();
  const run = data?.run?.full || data?.run?.preview ? data.run : data?.last_results_run;
  const res = run?.full || run?.preview;
  if (!res) return <p data-testid="results-empty" className="text-sm text-slate-500">No results yet — start a run on the Experiments page.</p>;
  return (
    <div className="space-y-6">
      <div data-testid="results-source-badge" className={`text-xs font-mono inline-block px-2 py-1 rounded ${run.full ? "bg-[#F0FDF4] text-[#166534]" : "bg-[#FFFBEB] text-[#D97706]"}`}>
        {run.full ? `FULL RUN · ${res.n_repeats} × ${res.n_folds}-fold CV` : `PREVIEW · ${res.n_repeats} × ${res.n_folds}-fold CV — full run in progress (${run.progress.done}/${run.progress.total})`}
      </div>
      <Panel testid="panel-metrics" title="Evaluation metrics · mean ± std"><MetricsTable res={res} /></Panel>
      <div className="grid xl:grid-cols-2 gap-6">
        <Panel testid="panel-bars" title="Metric comparison" caption="Error bars = ±1 std across folds."><MetricBars res={res} /></Panel>
        <Panel testid="panel-roc" title="Mean ROC curves" caption="Averaged over folds on a common FPR grid."><RocChart res={res} /></Panel>
      </div>
      <Panel testid="panel-confusion" title="Confusion matrices" caption="Counts averaged per full pass over the dataset (sum over folds ÷ shuffles).">
        <div className="grid sm:grid-cols-2 xl:grid-cols-5 gap-3">{res.models.map((m) => <Confusion key={m.name} m={m} />)}</div>
      </Panel>
      <div className="grid xl:grid-cols-5 gap-6">
        <Panel testid="panel-importance" title="Feature importance" className="xl:col-span-3"><Importance res={res} /></Panel>
        <Panel testid="panel-interpretation" title="Interpretation" className="xl:col-span-2">
          <div className="space-y-4 max-h-[520px] overflow-y-auto pr-2">
            {run.interpretation.map((p) => (
              <div key={p.heading} data-testid="interpretation-item">
                <h4 className="font-serif font-semibold text-slate-900">{p.heading}</h4>
                <p className="mt-1 text-sm text-slate-700 leading-relaxed">{p.text}</p>
              </div>
            ))}
          </div>
        </Panel>
      </div>
      {run.holdout && <Panel testid="panel-holdout" title="Different train/test splits"><Holdout rows={run.holdout} /></Panel>}
    </div>
  );
};

export default function Results() {
  return (
    <div>
      <PageHeader kicker="06 · Results" title="Model performance & interpretation">Binary at-risk classification at the 50% cut-off, five algorithms, SMOTE in training folds only.</PageHeader>
      <NeedsData><Body /></NeedsData>
    </div>
  );
}
