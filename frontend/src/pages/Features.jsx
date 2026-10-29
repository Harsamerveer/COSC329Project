import { Download } from "lucide-react";
import { useFeatureInfo, useReadyQuery } from "@/hooks/useOulad";
import { PageHeader, Panel, Stat, NeedsData } from "@/components/common";
import { API_BASE, fmtInt, SOURCE_COLORS } from "@/lib/api";

const SourceTag = ({ s }) => (
  <span className="text-[10px] font-mono uppercase tracking-wider px-1.5 py-0.5 rounded border" style={{ color: SOURCE_COLORS[s], borderColor: SOURCE_COLORS[s] + "55" }}>{s}</span>
);

const FeatureTable = ({ features }) => (
  <div className="overflow-x-auto">
    <table data-testid="feature-definitions-table" className="w-full text-sm">
      <thead>
        <tr className="bg-[#F8F6F0] text-[11px] uppercase tracking-wider text-slate-500">
          {["Feature", "Source", "Definition", "Mean · at risk", "Mean · not at risk"].map((h) => <th key={h} className="text-left px-3 py-2 font-semibold">{h}</th>)}
        </tr>
      </thead>
      <tbody>
        {features.map((f) => (
          <tr key={f.feature} className="border-b border-[#F0EBE1] align-top">
            <td className="px-3 py-2 font-mono text-[12px] text-slate-900 whitespace-nowrap">{f.feature}</td>
            <td className="px-3 py-2"><SourceTag s={f.source} /></td>
            <td className="px-3 py-2 text-slate-600 text-[13px] min-w-[280px]">{f.description}</td>
            <td className="px-3 py-2 font-mono text-[12px] text-[#DC2626]">{f.at_risk_mean}</td>
            <td className="px-3 py-2 font-mono text-[12px] text-[#166534]">{f.not_at_risk_mean}</td>
          </tr>
        ))}
      </tbody>
    </table>
  </div>
);

const Sample = () => {
  const { data } = useReadyQuery(["sample"], "/features/sample?n=15");
  if (!data) return <p className="text-sm text-slate-500">Loading sample…</p>;
  return (
    <div className="overflow-x-auto border border-[#E2DCD2] rounded">
      <table data-testid="feature-sample-table" className="text-[11px] font-mono whitespace-nowrap">
        <thead><tr className="bg-[#F8F6F0] text-slate-500">{data.columns.map((c) => <th key={c} className="px-2 py-1.5 text-left font-semibold">{c}</th>)}</tr></thead>
        <tbody>
          {data.rows.map((r, i) => (
            <tr key={i} className={`border-t border-[#F0EBE1] ${r[r.length - 1] === 1 ? "bg-[#FEF2F2]" : ""}`}>
              {r.map((v, j) => <td key={j} className="px-2 py-1">{v === null ? "—" : String(v)}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

const Body = () => {
  const { data } = useFeatureInfo();
  if (!data) return <p className="text-sm text-slate-500">Building feature vectors (≈5 s)…</p>;
  const st = data.stats;
  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <Stat testid="stat-samples" label="Students in dataset" value={fmtInt(st.n_samples)} sub={`of ${fmtInt(st.n_registrations)} registrations`} />
        <Stat testid="stat-excluded" label="Excluded" value={fmtInt(st.excluded_withdrawn_early)} sub="withdrew before cut-off" />
        <Stat testid="stat-at-risk-rate" label="At-risk rate" value={`${(st.at_risk_rate * 100).toFixed(1)}%`} sub={`${fmtInt(st.at_risk)} vs ${fmtInt(st.not_at_risk)}`} tone="text-[#DC2626]" />
        <Stat testid="stat-n-features" label="Features" value={st.n_features} sub={Object.entries(data.source_counts).map(([k, v]) => `${v} ${k}`).join(" · ")} />
      </div>
      <Panel testid="panel-tradeoffs" title="Feature choices & tradeoffs" caption="Every decision below affects what the classifiers can learn — and whether the evaluation is honest.">
        <div className="grid md:grid-cols-2 gap-x-8 gap-y-5">
          {data.tradeoffs.map((t, i) => (
            <article key={t.title} data-testid={`tradeoff-${i}`} className="border-l-2 border-[#1E3A8A] pl-4">
              <h4 className="font-serif font-semibold text-slate-900">{t.title}</h4>
              <p className="mt-1 text-sm text-slate-700 leading-relaxed">{t.choice}</p>
              <p className="mt-1 text-sm text-slate-500 leading-relaxed"><span className="font-semibold text-[#D97706]">Tradeoff · </span>{t.tradeoff}</p>
            </article>
          ))}
        </div>
      </Panel>
      <Panel testid="panel-feature-defs" title="Feature vector definition" caption="Group means show the raw separation between at-risk and not-at-risk students at the 50% cut-off.">
        <FeatureTable features={data.features} />
      </Panel>
      <Panel testid="panel-feature-sample" title="Sample feature vectors" caption="15 random students; red rows are at risk (target = 1).">
        <Sample />
        <a data-testid="btn-download-features-csv" href={`${API_BASE}/features/csv`} className="mt-4 inline-flex items-center gap-2 text-sm text-[#1E3A8A] hover:underline">
          <Download className="w-4 h-4" /> Download full feature matrix (CSV)
        </a>
      </Panel>
    </div>
  );
};

export default function Features() {
  return (
    <div>
      <PageHeader kicker="03 · Features" title="From click streams to feature vectors">
        One fixed-length vector per student-presentation, computed strictly from data available before the midterm cut-off.
      </PageHeader>
      <NeedsData><Body /></NeedsData>
    </div>
  );
}
