import { Download, FileText } from "lucide-react";
import { useLatestRun } from "@/hooks/useOulad";
import { PageHeader, Panel, NeedsData } from "@/components/common";
import { API_BASE } from "@/lib/api";

const SECTIONS = [
  "Dataset description and per-course outcome table",
  "Feature definitions with group means, plus every choice & tradeoff",
  "Methodology: pipeline, SMOTE-in-training-only, 25 × 10-fold CV, hold-out splits",
  "Results table (mean ± std), bar chart with error bars, ROC curves, confusion matrices",
  "Feature importance (RF, XGBoost) and logistic-regression coefficients",
  "Interpretation of each algorithm on this data, and limitations",
];

const Body = () => {
  const { data } = useLatestRun();
  const run = data?.run;
  const has = run?.full || run?.preview;
  return (
    <div className="grid lg:grid-cols-3 gap-6">
      <Panel testid="panel-report" title="Downloadable PDF report" className="lg:col-span-2">
        <ul className="space-y-2 text-sm text-slate-700">
          {SECTIONS.map((s) => <li key={s} className="flex gap-2"><FileText className="w-4 h-4 text-[#1E3A8A] mt-0.5 flex-shrink-0" />{s}</li>)}
        </ul>
        <div className="mt-6 flex flex-wrap gap-3 items-center">
          {has ? (
            <a data-testid="btn-download-report" href={`${API_BASE}/report/pdf`} className="bg-[#1E3A8A] hover:bg-[#1E40AF] text-white font-medium px-4 py-2 rounded-md text-sm inline-flex items-center gap-2 transition-colors">
              <Download className="w-4 h-4" /> Download report (PDF)
            </a>
          ) : <p data-testid="report-unavailable" className="text-sm text-slate-500">Run the experiments first — the report uses the latest results.</p>}
          <a data-testid="btn-download-csv" href={`${API_BASE}/features/csv`} className="border border-[#CBD5E1] hover:bg-[#F8F6F0] text-slate-700 font-medium px-4 py-2 rounded-md text-sm inline-flex items-center gap-2"><Download className="w-4 h-4" /> Feature matrix (CSV)</a>
        </div>
        {has && <p className="mt-3 text-xs text-slate-500">Uses the {run.full ? "full 25-shuffle" : "3-shuffle preview"} results{!run.full && " — download again after the full run completes"}.</p>}
      </Panel>
      <Panel testid="panel-report-note" title="Reproducibility">
        <p className="text-sm text-slate-600 leading-relaxed">All random seeds are fixed (CV shuffles start from seed 42; SMOTE and model seeds follow the fold index), so re-running produces identical numbers. Raw OULAD tables are downloaded from the UCI Machine Learning Repository mirror of the Open University release.</p>
      </Panel>
    </div>
  );
};

export default function Report() {
  return (
    <div>
      <PageHeader kicker="07 · Report" title="Research report">Methodology, tradeoffs, results and interpretation in one document.</PageHeader>
      <NeedsData><Body /></NeedsData>
    </div>
  );
}
