import { Link } from "react-router-dom";
import { Loader2, Play, Terminal } from "lucide-react";
import { toast } from "sonner";
import { useLatestRun, useRunExperiments } from "@/hooks/useOulad";
import { PageHeader, Panel, NeedsData, Btn } from "@/components/common";
import { Progress } from "@/components/ui/progress";
import { pm, MODEL_COLORS } from "@/lib/api";

const STEPS = [
  ["Build features", "29 features per student at the 50% cut-off"],
  ["Split", "RepeatedStratifiedKFold — 10 folds × N shuffles, seeded"],
  ["Scale", "StandardScaler fitted on the training fold"],
  ["SMOTE", "Synthetic at-risk samples created in the training fold only"],
  ["Fit 5 models", "RF · XGBoost · LR · linear SVM · Naive Bayes"],
  ["Evaluate", "Precision, recall, F1, ROC-AUC on the untouched test fold"],
];

const Preview = ({ res, label }) => (
  <div className="grid sm:grid-cols-2 xl:grid-cols-5 gap-3">
    {res.models.map((m) => (
      <div key={m.name} data-testid={`preview-card-${m.name}`} className="border border-[#E2DCD2] rounded-md p-4 bg-white">
        <p className="text-xs font-semibold flex items-center gap-2"><span className="w-2 h-2 rounded-full" style={{ background: MODEL_COLORS[m.name] }} />{m.name}</p>
        <p className="mt-2 text-[10px] font-mono uppercase text-slate-500">ROC-AUC</p>
        <p className="font-mono font-bold text-xl">{pm(m.metrics.roc_auc)}</p>
        <p className="mt-1 text-[11px] font-mono text-slate-500">F1 {pm(m.metrics.f1)}</p>
      </div>
    ))}
    <p className="sm:col-span-2 xl:col-span-5 text-xs text-slate-500">{label}</p>
  </div>
);

const Body = () => {
  const { data } = useLatestRun();
  const runMut = useRunExperiments();
  const run = data?.run;
  const active = run && ["queued", "running"].includes(run.status);
  const pct = run?.progress?.total ? (run.progress.done / run.progress.total) * 100 : 0;
  const start = () => runMut.mutate(undefined, { onSuccess: () => toast.success("Experiment run started") });
  return (
    <div className="space-y-6">
      <Panel testid="panel-run" title="Run benchmark">
        <div className="flex flex-wrap items-center gap-3">
          <Btn data-testid="btn-run-experiments" onClick={start} disabled={active || runMut.isPending}>
            {active ? <Loader2 className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
            {active ? "Running…" : run ? "Re-run experiments" : "Run experiments"}
          </Btn>
          <Link to="/logs" data-testid="link-backend-log"><Btn variant="outline"><Terminal className="w-4 h-4" /> Watch backend log</Btn></Link>
        </div>
        {run && (
          <div className="mt-5 space-y-2" data-testid="run-progress">
            <div className="flex justify-between text-xs font-mono text-slate-600">
              <span data-testid="run-stage">{run.stage}</span>
              <span>{run.progress.done}/{run.progress.total} fold jobs · {pct.toFixed(0)}%</span>
            </div>
            <Progress value={pct} className="h-2 bg-[#F3EFE6]" />
            <p className="text-xs text-slate-500">Status: <span className="font-mono" data-testid="run-status">{run.status}</span>{run.error && <span className="text-[#DC2626]"> — {run.error}</span>} · Quick preview (3 shuffles) appears first; the 25-shuffle run continues in the background (≈25 min).</p>
          </div>
        )}
      </Panel>
      {run?.preview && !run?.full && <Panel testid="panel-preview" title="Quick preview results"><Preview res={run.preview} label="3 × 10-fold CV. Full results replace these when the 25-shuffle run completes — see the Results page." /></Panel>}
      {run?.full && <Panel testid="panel-full-summary" title="Full run complete"><Preview res={run.full} label={`25 × 10-fold CV (${run.full.n_fits_per_model} fits per model). Open Results for the full analysis.`} /></Panel>}
      <div className="grid lg:grid-cols-2 gap-6">
        <Panel testid="panel-pipeline" title="Per-fold pipeline">
          <ol className="space-y-3">
            {STEPS.map(([t, d], i) => (
              <li key={t} className="flex gap-3">
                <span className="font-mono text-xs w-6 h-6 rounded-full bg-[#EFF6FF] text-[#1E3A8A] flex items-center justify-center flex-shrink-0">{i + 1}</span>
                <div><p className="text-sm font-semibold text-slate-900">{t}</p><p className="text-xs text-slate-500">{d}</p></div>
              </li>
            ))}
          </ol>
        </Panel>
        <Panel testid="panel-models" title="Classifiers & settings" caption="Fixed, light hyper-parameters — no tuning, so differences reflect the algorithms themselves.">
          <ul className="space-y-3 text-sm">
            {[["Random Forest", "150 trees, min_samples_leaf=2"], ["XGBoost", "250 trees, depth 5, lr 0.08, subsample 0.9"], ["Logistic Regression", "L2, C=1.0"], ["SVM (linear)", "LinearSVC C=0.5 + Platt calibration (tradeoff: linear kernel for speed)"], ["Naive Bayes", "Gaussian"]].map(([n, h]) => (
              <li key={n} className="flex gap-3"><span className="w-2 h-2 mt-1.5 rounded-full flex-shrink-0" style={{ background: MODEL_COLORS[n] }} /><span><strong>{n}</strong> <span className="text-slate-500">— {h}</span></span></li>
            ))}
          </ul>
          <p className="mt-4 text-xs text-slate-500">Also: hold-out comparison with 60/40, 70/30 and 80/20 stratified splits × 5 seeds.</p>
        </Panel>
      </div>
    </div>
  );
};

export default function Experiments() {
  return (
    <div>
      <PageHeader kicker="04 · Experiments" title="Repeated 10-fold cross-validation with SMOTE">
        25 independent shuffles × 10 folds = 250 train/test evaluations per algorithm. SMOTE balances only the training portion; every test fold keeps the real ~37% at-risk rate.
      </PageHeader>
      <NeedsData><Body /></NeedsData>
    </div>
  );
}
