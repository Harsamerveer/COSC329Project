import { Loader2 } from "lucide-react";
import { useDatasetStatus, useLoadDataset } from "@/hooks/useOulad";

export const Panel = ({ title, caption, children, className = "", testid }) => (
  <section data-testid={testid} className={`bg-white border border-[#E2DCD2] rounded-md p-5 sm:p-6 min-w-0 hover:border-[#CBD5E1] transition-colors duration-200 ${className}`}>
    {title && (
      <header className="mb-4">
        <h3 className="text-xs sm:text-sm font-bold uppercase tracking-wider text-slate-900">{title}</h3>
        {caption && <p className="mt-1 text-xs text-slate-500 leading-relaxed">{caption}</p>}
      </header>
    )}
    {children}
  </section>
);

export const PageHeader = ({ kicker, title, children }) => (
  <header className="mb-8 fade-up">
    <p className="text-[11px] font-mono uppercase tracking-[0.2em] text-[#1E3A8A] mb-2">{kicker}</p>
    <h1 className="text-2xl sm:text-3xl font-serif font-bold text-slate-900 tracking-tight">{title}</h1>
    {children && <div className="mt-3 max-w-3xl text-sm sm:text-base text-slate-600 leading-relaxed">{children}</div>}
  </header>
);

export const Stat = ({ label, value, sub, testid, tone = "text-slate-900" }) => (
  <div data-testid={testid} className="bg-white border border-[#E2DCD2] rounded-md p-4 sm:p-5">
    <p className="text-[10px] font-mono uppercase tracking-widest text-slate-500">{label}</p>
    <p className={`mt-2 text-2xl sm:text-3xl font-mono font-bold tracking-tight ${tone}`}>{value}</p>
    {sub && <p className="mt-1 text-xs text-slate-500">{sub}</p>}
  </div>
);

export const Btn = ({ variant = "primary", className = "", children, ...p }) => {
  const v = {
    primary: "bg-[#1E3A8A] hover:bg-[#1E40AF] text-white",
    secondary: "bg-[#F3EFE6] hover:bg-[#E2DCD2] text-slate-800 border border-[#E2DCD2]",
    outline: "border border-[#CBD5E1] hover:bg-[#F8F6F0] text-slate-700",
  }[variant];
  return (
    <button {...p} className={`${v} font-medium px-4 py-2 rounded-md text-sm inline-flex items-center gap-2 transition-colors disabled:opacity-50 disabled:cursor-not-allowed ${className}`}>
      {children}
    </button>
  );
};

export const LoadDatasetButton = () => {
  const load = useLoadDataset();
  const { data: st } = useDatasetStatus();
  const busy = st && ["downloading", "preparing"].includes(st.status);
  return (
    <Btn data-testid="btn-load-dataset" onClick={() => load.mutate()} disabled={busy || st?.ready || load.isPending}>
      {busy && <Loader2 className="w-4 h-4 animate-spin" />}
      {st?.ready ? "Dataset loaded" : busy ? "Preparing…" : "Load dataset"}
    </Btn>
  );
};

export const NeedsData = ({ children }) => {
  const { data: st } = useDatasetStatus();
  if (st?.ready) return children;
  return (
    <div data-testid="needs-data-notice" className="border border-dashed border-[#CBD5E1] bg-[#F8F6F0] rounded-md p-8 max-w-xl">
      <p className="font-serif text-lg text-slate-800">The dataset has not been prepared yet.</p>
      <p className="mt-2 text-sm text-slate-600">Load OULAD (7 tables, ~10.6M click records) to unlock this page.</p>
      {st && st.status !== "idle" && (
        <p className="mt-3 text-xs font-mono text-slate-500">{st.message} · {Math.round((st.progress || 0) * 100)}%</p>
      )}
      <div className="mt-5"><LoadDatasetButton /></div>
    </div>
  );
};

export const ChartTip = ({ active, payload, label, labelPrefix = "" }) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-white border border-[#E2DCD2] p-3 rounded-md shadow-md text-xs text-slate-800">
      <p className="font-mono text-slate-500 mb-1">{labelPrefix}{label}</p>
      {payload.map((p) => (
        <p key={p.dataKey} className="flex gap-2 items-center">
          <span className="w-2 h-2 rounded-full" style={{ background: p.color || p.fill }} />
          {p.name}: <span className="font-mono">{typeof p.value === "number" ? p.value.toLocaleString("en-US", { maximumFractionDigits: 3 }) : p.value}</span>
        </p>
      ))}
    </div>
  );
};
