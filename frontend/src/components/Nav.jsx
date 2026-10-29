import { NavLink } from "react-router-dom";
import { LayoutDashboard, BarChart3, SlidersHorizontal, PlayCircle, Award, FileText, Terminal } from "lucide-react";
import { useDatasetStatus, useLatestRun } from "@/hooks/useOulad";

const NAV = [
  ["/", "overview", "Overview", LayoutDashboard],
  ["/explore", "explore", "Explore", BarChart3],
  ["/features", "features", "Features", SlidersHorizontal],
  ["/experiments", "experiments", "Experiments", PlayCircle],
  ["/results", "results", "Results", Award],
  ["/logs", "logs", "Backend log", Terminal],
  ["/report", "report", "Report", FileText],
];

export const Sidebar = ({ onNavigate }) => (
  <nav className="flex flex-col gap-1">
    {NAV.map(([to, id, label, Icon]) => (
      <NavLink
        key={id}
        to={to}
        end={to === "/"}
        onClick={onNavigate}
        data-testid={`nav-item-${id}`}
        className={({ isActive }) =>
          `flex items-center gap-3 px-3 py-2 rounded-md text-sm transition-colors ${
            isActive ? "bg-white text-[#1E3A8A] font-semibold border border-[#E2DCD2]" : "text-slate-600 hover:bg-[#EAE3D6] border border-transparent"
          }`
        }
      >
        <Icon className="w-4 h-4" strokeWidth={1.75} />
        {label}
      </NavLink>
    ))}
  </nav>
);

export const StatusBar = () => {
  const { data: st } = useDatasetStatus();
  const { data } = useLatestRun();
  const run = data?.run;
  const dsTone = st?.ready ? "bg-[#F0FDF4] text-[#166534]" : st?.status === "error" ? "bg-[#FEF2F2] text-[#DC2626]" : "bg-[#FFFBEB] text-[#D97706]";
  const pct = run?.progress?.total ? Math.round((run.progress.done / run.progress.total) * 100) : 0;
  return (
    <div className="flex items-center gap-2 flex-wrap text-[11px] font-mono">
      <span data-testid="status-dataset" className={`px-2 py-1 rounded ${dsTone}`}>
        DATA · {st?.ready ? "READY" : (st?.status || "…").toUpperCase()}
      </span>
      {run && (
        <span data-testid="status-run" className="px-2 py-1 rounded bg-[#EFF6FF] text-[#1E3A8A]">
          RUN · {run.status.toUpperCase()} {run.status === "running" ? `${pct}%` : ""}
        </span>
      )}
    </div>
  );
};
