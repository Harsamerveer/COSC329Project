import { useEffect, useRef, useState } from "react";
import { Pause, Play, Trash2 } from "lucide-react";
import { api } from "@/lib/api";
import { PageHeader, Panel, Btn } from "@/components/common";

const SRC_COLOR = { data: "text-sky-300", features: "text-amber-300", experiments: "text-emerald-300", report: "text-violet-300" };
const FILTERS = ["all", "data", "features", "experiments", "report"];

export default function BackendLog() {
  const [events, setEvents] = useState([]);
  const [live, setLive] = useState(true);
  const [filter, setFilter] = useState("all");
  const since = useRef(null);
  const box = useRef(null);

  useEffect(() => {
    if (!live) return;
    let stop = false;
    const tick = async () => {
      const { data } = await api.get("/logs", { params: since.current ? { since: since.current } : { limit: 400 } });
      if (stop || !data.events.length) return;
      since.current = data.events[data.events.length - 1].ts;
      setEvents((e) => [...e, ...data.events].slice(-2000));
    };
    tick();
    const id = setInterval(tick, 2000);
    return () => { stop = true; clearInterval(id); };
  }, [live]);

  useEffect(() => { if (live && box.current) box.current.scrollTop = box.current.scrollHeight; }, [events, live]);

  const shown = filter === "all" ? events : events.filter((e) => e.source === filter);
  return (
    <div>
      <PageHeader kicker="05 · Backend log" title="What the backend is doing, step by step">
        Every backend step is written to MongoDB (<code className="font-mono text-xs">events</code> collection) and the server log: dataset download & preparation, feature building, each cross-validation fold with per-model scores, and report generation. Updates every 2 seconds.
      </PageHeader>
      <Panel testid="panel-backend-log">
        <div className="flex flex-wrap items-center gap-2 mb-4">
          {FILTERS.map((f) => (
            <button key={f} data-testid={`log-filter-${f}`} onClick={() => setFilter(f)}
              className={`text-xs font-mono px-2.5 py-1 rounded border transition-colors ${filter === f ? "bg-[#1E3A8A] text-white border-[#1E3A8A]" : "border-[#E2DCD2] text-slate-600 hover:bg-[#F3EFE6]"}`}>{f}</button>
          ))}
          <div className="ml-auto flex gap-2">
            <Btn variant="outline" data-testid="btn-log-toggle" onClick={() => setLive((l) => !l)}>{live ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}{live ? "Pause" : "Resume"}</Btn>
            <Btn variant="outline" data-testid="btn-log-clear" onClick={() => setEvents([])}><Trash2 className="w-4 h-4" />Clear view</Btn>
          </div>
        </div>
        <div ref={box} data-testid="log-console" className="log-scroll bg-[#0F172A] text-slate-200 rounded-md p-4 h-[65vh] overflow-y-auto font-mono text-[12px] leading-relaxed">
          {shown.length === 0 && <p className="text-slate-500">No events yet — load the dataset or run the experiments.</p>}
          {shown.map((e, i) => (
            <div key={i} data-testid="log-line" className="flex gap-3 hover:bg-white/5">
              <span className="text-slate-500 flex-shrink-0">{e.ts.slice(11, 19)}</span>
              <span className={`${SRC_COLOR[e.source] || "text-slate-300"} flex-shrink-0 w-24`}>[{e.source}]</span>
              <span className={e.level === "error" ? "text-red-400" : ""}>{e.message}</span>
            </div>
          ))}
        </div>
        <p className="mt-2 text-xs text-slate-500 font-mono">{shown.length} events shown · times in UTC</p>
      </Panel>
    </div>
  );
}
