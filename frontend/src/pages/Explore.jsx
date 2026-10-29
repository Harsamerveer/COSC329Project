import { useState } from "react";
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine, Legend } from "recharts";
import { useSummary, useReadyQuery } from "@/hooks/useOulad";
import { PageHeader, Panel, NeedsData, ChartTip } from "@/components/common";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { fmtInt, OUTCOME_COLORS } from "@/lib/api";

const OUTCOMES = ["Pass", "Distinction", "Fail", "Withdrawn"];

const WeeklyChart = ({ data, cutoffWeek, assessments, testid }) => (
  <div className="h-[300px]" data-testid={testid}>
    <ResponsiveContainer>
      <LineChart data={data} margin={{ left: 0, right: 12 }}>
        <CartesianGrid stroke="#E2DCD2" strokeDasharray="3 3" />
        <XAxis dataKey="week" tick={{ fontSize: 11 }} label={{ value: "week", position: "insideBottomRight", offset: -2, fontSize: 11 }} />
        <YAxis tick={{ fontSize: 11 }} />
        <Tooltip content={<ChartTip labelPrefix="week " />} />
        <Legend wrapperStyle={{ fontSize: 12 }} />
        {assessments.filter((a) => a.date != null && a.type !== "Exam").map((a) => (
          <ReferenceLine key={a.id_assessment} x={Math.floor(a.date / 7)} stroke="#CBD5E1" strokeDasharray="2 3" />
        ))}
        <ReferenceLine x={cutoffWeek} stroke="#1E3A8A" strokeWidth={2} label={{ value: "cut-off", fill: "#1E3A8A", fontSize: 11, position: "top" }} />
        {OUTCOMES.map((o) => <Line key={o} type="monotone" dataKey={o} stroke={OUTCOME_COLORS[o]} dot={false} strokeWidth={2} />)}
      </LineChart>
    </ResponsiveContainer>
  </div>
);

const CourseView = ({ course }) => {
  const { data: d } = useReadyQuery(["explore", course], `/explore/${course}`, !!course);
  if (!d) return <p className="text-sm text-slate-500">Loading course…</p>;
  const cw = Math.floor(d.cutoff / 7);
  return (
    <div className="space-y-6">
      <div className="grid lg:grid-cols-2 gap-6">
        <Panel testid="panel-weekly-material" title="Engagement · material clicks per student per week" caption={`Grouped by final result. Blue line = 50% cut-off (day ${d.cutoff} of ${d.length}); grey dashes = assessment deadlines.`}>
          <WeeklyChart data={d.weekly_material} cutoffWeek={cw} assessments={d.assessments} testid="chart-weekly-material" />
        </Panel>
        <Panel testid="panel-weekly-forum" title="Interaction · forum clicks per student per week" caption="Forum (forumng) activity is far sparser and decays faster than material use.">
          <WeeklyChart data={d.weekly_forum} cutoffWeek={cw} assessments={d.assessments} testid="chart-weekly-forum" />
        </Panel>
      </div>
      <div className="grid lg:grid-cols-5 gap-6">
        <Panel testid="panel-assessments" title="Assessment schedule & scores" caption="Mean score of submitted work by eventual outcome group." className="lg:col-span-3">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="bg-[#F8F6F0] text-[11px] uppercase tracking-wider text-slate-500">
                  {["Type", "Due day", "Weight", "Submitted", "Mean", "At risk", "Not at risk"].map((h) => <th key={h} className="text-left px-3 py-2 font-semibold">{h}</th>)}
                </tr>
              </thead>
              <tbody className="font-mono text-[13px]">
                {d.assessments.map((a) => (
                  <tr key={a.id_assessment} className={`border-b border-[#F0EBE1] ${a.date != null && a.date < d.cutoff ? "" : "text-slate-400"}`}>
                    <td className="px-3 py-2 font-sans">{a.type}</td>
                    <td className="px-3 py-2">{a.date ?? "—"}</td>
                    <td className="px-3 py-2">{a.weight}</td>
                    <td className="px-3 py-2">{fmtInt(a.n_submitted)}</td>
                    <td className="px-3 py-2">{a.mean_score ?? "—"}</td>
                    <td className="px-3 py-2 text-[#DC2626]">{a.mean_score_at_risk ?? "—"}</td>
                    <td className="px-3 py-2 text-[#166534]">{a.mean_score_not_at_risk ?? "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="mt-2 text-xs text-slate-500">Greyed rows fall after the cut-off and are not used as features.</p>
          </div>
        </Panel>
        <Panel testid="panel-activity" title="Clicks by activity type" className="lg:col-span-2">
          <div className="h-[320px]">
            <ResponsiveContainer>
              <BarChart data={d.activity.slice(0, 12)} layout="vertical" margin={{ left: 20 }}>
                <CartesianGrid stroke="#E2DCD2" strokeDasharray="3 3" horizontal={false} />
                <XAxis type="number" tick={{ fontSize: 10 }} />
                <YAxis type="category" dataKey="type" tick={{ fontSize: 11 }} width={80} />
                <Tooltip content={<ChartTip />} cursor={{ fill: "#F3EFE6" }} />
                <Bar dataKey="clicks" name="Clicks" fill="#1E3A8A" radius={[0, 3, 3, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Panel>
      </div>
    </div>
  );
};

const Body = () => {
  const { data: s } = useSummary();
  const [course, setCourse] = useState("");
  const cur = course || s?.courses?.[0]?.course;
  if (!s) return null;
  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        <span className="text-xs font-mono uppercase tracking-widest text-slate-500">Presentation</span>
        <Select value={cur} onValueChange={setCourse}>
          <SelectTrigger data-testid="select-course" className="w-56 bg-white"><SelectValue /></SelectTrigger>
          <SelectContent className="bg-white">
            {s.courses.map((c) => (
              <SelectItem key={c.course} value={c.course} data-testid={`select-course-${c.course}`}>
                {c.course.replace("_", " · ")} ({fmtInt(c.n_students)})
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
      <CourseView course={cur} />
    </div>
  );
};

export default function Explore() {
  return (
    <div>
      <PageHeader kicker="02 · Explore" title="Courses, students & activity over time">
        Each presentation's click streams split by eventual outcome. Separation between the green/blue (pass) and red/grey (at-risk) curves well before the cut-off is what makes early prediction possible.
      </PageHeader>
      <NeedsData><Body /></NeedsData>
    </div>
  );
}
