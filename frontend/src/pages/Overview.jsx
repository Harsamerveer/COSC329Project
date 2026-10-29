import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from "recharts";
import { useSummary } from "@/hooks/useOulad";
import { PageHeader, Panel, Stat, NeedsData, LoadDatasetButton, ChartTip } from "@/components/common";
import { fmtInt, OUTCOME_COLORS } from "@/lib/api";

const CourseTable = ({ courses }) => (
  <div className="overflow-x-auto">
    <table data-testid="courses-table" className="w-full text-sm border-collapse">
      <thead>
        <tr className="bg-[#F8F6F0] text-[11px] uppercase tracking-wider text-slate-500">
          {["Presentation", "Days", "Students", "Pass", "Distinction", "Fail", "Withdrawn", "At-risk %", "Assessments", "Clicks"].map((h) => (
            <th key={h} className="text-left font-semibold px-3 py-2 border-b border-[#E2DCD2]">{h}</th>
          ))}
        </tr>
      </thead>
      <tbody className="font-mono text-[13px]">
        {courses.map((c) => (
          <tr key={c.course} className="border-b border-[#F0EBE1] hover:bg-[#FBF9F5]">
            <td className="px-3 py-2 font-sans font-semibold text-slate-900">{c.course.replace("_", " · ")}</td>
            <td className="px-3 py-2">{c.length}</td>
            <td className="px-3 py-2">{fmtInt(c.n_students)}</td>
            <td className="px-3 py-2">{fmtInt(c.Pass)}</td>
            <td className="px-3 py-2">{fmtInt(c.Distinction)}</td>
            <td className="px-3 py-2">{fmtInt(c.Fail)}</td>
            <td className="px-3 py-2">{fmtInt(c.Withdrawn)}</td>
            <td className="px-3 py-2 text-[#DC2626]">{(((c.Fail + c.Withdrawn) / c.n_students) * 100).toFixed(1)}</td>
            <td className="px-3 py-2">{c.n_assessments}</td>
            <td className="px-3 py-2">{fmtInt(c.total_clicks)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  </div>
);

const Body = () => {
  const { data: s } = useSummary();
  if (!s) return <p className="text-sm text-slate-500">Loading summary…</p>;
  const outcomes = Object.entries(s.outcomes).map(([k, v]) => ({ name: k, value: v }));
  const atRisk = s.outcomes.Fail + s.outcomes.Withdrawn;
  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-6 gap-3">
        <Stat testid="stat-registrations" label="Registrations" value={fmtInt(s.n_registrations)} sub={`${fmtInt(s.n_students)} unique students`} />
        <Stat testid="stat-courses" label="Modules / presentations" value={`${s.n_modules} / ${s.n_presentations}`} sub="2013B – 2014J" />
        <Stat testid="stat-assessments" label="Assessments" value={s.n_assessments} sub={`${fmtInt(s.n_assessment_submissions)} submissions`} />
        <Stat testid="stat-vle-records" label="VLE records" value={`${(s.n_vle_records / 1e6).toFixed(1)}M`} sub={`${fmtInt(s.n_materials)} materials`} />
        <Stat testid="stat-clicks" label="Total clicks" value={`${(s.total_clicks / 1e6).toFixed(1)}M`} sub="engagement + forum" />
        <Stat testid="stat-at-risk" label="At-risk share" value={`${((atRisk / s.n_registrations) * 100).toFixed(1)}%`} sub="Fail + Withdrawn" tone="text-[#DC2626]" />
      </div>
      <div className="grid lg:grid-cols-5 gap-6">
        <Panel testid="panel-outcomes" title="Final result distribution" caption="Class imbalance before any filtering; this is why SMOTE is applied to the training folds." className="lg:col-span-3">
          <div className="h-[280px]">
            <ResponsiveContainer>
              <BarChart data={outcomes} margin={{ left: 8 }}>
                <CartesianGrid stroke="#E2DCD2" strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="name" tick={{ fontSize: 12 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip content={<ChartTip />} cursor={{ fill: "#F3EFE6" }} />
                <Bar dataKey="value" name="Students" radius={[3, 3, 0, 0]}>
                  {outcomes.map((o) => <Cell key={o.name} fill={OUTCOME_COLORS[o.name]} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Panel>
        <Panel testid="panel-task" title="The prediction task" className="lg:col-span-2">
          <div className="text-sm text-slate-700 leading-relaxed space-y-3">
            <p><span className="font-semibold text-[#DC2626]">At risk</span> = final result <em>Fail</em> or <em>Withdrawn</em>. <span className="font-semibold text-[#166534]">Not at risk</span> = <em>Pass</em> or <em>Distinction</em>.</p>
            <p>We simulate a <strong>midterm cut-off at 50%</strong> of each presentation's length: only clicks and assessments available before that day become features.</p>
            <p>Students who already withdrew before the cut-off are excluded — their outcome is known, so predicting it would be leakage.</p>
            <p className="text-xs text-slate-500">Engagement = course-material clicks · Interaction = forum clicks · Assessment = TMA/CMA submissions & scores.</p>
          </div>
        </Panel>
      </div>
      <Panel testid="panel-courses" title="Course presentations" caption="Seven modules (AAA–GGG), each offered in up to four presentations (B = February start, J = October start).">
        <CourseTable courses={s.courses} />
      </Panel>
    </div>
  );
};

export default function Overview() {
  return (
    <div>
      <PageHeader kicker="01 · Overview" title="Open University Learning Analytics Dataset">
        Early-warning prediction of students at risk of failing or withdrawing, built from VLE engagement, forum interaction and assessment records.
        <div className="mt-4"><LoadDatasetButton /></div>
      </PageHeader>
      <NeedsData><Body /></NeedsData>
    </div>
  );
}
