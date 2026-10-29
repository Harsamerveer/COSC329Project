import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import cm  # noqa: E402
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle  # noqa: E402

from . import content, data, features  # noqa: E402

PALETTE = ["#1E3A8A", "#D97706", "#0284C7", "#166534", "#DC2626"]
METRIC_LABELS = {"precision": "Precision", "recall": "Recall", "f1": "F1", "roc_auc": "ROC-AUC"}


def _img(fig, w=16):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    ratio = fig.get_size_inches()[1] / fig.get_size_inches()[0]
    return Image(buf, width=w * cm, height=w * cm * ratio)


def _table(rows, widths=None):
    t = Table(rows, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F3EFE6")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#E2DCD2")),
        ("FONTSIZE", (0, 0), (-1, -1), 7.5), ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    return t


def _bar_chart(models):
    fig, ax = plt.subplots(figsize=(8, 3.6))
    w = 0.16
    for i, m in enumerate(models):
        means = [m["metrics"][k]["mean"] for k in METRIC_LABELS]
        stds = [m["metrics"][k]["std"] for k in METRIC_LABELS]
        ax.bar([j + i * w for j in range(4)], means, w, yerr=stds, capsize=2, label=m["name"], color=PALETTE[i])
    ax.set_xticks([j + 2 * w for j in range(4)])
    ax.set_xticklabels(METRIC_LABELS.values())
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7, ncol=5, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    ax.grid(axis="y", alpha=0.3)
    return _img(fig)


def _roc_chart(models):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for i, m in enumerate(models):
        ax.plot(m["roc"]["fpr"], m["roc"]["tpr"], color=PALETTE[i],
                label=f"{m['name']} (AUC {m['metrics']['roc_auc']['mean']:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="#94A3B8")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(fontsize=7)
    return _img(fig, 12)


def _imp_chart(m):
    top = m["importance"][:15][::-1]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    cmap = {"engagement": "#1E3A8A", "interaction": "#D97706", "assessment": "#166534"}
    ax.barh([t["feature"] for t in top], [t["value"] for t in top], color=[cmap[t["source"]] for t in top])
    ax.set_title(f"{m['name']} - top features", fontsize=9)
    ax.tick_params(labelsize=7)
    return _img(fig, 14)


def build_pdf(run):
    ss = getSampleStyleSheet()
    H1, H2, P = ss["Title"], ss["Heading2"], ss["BodyText"]
    P.fontSize, P.leading = 9, 12.5
    s = []
    summ = data.summary()
    finfo = features.info(run["config"]["cutoff_fraction"])
    res = run.get("full") or run.get("preview")
    st = finfo["stats"]
    s += [Paragraph("Predicting At-Risk Students in OULAD", H1),
          Paragraph(f"Binary at-risk classification (Fail or Withdrawn vs. Pass or Distinction) at the 50% course cut-off. "
                    f"Results: {'full' if run.get('full') else 'preview'} run, {res['n_repeats']} x {res['n_folds']}-fold stratified CV, "
                    "SMOTE applied to training folds only.", P), Spacer(1, 8)]
    s += [Paragraph("1. Dataset", H2),
          Paragraph(f"OULAD contains {summ['n_registrations']:,} student-module registrations ({summ['n_students']:,} unique students) across "
                    f"{summ['n_modules']} modules and {summ['n_presentations']} presentations, {summ['n_assessments']} assessments with "
                    f"{summ['n_assessment_submissions']:,} submissions, and {summ['n_vle_records']:,} daily VLE click records over "
                    f"{summ['n_materials']:,} materials. Final results: " +
                    ", ".join(f"{k} {v:,}" for k, v in summ["outcomes"].items()) + ".", P)]
    rows = [["Course", "Days", "Students", "Pass", "Dist.", "Fail", "Withdr.", "Assess."]] + [
        [c["course"], c["length"], c["n_students"], c["Pass"], c["Distinction"], c["Fail"], c["Withdrawn"], c["n_assessments"]]
        for c in summ["courses"]]
    s += [Spacer(1, 4), _table(rows), Spacer(1, 6)]
    s += [Paragraph("2. Feature engineering", H2),
          Paragraph(f"After removing {st['excluded_withdrawn_early']:,} registrations that withdrew before the cut-off, "
                    f"{st['n_samples']:,} students remain ({st['at_risk']:,} at risk, {st['at_risk_rate'] * 100:.1f}%). Each becomes a vector of "
                    f"{st['n_features']} features from three sources: engagement ({finfo['source_counts'].get('engagement', 0)}), "
                    f"interaction ({finfo['source_counts'].get('interaction', 0)}) and assessment ({finfo['source_counts'].get('assessment', 0)}).", P)]
    frows = [["Feature", "Source", "Description", "Mean risk / ok"]] + [
        [Paragraph(f["feature"], P), f["source"], Paragraph(f["description"], P), f"{f['at_risk_mean']} / {f['not_at_risk_mean']}"]
        for f in finfo["features"]]
    s += [Spacer(1, 4), _table(frows, [4 * cm, 2 * cm, 8.5 * cm, 2.5 * cm]), Spacer(1, 6),
          Paragraph("Choices and tradeoffs", ss["Heading3"])]
    for t in finfo["tradeoffs"]:
        s.append(Paragraph(f"<b>{t['title']}.</b> {t['choice']} <i>Tradeoff:</i> {t['tradeoff']}", P))
    s += [PageBreak(), Paragraph("3. Methodology", H2),
          Paragraph("Each model runs inside an imbalanced-learn pipeline: StandardScaler -> SMOTE -> classifier, refitted on every "
                    "training fold, so synthetic minority samples never reach a test fold. 10-fold stratified CV is repeated over "
                    f"{res['n_repeats']} independent shuffles (seeded). Metrics are computed on the at-risk class at a 0.5 probability "
                    "threshold; ROC-AUC uses predicted probabilities. Std is reported across all folds and across trial means. "
                    "A separate hold-out comparison uses 60/40, 70/30 and 80/20 stratified splits with 5 seeds each.", P)]
    hrows = [["Model", "Settings"]] + [[m["name"], Paragraph(m["hyperparams"], P)] for m in res["models"]]
    s += [Spacer(1, 4), _table(hrows, [4 * cm, 13 * cm])]
    s += [Paragraph("4. Results", H2)]
    mrows = [["Model"] + list(METRIC_LABELS.values())] + [
        [m["name"]] + [f"{m['metrics'][k]['mean']:.3f} ± {m['metrics'][k]['std']:.3f}" for k in METRIC_LABELS]
        for m in res["models"]]
    s += [_table(mrows), Paragraph("Mean ± std across all folds.", P), Spacer(1, 4), _bar_chart(res["models"]),
          _roc_chart(res["models"])]
    crows = [["Model", "TN", "FP", "FN", "TP"]] + [
        [m["name"], m["confusion"][0][0], m["confusion"][0][1], m["confusion"][1][0], m["confusion"][1][1]] for m in res["models"]]
    s += [Paragraph("Confusion matrices (averaged per full pass over the data)", ss["Heading3"]), _table(crows)]
    for m in res["models"]:
        if m["importance"] and m["name"] in ("XGBoost", "Random Forest", "Logistic Regression"):
            s.append(_imp_chart(m))
    if run.get("holdout"):
        ho = [["Split", "Model"] + list(METRIC_LABELS.values())] + [
            [r["split"], r["model"]] + [f"{r[k]['mean']:.3f} ± {r[k]['std']:.3f}" for k in METRIC_LABELS] for r in run["holdout"]]
        s += [Paragraph("Hold-out split comparison", ss["Heading3"]), _table(ho)]
    s += [PageBreak(), Paragraph("5. Interpretation", H2)]
    for p in content.interpretation(run):
        s.append(Paragraph(f"<b>{p['heading']}.</b> {p['text']}", P))
        s.append(Spacer(1, 3))
    s += [Paragraph("6. Limitations", H2),
          Paragraph("Click counts are a noisy proxy for learning; forum data has no content; the fixed 0.5 threshold is not tuned; "
                    "hyper-parameters are fixed; pooled modelling may hide course-specific effects; excluding early withdrawals "
                    "means the model addresses students still enrolled at mid-course.", P)]
    buf = io.BytesIO()
    SimpleDocTemplate(buf, pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm, topMargin=1.6 * cm,
                      bottomMargin=1.6 * cm, title="OULAD At-Risk Prediction Report").build(s)
    return buf.getvalue()
