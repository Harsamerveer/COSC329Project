import threading

import numpy as np
import pandas as pd

from . import data, log

KEY = ["cid", "id_student"]
RECENT_DAYS = 14
PRE_COURSE_DAYS = 30
MATERIAL_TYPES = ["oucontent", "resource", "quiz", "homepage", "subpage", "url"]

FEATURES = [
    ("eng_clicks_rel", "engagement", "Course-material clicks before the cut-off, divided by the course-presentation mean (1.0 = average student)."),
    ("eng_log_clicks", "engagement", "log(1 + raw material clicks) - compresses the heavy right tail of click counts."),
    ("eng_active_frac", "engagement", "Share of days (pre-course window + days up to cut-off) with any material activity."),
    ("eng_clicks_per_day_rel", "engagement", "Clicks per active day relative to the course mean (intensity, not regularity)."),
    ("eng_materials_rel", "engagement", "Distinct course materials (VLE sites) opened, relative to the course mean."),
    ("eng_days_since_last", "engagement", "Days between the last material activity and the cut-off (inactivity / recency)."),
    ("eng_recent_ratio", "engagement", "Click rate in the last 14 days before the cut-off divided by the overall rate (>1 rising, <1 fading)."),
    ("eng_share_oucontent", "engagement", "Share of material clicks on OU content pages (main study text)."),
    ("eng_share_resource", "engagement", "Share of material clicks on downloadable resources."),
    ("eng_share_quiz", "engagement", "Share of material clicks on quizzes (self-assessment)."),
    ("eng_share_homepage", "engagement", "Share of material clicks on the module homepage (navigation, not study)."),
    ("eng_share_subpage", "engagement", "Share of material clicks on sub-pages."),
    ("eng_share_url", "engagement", "Share of material clicks on external links."),
    ("forum_clicks_rel", "interaction", "Discussion-forum (forumng) clicks relative to the course mean."),
    ("forum_active_frac", "interaction", "Share of days with any forum activity."),
    ("forum_share", "interaction", "Forum clicks as a share of all VLE clicks (social vs. solitary study)."),
    ("forum_days_since_last", "interaction", "Days between the last forum activity and the cut-off."),
    ("forum_recent_rel", "interaction", "Forum clicks in the last 14 days before cut-off, relative to the course mean."),
    ("forum_any", "interaction", "1 if the student used the forum at all before the cut-off."),
    ("asm_n_due", "assessment", "Number of non-exam assessments (TMA/CMA) due before the cut-off."),
    ("asm_submit_rate", "assessment", "Share of due assessments submitted before the cut-off."),
    ("asm_missed", "assessment", "Count of due assessments not submitted by the cut-off."),
    ("asm_mean_score", "assessment", "Mean score (0-100) over submitted assessments; 0 if nothing was submitted."),
    ("asm_weighted_score", "assessment", "Weight-averaged score over all due assessments with missed ones counted as 0."),
    ("asm_first_score", "assessment", "Score on the first (early) assessment of the course; 0 if missed."),
    ("asm_last_score", "assessment", "Score on the most recent assessment due before the cut-off; 0 if missed."),
    ("asm_score_trend", "assessment", "Last minus first score - captures early trajectory."),
    ("asm_min_score", "assessment", "Lowest score among due assessments (missed = 0)."),
    ("asm_mean_late_days", "assessment", "Mean days submitted after the deadline (0 if on time / none)."),
]
FEATURE_NAMES = [f[0] for f in FEATURES]
SOURCE_OF = {f[0]: f[1] for f in FEATURES}
_CACHE = {}
_LOCK = threading.Lock()


def _rel(s, cid):
    m = s.groupby(cid).transform("mean")
    return (s / m.replace(0, np.nan)).fillna(0)


def _vle_agg(v, cutoff_arr):
    g = v.groupby(KEY, observed=True)
    a = pd.DataFrame({"clicks": g["sum_click"].sum(), "days": g["date"].nunique(),
                      "last": g["date"].max(), "sites": g["id_site"].nunique()})
    recent = v[v["date"].values >= cutoff_arr[v["cid"].values] - RECENT_DAYS]
    a["recent"] = recent.groupby(KEY, observed=True)["sum_click"].sum()
    return a


def _assessment(pop, t):
    asm = t["assessments"]
    asm = asm[(asm["assessment_type"] != "Exam") & asm["date"].notna()]
    due = pop[KEY + ["cutoff"]].merge(asm[["cid", "id_assessment", "date", "weight"]], on="cid")
    due = due[due["date"] < due["cutoff"]]
    due = due.merge(t["sa"][["id_assessment", "id_student", "date_submitted", "score"]],
                    on=["id_assessment", "id_student"], how="left")
    due["sub"] = (due["date_submitted"].notna() & (due["date_submitted"] < due["cutoff"])).astype(int)
    due["score_eff"] = np.where(due["sub"] == 1, due["score"].fillna(0), 0.0)
    due["score_sub"] = np.where(due["sub"] == 1, due["score"].fillna(0), np.nan)
    due["late"] = np.where(due["sub"] == 1, (due["date_submitted"] - due["date"]).clip(lower=0), np.nan)
    due["w"] = due["weight"]
    due["ws"] = due["w"] * due["score_eff"]
    due = due.sort_values(KEY + ["date"])
    g = due.groupby(KEY)
    a = pd.DataFrame({
        "asm_n_due": g.size(), "n_sub": g["sub"].sum(), "asm_mean_score": g["score_sub"].mean(),
        "wsum": g["w"].sum(), "ws": g["ws"].sum(), "eff_mean": g["score_eff"].mean(),
        "asm_first_score": g["score_eff"].first(), "asm_last_score": g["score_eff"].last(),
        "asm_min_score": g["score_eff"].min(), "asm_mean_late_days": g["late"].mean(),
    })
    a["asm_submit_rate"] = a["n_sub"] / a["asm_n_due"]
    a["asm_missed"] = a["asm_n_due"] - a["n_sub"]
    a["asm_weighted_score"] = np.where(a["wsum"] > 0, a["ws"] / a["wsum"].replace(0, np.nan), a["eff_mean"])
    a["asm_score_trend"] = a["asm_last_score"] - a["asm_first_score"]
    return a


def build(frac=0.5):
    with _LOCK:
        if frac in _CACHE:
            return _CACHE[frac]
        log.event("features", f"Building feature vectors at cut-off {frac:.0%} of course length")
        t = data.get_tables()
        courses = t["courses"]
        cutoff_arr = np.round(courses.sort_values("cid")["module_presentation_length"].values * frac).astype(int)
        info = t["info"][KEY + ["course", "final_result"]].merge(
            t["reg"][KEY + ["date_unregistration"]], on=KEY, how="left")
        info["cutoff"] = cutoff_arr[info["cid"].values]
        early = info["date_unregistration"].notna() & (info["date_unregistration"] < info["cutoff"])
        pop = info[~early].reset_index(drop=True)
        log.event("features", f"Excluded {int(early.sum())} registrations that withdrew before the cut-off; {len(pop)} students remain")

        vle = t["vle"]
        v = vle[vle["date"].values < cutoff_arr[vle["cid"].values]]
        is_forum = (v["activity_type"] == "forumng").values
        mat, frm = v[~is_forum], v[is_forum]
        log.event("features", f"{len(v):,} VLE records before cut-off: {len(mat):,} material (engagement), {len(frm):,} forum (interaction)")
        m = _vle_agg(mat, cutoff_arr)
        f = _vle_agg(frm, cutoff_arr)
        types = mat[mat["activity_type"].isin(MATERIAL_TYPES)].groupby(
            KEY + ["activity_type"], observed=True)["sum_click"].sum().unstack(fill_value=0)

        idx = pd.MultiIndex.from_frame(pop[KEY])
        m, f = m.reindex(idx), f.reindex(idx)
        types = types.reindex(idx).fillna(0)
        cid = pop["cid"].values
        cutoff = pop["cutoff"].values
        window = cutoff + PRE_COURSE_DAYS
        X = pd.DataFrame(index=idx)
        mc = m["clicks"].fillna(0).values
        fc = f["clicks"].fillna(0).values
        X["eng_clicks_rel"] = _rel(pd.Series(mc, index=idx), cid).values
        X["eng_log_clicks"] = np.log1p(mc)
        X["eng_active_frac"] = m["days"].fillna(0).values / window
        cpd = np.where(m["days"].fillna(0).values > 0, mc / m["days"].fillna(1).values, 0)
        X["eng_clicks_per_day_rel"] = _rel(pd.Series(cpd, index=idx), cid).values
        X["eng_materials_rel"] = _rel(pd.Series(m["sites"].fillna(0).values, index=idx), cid).values
        X["eng_days_since_last"] = np.where(m["last"].notna(), cutoff - m["last"].fillna(0).values, window)
        rate_all = mc / window
        rate_recent = m["recent"].fillna(0).values / RECENT_DAYS
        X["eng_recent_ratio"] = np.clip(np.where(rate_all > 0, rate_recent / np.where(rate_all > 0, rate_all, 1), 0), 0, 10)
        for tname in MATERIAL_TYPES:
            col = types[tname].values if tname in types else np.zeros(len(idx))
            X[f"eng_share_{tname}"] = np.where(mc > 0, col / np.where(mc > 0, mc, 1), 0)
        X["forum_clicks_rel"] = _rel(pd.Series(fc, index=idx), cid).values
        X["forum_active_frac"] = f["days"].fillna(0).values / window
        X["forum_share"] = np.where(mc + fc > 0, fc / np.where(mc + fc > 0, mc + fc, 1), 0)
        X["forum_days_since_last"] = np.where(f["last"].notna(), cutoff - f["last"].fillna(0).values, window)
        X["forum_recent_rel"] = _rel(pd.Series(f["recent"].fillna(0).values, index=idx), cid).values
        X["forum_any"] = (fc > 0).astype(int)
        a = _assessment(pop, t).reindex(idx)
        log.event("features", "Computed assessment features from TMA/CMA due and submitted before the cut-off")
        for c in ["asm_n_due", "asm_submit_rate", "asm_missed", "asm_mean_score", "asm_weighted_score",
                  "asm_first_score", "asm_last_score", "asm_score_trend", "asm_min_score", "asm_mean_late_days"]:
            X[c] = a[c].fillna(0).values
        X = X[FEATURE_NAMES].astype(float)
        y = pop["final_result"].isin(["Fail", "Withdrawn"]).astype(int).values
        meta = pop[["course", "id_student", "final_result", "cutoff"]]
        res = {
            "X": X.reset_index(drop=True), "y": y, "meta": meta,
            "stats": {
                "cutoff_fraction": frac, "n_samples": int(len(pop)), "n_features": len(FEATURE_NAMES),
                "excluded_withdrawn_early": int(early.sum()), "n_registrations": int(len(info)),
                "at_risk": int(y.sum()), "not_at_risk": int(len(y) - y.sum()),
                "at_risk_rate": round(float(y.mean()), 4),
                "outcomes_kept": {k: int(v) for k, v in pop["final_result"].value_counts().items()},
            },
        }
        _CACHE[frac] = res
        log.event("features", f"Feature matrix ready: {X.shape[0]} x {X.shape[1]}, at-risk rate {y.mean():.1%}")
        return res


def info(frac=0.5):
    from .content import TRADEOFFS
    b = build(frac)
    X, y = b["X"], b["y"]
    desc = []
    for name, src, d in FEATURES:
        col = X[name]
        desc.append({"feature": name, "source": src, "description": d,
                     "mean": round(float(col.mean()), 3), "std": round(float(col.std()), 3),
                     "at_risk_mean": round(float(col[y == 1].mean()), 3),
                     "not_at_risk_mean": round(float(col[y == 0].mean()), 3)})
    counts = {}
    for _, s, _ in FEATURES:
        counts[s] = counts.get(s, 0) + 1
    return {"features": desc, "tradeoffs": TRADEOFFS, "stats": b["stats"], "source_counts": counts}


def sample(n=25, seed=7):
    b = build()
    X = b["X"].round(3)
    df = pd.concat([b["meta"].reset_index(drop=True), X], axis=1)
    df["at_risk"] = b["y"]
    df = df.sample(n=min(n, len(df)), random_state=seed)
    return {"columns": list(df.columns), "rows": df.astype(object).where(df.notna(), None).values.tolist()}
