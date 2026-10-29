import json
import os
import threading
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from . import log

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW = DATA_DIR / "raw"
PREP = DATA_DIR / "prepared"
OUTCOMES = ["Pass", "Distinction", "Fail", "Withdrawn"]
STATE = {"status": "idle", "message": "", "progress": 0.0}
_CACHE = {}
_LOCK = threading.Lock()


def _step(**kw):
    STATE.update(**kw)
    log.event("data", kw["message"], progress=round(STATE["progress"], 3))


def is_ready():
    return (PREP / "summary.json").exists()


def status():
    if STATE["status"] in ("idle", "ready") and is_ready():
        return {"status": "ready", "message": "Dataset prepared", "progress": 1.0, "ready": True}
    return {**STATE, "ready": False}


def _download():
    zp = DATA_DIR / "oulad.zip"
    if zp.exists():
        log.event("data", f"Using cached archive {zp.name} ({zp.stat().st_size / 1e6:.1f} MB)")
    else:
        _step(status="downloading", message="Downloading OULAD archive from " + os.environ["OULAD_URL"])
        r = requests.get(os.environ["OULAD_URL"], stream=True, timeout=180)
        r.raise_for_status()
        total, got = int(r.headers.get("content-length", 0)), 0
        tmp = zp.with_suffix(".part")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                got += len(chunk)
                if total:
                    STATE["progress"] = 0.35 * got / total
        tmp.rename(zp)
    _step(status="preparing", message="Extracting archive", progress=0.35)
    with zipfile.ZipFile(zp) as z:
        z.extractall(RAW)


def _read(name, **kw):
    return pd.read_csv(RAW / name, na_values="?", **kw)


def _with_course(df, cid_map):
    df["course"] = df["code_module"].astype(str) + "_" + df["code_presentation"].astype(str)
    df["cid"] = df["course"].map(cid_map).astype("int8")
    return df


def _prepare():
    _step(message="Reading course, student and assessment tables", progress=0.4)
    courses = _read("courses.csv")
    courses["course"] = courses["code_module"] + "_" + courses["code_presentation"]
    courses = courses.sort_values("course").reset_index(drop=True)
    cid_map = {c: i for i, c in enumerate(courses["course"])}
    courses["cid"] = courses["course"].map(cid_map).astype("int8")
    info = _with_course(_read("studentInfo.csv"), cid_map)
    reg = _with_course(_read("studentRegistration.csv"), cid_map)
    asm = _with_course(_read("assessments.csv"), cid_map)
    sa = _read("studentAssessment.csv")
    vle_meta = _read("vle.csv")
    vle_meta = _with_course(vle_meta, cid_map)

    _step(message="Reading 10M VLE click records (studentVle.csv)", progress=0.5)
    sv = _read(
        "studentVle.csv",
        dtype={"code_module": "category", "code_presentation": "category", "id_student": "int32",
               "id_site": "int32", "date": "int16", "sum_click": "int32"},
    )
    _step(message="Joining clicks with activity types", progress=0.7)
    mods, pres = sv["code_module"].cat, sv["code_presentation"].cat
    lut = {(i, j): cid_map.get(f"{m}_{p}", -1) for i, m in enumerate(mods.categories) for j, p in enumerate(pres.categories)}
    codes = mods.codes.astype("int32") * 100 + pres.codes.astype("int32")
    sv["cid"] = codes.map({i * 100 + j: v for (i, j), v in lut.items()}).astype("int8")
    site_type = vle_meta.set_index("id_site")["activity_type"]
    sv["activity_type"] = sv["id_site"].map(site_type).astype("category")
    sv = sv[["cid", "id_student", "id_site", "date", "sum_click", "activity_type"]]

    PREP.mkdir(parents=True, exist_ok=True)
    tables = dict(courses=courses, info=info, reg=reg, assessments=asm, sa=sa, vle_meta=vle_meta, vle=sv)
    _step(message="Saving prepared tables", progress=0.8)
    for k, v in tables.items():
        v.to_pickle(PREP / f"{k}.pkl")
    _step(message="Computing exploration statistics", progress=0.88)
    explore = _explore(tables)
    (PREP / "explore.json").write_text(json.dumps(explore))
    (PREP / "summary.json").write_text(json.dumps(_summary(tables)))
    with _LOCK:
        _CACHE.update(tables)


def load_async():
    if STATE["status"] in ("downloading", "preparing") or is_ready():
        return status()
    STATE.update(status="downloading", message="Starting", progress=0.0)

    def job():
        try:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            _download()
            _prepare()
            _step(status="ready", message="Dataset prepared: %d registrations, %d VLE records" % (len(_CACHE["info"]), len(_CACHE["vle"])), progress=1.0)
        except Exception as e:  # surfaced to the UI
            STATE.update(status="error", message=str(e), progress=0.0)
            log.event("data", f"Dataset preparation failed: {e}", level="error")

    threading.Thread(target=job, daemon=True).start()
    return status()


def get_tables():
    with _LOCK:
        if "vle" not in _CACHE:
            log.event("data", "Loading prepared tables from disk into memory")
            for k in ["courses", "info", "reg", "assessments", "sa", "vle_meta", "vle"]:
                _CACHE[k] = pd.read_pickle(PREP / f"{k}.pkl")
        return _CACHE


def _summary(t):
    info, courses, sv = t["info"], t["courses"], t["vle"]
    oc = info.groupby(["course", "final_result"]).size().unstack(fill_value=0)
    asm_n = t["assessments"].groupby("course").size()
    clicks = sv.groupby("cid")["sum_click"].sum()
    rows = []
    for _, c in courses.iterrows():
        r = {"course": c.course, "code_module": c.code_module, "code_presentation": c.code_presentation,
             "length": int(c.module_presentation_length), "n_students": int(oc.loc[c.course].sum()),
             "n_assessments": int(asm_n.get(c.course, 0)), "total_clicks": int(clicks.get(c.cid, 0))}
        for o in OUTCOMES:
            r[o] = int(oc.loc[c.course].get(o, 0))
        rows.append(r)
    act = sv.groupby("activity_type", observed=True)["sum_click"].sum().sort_values(ascending=False)
    outcomes = info["final_result"].value_counts()
    return {
        "n_registrations": int(len(info)), "n_students": int(info["id_student"].nunique()),
        "n_modules": int(courses["code_module"].nunique()), "n_presentations": int(len(courses)),
        "n_assessments": int(len(t["assessments"])), "n_assessment_submissions": int(len(t["sa"])),
        "n_vle_records": int(len(sv)), "n_materials": int(len(t["vle_meta"])),
        "total_clicks": int(sv["sum_click"].sum()),
        "outcomes": {o: int(outcomes.get(o, 0)) for o in OUTCOMES},
        "courses": rows,
        "activity_types": [{"type": k, "clicks": int(v)} for k, v in act.items()],
    }


def _explore(t):
    info, sv, courses = t["info"], t["vle"], t["courses"]
    res_map = info.set_index(["cid", "id_student"])["final_result"]
    df = sv.assign(week=(sv["date"] // 7).astype("int16"), forum=(sv["activity_type"] == "forumng"))
    g = df.groupby(["cid", "id_student", "week", "forum"], observed=True)["sum_click"].sum().reset_index()
    g["final_result"] = pd.MultiIndex.from_arrays([g["cid"], g["id_student"]]).map(res_map)
    tot = g.groupby(["cid", "week", "forum", "final_result"])["sum_click"].sum()
    n = info.groupby(["cid", "final_result"]).size()
    per = (tot / n.reindex(pd.MultiIndex.from_arrays([tot.index.get_level_values(0), tot.index.get_level_values(3)])).values)
    act = sv.groupby(["cid", "activity_type"], observed=True)["sum_click"].sum()
    asm, sa = t["assessments"], t["sa"]
    at_risk = info.set_index(["cid", "id_student"])["final_result"].isin(["Fail", "Withdrawn"])
    sa2 = sa.merge(asm[["id_assessment", "cid"]], on="id_assessment")
    sa2["at_risk"] = pd.MultiIndex.from_arrays([sa2["cid"], sa2["id_student"]]).map(at_risk)
    out = {}
    for _, c in courses.iterrows():
        cid = c.cid
        series = {}
        for forum, name in [(False, "weekly_material"), (True, "weekly_forum")]:
            try:
                s = per.xs((cid, forum), level=["cid", "forum"]).unstack(fill_value=0)
            except KeyError:
                series[name] = []
                continue
            s = s.reindex(columns=OUTCOMES, fill_value=0).sort_index()
            series[name] = [{"week": int(w), **{o: round(float(r[o]), 2) for o in OUTCOMES}} for w, r in s.iterrows()]
        a_rows = []
        for _, a in asm[asm["cid"] == cid].sort_values("date").iterrows():
            subs = sa2[sa2["id_assessment"] == a.id_assessment]
            a_rows.append({
                "id_assessment": int(a.id_assessment), "type": a.assessment_type,
                "date": None if pd.isna(a.date) else int(a.date), "weight": float(a.weight),
                "n_submitted": int(len(subs)),
                "mean_score": None if subs.empty else round(float(subs["score"].mean()), 1),
                "mean_score_at_risk": _m(subs[subs["at_risk"] == True]["score"]),
                "mean_score_not_at_risk": _m(subs[subs["at_risk"] == False]["score"]),
            })
        ac = act.xs(cid, level="cid").sort_values(ascending=False) if cid in act.index.get_level_values(0) else pd.Series(dtype=float)
        out[c.course] = {
            "course": c.course, "length": int(c.module_presentation_length),
            "cutoff": int(round(c.module_presentation_length * 0.5)),
            **series, "assessments": a_rows,
            "activity": [{"type": k, "clicks": int(v)} for k, v in ac.items()],
        }
    return out


def _m(s):
    s = s.dropna()
    return None if s.empty else round(float(s.mean()), 1)


def summary():
    return json.loads((PREP / "summary.json").read_text())


def explore(course):
    return json.loads((PREP / "explore.json").read_text()).get(course)
