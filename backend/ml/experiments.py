import os
import threading
import traceback
import uuid
from datetime import datetime, timezone

import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from joblib import Parallel, delayed
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

from . import features, log

MODELS = {
    "Random Forest": lambda s: RandomForestClassifier(n_estimators=150, min_samples_leaf=2, n_jobs=1, random_state=s),
    "XGBoost": lambda s: XGBClassifier(n_estimators=250, max_depth=5, learning_rate=0.08, subsample=0.9,
                                       colsample_bytree=0.9, tree_method="hist", n_jobs=1, random_state=s,
                                       eval_metric="logloss"),
    "Logistic Regression": lambda s: LogisticRegression(max_iter=3000, C=1.0),
    "SVM (linear)": lambda s: CalibratedClassifierCV(LinearSVC(C=0.5, dual=False, max_iter=5000), cv=3),
    "Naive Bayes": lambda s: GaussianNB(),
}
HYPERPARAMS = {
    "Random Forest": "150 trees, min_samples_leaf=2, Gini impurity",
    "XGBoost": "250 boosted trees, max_depth=5, learning_rate=0.08, subsample=0.9, colsample=0.9",
    "Logistic Regression": "L2 penalty, C=1.0, lbfgs solver",
    "SVM (linear)": "LinearSVC C=0.5, sigmoid (Platt) calibration with 3-fold CalibratedClassifierCV for probabilities",
    "Naive Bayes": "Gaussian NB (per-class normal likelihood per feature)",
}
METRICS = ["precision", "recall", "f1", "roc_auc"]
GRID = np.linspace(0, 1, 101)
N_FOLDS = 10
PREVIEW_REPEATS = 3
FULL_REPEATS = 25
HOLDOUT_SPLITS = [0.4, 0.3, 0.2]
HOLDOUT_SEEDS = 5
SEED = 42
N_JOBS = max(1, min(8, (os.cpu_count() or 2) - 2))
_RUN_LOCK = threading.Lock()


def _pipe(name, seed):
    return Pipeline([("scale", StandardScaler()), ("smote", SMOTE(random_state=seed)), ("clf", MODELS[name](seed))])


def _fold(X, y, tr, te, seed, tag):
    out = {}
    for name in MODELS:
        p = _pipe(name, seed)
        p.fit(X[tr], y[tr])
        prob = p.predict_proba(X[te])[:, 1]
        pred = (prob >= 0.5).astype(int)
        fpr, tpr, _ = roc_curve(y[te], prob)
        tpr_i = np.interp(GRID, fpr, tpr)
        tpr_i[0] = 0.0
        clf = p.named_steps["clf"]
        imp = None
        if hasattr(clf, "feature_importances_"):
            imp = clf.feature_importances_
        elif hasattr(clf, "coef_"):
            imp = clf.coef_[0]
        out[name] = {
            "precision": precision_score(y[te], pred, zero_division=0), "recall": recall_score(y[te], pred),
            "f1": f1_score(y[te], pred), "roc_auc": roc_auc_score(y[te], prob),
            "tpr": tpr_i, "cm": confusion_matrix(y[te], pred, labels=[0, 1]), "imp": imp,
        }
    return tag, out


def _run_parallel(X, y, jobs, on_done, label):
    results = []
    log.event("experiments", f"{label}: dispatching {len(jobs)} fold jobs to {N_JOBS} worker processes")
    gen = Parallel(n_jobs=N_JOBS, return_as="generator_unordered")(
        delayed(_fold)(X, y, tr, te, seed, tag) for tr, te, seed, tag in jobs)
    for r in gen:
        results.append(r)
        on_done()
        tag, res = r
        n_tr = len(jobs[0][0])
        smote_note = f"train {n_tr} rows -> SMOTE-balanced, test {len(y) - n_tr} rows at real class balance"
        scores = ", ".join(f"{k}: AUC {v['roc_auc']:.3f} F1 {v['f1']:.3f}" for k, v in res.items())
        log.event("experiments", f"{label} - fold {len(results)}/{len(jobs)} done ({smote_note}) | {scores}")
    return results


def _ms(a):
    a = np.asarray(a, dtype=float)
    return {"mean": round(float(a.mean()), 4), "std": round(float(a.std(ddof=1)) if len(a) > 1 else 0.0, 4)}


def _aggregate_cv(results, n_repeats):
    models = []
    for name in MODELS:
        rows = [(tag, r[name]) for tag, r in results]
        metrics = {}
        for m in METRICS:
            vals = np.array([r[m] for _, r in rows])
            trial = [np.mean([r[m] for t, r in rows if t == k]) for k in range(n_repeats)]
            metrics[m] = {**_ms(vals), "trial_std": round(float(np.std(trial, ddof=1)) if n_repeats > 1 else 0.0, 4)}
        tprs = np.array([r["tpr"] for _, r in rows])
        cm = sum(r["cm"] for _, r in rows) / n_repeats
        imp = None
        if rows[0][1]["imp"] is not None:
            iv = np.mean([r["imp"] for _, r in rows], axis=0)
            imp = sorted([{"feature": f, "source": features.SOURCE_OF[f], "value": round(float(v), 5)}
                          for f, v in zip(features.FEATURE_NAMES, iv)], key=lambda d: -abs(d["value"]))
        models.append({
            "name": name, "hyperparams": HYPERPARAMS[name], "metrics": metrics,
            "roc": {"fpr": [round(float(x), 3) for x in GRID], "tpr": [round(float(x), 4) for x in tprs.mean(0)],
                    "tpr_std": [round(float(x), 4) for x in tprs.std(0)]},
            "confusion": [[round(float(v), 1) for v in row] for row in cm],
            "importance": imp,
        })
    return {"n_repeats": n_repeats, "n_folds": N_FOLDS, "n_fits_per_model": len(results), "models": models}


def _cv_jobs(y, n_repeats):
    rskf = RepeatedStratifiedKFold(n_splits=N_FOLDS, n_repeats=n_repeats, random_state=SEED)
    return [(tr, te, SEED + i, i // N_FOLDS) for i, (tr, te) in enumerate(rskf.split(np.zeros(len(y)), y))]


def _holdout_jobs(y):
    jobs, idx = [], np.arange(len(y))
    for ts in HOLDOUT_SPLITS:
        for s in range(HOLDOUT_SEEDS):
            tr, te = train_test_split(idx, test_size=ts, stratify=y, random_state=SEED + s)
            jobs.append((tr, te, SEED + s, ts))
    return jobs


def _aggregate_holdout(results):
    rows = []
    for ts in HOLDOUT_SPLITS:
        sub = [r for tag, r in results if tag == ts]
        for name in MODELS:
            rows.append({"split": f"{int(round((1 - ts) * 100))}/{int(round(ts * 100))}", "model": name,
                         "n_seeds": len(sub), **{m: _ms([r[name][m] for r in sub]) for m in METRICS}})
    return rows


def _now():
    return datetime.now(timezone.utc).isoformat()


def start(db, frac=0.5, full_repeats=FULL_REPEATS):
    with _RUN_LOCK:
        active = db.runs.find_one({"status": {"$in": ["queued", "running"]}})
        if active:
            return active["_id"]
        rid = str(uuid.uuid4())
        total = PREVIEW_REPEATS * N_FOLDS + len(HOLDOUT_SPLITS) * HOLDOUT_SEEDS + full_repeats * N_FOLDS
        db.runs.insert_one({"_id": rid, "status": "queued", "stage": "Building feature vectors",
                            "progress": {"done": 0, "total": total}, "created_at": _now(),
                            "config": {"cutoff_fraction": frac, "n_folds": N_FOLDS, "preview_repeats": PREVIEW_REPEATS,
                                       "full_repeats": full_repeats, "holdout_splits": HOLDOUT_SPLITS,
                                       "holdout_seeds": HOLDOUT_SEEDS, "seed": SEED, "smote": "training folds only",
                                       "threshold": 0.5, "models": list(MODELS)},
                            "preview": None, "full": None, "holdout": None, "error": None})
    threading.Thread(target=_run, args=(db, rid, frac, full_repeats), daemon=True).start()
    return rid


def _run(db, rid, frac, full_repeats):
    def tick():
        db.runs.update_one({"_id": rid}, {"$inc": {"progress.done": 1}})

    def stage(s, **extra):
        log.event("experiments", f"Stage: {s}")
        db.runs.update_one({"_id": rid}, {"$set": {"stage": s, "status": "running", **extra}})

    try:
        stage("Building feature vectors")
        b = features.build(frac)
        X, y = b["X"].values, b["y"]
        db.runs.update_one({"_id": rid}, {"$set": {"dataset": b["stats"]}})
        stage(f"Quick preview: {PREVIEW_REPEATS} shuffles x {N_FOLDS} folds")
        prev = _run_parallel(X, y, _cv_jobs(y, PREVIEW_REPEATS), tick, "Preview CV")
        stage("Hold-out split comparison (60/40, 70/30, 80/20)", preview=_aggregate_cv(prev, PREVIEW_REPEATS))
        ho = _run_parallel(X, y, _holdout_jobs(y), tick, "Hold-out")
        stage(f"Full run: {full_repeats} shuffles x {N_FOLDS} folds", holdout=_aggregate_holdout(ho))
        full = _run_parallel(X, y, _cv_jobs(y, full_repeats), tick, "Full CV")
        db.runs.update_one({"_id": rid}, {"$set": {"full": _aggregate_cv(full, full_repeats), "status": "done",
                                                   "stage": "Complete", "finished_at": _now()}})
        log.event("experiments", "Run complete - results saved to MongoDB (collection 'runs')")
    except Exception as e:
        traceback.print_exc()
        log.event("experiments", f"Run failed: {e}", level="error")
        db.runs.update_one({"_id": rid}, {"$set": {"status": "error", "error": str(e), "stage": "Failed"}})
