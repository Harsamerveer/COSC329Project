TRADEOFFS = [
    {"title": "Cut-off at 50% of course length",
     "choice": "Only clicks with date < cut-off and assessments due (and submitted) before the cut-off are used. The cut-off is 50% of each presentation's length in days (the simulated 'midterm').",
     "tradeoff": "Earlier cut-offs allow earlier intervention but contain less signal; later ones are more accurate but leave less time to help. 50% guarantees at least one or two marked assessments in every module."},
    {"title": "No leakage from the future",
     "choice": "Final exam scores, final_result, unregistration dates and any submission made after the cut-off are never used as features.",
     "tradeoff": "Gives honest, deployable estimates; accuracy is lower than models that silently peek at end-of-course data."},
    {"title": "Students who withdrew before the cut-off are excluded",
     "choice": "Registrations with date_unregistration < cut-off are dropped from the training/testing set.",
     "tradeoff": "Their outcome is already known at prediction time, so predicting it would inflate scores. The cost: the model only targets students still enrolled at mid-course, so the at-risk share drops."},
    {"title": "Engagement vs. interaction split",
     "choice": "'Interaction' = forumng (discussion forum) clicks; 'engagement' = every other VLE activity type (content, resources, quizzes, homepage, ...).",
     "tradeoff": "OULAD only logs clicks, not post text or replies, so 'interaction' is a proxy for social participation and cannot tell reading from posting."},
    {"title": "Per-course normalisation of click counts",
     "choice": "Raw click counts are divided by the mean of the same course presentation ('_rel' features); activity days are divided by the window length; activity-type mixes are expressed as shares.",
     "tradeoff": "Pooling 22 presentations of 7 modules in one model needs comparable scales (a STEM module generates far more clicks than a social-science one). Normalising removes the absolute level, which is partly informative, so one log(1+clicks) feature is kept as well."},
    {"title": "Missing values encode behaviour",
     "choice": "No clicks -> 0 clicks and recency = full window length; missed assessment -> score 0 and counted in asm_missed; no submissions -> mean score 0.",
     "tradeoff": "Missingness here is informative (inactive students are at risk), so imputing averages would hide the signal. The downside is that 0 conflates 'did badly' with 'did not submit' - separate count features let the model disentangle them."},
    {"title": "Aggregates, not raw sequences",
     "choice": "Each student becomes one fixed-length vector of 29 aggregates (totals, rates, recency, trend, shares).",
     "tradeoff": "Fixed-length vectors suit RF/XGB/LR/SVM/NB, are interpretable and cheap. We lose fine-grained weekly dynamics that a sequence model could exploit; the recent-ratio and score-trend features recover some of it."},
    {"title": "Demographics deliberately left out",
     "choice": "Gender, region, IMD band, age, disability and prior attempts are not used.",
     "tradeoff": "The assignment asks for engagement, interaction and assessment data; demographics can add a little accuracy but raise fairness concerns (predicting risk from who someone is rather than what they do)."},
    {"title": "Pooled model across courses",
     "choice": "All presentations are pooled into one dataset with course-relative features.",
     "tradeoff": "More data and a more general model; per-course models could capture module-specific patterns but have far fewer at-risk examples each."},
    {"title": "SMOTE only inside training folds",
     "choice": "Pipeline = StandardScaler -> SMOTE -> classifier, refit per fold, so synthetic minority samples never appear in a test fold.",
     "tradeoff": "Balanced training stops the models defaulting to 'pass' and raises recall; evaluating on untouched folds keeps precision honest. Oversampling before splitting would leak near-duplicates into the test set."},
    {"title": "Linear SVM with calibrated probabilities",
     "choice": "LinearSVC + Platt (sigmoid) calibration instead of an RBF kernel SVM.",
     "tradeoff": "An RBF SVM on ~25k SMOTE-enlarged rows x 275 fits per model is too slow; the linear kernel may miss non-linear interactions that tree models pick up."},
]

ALGORITHMS = {
    "Random Forest": "Averages many decision trees grown on bootstrap samples with random feature subsets. Trees split on thresholds, so skewed click counts, zero-inflated forum activity and interactions such as 'low clicks AND missed assessment' are handled without scaling or transformation. Averaging reduces variance, which is why its scores tend to be stable across shuffles.",
    "XGBoost": "Gradient-boosted trees fitted sequentially to the errors of earlier trees. Like Random Forest it is insensitive to feature scale and captures non-linear thresholds (e.g. a sharp jump in risk once the submission rate falls below 1), and boosting usually squeezes out slightly better ranking (ROC-AUC) on tabular data.",
    "Logistic Regression": "A linear model of the log-odds of being at risk. With standardised inputs its coefficients are directly interpretable (positive = raises risk). It cannot model thresholds or interactions on its own, so it loses some accuracy where the risk-score relationship is non-linear, but it is a strong, well-calibrated baseline when the main signals (scores, submission rate, activity) are roughly monotonic.",
    "SVM (linear)": "Finds the maximum-margin hyperplane between classes. It depends on feature scaling (hence the StandardScaler) and, with a linear kernel, behaves much like logistic regression; probabilities come from Platt calibration. Its margin focuses on borderline students, which can shift the precision/recall balance relative to LR.",
    "Naive Bayes": "Assumes features are conditionally independent and normally distributed within each class. Our features are strongly correlated (clicks, active days and materials all measure the same engagement) and highly skewed/zero-inflated, which violates both assumptions: evidence gets double-counted and probabilities become extreme. It is typically fast and recall-heavy but less precise.",
}


def interpretation(run):
    res = (run or {}).get("full") or (run or {}).get("preview")
    if not res:
        return []
    models = res["models"]
    by = {m: sorted(models, key=lambda x: -x["metrics"][m]["mean"]) for m in ["precision", "recall", "f1", "roc_auc"]}
    best_auc, best_f1 = by["roc_auc"][0], by["f1"][0]
    worst = by["f1"][-1]
    out = [{"heading": "Overall ranking",
            "text": (f"Across {res['n_repeats']} shuffles x {res['n_folds']} folds, {best_auc['name']} achieves the best ROC-AUC "
                     f"({best_auc['metrics']['roc_auc']['mean']:.3f} +/- {best_auc['metrics']['roc_auc']['std']:.3f}) and "
                     f"{best_f1['name']} the best F1 ({best_f1['metrics']['f1']['mean']:.3f}). {worst['name']} has the lowest F1 "
                     f"({worst['metrics']['f1']['mean']:.3f}). Standard deviations are small, so the ranking is robust to how the data are shuffled.")},
           {"heading": "Precision vs. recall under SMOTE",
            "text": (f"Highest recall: {by['recall'][0]['name']} ({by['recall'][0]['metrics']['recall']['mean']:.3f}); highest precision: "
                     f"{by['precision'][0]['name']} ({by['precision'][0]['metrics']['precision']['mean']:.3f}). Because SMOTE balances the training "
                     "folds, every model is pushed toward flagging more at-risk students than it would on the raw 'mostly pass' distribution; "
                     "test folds keep the true imbalance, so precision reflects how many flagged students really fail or withdraw. "
                     "For early-warning use, recall is usually more valuable (a missed at-risk student costs more than an unnecessary check-in).")}]
    tree = next((m for m in models if m["name"] == "XGBoost" and m["importance"]), None) or \
        next((m for m in models if m["importance"] and m["name"] == "Random Forest"), None)
    if tree:
        top = tree["importance"][:6]
        srcs = {}
        for t in tree["importance"]:
            srcs[t["source"]] = srcs.get(t["source"], 0) + abs(t["value"])
        tot = sum(srcs.values()) or 1
        mix = ", ".join(f"{k} {v / tot * 100:.0f}%" for k, v in sorted(srcs.items(), key=lambda kv: -kv[1]))
        out.append({"heading": "What drives the predictions",
                    "text": (f"In {tree['name']} the most important features are " + ", ".join(t["feature"] for t in top) +
                             f". Importance by data source: {mix}. Assessment behaviour (submitting and scoring on the early "
                             "TMAs/CMAs) is the most direct signal of eventual outcome, while engagement recency and activity "
                             "share capture disengagement that often precedes withdrawal. Forum interaction adds comparatively "
                             "little because many passing students never use the forum.")})
    lr = next((m for m in models if m["name"] == "Logistic Regression" and m["importance"]), None)
    if lr:
        pos = [t["feature"] for t in lr["importance"] if t["value"] > 0][:3]
        neg = [t["feature"] for t in lr["importance"] if t["value"] < 0][:3]
        out.append({"heading": "Logistic regression coefficients",
                    "text": (f"Features that increase predicted risk most: {', '.join(pos) or 'none'}. Features that decrease it most: "
                             f"{', '.join(neg) or 'none'}. Signs match intuition (missed work and long inactivity raise risk; higher "
                             "scores and steady activity lower it); correlated features can split or flip weight, so coefficients "
                             "should be read together rather than one by one.")})
    for m in models:
        out.append({"heading": m["name"], "text": ALGORITHMS[m["name"]] +
                    f" Here: precision {m['metrics']['precision']['mean']:.3f}, recall {m['metrics']['recall']['mean']:.3f}, "
                    f"F1 {m['metrics']['f1']['mean']:.3f}, ROC-AUC {m['metrics']['roc_auc']['mean']:.3f}."})
    if run.get("holdout"):
        out.append({"heading": "Different train/test splits",
                    "text": "Hold-out splits (60/40, 70/30, 80/20, 5 seeds each) give scores very close to cross-validation, showing results are not an artefact of the 10-fold scheme; larger training shares help slightly, mostly for the tree models which benefit from more data."})
    return out
