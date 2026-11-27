import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

# Flip to True for a quick live demo on video; False runs the full 25-trial x 10-fold CV
DEMO_MODE = True

DATA_PATH = "data/features_midterm.csv"
RESULTS_DIR = "results"
METRICS = ["precision", "recall", "f1", "roc_auc"]


def make_models(seed):
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=seed),
        "Linear SVM": LinearSVC(C=1.0, max_iter=3000, random_state=seed),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=15, random_state=seed, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=seed,
            n_jobs=-1,
        ),
    }


os.makedirs(RESULTS_DIR, exist_ok=True)

# 1. Load the engineered midterm dataset
df = pd.read_csv(DATA_PATH)
X = df.drop(columns=["target"])
y = df["target"]
print(f"Loaded {len(df)} students and {X.shape[1]} features (At-Risk rate: {y.mean() * 100:.2f}%)")

n_trials = 1 if DEMO_MODE else 25
n_splits = 5 if DEMO_MODE else 10
suffix = "_demo" if DEMO_MODE else ""

# 2. Run repeated stratified k-fold CV with SMOTE strictly inside training folds
print(f"\nRunning {n_trials} trial(s) x {n_splits}-fold CV per model...")
rows = []
model_names = list(make_models(42).keys())

for name in model_names:
    t0 = time.time()
    print(f"\nTraining {name}...")

    for trial in range(n_trials):
        seed = 42 + trial
        clf = make_models(seed)[name]

        # StandardScaler + SMOTE inside imblearn Pipeline ensures SMOTE never touches test folds
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=seed)),
            ("clf", clf),
        ])

        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        scores = cross_validate(pipe, X, y, cv=cv, scoring=METRICS)

        for fold in range(n_splits):
            rows.append({
                "model": name,
                "trial": trial + 1,
                "fold": fold + 1,
                "precision": scores["test_precision"][fold],
                "recall": scores["test_recall"][fold],
                "f1": scores["test_f1"][fold],
                "roc_auc": scores["test_roc_auc"][fold],
            })

        if (trial + 1) % 5 == 0 or trial == n_trials - 1:
            print(f"  Trial {trial + 1}/{n_trials} done ({time.time() - t0:.1f}s)")

fold_df = pd.DataFrame(rows)

# 3. Calculate Mean and Standard Deviation for each model
summary = []
for name, grp in fold_df.groupby("model", sort=False):
    entry = {"Model": name}
    for m in METRICS:
        mean_val = grp[m].mean()
        std_val = grp[m].std()
        entry[f"{m}_mean"] = round(mean_val, 4)
        entry[f"{m}_std"] = round(std_val, 4)
        entry[m.upper()] = f"{mean_val:.4f} +/- {std_val:.4f}"
    summary.append(entry)

summary_df = pd.DataFrame(summary).sort_values("roc_auc_mean", ascending=False)

print("\nModel Comparison (Mean +/- SD):")
print(summary_df[["Model", "PRECISION", "RECALL", "F1", "ROC_AUC"]].to_string(index=False))

fold_df.to_csv(f"{RESULTS_DIR}/all_fold_scores{suffix}.csv", index=False)
summary_df.to_csv(f"{RESULTS_DIR}/model_metrics_summary{suffix}.csv", index=False)

# 4. Plot grouped bar chart of all 4 metrics
x = np.arange(len(summary_df))
width = 0.2
fig, ax = plt.subplots(figsize=(10, 5.5))

for i, (m, label) in enumerate(zip(METRICS, ["Precision", "Recall", "F1", "ROC-AUC"])):
    ax.bar(
        x + (i - 1.5) * width,
        summary_df[f"{m}_mean"],
        width,
        yerr=summary_df[f"{m}_std"],
        capsize=3,
        label=label,
    )

ax.set_ylabel("Score")
ax.set_title("OULAD Midterm At-Risk Models (Mean +/- SD)")
ax.set_xticks(x)
ax.set_xticklabels(summary_df["Model"], rotation=12, ha="right")
ax.set_ylim(0, 1.05)
ax.grid(axis="y", linestyle="--", alpha=0.5)
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/metrics_comparison{suffix}.png", dpi=200)
plt.close()

# 5. Fit once on an 80/20 split to plot ROC curves and show sample student predictions
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

fig, ax = plt.subplots(figsize=(7.5, 5.5))
xgb_pipe = None

for name, clf in make_models(42).items():
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=42)),
        ("clf", clf),
    ])
    pipe.fit(X_train, y_train)
    if name == "XGBoost":
        xgb_pipe = pipe

    # LinearSVC uses decision_function instead of predict_proba
    if hasattr(pipe, "predict_proba"):
        risk_scores = pipe.predict_proba(X_test)[:, 1]
    else:
        risk_scores = pipe.decision_function(X_test)

    fpr, tpr, _ = roc_curve(y_test, risk_scores)
    auc = roc_auc_score(y_test, risk_scores)
    ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")

ax.plot([0, 1], [0, 1], "--", color="gray", label="Random Guess")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate (Recall)")
ax.set_title("ROC Curves at 50% Midterm Cut-Off")
ax.grid(alpha=0.4)
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/roc_curves{suffix}.png", dpi=200)
plt.close()

# 6. Print sample predictions (both correct catches and errors) for the video demo
probs = xgb_pipe.predict_proba(X_test)[:, 1]
preds = (probs >= 0.50).astype(int)

demo = X_test[["material_clicks", "forum_clicks", "submission_rate", "mean_early_score"]].copy()
demo["risk_prob"] = probs.round(3)
demo["predicted"] = np.where(preds == 1, "At-Risk", "Pass")
demo["actual"] = np.where(y_test.values == 1, "At-Risk", "Pass")

caught = demo[(demo["predicted"] == "At-Risk") & (demo["actual"] == "At-Risk")].head(2)
passed = demo[(demo["predicted"] == "Pass") & (demo["actual"] == "Pass")].head(2)
false_alarm = demo[(demo["predicted"] == "At-Risk") & (demo["actual"] == "Pass")].head(1)
missed = demo[(demo["predicted"] == "Pass") & (demo["actual"] == "At-Risk")].head(1)

print("\nSample Student Predictions on Holdout Test Set (XGBoost):")
print(pd.concat([caught, passed, false_alarm, missed]).to_string(index=False))
print(f"\nDone! Saved CSVs and charts to '{RESULTS_DIR}/'.")