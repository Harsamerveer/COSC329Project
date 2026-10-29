# OULAD At-Risk Student Predictor

A web app that downloads the Open University Learning Analytics Dataset, turns each student's early-course activity into feature vectors, and trains five classifiers to flag students at risk of failing or dropping out.
It runs repeated 10-fold cross-validation with SMOTE applied to training folds only, and presents results in an interactive dashboard with a downloadable written report.

## Who it's for
- A student (or pair) completing the OULAD at-risk prediction assignment who needs reproducible experiments, results tables, charts and written interpretation.
- Instructors or reviewers who want to inspect the feature choices, tradeoffs and results in one place.

## Core features and experience
**Data exploration**
- Automatic download of the 7 OULAD tables from the official Open University source.
- Overview of courses (modules and presentations, course length), student counts, final-result breakdown (Pass / Distinction / Fail / Withdrawn) and the class imbalance.
- Per-course view of clicks over time, forum activity and assessment schedule.

**Feature engineering (documented in-app)**
- Three data sources, all computed only from data available before the cut-off day:
  - Engagement: course-material clicks (all VLE activity types except forums) — total clicks, active days, clicks per week, trend over recent weeks, days since last activity, number of distinct materials used.
  - Interaction: discussion-forum clicks — total, active days, share of total activity, recency.
  - Assessment: scores and submissions due before the cut-off — mean and weighted score, number submitted vs. due, missed assessments, late-submission days, first-assessment score.
- Clicks normalised per course so long and short courses are comparable.
- A "Feature choices & tradeoffs" page explaining each decision (leakage, missing values, normalisation, excluding demographics, handling students who withdrew before the cut-off).

**Experiments**
- Binary target: at-risk = Fail or Withdrawn; not at-risk = Pass or Distinction.
- Midterm cut-off: 50% of each course's length.
- Models: Random Forest, XGBoost, Logistic Regression, Support Vector Machine, Naive Bayes.
- SMOTE applied inside each training fold only; test folds keep the real class balance.
- 10-fold stratified cross-validation repeated over 25 independent shuffles, plus a separate hold-out split comparison (e.g. 70/30, 80/20).
- Metrics: precision, recall, F1, ROC-AUC — mean and standard deviation per model.
- Quick preview (3 shuffles) shows results within minutes; the full 25-shuffle run continues in the background with a progress indicator and replaces the preview when done.

**Results dashboard + report**
- Comparison table (mean ± std for all 4 metrics × 5 models), bar charts with error bars, ROC curves, confusion matrices, feature importance (Random Forest / XGBoost) and Logistic Regression coefficients.
- Written interpretation of why each algorithm performs as it does on this kind of data (e.g. tree models handling skewed click counts, Naive Bayes' independence assumption breaking on correlated click features, SVM sensitivity to scaling).
- Downloadable report (PDF) with methodology, feature tradeoffs, tables, charts and interpretation.

## User flow
1. Open the app → press "Load dataset"; download and summary statistics appear.
2. Browse the Explore page (courses, students, outcomes, activity over time).
3. Read the Features page describing feature vectors and tradeoffs; preview a sample of the feature table.
4. Press "Run experiments" → quick-preview results appear; full run progresses in the background.
5. Review the Results dashboard; download the report.

## UI/UX feel
A clean, research-notebook style dashboard: left navigation (Overview, Explore, Features, Experiments, Results, Report), data-dense tables, clear charts, readable prose explanations next to each result. Calm and academic rather than flashy.

## Implementation phases
**Phase 1 — MVP (built now): 1-person option, end to end**
- Dataset download and exploration, feature engineering with documented tradeoffs, midterm cut-off, binary at-risk classification with all 5 models, SMOTE in training only, 25 × 10-fold CV and split comparison, mean ± std for 4 metrics, quick preview then full run, dashboard and downloadable report with interpretation.

**Phase 2 — 2-person option additions**
- 5 evenly spaced cut-offs (20/40/60/80/100% of course length) with a line chart of metrics over time for a chosen algorithm.
- Grade-drop multi-class task: classes "major drop" (first assessment ≥ 80% and later average falls by ≥ 10 points), "fail/withdraw", and "stays the same"; macro-averaged precision, recall, F1 and one-vs-rest ROC-AUC.
- Data-source ablation: engagement only, interaction only, assessment only, each pair, and all three; chart and discussion of which sources matter most.
- Report extended with these sections.

**Phase 3 — Refinement**
- Per-course breakdown of results, hyper-parameter tuning, optional demographics as a fourth source, and probability-threshold tuning to trade precision for recall.

## Assumptions
- Both scope options were selected; the full 2-person scope is the goal, with the 1-person option delivered first (Phase 1) and the 2-person additions in Phase 2.
- At-risk = Fail or Withdrawn; Distinction counts as not at-risk.
- Midterm cut-off = 50% of each course presentation's length in days.
- Students who withdrew before the cut-off are excluded from that cut-off's dataset (their outcome is already known); this choice is documented as a tradeoff.
- "Interaction" = forum (forumng) clicks; "engagement" = all other VLE clicks. OULAD has no post content, only click counts.
- Demographic and registration data are not used as features in Phase 1, to stay with the three sources named in the assignment.
- Exam scores and final results are never used as features (to avoid leakage).
- All course presentations are pooled into one model, with course-normalised features.
- Quick preview = 3 shuffles × 10 folds; full run = 25 shuffles × 10 folds.
- To keep the full run feasible, the SVM uses a linear kernel with calibrated probabilities for ROC-AUC; this is noted as a tradeoff.
- Default model settings with light, fixed hyper-parameters; no tuning in Phase 1.
- Random seeds fixed so results are reproducible.
- No login; single-user app.
