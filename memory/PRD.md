# PRD — OULAD At-Risk Student Predictor (COSC329)

## Original problem statement
Predict "at-risk" students after an early assessment using OULAD. Explore the data, engineer feature vectors (engagement, forum interaction, assessment) and document the tradeoffs. Classify with RF, XGBoost, LR, SVM and NB, using SMOTE on the training data only. Report precision, recall, F1 and ROC-AUC. 1-person option: midterm cut-off, 10-fold CV × 25 shuffles, different splits, mean/std, interpretation. 2-person option: 5 cut-offs with a chart over time, multi-class grade drop, data-source ablation.
Later request: deploy the code; the professor wants to track backend activity step by step. GitHub repo: https://github.com/Harsamerveer/COSC329Project.git

## Architecture
- Backend (FastAPI, sync pymongo). Modules in `backend/ml/`:
  - `data.py`: download, prepare and cache tables
  - `features.py`: 29 features at the cut-off
  - `experiments.py`: joblib-parallel repeated CV and hold-out runs, with SMOTE inside an imblearn Pipeline
  - `content.py`: tradeoffs and interpretation text
  - `report.py`: reportlab/matplotlib PDF
  - `log.py`: event log stored in the Mongo `events` collection
- Run state is stored in the Mongo `runs` collection. Runs execute in a background thread in the server process; a backend reload interrupts them.
- Frontend: React with react-query and recharts. Pages: Overview, Explore, Features, Experiments, Results, Backend log and Report.

## Implemented (2026-10-29), Phase 1
- Dataset load (UCI mirror) and summary/explore stats
- Feature engineering with 11 documented tradeoffs, plus a CSV export
- 3×10 preview, then 60/40, 70/30 and 80/20 hold-out splits × 5 seeds, then the full 25×10 CV
- Metrics as mean ± std (fold-level and trial-level); ROC curves, confusion matrices, importances and interpretation
- PDF report
- Live backend log page

## Backlog
- P1 (Phase 2): 5 cut-offs (20–100%) with a metric-over-time chart; multi-class grade-drop task; data-source ablation (7 combinations)
- P2 (Phase 3): per-course breakdown, hyper-parameter tuning, demographics as a 4th source, threshold tuning
