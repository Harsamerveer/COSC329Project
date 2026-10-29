# COSC329 Project — OULAD At-Risk Student Predictor

A full-stack app (React + FastAPI + MongoDB) that predicts students at risk of failing or withdrawing, using the
[Open University Learning Analytics Dataset](https://analyse.kmi.open.ac.uk/open-dataset).

## Pipeline
1. **Data** (`backend/ml/data.py`): downloads the 7 OULAD tables (`OULAD_URL` in `backend/.env`), parses 10.6M VLE click
   records, and caches the prepared tables under `backend/data/prepared/` (not in git).
2. **Features** (`backend/ml/features.py`): 29 features per student, computed only from data before the 50% cut-off, in three groups:
   engagement (material clicks), interaction (forum clicks) and assessment (TMA/CMA submissions and scores). Students who
   withdrew before the cut-off are left out. The choices and their tradeoffs are written up in `backend/ml/content.py`.
3. **Experiments** (`backend/ml/experiments.py`): the pipeline is StandardScaler → SMOTE → classifier. SMOTE runs on training folds only.
   The models are Random Forest, XGBoost, Logistic Regression, linear SVM (calibrated) and Gaussian Naive Bayes.
   Evaluation uses 10-fold stratified CV repeated over 25 shuffles, plus 60/40, 70/30 and 80/20 hold-out splits.
   Reported metrics: precision, recall, F1 and ROC-AUC (mean ± std).
4. **Report** (`backend/ml/report.py`): a PDF with the methodology, results tables, charts and interpretation.

## Backend activity tracking
Each backend step is written to the server log and to the MongoDB `events` collection. You can read these events at
`GET /api/logs` or on the **Backend log** page.

## API
| Method | Path | Purpose |
|---|---|---|
| POST | /api/dataset/load | download + prepare OULAD |
| GET | /api/dataset/status, /api/dataset/summary | progress / overview stats |
| GET | /api/explore/{course} | weekly clicks, assessments per presentation |
| GET | /api/features/info, /api/features/sample, /api/features/csv | feature docs, sample, full matrix |
| POST | /api/experiments/run | start preview (3×10) → hold-out → full (25×10) run |
| GET | /api/experiments/latest | progress + results + interpretation |
| GET | /api/logs | backend event log |
| GET | /api/report/pdf | PDF report |

## Run locally
Backend: `pip install -r backend/requirements.txt && uvicorn server:app --port 8001` (run it from `backend/`, with `MONGO_URL`, `DB_NAME`, `OULAD_URL` set in `.env`).
Frontend: `cd frontend && yarn install && yarn start` (with `REACT_APP_BACKEND_URL` set).
