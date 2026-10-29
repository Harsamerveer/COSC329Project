import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.responses import Response
from pymongo import MongoClient
from starlette.middleware.cors import CORSMiddleware

load_dotenv(Path(__file__).parent / ".env")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

from ml import content, data, experiments, features, log, report  # noqa: E402

client = MongoClient(os.environ["MONGO_URL"])
db = client[os.environ["DB_NAME"]]
log.init(db)
db.runs.update_many({"status": {"$in": ["queued", "running"]}},
                    {"$set": {"status": "interrupted", "stage": "Interrupted by server restart"}})

app = FastAPI(title="OULAD At-Risk Predictor")
api = APIRouter(prefix="/api")


def _require_data():
    if not data.is_ready():
        raise HTTPException(409, "Dataset not loaded yet")


@api.get("/")
def root():
    return {"message": "OULAD At-Risk Predictor API"}


@api.get("/dataset/status")
def dataset_status():
    return data.status()


@api.post("/dataset/load")
def dataset_load():
    return data.load_async()


@api.get("/dataset/summary")
def dataset_summary():
    _require_data()
    return data.summary()


@api.get("/explore/{course}")
def explore(course: str):
    _require_data()
    res = data.explore(course)
    if res is None:
        raise HTTPException(404, "Unknown course")
    return res


@api.get("/features/info")
def features_info():
    _require_data()
    return features.info()


@api.get("/features/sample")
def features_sample(n: int = 25):
    _require_data()
    return features.sample(min(max(n, 1), 200))


@api.get("/features/csv")
def features_csv():
    _require_data()
    b = features.build()
    df = b["meta"].reset_index(drop=True).join(b["X"])
    df["at_risk"] = b["y"]
    return Response(df.to_csv(index=False), media_type="text/csv",
                    headers={"Content-Disposition": "attachment; filename=oulad_features_cutoff50.csv"})


def _public(run):
    if not run:
        return None
    run["id"] = run.pop("_id")
    run["interpretation"] = content.interpretation(run)
    return run


@api.post("/experiments/run")
def experiments_run():
    _require_data()
    rid = experiments.start(db)
    return _public(db.runs.find_one({"_id": rid}))


@api.get("/experiments/latest")
def experiments_latest():
    run = db.runs.find_one(sort=[("created_at", -1)])
    last = None
    if run and not (run.get("full") or run.get("preview")):
        last = db.runs.find_one({"$or": [{"full": {"$ne": None}}, {"preview": {"$ne": None}}]}, sort=[("created_at", -1)])
    return {"run": _public(run), "last_results_run": _public(last), "algorithms": content.ALGORITHMS}


@api.get("/logs")
def logs(since: str | None = None, limit: int = 300):
    return {"events": log.recent(min(limit, 1000), since)}


@api.get("/report/pdf")
def report_pdf():
    _require_data()
    run = db.runs.find_one({"$or": [{"full": {"$ne": None}}, {"preview": {"$ne": None}}]}, sort=[("created_at", -1)])
    if not run:
        raise HTTPException(409, "Run the experiments first")
    log.event("report", "Generating PDF report")
    return Response(report.build_pdf(run), media_type="application/pdf",
                    headers={"Content-Disposition": "attachment; filename=oulad_at_risk_report.pdf"})


app.include_router(api)
app.add_middleware(CORSMiddleware, allow_credentials=True, allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
                   allow_methods=["*"], allow_headers=["*"])


@app.on_event("shutdown")
def shutdown():
    client.close()
