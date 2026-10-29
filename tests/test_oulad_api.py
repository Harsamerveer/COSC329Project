# OULAD at-risk predictor API regression tests (read-only; never starts runs)
import os
import pytest
import requests

BASE_URL = os.environ.get("REACT_APP_BACKEND_URL").rstrip("/")
API = f"{BASE_URL}/api"


@pytest.fixture(scope="module")
def s():
    return requests.Session()


# Health / dataset
def test_root(s):
    r = s.get(f"{API}/", timeout=30)
    assert r.status_code == 200 and "message" in r.json()


def test_dataset_status(s):
    r = s.get(f"{API}/dataset/status", timeout=30)
    assert r.status_code == 200
    assert r.json()["ready"] is True


def test_dataset_summary(s):
    r = s.get(f"{API}/dataset/summary", timeout=60)
    assert r.status_code == 200
    d = r.json()
    print("summary keys:", list(d.keys()))
    txt = str(d)
    assert "32593" in txt or "32,593" in txt


# Explore
def test_explore_valid(s):
    r = s.get(f"{API}/explore/AAA_2013J", timeout=60)
    assert r.status_code == 200
    d = r.json()
    print("explore keys:", list(d.keys()))
    assert isinstance(d, dict) and len(d) > 0


def test_explore_other_course(s):
    r = s.get(f"{API}/explore/BBB_2013B", timeout=60)
    assert r.status_code == 200


def test_explore_bad(s):
    r = s.get(f"{API}/explore/BAD", timeout=30)
    assert r.status_code == 404
    assert r.json()["detail"] == "Unknown course"


# Features
def test_features_info(s):
    r = s.get(f"{API}/features/info", timeout=120)
    assert r.status_code == 200
    d = r.json()
    assert d["stats"]["n_samples"] == 24612
    assert d["stats"]["n_features"] == 29
    assert len(d["features"]) == 29
    assert len(d["tradeoffs"]) == 11
    assert sum(d["source_counts"].values()) == 29


def test_features_sample(s):
    r = s.get(f"{API}/features/sample", params={"n": 5}, timeout=60)
    assert r.status_code == 200
    d = r.json()
    assert len(d["rows"]) == 5
    assert "at_risk" in d["columns"] and len(d["columns"]) == len(d["rows"][0])


def test_features_sample_clamped(s):
    r = s.get(f"{API}/features/sample", params={"n": 5000}, timeout=60)
    assert r.status_code == 200 and len(r.json()["rows"]) == 200


def test_features_csv(s):
    r = s.get(f"{API}/features/csv", timeout=120)
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/csv")
    lines = r.text.strip().split("\n")
    assert len(lines) == 24613
    assert "at_risk" in lines[0] and "eng_clicks_rel" in lines[0]


# Experiments
def test_experiments_latest(s):
    r = s.get(f"{API}/experiments/latest", timeout=30)
    assert r.status_code == 200
    d = r.json()
    run = d["run"]
    assert run is not None and "_id" not in run
    assert run["status"] in ("queued", "running", "done")
    print("status:", run["status"], run["stage"], run["progress"])
    res = run.get("full") or run.get("preview")
    assert res is not None
    assert len(res["models"]) == 5
    for m in res["models"]:
        for k in ("precision", "recall", "f1", "roc_auc"):
            assert 0 <= m["metrics"][k]["mean"] <= 1
        assert len(m["roc"]["fpr"]) == 101
    assert isinstance(run["interpretation"], list) and len(run["interpretation"]) > 0
    assert len(run["holdout"]) == 15
    assert "algorithms" in d


# Logs
def test_logs_limit(s):
    r = s.get(f"{API}/logs", params={"limit": 5}, timeout=30)
    assert r.status_code == 200
    ev = r.json()["events"]
    assert len(ev) == 5
    for e in ev:
        assert {"ts", "level", "source", "message"} <= set(e)
        assert "_id" not in e
    assert [e["ts"] for e in ev] == sorted(e["ts"] for e in ev)


def test_logs_since(s):
    ev = s.get(f"{API}/logs", params={"limit": 5}, timeout=30).json()["events"]
    r = s.get(f"{API}/logs", params={"since": ev[-1]["ts"]}, timeout=30)
    assert r.status_code == 200
    assert all(e["ts"] > ev[-1]["ts"] for e in r.json()["events"])


# Report
def test_report_pdf(s):
    r = s.get(f"{API}/report/pdf", timeout=120)
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/pdf"
    assert r.content[:5] == b"%PDF-"
    assert len(r.content) > 10000
