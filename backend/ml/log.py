import logging
from datetime import datetime, timezone

logger = logging.getLogger("oulad")
_DB = {"db": None}


def init(db):
    _DB["db"] = db


def event(source, message, level="info", **data):
    getattr(logger, level if level in ("info", "warning", "error") else "info")(f"[{source}] {message}")
    db = _DB["db"]
    if db is not None:
        db.events.insert_one({"ts": datetime.now(timezone.utc).isoformat(), "level": level,
                              "source": source, "message": message, "data": data or None})


def recent(limit=300, since=None):
    q = {"ts": {"$gt": since}} if since else {}
    docs = list(_DB["db"].events.find(q, {"_id": 0}).sort("ts", -1).limit(limit))
    return docs[::-1]
