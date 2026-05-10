"""
api.py — Vault-Audit Phase 2
==============================
FastAPI application that wraps the middleware.
All inbound requests go through execute_query(); the DB is never
touched directly from this layer.

Run with:
    python api.py
  or (dev):
    uvicorn api:app --reload --port 8000 --loop asyncio
"""

from __future__ import annotations

import asyncio
import platform
from typing import Any, Union

# ── Windows event loop fix ────────────────────────────────────────────────────
# Python 3.8+ on Windows defaults to ProactorEventLoop, which deadlocks
# pymongo's synchronous blocking sockets inside uvicorn's async worker.
# Fix: explicitly create and set a SelectorEventLoop before anything else runs.
if platform.system() == "Windows":
    _loop = asyncio.SelectorEventLoop()
    asyncio.set_event_loop(_loop)

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

from middleware import execute_query, get_db

# ─── App ─────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Vault-Audit API",
    description="Security middleware for MongoDB — intercepts, scores, and logs every query.",
    version="0.2.0",
)


# ─── Request / Response models ────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query_type:     str                    = Field(..., pattern="^(READ|UPDATE|DELETE|INSERT)$")
    query_filter:   dict[str, Any]         = Field(default_factory=dict)
    user_id:        str                    = Field(..., min_length=1)
    update_payload: Union[dict[str, Any], None] = None


class QueryResponse(BaseModel):
    status:       str
    threat_score: float
    flagged:      bool
    record_count: int
    hash:         str
    data:         Union[list[dict[str, Any]], None] = None


# ─── Helper ──────────────────────────────────────────────────────────────────

def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.get("/health", tags=["meta"])
def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok", "service": "vault-audit"}


@app.post("/query", tags=["queries"])
def run_query(payload: QueryRequest, request: Request) -> Response:
    """
    Submit a query against `sensitive_data`.
    The middleware classifies it, logs it, and either returns results
    or blocks the request with HTTP 403.
    """
    ip = _client_ip(request)

    try:
        result = execute_query(
            query_type     = payload.query_type,
            query_filter   = payload.query_filter,
            user_id        = payload.user_id,
            ip_address     = ip,
            update_payload = payload.update_payload,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    if result["status"] == "blocked":
        return JSONResponse(
            status_code=403,
            content={
                "status":       "blocked",
                "threat_score": result["threat_score"],
                "flagged":      True,
                "record_count": result["record_count"],
                "hash":         "",
                "data":         None,
                "detail":       "Query blocked: threat score exceeds safe threshold.",
            },
        )

    return JSONResponse(content=QueryResponse(**result).model_dump())


@app.get("/audit/logs", tags=["audit"])
def get_audit_logs(
    limit: int = 20,
    flagged_only: bool = False,
) -> list[dict[str, Any]]:
    """
    Retrieve recent audit log entries.
    - `limit`        — max entries to return (default 20)
    - `flagged_only` — when true, return only flagged (suspicious) logs
    """
    db = get_db()
    filt: dict[str, Any] = {"flagged": True} if flagged_only else {}
    cursor = (
        db["audit_logs"]
        .find(filt, {"_id": 0})
        .sort("timestamp", -1)
        .limit(limit)
    )
    logs: list[dict[str, Any]] = []
    for doc in cursor:
        if "timestamp" in doc:
            doc["timestamp"] = doc["timestamp"].isoformat()
        logs.append(doc)
    return logs


@app.get("/audit/stats", tags=["audit"])
def get_audit_stats() -> dict[str, Any]:
    """High-level stats over all audit logs."""
    db = get_db()
    collection = db["audit_logs"]
    total   = collection.count_documents({})
    flagged = collection.count_documents({"flagged": True})
    blocked = collection.count_documents({"threat_score": {"$gte": 0.85}})

    pipeline: list[dict[str, Any]] = [
        {"$group": {"_id": None, "avg_threat": {"$avg": "$threat_score"}}}
    ]
    agg = list(collection.aggregate(pipeline))
    avg_threat = round(agg[0]["avg_threat"], 4) if agg else 0.0

    return {
        "total_queries":    total,
        "flagged":          flagged,
        "blocked":          blocked,
        "avg_threat_score": avg_threat,
    }


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        loop="asyncio",
    )