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

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, Response, HTMLResponse
from pathlib import Path
from pydantic import BaseModel, Field

from middleware import execute_query, get_db, verify_chain
import svm_engine                          # Phase 3

# ─── App ─────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):          # type: ignore[type-arg]
    try:
        svm_engine.load_model("svm_model.joblib")
    except FileNotFoundError:
        import logging
        logging.getLogger("vault-audit").warning(
            "svm_model.joblib not found — rule-based scorer will be used. "
            "Run train_svm.py to generate the model."
        )
    yield


app = FastAPI(
    title="Vault-Audit API",
    description="Security middleware for MongoDB — intercepts, scores, and logs every query.",
    version="0.2.0",
    lifespan=lifespan,
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
    reason:       Union[str, None] = None


# ─── Helper ──────────────────────────────────────────────────────────────────

def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


async def init_db() -> None:
    """Compatibility stub used by tests; real DB setup happens elsewhere."""
    return None


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.get("/", tags=["meta"])
def root() -> dict[str, Any]:
    """API entry point — links to docs and available endpoints."""
    return {
        "service":   "vault-audit",
        "version":   "0.2.0",
        "docs":      "/docs",
        "endpoints": {
            "health":       "GET  /health",
            "dashboard":    "GET  /dashboard",
            "query":        "POST /query",
            "audit_logs":   "GET  /audit/logs",
            "audit_stats":  "GET  /audit/stats",
            "audit_verify": "GET  /audit/verify",
            "audit_attacks": "GET  /audit/attacks",
        },
    }


@app.get("/dashboard", tags=["meta"], response_class=HTMLResponse)
def dashboard() -> HTMLResponse:
    """Interactive web UI for exploring queries, audit logs, and chain integrity."""
    html_path = Path(__file__).parent / "dashboard.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.get("/health", tags=["meta"])
def health() -> dict[str, Any]:
    """Liveness probe."""
    return {
        "status":  "ok",
        "service": "vault-audit",
        "svm":     svm_engine.model_info(),    # Phase 3
    }


@app.post("/query", tags=["queries"])
async def run_query(payload: QueryRequest, request: Request) -> Response:
    """
    Submit a query against `sensitive_data`.
    The middleware classifies it, logs it, and either returns results
    or blocks the request with HTTP 403.
    """
    ip = _client_ip(request)

    try:
        result = await asyncio.to_thread(
            execute_query,
            payload.query_type,
            payload.query_filter,
            payload.user_id,
            ip,
            payload.update_payload,
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
                "reason":       result.get("reason", ""),
            },
        )

    return JSONResponse(content=QueryResponse(**result).model_dump())


@app.get("/audit/logs", tags=["audit"])
async def get_audit_logs(
    limit: int = 20,
    flagged_only: bool = False,
) -> list[dict[str, Any]]:
    """
    Retrieve recent audit log entries.
    - `limit`        — max entries to return (default 20, max 500)
    - `flagged_only` — when true, return only flagged (suspicious) logs
    """
    def _fetch() -> list[dict[str, Any]]:
        lim = max(1, min(limit, 500))
        db = get_db()
        filt: dict[str, Any] = {"flagged": True} if flagged_only else {}
        cursor = (
            db["audit_logs"]
            .find(filt, {"_id": 0})
            .sort("timestamp", -1)
            .limit(lim)
        )
        logs: list[dict[str, Any]] = []
        for doc in cursor:
            if "timestamp" in doc:
                doc["timestamp"] = doc["timestamp"].isoformat()
            logs.append(doc)
        return logs

    return await asyncio.to_thread(_fetch)


@app.get("/audit/stats", tags=["audit"])
async def get_audit_stats() -> dict[str, Any]:
    """High-level stats over all audit logs."""
    def _fetch() -> dict[str, Any]:
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

    return await asyncio.to_thread(_fetch)


@app.get("/audit/verify", tags=["audit"])
async def get_audit_verify() -> dict[str, Any]:
    """
    Phase 4: walk the audit log chain and verify integrity.
    Returns valid=true on a clean chain, or valid=false with the first
    broken seq and a human-readable reason.
    """
    result = await asyncio.to_thread(verify_chain)
    return dict(result)


@app.get("/audit/attacks", tags=["audit"])
async def get_audit_attacks(limit: int = 50) -> list[dict[str, Any]]:
    """
    Phase 5: return only audit entries that were flagged as injection or
    rate-limit attacks, newest first.
    """
    def _fetch() -> list[dict[str, Any]]:
        lim = max(1, min(limit, 500))
        db = get_db()
        cursor = (
            db["audit_logs"]
            .find({"threat_score": {"$gte": 0.85}}, {"_id": 0})
            .sort("timestamp", -1)
            .limit(lim)
        )
        logs: list[dict[str, Any]] = []
        for doc in cursor:
            if "timestamp" in doc:
                doc["timestamp"] = doc["timestamp"].isoformat()
            logs.append(doc)
        return logs

    return await asyncio.to_thread(_fetch)


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