"""
middleware.py — Vault-Audit Phase 2
====================================
Sits between the caller and MongoDB. Every query goes through
`execute_query()`, which:
  1. Extracts a feature vector from the request
  2. Scores threat likelihood with a rule-based scorer
     (Phase 3 replaces this with the trained SVM)
  3. Writes a tamper-stamped audit log entry
  4. Returns the real query result (or blocks on high threat)
"""

from __future__ import annotations
import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any

from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database

from baseline import THRESHOLDS
import svm_engine                          # Phase 3

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("vault-audit")

# ─── MongoDB connection ───────────────────────────────────────────────────────
_client: MongoClient[dict[str, Any]] = MongoClient("mongodb://localhost:27017/")
_db: Database[dict[str, Any]] = _client["vault_audit_db"]


def get_db() -> Database[dict[str, Any]]:
    """Public accessor for the database — use this instead of _db directly."""
    return _db


# ─── Feature extraction ───────────────────────────────────────────────────────

SENSITIVE_FIELDS: set[str] = set(THRESHOLDS["sensitive_fields"])
HIGH_RISK_OPS: set[str]    = set(THRESHOLDS["high_risk_ops"])
MAX_SAFE_COUNT: int         = int(THRESHOLDS["max_safe_record_count"])


def extract_features(
    query_type: str,
    query_filter: dict[str, Any],
    record_count: int,
    user_id: str,
) -> dict[str, Any]:
    """
    Returns a flat dict of named features.
    The SVM in Phase 3 will consume the numeric subset;
    the string fields are kept for logging / explainability.
    """
    filter_fields: list[str] = list(query_filter.keys())
    has_sensitive  = int(bool(SENSITIVE_FIELDS & set(filter_fields)))
    is_empty_filter = int(len(filter_fields) == 0)
    is_high_risk_op = int(query_type in HIGH_RISK_OPS)
    exceeds_limit   = int(record_count > MAX_SAFE_COUNT)
    bulk_sensitive  = int(
        is_empty_filter == 1 and
        (has_sensitive == 1 or record_count > MAX_SAFE_COUNT)
    )

    return {
        # ── numeric (SVM inputs) ──────────────────────
        "record_count":     record_count,
        "is_empty_filter":  is_empty_filter,
        "has_sensitive":    has_sensitive,
        "is_high_risk_op":  is_high_risk_op,
        "exceeds_limit":    exceeds_limit,
        "bulk_sensitive":   bulk_sensitive,
        # ── metadata ─────────────────────────────────
        "query_type":       query_type,
        "filter_fields":    filter_fields,
        "user_id":          user_id,
    }


# ─── Rule-based threat scorer (Phase 3 → SVM) ────────────────────────────────

def score_threat(features: dict[str, Any]) -> float:
    """
    Phase 3: delegates to the calibrated SVM pipeline when the model is loaded.
    Falls back to the original rule-based scorer automatically when the model
    is not available (e.g. CI runs without svm_model.joblib, unit tests).
    """
    if svm_engine.is_model_loaded():
        return svm_engine.svm_score(features)

    # ── Fallback: original rule-based scorer (Phase 2) ────────────────────
    score = 0.0

    if features["exceeds_limit"]:      # record_count > MAX_SAFE_COUNT
        score += 0.35

    if features["is_empty_filter"]:    # no filter → full collection scan
        score += 0.45

    if features["is_high_risk_op"]:    # DELETE
        score += 0.30

    if features["has_sensitive"]:      # ssn / bank_account / salary in filter
        score += 0.10

    # bulk_sensitive: only fires when BOTH empty filter AND sensitive — small
    # additive nudge; the two parent signals already cover most of the score
    if features["bulk_sensitive"] and not features["is_empty_filter"]:
        score += 0.05

    return min(round(score, 4), 1.0)


# ─── Integrity hash ───────────────────────────────────────────────────────────

def _make_hash(
    timestamp: datetime,
    user_id: str,
    query_type: str,
    record_count: int,
    threat_score: float,
    query_filter: dict[str, Any],
) -> str:
    """
    SHA-256 over the core log fields.
    Phase 4 will chain hashes (Merkle-style) for tamper-proof chains.
    """
    payload = json.dumps(
        {
            "timestamp":    timestamp.isoformat(),
            "user_id":      user_id,
            "query_type":   query_type,
            "record_count": record_count,
            "threat_score": threat_score,
            "query_filter": query_filter,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


# ─── Audit log writer ─────────────────────────────────────────────────────────

def write_audit_log(
    user_id: str,
    query_type: str,
    record_count: int,
    threat_score: float,
    query_filter: dict[str, Any],
    ip_address: str = "127.0.0.1",
    flagged: bool = False,
) -> str:
    """Writes one entry to audit_logs. Returns the integrity hash."""
    ts = datetime.now(timezone.utc)
    integrity_hash = _make_hash(
        ts, user_id, query_type, record_count, threat_score, query_filter
    )

    audit_logs: Collection[dict[str, Any]] = _db["audit_logs"]
    audit_logs.insert_one(
        {
            "timestamp":      ts,
            "user_id":        user_id,
            "query_type":     query_type,
            "record_count":   int(record_count),
            "threat_score":   float(threat_score),
            "integrity_hash": integrity_hash,
            "query_filter":   query_filter,
            "ip_address":     ip_address,
            "flagged":        flagged,
        }
    )
    return integrity_hash


# ─── Public query executor ────────────────────────────────────────────────────

BLOCK_THRESHOLD = 0.85   # scores at or above this are blocked outright

QueryResult = dict[str, Any]


def execute_query(
    query_type: str,
    query_filter: dict[str, Any],
    user_id: str,
    ip_address: str = "127.0.0.1",
    update_payload: dict[str, Any] | None = None,
) -> QueryResult:
    """
    The single entry-point for all DB operations.

    Parameters
    ----------
    query_type     : "READ" | "UPDATE" | "DELETE" | "INSERT"
    query_filter   : MongoDB filter dict  (e.g. {"employee_id": "EMP-1001"})
    user_id        : caller identity string
    ip_address     : caller IP (forwarded by FastAPI)
    update_payload : required when query_type == "UPDATE"

    Returns
    -------
    {
        "status":       "ok" | "blocked",
        "threat_score": float,
        "flagged":      bool,
        "data":         list[dict] | None,   # None when blocked
        "record_count": int,
        "hash":         str,
    }
    """
    sensitive_data: Collection[dict[str, Any]] = _db["sensitive_data"]

    # ── 1. Pre-run feature extraction (record_count needs a DB call) ──────────
    # For READ we count matches before fetching; for others count affected docs.
    pre_count: int = int(sensitive_data.count_documents(query_filter))

    features = extract_features(query_type, query_filter, pre_count, user_id)
    threat_score = score_threat(features)
    flagged      = threat_score >= 0.50

    log_level = logging.WARNING if flagged else logging.INFO
    log.log(
        log_level,
        "query=%s user=%s filter=%s count=%d threat=%.2f flagged=%s",
        query_type, user_id, query_filter, pre_count, threat_score, flagged,
    )

    # ── 2. Block if score is too high ─────────────────────────────────────────
    if threat_score >= BLOCK_THRESHOLD:
        write_audit_log(
            user_id, query_type, pre_count,
            threat_score, query_filter, ip_address, flagged=True,
        )
        log.warning("🚫  BLOCKED query from user=%s  threat=%.2f", user_id, threat_score)
        return {
            "status":       "blocked",
            "threat_score": threat_score,
            "flagged":      True,
            "data":         None,
            "record_count": pre_count,
            "hash":         "",
        }

    # ── 3. Execute the real MongoDB operation ─────────────────────────────────
    data: list[dict[str, Any]] | None = None
    actual_count = pre_count

    if query_type == "READ":
        cursor = sensitive_data.find(query_filter, {"_id": 0})
        data   = list(cursor)
        actual_count = len(data)

    elif query_type == "UPDATE":
        if update_payload is None:
            raise ValueError("update_payload required for UPDATE queries")
        result = sensitive_data.update_many(query_filter, {"$set": update_payload})
        actual_count = result.modified_count

    elif query_type == "DELETE":
        result = sensitive_data.delete_many(query_filter)
        actual_count = result.deleted_count

    elif query_type == "INSERT":
        if update_payload is None:
            raise ValueError("update_payload (the document) required for INSERT")
        sensitive_data.insert_one(update_payload)
        actual_count = 1

    # ── 4. Write audit log ────────────────────────────────────────────────────
    integrity_hash = write_audit_log(
        user_id, query_type, actual_count,
        threat_score, query_filter, ip_address, flagged,
    )

    return {
        "status":       "ok",
        "threat_score": threat_score,
        "flagged":      flagged,
        "data":         data,
        "record_count": actual_count,
        "hash":         integrity_hash,
    }