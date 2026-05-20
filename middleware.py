"""
middleware.py — Vault-Audit Phase 4
====================================
Sits between the caller and MongoDB. Every query goes through
`execute_query()`, which:
  1. Extracts a feature vector from the request
  2. Scores threat likelihood with the trained SVM
     (falls back to the Phase 2 rule-based scorer if the model is absent)
  3. Writes a tamper-proof, hash-chained audit log entry
  4. Returns the real query result (or blocks on high threat)

Phase 4 additions
-----------------
* Audit entries form a chain: each `integrity_hash` covers the previous
  entry's hash plus a monotonic `seq` number.
* `verify_chain()` walks the log and recomputes every hash, surfacing
  the first break (tamper, missing entry, or out-of-order seq).
"""

from __future__ import annotations
import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, TypedDict

from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import DuplicateKeyError

from baseline import THRESHOLDS
import svm_engine                          # Phase 3
import re
import time

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("vault-audit")

# ─── MongoDB connection ───────────────────────────────────────────────────────
# Override MONGO_URI in production; defaults to localhost for local dev.
_MONGO_URI: str = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
_client: MongoClient[dict[str, Any]] = MongoClient(_MONGO_URI)
_db: Database[dict[str, Any]] = _client["vault_audit_db"]


def get_db() -> Database[dict[str, Any]]:
    """Public accessor for the database — use this instead of _db directly."""
    return _db


# ─── Feature extraction ───────────────────────────────────────────────────────

SENSITIVE_FIELDS: set[str] = set(THRESHOLDS["sensitive_fields"])
HIGH_RISK_OPS: set[str]    = set(THRESHOLDS["high_risk_ops"])
MAX_SAFE_COUNT: int         = int(THRESHOLDS["max_safe_record_count"])


# ─── Injection detection ─────────────────────────────────────────────────────

# Operators that execute arbitrary server-side code — always block immediately.
_CRITICAL_OPERATORS: frozenset[str] = frozenset({
    "$where", "$function", "$accumulator", "$expr",
})

# Operators that are suspicious but sometimes legitimate (score bump only).
_SUSPICIOUS_OPERATORS: frozenset[str] = frozenset({
    "$gt", "$gte", "$lt", "$lte", "$ne", "$in", "$nin",
    "$regex", "$text", "$mod",
})

# Patterns that suggest JS injection attempts inside string values.
_JS_PATTERNS: re.Pattern[str] = re.compile(
    r"(sleep\s*\(|function\s*\(|this\.|db\.|return\s+true|while\s*\(1\))",
    re.IGNORECASE,
)

_MAX_FILTER_DEPTH = 5   # deeply nested objects are a red flag


def _scan_filter(
    obj: Any,
    depth: int = 0,
) -> tuple[bool, bool, bool]:
    """
    Recursively walk query_filter and return (critical, suspicious, js_payload).
    - critical    : a code-execution operator was found → instant block
    - suspicious  : an operator appears at the TOP level with no field wrapper
                    (e.g. {"$gt": ""} is an injection attempt; {"salary": {"$gt": 0}}
                    is normal field-scoped usage and is NOT flagged here)
    - js_payload  : a string value matches JS injection patterns

    Depth convention
    ----------------
    depth=0  : top-level keys of the filter dict  — suspicious operators flagged
    depth=1  : values of field keys               — operators here are field-scoped (normal)
    depth>=2 : deeper nesting                     — treated as suspicious if > _MAX_FILTER_DEPTH
    """
    if depth > _MAX_FILTER_DEPTH:
        return False, True, False   # excessive nesting → suspicious

    critical   = False
    suspicious = False
    js_payload = False

    if isinstance(obj, dict):
        k: str
        v: Any
        for k, v in obj.items():
            if k in _CRITICAL_OPERATORS:
                critical = True
            elif k.startswith("$"):
                # Suspicious only if at the top level (unscoped operator injection).
                # Field-scoped operators like {"salary": {"$gt": 0}} appear at depth>=1
                # and are normal MongoDB query syntax, not injection.
                if depth == 0 and k in _SUSPICIOUS_OPERATORS:
                    suspicious = True
                elif depth == 0 and k not in _CRITICAL_OPERATORS:
                    suspicious = True   # unknown top-level operator
            c2, s2, j2 = _scan_filter(v, depth + 1)
            critical   = critical   or c2
            suspicious = suspicious or s2
            js_payload = js_payload or j2

    elif isinstance(obj, str):
        if _JS_PATTERNS.search(obj):
            js_payload = True

    elif isinstance(obj, list):
        item: Any
        for item in obj:
            c2, s2, j2 = _scan_filter(item, depth + 1)
            critical   = critical   or c2
            suspicious = suspicious or s2
            js_payload = js_payload or j2

    return critical, suspicious, js_payload


# ─── Rate limiting (DDoS detection) ──────────────────────────────────────────
# In-memory sliding window — no DB round-trips, zero latency overhead.
# Keyed by IP address. Resets when the process restarts (academic prototype).

RATE_LIMIT_WINDOW_SECONDS = 60    # sliding window size
RATE_LIMIT_MAX_REQUESTS   = 10    # max requests per IP per window

# { ip: [timestamp, ...] }
_rate_limit_store: dict[str, list[float]] = {}
_rate_limit_lock = __import__("threading").Lock()


def _check_rate_limit(ip_address: str) -> tuple[bool, int]:
    """
    In-memory sliding-window rate limiter.
    Returns (exceeded, request_count_in_window).
    Thread-safe via a module-level lock.
    """
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW_SECONDS

    with _rate_limit_lock:
        timestamps = _rate_limit_store.get(ip_address, [])
        # Prune expired entries and append current timestamp
        timestamps = [t for t in timestamps if t >= window_start]
        timestamps.append(now)
        _rate_limit_store[ip_address] = timestamps
        count = len(timestamps)

    exceeded = count > RATE_LIMIT_MAX_REQUESTS
    if exceeded:
        log.warning(
            "Rate limit exceeded for IP %s: %d requests in %ds window",
            ip_address, count, RATE_LIMIT_WINDOW_SECONDS,
        )
    return exceeded, count


def extract_features(
    query_type: str,
    query_filter: dict[str, Any],
    record_count: int,
    user_id: str,
    ip_address: str = "127.0.0.1",
) -> dict[str, Any]:
    """
    Returns a flat dict of named features.
    The SVM in Phase 3 will consume the numeric subset;
    the string fields are kept for logging / explainability.

    Phase 5 additions
    -----------------
    * injection_critical : code-execution operator detected (instant block)
    * injection_suspicious : comparison/regex operator in filter (score bump)
    * js_payload         : JS-like string value detected in filter
    * rate_limited       : IP exceeded request rate limit in sliding window
    """
    filter_fields: list[str] = list(query_filter.keys())
    has_sensitive   = int(bool(SENSITIVE_FIELDS & set(filter_fields)))
    is_empty_filter = int(len(filter_fields) == 0)
    is_high_risk_op = int(query_type in HIGH_RISK_OPS)
    exceeds_limit   = int(record_count > MAX_SAFE_COUNT)
    bulk_sensitive  = int(
        is_empty_filter == 1 and
        (has_sensitive == 1 or record_count > MAX_SAFE_COUNT)
    )

    # Phase 5: injection detection
    critical, suspicious, js_payload = _scan_filter(query_filter)

    # Phase 5: rate limiting
    rate_exceeded, req_count = _check_rate_limit(ip_address)

    return {
        # ── numeric (SVM inputs) ──────────────────────
        "record_count":          record_count,
        "is_empty_filter":       is_empty_filter,
        "has_sensitive":         has_sensitive,
        "is_high_risk_op":       is_high_risk_op,
        "exceeds_limit":         exceeds_limit,
        "bulk_sensitive":        bulk_sensitive,
        # ── Phase 5: attack detection ─────────────────
        "injection_critical":    int(critical),
        "injection_suspicious":  int(suspicious),
        "js_payload":            int(js_payload),
        "rate_limited":          int(rate_exceeded),
        "request_count":         req_count,
        # ── metadata ─────────────────────────────────
        "query_type":            query_type,
        "filter_fields":         filter_fields,
        "user_id":               user_id,
        "ip_address":            ip_address,
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
        score += 0.20

    # bulk_sensitive: additive nudge when empty filter AND sensitive fields
    # are both present (is_empty_filter already scored 0.45 above, but
    # bulk-exfil of sensitive data warrants an extra push toward the block
    # threshold).  The `not is_empty_filter` guard was incorrect — it made
    # this branch unreachable — removed.
    if features["bulk_sensitive"]:
        score += 0.05

    # ── Phase 5: injection and rate-limit scoring ──────────────────────────
    # Critical injection operators execute server-side code — instant block.
    if features.get("injection_critical"):
        score = 1.0
        return score

    # JS-like payload in a string value — instant block (code execution risk).
    if features.get("js_payload"):
        score = 1.0
        return score

    # Suspicious comparison/regex operators — flags the query; combine with
    # exfil signals to push toward blocked.
    if features.get("injection_suspicious"):
        score += 0.50

    # IP exceeded sliding-window rate limit — instant block.
    if features.get("rate_limited"):
        score = 1.0
        return score

    return min(round(score, 4), 1.0)


# ─── Integrity hash (Phase 4: chained) ────────────────────────────────────────

GENESIS_PREV_HASH = "GENESIS"


def _make_hash(
    timestamp: datetime,
    user_id: str,
    query_type: str,
    record_count: int,
    threat_score: float,
    query_filter: dict[str, Any],
    prev_hash: str,
    seq: int,
) -> str:
    """
    SHA-256 over the core log fields *plus* the previous entry's hash and
    this entry's sequence number. Any retrospective edit, insert, or delete
    anywhere in the log breaks every hash downstream from the tamper point.
    """
    payload = json.dumps(
        {
            "timestamp":    timestamp.isoformat(),
            "user_id":      user_id,
            "query_type":   query_type,
            "record_count": record_count,
            "threat_score": threat_score,
            "query_filter": query_filter,
            "prev_hash":    prev_hash,
            "seq":          seq,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


# ─── Audit log writer (Phase 4: chained, retry on seq collision) ──────────────

MAX_SEQ_RETRIES = 5


def _last_entry(audit_logs: Collection[dict[str, Any]]) -> dict[str, Any] | None:
    """Returns the entry with the highest seq, or None if the log is empty."""
    cursor = audit_logs.find({}, {"_id": 0, "seq": 1, "integrity_hash": 1}) \
                       .sort("seq", -1).limit(1)
    docs = list(cursor)
    return docs[0] if docs else None


def write_audit_log(
    user_id: str,
    query_type: str,
    record_count: int,
    threat_score: float,
    query_filter: dict[str, Any],
    ip_address: str = "127.0.0.1",
    flagged: bool = False,
) -> str:
    """
    Writes one chained entry to audit_logs. Returns the integrity hash.

    Concurrency
    -----------
    `seq` has a unique index. If two writers race and pick the same next-seq,
    one insert raises DuplicateKeyError; we re-read the tail and retry up to
    MAX_SEQ_RETRIES times. In the single-process academic setup this is
    essentially never exercised, but it makes the contract correct.
    """
    audit_logs: Collection[dict[str, Any]] = _db["audit_logs"]

    for attempt in range(MAX_SEQ_RETRIES):
        tail = _last_entry(audit_logs)
        if tail is None:
            next_seq:  int = 0
            prev_hash: str = GENESIS_PREV_HASH
        else:
            next_seq  = int(tail["seq"]) + 1
            prev_hash = str(tail["integrity_hash"])

        # MongoDB / BSON stores datetimes at millisecond precision. Truncate
        # microseconds before hashing so the write-time hash matches what
        # verify_chain() will recompute after reading the entry back.
        now = datetime.now(timezone.utc)
        ts  = now.replace(microsecond=(now.microsecond // 1000) * 1000)

        integrity_hash = _make_hash(
            ts, user_id, query_type, record_count,
            threat_score, query_filter, prev_hash, next_seq,
        )

        try:
            audit_logs.insert_one(
                {
                    "seq":            next_seq,
                    "timestamp":      ts,
                    "user_id":        user_id,
                    "query_type":     query_type,
                    "record_count":   int(record_count),
                    "threat_score":   float(threat_score),
                    "integrity_hash": integrity_hash,
                    "prev_hash":      prev_hash,
                    "query_filter":   query_filter,
                    "ip_address":     ip_address,
                    "flagged":        flagged,
                }
            )
            return integrity_hash
        except DuplicateKeyError:
            # Another writer beat us to this seq — re-read tail and try again.
            log.warning(
                "audit_log seq collision at seq=%d (attempt %d/%d) — retrying",
                next_seq, attempt + 1, MAX_SEQ_RETRIES,
            )
            continue

    raise RuntimeError(
        f"write_audit_log: failed to claim a seq after {MAX_SEQ_RETRIES} retries"
    )


# ─── Chain verification (Phase 4) ─────────────────────────────────────────────

class ChainVerification(TypedDict):
    valid:           bool
    total:           int
    first_break_seq: int | None
    reason:          str | None


def verify_chain() -> ChainVerification:
    """
    Walks audit_logs in seq order and recomputes each entry's hash, checking
    that:
      * seq starts at 0 and is gap-free,
      * prev_hash matches the previous entry's integrity_hash (or GENESIS for seq 0),
      * the recomputed hash equals the stored integrity_hash.

    Returns {valid, total, first_break_seq, reason}. On a clean chain,
    first_break_seq and reason are None.
    """
    audit_logs: Collection[dict[str, Any]] = _db["audit_logs"]
    # Project only the fields we need — avoids loading query_filter blobs
    # for every entry when the collection is large.
    _VERIFY_PROJECTION = {
        "_id": 0, "seq": 1, "timestamp": 1, "user_id": 1,
        "query_type": 1, "record_count": 1, "threat_score": 1,
        "integrity_hash": 1, "prev_hash": 1, "query_filter": 1,
    }
    cursor = audit_logs.find({}, _VERIFY_PROJECTION).sort("seq", 1)

    expected_seq    = 0
    expected_prev   = GENESIS_PREV_HASH
    total           = 0

    for entry in cursor:
        total += 1
        seq = int(entry.get("seq", -1))

        # ── 1. seq gap / out-of-order ──────────────────────────────────────
        if seq != expected_seq:
            return {
                "valid":           False,
                "total":           total,
                "first_break_seq": expected_seq,
                "reason":          (
                    f"seq mismatch: expected {expected_seq}, found {seq} "
                    "(entry deleted or out of order)"
                ),
            }

        # ── 2. prev_hash mismatch ──────────────────────────────────────────
        stored_prev = str(entry.get("prev_hash", ""))
        if stored_prev != expected_prev:
            return {
                "valid":           False,
                "total":           total,
                "first_break_seq": seq,
                "reason":          (
                    f"prev_hash mismatch at seq={seq}: "
                    f"expected {expected_prev[:12]}…, found {stored_prev[:12]}…"
                ),
            }

        # ── 3. recompute hash and compare ──────────────────────────────────
        ts_raw = entry["timestamp"]
        # MongoDB returns datetimes as naive UTC; re-tag for hash determinism.
        if ts_raw.tzinfo is None:
            ts_raw = ts_raw.replace(tzinfo=timezone.utc)
        recomputed = _make_hash(
            ts_raw,
            str(entry["user_id"]),
            str(entry["query_type"]),
            int(entry["record_count"]),
            float(entry["threat_score"]),
            dict(entry.get("query_filter", {})),
            stored_prev,
            seq,
        )
        stored_hash = str(entry["integrity_hash"])
        if recomputed != stored_hash:
            return {
                "valid":           False,
                "total":           total,
                "first_break_seq": seq,
                "reason":          (
                    f"integrity_hash mismatch at seq={seq}: "
                    f"recomputed {recomputed[:12]}…, stored {stored_hash[:12]}…"
                ),
            }

        expected_prev = stored_hash
        expected_seq += 1

    return {
        "valid":           True,
        "total":           total,
        "first_break_seq": None,
        "reason":          None,
    }


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

    features = extract_features(query_type, query_filter, pre_count, user_id, ip_address)
    threat_score = score_threat(features)
    flagged      = threat_score >= 0.50

    log_level = logging.WARNING if flagged else logging.INFO
    # Build a human-readable reason for logging / response
    _reasons: list[str] = []
    if features.get("injection_critical"):  _reasons.append("injection:critical-operator")
    if features.get("js_payload"):          _reasons.append("injection:js-payload")
    if features.get("injection_suspicious"):_reasons.append("injection:suspicious-operator")
    if features.get("rate_limited"):        _reasons.append("rate-limit-exceeded")
    _threat_reason = ", ".join(_reasons) if _reasons else "exfil-pattern"

    log.log(
        log_level,
        "query=%s user=%s ip=%s filter=%s count=%d threat=%.4f flagged=%s reason=%s",
        query_type, user_id, ip_address, query_filter,
        pre_count, threat_score, flagged, _threat_reason,
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
            "reason":       _threat_reason,
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