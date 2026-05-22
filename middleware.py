"""
middleware.py — Vault-Audit (post-review revision)
==================================================
Sits between the caller and MongoDB. Every query goes through
`execute_query()`, which:
  1. Scans both query_filter AND update_payload for injection (fix: S3)
  2. Extracts features and scores threat likelihood with the SVM
     (falls back to baseline.rule_score() if the model is absent)
  3. Writes a chained, tamper-evident audit log entry
  4. Returns the real query result (or blocks on high threat)

Review fixes applied
--------------------
* S3 — update_payload and INSERT bodies now scanned with the same scanner
       (mode='write') that previously only covered query_filter.
* S5 — MongoClient is lazy-initialised via get_client(); module import no
       longer opens a TCP connection. Tests can import this module without
       a live Mongo.
* S7 — _scan_filter is computed once in execute_query and passed into
       extract_features as a pre-computed tuple. No more duplicate work.
* S8 — rule scorer and weights live in baseline.py. This file imports
       them. The "two-files-must-stay-in-sync" comment is gone.
* S9 — Rate limiter keys on ip_address ONLY (was (user_id, ip_address)).
       A caller-supplied user_id cannot create fresh buckets anymore.
"""

from __future__ import annotations
import hashlib
import json
import logging
import os
import re
import threading
import time
from datetime import datetime, timezone
from typing import Any, TypedDict

from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import DuplicateKeyError

from baseline import (
    THRESHOLDS,
    BLOCK_THRESHOLD,
    FLAG_THRESHOLD,
    rule_score,
)
import svm_engine

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("vault-audit")

# ─── MongoDB connection (lazy — fix S5) ───────────────────────────────────────
_MONGO_URI: str = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
_client: MongoClient[dict[str, Any]] | None = None
_client_lock = threading.Lock()


def get_client() -> MongoClient[dict[str, Any]]:
    """
    Lazy MongoClient accessor. Opens the connection on first use, never at
    import time. This lets tests and tooling import middleware without a
    running Mongo (the previous module-level MongoClient(...) call would
    block during import).
    """
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:  # double-checked locking
                _client = MongoClient(
                    _MONGO_URI,
                    serverSelectionTimeoutMS=2000,
                    connectTimeoutMS=2000,
                    socketTimeoutMS=5000,
                )
    return _client


def get_db() -> Database[dict[str, Any]]:
    """Public accessor — use this everywhere instead of touching _client."""
    return get_client()["vault_audit_db"]


# ─── Feature extraction constants ─────────────────────────────────────────────
SENSITIVE_FIELDS: set[str] = set(THRESHOLDS["sensitive_fields"])
HIGH_RISK_OPS:    set[str] = set(THRESHOLDS["high_risk_ops"])
MAX_SAFE_COUNT:   int      = int(THRESHOLDS["max_safe_record_count"])


# ─── Injection detection ─────────────────────────────────────────────────────
_CRITICAL_OPERATORS: frozenset[str] = frozenset({
    "$where", "$function", "$accumulator", "$expr",
})
_SUSPICIOUS_OPERATORS: frozenset[str] = frozenset({
    "$gt", "$gte", "$lt", "$lte", "$ne", "$in", "$nin",
    "$regex", "$text", "$mod",
})
# Operators that are LEGITIMATE inside an update_payload — don't flag them
# in write-mode scans. Anything outside this set is still scrutinised.
_LEGITIMATE_UPDATE_OPERATORS: frozenset[str] = frozenset({
    "$set", "$unset", "$inc", "$mul", "$min", "$max", "$rename",
    "$currentDate", "$push", "$pull", "$addToSet", "$pop", "$each",
})
_JS_PATTERNS: re.Pattern[str] = re.compile(
    r"(sleep\s*\(|function\s*\(|this\.|db\.|return\s+true|while\s*\(1\))",
    re.IGNORECASE,
)
_MAX_FILTER_DEPTH = 5


def _scan_filter(
    obj: Any,
    depth: int = 0,
    mode: str = "read",
) -> tuple[bool, bool, bool]:
    """
    Recursively walk a MongoDB filter or write payload and return
    (critical, suspicious, js_payload).

    Parameters
    ----------
    mode : "read"  -> scanning a query_filter (top-level operators suspicious)
           "write" -> scanning an update_payload or INSERT body
                      (legitimate update operators like $set are allowed at
                      the top level; critical operators still block; JS
                      patterns in string values still block)

    fix S3
    ------
    Previously this only ran on query_filter. UPDATE and INSERT bodies were
    a free pass for injection — an attacker could ship
    update_payload = {"$where": "..."} and the scanner would never see it.
    """
    if depth > _MAX_FILTER_DEPTH:
        return False, True, False

    critical   = False
    suspicious = False
    js_payload = False

    if isinstance(obj, dict):
        k: str
        v: Any
        for k, v in obj.items():
            if k in _CRITICAL_OPERATORS:
                # Code-execution operators are blocked in BOTH modes.
                critical = True
            elif k.startswith("$"):
                if mode == "write":
                    # In write mode, legitimate update operators are fine;
                    # anything else at the top level is suspicious.
                    if depth == 0 and k not in _LEGITIMATE_UPDATE_OPERATORS:
                        suspicious = True
                else:  # read mode — original logic
                    if depth == 0 and k in _SUSPICIOUS_OPERATORS:
                        suspicious = True
                    elif depth == 0 and k not in _CRITICAL_OPERATORS:
                        suspicious = True
            c2, s2, j2 = _scan_filter(v, depth + 1, mode)
            critical   = critical   or c2
            suspicious = suspicious or s2
            js_payload = js_payload or j2

    elif isinstance(obj, str):
        if _JS_PATTERNS.search(obj):
            js_payload = True

    elif isinstance(obj, list):
        item: Any
        for item in obj:
            c2, s2, j2 = _scan_filter(item, depth + 1, mode)
            critical   = critical   or c2
            suspicious = suspicious or s2
            js_payload = js_payload or j2

    return critical, suspicious, js_payload


# ─── Rate limiting (fix S9 — key on IP only) ─────────────────────────────────
RATE_LIMIT_WINDOW_SECONDS = 60
RATE_LIMIT_MAX_REQUESTS   = 10

# Keyed on IP address ONLY. Previously keyed on (user_id, ip_address), but
# user_id is caller-supplied and unauthenticated — an attacker could iterate
# user_id values to get a fresh bucket per request.
_rate_limit_store: dict[str, list[float]] = {}
_rate_limit_lock = threading.Lock()


def _check_rate_limit(ip_address: str) -> tuple[bool, int]:
    """
    Sliding-window rate limiter keyed on ip_address.
    Returns (exceeded, request_count_in_window). Thread-safe.
    """
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW_SECONDS

    with _rate_limit_lock:
        timestamps = _rate_limit_store.get(ip_address, [])
        timestamps = [t for t in timestamps if t >= window_start]
        timestamps.append(now)
        _rate_limit_store[ip_address] = timestamps
        count = len(timestamps)

    exceeded = count > RATE_LIMIT_MAX_REQUESTS
    if exceeded:
        log.warning(
            "Rate limit exceeded for ip=%s: %d requests in %ds window",
            ip_address, count, RATE_LIMIT_WINDOW_SECONDS,
        )
    return exceeded, count


# ─── Feature extraction (fix S7 — scan results are now passed in) ────────────
def extract_features(
    query_type: str,
    query_filter: dict[str, Any],
    record_count: int,
    user_id: str,
    ip_address: str = "127.0.0.1",
    scan_result: tuple[bool, bool, bool] | None = None,
) -> dict[str, Any]:
    """
    Returns a flat dict of named features.

    fix S7
    ------
    Previously, this function called _scan_filter(query_filter) internally,
    duplicating the work execute_query() had already done. The scan_result
    can now be passed in as (critical, suspicious, js_payload). Falls back
    to scanning if None is passed (preserves backwards-compat with tests).
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

    if scan_result is None:
        critical, suspicious, js_payload = _scan_filter(query_filter)
    else:
        critical, suspicious, js_payload = scan_result

    return {
        "record_count":          record_count,
        "is_empty_filter":       is_empty_filter,
        "has_sensitive":         has_sensitive,
        "is_high_risk_op":       is_high_risk_op,
        "exceeds_limit":         exceeds_limit,
        "bulk_sensitive":        bulk_sensitive,
        "injection_critical":    int(critical),
        "injection_suspicious":  int(suspicious),
        "js_payload":            int(js_payload),
        "rate_limited":          0,
        "request_count":         0,
        "query_type":            query_type,
        "filter_fields":         filter_fields,
        "user_id":               user_id,
        "ip_address":            ip_address,
    }


# ─── Threat scorer (fix S8 — delegates to baseline.rule_score) ───────────────
def score_threat(features: dict[str, Any]) -> float:
    """
    Delegates to the calibrated SVM when loaded; otherwise to the canonical
    rule_score() in baseline.py. There is no longer a second copy of the
    weights in this file.
    """
    if svm_engine.is_model_loaded():
        return svm_engine.svm_score(features)

    return rule_score(
        record_count         = features["record_count"],
        is_empty_filter      = features["is_empty_filter"],
        has_sensitive        = features["has_sensitive"],
        is_high_risk_op      = features["is_high_risk_op"],
        exceeds_limit        = features["exceeds_limit"],
        bulk_sensitive       = features["bulk_sensitive"],
        injection_critical   = features.get("injection_critical", 0),
        js_payload           = features.get("js_payload", 0),
        injection_suspicious = features.get("injection_suspicious", 0),
        rate_limited         = features.get("rate_limited", 0),
    )


# ─── Integrity hash ───────────────────────────────────────────────────────────
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


# ─── Audit log writer ─────────────────────────────────────────────────────────
MAX_SEQ_RETRIES = 5


def _last_entry(audit_logs: Collection[dict[str, Any]]) -> dict[str, Any] | None:
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
    Writes one chained entry to audit_logs. Retries on seq collision.
    """
    audit_logs: Collection[dict[str, Any]] = get_db()["audit_logs"]

    for attempt in range(MAX_SEQ_RETRIES):
        tail = _last_entry(audit_logs)
        if tail is None:
            next_seq:  int = 0
            prev_hash: str = GENESIS_PREV_HASH
        else:
            next_seq  = int(tail["seq"]) + 1
            prev_hash = str(tail["integrity_hash"])

        # Truncate microseconds (BSON only stores millisecond precision)
        now = datetime.now(timezone.utc)
        ts  = now.replace(microsecond=(now.microsecond // 1000) * 1000)

        integrity_hash = _make_hash(
            ts, user_id, query_type, record_count,
            threat_score, query_filter, prev_hash, next_seq,
        )

        try:
            audit_logs.insert_one({
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
            })
            return integrity_hash
        except DuplicateKeyError:
            log.warning(
                "audit_log seq collision at seq=%d (attempt %d/%d) — retrying",
                next_seq, attempt + 1, MAX_SEQ_RETRIES,
            )
            continue

    raise RuntimeError(
        f"write_audit_log: failed to claim a seq after {MAX_SEQ_RETRIES} retries"
    )


# ─── Chain verification ───────────────────────────────────────────────────────
class ChainVerification(TypedDict):
    valid:           bool
    total:           int
    first_break_seq: int | None
    reason:          str | None


def verify_chain() -> ChainVerification:
    audit_logs: Collection[dict[str, Any]] = get_db()["audit_logs"]
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

        stored_prev = str(entry.get("prev_hash", ""))
        if stored_prev != expected_prev:
            return {
                "valid":           False,
                "total":           total,
                "first_break_seq": seq,
                "reason":          (
                    f"prev_hash mismatch at seq={seq}: "
                    f"expected {expected_prev[:12]}..., found {stored_prev[:12]}..."
                ),
            }

        ts_raw = entry["timestamp"]
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
                    f"recomputed {recomputed[:12]}..., stored {stored_hash[:12]}..."
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
    Returns a dict with status, threat_score, flagged, data, record_count, hash.
    """
    sensitive_data: Collection[dict[str, Any]] = get_db()["sensitive_data"]

    # ── 1. Pre-flight security checks (NO DB calls yet) ───────────────────────
    # Scan query_filter (read mode).
    f_critical, f_suspicious, f_js = _scan_filter(query_filter, mode="read")

    # fix S3: scan update_payload too. Critical operators and JS patterns are
    # blocked in writes; legitimate update operators ($set, $inc, ...) are
    # allowed via the "write" mode.
    p_critical = p_js = False
    if update_payload is not None:
        p_critical, _p_susp, p_js = _scan_filter(update_payload, mode="write")

    critical   = f_critical   or p_critical
    js_payload = f_js         or p_js
    suspicious = f_suspicious

    rate_exceeded, _req_count = _check_rate_limit(ip_address)

    if critical or js_payload or rate_exceeded:
        _reasons: list[str] = []
        if critical:      _reasons.append("injection:critical-operator")
        if js_payload:    _reasons.append("injection:js-payload")
        if rate_exceeded: _reasons.append("rate-limit-exceeded")
        _threat_reason = ", ".join(_reasons)

        log.warning(
            "PRE-FLIGHT BLOCK user=%s ip=%s filter=%s payload_present=%s reason=%s",
            user_id, ip_address, query_filter,
            update_payload is not None, _threat_reason,
        )
        write_audit_log(
            user_id, query_type, 0,
            1.0, query_filter, ip_address, flagged=True,
        )
        return {
            "status":       "blocked",
            "threat_score": 1.0,
            "flagged":      True,
            "data":         None,
            "record_count": 0,
            "hash":         "",
            "reason":       _threat_reason,
        }

    # ── 2. Pre-run feature extraction ─────────────────────────────────────────
    if query_type in ("READ", "DELETE"):
        pre_count: int = int(sensitive_data.count_documents(query_filter))
    else:
        pre_count = 0

    # fix S7: pass the scan tuple we already computed instead of re-scanning.
    features = extract_features(
        query_type, query_filter, pre_count, user_id, ip_address,
        scan_result=(critical, suspicious, js_payload),
    )
    threat_score = score_threat(features)
    flagged      = threat_score >= FLAG_THRESHOLD

    log_level = logging.WARNING if flagged else logging.INFO
    _reasons = []
    if features.get("injection_suspicious"):
        _reasons.append("injection:suspicious-operator")
    _threat_reason = ", ".join(_reasons) if _reasons else "exfil-pattern"

    log.log(
        log_level,
        "query=%s user=%s ip=%s filter=%s count=%d threat=%.4f flagged=%s reason=%s",
        query_type, user_id, ip_address, query_filter,
        pre_count, threat_score, flagged, _threat_reason,
    )

    # ── 3. Block if score is too high ─────────────────────────────────────────
    if threat_score >= BLOCK_THRESHOLD:
        write_audit_log(
            user_id, query_type, pre_count,
            threat_score, query_filter, ip_address, flagged=True,
        )
        log.warning("BLOCKED query from user=%s threat=%.2f", user_id, threat_score)
        return {
            "status":       "blocked",
            "threat_score": threat_score,
            "flagged":      True,
            "data":         None,
            "record_count": pre_count,
            "hash":         "",
            "reason":       _threat_reason,
        }

    # ── 4. Execute the real MongoDB operation ─────────────────────────────────
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

    # ── 5. Write audit log ────────────────────────────────────────────────────
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