"""
Baseline definitions for Vault-Audit.

This module is the **single source of truth** for:
  - threshold constants (max safe record count, sensitive fields, high-risk ops)
  - the rule-based threat scorer used as fallback AND for training-label
    generation in train_svm.py

Previously, `_rule_score` was duplicated in middleware.py and train_svm.py,
with a README warning that "the two must stay in sync." That maintenance
trap is now eliminated — both call `rule_score()` here.
"""
from typing import Any

# ─── What a NORMAL query looks like ──────────────────────────────────────────
NORMAL_EXAMPLES: list[dict[str, Any]] = [
    {"query_type": "READ",   "record_count": 1,  "filter_fields": ["employee_id"], "label": 0},
    {"query_type": "READ",   "record_count": 1,  "filter_fields": ["ssn"],         "label": 0},
    {"query_type": "UPDATE", "record_count": 1,  "filter_fields": ["employee_id"], "label": 0},
    {"query_type": "READ",   "record_count": 2,  "filter_fields": ["department"],  "label": 0},
]

# ─── What a THREAT query looks like ──────────────────────────────────────────
THREAT_EXAMPLES: list[dict[str, Any]] = [
    {"query_type": "READ",   "record_count": 10, "filter_fields": [],              "label": 1},
    {"query_type": "READ",   "record_count": 10, "filter_fields": ["salary"],      "label": 1},
    {"query_type": "DELETE", "record_count": 10, "filter_fields": [],              "label": 1},
    {"query_type": "READ",   "record_count": 5,  "filter_fields": ["ssn", "bank_account"], "label": 1},
]

# ─── Threshold rules ──────────────────────────────────────────────────────────
THRESHOLDS: dict[str, Any] = {
    "max_safe_record_count": 3,
    "sensitive_fields":      ["ssn", "bank_account", "salary"],
    "high_risk_ops":         ["DELETE"],
}

# ─── Scoring weights (single source of truth) ────────────────────────────────
# Both middleware.py (fallback) and train_svm.py (label generation) import these.
# Change here, and everything stays consistent — no two-file edits required.
WEIGHTS: dict[str, float] = {
    "exceeds_limit":       0.35,
    "is_empty_filter":     0.45,
    "is_high_risk_op":     0.30,
    "has_sensitive":       0.20,
    "bulk_sensitive":      0.05,
    "injection_suspicious": 0.50,
}

# ─── Decision thresholds ──────────────────────────────────────────────────────
FLAG_THRESHOLD:  float = 0.50
BLOCK_THRESHOLD: float = 0.85


def rule_score(
    record_count: int,
    is_empty_filter: int,
    has_sensitive: int,
    is_high_risk_op: int,
    exceeds_limit: int,
    bulk_sensitive: int,
    injection_critical: int = 0,
    js_payload: int = 0,
    injection_suspicious: int = 0,
    rate_limited: int = 0,
) -> float:
    """
    Canonical rule-based threat scorer.

    Used by:
      - middleware.score_threat() as the fallback when the SVM is unloaded
      - train_svm.py to generate labels for the synthetic training set

    Returns a score in [0.0, 1.0]. Any of the instant-block signals
    (injection_critical, js_payload, rate_limited) short-circuits to 1.0.
    """
    # Instant-block signals — code execution risk or DDoS.
    if injection_critical or js_payload or rate_limited:
        return 1.0

    score = 0.0
    if exceeds_limit:        score += WEIGHTS["exceeds_limit"]
    if is_empty_filter:      score += WEIGHTS["is_empty_filter"]
    if is_high_risk_op:      score += WEIGHTS["is_high_risk_op"]
    if has_sensitive:        score += WEIGHTS["has_sensitive"]
    if bulk_sensitive:       score += WEIGHTS["bulk_sensitive"]
    if injection_suspicious: score += WEIGHTS["injection_suspicious"]

    return min(round(score, 4), 1.0)


def label_from_score(score: float) -> int:
    """0 = clean | 1 = flagged (>=0.50) | 2 = blocked (>=0.85)."""
    if score >= BLOCK_THRESHOLD: return 2
    if score >= FLAG_THRESHOLD:  return 1
    return 0


if __name__ == "__main__":
    print("Normal query example:")
    print("  db.sensitive_data.find_one({'employee_id': 'EMP-1001'})")
    print()
    print("Threat query example (bulk exfil):")
    print("  db.sensitive_data.find({'salary': {'$gt': 0}})  <-- returns ALL records")