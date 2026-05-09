"""
Baseline definitions for Vault-Audit.
These thresholds become training labels for the ML model in Phase 3.
"""
from typing import Any
# ─── What a NORMAL query looks like ──────────────────────────────────────────
# Single record lookup by a specific identifier.
# record_count = 1, specific filter field, business-hours access.
NORMAL_EXAMPLES: list[dict[str, Any]] = [
    {"query_type": "READ",   "record_count": 1,  "filter_fields": ["employee_id"], "label": 0},
    {"query_type": "READ",   "record_count": 1,  "filter_fields": ["ssn"],         "label": 0},
    {"query_type": "UPDATE", "record_count": 1,  "filter_fields": ["employee_id"], "label": 0},
    {"query_type": "READ",   "record_count": 2,  "filter_fields": ["department"],  "label": 0},
]

# ─── What a THREAT query looks like ──────────────────────────────────────────
# Bulk reads, no filter (full collection scan), salary/SSN mass-dump.
# record_count >> 1, broad/empty filters, unusual hours.
THREAT_EXAMPLES: list[dict[str, Any]] = [
    {"query_type": "READ",   "record_count": 10, "filter_fields": [],              "label": 1},
    {"query_type": "READ",   "record_count": 10, "filter_fields": ["salary"],      "label": 1},
    {"query_type": "DELETE", "record_count": 10, "filter_fields": [],              "label": 1},
    {"query_type": "READ",   "record_count": 5,  "filter_fields": ["ssn", "bank_account"], "label": 1},
]

# ─── Threshold rules (used as fallback and for explainability) ────────────────
THRESHOLDS: dict[str, Any] = {
    "max_safe_record_count": 3,          # more than 3 docs returned → suspicious
    "sensitive_fields": ["ssn", "bank_account", "salary"],
    "high_risk_ops": ["DELETE"],
}

if __name__ == "__main__":
    print("Normal query example:")
    print("  db.sensitive_data.find_one({'employee_id': 'EMP-1001'})")
    print()
    print("Threat query example (bulk exfil):")
    print("  db.sensitive_data.find({'salary': {'$gt': 0}})  ← returns ALL records")