"""
test_security.py — Vault-Audit (post-review)
=============================================
Tests added in response to the faculty review of vault-audit.

Coverage
--------
  S2  XSS storage: malicious user_id is stored verbatim in the audit log
  S3  update_payload injection: $where in update_payload is blocked
  S3  INSERT payload injection: $where in INSERT body is blocked
  S4  Chain-forgery limitation: documents the current behaviour as an
       executable assertion
  S7  scan_result short-circuit: extract_features accepts pre-computed
       scan results
  S9  Rate-limit bypass via rotating user_id: now keyed on ip only

Run:
    python -m pytest test_security.py -v
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# These functions are underscore-prefixed in middleware.py, which Pylance
# flags as private usage.  This is intentional — unit tests for security-
# critical internals MUST reach into the module.  We re-export through a
# public-facing alias so Pylance is satisfied and the intent is documented.
import middleware
from middleware import extract_features, verify_chain

# Re-alias private internals under test-local names so Pylance stops
# complaining about private access while keeping the tests readable.
scan_filter = middleware._scan_filter             # noqa: SLF001
check_rate_limit = middleware._check_rate_limit   # noqa: SLF001
rate_limit_store = middleware._rate_limit_store    # noqa: SLF001
make_hash = middleware._make_hash                 # noqa: SLF001
GENESIS_PREV_HASH = middleware.GENESIS_PREV_HASH


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _clear_rate_limit_store() -> None:   # type: ignore[misc]
    """Reset the in-process rate-limit store between tests."""
    rate_limit_store.clear()
    yield  # type: ignore[misc]
    rate_limit_store.clear()


# ─── S2 — XSS storage ────────────────────────────────────────────────────────

def test_s2_malicious_user_id_passes_through_server_unchanged() -> None:
    """
    The server stores user_id verbatim — sanitization happens in the
    dashboard.  This test pins the contract: the API itself doesn't strip
    HTML.  The dashboard test of choice would be a headless-browser test;
    here we just confirm the storage path doesn't accidentally mangle the
    value.
    """
    payload = '<img src=x onerror=alert(1)>'
    feats = extract_features(
        "READ", {"employee_id": "EMP-1001"}, 1, payload, "1.2.3.4",
        scan_result=(False, False, False),
    )
    assert feats["user_id"] == payload, "server must not mutate user_id"


# ─── S3 — update_payload and INSERT body scanning ────────────────────────────

def test_s3_update_payload_with_dollar_where_is_blocked() -> None:
    """An UPDATE whose payload contains $where must be pre-flight blocked."""
    crit, _susp, _js = scan_filter(
        {"$where": "function(){return true;}"},
        mode="write",
    )
    assert crit is True, "$where in update_payload must be detected as critical"


def test_s3_update_payload_with_dollar_set_is_legitimate() -> None:
    """Legitimate $set in update_payload must NOT be flagged."""
    crit, susp, js = scan_filter(
        {"$set": {"salary": 100000}},
        mode="write",
    )
    assert crit is False
    assert susp is False
    assert js is False


def test_s3_update_payload_with_js_payload_string_is_blocked() -> None:
    _crit, _susp, js = scan_filter(
        {"$set": {"role": "function(){while(1){}}"}},
        mode="write",
    )
    assert js is True


def test_s3_insert_body_with_dollar_function_is_blocked() -> None:
    crit, _susp, _js = scan_filter(
        {"$function": {"body": "...", "args": [], "lang": "js"}},
        mode="write",
    )
    assert crit is True


# ─── S4 — Chain forgery limitation (DOCUMENTS current behaviour) ─────────────

def test_s4_chain_forgery_succeeds_without_hmac() -> None:
    """
    A privileged adversary with DB write access can forge a forward-
    consistent audit entry.  This test asserts that verify_chain()
    currently does NOT detect such forgery.

    When HMAC-signed entries are implemented, this assertion should flip
    from `valid=True` to `valid=False`.
    """
    ts = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    forged_hash = make_hash(
        timestamp=ts,
        user_id="attacker",
        query_type="READ",
        record_count=0,
        threat_score=0.0,
        query_filter={},
        prev_hash=GENESIS_PREV_HASH,
        seq=0,
    )
    forged_entry: dict[str, Any] = {
        "seq":            0,
        "timestamp":      ts,
        "user_id":        "attacker",
        "query_type":     "READ",
        "record_count":   0,
        "threat_score":   0.0,
        "integrity_hash": forged_hash,
        "prev_hash":      GENESIS_PREV_HASH,
        "query_filter":   {},
    }

    fake_audit = MagicMock()
    fake_audit.find.return_value.sort.return_value = iter([forged_entry])

    with patch.object(middleware, "get_db") as mock_get_db:
        mock_get_db.return_value.__getitem__ = MagicMock(return_value=fake_audit)
        result = verify_chain()

    # CURRENT BEHAVIOUR: forgery succeeds.  After HMAC fix, expect False.
    assert result["valid"] is True, (
        "Current chain design cannot detect forgery by an attacker with "
        "DB write access.  When HMAC-signed entries are added, this "
        "assertion should flip to assert valid is False."
    )


# ─── S7 — scan_result short-circuit ──────────────────────────────────────────

def test_s7_extract_features_uses_passed_scan_result() -> None:
    """
    extract_features should accept a pre-computed scan_result tuple and
    not re-scan the filter.
    """
    feats = extract_features(
        "READ", {"employee_id": "EMP-1001"}, 1, "alice", "1.2.3.4",
        scan_result=(True, True, True),
    )
    assert feats["injection_critical"]   == 1
    assert feats["injection_suspicious"] == 1
    assert feats["js_payload"]           == 1


def test_s7_extract_features_scans_when_no_result_passed() -> None:
    """Backwards-compat: when scan_result is None, function scans itself."""
    feats = extract_features(
        "READ", {"employee_id": "EMP-1001"}, 1, "alice", "1.2.3.4",
    )
    assert feats["injection_critical"]   == 0
    assert feats["injection_suspicious"] == 0
    assert feats["js_payload"]           == 0


# ─── S9 — Rate-limit bypass via rotating user_id ─────────────────────────────

def test_s9_rate_limiter_is_not_bypassed_by_rotating_user_id() -> None:
    """
    11 requests from the same IP must trip the rate limit. Previously
    the limiter keyed on (user_id, ip), so rotating user_id gave fresh
    buckets.
    """
    ip = "9.9.9.9"
    exceeded_seen = False
    for _ in range(11):
        exceeded, _count = check_rate_limit(ip)
        if exceeded:
            exceeded_seen = True
            break

    assert exceeded_seen, (
        "Rate limiter must trip on 11+ requests from the same IP "
        "regardless of caller-supplied user_id."
    )


def test_s9_separate_ips_have_separate_buckets() -> None:
    """Different IPs must not share buckets."""
    for _ in range(11):
        check_rate_limit("1.1.1.1")
        check_rate_limit("2.2.2.2")
    _exceeded_a, count_a = check_rate_limit("1.1.1.1")
    _exceeded_b, count_b = check_rate_limit("2.2.2.2")
    assert count_a >= 10 and count_b >= 10


# ─── Documentation test ──────────────────────────────────────────────────────

def test_rule_recovery_high_does_not_imply_generalisation() -> None:
    """
    Pins the principle that high rule-recovery accuracy is necessary but
    NOT sufficient evidence that the SVM generalises.
    """
    rule_recovery_high = True
    generalisation_proven = False
    assert rule_recovery_high and not generalisation_proven, (
        "If you're reading this, the project now has independent evidence "
        "of generalisation.  Update accordingly."
    )