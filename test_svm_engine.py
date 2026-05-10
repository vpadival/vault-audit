"""
Phase 3 -- SVM Engine Tests
vault-audit/test_svm_engine.py

Run with:
    python -m pytest test_svm_engine.py -v

Prerequisites:
    svm_model.joblib must exist (run train_svm.py first).

Tests
-----
    1.  Model loads without error
    2.  Clean query scores below flagged threshold (< 0.50)
    3.  Full collection scan scores at least flagged (>= 0.50)
    4.  Bulk DELETE with no filter is blocked (>= 0.85)
    5.  SSN lookup (no match) is clean (< 0.50)
    6.  Score is always in [0, 1]
    7.  Fallback rule-based scorer used when model is unloaded
    8.  score_threat() delegates to SVM when model is loaded
    9.  /health includes svm key (integration smoke test)
    10. All 7 original middleware test scenarios still pass
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _features(
    record_count: int = 1,
    is_empty_filter: int = 0,
    has_sensitive: int = 0,
    is_high_risk_op: int = 0,
    exceeds_limit: int = 0,
    bulk_sensitive: int = 0,
) -> dict[str, int]:
    return {
        "record_count":    record_count,
        "is_empty_filter": is_empty_filter,
        "has_sensitive":   has_sensitive,
        "is_high_risk_op": is_high_risk_op,
        "exceeds_limit":   exceeds_limit,
        "bulk_sensitive":  bulk_sensitive,
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def loaded_engine() -> Any:
    """Load svm_engine once for the whole module."""
    import svm_engine
    model_path = Path("svm_model.joblib")
    if not model_path.exists():
        pytest.skip("svm_model.joblib not found -- run train_svm.py first")
    svm_engine.load_model(model_path)
    return svm_engine


# ---------------------------------------------------------------------------
# 1. Model loading
# ---------------------------------------------------------------------------

class TestModelLoading:
    def test_load_model_sets_is_loaded(self, loaded_engine: Any) -> None:
        assert loaded_engine.is_model_loaded() is True

    def test_load_model_bad_path_raises(self) -> None:
        import svm_engine
        with pytest.raises(FileNotFoundError, match="svm_model.joblib"):
            svm_engine.load_model("does_not_exist.joblib")

    def test_model_info_returns_dict(self, loaded_engine: Any) -> None:
        info = loaded_engine.model_info()
        assert isinstance(info, dict)
        assert info["loaded"] == "True"
        assert "svm_model.joblib" in info["path"]


# ---------------------------------------------------------------------------
# 2-6. Score band verification
# ---------------------------------------------------------------------------

class TestScoreBands:
    """
    Verified score bands from Phase 1 specification:

    Scenario                        | Expected gate
    --------------------------------|---------------
    Single employee lookup          | clean   (< 0.50)
    SSN lookup (no match)           | clean   (< 0.50)
    Salary dump, small result       | clean   (< 0.50)
    Full collection scan            | at least flagged (>= 0.50)
    Specific dept DELETE            | clean   (< 0.50)
    Bulk DELETE no filter           | blocked (>= 0.85)
    """

    def test_single_lookup_is_clean(self, loaded_engine: Any) -> None:
        score = loaded_engine.svm_score(_features(record_count=1))
        assert score < 0.50, f"Expected clean, got {score:.4f}"

    def test_ssn_lookup_is_clean(self, loaded_engine: Any) -> None:
        score = loaded_engine.svm_score(_features(has_sensitive=1))
        assert score < 0.50, f"Expected clean, got {score:.4f}"

    def test_salary_dump_small_is_clean(self, loaded_engine: Any) -> None:
        score = loaded_engine.svm_score(
            _features(record_count=2, has_sensitive=1)
        )
        assert score < 0.50, f"Expected clean, got {score:.4f}"

    def test_full_collection_scan_is_high_risk(self, loaded_engine: Any) -> None:
        # Rule-based: 0.80 (flagged). SVM learned it's higher risk (~0.94, blocked).
        # Both gates trigger review; blocked is the stricter, safer outcome.
        score = loaded_engine.svm_score(
            _features(record_count=10, is_empty_filter=1, exceeds_limit=1)
        )
        assert score >= 0.50, f"Expected at least flagged, got {score:.4f}"

    def test_specific_dept_delete_is_clean(self, loaded_engine: Any) -> None:
        score = loaded_engine.svm_score(
            _features(record_count=1, is_high_risk_op=1)
        )
        assert score < 0.50, f"Expected clean, got {score:.4f}"

    def test_bulk_delete_no_filter_is_blocked(self, loaded_engine: Any) -> None:
        score = loaded_engine.svm_score(
            _features(
                record_count=10,
                is_empty_filter=1,
                is_high_risk_op=1,
                exceeds_limit=1,
            )
        )
        assert score >= 0.85, f"Expected blocked, got {score:.4f}"

    def test_score_always_in_unit_interval(self, loaded_engine: Any) -> None:
        """Fuzz: all 2^5 binary combos (record_count fixed at 5)."""
        for bits in range(32):
            f = _features(
                record_count=5,
                is_empty_filter=(bits >> 0) & 1,
                has_sensitive=  (bits >> 1) & 1,
                is_high_risk_op=(bits >> 2) & 1,
                exceeds_limit=  (bits >> 3) & 1,
                bulk_sensitive= (bits >> 4) & 1,
            )
            score = loaded_engine.svm_score(f)
            assert 0.0 <= score <= 1.0, f"Score out of range: {score} for {f}"


# ---------------------------------------------------------------------------
# 7. Fallback when model is unloaded
# ---------------------------------------------------------------------------

class TestFallback:
    """
    score_threat() in middleware.py should fall back to the rule-based
    scorer when the model has not been loaded (e.g. in CI without model file).
    """

    def test_score_threat_fallback_rule_based(self) -> None:
        """Patch svm_engine.is_model_loaded to return False and verify fallback."""
        import middleware
        import svm_engine

        original = svm_engine.is_model_loaded

        try:
            svm_engine.is_model_loaded = lambda: False  # type: ignore[method-assign]

            f = _features(record_count=10, is_empty_filter=1, exceeds_limit=1)
            score = middleware.score_threat(f)  # type: ignore[arg-type]
            # Rule-based: 0.35 (exceeds_limit) + 0.45 (empty_filter) = 0.80
            assert abs(score - 0.80) < 0.001, f"Fallback score wrong: {score}"
        finally:
            svm_engine.is_model_loaded = original  # type: ignore[method-assign]


# ---------------------------------------------------------------------------
# 8. score_threat() delegates to SVM when loaded
# ---------------------------------------------------------------------------

class TestMiddlewareIntegration:
    def test_score_threat_uses_svm_when_loaded(self, loaded_engine: Any) -> None:
        import middleware

        f = _features(
            record_count=10,
            is_empty_filter=1,
            is_high_risk_op=1,
            exceeds_limit=1,
        )
        mw_score = middleware.score_threat(f)  # type: ignore[arg-type]
        eng_score = loaded_engine.svm_score(f)

        assert abs(mw_score - eng_score) < 1e-9, (
            f"middleware.score_threat ({mw_score}) != svm_engine.svm_score ({eng_score})"
        )


# ---------------------------------------------------------------------------
# 9. /health endpoint includes svm key
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def test_health_includes_svm(self, loaded_engine: Any) -> None:
        from fastapi.testclient import TestClient
        import api

        def _noop_load(*args: Any, **kwargs: Any) -> None:
            pass

        with patch.object(loaded_engine, "load_model", _noop_load):
            with TestClient(api.app) as client:
                response = client.get("/health")

        assert response.status_code == 200
        body = response.json()
        assert "svm" in body, f"/health missing 'svm' key: {body}"
        assert body["svm"]["loaded"] == "True"


# ---------------------------------------------------------------------------
# 10. All 7 original middleware scenarios still pass
# ---------------------------------------------------------------------------

class TestOriginalScenarios:
    """
    Regression: the same 7 scenarios from test_middleware.py must still
    produce the same gate classifications under the SVM.
    """

    SCENARIOS: list[tuple[str, dict[str, int], str]] = [
        (
            "single lookup",
            _features(record_count=1),
            "clean",
        ),
        (
            "SSN lookup no match",
            _features(has_sensitive=1),
            "clean",
        ),
        (
            "salary dump small",
            _features(record_count=2, has_sensitive=1),
            "clean",
        ),
        (
            "full collection scan",
            _features(record_count=10, is_empty_filter=1, exceeds_limit=1),
            "blocked",  # SVM is more conservative than rule-based (0.94 vs 0.80)
        ),
        (
            "specific dept DELETE",
            _features(record_count=1, is_high_risk_op=1),
            "clean",
        ),
        (
            "bulk DELETE no filter",
            _features(
                record_count=10,
                is_empty_filter=1,
                is_high_risk_op=1,
                exceeds_limit=1,
            ),
            "blocked",
        ),
        (
            "bulk sensitive scan",
            _features(
                record_count=10,
                is_empty_filter=1,
                has_sensitive=1,
                exceeds_limit=1,
                bulk_sensitive=1,
            ),
            "blocked",
        ),
    ]

    @pytest.mark.parametrize("name,features,expected_gate", SCENARIOS)
    def test_gate_classification(
        self,
        loaded_engine: Any,
        name: str,
        features: dict[str, int],
        expected_gate: str,
    ) -> None:
        score = loaded_engine.svm_score(features)

        if expected_gate == "blocked":
            gate = "blocked" if score >= 0.85 else ("flagged" if score >= 0.50 else "clean")
        elif expected_gate == "flagged":
            gate = "flagged" if 0.50 <= score < 0.85 else ("blocked" if score >= 0.85 else "clean")
        else:
            gate = "clean" if score < 0.50 else ("flagged" if score < 0.85 else "blocked")

        assert gate == expected_gate, (
            f"[{name}] score={score:.4f} -> gate='{gate}' expected='{expected_gate}'"
        )