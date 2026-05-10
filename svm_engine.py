"""
Phase 3 — SVM Inference Engine
vault-audit/svm_engine.py

Drop-in replacement for the rule-based score_threat() in middleware.py.

Public API
----------
    load_model(path)        -> call once at app startup
    svm_score(features)     -> returns float threat score in [0, 1]
    is_model_loaded()       -> bool, used by /health route

Internal design
---------------
    predict_proba() returns  [P(clean), P(flagged), P(blocked)]
    Threat score             = P(flagged) + P(blocked)   in [0, 1]
    This preserves the >=0.50 flagged / >=0.85 blocked thresholds
    that api.py and middleware.py already use.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib  # type: ignore[import-untyped]
import numpy as np
from sklearn.pipeline import Pipeline

# ---------------------------------------------------------------------------
# Module-level state (loaded once at startup)
# ---------------------------------------------------------------------------

_pipeline: Pipeline | None = None
_model_path: str = ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_model(path: str | Path = "svm_model.joblib") -> None:
    """
    Load the serialized sklearn pipeline from *path*.

    Call this once from main startup (api.py lifespan).
    Raises FileNotFoundError if the model file is missing — run train_svm.py first.
    """
    global _pipeline, _model_path

    model_file = Path(path)
    if not model_file.exists():
        raise FileNotFoundError(
            f"SVM model not found at '{model_file}'. "
            "Run train_svm.py first to generate svm_model.joblib."
        )

    loaded: Any = joblib.load(model_file)  # type: ignore[no-untyped-call]
    _pipeline = loaded  # Pipeline from sklearn
    _model_path = str(model_file.resolve())
    print(f"[svm_engine] Model loaded from {_model_path}")


def is_model_loaded() -> bool:
    """Return True if a model has been loaded and is ready for inference."""
    return _pipeline is not None


def svm_score(features: dict[str, int | float]) -> float:
    """
    Compute a threat score in [0, 1] for the given feature vector.

    Parameters
    ----------
    features : dict with keys matching the contract in middleware.py:
        {
            "record_count":    int,
            "is_empty_filter": 0 | 1,
            "has_sensitive":   0 | 1,
            "is_high_risk_op": 0 | 1,
            "exceeds_limit":   0 | 1,
            "bulk_sensitive":  0 | 1,
        }

    Returns
    -------
    float
        P(flagged) + P(blocked) — compatible with the existing thresholds:
            score >= 0.50  ->  flagged
            score >= 0.85  ->  blocked

    Raises
    ------
    RuntimeError  if load_model() has not been called yet.
    """
    if _pipeline is None:
        raise RuntimeError(
            "SVM model is not loaded. Call svm_engine.load_model() at startup."
        )

    # Feature order MUST match train_svm.py FEATURE_NAMES and extract_features()
    vector = np.array([[
        features["record_count"],
        features["is_empty_filter"],
        features["has_sensitive"],
        features["is_high_risk_op"],
        features["exceeds_limit"],
        features["bulk_sensitive"],
    ]], dtype=float)

    # proba shape: (1, 3)  ->  [P(clean), P(flagged), P(blocked)]
    proba: np.ndarray = _pipeline.predict_proba(vector)  # type: ignore[no-untyped-call]
    p_flagged: float = float(proba[0, 1])
    p_blocked: float = float(proba[0, 2])

    threat_score = p_flagged + p_blocked
    # Clamp to [0, 1] to absorb any floating-point edge cases
    return max(0.0, min(1.0, threat_score))


def model_info() -> dict[str, str]:
    """Return metadata dict for the /health endpoint."""
    return {
        "loaded": str(is_model_loaded()),
        "path":   _model_path if _model_path else "not loaded",
    }