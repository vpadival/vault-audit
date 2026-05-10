"""
Phase 3 — SVM Training Script
vault-audit/train_svm.py

Generates a labelled synthetic dataset from the rule-based threat scorer,
trains a calibrated SVM classifier, and serializes it to svm_model.joblib.

Run once (or whenever you want to retrain):
    python train_svm.py

Output:
    svm_model.joblib    -- model loaded at startup by svm_engine.py
    training_report.txt -- classification report + feature importances
"""

from __future__ import annotations

import random
import textwrap
from pathlib import Path

import joblib  # type: ignore[import-untyped]
import numpy as np
from numpy import ndarray
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report  # type: ignore[import-untyped]
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

# ---------------------------------------------------------------------------
# 1.  Feature vector type alias
# ---------------------------------------------------------------------------
# Each sample is [record_count, is_empty_filter, has_sensitive,
#                  is_high_risk_op, exceeds_limit, bulk_sensitive]
# Labels: 0 = clean, 1 = flagged, 2 = blocked

FEATURE_NAMES: list[str] = [
    "record_count",
    "is_empty_filter",
    "has_sensitive",
    "is_high_risk_op",
    "exceeds_limit",
    "bulk_sensitive",
]


# ---------------------------------------------------------------------------
# 2.  Rule-based scorer (mirrors middleware.py exactly — single source of truth
#     for label generation; do NOT import middleware here to keep training
#     self-contained and runnable without a live MongoDB connection)
# ---------------------------------------------------------------------------

def _rule_score(
    record_count: int,
    is_empty_filter: int,
    has_sensitive: int,
    is_high_risk_op: int,
    exceeds_limit: int,
    bulk_sensitive: int,
) -> float:
    score = 0.0
    if exceeds_limit:
        score += 0.35
    if is_empty_filter:
        score += 0.45
    if is_high_risk_op:
        score += 0.30
    if has_sensitive:
        score += 0.10
    if bulk_sensitive:
        score += 0.05
    return min(score, 1.0)


def _label(score: float) -> int:
    """0 = clean  |  1 = flagged (>=0.50)  |  2 = blocked (>=0.85)"""
    if score >= 0.85:
        return 2
    if score >= 0.50:
        return 1
    return 0


# ---------------------------------------------------------------------------
# 3.  Synthetic dataset generator
# ---------------------------------------------------------------------------

def _generate_sample(rng: random.Random) -> tuple[list[int | float], int]:
    """Return one (feature_vector, label) pair."""
    is_empty_filter = rng.choices([0, 1], weights=[0.6, 0.4])[0]
    has_sensitive   = rng.choices([0, 1], weights=[0.7, 0.3])[0]
    is_high_risk_op = rng.choices([0, 1], weights=[0.8, 0.2])[0]

    # record_count: skew toward small queries; occasionally bulk
    record_count = rng.choices(
        population=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        weights=   [25, 20, 15, 10,  8,  7,  5,  4,  3,  3],
    )[0]

    exceeds_limit = 1 if record_count > 3 else 0

    # bulk_sensitive: non-overlapping bonus (empty filter AND sensitive fields,
    # but exceeds_limit is counted separately — mirrors middleware logic)
    bulk_sensitive = 1 if (is_empty_filter and has_sensitive) else 0

    score = _rule_score(
        record_count, is_empty_filter, has_sensitive,
        is_high_risk_op, exceeds_limit, bulk_sensitive,
    )
    label = _label(score)
    features: list[int | float] = [
        record_count, is_empty_filter, has_sensitive,
        is_high_risk_op, exceeds_limit, bulk_sensitive,
    ]
    return features, label


def generate_dataset(
    n_samples: int = 5_000,
    seed: int = 42,
) -> tuple[ndarray[tuple[int, int], np.dtype[np.float64]],
           ndarray[tuple[int],      np.dtype[np.int64]]]:
    """Return X (n_samples x 6) and y (n_samples,) arrays."""
    rng = random.Random(seed)
    X_list: list[list[int | float]] = []
    y_list: list[int] = []

    for _ in range(n_samples):
        features, label = _generate_sample(rng)
        X_list.append(features)
        y_list.append(label)

    # Deterministically add hard-coded edge cases to ensure all score bands
    # are well represented regardless of random seed
    hard_cases: list[tuple[list[int | float], int]] = [
        # clean -- single lookup
        ([1, 0, 0, 0, 0, 0], 0),
        # clean -- SSN lookup, no match
        ([0, 0, 1, 0, 0, 0], 0),
        # clean -- salary dump, small result
        ([2, 0, 1, 0, 0, 0], 0),
        # flagged -- full collection scan (score 0.80)
        ([10, 1, 0, 0, 1, 0], 1),
        # clean -- specific dept DELETE
        ([1, 0, 0, 1, 0, 0], 0),
        # blocked -- bulk DELETE no filter (score 1.00)
        ([10, 1, 0, 1, 1, 0], 2),
        # blocked -- empty filter + high risk + bulk sensitive
        ([5, 1, 1, 1, 1, 1], 2),
        # flagged -- bulk with sensitive fields
        ([4, 0, 1, 0, 1, 0], 1),
    ]
    # Add each hard case 50x for clear signal
    for feat, lbl in hard_cases:
        for _ in range(50):
            X_list.append(feat)
            y_list.append(lbl)

    X: ndarray[tuple[int, int], np.dtype[np.float64]] = np.array(X_list, dtype=float)
    y: ndarray[tuple[int], np.dtype[np.int64]] = np.array(y_list, dtype=int)
    return X, y


# ---------------------------------------------------------------------------
# 4.  Train
# ---------------------------------------------------------------------------

def train(output_dir: Path = Path(".")) -> None:
    print("=== vault-audit  Phase 3: SVM Training ===\n")

    # -- Dataset
    X, y = generate_dataset(n_samples=5_000, seed=42)
    counts = {lbl: int((y == lbl).sum()) for lbl in [0, 1, 2]}
    print(f"Dataset size : {len(X):,}  samples")
    print(f"Class counts : clean={counts[0]}  flagged={counts[1]}  blocked={counts[2]}\n")

    X_train: ndarray[tuple[int, int], np.dtype[np.float64]]
    X_test:  ndarray[tuple[int, int], np.dtype[np.float64]]
    y_train: ndarray[tuple[int], np.dtype[np.int64]]
    y_test:  ndarray[tuple[int], np.dtype[np.int64]]
    X_train, X_test, y_train, y_test = train_test_split(  # type: ignore[misc, arg-type]
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    # -- Pipeline: scaler + RBF-SVM + Platt scaling for predict_proba
    base_svm = SVC(
        kernel="rbf",
        C=10.0,
        gamma="scale",
        class_weight="balanced",   # handles any remaining imbalance
        probability=False,         # CalibratedClassifierCV adds Platt scaling
        random_state=42,
    )
    calibrated = CalibratedClassifierCV(base_svm, method="sigmoid", cv=5)
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    calibrated),
    ])

    # -- Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores: ndarray[tuple[int], np.dtype[np.float64]] = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="accuracy")  # type: ignore[misc, arg-type]
    print(f"CV accuracy  : {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    # -- Final fit on all training data
    pipeline.fit(X_train, y_train)  # type: ignore[arg-type]
    y_pred: ndarray[tuple[int], np.dtype[np.int64]] = pipeline.predict(X_test)  # type: ignore[assignment]

    report: str = classification_report(y_test, y_pred, target_names=["clean", "flagged", "blocked"], digits=4)  # type: ignore[assignment, arg-type]
    print("\nTest-set classification report:")
    print(report)

    # -- Verify predict_proba shape
    sample = np.array([[1, 0, 0, 0, 0, 0]], dtype=float)
    proba: ndarray[tuple[int, int], np.dtype[np.float64]] = pipeline.predict_proba(sample)  # type: ignore[assignment]
    assert proba.shape == (1, 3), "predict_proba must return 3-class probabilities"
    print(f"predict_proba smoke test OK  -> {proba}\n")

    # -- Serialize
    model_path = output_dir / "svm_model.joblib"
    joblib.dump(pipeline, model_path)  # type: ignore[no-untyped-call]
    print(f"Model saved  -> {model_path}")

    # -- Human-readable report
    report_path = output_dir / "training_report.txt"
    report_text = textwrap.dedent(f"""\
        vault-audit  Phase 3 -- SVM Training Report
        ============================================

        Dataset
        -------
        Total samples : {len(X):,}
        Class counts  : clean={counts[0]}  flagged={counts[1]}  blocked={counts[2]}

        Model
        -----
        Pipeline      : StandardScaler -> CalibratedClassifierCV(SVC, method=sigmoid, cv=5)
        Kernel        : RBF   C=10   gamma=scale   class_weight=balanced

        Cross-validation (StratifiedKFold n=5, on train split)
        -------------------------------------------------------
        Accuracy      : {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}

        Test-set classification report
        ------------------------------
        {report}

        Feature order (must match extract_features() in middleware.py)
        ---------------------------------------------------------------
        {chr(10).join(f"  [{i}] {name}" for i, name in enumerate(FEATURE_NAMES))}

        Thresholds applied in svm_engine.py
        ------------------------------------
          flagged  : P(flagged) + P(blocked) >= 0.50
          blocked  : P(flagged) + P(blocked) >= 0.85

        Notes
        -----
        Labels were generated by the rule-based scorer in middleware.py.
        The SVM generalises the same decision surface via learned weights,
        making it robust to feature values outside the rigid rule boundaries.
    """)
    report_path.write_text(report_text, encoding="utf-8")
    print(f"Report saved -> {report_path}\n")
    print("=== Training complete ===")


if __name__ == "__main__":
    train(output_dir=Path("."))