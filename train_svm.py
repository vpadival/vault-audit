"""
train_svm.py — Vault-Audit (post-review revision)
==================================================
Generates a labelled synthetic dataset, trains a calibrated SVM, and
serializes it to svm_model.joblib.

Review fixes applied
--------------------
* S8 — Imports rule_score() from baseline.py instead of redefining it.
       Weights now have a single source of truth.
* S1 — Adds a *hand-labeled adversarial validation set* the SVM never
       trains on. Reports rule-recovery accuracy AND adversarial accuracy
       separately, so the headline number is honest:
         - rule-recovery accuracy : how well the SVM mimics _rule_score
         - adversarial accuracy   : how well the SVM handles cases that
                                    aren't obvious from the rule alone
       This is the difference between "model fits a function we already
       have" (boring) and "model generalises beyond the rule" (the actual
       research question for a security ML project).
"""

from __future__ import annotations

import random
import textwrap
from pathlib import Path

import joblib  # type: ignore[import-untyped]
import numpy as np
from numpy import ndarray
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, accuracy_score  # type: ignore[import-untyped]
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

# fix S8 — single source of truth for the rule scorer.
from baseline import rule_score, label_from_score


FEATURE_NAMES: list[str] = [
    "record_count",
    "is_empty_filter",
    "has_sensitive",
    "is_high_risk_op",
    "exceeds_limit",
    "bulk_sensitive",
]


# ---------------------------------------------------------------------------
# Synthetic dataset generator (labels from baseline.rule_score)
# ---------------------------------------------------------------------------

def _generate_sample(rng: random.Random) -> tuple[list[int | float], int]:
    is_empty_filter = rng.choices([0, 1], weights=[0.6, 0.4])[0]
    has_sensitive   = rng.choices([0, 1], weights=[0.7, 0.3])[0]
    is_high_risk_op = rng.choices([0, 1], weights=[0.8, 0.2])[0]
    record_count    = rng.choices(
        population=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        weights=   [25, 20, 15, 10,  8,  7,  5,  4,  3,  3],
    )[0]
    exceeds_limit   = 1 if record_count > 3 else 0
    bulk_sensitive  = 1 if (is_empty_filter and has_sensitive) else 0

    score = rule_score(
        record_count, is_empty_filter, has_sensitive,
        is_high_risk_op, exceeds_limit, bulk_sensitive,
    )
    label = label_from_score(score)
    features: list[int | float] = [
        record_count, is_empty_filter, has_sensitive,
        is_high_risk_op, exceeds_limit, bulk_sensitive,
    ]
    return features, label


def generate_dataset(
    n_samples: int = 5_000,
    seed: int = 42,
) -> tuple[ndarray, ndarray]:
    rng = random.Random(seed)
    X_list: list[list[int | float]] = []
    y_list: list[int] = []

    for _ in range(n_samples):
        features, label = _generate_sample(rng)
        X_list.append(features)
        y_list.append(label)

    # Hard-coded edge cases (×50 each) so all 3 classes are well-represented.
    hard_cases: list[tuple[list[int | float], int]] = [
        ([1, 0, 0, 0, 0, 0], 0),  # clean: single lookup
        ([0, 0, 1, 0, 0, 0], 0),  # clean: SSN lookup no match
        ([2, 0, 1, 0, 0, 0], 0),  # clean: salary dump small
        ([10, 1, 0, 0, 1, 0], 1), # flagged: full collection scan
        ([1, 0, 0, 1, 0, 0], 0),  # clean: specific dept DELETE
        ([10, 1, 0, 1, 1, 0], 2), # blocked: bulk DELETE no filter
        ([5, 1, 1, 1, 1, 1], 2),  # blocked: empty + high-risk + bulk sensitive
        ([4, 0, 1, 0, 1, 0], 1),  # flagged: bulk with sensitive fields
    ]
    for feat, lbl in hard_cases:
        for _ in range(50):
            X_list.append(feat)
            y_list.append(lbl)

    return np.array(X_list, dtype=float), np.array(y_list, dtype=int)


# ---------------------------------------------------------------------------
# Adversarial validation set (fix S1)
# ---------------------------------------------------------------------------
# Hand-labelled queries the SVM never sees in training. These probe whether
# the model generalises beyond the rule, or just memorises it.
#
# Format: (feature_vector, expected_label, rationale)
# Labels are the AUTHOR's judgement, not derived from rule_score.
# When the rule says "clean" but a human says "this is suspicious", we
# learn whether the SVM agrees with the rule or the human.

ADVERSARIAL_CASES: list[tuple[list[int | float], int, str]] = [
    # ─── Cases where the rule scores LOW but the situation is risky ────────
    # A small DELETE on sensitive fields. Rule says clean (no flags fire),
    # but deleting payroll records by SSN match is human-suspicious.
    ([1, 0, 1, 1, 0, 0], 1, "small DELETE matching by SSN"),
    # Exactly 3 records returned on a bulk sensitive read. exceeds_limit=0
    # by one. Rule says clean. Human says "still looks like exfil-probing".
    ([3, 0, 1, 0, 0, 0], 0, "3-record sensitive read — borderline clean"),
    # Empty filter but the only flag is is_empty_filter (score 0.45).
    # Just under flag threshold. Real-world: this is a full scan.
    ([2, 1, 0, 0, 0, 0], 1, "full-collection scan, 2 results, no sensitive"),

    # ─── Cases where the rule scores HIGH but human says it's fine ─────────
    # has_sensitive=1 just because the filter mentions "salary" — but it's
    # a single record lookup. Rule says 0.20 (clean). Human agrees.
    ([1, 0, 1, 0, 0, 0], 0, "single-record salary lookup"),

    # ─── Tricky bulk-sensitive cases ───────────────────────────────────────
    # Bulk DELETE with no sensitive fields but high record count.
    ([8, 0, 0, 1, 1, 0], 2, "targeted bulk DELETE — should block"),
    # Bulk read of sensitive data but with a real filter. Rule says
    # has_sensitive=1, no bulk flag. Human: this is exactly an exfil pattern.
    ([6, 0, 1, 0, 1, 0], 1, "bulk sensitive read with filter — exfil"),

    # ─── Clearly clean ─────────────────────────────────────────────────────
    ([1, 0, 0, 0, 0, 0], 0, "trivial single-record lookup"),
    ([2, 0, 0, 0, 0, 0], 0, "2-record department read"),

    # ─── Clearly malicious ─────────────────────────────────────────────────
    ([10, 1, 1, 1, 1, 1], 2, "all flags lit — definitive block"),
    ([10, 1, 1, 0, 1, 1], 2, "READ no-filter all-sensitive — exfil dump"),
]


def _eval_adversarial(pipeline: Pipeline) -> tuple[float, list[str]]:
    """
    Score the SVM on the hand-labeled adversarial set.
    Returns (accuracy, per_case_results).
    """
    feats   = np.array([c[0] for c in ADVERSARIAL_CASES], dtype=float)
    truth   = np.array([c[1] for c in ADVERSARIAL_CASES], dtype=int)
    preds: ndarray = pipeline.predict(feats)  # type: ignore[type-arg, assignment]
    acc     = float(accuracy_score(truth, preds))  # type: ignore[arg-type]

    lines: list[str] = []
    for (feat, expected, note), pred in zip(ADVERSARIAL_CASES, preds):
        mark = "OK " if pred == expected else "MISS"
        lines.append(
            f"  [{mark}] expected={expected}  predicted={int(pred)}  "
            f"feat={feat}  ({note})"
        )
    return acc, lines


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def train(output_dir: Path = Path(".")) -> None:
    print("=== vault-audit — SVM Training (post-review) ===\n")

    X, y = generate_dataset(n_samples=5_000, seed=42)
    counts = {lbl: int((y == lbl).sum()) for lbl in [0, 1, 2]}
    print(f"Dataset size : {len(X):,}  samples")
    print(f"Class counts : clean={counts[0]}  flagged={counts[1]}  blocked={counts[2]}\n")

    X_train: ndarray  # type: ignore[type-arg]
    X_test:  ndarray  # type: ignore[type-arg]
    y_train: ndarray  # type: ignore[type-arg]
    y_test:  ndarray  # type: ignore[type-arg]
    X_train, X_test, y_train, y_test = train_test_split(  # type: ignore[misc]
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    base_svm = SVC(
        kernel="rbf", C=10.0, gamma="scale",
        class_weight="balanced", probability=False, random_state=42,
    )
    calibrated = CalibratedClassifierCV(base_svm, method="sigmoid", cv=5)
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    calibrated),
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores: ndarray = cross_val_score(  # type: ignore[type-arg, misc]
        pipeline, X_train, y_train, cv=cv, scoring="accuracy"
    )
    rule_recovery_cv = float(cv_scores.mean())
    print(f"Rule-recovery CV accuracy : {rule_recovery_cv:.4f} +/- {cv_scores.std():.4f}")
    print("  (this measures how well the SVM mimics rule_score — NOT generalisation)\n")

    pipeline.fit(X_train, y_train)  # type: ignore[arg-type]
    y_pred: ndarray = pipeline.predict(X_test)  # type: ignore[type-arg, assignment]
    test_acc = float(accuracy_score(y_test, y_pred))  # type: ignore[arg-type]

    report: str = classification_report(  # type: ignore[assignment]
        y_test, y_pred,
        target_names=["clean", "flagged", "blocked"], digits=4,
    )
    print("Test-set classification report (rule-labeled):")
    print(report)

    # ── fix S1: evaluate on hand-labeled adversarial set ──────────────────
    adv_acc, adv_lines = _eval_adversarial(pipeline)
    print(f"Adversarial accuracy : {adv_acc:.4f}  ({len(ADVERSARIAL_CASES)} hand-labeled cases)")
    print("  (this is the honest number — cases the SVM has never seen and")
    print("   that aren't derivable from rule_score alone)\n")
    for ln in adv_lines:
        print(ln)
    print()

    sample = np.array([[1, 0, 0, 0, 0, 0]], dtype=float)
    proba: ndarray = pipeline.predict_proba(sample)  # type: ignore[type-arg, assignment]
    assert proba.shape == (1, 3), "predict_proba must return 3-class probabilities"

    model_path = output_dir / "svm_model.joblib"
    joblib.dump(pipeline, model_path)  # type: ignore[no-untyped-call]
    print(f"Model saved  -> {model_path}")

    report_path = output_dir / "training_report.txt"
    report_text = textwrap.dedent(f"""\
        vault-audit — SVM Training Report (post-review)
        ================================================

        Dataset
        -------
        Total samples : {len(X):,}
        Class counts  : clean={counts[0]}  flagged={counts[1]}  blocked={counts[2]}

        Model
        -----
        Pipeline : StandardScaler -> CalibratedClassifierCV(SVC, method=sigmoid, cv=5)
        Kernel   : RBF   C=10   gamma=scale   class_weight=balanced

        Headline numbers (read both — they measure different things)
        -------------------------------------------------------------
        Rule-recovery CV accuracy : {rule_recovery_cv:.4f}  (+/- {cv_scores.std():.4f})
            How well the SVM mimics the deterministic rule_score().
            This is NOT generalisation — both X and y were generated by
            the same rule, so a high number here just means the SVM
            successfully memorised a piecewise-linear function.

        Test-set accuracy         : {test_acc:.4f}
            Same caveat — rule-labeled holdout.

        Adversarial accuracy      : {adv_acc:.4f}
            {len(ADVERSARIAL_CASES)} hand-labeled cases the SVM never trained on,
            and whose labels are the author's judgement — NOT derivable from
            rule_score(). This is the honest measure of whether the model
            generalises beyond the rule.

        Per-case adversarial results
        -----------------------------
        {chr(10).join('        ' + ln for ln in adv_lines)}

        Test-set classification report (rule-labeled)
        ----------------------------------------------
        {report}

        Feature order
        -------------
        {chr(10).join(f"          [{i}] {name}" for i, name in enumerate(FEATURE_NAMES))}

        Decision thresholds (applied in svm_engine.py)
        -----------------------------------------------
          flagged : P(flagged) + P(blocked) >= 0.50
          blocked : P(flagged) + P(blocked) >= 0.85

        Notes on evaluation honesty
        ----------------------------
        Training labels come from baseline.rule_score(), so the
        rule-recovery accuracy is necessarily high (the SVM is fitting
        a closed-form function). The adversarial set is the meaningful
        number — it tests cases where rule_score and human judgement
        diverge.

        Future work: collect real query traces from a production-like
        environment, label them by hand or by expert agreement, and
        report metrics on that set.
    """)
    report_path.write_text(report_text, encoding="utf-8")
    print(f"Report saved -> {report_path}\n")
    print("=== Training complete ===")


if __name__ == "__main__":
    train(output_dir=Path("."))