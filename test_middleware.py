"""
test_middleware.py — Vault-Audit Phase 2
=========================================
Fires a battery of test queries against the running API and prints
a colour-coded report.

Start the server first:
    python api.py

Then in a second terminal:
    python test_middleware.py
"""

from __future__ import annotations

import sys
from typing import Any, TypedDict

import requests
from pymongo import MongoClient
from pymongo.collection import Collection

BASE = "http://127.0.0.1:8000"

GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
RESET  = "\033[0m"
BOLD   = "\033[1m"


def colour(text: str, score: float, blocked: bool) -> str:
    if blocked:
        return f"{RED}{text}{RESET}"
    if score >= 0.50:
        return f"{YELLOW}{text}{RESET}"
    return f"{GREEN}{text}{RESET}"


# ─── Typed test case schema ───────────────────────────────────────────────────

class TestCase(TypedDict):
    label:          str
    expect_blocked: bool
    body:           dict[str, Any]


TEST_CASES: list[TestCase] = [
    # ── Normal (clean, score = 0.00) ─────────────────────────────────────────
    {
        "label":          "Single employee lookup",
        "expect_blocked": False,
        "body": {
            "query_type":   "READ",
            "query_filter": {"employee_id": "EMP-1001"},
            "user_id":      "alice",
        },
    },
    {
        "label":          "SSN lookup — one record",
        "expect_blocked": False,
        "body": {
            "query_type":   "READ",
            "query_filter": {"ssn": "000-00-0000"},
            "user_id":      "bob",
        },
    },
    {
        "label":          "Single UPDATE by employee_id",
        "expect_blocked": False,
        "body": {
            "query_type":     "UPDATE",
            "query_filter":   {"employee_id": "EMP-1002"},
            "user_id":        "alice",
            "update_payload": {"role": "Senior Engineer"},
        },
    },
    # ── Flagged (score ≥ 0.50, not blocked) ──────────────────────────────────
    {
        "label": "Full collection scan — blocked by SVM",
        "expect_blocked": True,
        "body": {
            "query_type":   "READ",
            "query_filter": {},
            "user_id":      "eve",
        },
    },
    {
        "label":          "Salary dump — large result, flagged",
        "expect_blocked": False,
        "body": {
            "query_type":   "READ",
            "query_filter": {"salary": {"$gt": 0}},
            "user_id":      "eve",
        },
    },
    # ── Blocked (score ≥ 0.85 → HTTP 403) ────────────────────────────────────
    {
        "label":          "Bulk DELETE — no filter  [BLOCKED]",
        "expect_blocked": True,
        "body": {
            "query_type":   "DELETE",
            "query_filter": {},
            "user_id":      "attacker",
        },
    },
    {
        "label":          "DELETE by dept — flagged, not blocked",
        "expect_blocked": False,
        "body": {
            "query_type":   "DELETE",
            "query_filter": {"department": "Finance"},
            "user_id":      "mallory",
        },
    },
]


def run_tests() -> None:
    # ── Health check ──────────────────────────────────────────────────────────
    try:
        r = requests.get(f"{BASE}/health", timeout=3)
        r.raise_for_status()
        print(f"{BOLD}✅  API is up — {r.json()}{RESET}\n")
    except Exception as exc:
        print(f"{RED}❌  Cannot reach API at {BASE}: {exc}{RESET}")
        print("    Start the server first:  python api.py")
        sys.exit(1)

    print(f"{'─'*76}")
    print(f"{'LABEL':<42} {'SCORE':>6}  {'FLAGGED':>8}  {'STATUS':>10}  {'PASS':>4}")
    print(f"{'─'*76}")

    passed = failed = 0

    for tc in TEST_CASES:
        try:
            resp = requests.post(f"{BASE}/query", json=tc["body"], timeout=5)
        except requests.exceptions.ConnectionError as exc:
            print(f"{RED}  CONNECTION ERROR: {exc}{RESET}")
            failed += 1
            continue

        body    = resp.json()
        score   = float(body.get("threat_score", 0.0))
        flagged = bool(body.get("flagged", False))
        blocked = resp.status_code == 403

        ok  = blocked == tc["expect_blocked"]
        sym = "✅" if ok else "❌"
        if ok:
            passed += 1
        else:
            failed += 1

        flag_str   = "⚠️  yes" if flagged else "no"
        status_str = "🚫 blocked" if blocked else "ok"
        row = (
            f"{tc['label']:<42} {score:>6.2f}  {flag_str:>8}  "
            f"{status_str:>10}  {sym}"
        )
        print(colour(row, score, blocked))

    print(f"{'─'*76}")
    result_line = f"{BOLD}Results: {GREEN}{passed} passed{RESET}{BOLD} / "
    result_line += f"{RED}{failed} failed{RESET}" if failed else f"0 failed{RESET}"
    print(f"\n{result_line}\n")

    # ── Audit stats ───────────────────────────────────────────────────────────
    stats: dict[str, Any] = requests.get(f"{BASE}/audit/stats", timeout=5).json()
    print(f"{BOLD}Audit stats:{RESET}")
    for k, v in stats.items():
        print(f"  {k:<22}: {v}")

    # ── Last flagged entries ──────────────────────────────────────────────────
    flagged_logs: list[dict[str, Any]] = requests.get(
        f"{BASE}/audit/logs",
        params={"limit": 5, "flagged_only": "true"},
        timeout=5,
    ).json()
    print(f"\n{BOLD}Last {len(flagged_logs)} flagged log(s):{RESET}")
    for entry in flagged_logs:
        print(
            f"  [{entry['timestamp'][:19]}]  "
            f"user={entry['user_id']:<12} "
            f"op={entry['query_type']:<7} "
            f"score={entry['threat_score']:.2f}  "
            f"hash={entry['integrity_hash'][:12]}…"
        )

    # ─── Phase 4: chain verification ──────────────────────────────────────────
    print(f"\n{BOLD}Phase 4 — audit chain verification{RESET}")
    print("─" * 76)
    run_chain_tests()


# ─── Phase 4: tamper-detection tests ──────────────────────────────────────────

def _verify() -> dict[str, Any]:
    """Hit GET /audit/verify and return the parsed body."""
    return requests.get(f"{BASE}/audit/verify", timeout=5).json()


def _pass(label: str, ok: bool, detail: str = "") -> tuple[str, bool]:
    sym    = "✅" if ok else "❌"
    colour = GREEN if ok else RED
    line   = f"  {sym} {label}"
    if detail:
        line += f"  — {detail}"
    print(f"{colour}{line}{RESET}")
    return label, ok


def run_chain_tests() -> None:
    """
    Three checks:
      1. Verifier reports clean on the chain produced by the queries above.
      2. Tampering with one entry's threat_score is detected (direct Mongo write).
      3. Deleting an entry from the middle is detected.

    The tamper/delete steps go through a direct MongoClient by design — going
    through the API would just create more legitimate log entries and defeat
    the test.
    """
    client: MongoClient[dict[str, Any]] = MongoClient("mongodb://localhost:27017/")
    audit_logs: Collection[dict[str, Any]] = client["vault_audit_db"]["audit_logs"]

    results: list[tuple[str, bool]] = []

    # ── 1. Clean chain ────────────────────────────────────────────────────────
    r = _verify()
    ok = bool(r.get("valid")) and r.get("first_break_seq") is None
    results.append(_pass(
        "Clean chain reports valid",
        ok,
        f"total={r.get('total')} valid={r.get('valid')}",
    ))
    if not ok:
        print(f"    {RED}stopping further chain tests — base chain is already broken{RESET}")
        print(f"    reason: {r.get('reason')}")
        return

    # ── 2. Tamper one entry's threat_score ────────────────────────────────────
    # Pick a middle-ish entry so verification has to walk past clean entries first.
    target = audit_logs.find_one(
        {"seq": {"$exists": True}},
        sort=[("seq", 1)],
        skip=1,  # not the genesis entry
    )
    if target is None:
        results.append(_pass("Tamper test", False, "no entry to tamper with"))
    else:
        original_score = float(target["threat_score"])
        tampered_score = 0.0 if original_score > 0.5 else 0.99
        audit_logs.update_one(
            {"_id": target["_id"]},
            {"$set": {"threat_score": tampered_score}},
        )
        r = _verify()
        detected = (not r.get("valid")) and r.get("first_break_seq") == target["seq"]
        results.append(_pass(
            "Tampered threat_score is detected",
            detected,
            f"break at seq={r.get('first_break_seq')}  reason={r.get('reason')}",
        ))
        # Restore so the next test starts from a known state.
        audit_logs.update_one(
            {"_id": target["_id"]},
            {"$set": {"threat_score": original_score}},
        )
        r = _verify()
        results.append(_pass(
            "Chain valid again after restore",
            bool(r.get("valid")),
            f"valid={r.get('valid')}",
        ))

    # ── 3. Delete an entry from the middle ────────────────────────────────────
    victim = audit_logs.find_one({}, sort=[("seq", 1)], skip=1)
    if victim is None:
        results.append(_pass("Deletion test", False, "no entry to delete"))
    else:
        snapshot = dict(victim)
        audit_logs.delete_one({"_id": victim["_id"]})
        r = _verify()
        detected = (not r.get("valid")) and r.get("first_break_seq") == victim["seq"]
        results.append(_pass(
            "Deleted entry is detected",
            detected,
            f"break at seq={r.get('first_break_seq')}  reason={r.get('reason')}",
        ))
        # Restore so the DB is left as found.
        audit_logs.insert_one(snapshot)
        r = _verify()
        results.append(_pass(
            "Chain valid again after restore",
            bool(r.get("valid")),
            f"valid={r.get('valid')}",
        ))

    passed = sum(1 for _, ok in results if ok)
    failed = len(results) - passed
    print(f"\n  {BOLD}Phase 4: {GREEN}{passed} passed{RESET}{BOLD}, "
          f"{RED if failed else ''}{failed} failed{RESET}")


if __name__ == "__main__":
    run_tests()