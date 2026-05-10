"""
test_middleware.py — Vault-Audit Phase 2
=========================================
Fires a battery of test queries against the running API and prints
a colour-coded report.

Start the server first:
    python api.py          ← recommended (handles Windows event loop)
  or:
    uvicorn api:app --reload --port 8000 --loop asyncio

Then in a second terminal:
    python test_middleware.py
"""

from __future__ import annotations
import sys
import requests

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


# Each case has an "expect_blocked" key so pass/fail is unambiguous
TEST_CASES: list[dict] = [
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
        "label":          "Full collection scan — flagged",
        "expect_blocked": False,
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

        body   = resp.json()
        score  = float(body.get("threat_score", 0.0))
        flagged = bool(body.get("flagged", False))
        status  = body.get("status", "?")
        blocked = (resp.status_code == 403)

        # Pass condition: expected_blocked matches actual HTTP status
        ok     = (blocked == tc["expect_blocked"])
        sym    = "✅" if ok else "❌"
        if ok:
            passed += 1
        else:
            failed += 1

        score_str  = f"{score:.2f}"
        flag_str   = "⚠️  yes" if flagged else "no"
        status_str = "🚫 blocked" if blocked else "ok"

        row = (
            f"{tc['label']:<42} {score_str:>6}  {flag_str:>8}  "
            f"{status_str:>10}  {sym}"
        )
        print(colour(row, score, blocked))

    print(f"{'─'*76}")
    summary = f"{BOLD}Results: {GREEN}{passed} passed{RESET}{BOLD} / "
    if failed:
        summary += f"{RED}{failed} failed{RESET}"
    else:
        summary += f"0 failed{RESET}"
    print(f"\n{summary}\n")

    # ── Audit stats ───────────────────────────────────────────────────────────
    stats = requests.get(f"{BASE}/audit/stats", timeout=5).json()
    print(f"{BOLD}Audit stats:{RESET}")
    for k, v in stats.items():
        print(f"  {k:<22}: {v}")

    # ── Last 5 flagged entries ────────────────────────────────────────────────
    flagged_logs = requests.get(
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


if __name__ == "__main__":
    run_tests()