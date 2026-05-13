"""
verify_chain.py — Vault-Audit Phase 4
======================================
Standalone CLI verifier for the chained audit log.

Walks audit_logs in seq order and recomputes each entry's SHA-256, checking
that prev_hash links match and no entries are missing. Exits 0 on a clean
chain, 1 on any tamper / gap.

Usage:
    python verify_chain.py
    python verify_chain.py --verbose      # print every entry checked
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

from pymongo import MongoClient
from pymongo.collection import Collection

from middleware import verify_chain


GREEN = "\033[92m"
RED   = "\033[91m"
BOLD  = "\033[1m"
DIM   = "\033[2m"
RESET = "\033[0m"


def _print_tail(verbose: bool, limit: int = 5) -> None:
    """Print the last few entries so the user can eyeball the chain links."""
    if not verbose:
        return
    client: MongoClient[dict[str, Any]] = MongoClient("mongodb://localhost:27017/")
    audit_logs: Collection[dict[str, Any]] = client["vault_audit_db"]["audit_logs"]
    entries = list(
        audit_logs.find({}, {"_id": 0, "seq": 1, "user_id": 1, "query_type": 1,
                             "prev_hash": 1, "integrity_hash": 1})
                  .sort("seq", -1).limit(limit)
    )
    if not entries:
        return
    print(f"\n{DIM}Last {len(entries)} entries (newest first):{RESET}")
    for e in reversed(entries):
        seq      = e.get("seq", "?")
        user     = e.get("user_id", "?")
        op       = e.get("query_type", "?")
        prev     = str(e.get("prev_hash", ""))[:10]
        digest   = str(e.get("integrity_hash", ""))[:10]
        print(f"  seq={seq:<4} user={user:<10} op={op:<6}  prev={prev}…  hash={digest}…")


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the Vault-Audit audit-log chain.")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print the last few entries after verification.")
    args = parser.parse_args()

    print(f"{BOLD}Vault-Audit — chain verification{RESET}")
    print("─" * 50)

    result = verify_chain()

    if result["valid"]:
        print(f"{GREEN}✅  CHAIN VALID{RESET}")
        print(f"    entries checked: {result['total']}")
        _print_tail(args.verbose)
        return 0

    print(f"{RED}❌  CHAIN BROKEN{RESET}")
    print(f"    entries checked: {result['total']}")
    print(f"    first break at seq: {result['first_break_seq']}")
    print(f"    reason: {result['reason']}")
    _print_tail(args.verbose)
    return 1


if __name__ == "__main__":
    sys.exit(main())