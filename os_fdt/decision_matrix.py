#!/usr/bin/env python3
"""
Print the matrix of dictator decisions for a tournament run.

Rows are dictator strategies, columns are recipient strategies.
Each cell shows S (SHARE) or T (TAKE).

Usage:
  python -m os_fdt.decision_matrix --run runs/2026-02-16_12-13-08
"""
import argparse
import json
from pathlib import Path


def load_decision_matrix(rounds_path: Path) -> tuple[list[str], dict[tuple[str, str], str]]:
    """Load rounds.jsonl and return (sorted strategy names, {(dictator, recipient): decision})."""
    strategy_names = set()
    decisions: dict[tuple[str, str], str] = {}

    with open(rounds_path) as f:
        for line in f:
            rd = json.loads(line)
            d, r = rd["dictator"], rd["recipient"]
            strategy_names.add(d)
            strategy_names.add(r)
            decisions[(d, r)] = rd["decision"]

    return sorted(strategy_names), decisions


def print_decision_matrix(names: list[str], decisions: dict[tuple[str, str], str]):
    """Print a compact decision matrix to stdout."""
    # Abbreviate: SHARE -> S, TAKE -> T
    def abbrev(decision: str) -> str:
        if decision == "SHARE":
            return "S"
        elif decision == "TAKE":
            return "T"
        return "?"

    name_w = max(len(n) for n in names)

    # Print column legend
    for i, n in enumerate(names):
        print(f"  {i}: {n}")
    print()

    # Header row with column indices
    col_w = max(3, len(str(len(names))) + 1)
    print(f"{'':>{name_w}}  " + "".join(f"{i:>{col_w}}" for i in range(len(names))))
    print("-" * (name_w + 2 + col_w * len(names)))

    for d_name in names:
        row = []
        for r_name in names:
            decision = decisions.get((d_name, r_name))
            row.append(abbrev(decision) if decision else ".")
        print(f"{d_name:>{name_w}}  " + "".join(f"{c:>{col_w}}" for c in row))


def main():
    ap = argparse.ArgumentParser(description="Print decision matrix for a tournament run.")
    ap.add_argument("--run", required=True, help="Directory with tournament run output (rounds.jsonl).")
    args = ap.parse_args()

    rounds_path = Path(args.run) / "rounds.jsonl"
    if not rounds_path.exists():
        print(f"Error: {rounds_path} not found")
        return 1

    names, decisions = load_decision_matrix(rounds_path)

    print(f"\nDecision matrix ({len(names)} strategies, S=SHARE T=TAKE)\n")
    print_decision_matrix(names, decisions)
    print()

    # Summary counts
    share_count = sum(1 for v in decisions.values() if v == "SHARE")
    take_count = sum(1 for v in decisions.values() if v == "TAKE")
    total = len(decisions)
    print(f"SHARE: {share_count}/{total} ({100*share_count/total:.1f}%)")
    print(f"TAKE:  {take_count}/{total} ({100*take_count/total:.1f}%)")

    return 0


if __name__ == "__main__":
    exit(main())
