#!/usr/bin/env python3
"""
Compare the decision matrices of two tournament runs (e.g., the same
strategy pool adjudicated by different oracle models).

Reports cell-level agreement over all ordered (dictator, recipient) pairs,
lists every disagreeing cell, and shows the leaderboard each oracle's
matrix induces.

Usage:
  python -m os_fdt.compare_decisions runs/tournament_9 runs/tournament_9_sol
"""
import argparse
import math
from pathlib import Path

from .decision_matrix import load_decision_matrix


def leaderboard(names, decisions, E: float = 60.0):
    """Score each strategy from a decision matrix (log utility, both roles)."""
    g, full = math.log(1 + E / 2), math.log(1 + E)
    scores = {n: 0.0 for n in names}
    for (d, r), dec in decisions.items():
        if dec == "SHARE":
            scores[d] += g
            scores[r] += g
        else:
            scores[d] += full
    return sorted(scores.items(), key=lambda t: -t[1])


def main():
    ap = argparse.ArgumentParser(
        description="Compare decision matrices from two tournament runs.")
    ap.add_argument("run_a", help="First run directory (rounds.jsonl inside).")
    ap.add_argument("run_b", help="Second run directory.")
    args = ap.parse_args()

    names_a, dec_a = load_decision_matrix(Path(args.run_a) / "rounds.jsonl")
    names_b, dec_b = load_decision_matrix(Path(args.run_b) / "rounds.jsonl")
    names = sorted(set(names_a) & set(names_b))
    if names_a != names_b:
        only = set(names_a) ^ set(names_b)
        print(f"Note: pools differ (only in one run: {sorted(only)}); "
              f"comparing the {len(names)}-strategy intersection.\n")

    common = sorted(c for c in set(dec_a) & set(dec_b)
                    if c[0] in names and c[1] in names)
    agree = [c for c in common if dec_a[c] == dec_b[c]]
    disagree = [c for c in common if dec_a[c] != dec_b[c]]

    print(f"Cells compared: {len(common)}")
    print(f"Agreement: {len(agree)}/{len(common)} "
          f"({100 * len(agree) / len(common):.1f}%)\n")
    if disagree:
        print(f"{'Dictator':<25} {'Recipient':<25} "
              f"{Path(args.run_a).name:<18} {Path(args.run_b).name}")
        print("-" * 88)
        for d, r in disagree:
            print(f"{d:<25} {r:<25} {dec_a[(d, r)]:<18} {dec_b[(d, r)]}")
        print()

    for label, dec in ((args.run_a, dec_a), (args.run_b, dec_b)):
        restricted = {c: d for c, d in dec.items()
                      if c[0] in names and c[1] in names}
        print(f"Leaderboard induced by {label} (intersection pool):")
        for rank, (n, s) in enumerate(leaderboard(names, restricted), 1):
            print(f"  {rank}. {n:<25} {s:.3f}")
        print()


if __name__ == "__main__":
    main()
