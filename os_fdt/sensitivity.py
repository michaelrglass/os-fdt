#!/usr/bin/env python3
"""
Sensitivity analysis: vary the relative reward of SHARE vs TAKE.

Fix TAKE reward at 1.0 and vary SHARE between 0.5 and 0.95.
For each value, build a payoff matrix from the tournament's decision matrix
and compute the softmax equilibrium.

Payoff structure for a round where i is dictator against j:
  SHARE: dictator gets s, recipient gets s
  TAKE:  dictator gets 1.0, recipient gets 0

Total payoff: A[i,j] = score_as_dictator(d_ij) + score_as_recipient(d_ji)

Usage:
  python -m os_fdt.sensitivity --run runs/2026-02-16_14-38-29
"""
import argparse
from pathlib import Path

import numpy as np

from .decision_matrix import load_decision_matrix
from .osdg_equilibria import softmax_equilibria_sample


def build_payoff_matrix(names: list[str],
                        decisions: dict[tuple[str, str], str],
                        share_reward: float,
                        take_reward: float = 1.0) -> np.ndarray:
    """
    Build payoff matrix A where A[i,j] = score_as_dictator(d_ij) + score_as_recipient(d_ji).

    SHARE: dictator gets share_reward, recipient gets share_reward
    TAKE:  dictator gets take_reward, recipient gets 0
    """
    n = len(names)
    A = np.zeros((n, n))

    for i, ni in enumerate(names):
        for j, nj in enumerate(names):
            d_ij = decisions.get((ni, nj))
            d_ji = decisions.get((nj, ni))

            # Score as dictator against j
            s_d = share_reward if d_ij == "SHARE" else take_reward

            # Score as recipient when j is dictator
            s_r = share_reward if d_ji == "SHARE" else 0.0

            A[i, j] = s_d + s_r

    return A


def main():
    ap = argparse.ArgumentParser(
        description="Sensitivity analysis: vary SHARE reward relative to TAKE.")
    ap.add_argument("--run", required=True,
                    help="Directory with tournament run output (rounds.jsonl).")
    ap.add_argument("--beta", type=float, default=20.0,
                    help="Inverse temperature for softmax (default: 20.0).")
    ap.add_argument("--samples", type=int, default=1000,
                    help="Random starting points per configuration (default: 1000).")
    ap.add_argument("--steps", type=int, default=10,
                    help="Number of SHARE reward values to test (default: 10).")
    ap.add_argument("--share-min", type=float, default=0.50,
                    help="Minimum SHARE reward (default: 0.50).")
    ap.add_argument("--share-max", type=float, default=0.95,
                    help="Maximum SHARE reward (default: 0.95).")
    args = ap.parse_args()

    rounds_path = Path(args.run) / "rounds.jsonl"
    if not rounds_path.exists():
        print(f"Error: {rounds_path} not found")
        return 1

    names, decisions = load_decision_matrix(rounds_path)
    n = len(names)

    share_values = np.linspace(args.share_min, args.share_max, args.steps)

    # Abbreviate names for the table
    name_w = max(len(s) for s in names)
    col_w = max(name_w, 8)

    print(f"\nSensitivity: TAKE=1.0, SHARE varies, beta={args.beta}, "
          f"{args.samples} samples\n")

    # Header
    print(f"{'SHARE':>7s}  {'Winner':<{col_w}s}", end="")
    for name in names:
        print(f"  {name:>{col_w}s}", end="")
    print("  basins")
    print("-" * (7 + 2 + col_w + (2 + col_w) * n + 8))

    for s in share_values:
        A = build_payoff_matrix(names, decisions, share_reward=s)
        clusters = softmax_equilibria_sample(A, args.beta, num_samples=args.samples)

        # Use the dominant equilibrium (largest basin)
        f, count = clusters[0]
        winner = names[np.argmax(f)]

        print(f"{s:7.3f}  {winner:<{col_w}s}", end="")
        for i in range(n):
            print(f"  {f[i]:{col_w}.4f}", end="")
        n_basins = len(clusters)
        basin_str = f"{count}/{args.samples}"
        if n_basins > 1:
            basin_str += f" ({n_basins} eq)"
        print(f"  {basin_str}")

    return 0


if __name__ == "__main__":
    exit(main())
