#!/usr/bin/env python3
"""
Open-Strategy Dictator Game — Author-Type Mixture Equilibria

Models a population of strategy authors who differ in how much they value
their own reward vs. the total reward of the tournament:

  - "own" authors     (lambda = 0):  maximize their strategy's own score
  - "mixed" authors   (lambda = m):  maximize (1-m)*own + m*total
  - "altruist" authors (lambda = 1): maximize total tournament reward

Each author type exponentially tilts toward strategies that perform well
for its objective against the common field distribution f (a random-utility
/ logit choice model with inverse temperature beta). The field is the
type-mass-weighted mixture of the per-type choice distributions, and we
solve for the rational-expectations fixed point:

  f = sum_types mass_t * softmax(beta * U_t(f)),
  U_t(f) = (1 - lambda_t) * A f + lambda_t * W f

where A[i,j] is strategy i's own payoff across both rounds against j and
W[i,j] is the total welfare of both rounds between i and j.

Because total reward is a property of the whole field, altruist authors
choosing as a correlated bloc should maximize V(f) = 0.5 * f' W f over
their own component, internalizing rounds they are not part of. The bloc
scan pins the altruist mass on each candidate strategy, lets the remaining
types re-equilibrate, and ranks candidates by resulting total welfare.

Usage:
  python -m os_fdt.author_types --run runs/tournament_9
  python -m os_fdt.author_types --run runs/tournament_9 \
      --frac-own 0.4 --frac-mixed 0.4 --frac-altruist 0.2 --beta 5 20
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def u(x: float) -> float:
    """Utility function u(x) = ln(1 + x)."""
    if x < -1:
        x = -1 + 1e-12
    return math.log1p(x)


def load_allocation_matrix(run_path: Path) -> Tuple[List[str], np.ndarray, float]:
    """
    Load rounds.jsonl and return (names, v, E) where v[i, j] is the average
    amount dictator i awards to itself against recipient j.
    """
    rounds_path = run_path / "rounds.jsonl"
    rounds_data = []
    strategy_names_set = set()
    E = None

    with open(rounds_path, "r") as f:
        for line in f:
            round_data = json.loads(line)
            rounds_data.append(round_data)
            strategy_names_set.add(round_data["dictator"])
            strategy_names_set.add(round_data["recipient"])
            if E is None:
                E = sum(round_data["allocation"].values())

    if E is None:
        raise ValueError(f"No rounds found in {rounds_path}")

    names = sorted(strategy_names_set)
    name_to_idx = {name: i for i, name in enumerate(names)}
    n = len(names)

    v_sum = np.zeros((n, n))
    v_count = np.zeros((n, n), dtype=int)
    for round_data in rounds_data:
        i = name_to_idx[round_data["dictator"]]
        j = name_to_idx[round_data["recipient"]]
        v_sum[i, j] += round_data["allocation"]["ME"]
        v_count[i, j] += 1

    v = np.where(v_count > 0, v_sum / np.maximum(v_count, 1), E / 2.0)
    return names, v, float(E)


def build_matrices(v: np.ndarray, E: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build (A, W) from the dictator self-allocation matrix v.

    A[i, j]: strategy i's own payoff over both rounds against j:
             u(v[i,j]) + u(E - v[j,i])
    W[i, j]: total welfare of both rounds between i and j:
             u(v[i,j]) + u(E - v[i,j]) + u(v[j,i]) + u(E - v[j,i])
    """
    n = v.shape[0]
    A = np.zeros((n, n))
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            A[i, j] = u(v[i, j]) + u(E - v[j, i])
            W[i, j] = (u(v[i, j]) + u(E - v[i, j])
                       + u(v[j, i]) + u(E - v[j, i]))
    return A, W


def type_utilities(A: np.ndarray, W: np.ndarray, f: np.ndarray, lam: float) -> np.ndarray:
    """U_lambda(f) = (1 - lambda) * A f + lambda * W f."""
    return (1.0 - lam) * (A @ f) + lam * (W @ f)


def mixture_equilibrium(A: np.ndarray, W: np.ndarray,
                        types: List[Tuple[float, float]], beta: float,
                        f0: np.ndarray,
                        pinned: Optional[Tuple[int, float]] = None,
                        max_iter: int = 20000, tol: float = 1e-13,
                        history: Optional[List[np.ndarray]] = None,
                        damping: float = 1.0
                        ) -> Tuple[np.ndarray, Dict[float, np.ndarray]]:
    """
    Fixed point of f = pinned_mass + sum_t mass_t * softmax(beta * U_t(f)).

    types: list of (lambda, mass) for the free (softmax-responding) types.
    pinned: optional (strategy_index, mass) held fixed (the altruist bloc).
    history: optional list that collects every iterate (f0 first, fixed
        point last) for trajectory plots.
    damping: step size in (0, 1]; 1.0 is plain fixed-point iteration,
        smaller values blend f <- (1-damping)*f + damping*T(f) and converge
        where plain iteration cycles.  Fixed points are unaffected.

    Returns (f, per_type_choice) where per_type_choice maps lambda to that
    type's choice distribution at the fixed point.
    """
    n = A.shape[0]
    f = f0.copy()
    if history is not None:
        history.append(f.copy())
    comps: Dict[float, np.ndarray] = {}
    for _ in range(max_iter):
        fn = np.zeros(n)
        if pinned is not None:
            fn[pinned[0]] = pinned[1]
        for lam, mass in types:
            e = beta * type_utilities(A, W, f, lam)
            e -= e.max()
            q = np.exp(e)
            q /= q.sum()
            comps[lam] = q
            fn += mass * q
        if damping < 1.0:
            fn = (1.0 - damping) * f + damping * fn
        if history is not None:
            history.append(fn.copy())
        if np.max(np.abs(fn - f)) < tol:
            return fn, comps
        f = fn
    return f, comps


def sample_equilibria(A: np.ndarray, W: np.ndarray,
                      types: List[Tuple[float, float]], beta: float,
                      num_samples: int,
                      pinned: Optional[Tuple[int, float]] = None,
                      cluster_tol: float = 1e-4,
                      rng: Optional[np.random.Generator] = None,
                      damping: float = 1.0,
                      max_iter: int = 20000
                      ) -> List[Tuple[np.ndarray, Dict[float, np.ndarray], int]]:
    """
    Multi-start the fixed-point iteration and cluster distinct equilibria.
    Returns [(f, per_type_choice, basin_count)] sorted by basin size.
    """
    n = A.shape[0]
    if rng is None:
        rng = np.random.default_rng()
    free_mass = sum(mass for _, mass in types)
    clusters: List[Tuple[np.ndarray, Dict[float, np.ndarray], int]] = []
    for _ in range(num_samples):
        f0 = rng.dirichlet(np.ones(n)) * free_mass
        if pinned is not None:
            f0[pinned[0]] += pinned[1]
        f, comps = mixture_equilibrium(A, W, types, beta, f0, pinned=pinned,
                                       damping=damping, max_iter=max_iter)
        for idx, (centroid, _, count) in enumerate(clusters):
            if np.max(np.abs(f - centroid)) < cluster_tol:
                clusters[idx] = (centroid, clusters[idx][1], count + 1)
                break
        else:
            clusters.append((f, comps, 1))
    clusters.sort(key=lambda c: -c[2])
    return clusters


def total_welfare(W: np.ndarray, f: np.ndarray) -> float:
    """Per-capita total tournament welfare V(f) = 0.5 * f' W f."""
    return 0.5 * float(f @ (W @ f))


def format_dist(names: List[str], f: np.ndarray, min_mass: float = 0.005) -> str:
    order = np.argsort(-f)
    return ", ".join(f"{names[i]}:{f[i]:.3f}" for i in order if f[i] > min_mass)


def main():
    ap = argparse.ArgumentParser(
        description="Author-type mixture equilibria for the Open-Strategy Dictator Game.")
    ap.add_argument("--run", required=True,
                    help="Directory with tournament run output (rounds.jsonl).")
    ap.add_argument("--frac-own", type=float, default=0.4,
                    help="Fraction of authors maximizing own reward (default: 0.4).")
    ap.add_argument("--frac-mixed", type=float, default=0.4,
                    help="Fraction of authors with a mixed objective (default: 0.4).")
    ap.add_argument("--frac-altruist", type=float, default=0.2,
                    help="Fraction of authors maximizing total reward (default: 0.2).")
    ap.add_argument("--mixed-lambda", type=float, default=0.5,
                    help="Weight on total reward for the mixed type (default: 0.5).")
    ap.add_argument("--beta", type=float, nargs="+", default=[5.0, 20.0],
                    help="Inverse temperature(s) for the logit choice model.")
    ap.add_argument("--samples", type=int, default=300,
                    help="Random starts per equilibrium computation (default: 300).")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed.")
    args = ap.parse_args()

    fracs = np.array([args.frac_own, args.frac_mixed, args.frac_altruist])
    if fracs.min() < 0 or fracs.sum() <= 0:
        raise ValueError("Type fractions must be non-negative with positive sum.")
    fracs = fracs / fracs.sum()
    frac_own, frac_mixed, frac_altruist = fracs
    lambdas = [0.0, args.mixed_lambda, 1.0]

    names, v, E = load_allocation_matrix(Path(args.run))
    n = len(names)
    A, W = build_matrices(v, E)
    rng = np.random.default_rng(args.seed)

    delta = u(E) - u(E / 2)
    g = u(E / 2)
    print(f"=== Author types over {n} strategies (E={E:g}) ===")
    print(f"own: {frac_own:.2f}  mixed(lambda={args.mixed_lambda:g}): {frac_mixed:.2f}  "
          f"altruist: {frac_altruist:.2f}")
    print(f"share cost delta={delta:.4f}, share gain g={g:.4f}, "
          f"direct altruism threshold lambda* = delta/g = {delta / g:.4f}")

    all_types = [(lam, mass) for lam, mass in zip(lambdas, fracs) if mass > 0]

    for beta in args.beta:
        print(f"\n=== Mixture equilibrium (beta={beta:g}, {args.samples} starts) ===")
        clusters = sample_equilibria(A, W, all_types, beta, args.samples, rng=rng)
        for eq_idx, (f, comps, count) in enumerate(clusters):
            pct = 100 * count / args.samples
            print(f"#{eq_idx + 1}  basin: {count}/{args.samples} ({pct:.1f}%)  "
                  f"V={total_welfare(W, f):.4f}")
            print(f"    field: {format_dist(names, f)}")
            for lam, mass in all_types:
                print(f"    type lambda={lam:g} (mass {mass:.2f}): "
                      f"{format_dist(names, comps[lam], min_mass=0.01)}")

    if frac_altruist <= 0:
        print("\n(no altruist mass; skipping bloc scan)")
        return

    free_types = [(lam, mass) for lam, mass in zip(lambdas[:2], fracs[:2]) if mass > 0]
    for beta in args.beta:
        print(f"\n=== Altruist bloc scan (mass {frac_altruist:.2f} pinned per strategy, "
              f"beta={beta:g}) ===")
        print("Free types re-equilibrate around the pinned bloc; candidates ranked "
              "by total welfare V.")
        rows = []
        for k in range(n):
            clusters = sample_equilibria(A, W, free_types, beta, args.samples,
                                         pinned=(k, frac_altruist), rng=rng)
            f = clusters[0][0]
            V = total_welfare(W, f)
            bloc_own = float((A @ f)[k])
            rows.append((V, bloc_own, k, f, len(clusters), clusters[0][2]))
        rows.sort(key=lambda r: -r[0])
        for V, bloc_own, k, f, n_basins, basin_count in rows:
            multi = "" if n_basins == 1 else (
                f"  [{n_basins} basins, main {basin_count}/{args.samples}]")
            print(f"- {names[k]:<25s} V={V:.4f}  bloc_own={bloc_own:.4f}{multi}")
            print(f"    field: {format_dist(names, f, min_mass=0.02)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
