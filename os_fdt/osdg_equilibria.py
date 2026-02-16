#!/usr/bin/env python3
"""
Open-Strategy Dictator Game (OSDG) — Symmetric NE & ESS Finder

Given:
  - A fixed endowment E
  - Pairwise dictator allocations v[i][j] (how much dictator i awards itself vs recipient j)
This script builds a symmetric normal-form game with payoffs:

  g(i, j) = u(v[i][j]) + u(E - v[j][i]),   with u(x) = ln(1 + x)

and computes:
  1) All pure symmetric NE.
  2) Mixed symmetric NE by enumerating supports up to --max-support.
  3) ESS checks:
     - Pure ESS: standard Maynard-Smith condition (exact).
     - Mixed ESS: check mutants supported on best responses to p
       via a small grid search (configurable).

Usage examples:
  python os_fdt/osdg_equilibria.py --run runs/2025-10-18_12-13-04

"""
from __future__ import annotations
import argparse
import itertools as it
import json
import math
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import numpy as np
import pandas as pd


def u(x: float) -> float:
    """Utility function u(x) = ln(1 + x)."""
    if x < -1:
        # Should never happen if inputs are valid; clamp to avoid domain errors.
        x = -1 + 1e-12
    return math.log1p(x)


def build_game_matrix(v: np.ndarray, E: float) -> np.ndarray:
    """
    Build symmetric payoff matrix A with A[i,j] = u(v[i,j]) + u(E - v[j,i]).
    v: (n x n) dictator self-allocation matrix (i->j).
    """
    n = v.shape[0]
    A = np.zeros_like(v, dtype=float)
    for i in range(n):
        for j in range(n):
            A[i, j] = u(float(v[i, j])) + u(float(E - v[j, i]))
    return A


def pure_symmetric_NE(A: np.ndarray, tol: float = 1e-9) -> List[int]:
    """
    Return indices i that are pure symmetric NE:
      A[i,i] >= A[k,i] for all k.
    """
    n = A.shape[0]
    res = []
    for i in range(n):
        col_i = A[:, i]
        if np.all(col_i <= A[i, i] + tol):
            res.append(i)
    return res


def solve_mixed_support(A: np.ndarray, support: List[int], tol: float = 1e-9) -> Optional[np.ndarray]:
    """
    Solve for a symmetric NE with given support S:
      (A p)_i = v* for all i in S,
      (A p)_k <= v* for k not in S,
      sum p = 1, p >= 0, p_i = 0 for i not in S.

    Implementation:
      - Solve equalization on S: A_SS p_S = v* 1_S with sum(p_S)=1.
      - Eliminate v* by subtracting one row: (A_SS - row_0) p_S = 0; sum(p_S)=1.
      - Solve least-squares; then verify inequalities.

    Returns full-length p or None if infeasible.
    """
    n = A.shape[0]
    S = support
    S_idx = {i: k for k, i in enumerate(S)}
    A_SS = A[np.ix_(S, S)]  # |S| x |S|
    m = len(S)
    if m == 1:
        p = np.zeros(n)
        p[S[0]] = 1.0
        # verify
        best = np.max(A[:, S[0]])
        if A[S[0], S[0]] + tol >= best:
            return p
        return None

    # Build linear system:
    # Equal-payoff constraints up to a constant: (A_SS[i] - A_SS[0]) · p_S = 0 for i=1..m-1
    M = (A_SS[1:, :] - A_SS[0:1, :])  # (m-1) x m
    # Add sum-to-1 constraint
    M = np.vstack([M, np.ones((1, m))])
    b = np.zeros(m)
    b[-1] = 1.0

    # Least squares (robust to mild degeneracy); then clip negatives to zero and renormalize.
    p_S, *_ = np.linalg.lstsq(M, b, rcond=None)
    p_S = np.maximum(p_S, 0.0)
    s = p_S.sum()
    if s <= tol:
        return None
    p_S /= s

    # Compute v* and verify best-response conditions
    Ap = A @ embed_support(p_S, S, n)
    v_star = Ap[S].mean()  # equalized within support (up to tol)
    # Verify all in S are ~ equal and max
    if not np.allclose(Ap[S], v_star, atol=1e-6):
        return None
    if np.max(Ap) > v_star + 1e-6:
        return None

    p_full = embed_support(p_S, S, n)
    return p_full


def embed_support(p_S: np.ndarray, S: List[int], n: int) -> np.ndarray:
    p = np.zeros(n)
    p[S] = p_S
    return p


def enumerate_mixed_NE(A: np.ndarray, max_support: int = 4, tol: float = 1e-9) -> List[np.ndarray]:
    """
    Enumerate supports up to size max_support and solve for symmetric mixed NE.
    Deduplicate (by L1 distance) and return list of distributions.
    """
    n = A.shape[0]
    sols: List[np.ndarray] = []

    def already_have(p: np.ndarray) -> bool:
        for q in sols:
            if np.linalg.norm(p - q, 1) < 1e-6:
                return True
        return False

    for m in range(2, min(max_support, n) + 1):
        for S in it.combinations(range(n), m):
            p = solve_mixed_support(A, list(S), tol=tol)
            if p is not None and not already_have(p):
                sols.append(p)
    return sols


def ess_check_pure(A: np.ndarray, i: int, tol: float = 1e-9) -> bool:
    """
    Pure ESS condition:
      1) i is a (symmetric) best response to i: A[i,i] >= A[k,i] for all k.
      2) For any tie k != i with A[k,i] == A[i,i], we require A[i,k] > A[k,k].
    """
    col = A[:, i]
    best = col.max()
    if A[i, i] + tol < best:
        return False
    # Ties (within tol)
    ties = [k for k, val in enumerate(col) if abs(val - best) <= tol and k != i]
    for k in ties:
        if not (A[i, k] > A[k, k] + tol):
            return False
    return True


def ess_check_mixed_grid(A: np.ndarray, p: np.ndarray, grid: int = 20, tol: float = 1e-7) -> bool:
    """
    Mixed ESS heuristic based on Maynard-Smith condition restricted to the
    set of best responses to p. Let C = argmax_i (A p)_i. For any q supported on C,
    ESS requires:  p' A q > q' A q.

    We sample q over the simplex on C with a uniform grid (Dirichlet-like lattice).
    If any q violates the strict inequality (within tol), we return False.

    Note: This is sufficient when |C| is small. Increase --ess-grid for stricter checks.
    """
    Ap = A @ p
    v_star = float(np.max(Ap))
    C = [i for i, val in enumerate(Ap) if v_star - val <= 1e-8]
    if len(C) == 0:
        # Shouldn't happen; p wouldn't be NE.
        return False

    # Fast exits for trivial cases
    if len(C) == 1:
        # Then any q supported on C is the pure at C[0], and strict inequality reduces to pure case.
        i = C[0]
        return (p @ (A @ unit_vec(len(p), i))) > (unit_vec(len(p), i) @ (A @ unit_vec(len(p), i))) + tol

    # Build all integer compositions of 'grid' into len(C) parts
    # Translate to q = counts / grid
    def compositions(total: int, parts: int):
        if parts == 1:
            yield (total,)
        else:
            for x in range(total + 1):
                for rest in compositions(total - x, parts - 1):
                    yield (x,) + rest

    pAp = float(p @ (A @ p))
    for counts in compositions(grid, len(C)):
        qC = np.array(counts, dtype=float) / float(grid)
        if qC.sum() == 0:
            continue
        # Skip q == p projected on C if identical (rare)
        q = np.zeros_like(p)
        for idx, s in enumerate(C):
            q[s] = qC[idx]

        # Only consider q that are best responses (they are, by construction)
        lhs = float(p @ (A @ q))      # p' A q
        rhs = float(q @ (A @ q))      # q' A q
        if not (lhs > rhs + tol):
            return False
    return True


def unit_vec(n: int, i: int) -> np.ndarray:
    v = np.zeros(n)
    v[i] = 1.0
    return v


def _dominance_lp(A: np.ndarray, i: int, strict: bool, tol: float
                  ) -> Optional[np.ndarray]:
    """
    Check whether strategy i is dominated by a mixture of other strategies.

    If strict=True: looks for p with sum_j p_j A[j,k] > A[i,k] for all k.
    If strict=False (weak): looks for p with sum_j p_j A[j,k] >= A[i,k] for all k,
                            with strict inequality for at least one k.

    Returns the dominating mixture (full-length vector) or None.
    """
    from scipy.optimize import linprog

    n = A.shape[0]
    others = [j for j in range(n) if j != i]
    m = len(others)

    # Variables: [p_0, ..., p_{m-1}, epsilon]
    # Maximize epsilon subject to:
    #   sum_j p_j (A[j,k] - A[i,k]) >= epsilon   for all k
    #   sum_j p_j = 1,  p_j >= 0,  epsilon >= 0
    c = np.zeros(m + 1)
    c[-1] = -1.0  # minimize -epsilon

    A_ub = np.zeros((n, m + 1))
    b_ub = np.zeros(n)
    for k in range(n):
        for idx, j in enumerate(others):
            A_ub[k, idx] = -(A[j, k] - A[i, k])
        A_ub[k, -1] = 1.0

    A_eq = np.zeros((1, m + 1))
    A_eq[0, :m] = 1.0
    b_eq = np.array([1.0])

    bounds = [(0, None)] * m + [(0, None)]

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method='highs')

    if not res.success:
        return None

    epsilon = -res.fun

    if strict and epsilon <= tol:
        return None

    if not strict:
        # Weak dominance: epsilon >= 0 (never worse) is sufficient,
        # but we also need at least one column strictly better.
        if epsilon < -tol:
            return None
        # Verify there's at least one column with strict improvement
        p_vec = res.x[:m]
        improvement = sum(p_vec[idx] * (A[j, k] - A[i, k])
                          for k in range(n)
                          for idx, j in enumerate(others))
        # Actually, check per-column: max improvement > tol
        max_gap = max(
            sum(p_vec[idx] * (A[j, k] - A[i, k]) for idx, j in enumerate(others))
            for k in range(n)
        )
        if max_gap <= tol:
            return None  # identical payoffs, not dominated

    p_full = np.zeros(n)
    for idx, j in enumerate(others):
        p_full[j] = res.x[idx]
    return p_full


def find_dominated_strategies(A: np.ndarray, tol: float = 1e-9
                              ) -> List[Tuple[int, str, List[int]]]:
    """
    Find dominated strategies (both strict and weak).

    Strict: there exists a mixture p (p_i=0) with p^T A[:,k] > A[i,k] for ALL k.
    Weak:   there exists a mixture p (p_i=0) with p^T A[:,k] >= A[i,k] for all k,
            and strictly > for at least one k.

    Returns list of (dominated_index, "strict"|"weak", [pure_dominator_indices]).
    The pure dominator list contains all j that individually dominate i
    (strictly or weakly, matching the kind). May be empty if only a mixture dominates.
    """
    n = A.shape[0]
    results: List[Tuple[int, str, List[int]]] = []

    for i in range(n):
        # Try strict first
        mix = _dominance_lp(A, i, strict=True, tol=tol)
        if mix is not None:
            pure_doms = [j for j in range(n)
                         if j != i and np.all(A[j, :] > A[i, :] + tol)]
            results.append((i, "strict", pure_doms))
            continue

        # Try weak
        mix = _dominance_lp(A, i, strict=False, tol=tol)
        if mix is not None:
            pure_doms = [j for j in range(n)
                         if j != i
                         and np.all(A[j, :] >= A[i, :] - tol)
                         and np.any(A[j, :] > A[i, :] + tol)]
            results.append((i, "weak", pure_doms))

    return results


def find_equivalent_strategies(A: np.ndarray, tol: float = 1e-9) -> List[List[int]]:
    """
    Find groups of strategically equivalent strategies.

    Strategies i and j are equivalent if:
    - Same row:    A[i,k] == A[j,k] for all k  (identical as dictator)
    - Same column: A[k,i] == A[k,j] for all k  (treated identically as recipient)

    Returns list of equivalence groups (each with 2+ members), sorted by first index.
    """
    n = A.shape[0]
    visited = [False] * n
    groups: List[List[int]] = []

    for i in range(n):
        if visited[i]:
            continue
        group = [i]
        for j in range(i + 1, n):
            if visited[j]:
                continue
            same_row = np.allclose(A[i, :], A[j, :], atol=tol)
            same_col = np.allclose(A[:, i], A[:, j], atol=tol)
            if same_row and same_col:
                group.append(j)
                visited[j] = True
        if len(group) > 1:
            groups.append(group)
        visited[i] = True

    return groups


def softmax_equilibrium(A: np.ndarray, beta: float,
                        f0: Optional[np.ndarray] = None,
                        pinned: Optional[Dict[int, float]] = None,
                        max_iter: int = 10000, tol: float = 1e-12) -> np.ndarray:
    """
    Find f* = softmax(beta * A @ f*) by fixed-point iteration.

    Pinned strategies keep their frequency fixed; the remaining strategies
    compete for the leftover mass via softmax.

    Args:
        A: (n x n) payoff matrix
        beta: inverse temperature (selection pressure)
        f0: initial frequency vector (default: uniform)
        pinned: dict mapping strategy index -> fixed frequency value
        max_iter: maximum iterations
        tol: convergence tolerance (L-inf norm of change in f)

    Returns:
        Equilibrium frequency vector f* on the simplex.
    """
    n = A.shape[0]
    f = np.ones(n) / n if f0 is None else f0.copy()

    if pinned is None:
        pinned = {}

    # Apply pins to initial vector
    free = [i for i in range(n) if i not in pinned]
    for i, v in pinned.items():
        f[i] = v

    pinned_mass = sum(pinned.values())
    free_mass = 1.0 - pinned_mass

    if free and free_mass > 0:
        # Renormalize free strategies to fill remaining mass
        free_sum = sum(f[i] for i in free)
        if free_sum > 0:
            for i in free:
                f[i] = f[i] / free_sum * free_mass
        else:
            for i in free:
                f[i] = free_mass / len(free)

    for _ in range(max_iter):
        rewards = A @ f                     # R_i(f)

        if not free:
            return f

        # Softmax over free strategies only
        log_f_free = beta * rewards[free]
        log_f_free -= log_f_free.max()       # numerical stability
        f_free = np.exp(log_f_free)
        f_free *= free_mass / f_free.sum()

        f_new = f.copy()
        for idx, i in enumerate(free):
            f_new[i] = f_free[idx]

        if np.max(np.abs(f_new - f)) < tol:
            return f_new
        f = f_new

    return f


def softmax_equilibria_sample(A: np.ndarray, beta: float,
                              num_samples: int = 1000,
                              pinned: Optional[Dict[int, float]] = None,
                              cluster_tol: float = 1e-4
                              ) -> List[Tuple[np.ndarray, int]]:
    """
    Sample random initial frequency vectors and find softmax fixed points.
    Cluster the results and return distinct equilibria with their basin counts.

    Args:
        A: (n x n) payoff matrix
        beta: inverse temperature
        num_samples: number of random starting points
        pinned: dict mapping strategy index -> fixed frequency value
        cluster_tol: L-inf distance to consider two fixed points identical

    Returns:
        List of (equilibrium_vector, count) sorted by count descending.
    """
    n = A.shape[0]
    rng = np.random.default_rng()

    if pinned is None:
        pinned = {}

    free = [i for i in range(n) if i not in pinned]
    pinned_mass = sum(pinned.values())
    free_mass = 1.0 - pinned_mass

    # Collect all fixed points
    clusters: List[Tuple[np.ndarray, int]] = []

    for _ in range(num_samples):
        # Build random initial vector: pinned values fixed, free from Dirichlet
        f0 = np.zeros(n)
        for i, v in pinned.items():
            f0[i] = v
        if free and free_mass > 0:
            f0_free = rng.dirichlet(np.ones(len(free))) * free_mass
            for idx, i in enumerate(free):
                f0[i] = f0_free[idx]

        f_star = softmax_equilibrium(A, beta, f0=f0, pinned=pinned)

        # Assign to existing cluster or create new one
        matched = False
        for idx, (centroid, count) in enumerate(clusters):
            if np.max(np.abs(f_star - centroid)) < cluster_tol:
                # Update centroid as running average
                clusters[idx] = (
                    (centroid * count + f_star) / (count + 1),
                    count + 1
                )
                matched = True
                break
        if not matched:
            clusters.append((f_star, 1))

    # Sort by count descending
    clusters.sort(key=lambda x: x[1], reverse=True)
    return clusters


def main():
    ap = argparse.ArgumentParser(description="Compute symmetric NE and ESS for the Open-Strategy Dictator Game.")
    ap.add_argument("--run", required=True, help="Directory with tournament run output (rounds.jsonl).")
    ap.add_argument("--max-support", type=int, default=4, help="Max support size to enumerate for mixed NE.")
    ap.add_argument("--ess-grid", type=int, default=20, help="Grid resolution for mixed ESS check (higher = stricter).")
    ap.add_argument("--beta", type=float, nargs="+", default=[1.0, 5.0, 20.0],
                    help="Inverse temperature(s) for softmax equilibrium (default: 1.0 5.0 20.0).")
    ap.add_argument("--samples", type=int, default=1000,
                    help="Number of random initial distributions to sample per beta (default: 1000).")
    ap.add_argument("--pin-freq", action="append", default=[],
                    metavar="STRATEGY=FREQ",
                    help='Pin a strategy frequency, e.g. --pin-freq "tiger=0.1". Repeatable.')
    ap.add_argument("--tol", type=float, default=1e-9, help="Numerical tolerance.")
    args = ap.parse_args()

    # Load tournament results from the run directory
    run_path = Path(args.run)
    rounds_path = run_path / "rounds.jsonl"

    # Calculate total endowment from first round
    # (Sum all allocations to get E)
    E = None

    # Collect all strategy names and build the v matrix
    strategy_names_set = set()
    rounds_data = []

    with open(rounds_path, "r") as f:
        for line in f:
            round_data = json.loads(line)
            rounds_data.append(round_data)

            # Collect strategy names
            strategy_names_set.add(round_data["dictator"])
            strategy_names_set.add(round_data["recipient"])

            # Calculate E from first round if not set
            if E is None:
                E = sum(round_data["allocation"].values())

    # Sort names for consistent ordering
    names = sorted(list(strategy_names_set))
    n = len(names)
    name_to_idx = {name: i for i, name in enumerate(names)}

    # Build v matrix: v[i, j] = amount dictator i awards to ITSELF when facing recipient j
    # Initialize with zeros and count how many times each pairing occurs
    v_sum = np.zeros((n, n), dtype=float)
    v_count = np.zeros((n, n), dtype=int)

    for round_data in rounds_data:
        dictator_name = round_data["dictator"]
        dictator_idx = name_to_idx[dictator_name]

        recipient_name = round_data["recipient"]
        recipient_idx = name_to_idx[recipient_name]

        # How much did this dictator award to itself?
        dictator_allocation = round_data["allocation"]["ME"]

        v_sum[dictator_idx, recipient_idx] += dictator_allocation
        v_count[dictator_idx, recipient_idx] += 1

    # Check that we have data
    if E is None:
        raise ValueError(f"No rounds found in {rounds_path}")

    # Average the allocations for each pairing
    v = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if v_count[i, j] > 0:
                v[i, j] = v_sum[i, j] / v_count[i, j]
            else:
                # No data for this pairing; use a default (e.g., equal split)
                v[i, j] = E / 2.0

    E = float(E)

    n = len(names)
    A = build_game_matrix(v, E)

    # Parse pinned frequencies
    pinned: Dict[int, float] = {}
    for pin_str in args.pin_freq:
        name, freq_str = pin_str.split("=", 1)
        name = name.strip()
        freq = float(freq_str.strip())
        if name not in name_to_idx:
            raise ValueError(f"Unknown strategy in --pin-freq: {name!r}. "
                             f"Available: {', '.join(names)}")
        if not 0.0 <= freq <= 1.0:
            raise ValueError(f"Pinned frequency must be in [0, 1], got {freq} for {name!r}")
        pinned[name_to_idx[name]] = freq

    if pinned:
        total_pinned = sum(pinned.values())
        if total_pinned > 1.0 + 1e-9:
            raise ValueError(f"Total pinned frequency {total_pinned:.4f} exceeds 1.0")

    print(f"\n=== Strategies (n={n}) ===")
    for i, s in enumerate(names):
        print(f"{i:2d}: {s}")

    # Dominated strategies
    dom = find_dominated_strategies(A, tol=args.tol)
    print("\n=== Dominated strategies ===")
    if not dom:
        print("None")
    else:
        for i, kind, pure_doms in dom:
            if pure_doms:
                dom_names = ", ".join(names[j] for j in pure_doms)
                print(f"- {names[i]}  {kind}ly dominated by: {dom_names}")
            else:
                print(f"- {names[i]}  {kind}ly dominated (by mixture only)")

    # Equivalent strategies
    equiv = find_equivalent_strategies(A, tol=args.tol)
    print("\n=== Equivalent strategies ===")
    if not equiv:
        print("None")
    else:
        for group in equiv:
            group_names = ", ".join(names[i] for i in group)
            print(f"- {{{group_names}}}")

    # Pure NE
    pures = pure_symmetric_NE(A, tol=args.tol)
    print("\n=== Pure symmetric NE ===")
    if not pures:
        print("None")
    else:
        for i in pures:
            is_ess = ess_check_pure(A, i, tol=args.tol)
            tag = "ESS" if is_ess else "not ESS"
            print(f"- {names[i]}  ({tag})")

    # Mixed NE (support size >= 2)
    mixes = enumerate_mixed_NE(A, max_support=args.max_support, tol=args.tol)
    print(f"\n=== Mixed symmetric NE (support ≤ {args.max_support}) ===")
    if not mixes:
        print("None")
    else:
        for p in mixes:
            support = [names[i] for i in np.where(p > 1e-8)[0]]
            val = float(p @ (A @ p))
            ess = ess_check_mixed_grid(A, p, grid=args.ess_grid, tol=1e-7)
            tag = "ESS?" if ess else "not ESS"
            probs = ", ".join(f"{names[i]}:{p[i]:.4f}" for i in np.where(p > 1e-8)[0])
            print(f"- support: {{{', '.join(support)}}} | value={val:.6f} | {tag}")
            print(f"  p = [{probs}]")

    # Optional: summarize best responses
    print("\n=== Best responses summary ===")
    for j in range(n):
        col = A[:, j]
        best = np.max(col)
        brs = [names[i] for i, x in enumerate(col) if best - x <= 1e-9]
        print(f"Against {names[j]}: best responses = {brs}")

    # Softmax equilibrium frequencies (multi-start)
    pin_label = ""
    if pinned:
        pin_parts = [f"{names[i]}={v:.2f}" for i, v in sorted(pinned.items())]
        pin_label = f", pinned: {', '.join(pin_parts)}"
    print(f"\n=== Softmax equilibria ({args.samples} random starts per beta{pin_label}) ===")
    for beta in args.beta:
        clusters = softmax_equilibria_sample(A, beta, num_samples=args.samples,
                                             pinned=pinned if pinned else None)
        print(f"\nbeta = {beta}  ({len(clusters)} distinct fixed point{'s' if len(clusters) != 1 else ''})")
        for eq_idx, (f, count) in enumerate(clusters):
            pct = 100 * count / args.samples
            R = A @ f
            print(f"  #{eq_idx + 1}  basin: {count}/{args.samples} ({pct:.1f}%)")
            order = np.argsort(-f)
            for i in order:
                if f[i] < 1e-6:
                    continue
                pin_tag = " [pinned]" if i in pinned else ""
                print(f"      {names[i]:<25s}  f={f[i]:.4f}  R={R[i]:.4f}{pin_tag}")

    print("\nDone.")


if __name__ == "__main__":
    main()
