#!/usr/bin/env python3
"""
Sensitivity analyses for the equilibrium results.

Four sweeps, selected with --sweep:

  share (default): vary the relative payoff of SHARE vs TAKE.
      Fix the TAKE payoff at 1.0 and vary the SHARE payoff sigma.  For each
      value, build a payoff matrix from the tournament's decision matrix and
      compute the softmax equilibria (multi-start).
      Payoff structure for a round where i is dictator against j:
        SHARE: dictator gets sigma, recipient gets sigma
        TAKE:  dictator gets 1.0, recipient gets 0
      Total payoff: A[i,j] = score_as_dictator(d_ij) + score_as_recipient(d_ji)

  types: vary the author-type composition (own / mixed / altruist masses)
      over a grid on the simplex and recompute the mixture equilibrium of
      author_types at each point.

  mixed-lambda: vary the mixed type's welfare weight lambda while holding
      the type masses fixed.

  bloc: vary the mass of the pinned altruist bloc and recompute the bloc
      scan ranking of candidate strategies at each mass.

All sweeps use damped fixed-point iteration (fixed points are unaffected;
plain iteration cycles for some payoff parameterizations) and a seeded RNG.

Usage:
  python -m os_fdt.sensitivity --run runs/tournament_9
  python -m os_fdt.sensitivity --run runs/tournament_9 --sweep types
"""
import argparse
import math
from pathlib import Path

import numpy as np

from .author_types import (build_matrices, load_allocation_matrix,
                           format_dist, sample_equilibria, total_welfare,
                           type_utilities)
from .decision_matrix import load_decision_matrix
from .osdg_equilibria import softmax_equilibria_sample, softmax_equilibrium


def softmax_residual(A: np.ndarray, beta: float, f: np.ndarray) -> float:
    """L-inf distance between f and one softmax fixed-point update."""
    e = beta * (A @ f)
    t = np.exp(e - e.max())
    t /= t.sum()
    return float(np.max(np.abs(t - f)))


def mixture_residual(A: np.ndarray, W: np.ndarray, types, beta: float,
                     f: np.ndarray, pinned=None) -> float:
    """L-inf distance between f and one mixture fixed-point update."""
    fn = np.zeros_like(f)
    if pinned is not None:
        fn[pinned[0]] = pinned[1]
    for lam, mass in types:
        e = beta * type_utilities(A, W, f, lam)
        q = np.exp(e - e.max())
        q /= q.sum()
        fn += mass * q
    return float(np.max(np.abs(fn - f)))


def check_residuals(residuals, context: str, tol: float = 1e-8) -> str:
    """Return a warning marker if any cluster failed to converge."""
    bad = [r for r in residuals if r > tol]
    if bad:
        print(f"WARNING: {context}: {len(bad)} cluster(s) with residual "
              f"> {tol:g} (max {max(bad):.2e}); reduce --damping.")
        return " !"
    return ""


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


def plot_share_sweep(rows, names, out_path: Path) -> None:
    """
    100% stacked area of the uniform-start equilibrium's composition as a
    function of the SHARE payoff sigma, with that basin's share of starts
    overlaid (drops below 1 where a rival equilibrium exists).  Untitled
    and in Computer Modern, following the paper's figure conventions.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import PercentFormatter

    from .trajectories import (AXIS_COLOR, FONT_RC, INK_MUTED, INK_PRIMARY,
                               assign_colors, _legend_handles)

    sigma = np.array([s for s, _, _, _ in rows])
    comp = np.vstack([f for _, f, _, _ in rows])
    frac = np.array([share for _, _, share, _ in rows])

    peak = comp.max(axis=0)
    order = [names[i] for i in
             sorted(range(len(names)), key=lambda i: (-peak[i], names[i]))]
    colors = assign_colors(order)

    plt.rcParams.update(FONT_RC)
    fig, ax = plt.subplots(figsize=(5.4, 2.7))
    series = [comp[:, names.index(s)] for s in order]
    polys = ax.stackplot(sigma, series, colors=[colors[s] for s in order])
    for poly in polys:
        poly.set_edgecolor("white")
        poly.set_linewidth(0.7)

    # Share of starts reaching the dominant equilibrium: 1.0 where the
    # fixed point is unique, lower where a rival basin exists.
    ax.plot(sigma, frac, color=INK_PRIMARY, linewidth=1.0,
            linestyle=(0, (4, 2)))

    for x, label in ((0.5, r"$\rho = 1$"), (0.8, r"$\rho = 4$")):
        ax.axvline(x, color="white", linewidth=0.9, linestyle=(0, (1, 2)))
        ax.text(x, 1.03, label, ha="center", fontsize=8, color=INK_MUTED)
    x_log = math.log(31) / math.log(61)
    ax.axvline(x_log, color=INK_PRIMARY, linewidth=0.9,
               linestyle=(0, (1, 2)))
    ax.text(x_log, -0.24, r"$\ln$ utility", ha="center", fontsize=8,
            color=INK_PRIMARY, transform=ax.get_xaxis_transform())

    ax.set_xlim(sigma[0], sigma[-1])
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.tick_params(colors=INK_MUTED, labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS_COLOR)
    ax.set_xlabel(r"Share payoff $\sigma$", fontsize=10, color=INK_PRIMARY)
    ax.set_ylabel("Equilibrium share", fontsize=10, color=INK_PRIMARY)

    handles = [Line2D([], [], color=INK_PRIMARY, linewidth=1.0,
                      linestyle=(0, (4, 2)), label="basin share")]
    handles += _legend_handles(order, colors)
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5),
              frameon=False, fontsize=8, handlelength=1.2, handleheight=1.0,
              labelcolor=INK_PRIMARY)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def sweep_share(args) -> int:
    """Vary the SHARE payoff sigma with TAKE normalized to 1."""
    rounds_path = Path(args.run) / "rounds.jsonl"
    if not rounds_path.exists():
        print(f"Error: {rounds_path} not found")
        return 1

    names, decisions = load_decision_matrix(rounds_path)
    n = len(names)

    share_values = np.linspace(args.share_min, args.share_max, args.steps)
    rng = np.random.default_rng(args.seed)

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

    rows = []
    for s in share_values:
        A = build_payoff_matrix(names, decisions, share_reward=s)
        clusters = softmax_equilibria_sample(A, args.beta,
                                             num_samples=args.samples,
                                             damping=args.damping, rng=rng,
                                             max_iter=args.max_iter)
        mark = check_residuals(
            [softmax_residual(A, args.beta, f) for f, _ in clusters],
            f"share={s:.3f}")

        # The equilibrium selected by the uniform prior (for the figure:
        # continuous in sigma, where "largest basin" can flip on sampling
        # noise when basins are near even), plus its share of starts.
        f_unif = softmax_equilibrium(A, args.beta, damping=args.damping,
                                     max_iter=args.max_iter)
        unif_count = next((c for f_c, c in clusters
                           if np.max(np.abs(f_c - f_unif)) < 1e-3),
                          clusters[0][1])
        rows.append((s, f_unif, unif_count / args.samples, clusters))

        # Report the dominant equilibrium (largest basin)
        f, count = clusters[0]
        winner = names[np.argmax(f)]

        print(f"{s:7.3f}  {winner:<{col_w}s}", end="")
        for i in range(n):
            print(f"  {f[i]:{col_w}.4f}", end="")
        n_basins = len(clusters)
        basin_str = f"{count}/{args.samples}"
        if n_basins > 1:
            basin_str += f" ({n_basins} eq)"
        print(f"  {basin_str}{mark}")

    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["share", "basin_rank", "basin_count"] + list(names))
            for s, _, _, clusters in rows:
                for rank, (f, count) in enumerate(clusters):
                    w.writerow([f"{s:.6f}", rank + 1, count]
                               + [f"{x:.6f}" for x in f])
        print(f"\nWrote {args.csv}")
    if args.figure:
        plot_share_sweep(rows, list(names), Path(args.figure))
        print(f"Wrote {args.figure}")
    return 0


def type_grid(step: float):
    """All (own, mixed, altruist) mass triples on a simplex grid."""
    k = round(1.0 / step)
    for i in range(k + 1):
        for j in range(k + 1 - i):
            yield (i / k, j / k, (k - i - j) / k)


def mixture_clusters(A, W, masses, lam_mixed, args, rng, context):
    """sample_equilibria for an (own, mixed, altruist) mass triple, with a
    convergence check on every distinct fixed point found.  Points where
    the damped iteration still cycles are retried with stronger damping."""
    types = [(lam, m) for lam, m in
             zip([0.0, lam_mixed, 1.0], masses) if m > 0]
    for damping in (args.damping, 0.05, 0.02):
        clusters = sample_equilibria(A, W, types, args.beta, args.samples,
                                     rng=rng, damping=damping,
                                     max_iter=args.max_iter)
        residuals = [mixture_residual(A, W, types, args.beta, f)
                     for f, _, _ in clusters]
        if max(residuals) <= 1e-8:
            return clusters, ""
    mark = check_residuals(residuals, context)
    return clusters, mark


def sweep_types(args) -> int:
    """Vary the author-type composition over a grid on the simplex."""
    names, v, E = load_allocation_matrix(Path(args.run))
    A, W = build_matrices(v, E)
    rng = np.random.default_rng(args.seed)

    print(f"\nSensitivity: author-type masses (own, mixed(lambda="
          f"{args.mixed_lambda:g}), altruist) on a step-{args.type_step:g} "
          f"grid, beta={args.beta:g}, {args.samples} starts per point\n")
    print(f"{'own':>5s} {'mixed':>6s} {'altr':>5s}  {'#eq':>3s} "
          f"{'basin':>7s}  {'V':>7s}  dominant equilibrium")
    for masses in type_grid(args.type_step):
        context = f"own={masses[0]:.2f} mixed={masses[1]:.2f} altr={masses[2]:.2f}"
        clusters, mark = mixture_clusters(A, W, masses, args.mixed_lambda,
                                          args, rng, context)
        f, _, count = clusters[0]
        extra = ""
        if len(clusters) > 1:
            f2, _, count2 = clusters[1]
            extra = (f"   [eq2: {count2}/{args.samples} "
                     f"V={total_welfare(W, f2):.4f} "
                     f"{format_dist(names, f2, min_mass=0.05)}]")
        print(f"{masses[0]:5.2f} {masses[1]:6.2f} {masses[2]:5.2f}  "
              f"{len(clusters):3d} {count:4d}/{args.samples}  "
              f"{total_welfare(W, f):7.4f}  "
              f"{format_dist(names, f, min_mass=0.02)}{extra}{mark}")
    return 0


def sweep_mixed_lambda(args) -> int:
    """Vary the mixed type's welfare weight at fixed type masses."""
    names, v, E = load_allocation_matrix(Path(args.run))
    A, W = build_matrices(v, E)
    rng = np.random.default_rng(args.seed)
    masses = (args.frac_own, args.frac_mixed, args.frac_altruist)

    print(f"\nSensitivity: mixed-type lambda, masses own={masses[0]:g} "
          f"mixed={masses[1]:g} altruist={masses[2]:g}, beta={args.beta:g}, "
          f"{args.samples} starts per point\n")
    for lam in np.linspace(0.0, 1.0, args.steps):
        clusters, mark = mixture_clusters(A, W, masses, lam, args, rng,
                                          f"lambda={lam:.2f}")
        f, _, count = clusters[0]
        print(f"lambda={lam:5.2f}  #eq={len(clusters)}  "
              f"basin {count}/{args.samples}  V={total_welfare(W, f):.4f}  "
              f"{format_dist(names, f, min_mass=0.02)}{mark}")
    return 0


def sweep_bloc(args) -> int:
    """Vary the pinned altruist-bloc mass; re-rank candidates at each mass."""
    names, v, E = load_allocation_matrix(Path(args.run))
    n = len(names)
    A, W = build_matrices(v, E)
    rng = np.random.default_rng(args.seed)

    print(f"\nSensitivity: altruist-bloc mass (remainder split evenly "
          f"between own and mixed(lambda={args.mixed_lambda:g})), "
          f"beta={args.beta:g}, {args.samples} starts per pin\n")
    for bloc in args.bloc_masses:
        free = (1.0 - bloc) / 2.0
        free_types = [(0.0, free), (args.mixed_lambda, free)]
        rows = []
        for k in range(n):
            clusters = sample_equilibria(A, W, free_types, args.beta,
                                         args.samples, pinned=(k, bloc),
                                         rng=rng, damping=args.damping,
                                         max_iter=args.max_iter)
            check_residuals(
                [mixture_residual(A, W, free_types, args.beta, f,
                                  pinned=(k, bloc))
                 for f, _, _ in clusters],
                f"bloc={bloc:.2f} pin={names[k]}")
            f, _, _ = clusters[0]
            rows.append((total_welfare(W, f), k, len(clusters)))
        rows.sort(key=lambda r: -r[0])
        ranking = "  >  ".join(
            f"{names[k]} ({V:.3f}{'' if ne == 1 else f', {ne} eq'})"
            for V, k, ne in rows[:4])
        worst = rows[-1]
        print(f"bloc={bloc:4.2f}:  {ranking}")
        print(f"           worst: {names[worst[1]]} ({worst[0]:.3f})   "
              f"generous rank: "
              f"{[names[k] for _, k, _ in rows].index('generous') + 1}/{n}")
    return 0


def main():
    ap = argparse.ArgumentParser(
        description="Sensitivity analyses for the equilibrium results.")
    ap.add_argument("--run", required=True,
                    help="Directory with tournament run output (rounds.jsonl).")
    ap.add_argument("--sweep", choices=["share", "types", "mixed-lambda",
                                        "bloc"], default="share",
                    help="Which sensitivity sweep to run (default: share).")
    ap.add_argument("--beta", type=float, default=20.0,
                    help="Inverse temperature for softmax (default: 20.0).")
    ap.add_argument("--samples", type=int, default=1000,
                    help="Random starting points per configuration (default: 1000).")
    ap.add_argument("--steps", type=int, default=10,
                    help="Number of values to test in 1-D sweeps (default: 10).")
    ap.add_argument("--share-min", type=float, default=0.50,
                    help="Minimum SHARE payoff (default: 0.50).")
    ap.add_argument("--share-max", type=float, default=0.95,
                    help="Maximum SHARE payoff (default: 0.95).")
    ap.add_argument("--type-step", type=float, default=0.1,
                    help="Grid step for the type-mass simplex (default: 0.1).")
    ap.add_argument("--mixed-lambda", type=float, default=0.5,
                    help="Welfare weight of the mixed type (default: 0.5).")
    ap.add_argument("--frac-own", type=float, default=0.4)
    ap.add_argument("--frac-mixed", type=float, default=0.4)
    ap.add_argument("--frac-altruist", type=float, default=0.2)
    ap.add_argument("--bloc-masses", type=float, nargs="+",
                    default=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
                    help="Altruist-bloc masses for the bloc sweep.")
    ap.add_argument("--damping", type=float, default=0.2,
                    help="Fixed-point damping in (0, 1] (default: 0.2).")
    ap.add_argument("--max-iter", type=int, default=100000,
                    help="Max fixed-point iterations (default: 100000).")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed.")
    ap.add_argument("--csv", help="Optional CSV output path (share sweep).")
    ap.add_argument("--figure",
                    help="Optional figure output path (share sweep).")
    args = ap.parse_args()

    return {"share": sweep_share, "types": sweep_types,
            "mixed-lambda": sweep_mixed_lambda, "bloc": sweep_bloc}[args.sweep](args)


if __name__ == "__main__":
    exit(main())
