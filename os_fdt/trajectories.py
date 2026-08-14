#!/usr/bin/env python3
"""
Open-Strategy Dictator Game — Fixed-Point Iteration Trajectories

Records how strategy frequencies shift across iterations of the two
fixed-point solvers:

  - softmax_equilibrium   (osdg_equilibria):  f = softmax(beta * A f)
  - mixture_equilibrium   (author_types):     f = sum_t mass_t * softmax(beta * U_t(f))

For each solver/beta configuration the basins of attraction are mapped by
seeded multi-start sampling and one example trajectory is traced per basin:
the uniform start for the basin it falls in, and the first sampled start
reaching each other basin. Each trajectory is truncated at the first iterate
within --tol (L-inf, default 0.01) of the fixed point, so the plotted path
shows the interesting transient rather than the long numerical tail.

Output:
  - a CSV per trajectory (one row per iteration, one column per strategy)
  - a 100% stacked area chart per trajectory
  - a combined multi-panel page (all trajectories, one shared legend)

All panels stack strategies in the same global order with the same colors,
so a single legend serves every chart.

Usage:
  python -m os_fdt.trajectories --run runs/tournament_9
  python -m os_fdt.trajectories --run runs/tournament_9 \
      --beta 1 5 20 --author-beta 5 20 --tol 0.01 --format pdf
"""
from __future__ import annotations
import argparse
import math
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from os_fdt.author_types import (build_matrices, load_allocation_matrix,
                                 mixture_equilibrium)
from os_fdt.osdg_equilibria import softmax_equilibrium

# Thematic strategy colors (light mode). Hue encodes the strategy family;
# shade separates members within a family:
#   greys       unconditional (ignore the recipient): light = give, dark = take
#   warm        second-order norm enforcers (judge how you treat others)
#   blues       direct reciprocity (judge how you would treat me)
#   aqua        guardian: reciprocity + protection (blend of the two above)
#   singletons  green = Kantian universalization, violet = intellect,
#               pink = broad warmth toward anything cooperative
STRATEGY_COLORS = {
    "generous": "#c3c2b7",
    "selfish": "#57554f",
    "cooperation-coalition": "#d95926",
    "chivalry": "#eda100",
    "mirror": "#2a78d6",
    "conditional-cooperator": "#6da7ec",
    "guardian": "#1baf7a",
    "universalizability": "#008300",
    "intelligence": "#4a3aa7",
    "anti-exploiter": "#e87ba4",
}
# Fallback slots (validated categorical palette) for strategies without a
# thematic color, assigned by prominence; beyond these, fold into "Other".
PALETTE = ["#eb6834", "#e34948", "#00a0c6", "#e87ba4", "#4a3aa7"]
OTHER_COLOR = "#8f8d85"
BASIN_COLORS = ["#2a78d6", "#eb6834", "#4a3aa7"]
INK_PRIMARY = "#0b0b0b"
INK_MUTED = "#898781"
AXIS_COLOR = "#c3c2b7"

FONT_RC = {
    "font.family": "serif",
    "font.serif": ["cmr10"],
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
    "svg.fonttype": "path",
}


def truncate_history(history: List[np.ndarray], tol: float = 0.01
                     ) -> List[np.ndarray]:
    """
    Keep iterates from the start up to (and including) the first one within
    tol (L-inf) of the fixed point, taken to be the last recorded iterate.
    """
    f_star = history[-1]
    for k, f in enumerate(history):
        if np.max(np.abs(f - f_star)) <= tol:
            return history[:max(k, 1) + 1]
    return list(history)


def history_to_dataframe(history: List[np.ndarray], names: List[str]
                         ) -> pd.DataFrame:
    df = pd.DataFrame(np.vstack(history), columns=names)
    df.insert(0, "iteration", range(len(history)))
    return df


def basin_representatives(solve: Callable[[np.ndarray], np.ndarray], n: int,
                          num_samples: int = 300, seed: int = 0,
                          cluster_tol: float = 1e-4
                          ) -> List[Tuple[np.ndarray, int]]:
    """
    Map basins of attraction by multi-start sampling and return one
    representative starting distribution per basin, largest basin first.

    The uniform start represents the basin it converges to; every other
    basin is represented by the first sampled Dirichlet start (seeded, so
    reproducible) that reaches it.

    Returns [(f0, sampled_basin_count)].
    """
    uniform = np.ones(n) / n
    # [fixed_point, sample_count, representative_start]
    clusters: List[list] = [[solve(uniform), 0, uniform]]
    rng = np.random.default_rng(seed)
    for _ in range(num_samples):
        f0 = rng.dirichlet(np.ones(n))
        f = solve(f0)
        for c in clusters:
            if np.max(np.abs(f - c[0])) < cluster_tol:
                c[1] += 1
                break
        else:
            clusters.append([f, 1, f0])
    clusters.sort(key=lambda c: -c[1])
    return [(c[2], c[1]) for c in clusters]


def sample_basin_labels(solve: Callable[[np.ndarray], np.ndarray], n: int,
                        num_samples: int = 2000, seed: int = 0,
                        cluster_tol: float = 1e-4
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample Dirichlet starting distributions and label each by the basin of
    attraction it falls in. Labels are re-indexed so 0 is the largest basin.

    Returns (X, labels): X[k] is the k-th start, labels[k] its basin index.
    """
    rng = np.random.default_rng(seed)
    fixed_points: List[np.ndarray] = []
    X = np.empty((num_samples, n))
    raw = np.empty(num_samples, dtype=int)
    for k in range(num_samples):
        f0 = rng.dirichlet(np.ones(n))
        f = solve(f0)
        for bi, fp in enumerate(fixed_points):
            if np.max(np.abs(f - fp)) < cluster_tol:
                raw[k] = bi
                break
        else:
            raw[k] = len(fixed_points)
            fixed_points.append(f)
        X[k] = f0
    by_size = sorted(range(len(fixed_points)),
                     key=lambda bi: -np.sum(raw == bi))
    relabel = {bi: rank for rank, bi in enumerate(by_size)}
    return X, np.array([relabel[b] for b in raw])


def _logistic_fit(Z: np.ndarray, y: np.ndarray, iters: int = 20000,
                  lr: float = 5.0) -> Tuple[np.ndarray, float, float]:
    """Plain-numpy logistic regression. Returns (weights, bias, accuracy)."""
    mu = Z.mean(axis=0)
    Zc = Z - mu
    w = np.zeros(Z.shape[1])
    b = 0.0
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(Zc @ w + b)))
        w -= lr * (Zc.T @ (p - y)) / len(y)
        b -= lr * float((p - y).mean())
    acc = float(((Zc @ w + b > 0) == y).mean())
    return w, b - float(w @ mu), acc


def discriminating_dims(X: np.ndarray, labels: np.ndarray, ndims: int
                        ) -> List[int]:
    """
    Coordinates of the initial distribution that best separate the two
    largest basins, ranked by standardized difference of class means.
    """
    d = np.abs(X[labels == 1].mean(axis=0) - X[labels == 0].mean(axis=0))
    d = d / X.std(axis=0)
    return sorted(np.argsort(-d)[:ndims].tolist())


def prominence_order(names: List[str], trajectories: List[List[np.ndarray]]
                     ) -> List[str]:
    """
    Global strategy order by prominence (max fixed-point mass across all
    trajectories). Every chart stacks in this order and every strategy keeps
    one color, so a single legend serves all panels.
    """
    peak = np.max(np.vstack([traj[-1] for traj in trajectories]), axis=0)
    order = sorted(range(len(names)), key=lambda i: (-peak[i], names[i]))
    return [names[i] for i in order]


def assign_colors(order: List[str]) -> Dict[str, str]:
    """
    Thematic color per strategy; strategies without one draw fallback slots
    (in prominence order, skipping colors already in use) and fold into
    "Other" once the fallbacks run out.
    """
    colors = {s: STRATEGY_COLORS[s] for s in order if s in STRATEGY_COLORS}
    fallback = iter([c for c in PALETTE if c not in colors.values()])
    for s in order:
        if s not in colors:
            colors[s] = next(fallback, OTHER_COLOR)
    return colors


def _stack_series(df: pd.DataFrame, order: List[str], colors: Dict[str, str]
                  ) -> List[Tuple[str, np.ndarray, str]]:
    """Series in stack order (most prominent at the bottom, 'Other' on top)."""
    shown = [s for s in order if colors[s] != OTHER_COLOR]
    folded = [s for s in order if colors[s] == OTHER_COLOR]
    series = [(s, df[s].to_numpy(), colors[s]) for s in shown]
    if folded:
        series.append(("Other", df[folded].sum(axis=1).to_numpy(), OTHER_COLOR))
    return series


def _draw_panel(ax, df: pd.DataFrame, order: List[str],
                colors: Dict[str, str], label: Optional[str] = None) -> None:
    from matplotlib.ticker import MaxNLocator, PercentFormatter

    series = _stack_series(df, order, colors)
    x = df["iteration"].to_numpy()
    polys = ax.stackplot(x, [vals for _, vals, _ in series],
                         colors=[c for _, _, c in series])
    for poly in polys:  # hairline white gap between adjacent bands
        poly.set_edgecolor("white")
        poly.set_linewidth(0.7)

    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.tick_params(colors=INK_MUTED, labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS_COLOR)
    if label:
        ax.set_title(label, fontsize=9, color=INK_PRIMARY)


def _legend_handles(order: List[str], colors: Dict[str, str]) -> list:
    """Legend top-to-bottom mirrors the stack: 'Other'/least prominent first."""
    from matplotlib.patches import Patch
    entries = [s for s in order if colors[s] != OTHER_COLOR]
    if len(entries) < len(order):
        entries.append("Other")
    return [Patch(facecolor=colors.get(s, OTHER_COLOR), label=s)
            for s in reversed(entries)]


def plot_stacked_area(df: pd.DataFrame, order: List[str],
                      colors: Dict[str, str], out_path: Path) -> None:
    """
    Standalone 100% stacked area chart. Untitled: figures are captioned in
    the paper. Computer Modern to match the paper body font.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(FONT_RC)
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    _draw_panel(ax, df, order, colors)
    ax.set_xlabel("Iteration", fontsize=10, color=INK_PRIMARY)
    ax.set_ylabel("Share of population", fontsize=10, color=INK_PRIMARY)
    ax.legend(handles=_legend_handles(order, colors), loc="center left",
              bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=9,
              handlelength=1.2, handleheight=1.0, labelcolor=INK_PRIMARY)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_combined(panels: List[Tuple[str, pd.DataFrame]], order: List[str],
                  colors: Dict[str, str], out_path: Path,
                  ncols: int = 2) -> None:
    """
    All trajectories on one page: a grid of stacked area panels sharing a
    single legend (and colors/stack order) across the figure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(FONT_RC)
    nrows = math.ceil(len(panels) / ncols)
    fig, axes = plt.subplots(nrows, ncols, layout="constrained",
                             figsize=(6.5, 2.1 * nrows + 0.7), squeeze=False)
    for k, (label, df) in enumerate(panels):
        ax = axes[k // ncols][k % ncols]
        _draw_panel(ax, df, order, colors, label=label)
        if k % ncols != 0:
            ax.set_yticklabels([])
        if k // ncols == nrows - 1:
            ax.set_xlabel("Iteration", fontsize=9, color=INK_PRIMARY)
    for k in range(len(panels), nrows * ncols):
        axes[k // ncols][k % ncols].set_visible(False)

    fig.supylabel("Share of population", fontsize=10, color=INK_PRIMARY)
    fig.legend(handles=list(reversed(_legend_handles(order, colors))),
               loc="outside lower center", ncol=3, frameon=False, fontsize=9,
               handlelength=1.2, handleheight=1.0, labelcolor=INK_PRIMARY)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_basin_scatter(X: np.ndarray, labels: np.ndarray, names: List[str],
                       dims: List[int], out_path: Path,
                       seed: int = 0) -> None:
    """
    Starting distributions projected onto the 2-3 most basin-discriminating
    coordinates, colored by the basin each converges to. In 2D the fitted
    linear decision boundary is drawn as a dashed line.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update(FONT_RC)
    point_colors = [BASIN_COLORS[min(b, len(BASIN_COLORS) - 1)]
                    for b in labels]
    # Shuffle draw order so the minority basin isn't buried.
    draw = np.random.default_rng(seed).permutation(len(labels))

    if len(dims) == 2:
        fig, ax = plt.subplots(figsize=(3.5, 3.0))
        ax.scatter(X[draw, dims[0]], X[draw, dims[1]],
                   c=[point_colors[k] for k in draw], s=4, linewidths=0)
        w, b, acc = _logistic_fit(X[:, dims], (labels > 0).astype(float))
        xs = np.linspace(0, X[:, dims[0]].max() * 1.02, 100)
        ys = -(w[0] * xs + b) / w[1]
        inside = (ys >= 0) & (ys <= X[:, dims[1]].max() * 1.02)
        ax.plot(xs[inside], ys[inside], "--", color=INK_MUTED, linewidth=1)
        ax.set_xlabel(f"initial {names[dims[0]]} share",
                      fontsize=9, color=INK_PRIMARY)
        ax.set_ylabel(f"initial {names[dims[1]]} share",
                      fontsize=9, color=INK_PRIMARY)
        ax.set_xlim(0, None)
        ax.set_ylim(0, None)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS_COLOR)
    else:
        fig = plt.figure(figsize=(5.2, 4.2))
        ax = fig.add_subplot(projection="3d")
        ax.scatter(X[draw, dims[0]], X[draw, dims[1]], X[draw, dims[2]],
                   c=[point_colors[k] for k in draw], s=5, linewidths=0)
        ax.set_xlabel(f"initial {names[dims[0]]}",
                      fontsize=9, color=INK_PRIMARY)
        ax.set_ylabel(f"initial {names[dims[1]]}",
                      fontsize=9, color=INK_PRIMARY)
        ax.set_zlabel(f"initial {names[dims[2]]}",
                      fontsize=9, color=INK_PRIMARY)

    ax.tick_params(colors=INK_MUTED, labelsize=8)
    handles = [Line2D([], [], marker="o", linestyle="", markersize=5,
                      color=BASIN_COLORS[b], label=f"basin {b + 1}")
               for b in sorted(set(labels))]
    ax.legend(handles=handles, frameon=False, fontsize=9,
              labelcolor=INK_PRIMARY, loc="upper right")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description="Trace fixed-point iterations of the OSDG equilibrium "
                    "solvers and emit CSVs plus 100%% stacked area charts.")
    ap.add_argument("--run", required=True,
                    help="Directory with tournament run output (rounds.jsonl).")
    ap.add_argument("--out", default=None,
                    help="Output directory (default: <run>/trajectories).")
    ap.add_argument("--tol", type=float, default=0.01,
                    help="Stop plotting once within this L-inf distance of "
                         "the fixed point (default: 0.01).")
    ap.add_argument("--format", choices=["pdf", "svg", "png"], default="pdf",
                    help="Chart file format (default: pdf, for LaTeX).")
    ap.add_argument("--samples", type=int, default=300,
                    help="Random starts per configuration when mapping "
                         "basins of attraction (default: 300).")
    ap.add_argument("--scatter-samples", type=int, default=2000,
                    help="Labeled starts for the basin scatter chart, drawn "
                         "for any configuration with multiple basins "
                         "(default: 2000).")
    ap.add_argument("--scatter-dims", type=int, choices=[2, 3], default=2,
                    help="Dimensions of the basin scatter chart (default: 2).")
    ap.add_argument("--seed", type=int, default=0,
                    help="RNG seed for basin sampling (default: 0).")
    ap.add_argument("--beta", type=float, nargs="+", default=[1.0, 5.0, 20.0],
                    help="Inverse temperature(s) for the softmax solver "
                         "(default: 1 5 20).")
    ap.add_argument("--author-beta", type=float, nargs="+", default=[5.0, 20.0],
                    help="Inverse temperature(s) for the author-type mixture "
                         "solver (default: 5 20).")
    ap.add_argument("--frac-own", type=float, default=0.4)
    ap.add_argument("--frac-mixed", type=float, default=0.4)
    ap.add_argument("--frac-altruist", type=float, default=0.2)
    ap.add_argument("--mixed-lambda", type=float, default=0.5)
    args = ap.parse_args()

    run_path = Path(args.run)
    out_dir = Path(args.out) if args.out else run_path / "trajectories"
    out_dir.mkdir(parents=True, exist_ok=True)

    names, v, E = load_allocation_matrix(run_path)
    n = len(names)
    A, W = build_matrices(v, E)

    fracs = np.array([args.frac_own, args.frac_mixed, args.frac_altruist])
    fracs = fracs / fracs.sum()
    types = [(lam, mass) for lam, mass
             in zip([0.0, args.mixed_lambda, 1.0], fracs) if mass > 0]

    solvers: List[Tuple[str, str, Callable]] = []
    for beta in args.beta:
        solvers.append((
            f"softmax_beta{beta:g}", f"softmax, $\\beta = {beta:g}$",
            lambda f0, hist, b=beta: softmax_equilibrium(
                A, b, f0=f0, history=hist)))
    for beta in args.author_beta:
        solvers.append((
            f"authors_beta{beta:g}", f"author types, $\\beta = {beta:g}$",
            lambda f0, hist, b=beta: mixture_equilibrium(
                A, W, types, b, f0, history=hist)[0]))

    # One trajectory per basin of attraction per configuration; for
    # configurations with multiple basins, also map which initial
    # distributions land where.
    jobs: List[Tuple[str, str, List[np.ndarray], int]] = []
    basin_maps: List[Tuple[str, Callable]] = []
    for stem, label, solve in solvers:
        reps = basin_representatives(lambda f0: solve(f0, None), n,
                                     num_samples=args.samples, seed=args.seed)
        if len(reps) > 1:
            basin_maps.append((stem, solve))
        for bi, (f0, count) in enumerate(reps):
            history: List[np.ndarray] = []
            solve(f0, history)
            traj = truncate_history(history, args.tol)
            if len(reps) > 1:
                jobs.append((f"{stem}_basin{bi + 1}",
                             f"{label} (basin {bi + 1})", traj, count))
            else:
                jobs.append((stem, label, traj, count))

    order = prominence_order(names, [traj for _, _, traj, _ in jobs])
    colors = assign_colors(order)
    folded = sorted(s for s in names if colors[s] == OTHER_COLOR)
    if folded:
        print(f"folded into 'Other' in charts (CSV keeps all columns): "
              f"{', '.join(folded)}")

    panels: List[Tuple[str, pd.DataFrame]] = []
    for stem, label, traj, count in jobs:
        df = history_to_dataframe(traj, names)
        csv_path = out_dir / f"{stem}.csv"
        fig_path = out_dir / f"{stem}.{args.format}"
        df.to_csv(csv_path, index=False, float_format="%.6f")
        plot_stacked_area(df, order, colors, fig_path)
        panels.append((label, df))
        f_star = traj[-1]
        top3 = ", ".join(f"{names[i]}:{f_star[i]:.2f}"
                         for i in np.argsort(-f_star)[:3])
        print(f"{stem}: {len(traj) - 1} iterations to within {args.tol:g} "
              f"of fixed point, basin {count}/{args.samples} ({top3})")
        print(f"  wrote {csv_path} and {fig_path}")

    combined_path = out_dir / f"all_trajectories.{args.format}"
    plot_combined(panels, order, colors, combined_path)
    print(f"combined page: {combined_path}")

    for stem, solve in basin_maps:
        X, blabels = sample_basin_labels(
            lambda f0: solve(f0, None), n,
            num_samples=args.scatter_samples, seed=args.seed)
        dims = discriminating_dims(X, blabels, args.scatter_dims)
        bdf = pd.DataFrame(X, columns=names)
        bdf["basin"] = blabels + 1
        csv_path = out_dir / f"{stem}_basin_map.csv"
        fig_path = out_dir / f"{stem}_basin_map.{args.format}"
        bdf.to_csv(csv_path, index=False, float_format="%.6f")
        plot_basin_scatter(X, blabels, names, dims, fig_path)
        y = (blabels > 0).astype(float)
        _, _, acc = _logistic_fit(X[:, dims], y)
        _, _, full_acc = _logistic_fit(X, y)
        dim_names = ", ".join(names[i] for i in dims)
        print(f"{stem} basin map over {args.scatter_samples} starts: "
              f"axes ({dim_names}) predict the basin with {acc:.1%} accuracy "
              f"(all {n} coordinates: {full_acc:.1%})")
        print(f"  wrote {csv_path} and {fig_path}")


if __name__ == "__main__":
    main()
