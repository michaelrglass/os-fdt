import numpy as np

from os_fdt.author_types import mixture_equilibrium
from os_fdt.osdg_equilibria import softmax_equilibrium
from os_fdt.trajectories import (_logistic_fit, basin_representatives,
                                 discriminating_dims, history_to_dataframe,
                                 sample_basin_labels, truncate_history)

A = np.array([[2.0, 0.5],
              [0.5, 1.0]])
W = A + A.T


def test_softmax_history_records_start_and_fixed_point():
    history = []
    f_star = softmax_equilibrium(A, beta=2.0, history=history)
    assert len(history) >= 2
    assert np.allclose(history[0], [0.5, 0.5])
    assert np.allclose(history[-1], f_star)
    for f in history:
        assert abs(f.sum() - 1.0) < 1e-9


def test_softmax_history_default_off():
    f_star = softmax_equilibrium(A, beta=2.0)
    history = []
    assert np.allclose(softmax_equilibrium(A, beta=2.0, history=history), f_star)


def test_mixture_history_records_start_and_fixed_point():
    history = []
    types = [(0.0, 0.6), (1.0, 0.4)]
    f0 = np.ones(2) / 2
    f_star, _ = mixture_equilibrium(A, W, types, beta=2.0, f0=f0,
                                    history=history)
    assert len(history) >= 2
    assert np.allclose(history[0], f0)
    assert np.allclose(history[-1], f_star)


def test_truncate_history_stops_within_tolerance():
    target = np.array([1.0, 0.0])
    history = [np.array([0.5, 0.5]), np.array([0.8, 0.2]),
               np.array([0.995, 0.005]), np.array([0.9999, 0.0001]), target]
    truncated = truncate_history(history, tol=0.01)
    assert len(truncated) == 3
    assert np.max(np.abs(truncated[-1] - target)) <= 0.01
    assert np.max(np.abs(truncated[-2] - target)) > 0.01


def test_truncate_history_keeps_at_least_two_points():
    history = [np.array([0.5, 0.5]), np.array([0.5, 0.5])]
    assert len(truncate_history(history, tol=0.01)) == 2


def test_basin_representatives_finds_both_coordination_basins():
    # Pure coordination at high beta: random starts split between the two
    # pure equilibria; the uniform start stays at the symmetric mixed point.
    coord = np.array([[2.0, 0.0],
                      [0.0, 2.0]])
    reps = basin_representatives(
        lambda f0: softmax_equilibrium(coord, beta=20.0, f0=f0),
        n=2, num_samples=50, seed=0)
    fixed_points = [softmax_equilibrium(coord, beta=20.0, f0=f0)
                    for f0, _ in reps]
    assert len(reps) >= 2
    assert sum(count for _, count in reps) == 50
    for fa, fb in zip(fixed_points, fixed_points[1:]):
        assert np.max(np.abs(fa - fb)) > 1e-3


def test_basin_representatives_single_basin_uses_uniform_start():
    reps = basin_representatives(
        lambda f0: softmax_equilibrium(A, beta=1.0, f0=f0),
        n=2, num_samples=20, seed=0)
    assert len(reps) == 1
    assert np.allclose(reps[0][0], [0.5, 0.5])


def test_sample_basin_labels_and_discriminating_dims():
    # Coordination game: the basin is decided by which strategy starts ahead,
    # and both coordinates discriminate equally well.
    coord = np.array([[2.0, 0.0],
                      [0.0, 2.0]])
    X, labels = sample_basin_labels(
        lambda f0: softmax_equilibrium(coord, beta=20.0, f0=f0),
        n=2, num_samples=100, seed=0)
    assert X.shape == (100, 2)
    assert set(labels) == {0, 1}
    assert np.sum(labels == 0) >= np.sum(labels == 1)  # 0 is largest basin
    dims = discriminating_dims(X, labels, ndims=1)
    _, _, acc = _logistic_fit(X[:, dims], (labels > 0).astype(float))
    assert acc > 0.95


def test_history_to_dataframe_shape():
    history = [np.array([0.5, 0.5]), np.array([0.7, 0.3])]
    df = history_to_dataframe(history, ["a", "b"])
    assert list(df.columns) == ["iteration", "a", "b"]
    assert list(df["iteration"]) == [0, 1]
    assert df["a"].iloc[1] == 0.7
