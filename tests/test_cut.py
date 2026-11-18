import numpy as np

from gwmaxcut import brute_force, cut_value, cut_values, cycle, erdos_renyi


def naive_cut(W, x):
    n = len(x)
    return sum(W[i, j] for i in range(n) for j in range(i + 1, n) if x[i] != x[j])


def test_cut_value_matches_definition():
    rng = np.random.default_rng(0)
    for _ in range(20):
        W = erdos_renyi(9, 0.5, rng, weighted=True)
        x = rng.choice([-1.0, 1.0], size=9)
        assert np.isclose(cut_value(W, x), naive_cut(W, x))


def test_cut_values_columns():
    rng = np.random.default_rng(1)
    W = erdos_renyi(10, 0.4, rng)
    X = rng.choice([-1.0, 1.0], size=(10, 7))
    assert np.allclose(cut_values(W, X), [cut_value(W, X[:, t]) for t in range(7)])


def test_brute_force_small_cases():
    assert brute_force(cycle(5))[0] == 4
    assert brute_force(cycle(6))[0] == 6
    K4 = np.ones((4, 4)) - np.eye(4)
    assert brute_force(K4)[0] == 4


def test_brute_force_against_itertools():
    import itertools

    rng = np.random.default_rng(2)
    W = erdos_renyi(8, 0.5, rng, weighted=True)
    best = max(naive_cut(W, x) for x in itertools.product([-1, 1], repeat=8))
    value, x = brute_force(W, block_bits=3)  # small blocks to exercise the loop
    assert np.isclose(value, best)
    assert np.isclose(cut_value(W, x), value)
