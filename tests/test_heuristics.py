import numpy as np

from gwmaxcut import cut_value, derandomized_cut, erdos_renyi, local_search, random_cut


def test_derandomized_reaches_half_the_edges():
    rng = np.random.default_rng(3)
    for _ in range(30):
        W = erdos_renyi(15, rng.uniform(0.1, 0.9), rng, weighted=True)
        value, x = derandomized_cut(W)
        assert value >= W.sum() / 4 - 1e-12
        assert np.isclose(cut_value(W, x), value)


def test_random_cut_mean_is_half():
    rng = np.random.default_rng(4)
    W = erdos_renyi(20, 0.5, rng)
    vals = np.array([random_cut(W, rng)[0] for _ in range(4000)])
    se = vals.std() / np.sqrt(len(vals))
    assert abs(vals.mean() - W.sum() / 4) < 4 * se


def test_local_search_is_one_flip_optimal():
    rng = np.random.default_rng(5)
    W = erdos_renyi(25, 0.3, rng, weighted=True)
    start = rng.choice([-1.0, 1.0], size=25)
    value, x = local_search(W, start)
    assert value >= cut_value(W, start)
    for v in range(25):
        y = x.copy()
        y[v] = -y[v]
        assert cut_value(W, y) <= value + 1e-9
