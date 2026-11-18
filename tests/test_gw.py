import numpy as np
import pytest

from gwmaxcut import brute_force, cycle, erdos_renyi, expected_cut, goemans_williamson, solve_sdp
from gwmaxcut.gw import ALPHA_GW, gram_vectors


def test_sdp_value_of_c5():
    # classical example: OPT = 4, SDP = (25 + 5 sqrt 5) / 8
    value, _ = solve_sdp(cycle(5))
    assert value == pytest.approx((25 + 5 * np.sqrt(5)) / 8, abs=1e-4)


def test_bipartite_graph_is_solved_exactly():
    W = np.zeros((6, 6))
    W[:3, 3:] = 1
    W = W + W.T
    res = goemans_williamson(W, n_rounds=20, rng=np.random.default_rng(0))
    assert res.sdp_value == pytest.approx(9, abs=1e-4)
    assert res.best_cut == 9


def test_gram_vectors_reproduce_X():
    rng = np.random.default_rng(6)
    W = erdos_renyi(12, 0.5, rng)
    _, X = solve_sdp(W)
    V = gram_vectors(X)
    assert np.allclose(V @ V.T, X, atol=1e-4)


def test_ordering_sdp_opt_gw():
    rng = np.random.default_rng(7)
    for _ in range(15):
        W = erdos_renyi(11, 0.5, rng)
        if W.sum() == 0:
            continue
        opt, _ = brute_force(W)
        res = goemans_williamson(W, n_rounds=50, rng=rng)
        assert res.sdp_value >= opt - 1e-4
        assert opt >= res.best_cut
        assert res.expected >= ALPHA_GW * res.sdp_value - 1e-4


def test_rounding_mean_matches_expectation():
    rng = np.random.default_rng(8)
    W = erdos_renyi(15, 0.4, rng)
    res = goemans_williamson(W, n_rounds=20000, rng=rng)
    se = res.cuts.std() / np.sqrt(len(res.cuts))
    assert abs(res.cuts.mean() - expected_cut(W, res.X)) < 4 * se


def test_same_seed_same_result():
    W = erdos_renyi(10, 0.5, np.random.default_rng(9))
    a = goemans_williamson(W, n_rounds=30, rng=np.random.default_rng(42))
    b = goemans_williamson(W, n_rounds=30, rng=np.random.default_rng(42))
    assert np.array_equal(a.cuts, b.cuts)
