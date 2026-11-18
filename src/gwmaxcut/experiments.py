"""Experiments of the thesis (chapter 5) plus a few extra ones.

Each function takes its section of the YAML config and returns a DataFrame
with one row per graph (or per graph and algorithm). Graph g of size n in
experiment e is always generated from the seed (seed, e, n, g), so any single
row can be recomputed in isolation.
"""

import time
from collections.abc import Callable

import numpy as np
import pandas as pd

from gwmaxcut.cut import cut_values
from gwmaxcut.exact import brute_force, n_partitions
from gwmaxcut.graphs import erdos_renyi
from gwmaxcut.gw import expected_cut, gram_vectors, round_hyperplane, solve_sdp
from gwmaxcut.heuristics import derandomized_cut, local_search, random_cut

EXPERIMENT_IDS = {
    "runtime": 1,
    "approximation": 2,
    "gw_bound": 3,
    "best_of_k": 4,
    "rounding_law": 5,
    "large_n": 6,
    "density": 7,
    "best_of_t": 8,
}


def _sizes(spec) -> list[int]:
    # either an explicit list, a single value or {from: a, to: b, step: s}
    if isinstance(spec, dict):
        return list(range(spec["from"], spec["to"] + 1, spec.get("step", 1)))
    return list(spec) if isinstance(spec, list) else [spec]


def _rng(seed: int, name: str, *keys: float) -> np.random.Generator:
    return np.random.default_rng([seed, EXPERIMENT_IDS[name], *(int(round(k * 1000)) for k in keys)])


def _timed(f: Callable, *args):
    t0 = time.perf_counter()
    out = f(*args)
    return out, time.perf_counter() - t0


def _gw_cuts(W, rounds, rng, solver):
    (sdp, X), t_sdp = _timed(solve_sdp, W, solver)
    S = round_hyperplane(gram_vectors(X), rounds, rng)
    return sdp, X, S, cut_values(W, S), t_sdp


def runtime(cfg: dict, seed: int, solver: str | None = None, log=print) -> pd.DataFrame:
    rows = []
    for n in _sizes(cfg["n"]):
        for r in range(cfg.get("repeats", 1)):
            rng = _rng(seed, "runtime", n, r)
            W = erdos_renyi(n, cfg["p"], rng)
            timings = {
                "random": _timed(random_cut, W, rng)[1],
                "derandomized": _timed(derandomized_cut, W)[1],
            }
            t0 = time.perf_counter()
            sdp, X = solve_sdp(W, solver)
            round_hyperplane(gram_vectors(X), 1, rng)
            timings["goemans_williamson"] = time.perf_counter() - t0
            if n <= cfg.get("brute_force_max_n", 25):
                timings["brute_force"] = _timed(brute_force, W)[1]
            for alg, sec in timings.items():
                evaluated = n_partitions(n) if alg == "brute_force" else np.nan
                rows.append({"n": n, "repeat": r, "algorithm": alg, "seconds": sec, "partitions": evaluated})
        log(f"runtime: n={n} done")
    return pd.DataFrame(rows)


def approximation(cfg: dict, seed: int, solver: str | None = None, log=print) -> pd.DataFrame:
    rows = []
    for n in _sizes(cfg["n"]):
        for g in range(cfg["graphs"]):
            rng = _rng(seed, "approximation", n, g)
            W = erdos_renyi(n, cfg["p"], rng)
            opt, _ = brute_force(W)
            rand = np.array([random_cut(W, rng)[0] for _ in range(cfg["runs"])])
            sdp, X, _, gw, _ = _gw_cuts(W, cfg["runs"], rng, solver)
            rows.append(
                {
                    "n": n,
                    "graph": g,
                    "edges": W.sum() / 2,
                    "opt": opt,
                    "sdp": sdp,
                    "random_mean": rand.mean(),
                    "random_var": rand.var(),
                    "gw_mean": gw.mean(),
                    "gw_var": gw.var(),
                    "derandomized": derandomized_cut(W)[0],
                }
            )
        log(f"approximation: n={n} done")
    return pd.DataFrame(rows)


def gw_bound(cfg: dict, seed: int, solver: str | None = None, log=print) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Thesis 5.3.3: N roundings per graph, limit of the running mean of cut/OPT.

    Besides the empirical limit we store the exact expectation of one rounding,
    sum w_ij arccos(X_ij)/pi, which is what the running mean converges to.
    """
    rows, curves = [], []
    keep_n = cfg.get("curve_n", 20)
    for n in _sizes(cfg["n"]):
        for g in range(cfg["graphs"]):
            rng = _rng(seed, "gw_bound", n, g)
            W = erdos_renyi(n, cfg["p"], rng)
            if W.sum() == 0:
                continue
            opt, _ = brute_force(W)
            sdp, X, _, cuts, _ = _gw_cuts(W, cfg["rounds"], rng, solver)
            ratio = cuts / opt
            rows.append(
                {
                    "n": n,
                    "graph": g,
                    "opt": opt,
                    "sdp": sdp,
                    "mean_ratio": ratio.mean(),
                    "exact_ratio": expected_cut(W, X) / opt,
                    "min_ratio": ratio.min(),
                    "best_ratio": ratio.max(),
                    "sdp_gap": opt / sdp,
                }
            )
            if n == keep_n:
                running = np.cumsum(ratio) / np.arange(1, len(ratio) + 1)
                curves.append(pd.DataFrame({"graph": g, "t": np.arange(1, len(ratio) + 1), "running_mean": running}))
        log(f"gw_bound: n={n} done")
    return pd.DataFrame(rows), pd.concat(curves, ignore_index=True) if curves else pd.DataFrame()


def best_of_k(cfg: dict, seed: int, solver: str | None = None, log=print) -> pd.DataFrame:
    rows = []
    k = cfg["k"]
    for n in _sizes(cfg["n"]):
        for g in range(cfg["graphs"]):
            rng = _rng(seed, "best_of_k", n, g)
            W = erdos_renyi(n, cfg["p"], rng)
            if W.sum() == 0:
                continue
            (opt, _), t_bf = _timed(brute_force, W)
            t0 = time.perf_counter()
            _, _, _, cuts, _ = _gw_cuts(W, k, rng, solver)
            t_gw = time.perf_counter() - t0
            rows.append(
                {
                    "n": n,
                    "graph": g,
                    "opt": opt,
                    "best_ratio": cuts.max() / opt,
                    "mean_ratio": cuts.mean() / opt,
                    "time_gw": t_gw,
                    "time_brute_force": t_bf,
                }
            )
        log(f"best_of_k: n={n} done")
    return pd.DataFrame(rows)


def rounding_law(cfg: dict, seed: int, solver: str | None = None, log=print) -> pd.DataFrame:
    """Check P(edge ij is cut) = theta_ij / pi edge by edge."""
    frames = []
    for g in range(cfg["graphs"]):
        rng = _rng(seed, "rounding_law", cfg["n"], g)
        W = erdos_renyi(cfg["n"], cfg["p"], rng)
        _, X = solve_sdp(W, solver)
        S = round_hyperplane(gram_vectors(X), cfg["rounds"], rng)
        i, j = np.nonzero(np.triu(W, k=1))
        freq = (S[i] != S[j]).mean(axis=1)
        theta = np.arccos(np.clip(X[i, j], -1, 1))
        frames.append(pd.DataFrame({"graph": g, "theta": theta, "empirical": freq, "theory": theta / np.pi}))
        log(f"rounding_law: graph {g} done")
    return pd.concat(frames, ignore_index=True)


def large_n(cfg: dict, seed: int, solver: str | None = None, log=print) -> pd.DataFrame:
    """Beyond brute force: the SDP value is an upper bound on OPT, so cut/SDP is
    a certified lower bound on cut/OPT."""
    rows = []
    for p in cfg["p"]:
        for n in _sizes(cfg["n"]):
            for g in range(cfg["graphs"]):
                rng = _rng(seed, "large_n", n, p, g)
                W = erdos_renyi(n, p, rng)
                sdp, X, S, cuts, t_sdp = _gw_cuts(W, cfg["rounds"], rng, solver)
                k = int(np.argmax(cuts))
                rows.append(
                    {
                        "n": n,
                        "p": p,
                        "graph": g,
                        "sdp": sdp,
                        "sdp_seconds": t_sdp,
                        "gw_mean": cuts.mean(),
                        "gw_expected": expected_cut(W, X),
                        "gw_best": cuts[k],
                        "gw_local_search": local_search(W, S[:, k])[0],
                        "derandomized": derandomized_cut(W)[0],
                        "random_mean": W.sum() / 4,
                    }
                )
            log(f"large_n: p={p} n={n} done")
    return pd.DataFrame(rows)


def density(cfg: dict, seed: int, solver: str | None = None, log=print) -> pd.DataFrame:
    rows = []
    n = cfg["n"]
    for p in cfg["p"]:
        for g in range(cfg["graphs"]):
            rng = _rng(seed, "density", n, p, g)
            W = erdos_renyi(n, p, rng)
            if W.sum() == 0:
                continue
            opt, _ = brute_force(W)
            sdp, X = solve_sdp(W, solver)
            rows.append({"p": p, "graph": g, "opt": opt, "sdp": sdp, "gw_expected": expected_cut(W, X)})
        log(f"density: p={p} done")
    df = pd.DataFrame(rows)
    df["gw_ratio"] = df.gw_expected / df.opt
    df["sdp_ratio"] = df.sdp / df.opt
    return df


def best_of_t(cfg: dict, seed: int, solver: str | None = None, log=print) -> pd.DataFrame:
    """Expected best-of-T ratio for T = 1..rounds, averaged over random orderings
    of the same pool of roundings."""
    rows = []
    n = cfg["n"]
    for g in range(cfg["graphs"]):
        rng = _rng(seed, "best_of_t", n, g)
        W = erdos_renyi(n, cfg["p"], rng)
        opt, _ = brute_force(W)
        _, _, _, cuts, _ = _gw_cuts(W, cfg["rounds"], rng, solver)
        perms = np.array([rng.permutation(cuts) for _ in range(cfg.get("orderings", 50))])
        curve = np.maximum.accumulate(perms, axis=1).mean(axis=0) / opt
        rows.append(pd.DataFrame({"graph": g, "T": np.arange(1, len(cuts) + 1), "best_ratio": curve}))
        log(f"best_of_t: graph {g} done")
    return pd.concat(rows, ignore_index=True)


REGISTRY = {
    "runtime": runtime,
    "approximation": approximation,
    "gw_bound": gw_bound,
    "best_of_k": best_of_k,
    "rounding_law": rounding_law,
    "large_n": large_n,
    "density": density,
    "best_of_t": best_of_t,
}
