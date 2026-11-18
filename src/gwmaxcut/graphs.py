"""Graphs are stored as dense symmetric weight matrices with zero diagonal.

Every algorithm in this package is at least quadratic in n, so a dense matrix
is the simplest representation that is also the fastest one in practice.
"""

from pathlib import Path

import networkx as nx
import numpy as np


def erdos_renyi(n: int, p: float, rng: np.random.Generator, weighted: bool = False) -> np.ndarray:
    upper = np.triu(rng.random((n, n)) < p, k=1).astype(float)
    if weighted:
        upper *= rng.random((n, n))
    return upper + upper.T


def cycle(n: int) -> np.ndarray:
    W = np.zeros((n, n))
    idx = np.arange(n)
    W[idx, (idx + 1) % n] = 1.0
    return W + W.T


def from_networkx(G: nx.Graph, weight: str = "weight") -> np.ndarray:
    W = nx.to_numpy_array(G, weight=weight, dtype=float)
    np.fill_diagonal(W, 0.0)
    return W


def from_edgelist(path: str | Path) -> np.ndarray:
    """Read lines of the form `u v` or `u v w` (0-based vertex labels)."""
    rows = [line.split() for line in Path(path).read_text().splitlines() if line.strip() and not line.startswith("#")]
    edges = [(int(r[0]), int(r[1]), float(r[2]) if len(r) > 2 else 1.0) for r in rows]
    n = 1 + max(max(u, v) for u, v, _ in edges)
    W = np.zeros((n, n))
    for u, v, w in edges:
        if u != v:
            W[u, v] = W[v, u] = w
    return W


def total_weight(W: np.ndarray) -> float:
    return float(W.sum() / 2)
