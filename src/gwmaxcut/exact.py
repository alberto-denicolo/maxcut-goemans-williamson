import numpy as np


def brute_force(W: np.ndarray, block_bits: int = 18) -> tuple[float, np.ndarray]:
    """Exact Max-Cut by enumerating all 2^(n-1) partitions.

    Vertex 0 is fixed on the +1 side (x and -x give the same cut), the other
    n-1 signs are read from the bits of an integer counter. Partitions are
    evaluated in blocks so that memory stays bounded for n around 25.
    """
    n = W.shape[0]
    if n <= 1:
        return 0.0, np.ones(n)
    m = n - 1
    total = 1 << m
    block = 1 << min(block_bits, m)
    shifts = np.arange(m, dtype=np.int64)
    half_sum = W.sum() / 4

    best_val, best_code = -np.inf, 0
    for start in range(0, total, block):
        codes = np.arange(start, min(start + block, total), dtype=np.int64)
        signs = 1.0 - 2.0 * ((codes[:, None] >> shifts) & 1)
        X = np.hstack([np.ones((len(codes), 1)), signs])
        vals = half_sum - np.einsum("ti,ti->t", X @ W, X) / 4
        k = int(np.argmax(vals))
        if vals[k] > best_val:
            best_val, best_code = float(vals[k]), int(codes[k])

    x = np.ones(n)
    x[1:] = 1.0 - 2.0 * ((best_code >> np.arange(m)) & 1)
    return best_val, x


def n_partitions(n: int) -> int:
    return 1 << max(n - 1, 0)
