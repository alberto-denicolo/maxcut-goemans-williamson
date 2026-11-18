import numpy as np

from gwmaxcut.cut import cut_value


def random_cut(W: np.ndarray, rng: np.random.Generator) -> tuple[float, np.ndarray]:
    """Each vertex on either side with probability 1/2: E[cut] = W_tot / 2 >= OPT / 2."""
    x = rng.choice([-1.0, 1.0], size=W.shape[0])
    return cut_value(W, x), x


def derandomized_cut(W: np.ndarray, order: np.ndarray | None = None) -> tuple[float, np.ndarray]:
    """Method of conditional expectations applied to random_cut.

    Placing vertex v only changes the expected cut through edges to vertices
    already placed, so v goes to the side opposite to most of that weight.
    The result is deterministic and always at least W_tot / 2.
    """
    n = W.shape[0]
    order = np.arange(n) if order is None else np.asarray(order)
    x = np.zeros(n)
    for v in order:
        # x[v] = -sign(sum_u w_uv x_u): unplaced vertices have x_u = 0
        x[v] = -1.0 if W[v] @ x > 0 else 1.0
    return cut_value(W, x), x


def local_search(W: np.ndarray, x: np.ndarray, max_flips: int | None = None) -> tuple[float, np.ndarray]:
    """Flip the single vertex with the largest gain until no flip improves the cut."""
    x = x.astype(float).copy()
    Wx = W @ x
    max_flips = 10 * W.shape[0] ** 2 if max_flips is None else max_flips
    for _ in range(max_flips):
        gain = x * Wx  # gain of flipping v is x_v * (Wx)_v
        v = int(np.argmax(gain))
        if gain[v] <= 1e-12:
            break
        x[v] = -x[v]
        Wx += 2 * x[v] * W[:, v]
    return cut_value(W, x), x
