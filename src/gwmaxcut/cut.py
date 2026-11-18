import numpy as np


def cut_value(W: np.ndarray, x: np.ndarray) -> float:
    """Weight of the cut defined by x in {-1, +1}^n."""
    return float((W.sum() - x @ W @ x) / 4)


def cut_values(W: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Cut values for many partitions at once; X has one partition per column."""
    return (W.sum() - np.einsum("it,it->t", X, W @ X)) / 4
