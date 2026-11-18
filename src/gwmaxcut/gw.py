import time
from dataclasses import dataclass, field

import cvxpy as cp
import numpy as np

from gwmaxcut.cut import cut_values

ALPHA_GW = 0.8785672057848516


@dataclass
class GWResult:
    sdp_value: float
    X: np.ndarray
    cuts: np.ndarray  # value of every rounding
    best_cut: float
    best_x: np.ndarray
    expected: float  # exact E[cut] of one rounding, from the SDP solution
    solve_time: float = 0.0
    info: dict = field(default_factory=dict)


def solve_sdp(W: np.ndarray, solver: str | None = None) -> tuple[float, np.ndarray]:
    """max 1/4 sum_ij w_ij (1 - X_ij)  s.t.  X PSD, diag(X) = 1."""
    n = W.shape[0]
    if solver is None:
        # Clarabel is more accurate, but on some sparse graphs with n >= ~150 it
        # needed more than 8 GB of memory; SCS solves the same instances in < 1 s.
        solver = "CLARABEL" if n <= 100 else "SCS"
    X = cp.Variable((n, n), symmetric=True)
    objective = cp.Maximize((W.sum() - cp.sum(cp.multiply(W, X))) / 4)
    problem = cp.Problem(objective, [X >> 0, cp.diag(X) == 1])
    problem.solve(solver=solver)
    if problem.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"SDP solver returned status {problem.status!r}")
    return float(problem.value), X.value


def gram_vectors(X: np.ndarray) -> np.ndarray:
    """Rows v_i with v_i . v_j = X_ij.

    The solver output is only PSD up to ~1e-8, so small negative eigenvalues
    are clipped. Eigenvectors must be scaled by sqrt(eigenvalue): dropping the
    scaling gives vectors whose inner products are not X.
    """
    evals, evecs = np.linalg.eigh((X + X.T) / 2)
    V = evecs * np.sqrt(np.clip(evals, 0.0, None))
    return V / np.linalg.norm(V, axis=1, keepdims=True)


def round_hyperplane(V: np.ndarray, n_rounds: int, rng: np.random.Generator) -> np.ndarray:
    """One random hyperplane per column; returns signs of shape (n, n_rounds)."""
    R = rng.standard_normal((V.shape[1], n_rounds))
    S = np.sign(V @ R)
    S[S == 0] = 1.0
    return S


def expected_cut(W: np.ndarray, X: np.ndarray) -> float:
    """E[cut] = sum_{i<j} w_ij arccos(X_ij) / pi  (Goemans-Williamson, Lemma 3.2)."""
    theta = np.arccos(np.clip(X, -1.0, 1.0))
    return float(np.sum(np.triu(W * theta, k=1)) / np.pi)


def goemans_williamson(
    W: np.ndarray,
    n_rounds: int = 1,
    rng: np.random.Generator | None = None,
    solver: str | None = None,
) -> GWResult:
    rng = np.random.default_rng() if rng is None else rng
    t0 = time.perf_counter()
    sdp_value, X = solve_sdp(W, solver=solver)
    solve_time = time.perf_counter() - t0

    S = round_hyperplane(gram_vectors(X), n_rounds, rng)
    cuts = cut_values(W, S)
    k = int(np.argmax(cuts))
    return GWResult(
        sdp_value=sdp_value,
        X=X,
        cuts=cuts,
        best_cut=float(cuts[k]),
        best_x=S[:, k].copy(),
        expected=expected_cut(W, X),
        solve_time=solve_time,
    )
