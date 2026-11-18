from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from gwmaxcut.gw import ALPHA_GW  # noqa: E402

COLORS = {
    "brute_force": "#444444",
    "random": "#d95f02",
    "derandomized": "#e6ab02",
    "goemans_williamson": "#1b9e77",
    "local_search": "#7570b3",
    "sdp": "#1f78b4",
}
LABELS = {
    "brute_force": "Brute force",
    "random": "Random 0.5",
    "derandomized": "Derandomized 0.5",
    "goemans_williamson": "Goemans-Williamson",
    "local_search": "GW + local search",
    "sdp": "SDP bound",
}

plt.rcParams.update(
    {
        "figure.figsize": (6.4, 4.0),
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "savefig.bbox": "tight",
        "savefig.dpi": 150,
    }
)


def _alpha_line(ax):
    ax.axhline(ALPHA_GW, color="black", ls="--", lw=1, label=r"$\alpha_{GW} \approx 0.878$")


def _save(fig, out: Path, name: str):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png")
    fig.savefig(out / f"{name}.pdf")
    plt.close(fig)


def plot_runtime(df: pd.DataFrame, out: Path):
    fig, ax = plt.subplots()
    med = df.groupby(["algorithm", "n"]).seconds.median().reset_index()
    for alg, d in med.groupby("algorithm"):
        ax.plot(d.n, d.seconds, "o-", ms=3, color=COLORS[alg], label=LABELS[alg])
    ax.set(yscale="log", xlabel="number of vertices n", ylabel="time [s] (median)", title="Running time on G(n, 0.4)")
    ax.legend()
    _save(fig, out, "runtime")


def plot_approximation(df: pd.DataFrame, out: Path):
    g = (
        df.groupby("n")
        .agg(
            opt=("opt", "mean"),
            gw=("gw_mean", "mean"),
            rnd=("random_mean", "mean"),
            gw_sd=("gw_var", lambda v: np.sqrt(v.mean())),
            rnd_sd=("random_var", lambda v: np.sqrt(v.mean())),
        )
        .reset_index()
    )

    fig, ax = plt.subplots()
    ax.plot(g.n, g.opt, "o-", ms=3, color=COLORS["brute_force"], label="Exact Max-Cut")
    for key, col in (("gw", "goemans_williamson"), ("rnd", "random")):
        ax.errorbar(g.n, g[key], yerr=g[f"{key}_sd"], fmt="o-", ms=3, capsize=2, color=COLORS[col], label=LABELS[col])
    ax.set(xlabel="number of vertices n", ylabel="cut value (mean over graphs)", title="Exact vs approximate cut")
    ax.legend()
    _save(fig, out, "approximation_values")

    r = df.assign(gw=df.gw_mean / df.opt, rnd=df.random_mean / df.opt, der=df.derandomized / df.opt)
    r = r.groupby("n")[["gw", "rnd", "der"]].mean().reset_index()
    fig, ax = plt.subplots()
    ax.plot(r.n, r.gw, "o-", ms=3, color=COLORS["goemans_williamson"], label=LABELS["goemans_williamson"])
    ax.plot(r.n, r.der, "s-", ms=3, color=COLORS["derandomized"], label=LABELS["derandomized"])
    ax.plot(r.n, r.rnd, "o-", ms=3, color=COLORS["random"], label=LABELS["random"])
    _alpha_line(ax)
    ax.axhline(0.5, color="grey", ls=":", lw=1)
    ax.set(xlabel="number of vertices n", ylabel="cut / exact Max-Cut", ylim=(0.45, 1.02), title="Approximation ratio")
    ax.legend(loc="lower right")
    _save(fig, out, "approximation_ratio")


def plot_gw_bound(df: pd.DataFrame, curves: pd.DataFrame, out: Path):
    g = (
        df.groupby("n")
        .agg(m=("mean_ratio", "mean"), s=("mean_ratio", "std"), exact=("exact_ratio", "mean"))
        .reset_index()
    )
    fig, ax = plt.subplots()
    ax.errorbar(
        g.n,
        g.m,
        yerr=g.s,
        fmt="o",
        ms=4,
        capsize=2,
        color=COLORS["goemans_williamson"],
        label="limit of the running mean (± sd over graphs)",
    )
    ax.plot(g.n, g.exact, "x", color="black", label=r"exact $\sum w_{ij}\theta_{ij}/\pi$ / OPT")
    _alpha_line(ax)
    ax.set(
        xlabel="number of vertices n", ylabel="E[GW cut] / OPT", ylim=(0.85, 1.01), title="Goemans-Williamson guarantee"
    )
    ax.legend(loc="lower right", fontsize=8)
    _save(fig, out, "gw_bound")

    if not curves.empty:
        fig, ax = plt.subplots()
        for _, c in curves.groupby("graph"):
            ax.plot(c.t, c.running_mean, lw=1, alpha=0.8)
        _alpha_line(ax)
        ax.set(
            xscale="log",
            xlabel="number of roundings",
            ylabel="running mean of cut / OPT",
            title="Convergence of the running mean",
        )
        ax.legend()
        _save(fig, out, "gw_running_mean")


def plot_best_of_k(df: pd.DataFrame, out: Path, k: int):
    g = (
        df.groupby("n")
        .agg(
            best=("best_ratio", "mean"),
            worst=("best_ratio", "min"),
            mean=("mean_ratio", "mean"),
            t_gw=("time_gw", "median"),
            t_bf=("time_brute_force", "median"),
        )
        .reset_index()
    )
    fig, ax = plt.subplots()
    ax.plot(g.n, g.best, "o-", ms=3, color=COLORS["goemans_williamson"], label=f"best of {k} roundings (mean)")
    ax.plot(g.n, g.worst, "v", ms=4, color=COLORS["goemans_williamson"], alpha=0.6, label=f"best of {k} (worst graph)")
    ax.plot(g.n, g["mean"], "o--", ms=3, color="grey", label="single rounding (mean)")
    _alpha_line(ax)
    ax.set(
        xlabel="number of vertices n", ylabel="cut / OPT", ylim=(0.8, 1.01), title=f"Repeating the rounding {k} times"
    )
    ax.legend(loc="lower left", fontsize=8)
    _save(fig, out, "best_of_k")

    fig, ax = plt.subplots()
    ax.plot(g.n, g.t_bf, "o-", ms=3, color=COLORS["brute_force"], label=LABELS["brute_force"])
    ax.plot(g.n, g.t_gw, "o-", ms=3, color=COLORS["goemans_williamson"], label=f"GW, 1 SDP + {k} roundings")
    ax.set(yscale="log", xlabel="number of vertices n", ylabel="time [s] (median)", title="Running time")
    ax.legend()
    _save(fig, out, "best_of_k_time")


def plot_rounding_law(df: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.scatter(df.theta, df.empirical, s=6, alpha=0.4, color=COLORS["goemans_williamson"], label="edges (empirical)")
    t = np.linspace(0, np.pi, 200)
    ax.plot(t, t / np.pi, color="black", lw=1, label=r"$\theta/\pi$")
    ax.set(
        xlabel=r"angle $\theta_{ij}$ between $v_i$ and $v_j$",
        ylabel="fraction of roundings cutting the edge",
        title="Random hyperplane: P(cut) = θ/π",
    )
    ax.legend()
    _save(fig, out, "rounding_law")


def plot_large_n(df: pd.DataFrame, out: Path):
    cols = {
        "gw_local_search": "local_search",
        "gw_expected": "goemans_williamson",
        "derandomized": "derandomized",
        "random_mean": "random",
    }
    ps = sorted(df.p.unique())
    fig, axes = plt.subplots(1, len(ps), figsize=(5 * len(ps), 4.2), sharey=True, squeeze=False)
    for ax, p in zip(axes[0], ps, strict=True):
        d = df[df.p == p]
        for c, key in cols.items():
            r = (d[c] / d.sdp).groupby(d.n).mean()
            label = "GW (expected cut)" if key == "goemans_williamson" else LABELS[key]
            ax.plot(r.index, r.values, "o-", ms=3, color=COLORS[key], label=label)
        _alpha_line(ax)
        ax.set(xlabel="number of vertices n", title=f"p = {p}", ylim=(0.5, 1.0))
    axes[0][0].set_ylabel("cut / SDP bound  (a lower bound on cut / OPT)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.1))
    fig.suptitle("Graphs too large for brute force")
    _save(fig, out, "large_n")

    fig, ax = plt.subplots()
    for p in ps:
        d = df[df.p == p].groupby("n").sdp_seconds.median()
        slope = np.polyfit(np.log(d.index), np.log(d.values), 1)[0]
        ax.plot(d.index, d.values, "o-", ms=3, label=f"p = {p}  (fitted slope {slope:.1f})")
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="number of vertices n",
        ylabel="SDP solve time [s], SCS",
        title="Cost of the SDP relaxation",
    )
    ax.legend()
    _save(fig, out, "sdp_time")


def plot_density(df: pd.DataFrame, out: Path):
    g = df.groupby("p")[["gw_ratio", "sdp_ratio"]].agg(["mean", "std"])
    fig, ax = plt.subplots()
    ax.errorbar(
        g.index,
        g.gw_ratio["mean"],
        yerr=g.gw_ratio["std"],
        fmt="o-",
        ms=3,
        capsize=2,
        color=COLORS["goemans_williamson"],
        label="E[GW cut] / OPT",
    )
    ax.errorbar(
        g.index,
        g.sdp_ratio["mean"],
        yerr=g.sdp_ratio["std"],
        fmt="s-",
        ms=3,
        capsize=2,
        color=COLORS["sdp"],
        label="SDP / OPT",
    )
    _alpha_line(ax)
    ax.axhline(1, color="grey", lw=0.8)
    ax.set(xlabel="edge probability p", ylabel="ratio", title="Effect of density (n = 20)")
    ax.legend()
    _save(fig, out, "density")


def plot_best_of_t(df: pd.DataFrame, out: Path):
    fig, ax = plt.subplots()
    for _, d in df.groupby("graph"):
        ax.plot(d["T"], d.best_ratio, lw=0.8, alpha=0.5, color=COLORS["goemans_williamson"])
    m = df.groupby("T").best_ratio.mean()
    ax.plot(m.index, m.values, color="black", lw=1.5, label="mean over graphs")
    _alpha_line(ax)
    ax.set(
        xscale="log",
        xlabel="number of roundings T",
        ylabel="E[best of T] / OPT",
        title="Diminishing returns of repeated rounding",
    )
    ax.legend(loc="lower right")
    _save(fig, out, "best_of_t")
