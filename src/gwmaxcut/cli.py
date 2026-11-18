import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from gwmaxcut import plotting
from gwmaxcut.experiments import REGISTRY
from gwmaxcut.graphs import from_edgelist
from gwmaxcut.gw import goemans_williamson
from gwmaxcut.heuristics import local_search


def _run(args):
    cfg = yaml.safe_load(Path(args.config).read_text())
    names = list(REGISTRY) if args.experiments == ["all"] else args.experiments
    unknown = set(names) - set(REGISTRY)
    if unknown:
        sys.exit(f"unknown experiment(s): {', '.join(sorted(unknown))}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in names:
        if name not in cfg:
            print(f"skipping {name}: no section in {args.config}")
            continue
        t0 = time.perf_counter()
        solver = cfg[name].get("solver", cfg.get("solver"))
        result = REGISTRY[name](cfg[name], cfg["seed"], solver)
        if name == "gw_bound":
            result, curves = result
            curves.to_csv(out / "gw_running_mean.csv", index=False)
        result.to_csv(out / f"{name}.csv", index=False)
        print(f"{name}: {len(result)} rows in {time.perf_counter() - t0:.1f}s")


def _plot(args):
    res, fig = Path(args.results), Path(args.figures)
    cfg = yaml.safe_load(Path(args.config).read_text())

    def load(name):
        path = res / f"{name}.csv"
        return pd.read_csv(path) if path.exists() else None

    if (df := load("runtime")) is not None:
        plotting.plot_runtime(df, fig)
    if (df := load("approximation")) is not None:
        plotting.plot_approximation(df, fig)
    if (df := load("gw_bound")) is not None:
        curves = load("gw_running_mean")
        plotting.plot_gw_bound(df, curves if curves is not None else pd.DataFrame(), fig)
    if (df := load("best_of_k")) is not None:
        plotting.plot_best_of_k(df, fig, cfg["best_of_k"]["k"])
    if (df := load("rounding_law")) is not None:
        plotting.plot_rounding_law(df, fig)
    if (df := load("large_n")) is not None:
        plotting.plot_large_n(df, fig)
    if (df := load("density")) is not None:
        plotting.plot_density(df, fig)
    if (df := load("best_of_t")) is not None:
        plotting.plot_best_of_t(df, fig)
    print(f"figures written to {fig}/")


def _solve(args):
    W = from_edgelist(args.edgelist)
    res = goemans_williamson(W, n_rounds=args.rounds, rng=np.random.default_rng(args.seed), solver=args.solver)
    value, x = res.best_cut, res.best_x
    if args.local_search:
        value, x = local_search(W, x)
    side = " ".join(str(i) for i in np.flatnonzero(x > 0))
    print(f"vertices: {W.shape[0]}   total weight: {W.sum() / 2:g}")
    print(f"SDP upper bound:   {res.sdp_value:.4f}")
    print(f"expected GW cut:   {res.expected:.4f}")
    print(f"best cut found:    {value:.4f}  ({value / res.sdp_value:.3f} of the bound)")
    print(f"side S: {side}")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="gwmaxcut", description="Max-Cut experiments and solver.")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run experiments and write CSV files")
    run.add_argument("experiments", nargs="+", help=f"'all' or any of: {', '.join(REGISTRY)}")
    run.add_argument("--config", default="configs/thesis.yaml")
    run.add_argument("--out", default="results")
    run.set_defaults(func=_run)

    plot = sub.add_parser("plot", help="draw figures from the CSV files")
    plot.add_argument("--config", default="configs/thesis.yaml")
    plot.add_argument("--results", default="results")
    plot.add_argument("--figures", default="figures")
    plot.set_defaults(func=_plot)

    solve = sub.add_parser("solve", help="approximate Max-Cut of a graph given as an edge list")
    solve.add_argument("edgelist")
    solve.add_argument("--rounds", type=int, default=100)
    solve.add_argument("--seed", type=int, default=0)
    solve.add_argument("--solver", default=None)
    solve.add_argument("--local-search", action="store_true")
    solve.set_defaults(func=_solve)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
