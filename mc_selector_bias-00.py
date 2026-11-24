"""Monte Carlo study for the manuscript (Proposition 1 and the baseline result).

It reuses the forecasting functions of mdpi_7sector_reviewer_ready_analysis-00.py
(same models, same 9-8-7 protocol, same bandwidth grid) on simulated annual series
of length 24 and reports:
  mc_grid_bias.csv   optimism gap of the min-over-grid inner score versus the honest
                     prequential inner score, by kernel and grid size (Proposition 1);
  mc_selection.csv   LWLR and baseline selection shares of the honest and the
                     optimistic adaptive selectors;
  mc_accuracy.csv    out-of-sample accuracy of all methods relative to Naive;
  mc_corridor.csv    Proposition 5: coverage of the sequential MAD corridor and the
                     probability that a downward shift of size delta*sigma is
                     classified as "Below trend", finite-sample (8 calibration +
                     7 diagnostic residuals, as in the 9-8-7 protocol) and large-sample
                     (200 past residuals), compared with 2*Phi(lambda)-1 and Phi(delta-lambda).
Data-generating process (log level z_t, y_t = exp(z_t)):
  smooth : z_t = z_{t-1} + mu + sigma * eps_t
  shocks : as smooth, plus pre-specified negative valuation shocks at the positions of
           1999, 2009, 2015 and 2016 and a partial rebound in 2017.
Usage: python mc_selector_bias-00.py [--reps 200] [--output-dir outputs-00/simulation]
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("analysis", HERE / "mdpi_7sector_reviewer_ready_analysis-00.py")
A = importlib.util.module_from_spec(spec)
sys.modules["analysis"] = A
spec.loader.exec_module(A)

T = 24
MU, SIGMA = 0.08, 0.15
SHOCKS = {4: -0.25, 14: -0.35, 20: -0.25, 21: -0.30, 22: 0.20}   # 1999, 2009, 2015, 2016, 2017
GRID_SIZES = (2, 5, 10, 20, 39)
SCENARIOS = ("smooth", "shocks")


def simulate(rng, scenario):
    eps = rng.standard_normal(T)
    z = np.empty(T)
    z[0] = np.log(rng.uniform(1.0, 6.0))
    for t in range(1, T):
        z[t] = z[t - 1] + MU + SIGMA * eps[t]
        if scenario == "shocks" and t in SHOCKS:
            z[t] += SHOCKS[t]
    return np.exp(z)


def grid_gap(base_rows, y, start=A.INNER_MIN_TRAIN_YEARS):
    """Optimistic minus honest inner RMSE for every outer origin, for one sub-grid."""
    honest_start = start + 1
    n = y.size
    honest = np.full(n, np.nan)
    for j in range(honest_start, n):
        scores = A._history_rmse(base_rows, y, start, j)
        honest[j] = base_rows[int(np.argmin(scores)), j]
    gaps, opts, hons = [], [], []
    for t in range(A.INITIAL_TRAIN_YEARS, n):
        opt = float(np.min(A._history_rmse(base_rows, y, honest_start, t)))
        hon = float(np.sqrt(np.mean((y[honest_start:t] - honest[honest_start:t]) ** 2)))
        gaps.append(opt - hon); opts.append(opt); hons.append(hon)
    return np.mean(gaps), np.mean(opts), np.mean(hons)


def one_replication(args):
    scenario, rep = args
    rng = np.random.default_rng(20260 + 100000 * SCENARIOS.index(scenario) + rep)
    y = simulate(rng, scenario)
    years = np.arange(1995, 1995 + T)
    x = years.astype(float) - 1995.0
    base, _, _ = A.prequential_base_predictions(x, y)
    rows = {"grid": [], "acc": [], "sel": []}
    for kernel in ("Gaussian LWLR", "Exponential LWLR"):
        for g in GRID_SIZES:
            idx = np.unique(np.round(np.linspace(0, len(A.BANDWIDTH_GRID) - 1, g)).astype(int))
            gap, opt, hon = grid_gap(base[kernel][idx], y)
            rows["grid"].append({"Scenario": scenario, "Rep": rep, "Kernel": kernel, "Grid_size": len(idx),
                                 "Gap": gap, "Optimistic": opt, "Honest": hon})
    roll = A.rolling_origin_predictions(years, y)
    preds = dict(roll.predictions)
    preds["Adaptive (honest)"] = roll.adaptive_predictions
    preds["Adaptive (optimistic)"] = roll.optimistic_adaptive_predictions
    for m, p in preds.items():
        met = A.regression_metrics(roll.actual, p, roll.mase_scales, roll.rmsse_scales)
        rows["acc"].append({"Scenario": scenario, "Rep": rep, "Method": m, "RMSE": met["RMSE"], "MASE": met["MASE"]})
    for name, chosen in (("Adaptive (honest)", roll.selected_models), ("Adaptive (optimistic)", roll.optimistic_selected_models)):
        chosen = np.asarray(chosen, dtype=object)
        rows["sel"].append({"Scenario": scenario, "Rep": rep, "Selector": name,
                            "LWLR_share": float(np.isin(chosen, ["Gaussian LWLR", "Exponential LWLR"]).mean()),
                            "Baseline_share": float(np.isin(chosen, ["Naive", "Drift"]).mean())})
    return rows


def corridor_experiment(reps=200_000, n_cal=8, n_diag=7, n_large=200, seed=2027):
    """Vectorised check of Proposition 5 with the same corridor rule as the analysis code."""
    from scipy.stats import norm
    rng = np.random.default_rng(seed)
    rows = []
    lambdas = A.MAD_LAMBDAS
    deltas = (0.0, 1.0, 2.0, 3.0, 4.0)
    for label, n_past, n_eval in (("finite (8 calibration + 7 diagnostic)", n_cal, n_diag), ("large sample (200 past)", n_large, 1)):
        e = rng.standard_normal((reps, n_past + n_eval))
        for lam in lambdas:
            near = []
            below_shift = {d: [] for d in deltas}
            for t in range(n_past, n_past + n_eval):
                past = e[:, :t]
                c = np.median(past, axis=1)
                mad = A.MAD_SCALE * np.median(np.abs(past - c[:, None]), axis=1)
                near.append(np.abs(e[:, t] - c) <= lam * mad)
                if t == n_past:
                    for d in deltas:
                        below_shift[d].append((e[:, t] - d) < c - lam * mad)
            rows.append({"Setting": label, "Lambda": lam, "Quantity": "Coverage (Near)",
                         "Delta_sigma": np.nan, "Simulated": float(np.mean(near)),
                         "Theoretical_limit": float(2 * norm.cdf(lam) - 1)})
            for d in deltas:
                rows.append({"Setting": label, "Lambda": lam, "Quantity": "P(Below | shift)",
                             "Delta_sigma": d, "Simulated": float(np.mean(below_shift[d])),
                             "Theoretical_limit": float(norm.cdf(d - lam))})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--output-dir", type=Path, default=HERE / "outputs-00" / "simulation")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--corridor-only", action="store_true", help="run only the Proposition 5 corridor experiment")
    args = ap.parse_args()
    out = args.output_dir; out.mkdir(parents=True, exist_ok=True)
    corridor = corridor_experiment()
    corridor.to_csv(out / "mc_corridor.csv", index=False, encoding="utf-8-sig")
    with pd.option_context("display.width", 200):
        print(corridor.round(4).to_string(index=False))
    if args.corridor_only:
        return
    jobs = [(s, r) for s in SCENARIOS for r in range(args.reps)]
    with Pool(args.workers) as pool:
        results = pool.map(one_replication, jobs, chunksize=4)
    grid = pd.DataFrame([g for res in results for g in res["grid"]])
    acc = pd.DataFrame([g for res in results for g in res["acc"]])
    sel = pd.DataFrame([g for res in results for g in res["sel"]])

    grid_sum = (grid.groupby(["Scenario", "Kernel", "Grid_size"])
                .agg(Mean_gap=("Gap", "mean"), SD_gap=("Gap", "std"),
                     Share_negative=("Gap", lambda g: float((g < 0).mean())),
                     Mean_optimistic=("Optimistic", "mean"), Mean_honest=("Honest", "mean"))
                .reset_index())
    grid_sum["SE_gap"] = grid_sum["SD_gap"] / np.sqrt(args.reps)
    naive = acc[acc.Method == "Naive"].set_index(["Scenario", "Rep"])["RMSE"]
    acc["Rel_RMSE"] = acc["RMSE"] / acc.set_index(["Scenario", "Rep"]).index.map(naive).to_numpy()
    acc["Rank_RMSE"] = acc.groupby(["Scenario", "Rep"])["RMSE"].rank()
    acc_sum = (acc.groupby(["Scenario", "Method"])
               .agg(Mean_rel_RMSE=("Rel_RMSE", "mean"), Median_rel_RMSE=("Rel_RMSE", "median"),
                    Share_beats_Naive=("Rel_RMSE", lambda r: float((r < 1).mean())),
                    Mean_MASE=("MASE", "mean"), Mean_rank_RMSE=("Rank_RMSE", "mean"))
               .reset_index().sort_values(["Scenario", "Mean_rank_RMSE"]))
    sel_sum = sel.groupby(["Scenario", "Selector"]).agg(LWLR_share=("LWLR_share", "mean"),
                                                        Baseline_share=("Baseline_share", "mean")).reset_index()
    design = pd.DataFrame([
        ("Replications per scenario", args.reps), ("Series length", T), ("Drift mu (log)", MU),
        ("Innovation sigma (log)", SIGMA), ("Shock positions (index: size)", "; ".join(f"{1995+k}: {v:+.2f}" for k, v in SHOCKS.items())),
        ("Protocol", A.TEMPORAL_PROTOCOL), ("Bandwidth grid", f"{A.BANDWIDTH_GRID[0]:.2f}-{A.BANDWIDTH_GRID[-1]:.2f} ({len(A.BANDWIDTH_GRID)})"),
        ("Seeds", "20260 + 100000*scenario_index + replication")], columns=["Parameter", "Value"])
    for name, df in (("mc_grid_bias", grid_sum), ("mc_accuracy", acc_sum), ("mc_selection", sel_sum), ("mc_design", design)):
        df.to_csv(out / f"{name}.csv", index=False, encoding="utf-8-sig")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(grid_sum.round(4)); print(acc_sum.round(3)); print(sel_sum.round(3))


if __name__ == "__main__":
    main()
