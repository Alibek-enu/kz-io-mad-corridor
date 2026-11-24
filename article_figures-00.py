"""Article-level summary figures built only from the CSV tables in outputs-00.

Usage:  python article_figures-00.py <outputs-00/tables> <output figure folder>
Produces (PNG 300 dpi + PDF):
  fig_workflow        research workflow (9-8-7 protocol and analysis stages)
  fig_mase_heatmap    series x method MASE matrix (scale-free accuracy)
  fig_selection       adaptive selection frequency, honest vs optimistic inner score
  fig_direction_map   MAD direction classes by series and diagnostic year
  fig_regime_map      K-means error regimes by series and diagnostic year + silhouette
Palette: Okabe-Ito (colour-blind safe).
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyBboxPatch, Patch

TAB = Path(sys.argv[1])
OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)

OI = {"black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
      "yellow": "#F0E442", "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7",
      "grey": "#BBBBBB"}
SECTORS = ["AGR", "MNG", "MNF", "SRV", "CST", "TRD", "TRN"]
SERIES = [f"{s}_{i}" for s in SECTORS for i in ("Buy", "Sell")]
METHODS = ["Naive", "Drift", "Linear", "Polynomial-2", "Exponential", "Logarithmic", "Power",
           "ETS", "Gaussian LWLR", "Exponential LWLR", "Adaptive rolling selector"]
plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans"})


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


# 1. Workflow -----------------------------------------------------------------
def workflow():
    fig, ax = plt.subplots(figsize=(12, 6.2))
    ax.set_xlim(0, 24); ax.set_ylim(0, 12.4); ax.axis("off")
    # timeline
    spans = [(1995, 2003, OI["sky"], "Initial training\n1995–2003 (N = 9)"),
             (2004, 2011, OI["orange"], "Residual calibration\n2004–2011 (N = 8)"),
             (2012, 2018, OI["vermillion"], "Dynamic diagnostics\n2012–2018 (N = 7)")]
    for a, b, c, label in spans:
        x0 = a - 1995; w = b - a + 1
        ax.add_patch(FancyBboxPatch((x0, 10.3), w - 0.08, 1.2, boxstyle="round,pad=0.02",
                                    fc=c, ec="none", alpha=0.85))
        ax.text(x0 + w / 2, 10.9, label, ha="center", va="center", fontsize=9.5, color="black")
    ax.text(12, 12.1, "9–8–7 temporal evaluation protocol (24 annual observations per series)",
            ha="center", fontsize=11, weight="bold")
    boxes = [
        (0.2, 6.6, "OECD ICIO 2021\nKazakhstan, 45 ISIC Rev.4\nindustries → 7 sectors\nBuy and Sell, 14 series"),
        (4.9, 6.6, "Expanding-window\nrolling origin\n210 one-step\nout-of-sample forecasts"),
        (9.6, 6.6, "10 candidate methods\nNaive, Drift, 5 global\nregressions, ETS,\nGaussian/Exp. LWLR"),
        (14.3, 6.6, "Nested inner validation\n(bandwidth, ETS variant)\nhonest prequential\nadaptive selector"),
        (19.0, 6.6, "Accuracy and inference\nRMSE, MAE, MASE, RMSSE\nFriedman, Wilcoxon–Holm\nbootstrap CI"),
        (2.6, 1.4, "Sequential MAD\ndiagnostic corridor\n(calibration 2004–2011)"),
        (7.6, 1.4, "Below / Near / Above\ntrend classes\n2012–2018 (98 cases)"),
        (12.6, 1.4, "Pooled K-means on\nnormalized |errors|\nerror regimes, silhouette"),
        (17.6, 1.4, "Ablation, disruption\nwindows, λ and protocol\nsensitivity, metadata"),
    ]
    for x, y, t in boxes:
        ax.add_patch(FancyBboxPatch((x, y), 4.4, 3.0, boxstyle="round,pad=0.08",
                                    fc="white", ec=OI["blue"], lw=1.6))
        ax.text(x + 2.2, y + 1.5, t, ha="center", va="center", fontsize=9.2)
    arrow = dict(arrowstyle="-|>", color="0.25", lw=1.4)
    for x in (4.6, 9.3, 14.0, 18.7):
        ax.annotate("", xy=(x + 0.3, 8.1), xytext=(x, 8.1), arrowprops=arrow)
    for x in (7.0, 12.0, 17.0):
        ax.annotate("", xy=(x + 0.6, 2.9), xytext=(x, 2.9), arrowprops=arrow)
    ax.annotate("", xy=(4.8, 4.5), xytext=(16.5, 6.5),
                arrowprops=dict(arrowstyle="-|>", color="0.25", lw=1.4, connectionstyle="arc3,rad=-0.15"))
    ax.text(10.6, 5.25, "adaptive out-of-sample residuals", fontsize=9, ha="center", style="italic")
    save(fig, "fig_workflow")


# 2. MASE heatmap ---------------------------------------------------------------
def mase_heatmap():
    m = pd.read_csv(TAB / "rolling_metrics.csv")
    p = m.pivot(index="Series", columns="Model", values="MASE").loc[SERIES, METHODS]
    fig, ax = plt.subplots(figsize=(12.5, 7))
    im = ax.imshow(p.to_numpy(), aspect="auto", cmap="viridis_r")
    ax.set_xticks(range(len(METHODS)), [x.replace("Adaptive rolling selector", "Adaptive selector")
                                       for x in METHODS], rotation=40, ha="right")
    ax.set_yticks(range(len(SERIES)), SERIES)
    best = p.to_numpy().argmin(axis=1)
    for i in range(p.shape[0]):
        for j in range(p.shape[1]):
            v = p.iat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                    color="white" if v > np.nanpercentile(p.to_numpy(), 60) else "black",
                    weight="bold" if j == best[i] else "normal")
        ax.add_patch(plt.Rectangle((best[i] - 0.5, i - 0.5), 1, 1, fill=False, ec=OI["vermillion"], lw=2))
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="MASE (lower is better)")
    save(fig, "fig_mase_heatmap")


# 3. Selection frequency ----------------------------------------------------------
def selection():
    s = pd.read_csv(TAB / "adaptive_selection_frequency.csv")
    x = np.arange(len(s)); w = 0.4
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - w / 2, s["Selections"], w, color=OI["blue"], label="Honest prequential inner score (primary)")
    ax.bar(x + w / 2, s["Optimistic_Selector_Selections_Ablation"], w, color=OI["orange"],
           label="Optimistic min-over-grid inner score (ablation)")
    ax.set_xticks(x, s["Model"], rotation=35, ha="right")
    ax.set_ylabel("Number of outer origins selected (of 210)")
    ax.grid(axis="y", alpha=0.3); ax.legend()
    save(fig, "fig_selection")


# 4. Direction map -----------------------------------------------------------------
def direction_map():
    y = pd.read_csv(TAB / "year_level_results.csv")
    d = y[y["Direction_Class"] != "Calibration"]
    code = {"Below trend": 0, "Near trend": 1, "Above trend": 2}
    p = d.pivot(index="Series", columns="Year", values="Direction_Class").loc[SERIES].replace(code)
    years = list(p.columns)
    cmap = ListedColormap([OI["blue"], "#E8E8E8", OI["vermillion"]])
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.2), gridspec_kw={"width_ratios": [2.2, 1]})
    ax = axes[0]
    ax.imshow(p.to_numpy(float), aspect="auto", cmap=cmap, vmin=-0.5, vmax=2.5)
    ax.set_xticks(range(len(years)), years); ax.set_yticks(range(len(SERIES)), SERIES)
    for i in range(p.shape[0]):
        for j in range(p.shape[1]):
            v = int(p.iat[i, j]); ax.text(j, i, "▼▪▲"[v], ha="center", va="center",
                                         color="white" if v != 1 else "0.35", fontsize=10)
    ax.set_xlabel("Diagnostic year"); ax.set_title("(a) Direction class, λ = 1.5")
    ax.legend(handles=[Patch(color=OI["blue"], label="Below trend"), Patch(color="#E8E8E8", label="Near trend"),
                       Patch(color=OI["vermillion"], label="Above trend")],
              loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=False)
    ax = axes[1]
    cnt = pd.crosstab(d["Year"], d["Direction_Class"]).reindex(columns=list(code), fill_value=0)
    bottom = np.zeros(len(cnt))
    for name, c in zip(code, [OI["blue"], "#BDBDBD", OI["vermillion"]]):
        ax.bar(cnt.index, cnt[name], bottom=bottom, color=c, label=name); bottom += cnt[name].to_numpy()
    ax.set_ylabel("Number of series (of 14)"); ax.set_xticks(cnt.index, cnt.index, rotation=45)
    ax.set_title("(b) Pooled counts by year"); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save(fig, "fig_direction_map")


# 5. Regime map ----------------------------------------------------------------------
def regime_map():
    y = pd.read_csv(TAB / "year_level_results.csv")
    d = y.dropna(subset=["Error_Regime"])
    code = {"Low-error regime": 0, "Medium-error regime": 1, "High-error regime": 2}
    p = d.pivot(index="Series", columns="Year", values="Error_Regime").loc[SERIES].replace(code)
    z = d.pivot(index="Series", columns="Year", values="Normalized_Absolute_Error").loc[SERIES]
    k = pd.read_csv(TAB / "kmeans_period_sensitivity.csv")
    cmap = ListedColormap([OI["sky"], OI["yellow"], OI["vermillion"]])
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.2), gridspec_kw={"width_ratios": [2.2, 1]})
    ax = axes[0]
    ax.imshow(p.to_numpy(float), aspect="auto", cmap=cmap, vmin=-0.5, vmax=2.5)
    ax.set_xticks(range(p.shape[1]), p.columns); ax.set_yticks(range(len(SERIES)), SERIES)
    for i in range(p.shape[0]):
        for j in range(p.shape[1]):
            ax.text(j, i, f"{z.iat[i, j]:.2f}", ha="center", va="center", fontsize=7.5)
    ax.set_xlabel("Diagnostic year"); ax.set_title("(a) K-means regime (k = 3); cell = normalized |error|")
    ax.legend(handles=[Patch(color=c, label=n) for n, c in zip(code, [OI["sky"], OI["yellow"], OI["vermillion"]])],
              loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=False)
    ax = axes[1]
    for period, c, mk in (("Diagnostic", OI["blue"], "o"), ("Full rolling period (sensitivity)", OI["orange"], "s")):
        t = k[k["Period"] == period]
        ax.plot(t["k"], t["Silhouette"], marker=mk, color=c, lw=2,
                label="2012–2018 (N = 98)" if period == "Diagnostic" else "2004–2018 (N = 210)")
    ax.set_xticks([2, 3, 4]); ax.set_xlabel("Number of clusters k"); ax.set_ylabel("Silhouette coefficient")
    ax.set_title("(b) Silhouette by k"); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    save(fig, "fig_regime_map")


# 6. Monte Carlo ------------------------------------------------------------------------
def monte_carlo():
    sim = TAB.parent / "simulation"
    if not (sim / "mc_grid_bias.csv").exists():
        print("simulation outputs not found; fig_mc skipped"); return
    g = pd.read_csv(sim / "mc_grid_bias.csv"); a = pd.read_csv(sim / "mc_accuracy.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1, 1.25]})
    ax = axes[0]
    styles = {("smooth", "Gaussian LWLR"): (OI["blue"], "o", "-"), ("smooth", "Exponential LWLR"): (OI["sky"], "s", "-"),
              ("shocks", "Gaussian LWLR"): (OI["vermillion"], "o", "--"), ("shocks", "Exponential LWLR"): (OI["orange"], "s", "--")}
    for (sc, ke), (c, mk, ls) in styles.items():
        t = g[(g.Scenario == sc) & (g.Kernel == ke)].sort_values("Grid_size")
        ax.errorbar(t.Grid_size, t.Mean_gap, yerr=1.96 * t.SE_gap, color=c, marker=mk, ls=ls, capsize=3,
                    label=f"{ke.replace(' LWLR','')} kernel, {sc}")
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set_xscale("log"); ax.minorticks_off(); ax.set_xticks([2, 5, 10, 20, 39], ["2", "5", "10", "20", "39"])
    ax.set_xlabel("Bandwidth grid size |H|"); ax.set_ylabel("Mean optimistic − honest inner RMSE")
    ax.set_title("(a) Optimism of the min-over-grid inner score"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax = axes[1]
    methods = ["Naive", "Drift", "ETS", "Exponential LWLR", "Gaussian LWLR", "Adaptive (honest)", "Adaptive (optimistic)", "Linear", "Polynomial-2"]
    yv = np.arange(len(methods)); h = 0.38
    for k, (sc, c) in enumerate((("smooth", OI["blue"]), ("shocks", OI["vermillion"]))):
        t = a[a.Scenario == sc].set_index("Method").reindex(methods)
        ax.barh(yv + (k - 0.5) * h, t.Mean_rel_RMSE, h, color=c, label=sc)
    ax.axvline(1, color="0.3", lw=1, ls=":"); ax.set_xlim(0.8, None)
    ax.set_yticks(yv, methods); ax.invert_yaxis(); ax.set_xlabel("Mean RMSE relative to Naive")
    ax.set_title("(b) Out-of-sample accuracy in simulation"); ax.legend(); ax.grid(axis="x", alpha=0.3)
    fig.tight_layout(); save(fig, "fig_mc")


workflow(); mase_heatmap(); selection(); direction_map(); regime_map(); monte_carlo()
print("figures written to", OUT)
