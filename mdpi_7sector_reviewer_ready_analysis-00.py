"""Leakage-free regression and diagnostic analysis for the 7-sector panel.

Expected input
--------------
Workbook: prepared_7sector_buy_sell_panel.xlsx
Sheet:    Panel_Long
Columns:  Year, Sector_Code, Sector, Buy_USD_bn, Sell_USD_bn

Methodological design (primary configuration: 9-8-7 temporal evaluation protocol)
-------------------------------------------------------------------------------
Stage                  Years       N   Purpose
Initial training       1995-2003   9   first outer training window
Residual calibration   2004-2011   8   out-of-sample forecasts; their residuals
                                       calibrate the sequential MAD scale
Dynamic diagnostics    2012-2018   7   MAD coverage/width and error regimes

1. Ten fixed methods are compared: Naive, Drift, five global regression forms,
   a non-seasonal exponential-smoothing benchmark (SES / Holt / damped Holt,
   parameters estimated by one-step SSE with fixed initial states; reported as
   "ETS"), Gaussian-kernel LWLR and exponential-kernel LWLR. Gaussian and
   exponential are temporal weighting kernels inside LWLR, not separate
   regression models.
2. Out-of-sample accuracy uses expanding-window rolling-origin validation.
   Under 9-8-7, 1995-2003 predict 2004, 1995-2004 predict 2005, ..., giving
   15 one-step forecasts per series (2004-2018) and 14 x 15 = 210 in total.
   Future observations never enter estimation, tuning or selection
   (leakage-free evaluation design).
3. LWLR bandwidth and the ETS variant are tuned at every origin by inner
   expanding validation on the current training sample only.
4. Adaptive rolling selector. At every outer origin t the candidate with the
   lowest inner RMSE is chosen. Inner errors are *honest prequential* errors:
   for a tuned candidate (LWLR, ETS) the inner one-step error at inner endpoint
   e uses the bandwidth/variant that was itself selected on data before e.
   Hence tuned and untuned candidates are scored on the same footing and the
   optimistic min-over-grid bias of the former selector is removed. The former
   ("optimistic") selector is retained only as an ablation.
5. Sequential MAD diagnostic corridor (not a prediction interval): the first 8
   adaptive residuals (2004-2011) calibrate; for 2012-2018 every boundary uses
   strictly earlier residuals only.
6. K-means error regimes (unsupervised, not confidence classes): primary fit on
   the pooled 2012-2018 diagnostic errors (14 x 7 = 98). Each series' absolute
   errors are scaled by the median absolute error of its own calibration
   period (2004-2011), so the scale is fixed before the diagnostic period.
   The full 2004-2018 pool (210) is reported as a sensitivity analysis.
7. MASE and RMSSE use origin-specific in-sample scales: each outer error is
   divided by the mean absolute (squared) first difference of the training
   sample available at that origin.
8. Statistical inference is run for RMSE, MASE and RMSSE. The seven-sector
   confirmatory comparison uses a small pre-specified comparator family
   (Naive, Drift, ETS) because with N=7 blocks the smallest attainable
   two-sided Wilcoxon p-value is 2/2^7 = 0.0156; with ten Holm-adjusted
   comparisons significance would be unattainable by construction.

The 5-5-14 protocol is available only as a sensitivity run
(--protocol 5-5-14); it writes to a separate output folder.

The script prints complete tables and shows all figures on screen. With
--save-results it also creates article-ready CSV tables and PNG/PDF figures.

Usage
-----
python mdpi_7sector_reviewer_ready_analysis-00.py --save-results --no-show
python mdpi_7sector_reviewer_ready_analysis-00.py --protocol 5-5-14 --save-results --no-show

Dependencies
------------
numpy, pandas, scipy, scikit-learn, matplotlib, openpyxl
"""

from __future__ import annotations

import argparse
import hashlib
import math
import platform
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import openpyxl
import pandas as pd
from matplotlib.lines import Line2D
from scipy.interpolate import PchipInterpolator
import scipy
import sklearn
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import binomtest, chi2, friedmanchisquare, rankdata, shapiro, wilcoxon
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score


# =============================================================================
# 1. Reproducible configuration
# =============================================================================
RANDOM_STATE = 42
INPUT_FILE = Path(__file__).resolve().parent / "prepared_7sector_buy_sell_panel.xlsx"
SHEET_NAME = "Panel_Long"
OUTPUT_DIRS = {
    "9-8-7": Path(__file__).resolve().parent / "outputs-00",
    "5-5-14": Path(__file__).resolve().parent / "outputs-00_protocol-5-5-14",
}

# Temporal evaluation protocols: (initial training years, MAD calibration forecasts).
# The remaining 24 - training - calibration years form the diagnostic period.
PROTOCOLS = {"9-8-7": (9, 8), "5-5-14": (5, 5)}
TEMPORAL_PROTOCOL = "9-8-7"
INITIAL_TRAIN_YEARS = 9       # 1995-2003 -> first outer forecast for 2004
INNER_MIN_TRAIN_YEARS = 3     # minimum history in inner (nested) validation
MAD_CALIBRATION_FORECASTS = 8 # 2004-2011 calibrate; diagnostics 2012-2018
MAD_MULTIPLIER = 1.5
MAD_LAMBDAS = (1.0, 1.5, 2.0, 2.5, 3.0)
MAD_SCALE = 1.4826
BOOTSTRAP_REPETITIONS = 5000
KMEANS_N_INIT = 100
KMEANS_STABILITY_REPETITIONS = 100
# Pre-specified (not test-detected) disruption periods; these are not formal
# structural breaks. Windows are clipped to the outer forecast years.
DISRUPTION_WINDOWS = {
    "Pre-2008": (2000, 2007),
    "Global-crisis": (2008, 2009),
    "Post-crisis": (2010, 2013),
    "2014-2016-shock": (2014, 2016),
    "Recent": (2017, 2018),
}

# Bandwidth in year units. A deterministic grid makes the analysis reproducible.
# 0.50, 0.75, ..., 10.00 (step 0.25). Below 0.5 the Gaussian weight of the
# second-nearest year falls under the numerical ridge and the local slope
# degenerates, so 0.5 is the smallest numerically meaningful bandwidth.
BANDWIDTH_GRID = np.linspace(0.5, 10.0, 39)

COMPARISON_METRICS = ("RMSE", "MASE", "RMSSE")
# Pre-specified comparator family for the seven-sector confirmatory tests.
CONFIRMATORY_COMPARATORS = ("Naive", "Drift", "ETS")

SECTOR_ORDER = ["AGR", "MNG", "MNF", "SRV", "CST", "TRD", "TRN"]
SECTOR_NAMES = {
    "AGR": "Agriculture",
    "MNG": "Mining",
    "MNF": "Manufacturing",
    "SRV": "Services",
    "CST": "Construction",
    "TRD": "Trade",
    "TRN": "Transportation",
}
VALUE_COLUMNS = ["Buy_USD_bn", "Sell_USD_bn"]
VALUE_LABELS = {"Buy_USD_bn": "Buy", "Sell_USD_bn": "Sell"}

MODEL_ORDER = [
    "Naive",
    "Drift",
    "Linear",
    "Polynomial-2",
    "Exponential",
    "Logarithmic",
    "Power",
    "ETS",
    "Gaussian LWLR",
    "Exponential LWLR",
]
ADAPTIVE_METHOD = "Adaptive rolling selector"
METHOD_ORDER = MODEL_ORDER + [ADAPTIVE_METHOD]

MODEL_COLORS = {
    "Naive": "#616161",
    "Drift": "#9E9E9E",
    "Linear": "#6A1B9A",
    "Polynomial-2": "#00838F",
    "Exponential": "#E65100",
    "Logarithmic": "#5D4037",
    "Power": "#7CB342",
    "ETS": "#EC407A",
    "Gaussian LWLR": "#1565C0",
    "Exponential LWLR": "#D32F2F",
    ADAPTIVE_METHOD: "#000000",
}

DIRECTION_ORDER = ["Calibration", "Below trend", "Near trend", "Above trend"]
DIRECTION_COLORS = {
    "Calibration": "#757575",
    "Below trend": "#1565C0",
    "Near trend": "#2E7D32",
    "Above trend": "#D32F2F",
}

ERROR_REGIME_ORDER = ["Low-error regime", "Medium-error regime", "High-error regime"]
ERROR_REGIME_COLORS = {
    "Low-error regime": "#2E7D32",
    "Medium-error regime": "#F9A825",
    "High-error regime": "#D32F2F",
}

plt.rcParams.update({
    "figure.max_open_warning": 100,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
})


# =============================================================================
# 2. Data structures
# =============================================================================
@dataclass
class RollingResult:
    years: np.ndarray
    actual: np.ndarray
    predictions: dict[str, np.ndarray]
    bandwidths: dict[str, np.ndarray]
    adaptive_predictions: np.ndarray
    selected_models: np.ndarray
    inner_scores: dict[str, np.ndarray]
    optimistic_adaptive_predictions: np.ndarray
    optimistic_selected_models: np.ndarray
    optimistic_inner_scores: dict[str, np.ndarray]
    ets_variants: np.ndarray
    mase_scales: np.ndarray
    rmsse_scales: np.ndarray
    lwlr_inner_fits: int
    all_inner_fits: int


# =============================================================================
# 3. Input and validation
# =============================================================================
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Leakage-free rolling analysis of the prepared 7-sector panel."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Path to prepared_7sector_buy_sell_panel.xlsx",
    )
    parser.add_argument("--sheet", default=SHEET_NAME, help="Panel sheet name")
    parser.add_argument(
        "--save-results", action="store_true",
        help="Save reproducibility tables and article figures.",
    )
    parser.add_argument(
        "--protocol", choices=sorted(PROTOCOLS), default="9-8-7",
        help="Temporal evaluation protocol (9-8-7 primary; 5-5-14 sensitivity).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Directory used only with --save-results "
             "(default: outputs-00 for 9-8-7, outputs-00_protocol-5-5-14 for 5-5-14).",
    )
    parser.add_argument(
        "--no-show", action="store_true",
        help="Do not open interactive figures (useful for automated reproduction).",
    )
    return parser.parse_args()


def resolve_input_path(cli_path: Path | None) -> Path:
    if cli_path is not None:
        path = cli_path.expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Input file does not exist: {path}")
        return path

    candidates = [INPUT_FILE, Path.cwd() / INPUT_FILE.name]
    for path in candidates:
        if path.exists():
            return path.resolve()
    raise FileNotFoundError(
        "prepared_7sector_buy_sell_panel.xlsx was not found. Place it beside "
        "this script or run: python mdpi_7sector_reviewer_ready_analysis-00.py "
        "--input D:/path/prepared_7sector_buy_sell_panel.xlsx"
    )


def load_and_validate_panel(path: Path, sheet_name: str) -> pd.DataFrame:
    panel = pd.read_excel(path, sheet_name=sheet_name, engine="openpyxl")
    required = {
        "Observation_ID", "Year", "Sector_Code", "Sector",
        "Buy_USD_bn", "Sell_USD_bn",
    }
    missing_columns = required.difference(panel.columns)
    if missing_columns:
        raise ValueError(f"Missing columns: {sorted(missing_columns)}")

    panel = panel[[
        "Observation_ID", "Year", "Sector_Code", "Sector",
        "Buy_USD_bn", "Sell_USD_bn",
    ]].copy()
    panel["Sector_Code"] = panel["Sector_Code"].astype(str).str.strip()
    panel["Sector"] = panel["Sector"].astype(str).str.strip()
    panel["Year"] = pd.to_numeric(panel["Year"], errors="raise").astype(int)
    for column in VALUE_COLUMNS:
        panel[column] = pd.to_numeric(panel[column], errors="raise")

    if panel.duplicated(["Sector_Code", "Year"]).any():
        duplicates = panel.loc[
            panel.duplicated(["Sector_Code", "Year"], keep=False),
            ["Sector_Code", "Year"],
        ]
        raise ValueError(f"Duplicate sector-year keys found:\n{duplicates}")

    if len(panel) != 168:
        raise ValueError(f"Expected 168 sector-year rows; found {len(panel)}.")

    expected_years = set(range(1995, 2019))
    observed_sectors = set(panel["Sector_Code"])
    if observed_sectors != set(SECTOR_ORDER):
        raise ValueError(
            f"Expected sectors {SECTOR_ORDER}; found {sorted(observed_sectors)}"
        )

    # The first occurrence of each code in the prepared panel must follow the
    # authoritative OECD aggregation order. This rejects panels produced with
    # the former, incorrectly labelled sector sequence.
    observed_order = panel.drop_duplicates("Sector_Code")["Sector_Code"].tolist()
    if observed_order != SECTOR_ORDER:
        raise ValueError(
            "Incorrect sector order in the input panel. "
            f"Expected {SECTOR_ORDER}; found {observed_order}. "
            "Use the corrected prepared_7sector_buy_sell_panel.xlsx file."
        )

    observed_mapping = (
        panel[["Sector_Code", "Sector"]]
        .drop_duplicates()
        .set_index("Sector_Code")["Sector"]
        .to_dict()
    )
    if observed_mapping != SECTOR_NAMES:
        raise ValueError(
            "Sector code/name mapping is inconsistent. "
            f"Expected {SECTOR_NAMES}; found {observed_mapping}."
        )

    for sector in SECTOR_ORDER:
        sector_years = set(panel.loc[panel["Sector_Code"] == sector, "Year"])
        if sector_years != expected_years:
            raise ValueError(f"{sector}: incomplete years {sorted(expected_years-sector_years)}")

    expected_ids = panel["Sector_Code"] + "_" + panel["Year"].astype(str)
    if not panel["Observation_ID"].astype(str).eq(expected_ids).all():
        bad = panel.loc[
            ~panel["Observation_ID"].astype(str).eq(expected_ids),
            ["Observation_ID", "Sector_Code", "Year"],
        ].head(10)
        raise ValueError(f"Invalid Observation_ID values:\n{bad.to_string(index=False)}")

    if panel[VALUE_COLUMNS].isna().any().any():
        raise ValueError("Panel contains missing Buy/Sell values.")
    if (panel[VALUE_COLUMNS] <= 0).any().any():
        raise ValueError("All Buy/Sell values must be positive for log-based models.")

    panel["Sector_Code"] = pd.Categorical(
        panel["Sector_Code"], categories=SECTOR_ORDER, ordered=True
    )
    return panel.sort_values(["Sector_Code", "Year"]).reset_index(drop=True)


def calculate_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 checksum of the exact input workbook bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


# =============================================================================
# 4. Regression models
# =============================================================================
def stable_solve(design: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(design, y, rcond=None)[0]


def global_model_predict(
    model_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_query: np.ndarray,
) -> np.ndarray:
    """Fit one global model on training observations and predict x_query."""
    x_train = np.asarray(x_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    x_query = np.atleast_1d(np.asarray(x_query, dtype=float))

    if model_name == "Linear":
        design = np.column_stack([np.ones(x_train.size), x_train])
        beta = stable_solve(design, y_train)
        return beta[0] + beta[1] * x_query

    if model_name == "Polynomial-2":
        design = np.column_stack([np.ones(x_train.size), x_train, x_train**2])
        beta = stable_solve(design, y_train)
        return beta[0] + beta[1] * x_query + beta[2] * x_query**2

    if model_name == "Exponential":
        design = np.column_stack([np.ones(x_train.size), x_train])
        beta = stable_solve(design, np.log(y_train))
        return np.exp(np.clip(beta[0] + beta[1] * x_query, -700, 700))

    if model_name == "Logarithmic":
        design = np.column_stack([np.ones(x_train.size), np.log1p(x_train)])
        beta = stable_solve(design, y_train)
        return beta[0] + beta[1] * np.log1p(x_query)

    if model_name == "Power":
        design = np.column_stack([np.ones(x_train.size), np.log1p(x_train)])
        beta = stable_solve(design, np.log(y_train))
        return np.exp(np.clip(beta[0] + beta[1] * np.log1p(x_query), -700, 700))

    raise KeyError(f"Unknown global model: {model_name}")


def baseline_predict(
    model_name: str, x_train: np.ndarray, y_train: np.ndarray, x_query: np.ndarray,
) -> np.ndarray:
    """Leakage-free persistence benchmarks required in forecast comparisons."""
    x_query = np.atleast_1d(np.asarray(x_query, dtype=float))
    if model_name == "Naive":
        return np.full(x_query.size, float(y_train[-1]))
    if model_name == "Drift":
        if len(y_train) < 2:
            return np.full(x_query.size, float(y_train[-1]))
        slope = (float(y_train[-1]) - float(y_train[0])) / (
            float(x_train[-1]) - float(x_train[0])
        )
        return float(y_train[-1]) + slope * (x_query - float(x_train[-1]))
    raise KeyError(model_name)


# ETS is a classical time-series benchmark, not one of the five regression
# functions.  It is implemented here without an external statsmodels
# dependency so that the reproducibility package remains lightweight.
ETS_VARIANTS = ("SES", "Holt", "Damped Holt")


def _ets_sse(y: np.ndarray, variant: str, parameters: np.ndarray) -> float:
    """One-step-ahead sum of squared errors for a small non-seasonal ETS family."""
    y = np.asarray(y, dtype=float)
    alpha = float(parameters[0])
    beta = float(parameters[1]) if variant != "SES" else 0.0
    phi = float(parameters[2]) if variant == "Damped Holt" else 1.0
    level = float(y[0])
    trend = float(y[1] - y[0]) if y.size > 1 else 0.0
    errors = []
    for observation in y[1:]:
        forecast = level if variant == "SES" else level + phi * trend
        errors.append(float(observation - forecast))
        if variant == "SES":
            level = alpha * observation + (1.0 - alpha) * level
        else:
            previous_level = level
            level = alpha * observation + (1.0 - alpha) * (level + phi * trend)
            trend = beta * (level - previous_level) + (1.0 - beta) * phi * trend
    return float(np.sum(np.square(errors)))


def ets_predict_one(y_train: np.ndarray, variant: str) -> float:
    """Estimate a low-parameter ETS variant and return its one-step forecast."""
    y = np.asarray(y_train, dtype=float)
    if y.size == 1:
        return float(y[-1])
    if variant == "SES":
        fit = minimize_scalar(
            lambda a: _ets_sse(y, variant, np.array([a])),
            bounds=(1e-4, 0.9999), method="bounded",
        )
        parameters = np.array([float(fit.x)])
    elif variant == "Holt":
        fit = minimize(
            lambda p: _ets_sse(y, variant, p), x0=np.array([0.4, 0.2]),
            bounds=((1e-4, 0.9999), (1e-4, 0.9999)), method="L-BFGS-B",
        )
        parameters = np.asarray(fit.x, dtype=float)
    elif variant == "Damped Holt":
        fit = minimize(
            lambda p: _ets_sse(y, variant, p), x0=np.array([0.4, 0.2, 0.9]),
            bounds=((1e-4, 0.9999), (1e-4, 0.9999), (0.80, 0.9999)),
            method="L-BFGS-B",
        )
        parameters = np.asarray(fit.x, dtype=float)
    else:
        raise KeyError(variant)

    alpha = float(parameters[0])
    beta = float(parameters[1]) if variant != "SES" else 0.0
    phi = float(parameters[2]) if variant == "Damped Holt" else 1.0
    level = float(y[0])
    trend = float(y[1] - y[0])
    for observation in y[1:]:
        if variant == "SES":
            level = alpha * observation + (1.0 - alpha) * level
        else:
            previous_level = level
            level = alpha * observation + (1.0 - alpha) * (level + phi * trend)
            trend = beta * (level - previous_level) + (1.0 - beta) * phi * trend
    return float(level if variant == "SES" else level + phi * trend)


def select_ets_variant_nested(y_train: np.ndarray) -> tuple[str, float, int]:
    """Select SES/Holt/damped Holt by training-only inner rolling validation."""
    best_variant, best_rmse, fit_count = ETS_VARIANTS[0], math.inf, 0
    for variant in ETS_VARIANTS:
        errors = []
        for endpoint in range(INNER_MIN_TRAIN_YEARS, len(y_train)):
            prediction = ets_predict_one(y_train[:endpoint], variant)
            errors.append(float(y_train[endpoint] - prediction))
            fit_count += 1
        rmse = float(np.sqrt(np.mean(np.square(errors))))
        if rmse < best_rmse:
            best_variant, best_rmse = variant, rmse
    return best_variant, best_rmse, fit_count


def gaussian_kernel(distance: np.ndarray, bandwidth: float) -> np.ndarray:
    return np.exp(-0.5 * (distance / bandwidth) ** 2)


def exponential_kernel(distance: np.ndarray, bandwidth: float) -> np.ndarray:
    return np.exp(-np.abs(distance) / bandwidth)


def lwlr_predict_one(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_query: float,
    bandwidth: float,
    kernel_name: str,
) -> float:
    distance = np.asarray(x_train, dtype=float) - float(x_query)
    if kernel_name == "Gaussian LWLR":
        weights = gaussian_kernel(distance, bandwidth)
    elif kernel_name == "Exponential LWLR":
        weights = exponential_kernel(distance, bandwidth)
    else:
        raise KeyError(kernel_name)
    # WLS is invariant to a common weight scale; normalising to max weight 1
    # makes the tiny ridge term relative rather than absolute.
    weights = weights / float(np.max(weights))

    # Centered local-linear design; intercept is the prediction at x_query.
    design = np.column_stack([np.ones(distance.size), distance])
    weighted_cross = design.T @ (design * weights[:, None])
    weighted_y = design.T @ (weights * np.asarray(y_train, dtype=float))
    ridge = 1e-10 * np.eye(2)
    beta = np.linalg.solve(weighted_cross + ridge, weighted_y)
    return float(beta[0])


def select_lwlr_bandwidth_nested(
    x_train: np.ndarray,
    y_train: np.ndarray,
    kernel_name: str,
) -> tuple[float, float, int]:
    """Inner rolling-origin selection using only the current outer training set."""
    best_h = float(BANDWIDTH_GRID[0])
    best_rmse = math.inf
    fit_count = 0

    for bandwidth in BANDWIDTH_GRID:
        errors = []
        for endpoint in range(INNER_MIN_TRAIN_YEARS, len(x_train)):
            prediction = lwlr_predict_one(
                x_train[:endpoint], y_train[:endpoint], x_train[endpoint],
                float(bandwidth), kernel_name,
            )
            fit_count += 1
            errors.append(y_train[endpoint] - prediction)
        rmse = float(np.sqrt(np.mean(np.square(errors))))
        if rmse < best_rmse:
            best_rmse = rmse
            best_h = float(bandwidth)
    return best_h, best_rmse, fit_count


FIXED_GLOBAL_MODELS = ["Linear", "Polynomial-2", "Exponential", "Logarithmic", "Power"]
UNTUNED_MODELS = ["Naive", "Drift"] + FIXED_GLOBAL_MODELS
TUNED_MODELS = ["ETS", "Gaussian LWLR", "Exponential LWLR"]


def _history_rmse(matrix: np.ndarray, y: np.ndarray, start: int, end: int) -> np.ndarray:
    """Row-wise RMSE of one-step predictions over inner endpoints start..end-1."""
    errors = y[start:end][None, :] - matrix[:, start:end]
    return np.sqrt(np.mean(np.square(errors), axis=1))


def prequential_base_predictions(x: np.ndarray, y: np.ndarray) -> tuple[dict, int, int]:
    """One-step predictions of every base candidate at every admissible endpoint.

    For endpoint k (k >= INNER_MIN_TRAIN_YEARS) a candidate is fitted on
    observations 0..k-1 only and predicts observation k. Because this
    prediction does not depend on which outer origin later uses it, it is
    computed once and reused by every inner and outer validation loop. The
    resulting forecasts are numerically identical to refitting inside each
    loop, but the design is transparent and the cost is linear in the grid.
    """
    n = y.size
    start = INNER_MIN_TRAIN_YEARS
    base: dict = {}
    lwlr_fits = 0
    all_fits = 0
    for model in UNTUNED_MODELS:
        row = np.full(n, np.nan)
        for k in range(start, n):
            if model in ("Naive", "Drift"):
                row[k] = baseline_predict(model, x[:k], y[:k], np.array([x[k]]))[0]
            else:
                row[k] = global_model_predict(model, x[:k], y[:k], np.array([x[k]]))[0]
            all_fits += 1
        base[model] = row
    ets = np.full((len(ETS_VARIANTS), n), np.nan)
    for v, variant in enumerate(ETS_VARIANTS):
        for k in range(start, n):
            ets[v, k] = ets_predict_one(y[:k], variant)
            all_fits += 1
    base["ETS"] = ets
    for kernel in ("Gaussian LWLR", "Exponential LWLR"):
        grid = np.full((BANDWIDTH_GRID.size, n), np.nan)
        for j, bandwidth in enumerate(BANDWIDTH_GRID):
            for k in range(start, n):
                grid[j, k] = lwlr_predict_one(x[:k], y[:k], x[k], float(bandwidth), kernel)
                lwlr_fits += 1
                all_fits += 1
        base[kernel] = grid
    return base, lwlr_fits, all_fits


def tuned_choice(base: dict, model: str, y: np.ndarray, endpoint: int) -> tuple[int, float]:
    """Select ETS variant / LWLR bandwidth using endpoints strictly before `endpoint`.

    Returns the selected row index and its (optimistic) minimum inner RMSE.
    Ties are resolved by the first grid element, as in the original loop.
    """
    scores = _history_rmse(base[model], y, INNER_MIN_TRAIN_YEARS, endpoint)
    index = int(np.argmin(scores))
    return index, float(scores[index])


def honest_prequential_predictions(base: dict, y: np.ndarray) -> dict[str, np.ndarray]:
    """One-step predictions in which tuning itself uses only earlier data.

    Untuned models: available from INNER_MIN_TRAIN_YEARS.
    Tuned models: available from INNER_MIN_TRAIN_YEARS + 1 (tuning needs at
    least one earlier inner endpoint).
    """
    n = y.size
    honest = {model: base[model].copy() for model in UNTUNED_MODELS}
    for model in TUNED_MODELS:
        row = np.full(n, np.nan)
        for endpoint in range(INNER_MIN_TRAIN_YEARS + 1, n):
            index, _ = tuned_choice(base, model, y, endpoint)
            row[endpoint] = base[model][index, endpoint]
        honest[model] = row
    return honest


def rolling_origin_predictions(years: np.ndarray, y: np.ndarray) -> RollingResult:
    x = years.astype(float) - float(years.min())
    forecast_indices = np.arange(INITIAL_TRAIN_YEARS, len(years))
    forecast_years = years[forecast_indices]
    actual = y[forecast_indices]
    size = forecast_indices.size

    base, lwlr_fits, all_fits = prequential_base_predictions(x, y)
    honest = honest_prequential_predictions(base, y)
    honest_start = INNER_MIN_TRAIN_YEARS + 1  # common inner window for all candidates

    predictions = {model: np.empty(size) for model in MODEL_ORDER}
    bandwidths = {"Gaussian LWLR": np.empty(size), "Exponential LWLR": np.empty(size)}
    ets_variants = np.empty(size, dtype=object)
    mase_scales = np.empty(size)
    rmsse_scales = np.empty(size)
    inner_scores = {model: np.empty(size) for model in MODEL_ORDER}
    optimistic_scores = {model: np.empty(size) for model in MODEL_ORDER}
    adaptive_predictions = np.empty(size)
    selected_models = np.empty(size, dtype=object)
    optimistic_predictions = np.empty(size)
    optimistic_selected = np.empty(size, dtype=object)

    for output_index, test_index in enumerate(forecast_indices):
        differences = np.diff(y[:test_index])
        mase_scales[output_index] = float(np.mean(np.abs(differences)))
        rmsse_scales[output_index] = float(np.mean(np.square(differences)))

        # Outer one-step forecasts: fitted/tuned on observations before test_index.
        for model in UNTUNED_MODELS:
            predictions[model][output_index] = base[model][test_index]
            optimistic_scores[model][output_index] = float(_history_rmse(
                base[model][None, :], y, INNER_MIN_TRAIN_YEARS, test_index
            )[0])
        for model in TUNED_MODELS:
            index, optimistic = tuned_choice(base, model, y, test_index)
            predictions[model][output_index] = base[model][index, test_index]
            optimistic_scores[model][output_index] = optimistic
            if model == "ETS":
                ets_variants[output_index] = ETS_VARIANTS[index]
            else:
                bandwidths[model][output_index] = float(BANDWIDTH_GRID[index])

        # Honest inner scores: identical inner window and honest tuning for all.
        for model in MODEL_ORDER:
            errors = y[honest_start:test_index] - honest[model][honest_start:test_index]
            inner_scores[model][output_index] = float(np.sqrt(np.mean(np.square(errors))))

        selected = min(MODEL_ORDER, key=lambda name: inner_scores[name][output_index])
        selected_models[output_index] = selected
        adaptive_predictions[output_index] = predictions[selected][output_index]

        # Ablation only: former selector with optimistic min-over-grid scores.
        legacy = min(MODEL_ORDER, key=lambda name: optimistic_scores[name][output_index])
        optimistic_selected[output_index] = legacy
        optimistic_predictions[output_index] = predictions[legacy][output_index]

    return RollingResult(
        years=forecast_years,
        actual=actual,
        predictions=predictions,
        bandwidths=bandwidths,
        adaptive_predictions=adaptive_predictions,
        selected_models=selected_models,
        inner_scores=inner_scores,
        optimistic_adaptive_predictions=optimistic_predictions,
        optimistic_selected_models=optimistic_selected,
        optimistic_inner_scores=optimistic_scores,
        ets_variants=ets_variants,
        mase_scales=mase_scales,
        rmsse_scales=rmsse_scales,
        lwlr_inner_fits=lwlr_fits,
        all_inner_fits=all_fits,
    )


# =============================================================================
# 5. Metrics, robust corridor and classifications
# =============================================================================
def regression_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    mase_scales: np.ndarray | None = None,
    rmsse_scales: np.ndarray | None = None,
) -> dict[str, float]:
    error = actual - predicted
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error**2)))
    mape = float(np.mean(np.abs(error / actual)) * 100)
    smape = float(np.mean(2 * np.abs(error) / (np.abs(actual) + np.abs(predicted))) * 100)
    ss_res = float(np.sum(error**2))
    ss_tot = float(np.sum((actual - np.mean(actual)) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    mase = np.nan
    rmsse = np.nan
    if mase_scales is not None:
        valid = np.asarray(mase_scales, dtype=float) > np.finfo(float).eps
        mase = float(np.mean(np.abs(error[valid]) / np.asarray(mase_scales)[valid]))
    if rmsse_scales is not None:
        valid = np.asarray(rmsse_scales, dtype=float) > np.finfo(float).eps
        rmsse = float(np.sqrt(np.mean(error[valid] ** 2 / np.asarray(rmsse_scales)[valid])))
    return {
        "MAE": mae, "RMSE": rmse, "MASE": mase, "RMSSE": rmsse,
        "MAPE": mape, "sMAPE": smape, "R2": r2,
    }


def ljung_box_test(residuals: np.ndarray, lags: int = 4) -> tuple[float, float]:
    """Small-sample Ljung-Box portmanteau statistic without extra dependency."""
    values = np.asarray(residuals, dtype=float) - float(np.mean(residuals))
    n = values.size
    denominator = float(np.sum(values**2))
    if denominator <= np.finfo(float).eps:
        return np.nan, np.nan
    effective_lags = min(lags, n - 2)
    correlations = [
        float(np.sum(values[k:] * values[:-k]) / denominator)
        for k in range(1, effective_lags + 1)
    ]
    q = n * (n + 2) * sum(
        correlations[k - 1] ** 2 / (n - k)
        for k in range(1, effective_lags + 1)
    )
    return float(q), float(chi2.sf(q, effective_lags))


def sequential_mad_classification(
    residuals: np.ndarray,
    calibration_forecasts: int | None = None,
    multiplier: float | None = None,
) -> pd.DataFrame:
    # Defaults are read at call time so that --protocol changes take effect.
    if calibration_forecasts is None:
        calibration_forecasts = MAD_CALIBRATION_FORECASTS
    if multiplier is None:
        multiplier = MAD_MULTIPLIER
    n = residuals.size
    classes = np.full(n, "Calibration", dtype=object)
    medians = np.full(n, np.nan)
    mads = np.full(n, np.nan)
    robust_sigmas = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    upper = np.full(n, np.nan)

    for i in range(calibration_forecasts, n):
        history = residuals[:i]  # strictly earlier out-of-sample errors
        median = float(np.median(history))
        mad = float(np.median(np.abs(history - median)))
        robust_sigma = MAD_SCALE * mad
        if robust_sigma <= np.finfo(float).eps:
            # Deterministic fallback for a degenerate early residual history.
            robust_sigma = float(np.std(history, ddof=1)) if history.size > 1 else 0.0
        lo = median - multiplier * robust_sigma
        hi = median + multiplier * robust_sigma

        medians[i], mads[i], robust_sigmas[i] = median, mad, robust_sigma
        lower[i], upper[i] = lo, hi
        if residuals[i] < lo:
            classes[i] = "Below trend"
        elif residuals[i] > hi:
            classes[i] = "Above trend"
        else:
            classes[i] = "Near trend"

    return pd.DataFrame({
        "Residual_Median": medians,
        "MAD": mads,
        "Robust_Sigma": robust_sigmas,
        "Residual_Lower": lower,
        "Residual_Upper": upper,
        "Corridor_Width": upper - lower,
        "Direction_Class": classes,
    })


def ordered_kmeans_classes(values_1d: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values_1d, dtype=float).reshape(-1, 1)
    fitted = KMeans(n_clusters=k, n_init=KMEANS_N_INIT, random_state=RANDOM_STATE)
    raw = fitted.fit_predict(values)
    order = np.argsort(fitted.cluster_centers_.ravel())
    if k == 2:
        names = ["Low-error regime", "High-error regime"]
    elif k == 3:
        names = ERROR_REGIME_ORDER
    else:
        names = [f"Ordered error regime {i + 1}" for i in range(k)]
    mapping = {int(label): names[rank] for rank, label in enumerate(order)}
    return np.array([mapping[int(label)] for label in raw], dtype=object), np.sort(fitted.cluster_centers_.ravel())


def kmeans_k_table(values_1d: np.ndarray, ks=(2, 3, 4)) -> pd.DataFrame:
    values = np.asarray(values_1d, dtype=float).reshape(-1, 1)
    rows = []
    for k in ks:
        candidate = KMeans(n_clusters=k, n_init=KMEANS_N_INIT, random_state=RANDOM_STATE)
        labels = candidate.fit_predict(values)
        rows.append({
            "k": k,
            "N": values.shape[0],
            "Inertia": float(candidate.inertia_),
            "Silhouette": float(silhouette_score(values, labels)),
        })
    return pd.DataFrame(rows)


def kmeans_error_diagnostics(
    absolute_errors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float], float, pd.DataFrame]:
    values = absolute_errors.reshape(-1, 1)
    model = KMeans(n_clusters=3, n_init=KMEANS_N_INIT, random_state=RANDOM_STATE)
    labels = model.fit_predict(values)
    centers = model.cluster_centers_.ravel()
    ordered_raw_labels = np.argsort(centers)
    mapping = {int(raw): ERROR_REGIME_ORDER[rank] for rank, raw in enumerate(ordered_raw_labels)}
    regimes = np.array([mapping[int(label)] for label in labels], dtype=object)
    sorted_centers = np.sort(centers)
    boundaries = (
        float((sorted_centers[0] + sorted_centers[1]) / 2),
        float((sorted_centers[1] + sorted_centers[2]) / 2),
    )
    silhouette = float(silhouette_score(values, labels))
    data_driven_k2_regimes, _ = ordered_kmeans_classes(absolute_errors, 2)
    return regimes, data_driven_k2_regimes, sorted_centers, boundaries, silhouette, kmeans_k_table(absolute_errors)


def pooled_kmeans_diagnostics(series_results: dict) -> tuple[pd.DataFrame, dict]:
    """Pooled K-means on scale-normalised adaptive errors.

    Normalisation: |e_{s,t}| / (median of |e_{s,tau}| over the series' own
    calibration forecasts + eps). The scale is therefore fixed before the
    diagnostic period starts. The primary clustering uses the diagnostic
    period only (9-8-7: 2012-2018, 14 x 7 = 98 pooled observations); the
    full rolling period (2004-2018, 210) is a sensitivity analysis. K-means
    itself is a retrospective, unsupervised partition of the pooled period.
    """
    calibration = MAD_CALIBRATION_FORECASTS
    rows = []
    for series, result in series_results.items():
        absolute = np.abs(result["adaptive_residuals"])
        scale = float(np.median(absolute[:calibration]))
        if scale <= np.finfo(float).eps:
            scale = float(np.mean(absolute[:calibration])) or 1.0
        result["error_scale"] = scale
        for index, (year, error) in enumerate(zip(result["rolling"].years, absolute)):
            rows.append({
                "Series": series, "Year": int(year),
                "Period": "Calibration" if index < calibration else "Diagnostic",
                "Absolute_Error": float(error),
                "Error_Scale": scale,
                "Normalized_Absolute_Error": float(error / (scale + 1e-12)),
            })
    pooled = pd.DataFrame(rows)
    diagnostic = pooled["Period"].eq("Diagnostic").to_numpy()
    z_diag = pooled.loc[diagnostic, "Normalized_Absolute_Error"].to_numpy()
    raw_diag = pooled.loc[diagnostic, "Absolute_Error"].to_numpy()

    regimes, k2_regimes, centers, boundaries, silhouette, k_table = kmeans_error_diagnostics(z_diag)
    pooled["Error_Regime"] = np.nan
    pooled["Data_Driven_k2_Regime"] = np.nan
    pooled["Raw_Error_Regime"] = np.nan
    pooled = pooled.astype({"Error_Regime": object, "Data_Driven_k2_Regime": object, "Raw_Error_Regime": object})
    pooled.loc[diagnostic, "Error_Regime"] = regimes
    pooled.loc[diagnostic, "Data_Driven_k2_Regime"] = k2_regimes

    stability_k3 = kmeans_perturbation_stability(z_diag, k=3)
    stability_k2 = kmeans_perturbation_stability(z_diag, k=2)

    # Ablation: raw (unnormalised) pooled errors, where large sectors dominate.
    raw_regimes, _, raw_centers, raw_boundaries, raw_silhouette, raw_k = (
        kmeans_error_diagnostics(raw_diag)
    )
    pooled.loc[diagnostic, "Raw_Error_Regime"] = raw_regimes

    # Sensitivity: full rolling period (calibration + diagnostics).
    full_k = kmeans_k_table(pooled["Normalized_Absolute_Error"].to_numpy())

    comparison = {
        "normalized_centers": centers,
        "normalized_boundaries": boundaries,
        "normalized_silhouette": silhouette,
        "normalized_stability_ari_k3": stability_k3,
        "normalized_stability_ari_k2": stability_k2,
        "normalized_k_table": k_table,
        "data_driven_optimal_k": int(k_table.loc[k_table["Silhouette"].idxmax(), "k"]),
        "raw_centers": raw_centers,
        "raw_boundaries": raw_boundaries,
        "raw_silhouette": raw_silhouette,
        "raw_k_table": raw_k,
        "raw_vs_normalized_ari": float(adjusted_rand_score(raw_regimes, regimes)),
        "full_period_k_table": full_k,
    }
    return pooled, comparison


def kmeans_perturbation_stability(
    values_1d: np.ndarray,
    k: int = 3,
    repetitions: int = KMEANS_STABILITY_REPETITIONS,
) -> float:
    """Label stability under small Gaussian jitter (5% of the SD), mean ARI."""
    rng = np.random.default_rng(RANDOM_STATE)
    values = np.asarray(values_1d, dtype=float).reshape(-1, 1)
    base = KMeans(n_clusters=k, n_init=KMEANS_N_INIT, random_state=RANDOM_STATE)
    base_labels = base.fit_predict(values)
    scale = max(float(np.std(values, ddof=1)) * 0.05, 1e-9)
    scores = []
    for rep in range(repetitions):
        perturbed = values + rng.normal(0.0, scale, size=values.shape)
        fitted = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE + rep + 1)
        scores.append(adjusted_rand_score(base_labels, fitted.fit_predict(perturbed)))
    return float(np.mean(scores))


# =============================================================================
# 6. Statistical comparison across 14 series
# =============================================================================
def holm_adjust(p_values: list[float]) -> list[float]:
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running_max = 0.0
    m = len(p)
    for rank, index in enumerate(order):
        value = min(1.0, (m - rank) * p[index])
        running_max = max(running_max, value)
        adjusted[index] = running_max
    return adjusted.tolist()


def sector_cluster_bootstrap_ci(
    metrics_df: pd.DataFrame,
    metric: str,
    repetitions: int = BOOTSTRAP_REPETITIONS,
) -> pd.DataFrame:
    """Percentile CI of the cross-series mean, resampling sectors (Buy+Sell kept together)."""
    rng = np.random.default_rng(RANDOM_STATE)
    rows = []
    pivot = metrics_df.pivot(index="Series", columns="Model", values=metric)[METHOD_ORDER]
    sectors = metrics_df[["Series", "Sector_Code"]].drop_duplicates().set_index("Series")
    sector_names = sorted(sectors["Sector_Code"].astype(str).unique())
    values_by_sector = {
        sector: pivot.loc[sectors.index[sectors["Sector_Code"].astype(str) == sector]].to_numpy(float)
        for sector in sector_names
    }
    for model_index, model in enumerate(METHOD_ORDER):
        sample_means = np.empty(repetitions)
        for b in range(repetitions):
            sampled_sectors = rng.choice(sector_names, size=len(sector_names), replace=True)
            sample_values = np.concatenate([
                values_by_sector[sector][:, model_index] for sector in sampled_sectors
            ])
            sample_means[b] = np.mean(sample_values)
        rows.append({
            "Model": model,
            f"Mean_{metric}": float(pivot[model].mean()),
            f"{metric}_CI_2.5%": float(np.quantile(sample_means, 0.025)),
            f"{metric}_CI_97.5%": float(np.quantile(sample_means, 0.975)),
        })
    return pd.DataFrame(rows)


def _rank_biserial(differences: np.ndarray) -> float:
    nonzero = differences[~np.isclose(differences, 0.0)]
    if not nonzero.size:
        return 0.0
    signed_ranks = rankdata(np.abs(nonzero)) * np.sign(nonzero)
    return float(np.sum(signed_ranks) / np.sum(np.abs(signed_ranks)))


def _wilcoxon(differences: np.ndarray) -> tuple[float, float]:
    if np.allclose(differences, 0.0):
        return np.nan, 1.0
    try:
        test = wilcoxon(differences, alternative="two-sided", zero_method="wilcox")
        return float(test.statistic), float(test.pvalue)
    except ValueError:
        return np.nan, 1.0


def _rank_summary(pivots: dict[str, pd.DataFrame], prefix: str = "") -> pd.DataFrame:
    data = {"Model": METHOD_ORDER}
    for metric, pivot in pivots.items():
        data[f"Mean_{prefix}{metric}"] = [pivot[m].mean() for m in METHOD_ORDER]
    for metric, pivot in pivots.items():
        ranks = pivot.rank(axis=1, method="average", ascending=True)
        data[f"Mean_{prefix}{metric}_Rank"] = [ranks[m].mean() for m in METHOD_ORDER]
    for metric, pivot in pivots.items():
        ranks = pivot.rank(axis=1, method="average", ascending=True)
        data[f"First_Place_Count_{metric}"] = [int((ranks[m] == 1).sum()) for m in METHOD_ORDER]
    return (
        pd.DataFrame(data)
        .sort_values([f"Mean_{prefix}RMSE_Rank", f"Mean_{prefix}RMSE"])
        .reset_index(drop=True)
    )


def _friedman_rows(pivots: dict[str, pd.DataFrame], blocks_label: str) -> pd.DataFrame:
    rows = []
    for metric in COMPARISON_METRICS:
        pivot = pivots[metric]
        statistic, p_value = friedmanchisquare(*(pivot[m].to_numpy() for m in METHOD_ORDER))
        rows.append({
            "Blocks": blocks_label,
            "Metric": metric,
            "N_Blocks": len(pivot),
            "Methods": len(METHOD_ORDER),
            "Friedman_chi2": float(statistic),
            "df": len(METHOD_ORDER) - 1,
            "p_value": float(p_value),
            "Kendalls_W": float(statistic / (len(pivot) * (len(METHOD_ORDER) - 1))),
        })
    return pd.DataFrame(rows)


def statistical_model_comparison(metrics_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Level 1: 14 Buy/Sell series as blocks; RMSE, MASE and RMSSE.

    The adaptive selector is the pre-specified reference. Holm correction is
    applied within each metric family (10 comparisons). Bootstrap CIs resample
    the seven sectors so that Buy and Sell of one sector stay together.
    """
    pivots = {
        metric: metrics_df.pivot(index="Series", columns="Model", values=metric)[METHOD_ORDER]
        for metric in ("RMSE", "MAE", "MASE", "RMSSE")
    }
    rank_summary = _rank_summary(pivots)
    omnibus = _friedman_rows(pivots, "14 Buy/Sell series")

    sector_of = (
        metrics_df[["Series", "Sector_Code"]].drop_duplicates()
        .assign(Sector_Code=lambda d: d["Sector_Code"].astype(str))
        .set_index("Series")["Sector_Code"]
    )
    reference = ADAPTIVE_METHOD
    rows = []
    for metric in COMPARISON_METRICS:
        pivot = pivots[metric]
        sector_index = sector_of.loc[pivot.index].to_numpy()
        sectors = sorted(set(sector_index))
        family = []
        for model in MODEL_ORDER:
            differences = pivot[reference].to_numpy() - pivot[model].to_numpy()
            rng = np.random.default_rng(RANDOM_STATE)
            by_sector = {s: differences[sector_index == s] for s in sectors}
            boot = np.empty(BOOTSTRAP_REPETITIONS)
            for b in range(BOOTSTRAP_REPETITIONS):
                sampled = rng.choice(sectors, size=len(sectors), replace=True)
                boot[b] = float(np.mean(np.concatenate([by_sector[s] for s in sampled])))
            statistic, p = _wilcoxon(differences)
            family.append({
                "Metric": metric,
                "Reference": reference,
                "Compared_Model": model,
                "N_Series": len(differences),
                "Mean_Difference_Reference_minus_Model": float(np.mean(differences)),
                "Median_Difference_Reference_minus_Model": float(np.median(differences)),
                "Wilcoxon_W": statistic,
                "Rank_Biserial_Correlation": _rank_biserial(differences),
                "Sector_Cluster_Bootstrap_CI_2.5%": float(np.quantile(boot, 0.025)),
                "Sector_Cluster_Bootstrap_CI_97.5%": float(np.quantile(boot, 0.975)),
                "Raw_p": p,
            })
        for row, adjusted in zip(family, holm_adjust([r["Raw_p"] for r in family])):
            row["Holm_Adjusted_p"] = adjusted
            row["Significant_0.05"] = adjusted < 0.05
        rows.extend(family)
    return rank_summary, pd.DataFrame(rows), omnibus


def sector_level_confirmatory_comparison(
    metrics_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Level 2: Buy and Sell averaged within sector; seven sectors as blocks.

    Pairwise tests use the pre-specified family CONFIRMATORY_COMPARATORS.
    With N = 7 the smallest attainable two-sided exact Wilcoxon p-value is
    2 / 2^7 = 0.015625; the Holm-adjusted floor is m times this value and is
    reported so that non-significance by construction is visible.
    """
    sector_metrics = (
        metrics_df.groupby(["Sector_Code", "Sector", "Model"], observed=True)
        .agg(
            Sector_Mean_MAE=("MAE", "mean"),
            Sector_Mean_RMSE=("RMSE", "mean"),
            Sector_Mean_MASE=("MASE", "mean"),
            Sector_Mean_RMSSE=("RMSSE", "mean"),
        )
        .reset_index()
    )
    pivots = {
        metric: sector_metrics.pivot(
            index="Sector_Code", columns="Model", values=f"Sector_Mean_{metric}"
        )[METHOD_ORDER]
        for metric in ("RMSE", "MAE", "MASE", "RMSSE")
    }
    rank_summary = _rank_summary(pivots, prefix="Sector_")
    omnibus = _friedman_rows(pivots, "Seven sectors; Buy and Sell averaged within sector")

    reference = ADAPTIVE_METHOD
    m = len(CONFIRMATORY_COMPARATORS)
    rows = []
    for metric in COMPARISON_METRICS:
        pivot = pivots[metric]
        n_blocks = len(pivot)
        minimum_p = 2.0 / (2 ** n_blocks)
        rng = np.random.default_rng(RANDOM_STATE)
        family = []
        for model in CONFIRMATORY_COMPARATORS:
            differences = pivot[reference].to_numpy() - pivot[model].to_numpy()
            statistic, p = _wilcoxon(differences)
            boot = np.array([
                float(np.mean(rng.choice(differences, size=n_blocks, replace=True)))
                for _ in range(BOOTSTRAP_REPETITIONS)
            ])
            family.append({
                "Metric": metric,
                "Reference": reference,
                "Compared_Model": model,
                "N_Sectors": n_blocks,
                "Mean_Difference_Reference_minus_Model": float(np.mean(differences)),
                "Median_Difference_Reference_minus_Model": float(np.median(differences)),
                "Sectors_Reference_Better": int(np.sum(differences < 0)),
                "Wilcoxon_W": statistic,
                "Rank_Biserial_Correlation": _rank_biserial(differences),
                "Bootstrap_CI_2.5%": float(np.quantile(boot, 0.025)),
                "Bootstrap_CI_97.5%": float(np.quantile(boot, 0.975)),
                "Raw_p": p,
                "Minimum_Attainable_Raw_p": minimum_p,
                "Minimum_Attainable_Holm_p": min(1.0, m * minimum_p),
            })
        for row, adjusted in zip(family, holm_adjust([r["Raw_p"] for r in family])):
            row["Holm_Adjusted_p"] = adjusted
            row["Significant_0.05"] = adjusted < 0.05
        rows.extend(family)
    return sector_metrics, rank_summary, omnibus, pd.DataFrame(rows)


# =============================================================================
# 7. Full analysis
# =============================================================================
def analyze_panel(panel: pd.DataFrame):
    metrics_rows = []
    best_rows = []
    year_rows = []
    bandwidth_rows = []
    selection_rows = []
    series_results = {}
    total_lwlr_inner_fits = 0
    total_all_inner_fits = 0

    for sector_code in SECTOR_ORDER:
        sector_frame = panel.loc[panel["Sector_Code"] == sector_code].sort_values("Year")
        sector_name = str(sector_frame["Sector"].iloc[0])
        years = sector_frame["Year"].to_numpy(dtype=int)

        for value_column in VALUE_COLUMNS:
            indicator = VALUE_LABELS[value_column]
            series_name = f"{sector_code}_{indicator}"
            y = sector_frame[value_column].to_numpy(dtype=float)
            rolling = rolling_origin_predictions(years, y)
            total_lwlr_inner_fits += rolling.lwlr_inner_fits
            total_all_inner_fits += rolling.all_inner_fits

            model_metrics = {}
            for model in MODEL_ORDER:
                values = regression_metrics(
                    rolling.actual, rolling.predictions[model],
                    rolling.mase_scales, rolling.rmsse_scales,
                )
                model_metrics[model] = values
                metrics_rows.append({
                    "Series": series_name,
                    "Sector_Code": sector_code,
                    "Sector": sector_name,
                    "Indicator": indicator,
                    "Model": model,
                    **values,
                })

            adaptive_metrics = regression_metrics(
                rolling.actual, rolling.adaptive_predictions,
                rolling.mase_scales, rolling.rmsse_scales,
            )
            model_metrics[ADAPTIVE_METHOD] = adaptive_metrics
            metrics_rows.append({
                "Series": series_name,
                "Sector_Code": sector_code,
                "Sector": sector_name,
                "Indicator": indicator,
                "Model": ADAPTIVE_METHOD,
                **adaptive_metrics,
            })

            # Retrospective fixed-model winner is reported for comparison only;
            # it is never used to construct adaptive residuals or MAD bands.
            best_model = min(MODEL_ORDER, key=lambda m: model_metrics[m]["RMSE"])
            adaptive_prediction = rolling.adaptive_predictions
            residuals = rolling.actual - adaptive_prediction
            absolute_errors = np.abs(residuals)
            mad_table = sequential_mad_classification(residuals)

            try:
                shapiro_p = float(shapiro(residuals).pvalue)
            except ValueError:
                shapiro_p = np.nan
            durbin_watson = float(np.sum(np.diff(residuals) ** 2) / np.sum(residuals**2))
            ljung_q, ljung_p = ljung_box_test(residuals, lags=4)

            best_rows.append({
                "Series": series_name,
                "Sector": sector_name,
                "Indicator": indicator,
                "Best_Fixed_Model_Retrospective": best_model,
                "Adaptive_MAE": adaptive_metrics["MAE"],
                "Adaptive_RMSE": adaptive_metrics["RMSE"],
                "Adaptive_MASE": adaptive_metrics["MASE"],
                "Adaptive_RMSSE": adaptive_metrics["RMSSE"],
                "Adaptive_MAPE": adaptive_metrics["MAPE"],
                "Adaptive_sMAPE": adaptive_metrics["sMAPE"],
                "Adaptive_R2": adaptive_metrics["R2"],
                "Residual_Shapiro_p": shapiro_p,
                "Durbin_Watson": durbin_watson,
                "Ljung_Box_Q_lag4": ljung_q,
                "Ljung_Box_p_lag4": ljung_p,
            })

            for i, year in enumerate(rolling.years):
                record = {
                    "Series": series_name,
                    "Sector_Code": sector_code,
                    "Sector": sector_name,
                    "Indicator": indicator,
                    "Year": int(year),
                    "Actual": float(rolling.actual[i]),
                    "Selected_Model": str(rolling.selected_models[i]),
                    "Prediction": float(adaptive_prediction[i]),
                    "Residual": float(residuals[i]),
                    "Absolute_Error": float(absolute_errors[i]),
                    "APE_percent": float(absolute_errors[i] / rolling.actual[i] * 100),
                    "MASE": float(absolute_errors[i] / rolling.mase_scales[i]),
                    "RMSSE_Component": float(residuals[i] ** 2 / rolling.rmsse_scales[i]),
                    **mad_table.iloc[i].to_dict(),
                }
                if np.isfinite(record["Residual_Lower"]):
                    record["Value_Lower"] = record["Prediction"] + record["Residual_Lower"]
                    record["Value_Upper"] = record["Prediction"] + record["Residual_Upper"]
                else:
                    record["Value_Lower"] = np.nan
                    record["Value_Upper"] = np.nan
                year_rows.append(record)

                selection_rows.append({
                    "Series": series_name,
                    "Year": int(year),
                    "Selected_Model": str(rolling.selected_models[i]),
                    "Selected_ETS_Variant": str(rolling.ets_variants[i]),
                    "Optimistic_Selector_Model_Ablation": str(rolling.optimistic_selected_models[i]),
                    **{
                        f"Honest_Inner_RMSE_{model}": float(rolling.inner_scores[model][i])
                        for model in MODEL_ORDER
                    },
                    **{
                        f"Optimistic_Inner_RMSE_{model}": float(rolling.optimistic_inner_scores[model][i])
                        for model in MODEL_ORDER
                    },
                })

            for model in ["Gaussian LWLR", "Exponential LWLR"]:
                for year, bandwidth in zip(rolling.years, rolling.bandwidths[model]):
                    bandwidth_rows.append({
                        "Series": series_name,
                        "Model": model,
                        "Forecast_Year": int(year),
                        "Selected_Bandwidth": float(bandwidth),
                    })

            series_results[series_name] = {
                "sector_code": sector_code,
                "sector_name": sector_name,
                "indicator": indicator,
                "historical_years": years,
                "historical_values": y,
                "rolling": rolling,
                "best_model": best_model,
                "adaptive_prediction": adaptive_prediction,
                "adaptive_residuals": residuals,
                "mad_table": mad_table,
            }

    # Primary error clustering is pooled and normalized after every series has
    # generated leakage-free adaptive rolling errors.
    pooled_kmeans, kmeans_comparison = pooled_kmeans_diagnostics(series_results)
    year_df = pd.DataFrame(year_rows).merge(
        pooled_kmeans[[
            "Series", "Year", "Period", "Error_Scale", "Normalized_Absolute_Error",
            "Data_Driven_k2_Regime", "Error_Regime", "Raw_Error_Regime",
        ]],
        on=["Series", "Year"], how="left", validate="one_to_one",
    )
    for series, result in series_results.items():
        subset = pooled_kmeans.loc[
            (pooled_kmeans["Series"] == series) & pooled_kmeans["Period"].eq("Diagnostic")
        ].sort_values("Year")
        result["regime_years"] = subset["Year"].to_numpy()
        result["error_regimes"] = subset["Error_Regime"].to_numpy()
        result["normalized_errors"] = subset["Normalized_Absolute_Error"].to_numpy()
        result["k_boundaries"] = kmeans_comparison["normalized_boundaries"]

    return (
        pd.DataFrame(metrics_rows),
        pd.DataFrame(best_rows),
        year_df,
        pd.DataFrame(bandwidth_rows),
        pd.DataFrame(selection_rows),
        pooled_kmeans,
        kmeans_comparison,
        series_results,
        total_lwlr_inner_fits,
        total_all_inner_fits,
    )


# =============================================================================
# 8. Sensitivity analysis
# =============================================================================
def calibration_sensitivity(series_results: dict) -> pd.DataFrame:
    """Calibration-length sensitivity around the primary value (main +/- 2)."""
    n_forecasts = 24 - INITIAL_TRAIN_YEARS
    grid = [
        c for c in range(MAD_CALIBRATION_FORECASTS - 2, MAD_CALIBRATION_FORECASTS + 3)
        if 3 <= c <= n_forecasts - 3
    ]
    rows = []
    for calibration in grid:
        counts = {name: 0 for name in DIRECTION_ORDER}
        widths = []
        inside_total = 0
        evaluated_total = 0
        for result in series_results.values():
            residuals = result["adaptive_residuals"]
            table = sequential_mad_classification(residuals, calibration_forecasts=calibration)
            for name in DIRECTION_ORDER:
                counts[name] += int((table["Direction_Class"] == name).sum())
            widths.extend(table["Corridor_Width"].dropna().tolist())
            evaluated = (table["Direction_Class"] != "Calibration").to_numpy()
            evaluated_total += int(evaluated.sum())
            inside_total += int((table["Direction_Class"] == "Near trend").sum())
        rows.append({
            "Calibration_Forecasts": calibration,
            "Primary_Configuration": calibration == MAD_CALIBRATION_FORECASTS,
            "Evaluated_N": evaluated_total,
            **counts,
            "Empirical_Coverage_percent": 100.0 * inside_total / evaluated_total,
            "Mean_Corridor_Width": float(np.mean(widths)),
            "Median_Corridor_Width": float(np.median(widths)),
        })
    return pd.DataFrame(rows)


def mad_lambda_sensitivity(series_results: dict) -> pd.DataFrame:
    """Coverage-width trade-off for pre-specified MAD multipliers."""
    rows = []
    for multiplier in MAD_LAMBDAS:
        for series, result in series_results.items():
            residuals = result["adaptive_residuals"]
            table = sequential_mad_classification(residuals, multiplier=multiplier)
            evaluated = table["Direction_Class"] != "Calibration"
            residual_eval = residuals[evaluated.to_numpy()]
            lower = table.loc[evaluated, "Residual_Lower"].to_numpy(float)
            upper = table.loc[evaluated, "Residual_Upper"].to_numpy(float)
            inside = (residual_eval >= lower) & (residual_eval <= upper)
            width = upper - lower
            actual = result["rolling"].actual[evaluated.to_numpy()]
            normalization = float(np.mean(np.abs(actual)))  # mean level of evaluated years
            rows.append({
                "Lambda": multiplier,
                "Series": series,
                "Evaluated_N": int(evaluated.sum()),
                "Coverage_percent": float(np.mean(inside) * 100),
                "Mean_Width": float(np.mean(width)),
                "Level_Normalized_Mean_Width": float(np.mean(width) / normalization) if normalization > 0 else np.nan,
                "Below_Count": int(np.sum(residual_eval < lower)),
                "Near_Count": int(np.sum(inside)),
                "Above_Count": int(np.sum(residual_eval > upper)),
            })
    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby("Lambda", as_index=False)
        .agg(
            Series=("Series", "count"), Evaluated_N=("Evaluated_N", "sum"),
            Mean_Coverage_percent=("Coverage_percent", "mean"),
            Mean_Width=("Mean_Width", "mean"),
            Mean_Level_Normalized_Width=("Level_Normalized_Mean_Width", "mean"),
            Below_Count=("Below_Count", "sum"), Near_Count=("Near_Count", "sum"),
            Above_Count=("Above_Count", "sum"),
        )
    )
    return detail, summary


def direction_asymmetry_test(series_results: dict) -> pd.DataFrame:
    """Exact sign tests for directional asymmetry of MAD-corridor exceedances.

    Proposition 2 of the manuscript: if the diagnostic residuals are jointly
    symmetric about a common centre, Below and Above counts have the same
    distribution; if, in addition, exceedance directions are conditionally
    independent, Below | (Below + Above = n) ~ Binomial(n, 1/2).
    Three versions are reported for each lambda:
      pooled   - all exceedances pooled (assumes conditional independence);
      year     - sign test over diagnostic years on (Below - Above) per year,
                 robust to cross-series dependence within a year;
      series   - sign test over the 14 series on (Below - Above) per series,
                 robust to serial dependence within a series.
    """
    rows = []
    for multiplier in MAD_LAMBDAS:
        records = []
        for series, result in series_results.items():
            table = sequential_mad_classification(result["adaptive_residuals"], multiplier=multiplier)
            years = result["rolling"].years
            for year, cls in zip(years, table["Direction_Class"].to_numpy()):
                if cls != "Calibration":
                    records.append((series, int(year), cls))
        frame = pd.DataFrame(records, columns=["Series", "Year", "Class"])
        below = int((frame["Class"] == "Below trend").sum())
        above = int((frame["Class"] == "Above trend").sum())
        n_out = below + above
        pooled = binomtest(below, n_out, 0.5, alternative="two-sided").pvalue if n_out else np.nan
        pooled_one = binomtest(below, n_out, 0.5, alternative="greater").pvalue if n_out else np.nan
        rows.append({"Lambda": multiplier, "Level": "pooled exceedances", "Blocks": n_out,
                     "Below_or_positive": below, "Above_or_negative": above, "Ties": 0,
                     "p_two_sided": float(pooled), "p_one_sided_below": float(pooled_one)})
        for level, key in (("diagnostic years", "Year"), ("series", "Series")):
            diff = frame.groupby(key)["Class"].agg(
                lambda c: int((c == "Below trend").sum()) - int((c == "Above trend").sum())
            )
            pos, neg = int((diff > 0).sum()), int((diff < 0).sum())
            ties = int((diff == 0).sum())
            n_eff = pos + neg
            p2 = binomtest(pos, n_eff, 0.5, alternative="two-sided").pvalue if n_eff else np.nan
            p1 = binomtest(pos, n_eff, 0.5, alternative="greater").pvalue if n_eff else np.nan
            rows.append({"Lambda": multiplier, "Level": level, "Blocks": n_eff,
                         "Below_or_positive": pos, "Above_or_negative": neg, "Ties": ties,
                         "p_two_sided": float(p2), "p_one_sided_below": float(p1)})
    return pd.DataFrame(rows)


def pettitt_test(values: np.ndarray) -> tuple[int, float, float]:
    """Exploratory Pettitt single-change test; low powered for n=24."""
    x = np.asarray(values, dtype=float)
    n = x.size
    statistics = np.empty(n - 1)
    for split in range(1, n):
        statistics[split - 1] = np.sum(
            np.sign(x[:split, None] - x[None, split:])
        )
    index = int(np.argmax(np.abs(statistics)))
    k_stat = float(np.abs(statistics[index]))
    p_value = float(min(1.0, 2.0 * np.exp((-6.0 * k_stat**2) / (n**3 + n**2))))
    return index + 1, k_stat, p_value


def exploratory_pettitt_table(series_results: dict) -> pd.DataFrame:
    rows = []
    for series, result in series_results.items():
        split, statistic, p_value = pettitt_test(result["historical_values"])
        years = result["historical_years"]
        rows.append({
            "Series": series,
            "Candidate_Change_Year": int(years[split]),
            "Pettitt_K": statistic,
            "Approximate_p": p_value,
            "Exploratory_Only": True,
        })
    return pd.DataFrame(rows)


def exploratory_tables(panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    long = panel.melt(
        id_vars=["Year", "Sector_Code", "Sector"], value_vars=VALUE_COLUMNS,
        var_name="Indicator_Column", value_name="Value_USD_bn",
    )
    long["Indicator"] = long["Indicator_Column"].map(VALUE_LABELS)
    descriptive = (
        long.groupby(["Sector_Code", "Sector", "Indicator"], observed=True)["Value_USD_bn"]
        .agg(N="count", Mean="mean", SD="std", Minimum="min", Median="median", Maximum="max")
        .reset_index()
    )
    growth_rows = []
    for (code, sector, indicator), frame in long.groupby(
        ["Sector_Code", "Sector", "Indicator"], observed=True
    ):
        frame = frame.sort_values("Year")
        values = frame["Value_USD_bn"].to_numpy(float)
        yoy = pd.Series(values).pct_change() * 100
        cagr = ((values[-1] / values[0]) ** (1 / (len(values) - 1)) - 1) * 100
        growth_rows.append({
            "Sector_Code": code, "Sector": sector, "Indicator": indicator,
            "CAGR_1995_2018_percent": float(cagr),
            "Mean_YoY_percent": float(yoy.mean()),
            "Median_YoY_percent": float(yoy.median()),
            "YoY_SD_percent": float(yoy.std(ddof=1)),
        })
    growth = pd.DataFrame(growth_rows)
    buy_sell = pd.DataFrame([
        {
            "Sector_Code": code,
            "Buy_Sell_Pearson_r": float(
                frame["Buy_USD_bn"].corr(frame["Sell_USD_bn"])
            ),
        }
        for code, frame in panel.groupby("Sector_Code", observed=True)
    ])
    buy_corr = panel.pivot(index="Year", columns="Sector_Code", values="Buy_USD_bn").corr()
    sell_corr = panel.pivot(index="Year", columns="Sector_Code", values="Sell_USD_bn").corr()
    return {
        "eda_descriptive": descriptive,
        "eda_growth": growth,
        "eda_buy_sell_correlation": buy_sell,
        "eda_buy_sector_correlation": buy_corr,
        "eda_sell_sector_correlation": sell_corr,
    }


def corridor_quality(year_df: pd.DataFrame) -> pd.DataFrame:
    """Empirical coverage and width of the sequential MAD diagnostic corridor.

    These are descriptive diagnostics; the corridor carries no nominal or
    probabilistic coverage guarantee and is not a prediction interval.
    """
    rows = []
    for series, frame in year_df.groupby("Series"):
        evaluated = frame.dropna(subset=["Value_Lower", "Value_Upper"]).copy()
        inside = (
            (evaluated["Actual"] >= evaluated["Value_Lower"])
            & (evaluated["Actual"] <= evaluated["Value_Upper"])
        )
        data_range = float(frame["Actual"].max() - frame["Actual"].min())
        mean_width = float((evaluated["Value_Upper"] - evaluated["Value_Lower"]).mean())
        rows.append({
            "Series": series,
            "Evaluated_N": len(evaluated),
            "Empirical_Coverage_percent": float(inside.mean() * 100),
            "Mean_Corridor_Width": mean_width,
            "Range_Normalized_Width": float(mean_width / data_range) if data_range > 0 else np.nan,
            "Below_rate_percent": float((evaluated["Actual"] < evaluated["Value_Lower"]).mean() * 100),
            "Above_rate_percent": float((evaluated["Actual"] > evaluated["Value_Upper"]).mean() * 100),
        })
    return pd.DataFrame(rows)


def disruption_period_sensitivity(metrics_df: pd.DataFrame, series_results: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    window_rows = []
    exclusion_rows = []
    for series, result in series_results.items():
        rolling = result["rolling"]
        for label, (start, end) in DISRUPTION_WINDOWS.items():
            mask = (rolling.years >= start) & (rolling.years <= end)
            if mask.sum() == 0:
                continue
            for method in METHOD_ORDER:
                pred = (
                    rolling.adaptive_predictions if method == ADAPTIVE_METHOD
                    else rolling.predictions[method]
                )
                window_rows.append({
                    "Series": series, "Window": label,
                    "Start": int(rolling.years[mask].min()), "End": int(rolling.years[mask].max()), "N": int(mask.sum()),
                    "Model": method,
                    **regression_metrics(
                        rolling.actual[mask], pred[mask],
                        rolling.mase_scales[mask], rolling.rmsse_scales[mask],
                    ),
                })
        for excluded_label in ("Global-crisis", "2014-2016-shock"):
            start, end = DISRUPTION_WINDOWS[excluded_label]
            keep = ~((rolling.years >= start) & (rolling.years <= end))
            for method in METHOD_ORDER:
                pred = (
                    rolling.adaptive_predictions if method == ADAPTIVE_METHOD
                    else rolling.predictions[method]
                )
                exclusion_rows.append({
                    "Series": series, "Excluded_Window": excluded_label,
                    "Remaining_N": int(keep.sum()), "Model": method,
                    **regression_metrics(
                        rolling.actual[keep], pred[keep],
                        rolling.mase_scales[keep], rolling.rmsse_scales[keep],
                    ),
                })
    return pd.DataFrame(window_rows), pd.DataFrame(exclusion_rows)


def fixed_bandwidth_predictions(years: np.ndarray, y: np.ndarray, kernel: str) -> tuple[np.ndarray, float, int]:
    x = years.astype(float) - float(years.min())
    h, _, fits = select_lwlr_bandwidth_nested(
        x[:INITIAL_TRAIN_YEARS], y[:INITIAL_TRAIN_YEARS], kernel
    )
    preds = []
    for test_index in range(INITIAL_TRAIN_YEARS, len(years)):
        preds.append(lwlr_predict_one(x[:test_index], y[:test_index], x[test_index], h, kernel))
        fits += 1
    return np.asarray(preds), h, fits


def ablation_analysis(series_results: dict, year_df: pd.DataFrame, kmeans_comparison: dict) -> dict[str, pd.DataFrame]:
    bandwidth_rows = []
    corridor_rows = []
    for series, result in series_results.items():
        years = result["historical_years"]
        y = result["historical_values"]
        rolling = result["rolling"]
        for kernel in ("Gaussian LWLR", "Exponential LWLR"):
            fixed_pred, fixed_h, fits = fixed_bandwidth_predictions(years, y, kernel)
            for mode, pred in (
                ("Dynamic bandwidth", rolling.predictions[kernel]),
                ("Fixed initial bandwidth", fixed_pred),
            ):
                bandwidth_rows.append({
                    "Series": series, "Kernel": kernel, "Bandwidth_Mode": mode,
                    "Fixed_h_if_applicable": fixed_h if mode.startswith("Fixed") else np.nan,
                    "Additional_Fits": fits if mode.startswith("Fixed") else 0,
                    **regression_metrics(rolling.actual, pred, rolling.mase_scales, rolling.rmsse_scales),
                })

        residuals = result["adaptive_residuals"]
        dynamic = sequential_mad_classification(residuals)
        calibration = residuals[:MAD_CALIBRATION_FORECASTS]
        center = float(np.median(calibration))
        sigma = MAD_SCALE * float(np.median(np.abs(calibration - center)))
        lo, hi = center - MAD_MULTIPLIER * sigma, center + MAD_MULTIPLIER * sigma
        eval_resid = residuals[MAD_CALIBRATION_FORECASTS:]
        for mode, lower, upper in (
            (
                "Dynamic sequential MAD",
                dynamic["Residual_Lower"].to_numpy()[MAD_CALIBRATION_FORECASTS:],
                dynamic["Residual_Upper"].to_numpy()[MAD_CALIBRATION_FORECASTS:],
            ),
            ("Static calibration MAD", np.full(eval_resid.size, lo), np.full(eval_resid.size, hi)),
        ):
            inside = (eval_resid >= lower) & (eval_resid <= upper)
            widths = upper - lower
            corridor_rows.append({
                "Series": series, "Corridor_Mode": mode, "N": eval_resid.size,
                "Empirical_Coverage_percent": float(inside.mean() * 100),
                "Mean_Width": float(np.mean(widths)),
                "Median_Width": float(np.median(widths)),
            })

    kmeans_rows = []
    for mode, prefix in (("Within-series normalized", "normalized"), ("Raw pooled errors", "raw")):
        k_table = kmeans_comparison[f"{prefix}_k_table"].set_index("k")
        kmeans_rows.append({
            "Mode": mode,
            "Period": "Diagnostic",
            "N": int(k_table.loc[3, "N"]),
            "Silhouette_k2": float(k_table.loc[2, "Silhouette"]),
            "Silhouette_k3": float(k_table.loc[3, "Silhouette"]),
            "Silhouette_k4": float(k_table.loc[4, "Silhouette"]),
            "Boundary_1_k3": kmeans_comparison[f"{prefix}_boundaries"][0],
            "Boundary_2_k3": kmeans_comparison[f"{prefix}_boundaries"][1],
            "Raw_vs_Normalized_ARI_k3": kmeans_comparison["raw_vs_normalized_ari"],
        })

    # Selector ablation: honest prequential inner scores versus the former
    # optimistic min-over-grid inner scores for tuned candidates.
    selector_rows = []
    for series, result in series_results.items():
        rolling = result["rolling"]
        for mode, pred, chosen in (
            ("Honest prequential inner score (primary)", rolling.adaptive_predictions, rolling.selected_models),
            ("Optimistic min-over-grid inner score (ablation)", rolling.optimistic_adaptive_predictions,
             rolling.optimistic_selected_models),
        ):
            chosen = np.asarray(chosen, dtype=object)
            selector_rows.append({
                "Series": series,
                "Selector_Inner_Score": mode,
                "LWLR_Selection_Share": float(np.isin(chosen, ["Gaussian LWLR", "Exponential LWLR"]).mean()),
                "Baseline_Selection_Share": float(np.isin(chosen, ["Naive", "Drift"]).mean()),
                **regression_metrics(rolling.actual, pred, rolling.mase_scales, rolling.rmsse_scales),
            })

    return {
        "ablation_bandwidth": pd.DataFrame(bandwidth_rows),
        "ablation_corridor": pd.DataFrame(corridor_rows),
        "ablation_kmeans_scaling": pd.DataFrame(kmeans_rows),
        "ablation_selector_inner_score": pd.DataFrame(selector_rows),
    }


FIRST_YEAR = 1995


def period_label(offset: int, length: int) -> str:
    return f"{FIRST_YEAR + offset}-{FIRST_YEAR + offset + length - 1}"


def diagnostic_length() -> int:
    return 24 - INITIAL_TRAIN_YEARS - MAD_CALIBRATION_FORECASTS


def computation_and_versions(total_lwlr_fits: int, total_all_fits: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    computation = pd.DataFrame([
        {"Parameter": "Temporal protocol", "Value": TEMPORAL_PROTOCOL},
        {"Parameter": "Initial training years", "Value": f"{period_label(0, INITIAL_TRAIN_YEARS)} (N={INITIAL_TRAIN_YEARS})"},
        {"Parameter": "Residual calibration years", "Value": f"{period_label(INITIAL_TRAIN_YEARS, MAD_CALIBRATION_FORECASTS)} (N={MAD_CALIBRATION_FORECASTS})"},
        {"Parameter": "Dynamic diagnostic years", "Value": f"{period_label(INITIAL_TRAIN_YEARS + MAD_CALIBRATION_FORECASTS, diagnostic_length())} (N={diagnostic_length()})"},
        {"Parameter": "Series", "Value": 14},
        {"Parameter": "Outer forecasts per series", "Value": 24 - INITIAL_TRAIN_YEARS},
        {"Parameter": "Total outer forecasts", "Value": 14 * (24 - INITIAL_TRAIN_YEARS)},
        {"Parameter": "Pooled diagnostic observations (MAD, K-means)", "Value": 14 * diagnostic_length()},
        {"Parameter": "Candidate fixed models", "Value": len(MODEL_ORDER)},
        {"Parameter": "Bandwidth candidates", "Value": len(BANDWIDTH_GRID)},
        {"Parameter": "Bandwidth grid range", "Value": f"{BANDWIDTH_GRID[0]:.2f}-{BANDWIDTH_GRID[-1]:.2f}"},
        {"Parameter": "Unique LWLR local fits (cached prequential design)", "Value": total_lwlr_fits},
        {"Parameter": "Unique candidate fits (all models)", "Value": total_all_fits},
        {"Parameter": "Adaptive inner score", "Value": "Honest prequential RMSE, common inner window"},
        {"Parameter": "Random seed", "Value": RANDOM_STATE},
        {"Parameter": "Bootstrap repetitions", "Value": BOOTSTRAP_REPETITIONS},
        {"Parameter": "K-means n_init", "Value": KMEANS_N_INIT},
    ])
    versions = pd.DataFrame([
        {"Package": "Python", "Version": platform.python_version()},
        {"Package": "NumPy", "Version": np.__version__},
        {"Package": "pandas", "Version": pd.__version__},
        {"Package": "SciPy", "Version": scipy.__version__},
        {"Package": "scikit-learn", "Version": sklearn.__version__},
        {"Package": "Matplotlib", "Version": plt.matplotlib.__version__},
        {"Package": "openpyxl", "Version": openpyxl.__version__},
    ])
    return computation, versions


def build_run_metadata(
    input_path: Path,
    panel: pd.DataFrame,
    total_lwlr_fits: int,
    total_all_fits: int,
    sheet_name: str = SHEET_NAME,
) -> pd.DataFrame:
    """Create an auditable record of the data and computational configuration."""
    metadata = [
        ("Execution_timestamp_UTC", pd.Timestamp.now(tz="UTC").isoformat()),
        ("Input_file", input_path.name),
        ("Input_file_size_bytes", input_path.stat().st_size),
        ("Dataset_SHA256", calculate_sha256(input_path)),
        ("Sheet", sheet_name),
        ("Observations", len(panel)),
        ("Minimum_year", int(panel["Year"].min())),
        ("Maximum_year", int(panel["Year"].max())),
        ("Sectors", len(SECTOR_ORDER)),
        ("Sector_order", " | ".join(SECTOR_ORDER)),
        ("Sector_mapping", " | ".join(f"{code}={SECTOR_NAMES[code]}" for code in SECTOR_ORDER)),
        ("Series", len(SECTOR_ORDER) * len(VALUE_COLUMNS)),
        ("Fixed_candidate_models", len(MODEL_ORDER)),
        ("Fixed_model_order", " | ".join(MODEL_ORDER)),
        ("Adaptive_selector", ADAPTIVE_METHOD),
        ("Pairwise_reference", ADAPTIVE_METHOD),
        ("ETS_definition", "SES / Holt / damped Holt; SSE-estimated; fixed initial states; variant tuned by inner validation"),
        ("Temporal_protocol", TEMPORAL_PROTOCOL),
        ("Initial_training_period", period_label(0, INITIAL_TRAIN_YEARS)),
        ("Residual_calibration_period", period_label(INITIAL_TRAIN_YEARS, MAD_CALIBRATION_FORECASTS)),
        ("Dynamic_diagnostic_period", period_label(INITIAL_TRAIN_YEARS + MAD_CALIBRATION_FORECASTS, diagnostic_length())),
        ("Diagnostic_length", diagnostic_length()),
        ("Initial_training_years", INITIAL_TRAIN_YEARS),
        ("Inner_minimum_training_years", INNER_MIN_TRAIN_YEARS),
        ("Outer_forecasts_per_series", 24 - INITIAL_TRAIN_YEARS),
        ("Total_outer_forecasts", len(SECTOR_ORDER) * len(VALUE_COLUMNS) * (24 - INITIAL_TRAIN_YEARS)),
        ("Bandwidth_candidates", len(BANDWIDTH_GRID)),
        ("Bandwidth_grid", " | ".join(f"{value:.6g}" for value in BANDWIDTH_GRID)),
        ("Unique_LWLR_local_fits", total_lwlr_fits),
        ("Unique_candidate_fits", total_all_fits),
        ("Adaptive_inner_score", "Honest prequential RMSE; tuned candidates re-tuned before each inner endpoint"),
        ("Comparison_metrics", " | ".join(COMPARISON_METRICS)),
        ("Confirmatory_comparators", " | ".join(CONFIRMATORY_COMPARATORS)),
        ("KMeans_primary_period", "Diagnostic period; scale = calibration-period median absolute error"),
        ("MAD_calibration_forecasts", MAD_CALIBRATION_FORECASTS),
        ("MAD_main_lambda", MAD_MULTIPLIER),
        ("MAD_lambda_grid", " | ".join(map(str, MAD_LAMBDAS))),
        ("MAD_consistency_constant", MAD_SCALE),
        ("Random_seed", RANDOM_STATE),
        ("Bootstrap_repetitions", BOOTSTRAP_REPETITIONS),
        ("KMeans_n_init", KMEANS_N_INIT),
        ("KMeans_stability_repetitions", KMEANS_STABILITY_REPETITIONS),
        ("Python_version", platform.python_version()),
        ("NumPy_version", np.__version__),
        ("pandas_version", pd.__version__),
        ("SciPy_version", scipy.__version__),
        ("scikit_learn_version", sklearn.__version__),
        ("Matplotlib_version", plt.matplotlib.__version__),
        ("openpyxl_version", openpyxl.__version__),
    ]
    return pd.DataFrame(metadata, columns=["Parameter", "Value"])


# =============================================================================
# 9. Console tables
# =============================================================================
def print_complete_results(
    input_path: Path,
    panel: pd.DataFrame,
    metrics_df: pd.DataFrame,
    best_df: pd.DataFrame,
    year_df: pd.DataFrame,
    bandwidth_df: pd.DataFrame,
    selection_df: pd.DataFrame,
    pooled_kmeans: pd.DataFrame,
    kmeans_comparison: dict,
    series_results: dict,
    total_lwlr_inner_fits: int,
    total_all_inner_fits: int,
    sheet_name: str = SHEET_NAME,
):
    rank_summary, wilcoxon_table, omnibus = statistical_model_comparison(metrics_df)
    (
        sector_confirmatory_metrics,
        sector_confirmatory_ranks,
        sector_confirmatory_friedman,
        sector_confirmatory_pairwise,
    ) = sector_level_confirmatory_comparison(metrics_df)
    rmse_pivot = metrics_df.pivot(index="Series", columns="Model", values="RMSE")[METHOD_ORDER]
    mae_pivot = metrics_df.pivot(index="Series", columns="Model", values="MAE")[METHOD_ORDER]
    bootstrap_table = sector_cluster_bootstrap_ci(metrics_df, "RMSE")
    for metric in ("MAE", "MASE", "RMSSE"):
        bootstrap_table = bootstrap_table.merge(sector_cluster_bootstrap_ci(metrics_df, metric), on="Model")
    sensitivity = calibration_sensitivity(series_results)
    lambda_detail, lambda_summary = mad_lambda_sensitivity(series_results)
    asymmetry = direction_asymmetry_test(series_results)
    eda = exploratory_tables(panel)
    corridor = corridor_quality(year_df)
    disruption_windows, disruption_exclusions = disruption_period_sensitivity(
        metrics_df, series_results
    )
    pettitt = exploratory_pettitt_table(series_results)
    ablations = ablation_analysis(series_results, year_df, kmeans_comparison)
    computation, versions = computation_and_versions(
        total_lwlr_inner_fits, total_all_inner_fits
    )
    run_metadata = build_run_metadata(
        input_path, panel, total_lwlr_inner_fits, total_all_inner_fits, sheet_name
    )

    direction_counts = (
        year_df.groupby(["Series", "Direction_Class"], observed=False)
        .size().unstack(fill_value=0).reindex(columns=DIRECTION_ORDER, fill_value=0)
    )
    error_regime_counts = (
        year_df.dropna(subset=["Error_Regime"])
        .groupby(["Series", "Error_Regime"], observed=False)
        .size().unstack(fill_value=0).reindex(columns=ERROR_REGIME_ORDER, fill_value=0)
    )
    bandwidth_summary = (
        bandwidth_df.groupby(["Series", "Model"])["Selected_Bandwidth"]
        .agg(["min", "median", "max"]).reset_index()
    )
    lower_h, upper_h = float(BANDWIDTH_GRID[0]), float(BANDWIDTH_GRID[-1])
    bandwidth_boundary = (
        bandwidth_df.assign(
            At_Lower=lambda d: np.isclose(d["Selected_Bandwidth"], lower_h),
            At_Upper=lambda d: np.isclose(d["Selected_Bandwidth"], upper_h),
        )
        .groupby("Model")
        .agg(
            Outer_Origins=("Selected_Bandwidth", "size"),
            Median_Bandwidth=("Selected_Bandwidth", "median"),
            Lower_Boundary_Count=("At_Lower", "sum"),
            Upper_Boundary_Count=("At_Upper", "sum"),
        )
        .reset_index()
    )
    bandwidth_boundary["Lower_Boundary_percent"] = 100 * bandwidth_boundary["Lower_Boundary_Count"] / bandwidth_boundary["Outer_Origins"]
    bandwidth_boundary["Upper_Boundary_percent"] = 100 * bandwidth_boundary["Upper_Boundary_Count"] / bandwidth_boundary["Outer_Origins"]
    bandwidth_boundary["Grid"] = f"{lower_h:.2f}-{upper_h:.2f} ({len(BANDWIDTH_GRID)} values)"
    selection_frequency = (
        selection_df["Selected_Model"].value_counts()
        .reindex(MODEL_ORDER, fill_value=0).rename_axis("Model").reset_index(name="Selections")
    )
    selection_frequency["Share_percent"] = 100 * selection_frequency["Selections"] / len(selection_df)
    optimistic = (
        selection_df["Optimistic_Selector_Model_Ablation"].value_counts()
        .reindex(MODEL_ORDER, fill_value=0)
    )
    selection_frequency["Optimistic_Selector_Selections_Ablation"] = optimistic.to_numpy()
    k_summary = kmeans_comparison["normalized_k_table"].copy()
    k_summary.insert(0, "Period", "Diagnostic")
    k_summary.insert(1, "Scaling", "Within-series normalized")
    k_summary["Stability_ARI"] = k_summary["k"].map({
        2: kmeans_comparison["normalized_stability_ari_k2"],
        3: kmeans_comparison["normalized_stability_ari_k3"],
    })
    full_k = kmeans_comparison["full_period_k_table"].copy()
    full_k.insert(0, "Period", "Full rolling period (sensitivity)")
    full_k.insert(1, "Scaling", "Within-series normalized")
    kmeans_period_sensitivity = pd.concat([k_summary.drop(columns="Stability_ARI"), full_k], ignore_index=True)

    print("\n" + "=" * 118)
    print("LEAKAGE-FREE 7-SECTOR REGRESSION, ROBUST CORRIDOR AND CLASSIFICATION ANALYSIS")
    print("=" * 118)
    print(f"Input file: {input_path}")
    print(f"Dataset SHA-256: {calculate_sha256(input_path)}")
    print(f"Panel observations: {len(panel)} = 7 sectors × 24 years")
    print("Series analyzed: 14 = 7 sectors × Buy/Sell")
    print(f"Outer rolling forecasts per series: {24 - INITIAL_TRAIN_YEARS}")
    print(f"Candidate models: {len(MODEL_ORDER)}")
    print(f"Bandwidth candidates per inner loop: {len(BANDWIDTH_GRID)}")
    print(f"Temporal protocol: {TEMPORAL_PROTOCOL} "
          f"(training {period_label(0, INITIAL_TRAIN_YEARS)}, "
          f"calibration {period_label(INITIAL_TRAIN_YEARS, MAD_CALIBRATION_FORECASTS)}, "
          f"diagnostics {period_label(INITIAL_TRAIN_YEARS + MAD_CALIBRATION_FORECASTS, diagnostic_length())})")
    print(f"Unique LWLR local fits (cached prequential design): {total_lwlr_inner_fits:,}")
    print(f"Unique candidate fits across all models: {total_all_inner_fits:,}")
    print("Forecast protocol: expanding rolling origin; no future observation enters a forecast.")
    print("Model selection: nested inner rolling-origin at each outer origin; no future test error selects an earlier model.")
    print(
        f"Sequential MAD protocol: first {MAD_CALIBRATION_FORECASTS} rolling residuals "
        "are calibration; each later boundary uses earlier residuals only."
    )

    with pd.option_context(
        "display.max_rows", None, "display.max_columns", None,
        "display.width", 240, "display.float_format", "{:,.6f}".format,
    ):
        print("\nTABLE 1. Exploratory descriptive statistics")
        print(eda["eda_descriptive"].to_string(index=False))

        print("\nTABLE 2. Growth-rate diagnostics")
        print(eda["eda_growth"].to_string(index=False))

        print("\nTABLE 3. Buy-Sell correlations by sector")
        print(eda["eda_buy_sell_correlation"].to_string(index=False))

        print("\nTABLE 4. Complete rolling-origin metrics (14 series × 11 methods: 10 fixed and 1 adaptive)")
        print(metrics_df.sort_values(["Series", "RMSE"]).to_string(index=False))

        print("\nTABLE 5. Adaptive method and retrospective fixed-model benchmark")
        print(best_df.sort_values("Series").to_string(index=False))

        print("\nTABLE 6. Cross-series ranking (RMSE, MAE, MASE, RMSSE)")
        print(rank_summary.to_string(index=False))

        print("\nTABLE 7. Friedman omnibus tests by metric")
        print(omnibus.to_string(index=False))

        print("\nTABLE 8. Pairwise Wilcoxon tests (Holm within each metric family)")
        print(wilcoxon_table.to_string(index=False))

        print("\nTABLE 9. Sector-cluster bootstrap 95% confidence intervals")
        print(bootstrap_table.to_string(index=False))

        print("\nTABLE 9A. Seven-sector confirmatory metrics (Buy and Sell averaged within sector)")
        print(sector_confirmatory_metrics.to_string(index=False))

        print("\nTABLE 9B. Seven-sector confirmatory ranking")
        print(sector_confirmatory_ranks.to_string(index=False))

        print("\nTABLE 9C. Seven-sector confirmatory Friedman tests by metric")
        print(sector_confirmatory_friedman.to_string(index=False))

        print("\nTABLE 9D. Seven-sector confirmatory Wilcoxon tests (pre-specified comparators)")
        print(sector_confirmatory_pairwise.to_string(index=False))

        print("\nTABLE 10. Nested LWLR bandwidth summary")
        print(bandwidth_summary.to_string(index=False))

        print("\nTABLE 10A. Bandwidth-grid boundary diagnostics")
        print(bandwidth_boundary.to_string(index=False))

        print("\nTABLE 10B. Adaptive model-selection frequency")
        print(selection_frequency.to_string(index=False))

        print("\nTABLE 11. Dynamic MAD direction-class counts")
        print(direction_counts.to_string())

        print("\nTABLE 12. Pooled normalized K-means error-regime counts (diagnostic period)")
        print(error_regime_counts.to_string())

        print("\nTABLE 13. Pooled normalized K-means k=2,3,4 silhouette and stability (diagnostic period)")
        print(k_summary.to_string(index=False))

        print("\nTABLE 13A. K-means period sensitivity (diagnostic vs full rolling period)")
        print(kmeans_period_sensitivity.to_string(index=False))
        print(
            f"Data-driven silhouette optimum: k={kmeans_comparison['data_driven_optimal_k']}; "
            "k=3 is retained as an operational low/medium/high error-regime partition."
        )

        print("\nTABLE 14. Dynamic MAD empirical coverage and width diagnostics")
        print(corridor.to_string(index=False))

        print("\nTABLE 15. Calibration-length sensitivity analysis")
        print(sensitivity.to_string(index=False))

        print("\nTABLE 15A. MAD lambda coverage-width sensitivity (summary)")
        print(lambda_summary.to_string(index=False))

        print("\nTABLE 15A2. Exact sign tests for directional asymmetry of corridor exceedances")
        print(asymmetry.to_string(index=False))

        print("\nTABLE 15B. MAD lambda coverage-width sensitivity (series detail)")
        print(lambda_detail.to_string(index=False))

        print("\nTABLE 16. Pre-specified disruption-period sensitivity")
        print(disruption_windows.to_string(index=False))

        print("\nTABLE 17. Pre-specified disruption-period exclusion sensitivity")
        print(disruption_exclusions.to_string(index=False))

        print("\nTABLE 17A. Exploratory Pettitt change-point diagnostics")
        print(pettitt.to_string(index=False))

        print("\nTABLE 18. Bandwidth ablation")
        print(ablations["ablation_bandwidth"].to_string(index=False))

        print("\nTABLE 19. Dynamic versus static MAD ablation")
        print(ablations["ablation_corridor"].to_string(index=False))

        print("\nTABLE 20. K-means scaling ablation")
        print(ablations["ablation_kmeans_scaling"].to_string(index=False))

        print("\nTABLE 20A. Adaptive-selector inner-score ablation")
        print(ablations["ablation_selector_inner_score"].to_string(index=False))

        print("\nTABLE 21. Computation design")
        print(computation.to_string(index=False))

        print("\nTABLE 22. Software versions")
        print(versions.to_string(index=False))

        print("\nTABLE 22A. Dataset checksum and run metadata")
        print(run_metadata.to_string(index=False))

        print("\nTABLE 23. Complete year-level predictions and classifications")
        year_columns = [
            "Series", "Year", "Actual", "Selected_Model", "Prediction", "Residual",
            "Absolute_Error", "APE_percent", "Residual_Median", "MAD",
            "MASE", "RMSSE_Component",
            "Robust_Sigma", "Value_Lower", "Value_Upper", "Corridor_Width",
            "Direction_Class", "Normalized_Absolute_Error",
            "Data_Driven_k2_Regime", "Error_Regime",
        ]
        print(year_df[year_columns].sort_values(["Series", "Year"]).to_string(index=False))

    tables = {
        **eda,
        "rolling_metrics": metrics_df,
        "adaptive_summary": best_df,
        "rank_summary": rank_summary,
        "friedman_test": omnibus,
        "wilcoxon_holm": wilcoxon_table,
        "bootstrap_ci": bootstrap_table,
        "sector_confirmatory_metrics": sector_confirmatory_metrics,
        "sector_confirmatory_rank_summary": sector_confirmatory_ranks,
        "sector_confirmatory_friedman": sector_confirmatory_friedman,
        "sector_confirmatory_wilcoxon_holm": sector_confirmatory_pairwise,
        "bandwidths": bandwidth_df,
        "bandwidth_boundary": bandwidth_boundary,
        "adaptive_selection_frequency": selection_frequency,
        "model_selection_trace": selection_df,
        "year_level_results": year_df,
        "pooled_kmeans": pooled_kmeans,
        "kmeans_k_selection": k_summary,
        "kmeans_period_sensitivity": kmeans_period_sensitivity,
        "corridor_quality": corridor,
        "calibration_sensitivity": sensitivity,
        "mad_lambda_sensitivity_detail": lambda_detail,
        "mad_lambda_sensitivity_summary": lambda_summary,
        "direction_asymmetry": asymmetry,
        "disruption_windows": disruption_windows,
        "disruption_exclusions": disruption_exclusions,
        "exploratory_pettitt": pettitt,
        "computation_design": computation,
        "software_versions": versions,
        "run_metadata": run_metadata,
        **ablations,
    }
    return rank_summary, rmse_pivot, mae_pivot, sensitivity, tables


# =============================================================================
# 10. Figures (screen only)
# =============================================================================
def add_common_axis_style(ax):
    ax.grid(alpha=0.22)
    ax.tick_params(axis="x", rotation=45)


def plot_dataset_overview(panel: pd.DataFrame):
    fig, axes = plt.subplots(4, 2, figsize=(14, 16), sharex=True)
    axes = axes.ravel()
    for idx, sector_code in enumerate(SECTOR_ORDER):
        ax = axes[idx]
        data = panel.loc[panel["Sector_Code"] == sector_code].sort_values("Year")
        ax.plot(data["Year"], data["Buy_USD_bn"], "o-", ms=3.5, label="Buy", color="#F28E2B")
        ax.plot(data["Year"], data["Sell_USD_bn"], "o-", ms=3.5, label="Sell", color="#4E79A7")
        ax.set_title(f"{sector_code}: {data['Sector'].iloc[0]}")
        ax.set_ylabel("Billion USD")
        add_common_axis_style(ax)
    axes[7].axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2)
    fig.suptitle("Figure 1. Seven-sector Buy and Sell observations, 1995–2018", fontsize=15)
    fig.tight_layout(rect=[0, 0.035, 1, 0.97])


def plot_model_comparisons(series_results: dict):
    figure_number = 2
    for sector_code in SECTOR_ORDER:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharex=True)
        for ax, indicator in zip(axes, ["Buy", "Sell"]):
            result = series_results[f"{sector_code}_{indicator}"]
            rolling = result["rolling"]
            ax.plot(
                result["historical_years"], result["historical_values"],
                "ko-", ms=4, lw=1.4, label="Observed",
            )
            for model in MODEL_ORDER:
                ax.plot(
                    rolling.years, rolling.predictions[model],
                    lw=1.15, alpha=0.65, color=MODEL_COLORS[model], label=model,
                )
            ax.plot(
                rolling.years, rolling.adaptive_predictions,
                lw=3.0, color=MODEL_COLORS[ADAPTIVE_METHOD],
                label=ADAPTIVE_METHOD,
            )
            ax.axvline(rolling.years[0], color="0.4", ls=":", lw=1)
            ax.set_title(f"{indicator}; leakage-free adaptive selection")
            ax.set_xlabel("Year")
            ax.set_ylabel(f"{indicator}, billion USD")
            add_common_axis_style(ax)
        axes[1].legend(loc="best", fontsize=8)
        fig.suptitle(
            f"Figure {figure_number}. Rolling-origin model comparison: "
            f"{result['sector_name']} ({sector_code})",
            fontsize=14,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        figure_number += 1


def plot_metric_heatmaps(rmse_pivot: pd.DataFrame, mae_pivot: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    for ax, data, title in [
        (axes[0], rmse_pivot, "Rolling-origin RMSE"),
        (axes[1], mae_pivot, "Rolling-origin MAE"),
    ]:
        image = ax.imshow(data.to_numpy(), aspect="auto", cmap="YlOrRd")
        ax.set_xticks(range(len(METHOD_ORDER)), METHOD_ORDER, rotation=50, ha="right")
        ax.set_yticks(range(len(data.index)), data.index)
        ax.set_title(title)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                value = data.iloc[i, j]
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Figure 9. Model-error heatmaps across 14 series", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.96])


def plot_rank_summary(rank_summary: pd.DataFrame):
    ordered = rank_summary.sort_values("Mean_RMSE_Rank").reset_index(drop=True)
    metrics = [("RMSE", "#1565C0"), ("MASE", "#F9A825"), ("RMSSE", "#6A1B9A")]
    positions = np.arange(len(ordered))
    width = 0.27
    fig, ax = plt.subplots(figsize=(13, 6))
    for offset, (metric, color) in zip((-width, 0.0, width), metrics):
        ax.bar(positions + offset, ordered[f"Mean_{metric}_Rank"], width, color=color, label=f"{metric} rank")
    for pos, count in zip(positions, ordered["First_Place_Count_RMSE"]):
        ax.text(pos - width, ordered.loc[pos, "Mean_RMSE_Rank"], f"{int(count)}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(positions, ordered["Model"], rotation=35, ha="right")
    ax.set_ylabel("Mean rank across 14 series (lower is better)")
    ax.set_title("Figure 10. Mean model rank by RMSE, MASE and RMSSE (numbers: RMSE first places)")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()


def smooth_curve(x: np.ndarray, y: np.ndarray, points: int = 300):
    dense_x = np.linspace(float(np.min(x)), float(np.max(x)), points)
    dense_y = PchipInterpolator(x, y)(dense_x)
    return dense_x, dense_y


def plot_dynamic_corridors(series_results: dict):
    figure_number = 11
    for sector_code in SECTOR_ORDER:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        for ax, indicator in zip(axes, ["Buy", "Sell"]):
            result = series_results[f"{sector_code}_{indicator}"]
            rolling = result["rolling"]
            mad = result["mad_table"]
            center_x, center_y = smooth_curve(rolling.years, result["adaptive_prediction"])
            evaluation = mad["Direction_Class"].to_numpy() != "Calibration"
            eval_years = rolling.years[evaluation]
            lower_values = result["adaptive_prediction"][evaluation] + mad.loc[evaluation, "Residual_Lower"].to_numpy()
            upper_values = result["adaptive_prediction"][evaluation] + mad.loc[evaluation, "Residual_Upper"].to_numpy()
            dense_eval_years = np.linspace(eval_years.min(), eval_years.max(), 250)
            dense_lower = PchipInterpolator(eval_years, lower_values)(dense_eval_years)
            dense_width = PchipInterpolator(eval_years, upper_values - lower_values)(dense_eval_years)
            dense_upper = dense_lower + np.maximum(dense_width, 0)

            ax.fill_between(dense_eval_years, dense_lower, dense_upper, color="#A5D6A7", alpha=0.30, label="Dynamic MAD corridor")
            ax.plot(center_x, center_y, color="black", lw=2.2, label="Adaptive rolling centre")
            ax.plot(dense_eval_years, dense_lower, "--", color="#1565C0", lw=1.6, label="Lower boundary")
            ax.plot(dense_eval_years, dense_upper, "--", color="#D32F2F", lw=1.6, label="Upper boundary")
            ax.axvspan(rolling.years[0], rolling.years[MAD_CALIBRATION_FORECASTS-1], color="#BDBDBD", alpha=0.16, label="Residual calibration")
            ax.plot(result["historical_years"], result["historical_values"], color="0.70", lw=1)
            classes = mad["Direction_Class"].to_numpy()
            for name in DIRECTION_ORDER:
                mask = classes == name
                ax.scatter(
                    rolling.years[mask], rolling.actual[mask], s=62,
                    color=DIRECTION_COLORS[name], edgecolor="black", linewidth=0.4,
                    label=name, zorder=5,
                )
            ax.set_title(f"{indicator}: adaptive selector")
            ax.set_xlabel("Year")
            ax.set_ylabel(f"{indicator}, billion USD")
            add_common_axis_style(ax)

            # Exact geometry/class agreement at every classifiable observation.
            geometric = np.full(len(rolling.years), "Calibration", dtype=object)
            value_lower = result["adaptive_prediction"] + mad["Residual_Lower"].to_numpy()
            value_upper = result["adaptive_prediction"] + mad["Residual_Upper"].to_numpy()
            geometric[evaluation] = "Near trend"
            geometric[evaluation & (rolling.actual < value_lower)] = "Below trend"
            geometric[evaluation & (rolling.actual > value_upper)] = "Above trend"
            assert np.array_equal(geometric, classes), f"Plot/class mismatch: {sector_code}_{indicator}"

        axes[1].legend(loc="best", fontsize=7.5)
        fig.suptitle(
            f"Figure {figure_number}. Sequential dynamic MAD corridors: "
            f"{result['sector_name']} ({sector_code})",
            fontsize=14,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        figure_number += 1


def plot_kmeans_diagnostics(series_results: dict):
    figure_number = 18
    for sector_code in SECTOR_ORDER:
        fig, axes = plt.subplots(1, 2, figsize=(16, 5.7), sharex=True)
        for ax, indicator in zip(axes, ["Buy", "Sell"]):
            result = series_results[f"{sector_code}_{indicator}"]
            years = result["regime_years"]
            errors = result["normalized_errors"]
            regimes = result["error_regimes"]
            for name in ERROR_REGIME_ORDER:
                mask = regimes == name
                ax.scatter(
                    years[mask], errors[mask], s=65,
                    color=ERROR_REGIME_COLORS[name], edgecolor="black", linewidth=0.4,
                    label=name,
                )
            ax.axhline(result["k_boundaries"][0], color="#2E7D32", ls="--", lw=1.4, label="K-means boundary 1")
            ax.axhline(result["k_boundaries"][1], color="#D32F2F", ls="--", lw=1.4, label="K-means boundary 2")
            ax.set_title(f"{indicator}: normalized absolute rolling error (diagnostic period)")
            ax.set_xlabel("Year")
            ax.set_ylabel("Normalized absolute error")
            add_common_axis_style(ax)
        axes[1].legend(loc="best", fontsize=8)
        fig.suptitle(
            f"Figure {figure_number}. K-means error regimes: "
            f"{result['sector_name']} ({sector_code})",
            fontsize=14,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        figure_number += 1


def plot_sensitivity(sensitivity: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for name in ["Below trend", "Near trend", "Above trend"]:
        axes[0].plot(
            sensitivity["Calibration_Forecasts"], sensitivity[name],
            "o-", lw=2, color=DIRECTION_COLORS[name], label=name,
        )
    axes[0].set_xlabel("Number of calibration forecasts")
    axes[0].set_ylabel("Total class count across 14 series")
    axes[0].set_title("Direction-class sensitivity")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(
        sensitivity["Calibration_Forecasts"], sensitivity["Mean_Corridor_Width"],
        "o-", lw=2, color="#6A1B9A", label="Mean width",
    )
    axes[1].plot(
        sensitivity["Calibration_Forecasts"], sensitivity["Median_Corridor_Width"],
        "s--", lw=2, color="#00838F", label="Median width",
    )
    axes[1].set_xlabel("Number of calibration forecasts")
    axes[1].set_ylabel("Corridor width, billion USD")
    axes[1].set_title("Dynamic-corridor width sensitivity")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    fig.suptitle("Figure 25. Sensitivity to the residual-calibration length", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])


def plot_additional_diagnostics(tables: dict[str, pd.DataFrame]):
    growth = tables["eda_growth"]
    fig, ax = plt.subplots(figsize=(13, 6))
    pivot = growth.pivot(index="Sector_Code", columns="Indicator", values="CAGR_1995_2018_percent")
    pivot.plot(kind="bar", ax=ax, color=["#F28E2B", "#4E79A7"])
    ax.set_ylabel("CAGR, %")
    ax.set_title("Figure 26. Compound annual growth by sector and indicator")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, key, title in (
        (axes[0], "eda_buy_sector_correlation", "Buy"),
        (axes[1], "eda_sell_sector_correlation", "Sell"),
    ):
        data = tables[key]
        image = ax.imshow(data.to_numpy(), vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_xticks(range(len(data.columns)), data.columns, rotation=45)
        ax.set_yticks(range(len(data.index)), data.index)
        ax.set_title(f"{title} sector correlation")
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.text(j, i, f"{data.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Figure 27. Cross-sector Pearson correlations")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    selection = tables["model_selection_trace"]["Selected_Model"].value_counts().reindex(MODEL_ORDER, fill_value=0)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(selection.index, selection.values, color=[MODEL_COLORS[m] for m in selection.index])
    ax.set_ylabel("Number of outer origins selected")
    ax.set_title("Figure 28. Leakage-free adaptive model-selection frequency")
    ax.tick_params(axis="x", rotation=40)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    corridor = tables["corridor_quality"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    axes[0].bar(corridor["Series"], corridor["Empirical_Coverage_percent"], color="#2E7D32")
    axes[0].set_ylabel("Empirical coverage, %")
    axes[0].set_title("Dynamic MAD empirical coverage")
    axes[1].bar(corridor["Series"], corridor["Range_Normalized_Width"], color="#6A1B9A")
    axes[1].set_ylabel("Normalized mean width")
    axes[1].set_title("Dynamic MAD normalized width")
    for ax in axes:
        ax.tick_params(axis="x", rotation=55)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Figure 29. Dynamic MAD empirical coverage and width (not a prediction interval)")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    disruption = (
        tables["disruption_windows"].groupby(["Window", "Model"])["RMSE"]
        .mean().unstack()
    )
    fig, ax = plt.subplots(figsize=(14, 7))
    disruption[METHOD_ORDER].plot(kind="bar", ax=ax, colormap="tab20")
    ax.set_ylabel("Mean RMSE across series")
    ax.set_xlabel("Pre-specified disruption window (forecast years)")
    ax.set_title("Figure 30. Pre-specified disruption-period sensitivity")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    ablation = tables["ablation_bandwidth"]
    summary = ablation.groupby(["Kernel", "Bandwidth_Mode"])["RMSE"].mean().unstack()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    summary.plot(kind="bar", ax=axes[0], color=["#1565C0", "#90CAF9"])
    axes[0].set_ylabel("Mean RMSE")
    axes[0].set_title("Bandwidth ablation")
    corridor_ablation = tables["ablation_corridor"].groupby("Corridor_Mode")[["Empirical_Coverage_percent", "Mean_Width"]].mean()
    axes[1].bar(corridor_ablation.index, corridor_ablation["Empirical_Coverage_percent"], color=["#2E7D32", "#81C784"])
    axes[1].set_ylabel("Mean empirical coverage, %")
    axes[1].set_title("MAD corridor ablation")
    for ax in axes:
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Figure 31. Ablation analysis")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    lambda_table = tables["mad_lambda_sensitivity_summary"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    axes[0].plot(
        lambda_table["Lambda"], lambda_table["Mean_Coverage_percent"],
        "o-", lw=2, color="#2E7D32",
    )
    axes[0].set_xlabel(r"MAD multiplier $\lambda$")
    axes[0].set_ylabel("Mean empirical coverage, %")
    axes[0].set_title("Coverage sensitivity")
    axes[1].plot(
        lambda_table["Lambda"], lambda_table["Mean_Level_Normalized_Width"],
        "s-", lw=2, color="#6A1B9A",
    )
    axes[1].set_xlabel(r"MAD multiplier $\lambda$")
    axes[1].set_ylabel("Mean normalized corridor width")
    axes[1].set_title("Width sensitivity")
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.suptitle("Figure 32. Dynamic MAD coverage-width trade-off")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    k_table = tables["kmeans_k_selection"]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(k_table["k"], k_table["Silhouette"], "o-", lw=2, color="#1565C0")
    best = k_table.loc[k_table["Silhouette"].idxmax()]
    ax.scatter([best["k"]], [best["Silhouette"]], s=110, color="#D32F2F", zorder=5)
    ax.annotate(
        f"Data-driven optimum: k={int(best['k'])}",
        (best["k"], best["Silhouette"]), xytext=(8, 10), textcoords="offset points",
    )
    ax.set_xlabel("Number of clusters k")
    ax.set_ylabel("Silhouette coefficient")
    ax.set_title("Figure 33. Data-driven K-means cluster-number diagnostic")
    ax.grid(alpha=0.25)
    fig.tight_layout()


def show_all_figures(panel, series_results, rank_summary, rmse_pivot, mae_pivot, sensitivity, tables):
    plot_dataset_overview(panel)
    plot_model_comparisons(series_results)
    plot_metric_heatmaps(rmse_pivot, mae_pivot)
    plot_rank_summary(rank_summary)
    plot_dynamic_corridors(series_results)
    plot_kmeans_diagnostics(series_results)
    plot_sensitivity(sensitivity)
    plot_additional_diagnostics(tables)


def _safe_save(save_callable, target: Path, attempts: int = 5) -> Path | None:
    """Save via a temporary file and atomic replace.

    On Windows a figure that is open in a viewer, Word, the Explorer preview pane
    or a synchronisation client can be locked, and writing to it raises
    OSError (Errno 13 or 22). The new file is first written under a temporary
    name and then moved over the target; on repeated failure it is kept under a
    '_new' suffix so that the run never stops and nothing is silently lost.
    """
    import os
    import time
    temporary = target.with_name(f"{target.stem}.tmp{target.suffix}")
    for attempt in range(attempts):
        try:
            save_callable(temporary)
            os.replace(temporary, target)
            return target
        except OSError:
            time.sleep(0.5 * (attempt + 1))
    fallback = target.with_name(f"{target.stem}_new{target.suffix}")
    try:
        if temporary.exists():
            os.replace(temporary, fallback)
        else:
            save_callable(fallback)
        print(f"WARNING: {target.name} is locked by another program; saved as {fallback.name}")
        return fallback
    except OSError as error:
        print(f"WARNING: could not save {target.name}: {error}")
        return None


def save_tables_and_figures(output_dir: Path, tables: dict[str, pd.DataFrame]):
    output_dir = output_dir.expanduser().resolve()
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        _safe_save(lambda path, t=table: t.to_csv(path, index=False, encoding="utf-8-sig"),
                   table_dir / f"{name}.csv")
    for number in plt.get_fignums():
        figure = plt.figure(number)
        safe = f"figure_{number:02d}"
        _safe_save(lambda path, fig=figure: fig.savefig(path, dpi=300, bbox_inches="tight", format="png"),
                   figure_dir / f"{safe}.png")
        _safe_save(lambda path, fig=figure: fig.savefig(path, bbox_inches="tight", format="pdf"),
                   figure_dir / f"{safe}.pdf")
    print(f"\nReproducibility outputs saved to: {output_dir}")


# =============================================================================
# 11. Entry point
# =============================================================================
if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    args = parse_arguments()
    TEMPORAL_PROTOCOL = args.protocol
    INITIAL_TRAIN_YEARS, MAD_CALIBRATION_FORECASTS = PROTOCOLS[TEMPORAL_PROTOCOL]
    output_dir = args.output_dir if args.output_dir is not None else OUTPUT_DIRS[TEMPORAL_PROTOCOL]
    input_path = resolve_input_path(args.input)
    panel_df = load_and_validate_panel(input_path, args.sheet)

    (
        metrics_df,
        best_df,
        year_df,
        bandwidth_df,
        selection_df,
        pooled_kmeans_df,
        kmeans_comparison,
        SERIES_RESULTS,
        total_lwlr_inner_fits,
        total_all_inner_fits,
    ) = analyze_panel(panel_df)

    rank_summary, rmse_pivot, mae_pivot, sensitivity, result_tables = print_complete_results(
        input_path,
        panel_df,
        metrics_df,
        best_df,
        year_df,
        bandwidth_df,
        selection_df,
        pooled_kmeans_df,
        kmeans_comparison,
        SERIES_RESULTS,
        total_lwlr_inner_fits,
        total_all_inner_fits,
        args.sheet,
    )

    show_all_figures(
        panel_df, SERIES_RESULTS, rank_summary, rmse_pivot, mae_pivot, sensitivity,
        result_tables,
    )
    if args.save_results:
        save_tables_and_figures(output_dir, result_tables)
    if not args.no_show:
        plt.show()
