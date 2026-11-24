# Sequential Median–MAD Diagnostic Corridor and Leakage-Free Forecast Evaluation for Kazakhstan's Seven-Sector Input–Output Flows

Code, data and results accompanying the manuscript *“A Sequential Median–MAD Diagnostic Corridor for Short Sectoral
Input–Output Flows with Leakage-Free Forecast Evaluation: Evidence from Kazakhstan”* (submitted to *Mathematics*, MDPI).

The repository reproduces every table and figure of the paper from the original OECD input–output matrices.

## What the code does

* Builds two annual series per sector from the OECD Input–Output Tables (IOTs), 2021 edition, for Kazakhstan, 1995–2018:
  **Buy** — intermediate demand for the sector's output (row sums), and **Sell** — intermediate supply received by the
  sector (column sums), for seven sectors (AGR, MNG, MNF, SRV, CST, TRD, TRN): 14 series × 24 years.
* Evaluates one-step-ahead forecasts under the **9–8–7 temporal evaluation protocol**
  (training 1995–2003, residual calibration 2004–2011, diagnostics 2012–2018) with expanding-window
  rolling-origin validation and nested inner validation, so that no future information enters estimation,
  tuning, model selection or the corridor (leakage-free design).
* Compares Naïve, Drift, five global regressions, ETS (SES/Holt/damped Holt), Gaussian and exponential-kernel
  locally weighted linear regression (LWLR) and an adaptive rolling selector with an honest prequential inner score.
* Builds the **sequential median–MAD diagnostic corridor** from past out-of-sample residuals and classifies each
  diagnostic observation as Below / Near / Above trend; tests directional symmetry with exact sign tests.
* Statistical inference (Friedman, Kendall's W, Wilcoxon with Holm correction, rank-biserial correlation, sector-cluster
  bootstrap), K-means error regimes, ablation and disruption-period sensitivity.
* Monte Carlo experiments for the optimism of min-over-grid inner scores and for the coverage / shock-detection
  probability of the corridor.

## Repository structure

| Path | Content |
|---|---|
| `data/oecd_iot2021_kaz_ttl/KAZ_TTL_<year>.csv` | Original OECD IOTs 2021 TTL intermediate-flow matrices for Kazakhstan, 45 × 45, million USD, current prices (rows = supplying industry, columns = using industry) |
| `aggregate_45_to_7.py` | Aggregates the 45 × 45 matrices to 7 × 7 and rebuilds the Buy/Sell panel |
| `prepared_7sector_buy_sell_panel.xlsx` | Analysis panel used by the main script (billion USD) |
| `data/rebuilt_7sector_buy_sell_panel.csv` | Panel rebuilt by `aggregate_45_to_7.py` (identical to the analysis panel up to 2·10⁻¹⁴) |
| `data/SHA256SUMS.txt` | SHA-256 checksums of all data files |
| `mdpi_7sector_reviewer_ready_analysis-00.py` | Main analysis (all empirical tables and figures) |
| `mc_selector_bias-00.py` | Monte Carlo experiments (Proposition 1 and Proposition 5) |
| `article_figures-00.py` | Summary figures of the paper built from the CSV tables |
| `outputs-00/tables`, `outputs-00/figures` | Results of the final 9–8–7 run |
| `outputs-00/simulation` | Monte Carlo results (500 replications per scenario; corridor experiment 200,000 replications) |
| `outputs-00/article_figures` | Summary figures used in the paper |
| `outputs-00_protocol-5-5-14/tables` | Sensitivity run under the 5–5–14 protocol (not pooled with the 9–8–7 results) |

## Sector aggregation (ISIC Rev.4)

| Sector | OECD industries |
|---|---|
| Agriculture (AGR) | 01T02, 03 |
| Mining (MNG) | 05T06, 07T08, 09 |
| Manufacturing (MNF) | 10T12, 13T15, 16, 17T18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31T33 |
| Services (SRV) | 35, 36T39, 55T56, 58T60, 61, 62T63, 64T66, 68, 69T75, 77T82, 84, 85, 86T88, 90T93, 94T96, 97T98 |
| Construction (CST) | 41T43 |
| Trade (TRD) | 45T47 |
| Transportation (TRN) | 49, 50, 51, 52, 53 |

Within-sector flows (the diagonal of the aggregated matrix) are included in both Buy and Sell; they account for
31–37% of all intermediate flows.

## How to reproduce

Python 3.13 was used. Install the pinned packages and run the scripts from the repository root:

```bash
pip install -r requirements.txt

# 1. Rebuild the Buy/Sell panel from the OECD matrices and compare it with the analysis panel
python aggregate_45_to_7.py

# 2. Main analysis, 9-8-7 protocol (writes outputs-00/tables and outputs-00/figures)
python mdpi_7sector_reviewer_ready_analysis-00.py --save-results --no-show

# 3. Sensitivity run, 5-5-14 protocol (writes outputs-00_protocol-5-5-14)
python mdpi_7sector_reviewer_ready_analysis-00.py --protocol 5-5-14 --save-results --no-show

# 4. Monte Carlo experiments (writes outputs-00/simulation)
python mc_selector_bias-00.py --reps 500 --workers 2

# 5. Summary figures of the paper
python article_figures-00.py outputs-00/tables outputs-00/article_figures
```

The main script records the SHA-256 checksum of the input panel, the library versions, the random seed (42) and all
tuning parameters in `outputs-00/tables/run_metadata.csv` and `software_versions.csv`.

Expected SHA-256 of the analysis panel:
`dfd4227b1879c7ebd0d78bfa31d6d192e5eabe5ecc9f8df1b72f1c208d061f4e`

## Data source

OECD. *Input-Output Tables (IOTs)*, 2021 edition; variable TTL (total: domestic output plus imports);
country KAZ; unit US dollar, millions, current prices; years 1995–2018.
Available online: https://www.oecd.org/sti/ind/input-outputtables.htm

## Licence

Code: MIT (see `LICENSE`). Derived data, tables and figures: CC BY 4.0; original OECD values remain subject to the
OECD terms and conditions (see `LICENSE-DATA.md`).

## Funding

This research was funded by the Committee of Science of the Ministry of Science and Higher Education of the
Republic of Kazakhstan (Grant No. AP26196337).

## Citation

See `CITATION.cff`. The citation of the published article will be added after publication.
