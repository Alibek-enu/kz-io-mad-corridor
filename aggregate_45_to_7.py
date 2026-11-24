"""Rebuild the seven-sector Buy/Sell panel from the OECD IOTs 2021 TTL matrices (Kazakhstan, 1995-2018).

Input : data/oecd_iot2021_kaz_ttl/KAZ_TTL_<year>.csv  (45 x 45 intermediate flows, million USD;
        rows = supplying industry TTL_*, columns = using industry D*)
Output: data/rebuilt_7sector_buy_sell_panel.csv and a comparison with prepared_7sector_buy_sell_panel.xlsx

Definitions (Equation (1) of the paper), with S the 7 x 45 aggregation matrix:
    Buy_s  = 1e-3 * sum over rows i in sector s, all columns j   (row sums of S Z S^T)
    Sell_s = 1e-3 * sum over all rows i, columns j in sector s   (column sums of S Z S^T)
Within-sector flows (the diagonal of S Z S^T) are included in both indicators.

Usage: python aggregate_45_to_7.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "oecd_iot2021_kaz_ttl"
SECTOR_ORDER = ["AGR", "MNG", "MNF", "SRV", "CST", "TRD", "TRN"]
MAPPING = {
    "AGR": ["01T02", "03"],
    "MNG": ["05T06", "07T08", "09"],
    "MNF": ["10T12", "13T15", "16", "17T18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28",
            "29", "30", "31T33"],
    "SRV": ["35", "36T39", "55T56", "58T60", "61", "62T63", "64T66", "68", "69T75", "77T82", "84", "85",
            "86T88", "90T93", "94T96", "97T98"],
    "CST": ["41T43"],
    "TRD": ["45T47"],
    "TRN": ["49", "50", "51", "52", "53"],
}


def aggregation_matrix(codes):
    s = np.zeros((len(SECTOR_ORDER), len(codes)))
    for k, sec in enumerate(SECTOR_ORDER):
        for code in MAPPING[sec]:
            s[k, codes.index(code)] = 1.0
    assert np.all(s.sum(axis=0) == 1), "each industry must belong to exactly one sector"
    return s


rows = []
for year in range(1995, 2019):
    z = pd.read_csv(RAW / f"KAZ_TTL_{year}.csv", index_col=0)
    row_codes = [c.replace("TTL_", "") for c in z.index]
    col_codes = [c.replace("D", "", 1) for c in z.columns]
    assert row_codes == col_codes and len(row_codes) == 45
    S = aggregation_matrix(row_codes)
    zt = S @ z.to_numpy(float) @ S.T  # 7 x 7 aggregated matrix
    buy, sell = 1e-3 * zt.sum(axis=1), 1e-3 * zt.sum(axis=0)
    for k, sec in enumerate(SECTOR_ORDER):
        rows.append({"Year": year, "Sector_Code": sec, "Buy_USD_bn": buy[k], "Sell_USD_bn": sell[k],
                     "Within_sector_USD_bn": 1e-3 * zt[k, k]})

panel = pd.DataFrame(rows)
out = ROOT / "data" / "rebuilt_7sector_buy_sell_panel.csv"
panel.to_csv(out, index=False)
print(f"written {out.relative_to(ROOT)} ({len(panel)} rows)")

prepared = ROOT / "prepared_7sector_buy_sell_panel.xlsx"
if prepared.exists():
    p = pd.read_excel(prepared, sheet_name="Panel_Long")[["Year", "Sector_Code", "Buy_USD_bn", "Sell_USD_bn"]]
    m = panel.merge(p, on=["Year", "Sector_Code"], suffixes=("", "_prepared"))
    d_buy = (m.Buy_USD_bn - m.Buy_USD_bn_prepared).abs().max()
    d_sell = (m.Sell_USD_bn - m.Sell_USD_bn_prepared).abs().max()
    print(f"compared {len(m)} sector-years with {prepared.name}: max |diff| Buy = {d_buy:.2e}, Sell = {d_sell:.2e}")
