"""Shared data loading/cleaning pipeline for the NU-ITI creep and shrinkage
databases, replicating the preprocessing that is repeated in every notebook
(CreepABNT, CreepB4, ShrinkageABNT, ShrinkageB4, XGBoost_NR, XGBoost_shrinkage_NR).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
CREEP_CSV = str(_ROOT / "Creep.csv")
SHRINKAGE_CSV = str(_ROOT / "Shrinkage.csv")

# --- Creep (Creep.csv) -------------------------------------------------

CREEP_NUMERIC_COLS = [
    "x3", "x4", "x17", "x18", "x19", "x43", "x45", "x47", "x49", "x50", "x52", "x55", "x57",
]
CREEP_IQR_COLS = ["ln_J_div_J0", "log_x3", "x17", "x19", "x43", "x57", "x18", "x45"]

# semantic name -> raw column, shared by every creep model/formula
CREEP_COLMAP = {
    "duration": "x3",      # duration of loading, t - t0 (days)
    "wc": "x17",           # water/cement
    "ac": "x18",           # aggregate/cement
    "cement_kg": "x19",    # cement content (kg/m3)
    "fc28": "x43",         # compressive strength at 28d (MPa)
    "e28": "x45",          # elastic modulus at 28d (MPa)
    "length_radius": "x47",
    "height": "x49",
    "vs_ratio": "x50",     # volume/surface ratio
    "t0": "x52",           # age at loading (days)
    "temp": "x55",         # temperature (degC)
    "humidity": "x57",     # environment humidity (%)
}
CREEP_CEMENT_PREFIX = "x20"
CREEP_CEMENT_FULL_CATS = ["N", "R", "RS", "SL"]

# --- Shrinkage (Shrinkage.csv) ------------------------------------------

SHRINK_NUMERIC_COLS = [
    "x3", "x4", "x16", "x17", "x18", "x42", "x44", "x46", "x48", "x49", "x51", "x53", "x55",
]
SHRINK_IQR_COLS = ["x4", "log_x3", "x16", "x18", "x42"]

SHRINK_COLMAP = {
    "duration": "x3",      # duration of drying, t - tc (days)
    "wc": "x16",
    "ac": "x17",
    "cement_kg": "x18",
    "fc28": "x42",
    "e28": "x44",
    "length_radius": "x46",
    "height": "x48",
    "vs_ratio": "x49",
    "t0": "x51",           # age at start of drying, tc (days)
    "temp": "x53",         # curing temperature (degC)
    "humidity": "x55",
}
SHRINK_CEMENT_PREFIX = "x19"
# 'N' has no dummy of its own in the shrinkage notebooks: it is the implicit
# reference category (all dummies == 0).
SHRINK_CEMENT_FULL_CATS = ["R", "RS", "SL"]

# Both quantities merge N and R into a single category for the ML (RF/XGBoost)
# feature set, matching XGBoost_NR.ipynb / XGBoost_shrinkage_NR.ipynb.
ML_CEMENT_CATS = ["N_R", "RS", "SL"]

E28_FALLBACK_K = 4734.0  # used by the dataset-cleaning step (and by B3/B4) when E28 is missing


def _fix_decimal(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    return df


def _iqr_filter(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    q1 = df[cols].quantile(0.25)
    q3 = df[cols].quantile(0.75)
    iqr = q3 - q1
    mask = pd.Series(True, index=df.index)
    for c in cols:
        mask &= ~((df[c] < (q1[c] - 1.5 * iqr[c])) | (df[c] > (q3[c] + 1.5 * iqr[c])))
    return df[mask]


def load_creep_raw() -> pd.DataFrame:
    df = pd.read_csv(CREEP_CSV, sep=";")
    df.columns = [f"x{i}" for i in range(1, 64)]
    df = df.iloc[2:29197].copy()
    df = _fix_decimal(df, CREEP_NUMERIC_COLS)

    idx0 = df.groupby("x2")["x3"].idxmin()
    j0_map = df.loc[idx0].set_index("x2")["x4"]
    df["J0"] = df["x2"].map(j0_map)
    df["J_div_J0"] = df["x4"] / df["J0"]
    df["ln_J_div_J0"] = np.log(df["J_div_J0"] + 1)
    df["log_x3"] = np.log10(df["x3"] + 1)

    cond = df["x45"].isnull()
    df.loc[cond, "x45"] = E28_FALLBACK_K * df.loc[cond, "x43"] ** 0.5

    df = df[df["x15"].isin(["total", "total?"])].copy()
    df.dropna(subset=CREEP_NUMERIC_COLS, inplace=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df


def load_shrinkage_raw() -> pd.DataFrame:
    df = pd.read_csv(SHRINKAGE_CSV, sep=";")
    df.columns = [f"x{i}" for i in range(1, 60)]
    df = df.iloc[2:32320].copy()
    df = _fix_decimal(df, SHRINK_NUMERIC_COLS)

    df["log_x3"] = np.log(df["x3"] + 1)

    cond = df["x44"].isnull()
    df.loc[cond, "x44"] = E28_FALLBACK_K * df.loc[cond, "x42"] ** 0.5

    df = df[df["x15"].isin(["total", "total?"])].copy()
    df.dropna(subset=SHRINK_NUMERIC_COLS, inplace=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df


def build_creep_dataset(merge_cement_nr: bool) -> pd.DataFrame:
    """Returns df_modif equivalent used across the creep notebooks."""
    df = load_creep_raw()
    cement_col = df["x20"].replace({"N": "N_R", "R": "N_R"}) if merge_cement_nr else df["x20"]
    dummies = pd.get_dummies(cement_col, prefix=CREEP_CEMENT_PREFIX).astype(float)
    keep = CREEP_NUMERIC_COLS + ["J0", "ln_J_div_J0", "log_x3"]
    df_full = pd.concat([df[keep], dummies], axis=1)
    df_full = _iqr_filter(df_full, CREEP_IQR_COLS).dropna()
    return df_full


def build_shrinkage_dataset(merge_cement_nr: bool) -> pd.DataFrame:
    """Returns df_modif equivalent used across the shrinkage notebooks."""
    df = load_shrinkage_raw()
    cement_col = df["x19"].replace({"N": "N_R", "R": "N_R"}) if merge_cement_nr else df["x19"]
    dummies = pd.get_dummies(cement_col, prefix=SHRINK_CEMENT_PREFIX).astype(float)
    keep = SHRINK_NUMERIC_COLS + ["log_x3"]
    df_full = pd.concat([df[keep], dummies], axis=1)
    if not merge_cement_nr:
        df_full = df_full.drop(columns=[f"{SHRINK_CEMENT_PREFIX}_N"], errors="ignore")
    df_full = _iqr_filter(df_full, SHRINK_IQR_COLS).dropna()
    return df_full


def cement_label(cement_type: str, merge_nr: bool) -> str:
    if merge_nr and cement_type in ("N", "R"):
        return "N_R"
    return cement_type


def make_feature_row(
    colmap: dict,
    values: dict,
    cement_prefix: str,
    cement_categories: list[str],
    cement_type: str,
    merge_nr: bool,
) -> dict:
    """Builds one row of native (xN) feature columns from semantic property
    names + the selected cement type, matching whichever one-hot scheme
    (full N/R/RS/SL for the formula models, merged N_R/RS/SL for RF/XGBoost)
    the target model expects.
    """
    row = {colmap[k]: v for k, v in values.items() if k in colmap}
    label = cement_label(cement_type, merge_nr)
    for cat in cement_categories:
        row[f"{cement_prefix}_{cat}"] = 1.0 if cat == label else 0.0
    return row
