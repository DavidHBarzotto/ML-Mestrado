"""Random Forest + XGBoost training and curve prediction for creep and
shrinkage, ported from XGBoost_NR.ipynb / XGBoost_shrinkage_NR.ipynb.

Hyperparameters below are the *final* ones already chosen in those notebooks
via GridSearchCV -- the search itself is not re-run here (it was already run
once by the author; re-running it on every app start would be slow and adds
no value since the winning params are fixed).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.ndimage import gaussian_filter1d
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from app import data_pipeline as dp

RF_PARAMS = dict(n_estimators=100, random_state=42)

XGB_PARAMS_CREEP = dict(
    n_estimators=500, learning_rate=0.1, max_depth=5, subsample=1.0,
    colsample_bytree=0.8, objective="reg:squarederror", random_state=42,
)
XGB_PARAMS_SHRINK = dict(
    n_estimators=1000, learning_rate=0.1, max_depth=3, subsample=1.0,
    colsample_bytree=1.0, objective="reg:squarederror", random_state=42,
)

FEATURE_ORDER_CREEP = [
    "log_x3", "x17", "x18", "x19", "x43", "x45", "x47", "x49", "x50",
    "x52", "x55", "x57", "x20_N_R", "x20_RS", "x20_SL",
]
FEATURE_ORDER_SHRINK = [
    "log_x3", "x16", "x17", "x18", "x42", "x44", "x46", "x48", "x49",
    "x51", "x53", "x55", "x19_N_R", "x19_RS", "x19_SL",
]


def _metrics(y_true, y_pred) -> dict:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def converter_para_J(y_log, j0_values) -> np.ndarray:
    """Reverts y = ln(J/J0 + 1) back to absolute J."""
    j_div_j0 = np.exp(np.asarray(y_log, dtype=float)) - 1
    return j_div_j0 * np.asarray(j0_values, dtype=float)


def _train_creep() -> dict:
    df = dp.build_creep_dataset(merge_cement_nr=True)
    X = df[FEATURE_ORDER_CREEP]
    y = df["ln_J_div_J0"]
    trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.3, random_state=42)
    j0_test = df.loc[testX.index, "J0"]

    rf = RandomForestRegressor(**RF_PARAMS).fit(trainX, trainY)
    rf_pred_J = converter_para_J(rf.predict(testX), j0_test)
    rf_metrics = _metrics(converter_para_J(testY, j0_test), rf_pred_J)

    scaler = MinMaxScaler().fit(trainX)
    trainX_s = scaler.transform(trainX)
    testX_s = scaler.transform(testX)
    xgb_model = xgb.XGBRegressor(**XGB_PARAMS_CREEP).fit(trainX_s, trainY)
    xgb_pred_J = converter_para_J(xgb_model.predict(testX_s), j0_test)
    xgb_metrics = _metrics(converter_para_J(testY, j0_test), xgb_pred_J)

    return {
        "rf": {"model": rf, "scaler": None, "metrics": rf_metrics},
        "xgb": {"model": xgb_model, "scaler": scaler, "metrics": xgb_metrics},
        "feature_order": FEATURE_ORDER_CREEP,
    }


def _train_shrinkage() -> dict:
    df = dp.build_shrinkage_dataset(merge_cement_nr=True)
    X = df[FEATURE_ORDER_SHRINK]
    y = df["x4"].abs()
    trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.3, random_state=42)

    rf = RandomForestRegressor(**RF_PARAMS).fit(trainX, trainY)
    rf_metrics = _metrics(testY, rf.predict(testX))

    scaler = MinMaxScaler().fit(trainX)
    trainX_s = scaler.transform(trainX)
    testX_s = scaler.transform(testX)
    xgb_model = xgb.XGBRegressor(**XGB_PARAMS_SHRINK).fit(trainX_s, trainY)
    xgb_metrics = _metrics(testY, xgb_model.predict(testX_s))

    return {
        "rf": {"model": rf, "scaler": None, "metrics": rf_metrics},
        "xgb": {"model": xgb_model, "scaler": scaler, "metrics": xgb_metrics},
        "feature_order": FEATURE_ORDER_SHRINK,
    }


def train_all() -> dict:
    return {"creep": _train_creep(), "shrinkage": _train_shrinkage()}


def predict_curve(quantity: str, model_key: str, artifacts: dict, tempos: np.ndarray,
                   props: dict, cement_type: str) -> np.ndarray:
    """Sweeps `tempos` (days) holding every other property fixed and returns
    the predicted curve (J in microstrain/MPa for creep, strain in
    microstrain for shrinkage), mirroring predict_xgboost_puro from
    XGBoost_NR.ipynb but generalized to RF and to both quantities.
    """
    entry = artifacts[quantity][model_key]
    model = entry["model"]
    scaler = entry["scaler"]
    feature_order = artifacts[quantity]["feature_order"]

    colmap = dp.CREEP_COLMAP if quantity == "creep" else dp.SHRINK_COLMAP
    prefix = dp.CREEP_CEMENT_PREFIX if quantity == "creep" else dp.SHRINK_CEMENT_PREFIX
    # Matches the transform actually used when *training* each model:
    # creep used log10(t+1), shrinkage used ln(t+1) (see data_pipeline.py).
    log_fn = np.log10 if quantity == "creep" else np.log

    rows = []
    for t in tempos:
        row = dp.make_feature_row(colmap, props, prefix, dp.ML_CEMENT_CATS, cement_type, merge_nr=True)
        row["log_x3"] = log_fn(t + 1)
        rows.append(row)
    X_plot = pd.DataFrame(rows).reindex(columns=feature_order, fill_value=0.0)

    X_in = scaler.transform(X_plot) if scaler is not None else X_plot
    y_pred = model.predict(X_in)
    tempos = np.asarray(tempos, dtype=float)

    if quantity == "creep":
        if len(y_pred) > 1:
            y_pred = gaussian_filter1d(y_pred, sigma=3)
        e28 = props["e28"]
        j0 = (1.0 / e28) * 1e6
        predictions_J = (np.exp(y_pred) - 1) * j0

        mask_imediato = tempos < 0.1
        predictions_J[mask_imediato] = j0

        mask_trans = (tempos >= 0.1) & (tempos < 1.0)
        if np.any(mask_trans):
            peso = (tempos[mask_trans] - 0.1) / 0.9
            predictions_J[mask_trans] = (1 - peso) * j0 + peso * predictions_J[mask_trans]

        return np.maximum.accumulate(predictions_J)

    if len(y_pred) > 1:
        y_pred = gaussian_filter1d(y_pred, sigma=1.5)
    return y_pred
