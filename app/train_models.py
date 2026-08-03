"""Standalone training script.

Run once (and again whenever Creep.csv / Shrinkage.csv change):

    python app/train_models.py

Builds the RF + XGBoost models for creep and shrinkage (hyperparameters are
already fixed -- see ml_models.py -- so this does NOT re-run GridSearchCV),
evaluates the closed-form formula models against the full cleaned dataset,
and writes everything the Streamlit app needs into models/:

    models/artifacts.joblib   -- RF/XGBoost models + scalers (dict)
    models/metrics.json       -- R²/RMSE/MAE for all 10 model x quantity pairs
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import data_pipeline as dp
from app import formulas as f
from app import ml_models as ml

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def _reg_metrics(y_true, y_pred) -> dict:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def _calibrated_metrics(y_true, y_pred_raw) -> dict:
    """R²/RMSE/MAE after a linear (scale + bias) calibration of the raw
    formula output against the target -- this is exactly the "R² após
    ajuste linear" step the B4 sections of the original notebooks compute
    (`LinearRegression().fit(J_train, y)`), used there because B4's absolute
    scale doesn't line up with this specific database's convention even when
    its overall shape/correlation is reasonable. This only affects the
    reported metric, not the curve the app predicts for user-entered inputs.
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred_raw, dtype=float)
    mask = np.isfinite(y_true_arr) & np.isfinite(y_pred_arr)
    reg = LinearRegression().fit(y_pred_arr[mask].reshape(-1, 1), y_true_arr[mask])
    y_calibrated = reg.predict(y_pred_arr[mask].reshape(-1, 1))
    return _reg_metrics(y_true_arr[mask], y_calibrated)


def evaluate_formulas() -> dict:
    """R²/RMSE/MAE of each closed-form model against the full cleaned
    database (these are physics formulas, not fitted to this dataset, so
    this is a "how well does the norm/standard match reality" check, not a
    held-out ML test score). ABNT and B4 are BOTH reported after the same
    linear calibration (scale + bias) their own notebooks used -- comparing
    a raw formula's R² against a calibrated one isn't apples-to-apples,
    since a raw output can carry a scale/bias mismatch that swamps the R²
    even when the underlying shape/correlation is fine.
    """
    results = {}

    df_c = dp.build_creep_dataset(merge_cement_nr=False)
    y_c = df_c["x4"].abs()
    results["creep_abnt"] = _calibrated_metrics(y_c, f.fluencia_abnt(df_c)["J_total"])
    results["creep_b3"] = _calibrated_metrics(y_c, f.calcular_fluencia_B3_corrigido(df_c))
    results["creep_b4"] = _calibrated_metrics(y_c, f.calcular_fluencia_B4(df_c))

    df_s = dp.build_shrinkage_dataset(merge_cement_nr=False)
    y_s = df_s["x4"].abs()
    results["shrinkage_abnt"] = _calibrated_metrics(y_s, f.modelo_abnt(df_s)["ecs_ue"])
    results["shrinkage_b3"] = _calibrated_metrics(y_s, f.calcular_retracao_b3_corrigido(df_s))
    results["shrinkage_b4"] = _calibrated_metrics(y_s, f.calcular_retracao_b4_baseline_tc(df_s))

    return results


def main() -> None:
    MODELS_DIR.mkdir(exist_ok=True)

    print("Treinando modelos de fluencia e retracao (RF + XGBoost)...")
    ml_artifacts = ml.train_all()

    print("Avaliando modelos de formula (ABNT / B3 / B4)...")
    formula_metrics = evaluate_formulas()

    joblib.dump(ml_artifacts, MODELS_DIR / "artifacts.joblib")

    metrics = {"formulas": formula_metrics}
    for quantity, entry in ml_artifacts.items():
        metrics[quantity] = {
            "rf": entry["rf"]["metrics"],
            "xgb": entry["xgb"]["metrics"],
        }

    with open(MODELS_DIR / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, ensure_ascii=False)

    print(f"Modelos salvos em {MODELS_DIR}/artifacts.joblib")
    print(f"Metricas salvas em {MODELS_DIR}/metrics.json")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
