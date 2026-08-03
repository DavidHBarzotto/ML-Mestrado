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
from sklearn.model_selection import train_test_split

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


def _split_calibrated_metrics(df, y, predict_fn) -> dict:
    """R²/RMSE/MAE on a held-out 30% test split (test_size=0.3,
    random_state=42 -- the same split used everywhere else in this app,
    including the RF/XGBoost training), after a linear (scale + bias)
    calibration fit ONLY on the training split and applied to the test
    split. This matters more than it may look: with only ~8000 rows and a
    formula whose absolute scale doesn't match this database's convention,
    a handful of extreme-error rows (some very-long-duration specimens the
    closed-form formulas fit badly) can swing a *whole-dataset* R² by more
    than a full point depending on how they happen to fall across a random
    split. Evaluating strictly out-of-sample, the same way the ML models
    are evaluated, gives a number that isn't an artifact of which random
    subset you happen to look at.
    """
    trainX, testX, trainY, testY = train_test_split(df, y, test_size=0.3, random_state=42)

    pred_train = np.asarray(predict_fn(trainX), dtype=float)
    pred_test = np.asarray(predict_fn(testX), dtype=float)
    trainY_arr = np.asarray(trainY, dtype=float)
    testY_arr = np.asarray(testY, dtype=float)

    mask_train = np.isfinite(pred_train) & np.isfinite(trainY_arr)
    mask_test = np.isfinite(pred_test) & np.isfinite(testY_arr)

    reg = LinearRegression().fit(pred_train[mask_train].reshape(-1, 1), trainY_arr[mask_train])
    y_calibrated_test = reg.predict(pred_test[mask_test].reshape(-1, 1))
    return _reg_metrics(testY_arr[mask_test], y_calibrated_test)


def evaluate_formulas() -> dict:
    """R²/RMSE/MAE of each closed-form model on a held-out test split (see
    `_split_calibrated_metrics`) -- these are physics formulas, not fitted
    to this dataset, so a linear scale/bias calibration (fit on train only)
    is applied first, matching what the original notebooks do; this makes
    ABNT and B4 directly comparable to each other and to the RF/XGBoost
    test metrics reported elsewhere in the app.
    """
    results = {}

    df_c = dp.build_creep_dataset(merge_cement_nr=False)
    y_c = df_c["x4"].abs()
    results["creep_abnt"] = _split_calibrated_metrics(df_c, y_c, lambda d: f.fluencia_abnt(d)["J_total"])
    results["creep_b3"] = _split_calibrated_metrics(df_c, y_c, f.calcular_fluencia_B3_corrigido)
    results["creep_b4"] = _split_calibrated_metrics(df_c, y_c, f.calcular_fluencia_B4)

    df_s = dp.build_shrinkage_dataset(merge_cement_nr=False)
    y_s = df_s["x4"].abs()
    results["shrinkage_abnt"] = _split_calibrated_metrics(df_s, y_s, lambda d: f.modelo_abnt(d)["ecs_ue"])
    results["shrinkage_b3"] = _split_calibrated_metrics(df_s, y_s, f.calcular_retracao_b3_corrigido)
    results["shrinkage_b4"] = _split_calibrated_metrics(df_s, y_s, f.calcular_retracao_b4_baseline_tc)

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
