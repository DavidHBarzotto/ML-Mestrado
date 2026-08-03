"""Standalone training script.

Run once (and again whenever Creep.csv / Shrinkage.csv change):

    python app/train_models.py

Builds the RF + XGBoost models for creep and shrinkage (hyperparameters are
already fixed -- see ml_models.py -- so this does NOT re-run GridSearchCV)
and writes them to models/artifacts.joblib, which the Streamlit app loads
on startup.
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import ml_models as ml

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def main() -> None:
    MODELS_DIR.mkdir(exist_ok=True)

    print("Treinando modelos de fluencia e retracao (RF + XGBoost)...")
    ml_artifacts = ml.train_all()

    joblib.dump(ml_artifacts, MODELS_DIR / "artifacts.joblib")
    print(f"Modelos salvos em {MODELS_DIR}/artifacts.joblib")


if __name__ == "__main__":
    main()
