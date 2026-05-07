"""
scripts/train_sklearn.py — Entrena LogisticRegression y registra en MLflow.

Uso::

    python scripts/train_sklearn.py
    python scripts/train_sklearn.py --config configs/model_config.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("scripts.train_sklearn")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from titanic_survival.data.config import load_model_config
from titanic_survival.evaluation.metrics import evaluate_model
from titanic_survival.evaluation.mlflow_logger import log_sklearn_run, setup_mlflow
from titanic_survival.features.engineering import FEATURES, TARGET
from titanic_survival.models.sklearn_model import SklearnConfig, SklearnSurvivalModel


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Entrena LogisticRegression con GridSearchCV")
    p.add_argument("--config", default="configs/model_config.yaml")
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    config = load_model_config(args.config)

    seed      = config["project"]["seed"]
    np.random.seed(seed)

    # ── 1. Cargar datos procesados
    data_path = Path(config["data"]["processed_path"])
    if data_path.is_dir():
        data_path = data_path / "train_fe.parquet"
    df = pd.read_parquet(data_path)
    logger.info("Datos cargados: %d filas", len(df))

    X = df[FEATURES]
    y = df[TARGET]

    # ── 2. Split estratificado 70/15/15
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=seed,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=0.15 / 0.85, stratify=y_tv, random_state=seed,
    )
    logger.info("Train=%d | Val=%d | Test=%d", len(X_train), len(X_val), len(X_test))

    # ── 3. Configurar y entrenar
    sk_config = config["sklearn"]
    hp = sk_config["hyperparameters"]
    model_config = SklearnConfig(
        C=hp.get("C", 1.0),
        max_iter=hp.get("max_iter", 2000),
        class_weight=hp.get("class_weight", "balanced"),
        solver=hp.get("solver", "lbfgs"),
        penalty=hp.get("penalty", "l2"),
        cv_folds=sk_config["search"].get("cv_folds", 5),
        scoring=sk_config["search"].get("scoring", "f1"),
        random_state=seed,
    )
    model = SklearnSurvivalModel(config=model_config)

    param_grid = sk_config["search"].get("param_grid", None)
    if param_grid:
        param_grid = {f"classifier__{k}": v for k, v in param_grid.items()}
    model.train(X_train, y_train, param_grid=param_grid)

    # ── 4. Evaluar en test set
    result = evaluate_model(model, X_test, y_test, model_name="LogisticRegression")
    logger.info("\n%s", result)

    # ── 5. Guardar modelo
    out_path = Path(config["sklearn"]["output_path"])
    model.save(out_path)

    # ── 6. Guardar métricas JSON (para DVC)
    metrics_file = out_path.with_name("metrics_sklearn.json")
    with metrics_file.open("w") as fh:
        json.dump(result.to_dict(), fh, indent=2)

    # ── 7. Registrar en MLflow
    mlflow_cfg = config.get("mlflow", {})
    setup_mlflow(
        mlflow_cfg.get("tracking_uri", "file:./mlruns"),
        mlflow_cfg.get("experiment_name", "titanic-survival-predictor"),
    )
    run_id = log_sklearn_run(
        result=result,
        best_params=model.best_params,
        model_path=out_path,
        config=config,
    )

    logger.info("✓ Entrenamiento sklearn completado. MLflow run_id=%s", run_id)
    logger.info("  F1=%.4f | AUC=%.4f | Recall=%.4f", result.f1, result.auroc, result.recall)


if __name__ == "__main__":
    main()
