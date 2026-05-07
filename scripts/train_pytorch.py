"""
scripts/train_pytorch.py — Entrena TitanicMLP y registra en MLflow.

Uso::

    python scripts/train_pytorch.py
    python scripts/train_pytorch.py --config configs/model_config.yaml --amp
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
logger = logging.getLogger("scripts.train_pytorch")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline as SKPipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from titanic_survival.data.config import load_model_config
from titanic_survival.evaluation.metrics import evaluate_model
from titanic_survival.evaluation.mlflow_logger import log_pytorch_run, setup_mlflow
from titanic_survival.features.engineering import (
    CATEGORICAL_FEATURES,
    FEATURES,
    NUMERIC_FEATURES,
    TARGET,
)
from titanic_survival.models.pytorch_model import PytorchSurvivalModel


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Entrena TitanicMLP con PyTorch")
    p.add_argument("--config", default="configs/model_config.yaml")
    p.add_argument("--amp", action="store_true", help="Mixed precision bfloat16 (RTX 4070)")
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    config = load_model_config(args.config)

    seed = config["project"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ── 1. Cargar datos
    data_path = Path(config["data"]["processed_path"])
    if data_path.is_dir():
        data_path = data_path / "train_fe.parquet"
    df = pd.read_parquet(data_path)
    logger.info("Datos cargados: %d filas", len(df))

    X = df[FEATURES]
    y = df[TARGET].values

    # ── 2. Split 70/15/15
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=seed,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=0.15 / 0.85, stratify=y_tv, random_state=seed,
    )
    logger.info("Train=%d | Val=%d | Test=%d", len(X_train), len(X_val), len(X_test))

    # ── 3. Preprocesador sklearn (ajustado SOLO en train)
    num_pipe = SKPipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    cat_pipe = SKPipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe",     OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer([
        ("num", num_pipe, NUMERIC_FEATURES),
        ("cat", cat_pipe, CATEGORICAL_FEATURES),
    ], remainder="drop")

    X_train_np = preprocessor.fit_transform(X_train)
    X_val_np   = preprocessor.transform(X_val)
    X_test_np  = preprocessor.transform(X_test)

    in_dim = X_train_np.shape[1]
    logger.info("IN_DIM=%d features después del preprocesamiento", in_dim)

    # ── 4. Configurar y entrenar
    pt_cfg   = config["pytorch"]
    arch     = pt_cfg["architecture"]
    train_cfg = pt_cfg["training"]
    use_amp  = args.amp and torch.cuda.is_available()

    model = PytorchSurvivalModel(
        in_dim=in_dim,
        hidden_dims=tuple(arch["hidden_dims"]),
        dropouts=tuple(arch["dropouts"]),
    )
    logger.info("Device: %s", model.device)

    # Combinar train+val para el entrenamiento completo
    X_trainval = np.vstack([X_train_np, X_val_np])
    y_trainval = np.concatenate([y_train, y_val])

    history = model.train(
        X=X_trainval,
        y=y_trainval,
        pos_weight_factor=train_cfg.get("pos_weight_factor", 2.0),
        epochs=train_cfg.get("epochs", 80),
        batch_size=train_cfg.get("batch_size", 32),
        lr=train_cfg.get("learning_rate", 1e-3),
        weight_decay=train_cfg.get("weight_decay", 1e-2),
        clip_norm=train_cfg.get("clip_norm", 1.0),
        use_amp=use_amp,
        early_stopping_patience=pt_cfg.get("early_stopping", {}).get("patience", 10),
        seed=seed,
    )

    # ── 5. Evaluar en test
    # Crear wrapper de predict para pasar arrays numpy directamente
    X_test_df = pd.DataFrame(X_test_np)
    result = evaluate_model(model, X_test_df, y_test, model_name="TitanicMLP")
    logger.info("\n%s", result)

    # ── 6. Guardar modelo
    out_path = Path(pt_cfg["output_path"])
    model.save(out_path)

    # También guardar el preprocesador para usarlo en la app
    import joblib
    prep_path = out_path.parent / "pytorch_preprocessor.joblib"
    with prep_path.open("wb") as fh:
        joblib.dump(preprocessor, fh)
    logger.info("Preprocesador guardado: %s", prep_path)

    # ── 7. Guardar métricas JSON (para DVC)
    metrics_file = out_path.parent / "metrics_pytorch.json"
    metrics_data = result.to_dict()
    metrics_data["best_epoch"] = history.get("best_epoch", -1)
    with metrics_file.open("w") as fh:
        json.dump(metrics_data, fh, indent=2)

    # ── 8. Registrar en MLflow
    mlflow_cfg = config.get("mlflow", {})
    setup_mlflow(
        mlflow_cfg.get("tracking_uri", "file:./mlruns"),
        mlflow_cfg.get("experiment_name", "titanic-survival-predictor"),
    )
    run_id = log_pytorch_run(
        result=result,
        history=history,
        model_path=out_path,
        config=config,
    )

    logger.info("✓ Entrenamiento PyTorch completado. MLflow run_id=%s", run_id)
    logger.info("  F1=%.4f | AUC=%.4f | Recall=%.4f", result.f1, result.auroc, result.recall)


if __name__ == "__main__":
    main()
