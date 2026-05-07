"""
mlflow_logger.py — Logging de experimentos en MLflow.

Registra params, métricas y artefactos de cada modelo.
Al abrir la UI de MLflow se puede comparar todos los experimentos
y entender las decisiones tomadas.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import mlflow
import mlflow.pytorch
import mlflow.sklearn

from titanic_survival.models.base import EvaluationResult

logger = logging.getLogger(__name__)


def setup_mlflow(tracking_uri: str, experiment_name: str) -> str:
    """
    Configura MLflow y retorna el experiment_id.

    Parameters
    ----------
    tracking_uri : str
        URI del servidor de tracking (ej. 'file:./mlruns').
    experiment_name : str
        Nombre del experimento.

    Returns
    -------
    str
        ID del experimento.
    """
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    exp = mlflow.get_experiment_by_name(experiment_name)
    logger.info(
        "MLflow configurado: experiment=%s id=%s",
        experiment_name, exp.experiment_id,
    )
    return exp.experiment_id


def log_sklearn_run(
    result: EvaluationResult,
    best_params: dict,
    model_path: Path,
    config: dict,
    run_name: str = "sklearn_logistic_regression",
) -> str:
    """
    Registra un run de sklearn en MLflow.

    Returns
    -------
    str
        run_id del experimento registrado.
    """
    with mlflow.start_run(run_name=run_name) as run:
        # Params: hiperparámetros del modelo
        mlflow.log_params({
            "model_type":    "LogisticRegression",
            "search_method": "GridSearchCV",
            "cv_folds":      config.get("sklearn", {}).get("search", {}).get("cv_folds", 5),
            **{k.replace("classifier__", ""): str(v) for k, v in best_params.items()},
        })

        # Métricas finales
        mlflow.log_metrics({
            "f1":            result.f1,
            "auroc":         result.auroc,
            "recall":        result.recall,
            "precision":     result.precision,
            "accuracy":      result.accuracy,
            "avg_precision": result.avg_precision,
        })

        # Tags
        mlflow.set_tags({
            "model_family":   "sklearn",
            "dataset":        "titanic",
            "validated_with": "pandera",
        })

        # Artefacto: modelo serializado
        mlflow.log_artifact(str(model_path))

        run_id = run.info.run_id
        logger.info("MLflow sklearn run registrado: %s", run_id)
    return run_id


def log_pytorch_run(
    result: EvaluationResult,
    history: dict,
    model_path: Path,
    config: dict,
    run_name: str = "pytorch_mlp",
) -> str:
    """
    Registra un run de PyTorch en MLflow con métricas por época.

    Returns
    -------
    str
        run_id del experimento registrado.
    """
    pt_cfg = config.get("pytorch", {})
    train_cfg = pt_cfg.get("training", {})

    with mlflow.start_run(run_name=run_name) as run:
        # Params
        mlflow.log_params({
            "model_type":       "TitanicMLP",
            "hidden_dims":      str(pt_cfg.get("architecture", {}).get("hidden_dims")),
            "activation":       pt_cfg.get("architecture", {}).get("activation", "gelu"),
            "optimizer":        pt_cfg.get("optimizer", "AdamW"),
            "scheduler":        pt_cfg.get("scheduler", {}).get("name", "CosineAnnealingLR"),
            "lr":               train_cfg.get("learning_rate", 1e-3),
            "weight_decay":     train_cfg.get("weight_decay", 1e-2),
            "epochs":           train_cfg.get("epochs", 80),
            "batch_size":       train_cfg.get("batch_size", 32),
            "pos_weight_factor": train_cfg.get("pos_weight_factor", 2.0),
            "clip_norm":        train_cfg.get("clip_norm", 1.0),
            "early_stopping_patience": config.get("pytorch", {}).get(
                "early_stopping", {}).get("patience", 10),
        })

        # Métricas por época
        for epoch, (tl, vl) in enumerate(
            zip(history["train_loss"], history["val_loss"]), start=1
        ):
            mlflow.log_metrics({"train_loss": tl, "val_loss": vl}, step=epoch)

        # Métricas finales
        mlflow.log_metrics({
            "f1":            result.f1,
            "auroc":         result.auroc,
            "recall":        result.recall,
            "precision":     result.precision,
            "accuracy":      result.accuracy,
            "avg_precision": result.avg_precision,
            "best_epoch":    history.get("best_epoch", -1),
        })

        mlflow.set_tags({
            "model_family":   "pytorch",
            "dataset":        "titanic",
            "validated_with": "pandera",
            "device":         "cuda" if history.get("use_amp") else "cpu",
        })

        # Artefacto: modelo serializado
        mlflow.log_artifact(str(model_path))

        run_id = run.info.run_id
        logger.info("MLflow PyTorch run registrado: %s", run_id)
    return run_id
