"""
metrics.py — Evaluación de modelos de clasificación binaria.

Calcula todas las métricas relevantes y devuelve un EvaluationResult
(dataclass tipado) que puede loguearse en MLflow o renderizarse en Streamlit.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from titanic_survival.models.base import BaseModel, EvaluationResult

logger = logging.getLogger(__name__)


def evaluate_model(
    model: BaseModel,
    X: "pd.DataFrame",  # noqa: F821
    y: "pd.Series",      # noqa: F821
    model_name: str = "Model",
    threshold: float = 0.5,
) -> EvaluationResult:
    """
    Evalúa un modelo sobre un conjunto de datos.

    Parameters
    ----------
    model : BaseModel
        Modelo entrenado que implementa predict() y predict_proba().
    X : pd.DataFrame
        Features de evaluación.
    y : pd.Series | np.ndarray
        Etiquetas verdaderas.
    model_name : str
        Nombre descriptivo para el resultado.
    threshold : float
        Umbral de decisión (default 0.5).

    Returns
    -------
    EvaluationResult
        Resultado con todas las métricas calculadas.
    """
    import pandas as pd

    y_true  = np.asarray(y)
    y_proba = model.predict_proba(X)
    y_pred  = (y_proba >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred)

    result = EvaluationResult(
        model_name=model_name,
        accuracy=accuracy_score(y_true, y_pred),
        f1=f1_score(y_true, y_pred, zero_division=0),
        auroc=roc_auc_score(y_true, y_proba),
        recall=recall_score(y_true, y_pred, zero_division=0),
        precision=precision_score(y_true, y_pred, zero_division=0),
        avg_precision=average_precision_score(y_true, y_proba),
        confusion_matrix=cm,
    )

    logger.info("Evaluación %s: F1=%.4f AUC=%.4f Recall=%.4f",
                model_name, result.f1, result.auroc, result.recall)
    return result


def compare_models(results: list[EvaluationResult]) -> "pd.DataFrame":
    """
    Genera una tabla comparativa de resultados de múltiples modelos.

    Parameters
    ----------
    results : list[EvaluationResult]
        Lista de resultados de evaluación.

    Returns
    -------
    pd.DataFrame
        Tabla con una fila por modelo y columnas de métricas.
    """
    import pandas as pd

    rows = [r.to_dict() for r in results]
    df = pd.DataFrame(rows).set_index("model_name")
    metric_cols = ["f1", "auroc", "recall", "precision", "accuracy", "avg_precision"]
    return df[[c for c in metric_cols if c in df.columns]]
