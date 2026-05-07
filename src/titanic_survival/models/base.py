"""
base.py — Interfaz común para todos los modelos del proyecto.

Patrón Strategy: BaseModel define el contrato que todos los modelos
deben cumplir. Esto permite intercambiar modelos (sklearn, PyTorch)
sin modificar el código que los consume (la app de Streamlit,
los scripts de evaluación, MLflow).

EvaluationResult usa dataclass en lugar de dict — con campos nombrados,
el IDE autocompleta correctamente y los errores de typo se detectan
antes de ejecutar.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Estructura de datos para resultados de evaluación
# Dataclass en lugar de dict: campos tipados, autodocumentados
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EvaluationResult:
    """
    Resultado de evaluación de un modelo de clasificación binaria.

    Usa dataclass en lugar de dict para que el IDE autocomplete
    correctamente y los errores de typo se detecten en tiempo de edición.

    Attributes
    ----------
    model_name : str
        Nombre descriptivo del modelo (ej. 'LogisticRegression', 'TitanicMLP').
    accuracy : float
        Exactitud global — engañosa con desbalance de clases.
    f1 : float
        F1-score — métrica principal con desbalance.
    auroc : float
        Área bajo la curva ROC — capacidad discriminativa.
    recall : float
        Recall de la clase positiva (supervivientes).
    precision : float
        Precisión de la clase positiva.
    avg_precision : float
        Average Precision (AUC-PR) — mejor con desbalance.
    confusion_matrix : np.ndarray
        Matriz de confusión [[TN, FP], [FN, TP]].
    extra : dict
        Métricas adicionales específicas del modelo (ej. training time).
    """

    model_name: str
    accuracy:       float = 0.0
    f1:             float = 0.0
    auroc:          float = 0.0
    recall:         float = 0.0
    precision:      float = 0.0
    avg_precision:  float = 0.0
    confusion_matrix: np.ndarray = field(default_factory=lambda: np.zeros((2, 2)))
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serializa el resultado a dict para MLflow y JSON."""
        return {
            "model_name":     self.model_name,
            "accuracy":       round(self.accuracy, 6),
            "f1":             round(self.f1, 6),
            "auroc":          round(self.auroc, 6),
            "recall":         round(self.recall, 6),
            "precision":      round(self.precision, 6),
            "avg_precision":  round(self.avg_precision, 6),
            **self.extra,
        }

    def __str__(self) -> str:
        return (
            f"{self.model_name}\n"
            f"  F1:       {self.f1:.4f}\n"
            f"  AUC-ROC:  {self.auroc:.4f}\n"
            f"  Recall:   {self.recall:.4f}\n"
            f"  Precision:{self.precision:.4f}\n"
            f"  Accuracy: {self.accuracy:.4f}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Clase abstracta — contrato común para todos los modelos
# ─────────────────────────────────────────────────────────────────────────────

class BaseModel(ABC):
    """
    Interfaz común para modelos de clasificación binaria.

    Patrón Strategy: el código cliente (app Streamlit, evaluación)
    trabaja contra esta interfaz — no conoce si el modelo interno
    es sklearn o PyTorch. Para agregar un nuevo modelo basta con
    implementar esta interfaz sin tocar nada más.

    Methods
    -------
    train(X, y) → None
        Entrena el modelo con los datos proporcionados.
    predict(X) → np.ndarray
        Devuelve predicciones de clase (0 o 1).
    predict_proba(X) → np.ndarray
        Devuelve probabilidades de la clase positiva (0 a 1).
    save(path) → None
        Serializa el modelo a disco.
    load(path) → BaseModel
        Carga un modelo serializado (classmethod).
    """

    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Entrena el modelo."""
        ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Devuelve predicciones de clase (0 o 1)."""
        ...

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Devuelve probabilidades de la clase positiva [0, 1]."""
        ...

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Serializa el modelo a disco."""
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> "BaseModel":
        """Carga un modelo serializado desde disco."""
        ...
