"""
sklearn_model.py — Pipeline de LogisticRegression con scikit-learn.

Implementa BaseModel usando un Pipeline de sklearn con:
    - ColumnTransformer (StandardScaler + OneHotEncoder)
    - LogisticRegression con class_weight='balanced'
    - GridSearchCV para búsqueda de hiperparámetros

El Pipeline garantiza que el preprocesador viaja con el modelo:
serializar el Pipeline es serializar TODO — no hay preprocesamiento
separado que se olvide en producción.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from titanic_survival.features.engineering import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from titanic_survival.models.base import BaseModel

logger = logging.getLogger(__name__)


@dataclass
class SklearnConfig:
    """Configuración del modelo sklearn — dataclass en vez de dict."""

    C: float = 1.0
    max_iter: int = 2000
    class_weight: str = "balanced"
    solver: str = "lbfgs"
    penalty: str = "l2"
    cv_folds: int = 5
    scoring: str = "f1"
    random_state: int = 42


class SklearnSurvivalModel(BaseModel):
    """
    Modelo de supervivencia basado en LogisticRegression + sklearn Pipeline.

    El Pipeline integra preprocesamiento + modelo en un solo objeto,
    garantizando que al serializar el modelo también se serializa el
    preprocesador — eliminando el riesgo de training-serving skew.

    Parameters
    ----------
    config : SklearnConfig
        Hiperparámetros del modelo.
    """

    def __init__(self, config: SklearnConfig | None = None) -> None:
        self.config = config or SklearnConfig()
        self._pipeline: Pipeline | None = None
        self._best_params: dict = {}
        self._cv_score: float = 0.0

    def _build_pipeline(self) -> Pipeline:
        """Construye el Pipeline completo (preprocesador + clasificador)."""
        numeric_transformer = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        categorical_transformer = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        preprocessor = ColumnTransformer(
            [
                ("num", numeric_transformer, NUMERIC_FEATURES),
                ("cat", categorical_transformer, CATEGORICAL_FEATURES),
            ],
            remainder="drop",
        )

        return Pipeline(
            [
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    LogisticRegression(
                        C=self.config.C,
                        max_iter=self.config.max_iter,
                        class_weight=self.config.class_weight,
                        solver=self.config.solver,
                        penalty=self.config.penalty,
                        random_state=self.config.random_state,
                    ),
                ),
            ]
        )

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        param_grid: dict | None = None,
    ) -> None:
        """
        Entrena el modelo con búsqueda de hiperparámetros via GridSearchCV.

        Parameters
        ----------
        X : pd.DataFrame
            Features de entrenamiento.
        y : pd.Series
            Etiquetas binarias (0/1).
        param_grid : dict, optional
            Espacio de búsqueda para GridSearchCV. Si None usa el default.
        """
        logger.info("Iniciando entrenamiento LogisticRegression...")

        if param_grid is None:
            param_grid = {
                "classifier__C": [0.01, 0.1, 1.0, 10.0],
                "classifier__penalty": ["l2"],
                "classifier__solver": ["lbfgs", "liblinear"],
            }

        n_combos = 1
        for v in param_grid.values():
            n_combos *= len(v)

        logger.info(
            "GridSearchCV: %d combinaciones × %d folds = %d fits",
            n_combos,
            self.config.cv_folds,
            n_combos * self.config.cv_folds,
        )

        pipeline = self._build_pipeline()
        skf = StratifiedKFold(
            n_splits=self.config.cv_folds,
            shuffle=True,
            random_state=self.config.random_state,
        )

        search = GridSearchCV(
            pipeline,
            param_grid=param_grid,
            cv=skf,
            scoring=self.config.scoring,
            n_jobs=-1,
            verbose=0,
            return_train_score=True,
        )
        search.fit(X, y)

        self._pipeline = search.best_estimator_
        self._best_params = search.best_params_
        self._cv_score = search.best_score_

        logger.info(
            "Entrenamiento completado. Mejor F1 CV=%.4f | params=%s",
            self._cv_score,
            self._best_params,
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predicciones de clase (0 o 1)."""
        if self._pipeline is None:
            raise RuntimeError("Modelo no entrenado. Llama train() primero.")
        return self._pipeline.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Probabilidades de la clase positiva (supervivencia)."""
        if self._pipeline is None:
            raise RuntimeError("Modelo no entrenado. Llama train() primero.")
        return self._pipeline.predict_proba(X)[:, 1]

    def save(self, path: str | Path) -> None:
        """
        Serializa el Pipeline completo con joblib.

        Usa context manager para garantizar el cierre correcto del archivo.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            joblib.dump(self._pipeline, fh)
        size_kb = path.stat().st_size / 1024
        logger.info("Modelo sklearn guardado: %s (%.1f KB)", path, size_kb)

        # Guardar métricas de entrenamiento junto al modelo
        metrics_path = path.with_suffix(".json")
        metrics = {
            "best_cv_f1": round(self._cv_score, 6),
            "best_params": self._best_params,
        }
        with metrics_path.open("w") as fh:
            json.dump(metrics, fh, indent=2)
        logger.info("Métricas sklearn guardadas: %s", metrics_path)

    @classmethod
    def load(cls, path: str | Path) -> SklearnSurvivalModel:
        """Carga un modelo serializado desde disco."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Modelo no encontrado: {path}")
        instance = cls()
        with path.open("rb") as fh:
            instance._pipeline = joblib.load(fh)
        logger.info("Modelo sklearn cargado desde: %s", path)
        return instance

    @property
    def best_params(self) -> dict:
        return self._best_params

    @property
    def cv_score(self) -> float:
        return self._cv_score
