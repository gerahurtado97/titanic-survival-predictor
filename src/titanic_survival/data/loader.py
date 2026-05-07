"""
loader.py — Carga y validación de datos del Titanic.

Responsabilidades:
    - Leer el CSV crudo con dtypes correctos
    - Aplicar el schema de Pandera antes de cualquier procesamiento
    - Proveer splits reproducibles (train/val/test estratificados)

Patrón: los datos NUNCA entran al pipeline sin pasar por el schema.
Si la validación falla, el pipeline falla con mensaje claro.
"""

from __future__ import annotations

import logging
import time
from functools import wraps
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Decorador @timer — registra el tiempo de ejecución de funciones críticas
# ─────────────────────────────────────────────────────────────────────────────


def timer(func: Any) -> Any:
    """
    Decorador que registra el tiempo de ejecución de una función.

    Propósito real: en pipelines de ML con datasets grandes, saber cuánto
    tarda cada etapa (carga, preprocesamiento, entrenamiento) es fundamental
    para optimizar y detectar regresiones de rendimiento.

    Examples
    --------
    >>> @timer
    ... def cargar_datos(ruta): ...
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        logger.info("[timer] %s completado en %.3f s", func.__name__, elapsed)
        return result

    return wrapper


# ─────────────────────────────────────────────────────────────────────────────
# Tipos y dtypes correctos del dataset Titanic
# ─────────────────────────────────────────────────────────────────────────────

TITANIC_DTYPES: dict[str, str] = {
    "PassengerId": "int64",
    "Survived": "int64",
    "Pclass": "int64",
    "Name": "string",
    "Sex": "string",
    "Age": "float64",
    "SibSp": "int64",
    "Parch": "int64",
    "Ticket": "string",
    "Fare": "float64",
    "Cabin": "string",
    "Embarked": "string",
}


@timer
def load_raw(path: str | Path) -> pd.DataFrame:
    """
    Carga el CSV crudo del Titanic con dtypes explícitos.

    Usar dtypes explícitos en lugar de inferencia automática garantiza
    que los tipos sean correctos sin importar el contenido del archivo,
    y detecta inconsistencias en los datos al momento de la carga.

    Parameters
    ----------
    path : str | Path
        Ruta al archivo CSV crudo (train.csv de Kaggle).

    Returns
    -------
    pd.DataFrame
        DataFrame con dtypes correctos y PassengerId como índice.

    Raises
    ------
    FileNotFoundError
        Si el archivo no existe en la ruta indicada.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Archivo de datos no encontrado: {path}\n"
            "Descarga el dataset de Kaggle y colócalo en data/raw/train.csv\n"
            "O ejecuta: dvc pull"
        )

    logger.info("Cargando datos desde: %s", path)

    # Cargar con dtypes explícitos — nunca confiar en la inferencia automática
    df = pd.read_csv(
        path,
        dtype={k: v for k, v in TITANIC_DTYPES.items() if v != "string"},
        na_values=["", "NA", "N/A", "nan", "NaN"],
    )

    # Convertir columnas string correctamente (pandas StringDtype es más seguro que object)
    for col in ["Name", "Sex", "Ticket", "Cabin", "Embarked"]:
        if col in df.columns:
            df[col] = df[col].astype("string")

    logger.info(
        "Dataset cargado: %d filas × %d columnas | nulos: %d",
        len(df),
        df.shape[1],
        df.isnull().sum().sum(),
    )
    return df
