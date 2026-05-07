"""
engineering.py — Feature Engineering del dataset Titanic.

Todas las transformaciones se aplican de forma idéntica sobre train y test
para evitar data leakage. El preprocesador sklearn (StandardScaler + OHE)
se ajusta SOLO en el train set, dentro del Pipeline de sklearn.

Features derivados:
    Title          — título del nombre (encapsula sexo + estatus + edad)
    FamilySize     — SibSp + Parch + 1
    IsAlone        — 1 si viaja solo
    FamilyCategory — binning de FamilySize (Solo/Pequeña/Mediana/Grande)
    FarePerPerson  — tarifa normalizada por tamaño de familia
    FareLog        — log(1+Fare) para reducir sesgo de distribución
    HasCabin       — 1 si se conoce la cabina (proxy de clase alta)
    CabinDeck      — primera letra de la cabina
    AgeBin         — grupos de edad

Motivación de cada feature documentada en el EDA (notebooks/eda.ipynb).
"""

from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Títulos que se agrupan en "Rare" por baja frecuencia
_RARE_TITLES: frozenset[str] = frozenset(
    {
        "Don",
        "Rev",
        "Dr",
        "Mme",
        "Ms",
        "Major",
        "Lady",
        "Sir",
        "Mlle",
        "Col",
        "Capt",
        "the Countess",
        "Jonkheer",
    }
)


def _extract_title(name: str) -> str:
    """Extrae el título del nombre completo y agrupa los raros en 'Rare'."""
    match = re.search(r",\s*([^.]+)\.", name)
    if not match:
        return "Unknown"
    title = match.group(1).strip()
    return "Rare" if title in _RARE_TITLES else title


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica todas las transformaciones de Feature Engineering.

    Diseñada para ser llamada de forma idéntica sobre train y test
    (sin ajuste sobre los datos — sin riesgo de data leakage).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame crudo del Titanic, ya validado con Pandera.

    Returns
    -------
    pd.DataFrame
        DataFrame con las columnas originales más los features engineered.
    """
    df = df.copy()

    # ── 1. Título del nombre
    df["Title"] = df["Name"].apply(_extract_title)

    # ── 2. Tamaño de familia
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    # ── 3. Categoría de familia (relación no lineal con supervivencia)
    df["FamilyCategory"] = pd.cut(
        df["FamilySize"],
        bins=[0, 1, 3, 5, 20],
        labels=["Solo", "Pequeña", "Mediana", "Grande"],
    ).astype(str)

    # ── 4. Tarifa por persona
    df["FarePerPerson"] = df["Fare"] / df["FamilySize"]

    # ── 5. Log de tarifa (reduce sesgo extremo)
    df["FareLog"] = np.log1p(df["Fare"])

    # ── 6. Indicador de cabina conocida
    df["HasCabin"] = df["Cabin"].notna().astype(int)

    # ── 7. Deck de la cabina
    df["CabinDeck"] = df["Cabin"].fillna("U").str[0].replace("T", "U")

    # ── 8. Bins de edad (NaN → 'Desconocido' para el imputer categórico)
    age_bin = pd.cut(
        df["Age"],
        bins=[0, 12, 18, 35, 60, 120],
        labels=["Niño", "Adolescente", "Adulto_Joven", "Adulto", "Mayor"],
    )
    df["AgeBin"] = age_bin.cat.add_categories("Desconocido").fillna("Desconocido").astype(str)

    logger.debug(
        "Feature engineering completado: %d filas, %d columnas",
        len(df),
        df.shape[1],
    )
    return df


# Columnas que entran al preprocesador sklearn
NUMERIC_FEATURES: list[str] = [
    "Pclass",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "FamilySize",
    "IsAlone",
    "HasCabin",
    "FarePerPerson",
    "FareLog",
]

CATEGORICAL_FEATURES: list[str] = [
    "Sex",
    "Embarked",
    "Title",
    "CabinDeck",
    "FamilyCategory",
    "AgeBin",
]

FEATURES: list[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET: str = "Survived"
