"""
schema.py — Schema de validación Pandera para el dataset Titanic.

Principio: los datos NUNCA entran al pipeline sin validación.
Si el schema falla, el pipeline falla con un mensaje claro —
no continúa silenciosamente con datos corruptos.

El schema valida:
    - Tipos de columna correctos
    - Rangos válidos para variables numéricas
    - Categorías válidas para variables categóricas
    - Ausencia de PassengerId duplicados (check DataFrame-level)
    - Consistencia entre SibSp + Parch y tamaño de familia
"""

from __future__ import annotations

import logging

import pandas as pd
import pandera as pa
from pandera import Check, Column, DataFrameSchema

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Schema del dataset CRUDO (train.csv tal como viene de Kaggle)
# ─────────────────────────────────────────────────────────────────────────────

RAW_SCHEMA = DataFrameSchema(
    columns={
        "PassengerId": Column(
            int,
            checks=Check.greater_than(0),
            nullable=False,
            description="ID único del pasajero",
        ),
        "Survived": Column(
            int,
            checks=Check.isin([0, 1], error="Survived solo puede ser 0 o 1"),
            nullable=False,
            description="Target: 0=no sobrevivió, 1=sobrevivió",
        ),
        "Pclass": Column(
            int,
            checks=Check.isin([1, 2, 3], error="Pclass debe ser 1, 2 o 3"),
            nullable=False,
            description="Clase del boleto — proxy de estatus socioeconómico",
        ),
        "Name": Column(
            str,
            checks=Check.str_length(min_value=2, max_value=200),
            nullable=False,
        ),
        "Sex": Column(
            str,
            checks=Check.isin(["male", "female"]),
            nullable=False,
        ),
        "Age": Column(
            float,
            checks=[
                Check.greater_than(0, error="Edad debe ser > 0"),
                Check.less_than_or_equal_to(100, error="Edad no puede superar 100"),
            ],
            nullable=True,   # 177 nulos — se imputarán en el preprocesador
        ),
        "SibSp": Column(
            int,
            checks=Check.greater_than_or_equal_to(0),
            nullable=False,
        ),
        "Parch": Column(
            int,
            checks=Check.greater_than_or_equal_to(0),
            nullable=False,
        ),
        "Ticket": Column(str, nullable=False),
        "Fare": Column(
            float,
            checks=Check.greater_than_or_equal_to(0, error="Fare no puede ser negativo"),
            nullable=False,
        ),
        "Cabin": Column(str, nullable=True),   # 77% nulos — esperado
        "Embarked": Column(
            str,
            checks=Check.isin(["C", "Q", "S"], error="Embarked debe ser C, Q o S"),
            nullable=True,   # 2 nulos
        ),
    },
    checks=[
        # CHECK DataFrame-level: sin PassengerId duplicados
        Check(
            lambda df: df["PassengerId"].duplicated().sum() == 0,
            error="PassengerId tiene duplicados — cada pasajero debe ser único",
        ),
        # CHECK cross-column: tamaño de familia razonable
        Check(
            lambda df: (df["SibSp"] + df["Parch"]).max() <= 10,
            error="Familia de más de 11 personas — verificar datos",
        ),
    ],
    strict=False,   # Permite columnas extra sin fallar
    coerce=False,
    name="TitanicRawSchema",
)


# ─────────────────────────────────────────────────────────────────────────────
# Schema del dataset ENGINEERED (después del feature engineering)
# ─────────────────────────────────────────────────────────────────────────────

ENGINEERED_SCHEMA = DataFrameSchema(
    columns={
        "Age":          Column(float, nullable=True),
        "Fare":         Column(float, checks=Check.greater_than_or_equal_to(0), nullable=False),
        "Pclass":       Column(int,   checks=Check.isin([1, 2, 3]), nullable=False),
        "SibSp":        Column(int,   checks=Check.greater_than_or_equal_to(0), nullable=False),
        "Parch":        Column(int,   checks=Check.greater_than_or_equal_to(0), nullable=False),
        "FamilySize":   Column(int,   checks=[
            Check.greater_than_or_equal_to(1),
            Check.less_than_or_equal_to(20),
        ], nullable=False),
        "IsAlone":      Column(int,   checks=Check.isin([0, 1]), nullable=False),
        "HasCabin":     Column(int,   checks=Check.isin([0, 1]), nullable=False),
        "FarePerPerson": Column(float, checks=Check.greater_than_or_equal_to(0), nullable=False),
        "FareLog":      Column(float, checks=Check.greater_than_or_equal_to(0), nullable=False),
        "Sex":          Column(str,   checks=Check.isin(["male", "female"]), nullable=False),
        "Embarked":     Column(str,   checks=Check.isin(["C", "Q", "S"]), nullable=True),
        "Title":        Column(str,   checks=Check.isin(
            ["Mr", "Mrs", "Miss", "Master", "Rare", "Unknown"],
        ), nullable=False),
        "CabinDeck":    Column(str,   checks=Check.isin(
            ["A", "B", "C", "D", "E", "F", "G", "U"],
        ), nullable=False),
        "FamilyCategory": Column(str, checks=Check.isin(
            ["Solo", "Pequeña", "Mediana", "Grande"],
        ), nullable=False),
        "AgeBin":       Column(str,   checks=Check.isin(
            ["Niño", "Adolescente", "Adulto_Joven", "Adulto", "Mayor", "Desconocido"],
        ), nullable=False),
    },
    checks=[
        # CHECK cross-column: IsAlone consistente con FamilySize
        Check(
            lambda df: ((df["IsAlone"] == 1) == (df["FamilySize"] == 1)).all(),
            error="IsAlone inconsistente con FamilySize — bug en feature_engineering()",
        ),
        # CHECK cross-column: FarePerPerson <= Fare (FamilySize >= 1 siempre)
        Check(
            lambda df: (df["FarePerPerson"] <= df["Fare"] + 1e-6).all(),
            error="FarePerPerson > Fare — bug en cálculo",
        ),
    ],
    strict=False,
    name="TitanicEngineeredSchema",
)


def validate_raw(df: pd.DataFrame) -> pd.DataFrame:
    """
    Valida el DataFrame crudo contra RAW_SCHEMA.

    Si la validación falla, lanza SchemaErrors con un mensaje claro.
    El pipeline NUNCA debe continuar si esta función falla.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame crudo cargado desde train.csv.

    Returns
    -------
    pd.DataFrame
        El mismo DataFrame si pasa la validación.

    Raises
    ------
    pandera.errors.SchemaErrors
        Si alguna columna o check no cumple el schema.
    """
    logger.info("Validando schema del dataset crudo...")
    try:
        validated = RAW_SCHEMA.validate(df, lazy=True)
        logger.info("✓ Dataset crudo pasa la validación (%d filas)", len(validated))
        return validated
    except pa.errors.SchemaErrors as exc:
        n = len(exc.failure_cases)
        logger.error(
            "❌ Validación fallida: %d problema(s) encontrados\n%s",
            n,
            exc.failure_cases[["check", "column", "failure_case"]].head(10),
        )
        raise


def validate_engineered(df: pd.DataFrame) -> pd.DataFrame:
    """
    Valida el DataFrame después del feature engineering.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con features engineered.

    Returns
    -------
    pd.DataFrame
        El mismo DataFrame si pasa la validación.
    """
    logger.info("Validando schema del dataset engineered...")
    try:
        validated = ENGINEERED_SCHEMA.validate(df, lazy=True)
        logger.info("✓ Features engineered pasan la validación (%d filas)", len(validated))
        return validated
    except pa.errors.SchemaErrors as exc:
        n = len(exc.failure_cases)
        logger.error(
            "❌ Validación de features fallida: %d problema(s)\n%s",
            n,
            exc.failure_cases[["check", "column", "failure_case"]].head(10),
        )
        raise
