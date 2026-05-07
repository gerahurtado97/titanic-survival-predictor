"""
tests/unit/test_schema.py — Tests del schema de validación Pandera.

Verifica que:
    - El schema acepta datos válidos
    - El schema rechaza datos inválidos con mensajes claros
    - Los checks cross-column funcionan correctamente
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pandera as pa
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from titanic_survival.data.schema import validate_raw


@pytest.fixture()
def valid_raw_df() -> pd.DataFrame:
    """DataFrame mínimo que pasa el schema crudo."""
    return pd.DataFrame(
        [
            {
                "PassengerId": 1,
                "Survived": 0,
                "Pclass": 3,
                "Name": "Braund, Mr. Owen Harris",
                "Sex": "male",
                "Age": 22.0,
                "SibSp": 1,
                "Parch": 0,
                "Ticket": "A/5 21171",
                "Fare": 7.25,
                "Cabin": None,
                "Embarked": "S",
            },
            {
                "PassengerId": 2,
                "Survived": 1,
                "Pclass": 1,
                "Name": "Cumings, Mrs. John Bradley",
                "Sex": "female",
                "Age": 38.0,
                "SibSp": 1,
                "Parch": 0,
                "Ticket": "PC 17599",
                "Fare": 71.28,
                "Cabin": "C85",
                "Embarked": "C",
            },
        ]
    )


class TestRawSchema:
    def test_valid_data_passes(self, valid_raw_df):
        """Datos válidos deben pasar la validación sin errores."""
        result = validate_raw(valid_raw_df)
        assert len(result) == len(valid_raw_df)

    def test_invalid_survived_raises(self, valid_raw_df):
        """Survived=2 debe fallar la validación."""
        bad = valid_raw_df.copy()
        bad.loc[0, "Survived"] = 2
        with pytest.raises(pa.errors.SchemaErrors):
            validate_raw(bad)

    def test_invalid_pclass_raises(self, valid_raw_df):
        """Pclass=4 debe fallar la validación."""
        bad = valid_raw_df.copy()
        bad.loc[0, "Pclass"] = 4
        with pytest.raises(pa.errors.SchemaErrors):
            validate_raw(bad)

    def test_negative_fare_raises(self, valid_raw_df):
        """Fare negativo debe fallar la validación."""
        bad = valid_raw_df.copy()
        bad.loc[0, "Fare"] = -10.0
        with pytest.raises(pa.errors.SchemaErrors):
            validate_raw(bad)

    def test_negative_age_raises(self, valid_raw_df):
        """Edad negativa debe fallar."""
        bad = valid_raw_df.copy()
        bad.loc[0, "Age"] = -5.0
        with pytest.raises(pa.errors.SchemaErrors):
            validate_raw(bad)

    def test_duplicate_passenger_id_raises(self, valid_raw_df):
        """PassengerId duplicado debe fallar el check DataFrame-level."""
        bad = valid_raw_df.copy()
        bad.loc[1, "PassengerId"] = 1  # duplicado del id 1
        with pytest.raises(pa.errors.SchemaErrors):
            validate_raw(bad)

    def test_nullable_age_allowed(self, valid_raw_df):
        """Age nulo debe ser permitido (177 nulos en dataset real)."""
        df_with_nan = valid_raw_df.copy()
        df_with_nan.loc[0, "Age"] = None
        result = validate_raw(df_with_nan)
        assert pd.isna(result.loc[0, "Age"])

    def test_nullable_cabin_allowed(self, valid_raw_df):
        """Cabin nulo debe ser permitido (77% nulos)."""
        df_no_cabin = valid_raw_df.copy()
        df_no_cabin["Cabin"] = None
        result = validate_raw(df_no_cabin)
        assert result["Cabin"].isna().all()
