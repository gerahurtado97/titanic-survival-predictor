"""
tests/unit/test_feature_engineering.py — Tests del feature engineering.

Testea el pipeline de transformaciones, no el modelo.
Patrón AAA: Arrange / Act / Assert.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from titanic_survival.features.engineering import (
    CATEGORICAL_FEATURES,
    FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    _extract_title,
    feature_engineering,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def sample_row() -> dict:
    """Fila mínima válida del dataset Titanic."""
    return {
        "PassengerId": 1,
        "Survived": 1,
        "Pclass": 1,
        "Name": "Braund, Mr. Owen Harris",
        "Sex": "male",
        "Age": 22.0,
        "SibSp": 1,
        "Parch": 0,
        "Ticket": "A/5 21171",
        "Fare": 7.25,
        "Cabin": None,
        "Embarked": "S",
    }


@pytest.fixture()
def sample_df(sample_row) -> pd.DataFrame:
    """DataFrame de una fila para tests rápidos."""
    return pd.DataFrame([sample_row])


@pytest.fixture()
def multi_row_df() -> pd.DataFrame:
    """DataFrame multi-fila para tests de invariantes."""
    return pd.DataFrame([
        {
            "PassengerId": 1, "Survived": 0, "Pclass": 3,
            "Name": "Braund, Mr. Owen Harris", "Sex": "male",
            "Age": 22.0, "SibSp": 1, "Parch": 0,
            "Ticket": "A/5 21171", "Fare": 7.25, "Cabin": None, "Embarked": "S",
        },
        {
            "PassengerId": 2, "Survived": 1, "Pclass": 1,
            "Name": "Cumings, Mrs. John Bradley", "Sex": "female",
            "Age": 38.0, "SibSp": 1, "Parch": 0,
            "Ticket": "PC 17599", "Fare": 71.28, "Cabin": "C85", "Embarked": "C",
        },
        {
            "PassengerId": 3, "Survived": 1, "Pclass": 3,
            "Name": "Heikkinen, Miss. Laina", "Sex": "female",
            "Age": None,   # NaN intencional
            "SibSp": 0, "Parch": 0,
            "Ticket": "STON/O2.", "Fare": 7.925, "Cabin": None, "Embarked": "S",
        },
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Tests de _extract_title
# ─────────────────────────────────────────────────────────────────────────────

class TestExtractTitle:
    def test_mr(self):
        assert _extract_title("Braund, Mr. Owen Harris") == "Mr"

    def test_mrs(self):
        assert _extract_title("Cumings, Mrs. John Bradley") == "Mrs"

    def test_miss(self):
        assert _extract_title("Heikkinen, Miss. Laina") == "Miss"

    def test_master(self):
        assert _extract_title("Palsson, Master. Gosta Leonard") == "Master"

    @pytest.mark.parametrize("name,expected", [
        ("Montvila, Rev. Juozas", "Rare"),
        ("Graham, Miss. Margaret Edith", "Miss"),
        ("Bonnell, Dr. Fortescue", "Rare"),
    ])
    def test_rare_and_common(self, name: str, expected: str):
        assert _extract_title(name) == expected

    def test_unknown_name(self):
        assert _extract_title("Nombre Sin Título") == "Unknown"


# ─────────────────────────────────────────────────────────────────────────────
# Tests de feature_engineering
# ─────────────────────────────────────────────────────────────────────────────

class TestFeatureEngineering:
    def test_returns_dataframe(self, sample_df):
        result = feature_engineering(sample_df)
        assert isinstance(result, pd.DataFrame)

    def test_no_mutation_of_input(self, sample_df):
        """feature_engineering no debe modificar el DataFrame original."""
        original_cols = set(sample_df.columns)
        _ = feature_engineering(sample_df)
        assert set(sample_df.columns) == original_cols

    def test_all_features_present(self, sample_df):
        result = feature_engineering(sample_df)
        for f in FEATURES:
            assert f in result.columns, f"Feature '{f}' no encontrado"

    def test_family_size_formula(self, sample_df):
        """FamilySize = SibSp + Parch + 1."""
        result = feature_engineering(sample_df)
        expected = int(sample_df.iloc[0]["SibSp"] + sample_df.iloc[0]["Parch"] + 1)
        assert result.iloc[0]["FamilySize"] == expected

    def test_is_alone_when_solo(self, multi_row_df):
        """Un pasajero con SibSp=0, Parch=0 debe tener IsAlone=1."""
        result = feature_engineering(multi_row_df)
        # La fila 3 (idx 2): SibSp=0, Parch=0 → IsAlone=1
        assert result.iloc[2]["IsAlone"] == 1

    def test_is_alone_consistency(self, multi_row_df):
        """IsAlone debe ser consistente con FamilySize en TODAS las filas."""
        result = feature_engineering(multi_row_df)
        mask_alone     = result["FamilySize"] == 1
        mask_is_alone  = result["IsAlone"] == 1
        assert (mask_alone == mask_is_alone).all(), \
            "IsAlone inconsistente con FamilySize — bug en feature_engineering()"

    def test_fare_log_non_negative(self, multi_row_df):
        """FareLog = log1p(Fare) debe ser >= 0 siempre que Fare >= 0."""
        result = feature_engineering(multi_row_df)
        assert (result["FareLog"] >= 0).all()

    def test_fare_per_person_le_fare(self, multi_row_df):
        """FarePerPerson <= Fare porque FamilySize >= 1."""
        result = feature_engineering(multi_row_df)
        assert (result["FarePerPerson"] <= result["Fare"] + 1e-9).all()

    def test_has_cabin_binary(self, multi_row_df):
        result = feature_engineering(multi_row_df)
        assert set(result["HasCabin"].unique()).issubset({0, 1})

    def test_has_cabin_reflects_cabin_column(self, multi_row_df):
        """HasCabin=1 exactamente cuando Cabin no es nulo."""
        result = feature_engineering(multi_row_df)
        expected_has_cabin = multi_row_df["Cabin"].notna().astype(int)
        pd.testing.assert_series_equal(
            result["HasCabin"].reset_index(drop=True),
            expected_has_cabin.reset_index(drop=True),
            check_names=False,
        )

    def test_age_bin_desconocido_for_nan(self, multi_row_df):
        """AgeBin='Desconocido' cuando Age es NaN."""
        result = feature_engineering(multi_row_df)
        # Fila con Age=None (idx 2)
        assert result.iloc[2]["AgeBin"] == "Desconocido"

    def test_age_bin_nino(self):
        """Edad 10 → AgeBin='Niño'."""
        row = pd.DataFrame([{
            "PassengerId": 99, "Survived": 1, "Pclass": 1,
            "Name": "Test, Master. Kid", "Sex": "male", "Age": 10.0,
            "SibSp": 0, "Parch": 2, "Ticket": "T123", "Fare": 50.0,
            "Cabin": None, "Embarked": "S",
        }])
        result = feature_engineering(row)
        assert result.iloc[0]["AgeBin"] == "Niño"

    def test_family_category_solo(self, sample_df):
        """Pasajero solo → FamilyCategory='Solo'."""
        # SibSp=1, Parch=0 → FamilySize=2 → Pequeña
        result = feature_engineering(sample_df)
        assert result.iloc[0]["FamilyCategory"] == "Pequeña"

    @pytest.mark.parametrize("sibsp,parch,expected_cat", [
        (0, 0, "Solo"),
        (1, 0, "Pequeña"),
        (2, 0, "Pequeña"),
        (3, 0, "Mediana"),
        (4, 1, "Grande"),
    ])
    def test_family_category_parametrize(self, sibsp: int, parch: int, expected_cat: str):
        row = pd.DataFrame([{
            "PassengerId": 1, "Survived": 0, "Pclass": 2,
            "Name": "Test, Mr. User", "Sex": "male", "Age": 30.0,
            "SibSp": sibsp, "Parch": parch, "Ticket": "T1", "Fare": 15.0,
            "Cabin": None, "Embarked": "S",
        }])
        result = feature_engineering(row)
        assert result.iloc[0]["FamilyCategory"] == expected_cat, \
            f"SibSp={sibsp} Parch={parch}: esperado {expected_cat}, obtenido {result.iloc[0]['FamilyCategory']}"
