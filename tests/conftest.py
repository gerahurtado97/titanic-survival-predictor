"""
tests/conftest.py — Fixtures compartidas entre todos los tests.

Las fixtures definidas aquí se inyectan automáticamente en cualquier
test del directorio tests/ por nombre (pytest magic).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

# Añadir src/ al path para todos los tests
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture(scope="session")
def device() -> torch.device:
    """Device disponible para tests de PyTorch."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture()
def minimal_raw_df() -> pd.DataFrame:
    """DataFrame mínimo válido del Titanic para tests rápidos."""
    return pd.DataFrame([
        {
            "PassengerId": 1, "Survived": 0, "Pclass": 3,
            "Name": "Braund, Mr. Owen Harris", "Sex": "male",
            "Age": 22.0, "SibSp": 1, "Parch": 0,
            "Ticket": "A/5", "Fare": 7.25, "Cabin": None, "Embarked": "S",
        },
        {
            "PassengerId": 2, "Survived": 1, "Pclass": 1,
            "Name": "Cumings, Mrs. John B.", "Sex": "female",
            "Age": 38.0, "SibSp": 1, "Parch": 0,
            "Ticket": "PC 17599", "Fare": 71.28, "Cabin": "C85", "Embarked": "C",
        },
    ])


@pytest.fixture()
def small_numpy_data() -> tuple[np.ndarray, np.ndarray]:
    """Datos sintéticos pequeños para tests del modelo."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((50, 38)).astype(np.float32)
    y = rng.integers(0, 2, size=50)
    return X, y
