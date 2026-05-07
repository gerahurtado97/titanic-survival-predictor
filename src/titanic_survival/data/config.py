"""
config.py — Configuración centralizada con validación automática.

Usa pydantic-settings para:
- Cargar valores desde .env y variables de entorno
- Validar tipos y rangos al arrancar (no en runtime)
- Detectar configuraciones inválidas antes de procesar datos

Patrón: la configuración se instancia UNA vez con get_settings()
y se reutiliza en todo el proyecto.
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """
    Configuración centralizada del pipeline.

    Carga desde .env y variables de entorno automáticamente.
    Falla al instanciarse si algún valor es inválido — no en runtime.
    """

    # ── Entorno ────────────────────────────────────────────────
    environment: str = Field(
        default="development",
        description="Entorno: development | staging | production",
    )
    log_level: str = Field(
        default="INFO",
        description="Nivel de logging: DEBUG | INFO | WARNING | ERROR",
    )

    # ── Rutas ─────────────────────────────────────────────────
    data_raw_path: str = Field(
        default="data/raw/train.csv",
        description="Ruta al CSV crudo del Titanic",
    )
    data_processed_path: str = Field(
        default="data/processed/",
        description="Directorio de datos procesados",
    )
    models_path: str = Field(
        default="models/",
        description="Directorio donde se serializan los modelos",
    )
    mlflow_tracking_uri: str = Field(
        default="file:./mlruns",
        description="URI del servidor de MLflow tracking",
    )

    # ── Reproducibilidad ──────────────────────────────────────
    random_seed: int = Field(
        default=42,
        ge=0,
        description="Semilla global para reproducibilidad",
    )

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid:
            raise ValueError(f"log_level debe ser uno de {valid}, recibido: {v}")
        return v.upper()

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        valid = {"development", "staging", "production"}
        if v.lower() not in valid:
            raise ValueError(f"environment debe ser uno de {valid}")
        return v.lower()

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",
    }


def get_settings() -> Settings:
    """
    Retorna la configuración de la aplicación.

    Returns
    -------
    Settings
        Instancia validada de la configuración.
    """
    settings = Settings()
    logger.info(
        "Configuración cargada: environment=%s, seed=%d",
        settings.environment,
        settings.random_seed,
    )
    return settings


def load_model_config(config_path: str | Path = "configs/model_config.yaml") -> dict:
    """
    Carga la configuración de hiperparámetros desde YAML.

    Los hiperparámetros residen en YAML (no hardcodeados) para
    facilitar la búsqueda de hiperparámetros y el tracking con DVC.

    Parameters
    ----------
    config_path : str | Path
        Ruta al archivo YAML de configuración del modelo.

    Returns
    -------
    dict
        Configuración completa del pipeline.

    Raises
    ------
    FileNotFoundError
        Si el archivo de configuración no existe.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Archivo de configuración no encontrado: {path}")

    with path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    logger.info("Configuración de modelos cargada desde: %s", path)
    return cfg
