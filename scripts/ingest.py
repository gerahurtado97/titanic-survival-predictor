"""
scripts/ingest.py — Etapa 1 del pipeline: carga, validación y feature engineering.

Corresponde al stage 'ingest' de dvc.yaml.

Uso::

    python scripts/ingest.py
    python scripts/ingest.py --config configs/model_config.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Configurar logging antes de cualquier import del proyecto
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("scripts.ingest")

# Añadir src/ al path para imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from titanic_survival.data.config import load_model_config
from titanic_survival.data.loader import load_raw
from titanic_survival.data.schema import validate_engineered, validate_raw
from titanic_survival.features.engineering import FEATURES, TARGET, feature_engineering


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Carga, valida y aplica feature engineering")
    p.add_argument("--config", default="configs/model_config.yaml")
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    config = load_model_config(args.config)

    raw_path  = Path(config["data"]["raw_path"])
    out_path  = Path(config["data"]["processed_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── 1. Cargar datos crudos
    df_raw = load_raw(raw_path)

    # ── 2. Validar schema crudo (falla con mensaje claro si hay problemas)
    df_raw = validate_raw(df_raw)

    # ── 3. Feature engineering
    df_fe = feature_engineering(df_raw)

    # ── 4. Validar features engineered
    features_df = df_fe[FEATURES]
    validate_engineered(features_df)

    # ── 5. Guardar en Parquet (más eficiente que CSV, preserva tipos)
    out_file = out_path if out_path.suffix == ".parquet" else Path(str(out_path).rstrip("/") + "/train_fe.parquet")
    out_file.parent.mkdir(parents=True, exist_ok=True)

    df_fe[FEATURES + [TARGET]].to_parquet(out_file, index=False, engine="pyarrow")
    size_kb = out_file.stat().st_size / 1024
    logger.info("✓ Datos procesados guardados: %s (%.1f KB)", out_file, size_kb)
    logger.info("  Filas: %d | Columnas: %d", len(df_fe), len(FEATURES) + 1)


if __name__ == "__main__":
    main()
