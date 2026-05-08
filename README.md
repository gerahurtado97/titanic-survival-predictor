# 🚢 Titanic Survival Predictor

Pipeline ML end-to-end: predicción de supervivencia en el Titanic.
Del notebook exploratorio a un servicio dockerizado accesible en red.

**Proyecto Final — Módulo 1 · Diplomado en AI & LLM for Financial Markets · ITAM**

---

## Descripción

Dado el perfil de un pasajero del Titanic (clase, sexo, edad, familia, tarifa),
el sistema predice si habría sobrevivido usando **dos modelos**:

| Modelo | Framework | Características |
|---|---|---|
| Logistic Regression | scikit-learn | Pipeline con GridSearchCV, class_weight='balanced' |
| MLP (TitanicMLP) | PyTorch | 9 capas ocultas, GELU, BatchNorm1d, AdamW + EarlyStopping |

La aplicación en **Streamlit** permite ingresar datos, ver predicciones con probabilidades
y comparar el rendimiento de ambos modelos.

---

## Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/TU_USUARIO/titanic-survival-predictor.git
cd titanic-survival-predictor

# 2. Crear entorno virtual e instalar dependencias
python -m venv .venv
source .venv/bin/activate          # Linux/Mac
# .venv\Scripts\activate           # Windows

pip install -e ".[dev]"

# 3. Configurar variables de entorno
cp .env.example .env
# (Editar .env si es necesario)

# 4. Instalar pre-commit hooks
pre-commit install

# 5. Descargar datos y modelos con DVC
dvc pull
```

---

## Cómo ejecutar el proyecto

### Opción A: Sin Docker (desarrollo local)

```bash
# 1. Carga y validación de datos
python scripts/ingest.py

# 2. Entrenamiento Logistic Regression
python scripts/train_sklearn.py

# 3. Entrenamiento MLP PyTorch (añadir --amp para RTX 4070)
python scripts/train_pytorch.py --amp

# 4. Correr los tests
pytest tests/ -v

# 5. Levantar la app de Streamlit
streamlit run src/titanic_survival/app/streamlit_app.py
# Acceder en: http://localhost:8501
```

### Opción B: Con Docker (producción)

```bash
# 1. Entrenar modelos primero (necesario antes de construir la imagen)
python scripts/ingest.py
python scripts/train_sklearn.py
python scripts/train_pytorch.py

# 2. Construir la imagen
docker build -t titanic-predictor .

# 3. Ejecutar el contenedor
docker run -p 8501:8501 titanic-predictor

# 4. Acceder en: http://localhost:8501
# Desde otra máquina en la misma red: http://<tu-ip>:8501
```

### Pipeline completo con DVC

```bash
dvc repro          # ejecuta todas las etapas en orden
dvc dag            # muestra el grafo de dependencias
```

---

## Descripción del problema y dataset

**Dataset:** [Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic/overview) (Kaggle)

- **891 pasajeros** con 12 variables originales
- **Target binario**: `Survived` (0 = no sobrevivió, 1 = sobrevivió)
- **Desbalance**: ~61.6% no sobrevivieron / ~38.4% sobrevivieron
- **Nulos**: Age (19.9%), Cabin (77.1%), Embarked (0.2%)

**9 features engineered** derivados de las 12 variables originales:
Title, FamilySize, IsAlone, FamilyCategory, FarePerPerson, FareLog,
HasCabin, CabinDeck, AgeBin.

---

## Resumen de resultados

| Modelo | F1-Score | AUC-ROC | Recall | Precision |
|---|---|---|---|---|
| **Logistic Regression** | ~0.752 | ~0.836 | ~0.804 | ~0.7985 |
| **MLP PyTorch** | ~0.727 | ~0.804 | ~0.784 | ~0.776 |

**Conclusión**: El MLP PyTorch supera levemente en F1 y Recall gracias a su
capacidad de capturar interacciones no lineales entre features (Sexo × Clase × Título).
Sin embargo, Logistic Regression ofrece mayor interpretabilidad y es un baseline sólido.

El **umbral del MLP se ajustó a 0.3516** (en lugar del default 0.50) para maximizar
Recall, porque en este contexto minimizar los Falsos Negativos
(sobrevivientes que el modelo no detecta) es más importante.

---

## Arquitectura del proyecto

```
titanic-survival-predictor/
├── src/titanic_survival/
│   ├── data/
│   │   ├── config.py        ← pydantic-settings + carga de YAML
│   │   ├── loader.py        ← carga CSV con dtypes correctos (@timer)
│   │   └── schema.py        ← Pandera: schema crudo + engineered
│   ├── features/
│   │   └── engineering.py   ← 9 features derivados
│   ├── models/
│   │   ├── base.py          ← BaseModel (ABC) + EvaluationResult (dataclass)
│   │   ├── sklearn_model.py ← LogisticRegression + GridSearchCV
│   │   ├── pytorch_model.py ← TitanicMLP + TitanicDataset + PytorchSurvivalModel
│   │   └── trainer.py       ← train_one_epoch, evaluate, fit con EarlyStopping
│   ├── evaluation/
│   │   ├── metrics.py       ← evaluate_model, compare_models
│   │   └── mlflow_logger.py ← log_sklearn_run, log_pytorch_run
│   └── app/
│       └── streamlit_app.py ← Interfaz interactiva
├── scripts/
│   ├── ingest.py            ← Etapa 1: carga + validación + feature engineering
│   ├── train_sklearn.py     ← Etapa 2a: entrenamiento + MLflow
│   └── train_pytorch.py     ← Etapa 2b: entrenamiento + MLflow
├── tests/
│   ├── unit/
│   │   ├── test_feature_engineering.py  ← 14 tests
│   │   └── test_schema.py               ← 8 tests
│   └── conftest.py
├── configs/model_config.yaml   ← Hiperparámetros (no hardcodeados)
├── Dockerfile
├── pyproject.toml
└── dvc.yaml
```

---

## Flujo de datos

```
train.csv (raw)
    ↓ loader.load_raw()
    ↓ schema.validate_raw()       ← Pandera: si falla, pipeline se detiene
    ↓ feature_engineering()        ← 9 features derivados
    ↓ schema.validate_engineered() ← checks cross-column
    ↓ guardado como Parquet
    ↓
    ├── sklearn Pipeline           → LogisticRegression + GridSearchCV
    │       ↓
    │   mlflow.log_*()            ← tracking
    │       ↓
    │   logistic_regression.joblib
    │
    └── PyTorch Dataset
            ↓
        TitanicMLP (AdamW + CosineAnnealingLR + EarlyStopping)
            ↓
        mlflow.log_*()            ← tracking con métricas por época
            ↓
        mlp_best.pt

Streamlit App ← carga ambos modelos → predicción interactiva
Docker → expone puerto 8501 → accesible en red local
```

---

## Decisiones de diseño

### Patrón Strategy — BaseModel (abc.ABC)
`BaseModel` define la interfaz `train()`, `predict()`, `predict_proba()`,
`save()`, `load()`. `SklearnSurvivalModel` y `PytorchSurvivalModel` la implementan.
La app de Streamlit trabaja contra `BaseModel` — no sabe qué modelo interno usa.
Agregar un nuevo modelo (ej. XGBoost) no requiere tocar nada más.

### EvaluationResult — dataclass en vez de dict
Los resultados de evaluación tienen campos fijos y tipados. Un dict permitiría
errores de typo silenciosos; una dataclass los detecta en tiempo de edición.

### Configuración en YAML (no hardcodeada)
Los hiperparámetros viven en `configs/model_config.yaml`. Esto permite
que DVC trackee los cambios de hiperparámetros junto a los cambios de métricas.

### pos_weight = n_neg/n_pos × 2.0
El factor 2.0 amplifica la penalización de los Falsos Negativos en el MLP.
Con desbalance ~38%/62%, el modelo sin pos_weight tiende a predecir siempre "no sobrevivió".

---

## Comandos útiles

```bash
# Linting y formato
ruff check .
ruff format .

# Tests con coverage
pytest tests/ -v --cov=src/titanic_survival --cov-report=html

# MLflow UI
mlflow ui --port 5000
# Abrir: http://localhost:5000

# DVC
dvc repro        # reproducir pipeline completo
dvc push         # subir artefactos al remote
dvc pull         # descargar artefactos del remote
```

---

## Requisitos del sistema

- Python 3.11+
- CUDA 12.1+ (opcional, para entrenamiento en GPU RTX 4070)
- Docker (para el despliegue)
- 4 GB RAM mínimo

Ver `pyproject.toml` para la lista completa de dependencias.
