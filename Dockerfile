# ─────────────────────────────────────────────────────────────
# Dockerfile — Titanic Survival Predictor
# Imagen slim para producción: solo lo necesario para servir la app
#
# Construir:   docker build -t titanic-predictor .
# Ejecutar:    docker run -p 8501:8501 titanic-predictor
# Acceder:     http://localhost:8501
# ─────────────────────────────────────────────────────────────

FROM python:3.11-slim

# Metadatos
LABEL maintainer="ITAM Diplomado AI & LLM"
LABEL description="Titanic Survival Predictor — Streamlit App"
LABEL version="0.1.0"

# Evitar prompts interactivos durante la instalación
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Directorio de trabajo
WORKDIR /app

# ── Instalar dependencias del sistema (mínimas)
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ── Copiar solo pyproject.toml primero (Docker cache layer)
COPY pyproject.toml ./

# ── Instalar dependencias Python
# Se instala directamente con pip (sin Poetry en producción — más simple)
# CPU-only torch para reducir el tamaño de la imagen (GPU no disponible en Docker sin nvidia-docker)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir \
        numpy>=1.26 \
        pandas>=2.2 \
        scikit-learn>=1.3 \
        pydantic-settings>=2.0 \
        python-dotenv>=1.0 \
        pandera>=0.21.0 \
        multimethod==1.9.3 \
        mlflow>=2.9.0 \
        streamlit>=1.35.0 \
        matplotlib>=3.8 \
        seaborn>=0.13 \
        pyyaml>=6.0 \
        joblib>=1.3 \
        tqdm>=4.66 \
        pyarrow>=14.0

# ── Copiar el código fuente
COPY src/ ./src/
COPY configs/ ./configs/

# ── Copiar modelos serializados (se entrenan FUERA del contenedor)
# Los modelos se copian en lugar de reentrenarse dentro del contenedor
COPY models/ ./models/

# ── Copiar datos procesados (para las visualizaciones del EDA)
COPY data/processed/ ./data/processed/

# ── Variables de entorno para la app
ENV ENVIRONMENT=production
ENV LOG_LEVEL=INFO
ENV PYTHONPATH=/app/src

# ── Exponer el puerto de Streamlit
EXPOSE 8501

# ── Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# ── Arrancar la app de Streamlit
CMD ["streamlit", "run", "src/titanic_survival/app/streamlit_app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
