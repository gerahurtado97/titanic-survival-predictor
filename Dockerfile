FROM python:3.11-slim

LABEL maintainer="ITAM Diplomado AI & LLM"
LABEL description="Titanic Survival Predictor — Streamlit App"
LABEL version="0.1.0"

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app/src
ENV ENVIRONMENT=production
ENV LOG_LEVEL=INFO

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir \
        "numpy>=1.26,<2.0" \
        "pandas>=2.2" \
        "scikit-learn>=1.3" \
        "pydantic-settings>=2.0" \
        "python-dotenv>=1.0" \
        "pandera>=0.21.0" \
        "mlflow>=2.9.0" \
        "streamlit>=1.35.0" \
        "matplotlib>=3.8" \
        "seaborn>=0.13" \
        "pyyaml>=6.0" \
        "joblib>=1.3" \
        "tqdm>=4.66" \
        "pyarrow>=14.0"

COPY src/ ./src/
COPY configs/ ./configs/
COPY models/ ./models/
COPY data/processed/ ./data/processed/

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "src/titanic_survival/app/streamlit_app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
