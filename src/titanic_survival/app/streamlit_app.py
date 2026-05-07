"""
streamlit_app.py — Aplicación Streamlit para predicción de supervivencia.

Interfaz interactiva que permite:
    - Ingresar datos de un pasajero
    - Obtener predicción de supervivencia con probabilidad
    - Comparar ambos modelos (sklearn vs PyTorch)
    - Visualizar distribuciones del dataset y métricas de los modelos

Corre con: streamlit run src/titanic_survival/app/streamlit_app.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# ── Añadir src/ al path para imports
_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(_root / "src"))

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Configuración de la página
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Carga de modelos (cache para no recargar en cada interacción)
# ─────────────────────────────────────────────────────────────────────────────


@st.cache_resource
def load_models() -> dict:
    """
    Carga ambos modelos desde disco una sola vez.

    @st.cache_resource garantiza que los modelos se carguen UNA vez
    y se reutilicen en todas las interacciones del usuario.
    """
    from titanic_survival.models.pytorch_model import PytorchSurvivalModel
    from titanic_survival.models.sklearn_model import SklearnSurvivalModel

    models = {}
    sk_path = _root / "models" / "logistic_regression.joblib"
    pt_path = _root / "models" / "mlp_best.pt"

    if sk_path.exists():
        models["sklearn"] = SklearnSurvivalModel.load(sk_path)
    else:
        st.warning(f"⚠️ Modelo sklearn no encontrado en {sk_path}. Ejecuta scripts/train_sklearn.py")

    if pt_path.exists():
        models["pytorch"] = PytorchSurvivalModel.load(pt_path)
    else:
        st.warning(f"⚠️ Modelo PyTorch no encontrado en {pt_path}. Ejecuta scripts/train_pytorch.py")

    return models


@st.cache_resource
def load_preprocessor():
    """Carga el preprocesador del modelo PyTorch."""
    import joblib

    prep_path = _root / "models" / "pytorch_preprocessor.joblib"
    if prep_path.exists():
        with prep_path.open("rb") as fh:
            return joblib.load(fh)
    return None


@st.cache_data
def load_metrics() -> dict:
    """Carga las métricas guardadas de ambos modelos."""
    metrics = {}
    sk_m = _root / "models" / "metrics_sklearn.json"
    pt_m = _root / "models" / "metrics_pytorch.json"
    if sk_m.exists():
        with sk_m.open() as fh:
            metrics["sklearn"] = json.load(fh)
    if pt_m.exists():
        with pt_m.open() as fh:
            metrics["pytorch"] = json.load(fh)
    return metrics


@st.cache_data
def load_dataset() -> pd.DataFrame | None:
    """Carga el dataset procesado para visualizaciones."""
    pt = _root / "data" / "processed" / "train_fe.parquet"
    if pt.exists():
        return pd.read_parquet(pt)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Construcción del input del usuario
# ─────────────────────────────────────────────────────────────────────────────


def build_input_dataframe(
    pclass: int,
    sex: str,
    age: float,
    sibsp: int,
    parch: int,
    fare: float,
    embarked: str,
    cabin_known: bool,
) -> pd.DataFrame:
    """
    Construye un DataFrame con los features engineered a partir del input del usuario.

    Los features engineered se calculan aquí igual que en feature_engineering(),
    pero para una sola observación. Esto garantiza consistencia entre
    entrenamiento e inferencia.
    """
    family_size = sibsp + parch + 1
    is_alone = int(family_size == 1)

    if family_size == 1:
        family_cat = "Solo"
    elif family_size <= 3:
        family_cat = "Pequeña"
    elif family_size <= 5:
        family_cat = "Mediana"
    else:
        family_cat = "Grande"

    fare_per_person = fare / family_size
    fare_log = np.log1p(fare)
    has_cabin = int(cabin_known)
    cabin_deck = "C" if cabin_known else "U"  # asumimos deck C si tiene cabina

    if age <= 12:
        age_bin = "Niño"
    elif age <= 18:
        age_bin = "Adolescente"
    elif age <= 35:
        age_bin = "Adulto_Joven"
    elif age <= 60:
        age_bin = "Adulto"
    else:
        age_bin = "Mayor"

    # Título inferido desde sexo y edad (simplificado para la app)
    title = (
        ("Mrs" if age >= 18 else "Miss") if sex == "female" else ("Master" if age < 15 else "Mr")
    )

    return pd.DataFrame(
        [
            {
                "Pclass": pclass,
                "Age": age,
                "SibSp": sibsp,
                "Parch": parch,
                "Fare": fare,
                "FamilySize": family_size,
                "IsAlone": is_alone,
                "HasCabin": has_cabin,
                "FarePerPerson": fare_per_person,
                "FareLog": fare_log,
                "Sex": sex,
                "Embarked": embarked,
                "Title": title,
                "CabinDeck": cabin_deck,
                "FamilyCategory": family_cat,
                "AgeBin": age_bin,
            }
        ]
    )


# ─────────────────────────────────────────────────────────────────────────────
# Visualizaciones
# ─────────────────────────────────────────────────────────────────────────────


def plot_survival_by_feature(df: pd.DataFrame, feature: str, ax: plt.Axes) -> None:
    """Gráfica de tasa de supervivencia por categoría de un feature."""
    surv = df.groupby(feature)["Survived"].agg(["mean", "count"]).reset_index()
    surv.columns = [feature, "rate", "count"]
    surv = surv.sort_values("rate", ascending=False)

    colors = ["#2ECC71" if r >= df["Survived"].mean() else "#E74C3C" for r in surv["rate"]]
    bars = ax.bar(surv[feature].astype(str), surv["rate"], color=colors, edgecolor="white")
    ax.axhline(df["Survived"].mean(), ls="--", color="gray", alpha=0.7, label="Promedio global")
    ax.set_ylim(0, 1)
    ax.set_title(f"Supervivencia por {feature}", fontsize=11, fontweight="bold")
    ax.set_ylabel("Tasa de supervivencia")
    ax.legend(fontsize=8)
    for bar, row in zip(bars, surv.itertuples(), strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{row.rate:.0%}\n(n={row.count})",
            ha="center",
            fontsize=8,
        )


def plot_model_comparison(metrics: dict) -> plt.Figure:
    """Gráfica de comparación de métricas entre modelos."""
    if not metrics:
        return None

    metric_names = ["f1", "auroc", "recall", "precision", "accuracy"]
    labels_es = ["F1-Score", "AUC-ROC", "Recall", "Precision", "Accuracy"]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(metric_names))
    width = 0.35
    colors = {"sklearn": "#3498DB", "pytorch": "#E74C3C"}

    for i, (model_key, color) in enumerate(colors.items()):
        if model_key not in metrics:
            continue
        vals = [metrics[model_key].get(m, 0) for m in metric_names]
        offset = (i - 0.5) * width
        bars = ax.bar(
            x + offset,
            vals,
            width,
            label=model_key.upper(),
            color=color,
            alpha=0.85,
            edgecolor="white",
        )
        ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)

    ax.set_xticks(x)
    ax.set_xticklabels(labels_es)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Comparación de Modelos — Test Set", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_confusion_matrix(cm: np.ndarray, model_name: str) -> plt.Figure:
    """Visualiza una confusion matrix."""
    fig, ax = plt.subplots(figsize=(4, 3.5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    labels = ["No sobrevivió", "Sobrevivió"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                str(int(cm[i, j])),
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
                fontsize=12,
            )
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    ax.set_title(f"Confusion Matrix\n{model_name}", fontsize=10, fontweight="bold")
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Layout principal de la app
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    # ── Header
    st.title("🚢 Titanic Survival Predictor")
    st.markdown(
        "Predicción de supervivencia en el Titanic usando **Logistic Regression** "
        "(scikit-learn) y **MLP** (PyTorch). Proyecto Final Módulo 1 — ITAM."
    )
    st.divider()

    # ── Cargar recursos
    with st.spinner("Cargando modelos..."):
        models = load_models()
        preprocessor = load_preprocessor()
        metrics = load_metrics()
        df = load_dataset()

    # ── Sidebar: input del usuario
    st.sidebar.header("🎯 Datos del Pasajero")
    st.sidebar.markdown("Ingresa los datos para predecir la supervivencia.")

    pclass = st.sidebar.selectbox(
        "Clase del boleto", [1, 2, 3], format_func=lambda x: f"{x}ª clase"
    )
    sex = st.sidebar.radio(
        "Sexo",
        ["female", "male"],
        format_func=lambda x: "Femenino" if x == "female" else "Masculino",
    )
    age = st.sidebar.slider("Edad", min_value=0.5, max_value=80.0, value=30.0, step=0.5)
    sibsp = st.sidebar.slider("Hermanos / Cónyuge a bordo (SibSp)", 0, 8, 0)
    parch = st.sidebar.slider("Padres / Hijos a bordo (Parch)", 0, 6, 0)
    fare = st.sidebar.number_input(
        "Tarifa del boleto (£)", min_value=0.0, max_value=520.0, value=32.0, step=0.5
    )
    embarked = st.sidebar.selectbox(
        "Puerto de embarque",
        ["S", "C", "Q"],
        format_func={"S": "Southampton", "C": "Cherbourg", "Q": "Queenstown"}.get,
    )
    cabin_known = st.sidebar.checkbox("¿Cabina conocida?", value=False)

    st.sidebar.button("🔮 Predecir", type="primary", use_container_width=True)  # triggers rerun

    # ── Tabs principales
    tab_pred, tab_eda, tab_models = st.tabs(
        ["📊 Predicción", "🔍 Exploración de Datos", "⚖️ Comparación de Modelos"]
    )

    # ════════════════════════════════════════
    # TAB 1: PREDICCIÓN
    # ════════════════════════════════════════
    with tab_pred:
        st.subheader("Resultado de la Predicción")

        if True:  # Mostrar siempre con valores default (predict_btn triggers rerun)
            X_input = build_input_dataframe(
                pclass,
                sex,
                age,
                sibsp,
                parch,
                fare,
                embarked,
                cabin_known,
            )

            col1, col2 = st.columns(2)

            # ── Modelo sklearn
            with col1:
                st.markdown("### 🔵 Logistic Regression (sklearn)")
                if "sklearn" in models:
                    try:
                        proba_sk = models["sklearn"].predict_proba(X_input)[0]
                        pred_sk = int(proba_sk >= 0.5)
                        emoji = "✅ Sobrevive" if pred_sk == 1 else "❌ No sobrevive"

                        st.metric("Predicción", emoji)
                        st.metric("Probabilidad de supervivencia", f"{proba_sk:.1%}")

                        # Barra de probabilidad
                        fig_prob, ax_p = plt.subplots(figsize=(5, 1.2))
                        ax_p.barh(["Prob."], [proba_sk], color="#2ECC71", height=0.5)
                        ax_p.barh(
                            ["Prob."], [1 - proba_sk], left=[proba_sk], color="#E74C3C", height=0.5
                        )
                        ax_p.axvline(0.5, color="black", lw=1, ls="--")
                        ax_p.set_xlim(0, 1)
                        ax_p.set_xticks([0, 0.25, 0.5, 0.75, 1])
                        ax_p.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
                        ax_p.set_title(f"Supervivencia: {proba_sk:.1%}", fontsize=9)
                        fig_prob.tight_layout()
                        st.pyplot(fig_prob)
                        plt.close(fig_prob)

                    except Exception as e:
                        st.error(f"Error en predicción sklearn: {e}")
                else:
                    st.info(
                        "Modelo sklearn no disponible. Ejecuta: `python scripts/train_sklearn.py`"
                    )

            # ── Modelo PyTorch
            with col2:
                st.markdown("### 🔴 MLP (PyTorch)")
                if "pytorch" in models and preprocessor is not None:
                    try:
                        X_proc = preprocessor.transform(X_input)
                        proba_pt = models["pytorch"].predict_proba(X_proc)[0]
                        pred_pt = int(proba_pt >= 0.3516)  # umbral optimizado para Recall
                        emoji_pt = "✅ Sobrevive" if pred_pt == 1 else "❌ No sobrevive"

                        st.metric("Predicción", emoji_pt)
                        st.metric("Probabilidad de supervivencia", f"{proba_pt:.1%}")

                        fig_prob2, ax_p2 = plt.subplots(figsize=(5, 1.2))
                        ax_p2.barh(["Prob."], [proba_pt], color="#2ECC71", height=0.5)
                        ax_p2.barh(
                            ["Prob."], [1 - proba_pt], left=[proba_pt], color="#E74C3C", height=0.5
                        )
                        ax_p2.axvline(0.5, color="black", lw=1, ls="--")
                        ax_p2.set_xlim(0, 1)
                        ax_p2.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
                        ax_p2.set_title(f"Supervivencia: {proba_pt:.1%}", fontsize=9)
                        fig_prob2.tight_layout()
                        st.pyplot(fig_prob2)
                        plt.close(fig_prob2)

                    except Exception as e:
                        st.error(f"Error en predicción PyTorch: {e}")
                else:
                    st.info(
                        "Modelo PyTorch no disponible. Ejecuta: `python scripts/train_pytorch.py`"
                    )

            # ── Perfil del pasajero ingresado
            st.divider()
            st.subheader("Perfil del pasajero ingresado")
            fam_size = sibsp + parch + 1
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Clase": f"{pclass}ª",
                            "Sexo": "Femenino" if sex == "female" else "Masculino",
                            "Edad": age,
                            "Familia": fam_size,
                            "Tarifa (£)": fare,
                            "Embarque": {"S": "Southampton", "C": "Cherbourg", "Q": "Queenstown"}[
                                embarked
                            ],
                            "Cabina": "Conocida" if cabin_known else "Desconocida",
                        }
                    ]
                ),
                use_container_width=True,
            )

    # ════════════════════════════════════════
    # TAB 2: EDA
    # ════════════════════════════════════════
    with tab_eda:
        st.subheader("Exploración del Dataset Titanic")

        if df is not None:
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Pasajeros totales", len(df))
            col_b.metric("Sobrevivieron", f"{df['Survived'].sum()} ({df['Survived'].mean():.1%})")
            col_c.metric(
                "No sobrevivieron", f"{(1-df['Survived']).sum()} ({(1-df['Survived'].mean()):.1%})"
            )

            st.divider()

            # Gráfica 1: distribución de edad
            st.markdown("#### Distribución de Edad por Supervivencia")
            fig1, ax1 = plt.subplots(figsize=(9, 3.5))
            for s, lbl, c in [(0, "No sobrevivió", "#E74C3C"), (1, "Sobrevivió", "#2ECC71")]:
                data = df[df["Survived"] == s]["Age"].dropna()
                ax1.hist(data, bins=25, alpha=0.5, color=c, label=lbl, edgecolor="white")
            ax1.set_xlabel("Edad")
            ax1.set_ylabel("Frecuencia")
            ax1.set_title("Distribución de Edad por Supervivencia")
            ax1.legend()
            st.pyplot(fig1)
            plt.close(fig1)

            # Gráfica 2: supervivencia por Sexo y Clase
            st.markdown("#### Supervivencia por Sexo y Clase")
            fig2, axes2 = plt.subplots(1, 2, figsize=(11, 4))
            plot_survival_by_feature(df, "Sex", axes2[0])
            axes2[0].set_xticklabels(["Femenino", "Masculino"])
            plot_survival_by_feature(df, "Pclass", axes2[1])
            axes2[1].set_xticklabels(["1ª", "2ª", "3ª"])
            fig2.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)

            # Gráfica 3: heatmap Pclass × Sex
            st.markdown("#### Mapa de calor: Clase × Sexo")
            import seaborn as sns

            pivot = df.pivot_table(values="Survived", index="Pclass", columns="Sex", aggfunc="mean")
            fig3, ax3 = plt.subplots(figsize=(6, 3))
            sns.heatmap(
                pivot,
                annot=True,
                fmt=".2%",
                cmap="RdYlGn",
                ax=ax3,
                linewidths=0.5,
                linecolor="white",
            )
            ax3.set_title("Tasa de supervivencia: Clase × Sexo")
            ax3.set_yticklabels(["1ª", "2ª", "3ª"], rotation=0)
            fig3.tight_layout()
            st.pyplot(fig3)
            plt.close(fig3)

        else:
            st.info("Dataset no disponible. Ejecuta: `python scripts/ingest.py`")

    # ════════════════════════════════════════
    # TAB 3: COMPARACIÓN DE MODELOS
    # ════════════════════════════════════════
    with tab_models:
        st.subheader("Comparación de Modelos")

        if metrics:
            # Tabla de métricas
            rows = []
            for model_key, m in metrics.items():
                rows.append(
                    {
                        "Modelo": "Logistic Regression"
                        if model_key == "sklearn"
                        else "MLP PyTorch",
                        "F1-Score": f"{m.get('f1', 0):.4f}",
                        "AUC-ROC": f"{m.get('auroc', 0):.4f}",
                        "Recall": f"{m.get('recall', 0):.4f}",
                        "Precision": f"{m.get('precision', 0):.4f}",
                        "Accuracy": f"{m.get('accuracy', 0):.4f}",
                    }
                )
            st.dataframe(pd.DataFrame(rows).set_index("Modelo"), use_container_width=True)

            # Gráfica de barras comparativa
            fig_cmp = plot_model_comparison(metrics)
            if fig_cmp:
                st.pyplot(fig_cmp)
                plt.close(fig_cmp)

            st.divider()
            st.markdown("""
            **¿Por qué estas dos métricas son las más importantes?**
            - **F1-Score**: métrica principal con datasets desbalanceados (~38% positivos).
              Equilibra Precision y Recall — un modelo que predice siempre "no sobrevivió"
              tendría accuracy de 62% pero F1 de 0.
            - **AUC-ROC**: mide la capacidad discriminativa independientemente del umbral.
              Un valor de 0.86+ indica que el modelo separa bien las dos clases.

            **Sobre el umbral del MLP**: se usa 0.3516 (calibrado para maximizar Recall)
            porque minimizar Falsos Negativos (sobrevivientes que el modelo no detecta)
            es más importante que minimizar Falsos Positivos en este contexto.
            """)
        else:
            st.info("Métricas no disponibles. Entrena los modelos primero.")

        # Info de los modelos
        st.divider()
        with st.expander("ℹ️ Acerca de los modelos"):
            st.markdown("""
            #### Logistic Regression (scikit-learn)
            - **Pipeline**: ColumnTransformer (StandardScaler + OneHotEncoder) + LogisticRegression
            - **Hiperparámetros**: optimizados con GridSearchCV (5-fold StratifiedKFold)
            - **class_weight='balanced'**: compensa el desbalance de clases
            - **Ventaja**: interpretable, rápido, buen baseline

            #### MLP (PyTorch)
            - **Arquitectura**: Input(38) → 256×4 → 128×3 → 64 → 32 → 1 (logit)
            - **Activación**: GELU (evita dead neurons, estándar en Transformers)
            - **Regularización**: BatchNorm1d + Dropout decreciente (0.3 → 0.2)
            - **Optimizer**: AdamW + CosineAnnealingLR + EarlyStopping (patience=10)
            - **pos_weight = n_neg/n_pos × 2.0**: maximiza Recall
            - **Ventaja**: captura interacciones no lineales complejas

            #### Features Engineered (ambos modelos)
            | Feature | Descripción |
            |---|---|
            | Title | Título del nombre (Mr./Mrs./Miss./Master./Rare) |
            | FamilySize | SibSp + Parch + 1 |
            | IsAlone | 1 si viaja solo |
            | FamilyCategory | Solo/Pequeña/Mediana/Grande |
            | FareLog | log(1+Fare) — reduce sesgo |
            | HasCabin | 1 si tiene cabina registrada |
            | AgeBin | Niño/Adolescente/Adulto_Joven/Adulto/Mayor |
            """)


if __name__ == "__main__":
    main()
