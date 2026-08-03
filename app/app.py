"""Streamlit GUI for the creep (J) and shrinkage (e) prediction models
ported from CreepABNT, CreepB4, ShrinkageABNT, ShrinkageB4, XGBoost_NR and
XGBoost_shrinkage_NR.

Run with:
    streamlit run app/app.py
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import data_pipeline as dp
from app import formulas as f
from app import ml_models as ml

MODELS_DIR = ROOT / "models"

# Models that only partially use the selected cement type -- see the
# "Limitações conhecidas" note in the sidebar for details.
PARTIAL_CEMENT_EFFECT = {("shrinkage", "B4")}

# --- Estilo dos gráficos (aparência LaTeX/artigo científico) -----------
_HAS_LATEX = shutil.which("latex") is not None and shutil.which("dvipng") is not None
plt.rcParams.update({
    "text.usetex": _HAS_LATEX,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "cmr10", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 13,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.linewidth": 0.9,
    "lines.linewidth": 2.0,
    "figure.dpi": 150,
    "axes.grid": True,
    "grid.linestyle": ":",
    "grid.alpha": 0.5,
    "legend.frameon": True,
    "legend.edgecolor": "black",
    "legend.fancybox": False,
})

st.set_page_config(page_title="Fluência & Retração do Concreto", layout="wide")


def style_axes(ax) -> None:
    """Applies a consistent, journal-style look to a plot axis."""
    ax.tick_params(direction="in", which="both", top=True, right=True)
    ax.minorticks_on()
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)
    ax.set_axisbelow(True)


@st.cache_resource
def load_artifacts():
    """Loads the cached RF/XGBoost models if `python app/train_models.py` was
    already run locally; otherwise trains them once, in-process (~15-20s,
    hyperparameters are already fixed -- no GridSearch is re-run). This lets
    the app be deployed straight from GitHub without shipping the ~90MB
    model file: Streamlit Cloud will just train on first load and cache the
    result in memory for the life of that instance.
    """
    path = MODELS_DIR / "artifacts.joblib"
    if path.exists():
        return joblib.load(path)
    with st.spinner("Treinando Random Forest e XGBoost pela primeira vez (~20s)..."):
        return ml.train_all()


@st.cache_data
def load_metrics():
    path = MODELS_DIR / "metrics.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


artifacts = load_artifacts()
metrics = load_metrics()

st.title("Previsão de Fluência (J) e Retração (ε) do Concreto")
st.caption(
    "ABNT NBR 6118, ACI 209 (B4), Random Forest e XGBoost treinados na base NU-ITI "
    "(Bažant & Li) — insira as propriedades do concreto e compare as curvas previstas."
)

if artifacts is None:
    st.warning(
        "Os modelos de Machine Learning (Random Forest / XGBoost) ainda não foram "
        "treinados. Rode `python app/train_models.py` no terminal e recarregue esta "
        "página para habilitar essas abas (as fórmulas ABNT/B4 já funcionam)."
    )

# --- Entradas -------------------------------------------------------

quantity_label = st.sidebar.radio("Grandeza", ["Fluência (J)", "Retração (ε)"])
quantity = "creep" if quantity_label.startswith("Fluência") else "shrinkage"

CEMENT_OPTIONS = ["N", "R", "RS", "SL"]

st.sidebar.header("Propriedades do concreto")
fc28 = st.sidebar.number_input("fc28 — resistência aos 28 dias (MPa)", 5.0, 120.0, 33.3, step=1.0)
e28_input = st.sidebar.number_input(
    "E28 — módulo de elasticidade aos 28 dias (MPa) — 0 = calcular automaticamente",
    0.0, 60000.0, 29800.0, step=100.0,
)
wc = st.sidebar.number_input("a/c — relação água/cimento", 0.2, 1.2, 0.56, step=0.01)
ac = st.sidebar.number_input("agr/c — relação agregado/cimento", 0.5, 10.0, 3.814, step=0.1)
cement_kg = st.sidebar.number_input("Cimento (kg/m³)", 100.0, 600.0, 280.0, step=10.0)
vs_ratio = st.sidebar.number_input("Relação Volume/Superfície (V/S, mm)", 5.0, 300.0, 37.5, step=1.0)
length_radius = st.sidebar.number_input("Length/Radius (mm)", 10.0, 1000.0, 150.0, step=10.0)
height = st.sidebar.number_input("Altura / Height (mm)", 10.0, 2000.0, 300.0, step=10.0)
if quantity == "creep":
    t0 = st.sidebar.number_input("Idade de carregamento, t0 (dias)", 0.5, 365.0, 7.0, step=1.0)
else:
    t0 = st.sidebar.number_input("Idade de início da secagem, tc (dias)", 0.5, 365.0, 7.0, step=1.0)
humidity = st.sidebar.number_input("Umidade relativa do ambiente (%)", 40.0, 100.0, 60.0, step=1.0)
temp = st.sidebar.number_input("Temperatura (°C)", 0.0, 60.0, 23.0, step=1.0)
cement_type = st.sidebar.selectbox("Tipo de cimento", CEMENT_OPTIONS, index=1)

st.sidebar.header("Faixa de tempo da curva")
t_max = st.sidebar.slider("Duração máxima (dias)", 28, 3650, 84)
n_points = st.sidebar.slider("Pontos da curva", 20, 200, 84)
log_x = st.sidebar.checkbox("Eixo do tempo em escala logarítmica", value=(quantity == "creep"))

with st.sidebar.expander("Limitações conhecidas"):
    st.markdown(
        "- **B4 (retração)** — o tipo de cimento agora altera a parcela de "
        "secagem (usando as mesmas tabelas por tipo do modelo B4 de fluência), "
        "mas a parcela autógena continua fixa nos parâmetros 'Tipo R', pois não "
        "há calibração por tipo de cimento para essa parcela na literatura de "
        "origem.\n"
        "- As métricas das fórmulas ABNT/B4 comparam a fórmula fechada contra "
        "toda a base de dados (não é um teste de ML com dados nunca vistos)."
    )

e28_user = e28_input if e28_input > 0 else None


def resolve_e28(model_kind: str) -> float:
    """Mirrors each notebook's own handling of a missing E28:
    - ABNT formulas have their own internal fallback (0.9*5600*sqrt(fc28)),
      so we pass NaN through and let it trigger.
    - Every other model (B4, RF, XGBoost) has no such fallback in the
      notebook, so we impute it ourselves the same way the *dataset*
      preprocessing does for missing raw E28 (4734*sqrt(fc28)).
    """
    if e28_user is not None:
        return e28_user
    if model_kind == "abnt":
        return float("nan")
    return dp.E28_FALLBACK_K * np.sqrt(fc28)


def base_props(e28_value: float) -> dict:
    return {
        "wc": wc, "ac": ac, "cement_kg": cement_kg, "fc28": fc28, "e28": e28_value,
        "length_radius": length_radius, "height": height, "vs_ratio": vs_ratio,
        "t0": t0, "temp": temp, "humidity": humidity,
    }


tempos = np.linspace(0.01 if quantity == "creep" else 1.0, float(t_max), int(n_points))

colmap = dp.CREEP_COLMAP if quantity == "creep" else dp.SHRINK_COLMAP
cement_prefix = dp.CREEP_CEMENT_PREFIX if quantity == "creep" else dp.SHRINK_CEMENT_PREFIX
cement_cats_formula = dp.CREEP_CEMENT_FULL_CATS if quantity == "creep" else dp.SHRINK_CEMENT_FULL_CATS


def curve_abnt() -> np.ndarray:
    props = base_props(resolve_e28("abnt"))
    fn = f.fluencia_abnt if quantity == "creep" else f.modelo_abnt
    out_col = "J_total" if quantity == "creep" else "ecs_ue"
    return f.sweep_formula(fn, colmap, cement_prefix, cement_cats_formula, tempos, props,
                            cement_type, merge_nr=False, output_col=out_col)


def curve_b4() -> np.ndarray:
    props = base_props(resolve_e28("b4"))
    fn = f.calcular_fluencia_B4 if quantity == "creep" else f.calcular_retracao_b4_baseline_tc
    return f.sweep_formula(fn, colmap, cement_prefix, cement_cats_formula, tempos, props,
                            cement_type, merge_nr=False)


def curve_ml(model_key: str):
    if artifacts is None:
        return None
    props = base_props(resolve_e28(model_key))
    return ml.predict_curve(quantity, model_key, artifacts, tempos, props, cement_type)


Y_LABEL = r"Conformidade à fluência, $J(t,t_0)$ ($\mu\varepsilon$/MPa)" if quantity == "creep" \
    else r"Deformação de retração, $\varepsilon_{cs}$ ($\mu\varepsilon$)"
X_LABEL = r"Tempo de carregamento, $t - t_0$ (dias)" if quantity == "creep" \
    else r"Tempo de secagem, $t - t_c$ (dias)"


def formula_metrics(name: str) -> dict | None:
    key = {"ABNT": "abnt", "B4": "b4"}[name]
    return metrics.get("formulas", {}).get(f"{quantity}_{key}")


def ml_metrics(model_key: str) -> dict | None:
    cached = metrics.get(quantity, {}).get(model_key)
    if cached:
        return cached
    if artifacts is not None:
        return artifacts[quantity][model_key].get("metrics")
    return None


def render_tab(tab, name: str, curve_fn, model_metrics: dict | None) -> None:
    with tab:
        if (quantity, name) in PARTIAL_CEMENT_EFFECT:
            st.caption(
                "⚠️ O tipo de cimento afeta apenas a parcela de secagem deste modelo — "
                "a parcela autógena usa um único conjunto de parâmetros para todos os tipos "
                "(sem calibração por tipo disponível na literatura de origem)."
            )
        try:
            y = curve_fn()
        except Exception as exc:  # noqa: BLE001 - surface any formula error to the user
            st.error(f"Não foi possível calcular esta curva: {exc}")
            return
        if y is None:
            st.info("Modelo ainda não treinado — rode `python app/train_models.py`.")
            return

        fig, ax = plt.subplots(figsize=(7.5, 5))
        if log_x:
            ax.semilogx(tempos, y, color="black", linewidth=2, solid_capstyle="round")
        else:
            ax.plot(tempos, y, color="black", linewidth=2, solid_capstyle="round")
        ax.set_xlabel(X_LABEL)
        ax.set_ylabel(Y_LABEL)
        ax.set_title(f"Modelo {name} -- cimento {cement_type}" if _HAS_LATEX
                     else f"Modelo {name} — cimento {cement_type}")
        style_axes(ax)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        if model_metrics:
            c1, c2, c3 = st.columns(3)
            c1.metric("R²", f"{model_metrics['r2']:.3f}")
            c2.metric("RMSE", f"{model_metrics['rmse']:.2f}")
            c3.metric("MAE", f"{model_metrics['mae']:.2f}")

        df_out = pd.DataFrame({"tempo_dias": tempos, "valor": y})
        st.download_button(
            "Baixar curva (CSV)", df_out.to_csv(index=False).encode("utf-8"),
            file_name=f"{quantity}_{name.lower().replace(' ', '_')}.csv", mime="text/csv",
            key=f"download_{quantity}_{name}",
        )
        st.dataframe(df_out, width="stretch", height=200)


tab_abnt, tab_b4, tab_rf, tab_xgb, tab_all = st.tabs(
    ["ABNT", "B4", "Random Forest", "XGBoost", "Comparar todos"]
)

render_tab(tab_abnt, "ABNT", curve_abnt, formula_metrics("ABNT"))
render_tab(tab_b4, "B4", curve_b4, formula_metrics("B4"))
render_tab(tab_rf, "Random Forest", lambda: curve_ml("rf"), ml_metrics("rf"))
render_tab(tab_xgb, "XGBoost", lambda: curve_ml("xgb"), ml_metrics("xgb"))

with tab_all:
    st.caption("Sobreposição das curvas de todos os modelos disponíveis para as mesmas propriedades.")
    series = {
        "ABNT": ("black", "-", curve_abnt),
        "B4": ("tab:blue", "--", curve_b4),
        "Random Forest": ("tab:green", "-.", lambda: curve_ml("rf")),
        "XGBoost": ("tab:red", ":", lambda: curve_ml("xgb")),
    }
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    combined = {"tempo_dias": tempos}
    for label, (color, linestyle, fn) in series.items():
        try:
            y = fn()
        except Exception:
            continue
        if y is None:
            continue
        combined[label] = y
        if log_x:
            ax.semilogx(tempos, y, label=label, color=color, linestyle=linestyle, linewidth=2)
        else:
            ax.plot(tempos, y, label=label, color=color, linestyle=linestyle, linewidth=2)
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(Y_LABEL)
    ax.set_title(f"Comparação de modelos -- cimento {cement_type}" if _HAS_LATEX
                 else f"Comparação de modelos — cimento {cement_type}")
    style_axes(ax)
    ax.legend(loc="best")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    df_all = pd.DataFrame(combined)
    st.download_button(
        "Baixar comparação (CSV)", df_all.to_csv(index=False).encode("utf-8"),
        file_name=f"{quantity}_comparacao.csv", mime="text/csv", key=f"download_all_{quantity}",
    )
    st.dataframe(df_all, width="stretch", height=250)
