"""Streamlit GUI for the creep (J) and shrinkage (e) prediction models
ported from CreepABNT, CreepB4, ShrinkageABNT, ShrinkageB4, XGBoost_NR and
XGBoost_shrinkage_NR.

Run with:
    streamlit run app/app.py
"""
from __future__ import annotations

import re
import sys
import urllib.request
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
CEMENT_EFFECT_NOTE = {
    ("shrinkage", "NBR 6118:2023"): (
        "⚠️ Pela NBR 6118, a **retração** não depende do tipo de cimento (só a "
        "**fluência** depende, pelo coeficiente de endurecimento) — por isso "
        "trocar o cimento aqui não altera a curva."
    ),
}

# --- Estilo dos gráficos (aparência LaTeX/artigo científico) -----------
# Deliberately NOT using text.usetex=True: it depends on a full system LaTeX
# install (latex/dvipng/ghostscript + the right packages for UTF-8 Portuguese
# accents), which we can't guarantee on Streamlit Community Cloud's shared
# containers. "DejaVu Serif" ships with matplotlib itself, renders accented
# text correctly everywhere, and mathtext's "cm" fontset still gives the
# LaTeX/Computer-Modern look for the actual math notation ($t-t_c$, etc.).
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9.5,
    "legend.fontsize": 8.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.6,
    "figure.dpi": 160,
    "axes.grid": True,
    "grid.linestyle": ":",
    "grid.alpha": 0.5,
    "legend.frameon": True,
    "legend.edgecolor": "black",
    "legend.fancybox": False,
})

st.set_page_config(page_title="Fluência & Retração do Concreto", layout="wide")

# --- Layout compacto: reduz espacamento padrao do Streamlit e o tamanho
# das fontes dos widgets, para caber mais coisa na tela sem rolar. ---
st.markdown("""
<style>
    .block-container {padding-top: 3.2rem; padding-bottom: 1rem; max-width: 1400px;}
    div[data-testid="stVerticalBlockBorderWrapper"], div[data-testid="stVerticalBlock"] {gap: 0.35rem;}
    section[data-testid="stSidebar"] .block-container {padding-top: 1.8rem;}
    div[data-testid="stSidebarUserContent"] {gap: 0.2rem;}
    h1 {font-size: 1.35rem !important; margin-bottom: 0.2rem;}
    h3 {font-size: 1rem !important; margin-top: 0.4rem; margin-bottom: 0.2rem;}
    .stCaption, [data-testid="stCaptionContainer"] {font-size: 0.78rem !important;}
    label, .stNumberInput label, .stSelectbox label, .stSlider label {font-size: 0.8rem !important;}
    div[data-testid="stNumberInput"] input, div[data-testid="stSelectbox"] {font-size: 0.82rem !important;}
    .stTabs [data-baseweb="tab-list"] {gap: 4px;}
    .stTabs [data-baseweb="tab"] {font-size: 0.85rem; padding: 0.35rem 0.7rem; height: auto;}
    div[data-testid="stExpander"] p {font-size: 0.78rem;}
</style>
""", unsafe_allow_html=True)


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def style_axes(ax) -> None:
    """Applies a consistent, journal-style look to a plot axis."""
    ax.tick_params(direction="in", which="both", top=True, right=True)
    ax.minorticks_on()
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)
    ax.set_axisbelow(True)


# Optional: URL of a GitHub Release asset with the pre-trained artifacts.joblib.
# If set and reachable, the app downloads it instead of training live -- much
# faster and lighter than training on Streamlit Community Cloud's shared,
# memory-constrained free containers (live training there has been observed
# to hang for several minutes instead of the ~20s seen on a normal machine).
ARTIFACT_RELEASE_URL = (
    "https://github.com/DavidHBarzotto/ML-Mestrado/releases/download/Modelos-treinados-v1/artifacts.joblib"
)


@st.cache_resource
def load_artifacts():
    """Loads the cached RF/XGBoost models if present locally (from
    `python app/train_models.py` or a previous download); otherwise tries to
    download a pre-trained copy from ARTIFACT_RELEASE_URL; if that also
    isn't available, falls back to training in-process (~15-20s on a normal
    machine, hyperparameters are already fixed -- no GridSearch is re-run).
    """
    path = MODELS_DIR / "artifacts.joblib"
    if path.exists():
        return joblib.load(path)

    if ARTIFACT_RELEASE_URL:
        try:
            with st.spinner("Baixando modelos pré-treinados..."):
                MODELS_DIR.mkdir(exist_ok=True)
                tmp_path = path.with_suffix(".part")
                urllib.request.urlretrieve(ARTIFACT_RELEASE_URL, tmp_path)
                tmp_path.rename(path)
            return joblib.load(path)
        except Exception:
            if path.exists():
                path.unlink(missing_ok=True)

    with st.spinner("Treinando Random Forest e XGBoost pela primeira vez (~20s)..."):
        return ml.train_all()


artifacts = load_artifacts()

st.title("Fluência (J) e Retração (ε) do Concreto")
st.caption(
    "NBR 6118:2023 · ACI 209 (B4) · Random Forest · XGBoost — base NU-ITI (Bažant & Li)"
)

if artifacts is None:
    st.warning(
        "Modelos de ML ainda não treinados. Rode `python app/train_models.py` e "
        "recarregue a página (as fórmulas NBR 6118:2023/B4 já funcionam sem isso)."
    )

# --- Entradas -------------------------------------------------------

quantity_label = st.sidebar.radio("Grandeza", ["Fluência (J)", "Retração (ε)"], horizontal=True)
quantity = "creep" if quantity_label.startswith("Fluência") else "shrinkage"

CEMENT_OPTIONS = ["N/R", "RS", "SL"]

decimals = st.sidebar.number_input("Casas decimais (entradas e tabelas)", 0, 6, 2, step=1)
fmt = f"%.{int(decimals)}f"

st.sidebar.markdown("**Propriedades do concreto**")
c1, c2 = st.sidebar.columns(2)
fc28 = c1.number_input("fc28 (MPa)", 5.0, 120.0, 33.3, step=1.0, format=fmt, help="Resistência à compressão aos 28 dias")
e28_input = c2.number_input("E28 (MPa)", 0.0, 60000.0, 29800.0, step=100.0, format=fmt, help="Módulo de elasticidade aos 28 dias — 0 = calcular automaticamente")

c1, c2 = st.sidebar.columns(2)
wc = c1.number_input("a/c", 0.2, 1.2, 0.56, step=0.01, format=fmt, help="Relação água/cimento")
ac = c2.number_input("agr/c", 0.5, 10.0, 3.814, step=0.1, format=fmt, help="Relação agregado/cimento")

c1, c2 = st.sidebar.columns(2)
cement_kg = c1.number_input("Cimento (kg/m³)", 100.0, 600.0, 280.0, step=10.0, format=fmt)
vs_ratio = c2.number_input("V/S (mm)", 5.0, 300.0, 37.5, step=1.0, format=fmt, help="Relação Volume/Superfície")

c1, c2 = st.sidebar.columns(2)
length_radius = c1.number_input("L/R (mm)", 10.0, 1000.0, 150.0, step=10.0, format=fmt, help="Length/Radius")
height = c2.number_input("Altura (mm)", 10.0, 2000.0, 300.0, step=10.0, format=fmt)

c1, c2 = st.sidebar.columns(2)
if quantity == "creep":
    t0 = c1.number_input("t0 (dias)", 0.5, 365.0, 7.0, step=1.0, format=fmt, help="Idade de carregamento")
else:
    t0 = c1.number_input("tc (dias)", 0.5, 365.0, 7.0, step=1.0, format=fmt, help="Idade de início da secagem")
humidity = c2.number_input("Umidade (%)", 40.0, 100.0, 60.0, step=1.0, format=fmt)

c1, c2 = st.sidebar.columns(2)
temp = c1.number_input("Temp. (°C)", 0.0, 60.0, 23.0, step=1.0, format=fmt)
cement_display = c2.selectbox("Cimento", CEMENT_OPTIONS, index=0)
# N and R are kept as a single option everywhere in the UI: RF/XGBoost were
# already trained with N and R merged into one category, and ABNT/B4 group
# N with RS in their own tables -- so "N" and "R" almost never behave
# differently enough to justify separate menu entries, and having both
# confused more than it helped. "N" is used internally as the representative
# value wherever a formula needs one specific flag.
cement_type = "N" if cement_display == "N/R" else cement_display

st.sidebar.markdown("**Faixa de tempo da curva**")
c1, c2 = st.sidebar.columns(2)
t_min = c1.number_input("Tempo inicial (dias)", 0.01, 3650.0, 0.01, step=0.1, format="%.2f")
t_max = c2.number_input("Tempo final (dias)", 0.1, 3650.0, 84.0, step=1.0)
if t_max <= t_min:
    st.sidebar.warning("Tempo final ajustado — deve ser maior que o inicial.")
    t_max = t_min + 1.0

n_points = st.sidebar.slider("Pontos da curva", 20, 200, 84)
log_x = st.sidebar.checkbox("Eixo do tempo em escala logarítmica", value=False)

with st.sidebar.expander("Limitações conhecidas"):
    st.markdown(
        "- **NBR 6118:2023 (retração)** não depende do tipo de cimento — a "
        "norma só faz essa distinção na fórmula da fluência.\n"
        "- **NBR 6118:2023** trata os cimentos N e RS como idênticos — é "
        "assim que a norma agrupa esses tipos, não é uma limitação da "
        "modelagem.\n"
        "- O seletor de cimento junta **N e R em uma única opção** ('N/R') em "
        "todos os modelos — internamente, o Random Forest/XGBoost já foram "
        "treinados tratando N e R como a mesma categoria, e nas fórmulas "
        "NBR 6118:2023/B4 essa opção usa os parâmetros do cimento N."
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


tempos = np.linspace(float(t_min), float(t_max), int(n_points))

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


def render_tab(tab, name: str, curve_fn) -> None:
    with tab:
        note = CEMENT_EFFECT_NOTE.get((quantity, name))
        if note:
            st.caption(note)
        try:
            y = curve_fn()
        except Exception as exc:  # noqa: BLE001 - surface any formula error to the user
            st.error(f"Não foi possível calcular esta curva: {exc}")
            return
        if y is None:
            st.info("Modelo ainda não treinado — rode `python app/train_models.py`.")
            return

        col_plot, col_data = st.columns([3, 2])

        with col_plot:
            fig, ax = plt.subplots(figsize=(5.4, 3.6))
            if log_x:
                ax.semilogx(tempos, y, color="black", linewidth=1.6, solid_capstyle="round")
            else:
                ax.plot(tempos, y, color="black", linewidth=1.6, solid_capstyle="round")
            ax.set_xlabel(X_LABEL)
            ax.set_ylabel(Y_LABEL)
            ax.set_title(f"Modelo {name} — cimento {cement_display}")
            style_axes(ax)
            fig.tight_layout()
            st.pyplot(fig, width="content")
            plt.close(fig)

        df_out = pd.DataFrame({"tempo_dias": tempos, "valor": y}).round(int(decimals))
        with col_data:
            st.download_button(
                "Baixar curva (CSV)", df_out.to_csv(index=False).encode("utf-8"),
                file_name=f"{quantity}_{_slug(name)}.csv", mime="text/csv",
                key=f"download_{quantity}_{name}",
            )
            st.dataframe(df_out, width="stretch", height=300)


tab_abnt, tab_b4, tab_rf, tab_xgb, tab_all = st.tabs(
    ["NBR 6118:2023", "B4", "Random Forest", "XGBoost", "Comparar todos"]
)

render_tab(tab_abnt, "NBR 6118:2023", curve_abnt)
render_tab(tab_b4, "B4", curve_b4)
render_tab(tab_rf, "Random Forest", lambda: curve_ml("rf"))
render_tab(tab_xgb, "XGBoost", lambda: curve_ml("xgb"))

with tab_all:
    st.caption("Sobreposição das curvas de todos os modelos disponíveis para as mesmas propriedades.")
    series = {
        "NBR 6118:2023": ("black", "-", curve_abnt),
        "B4": ("tab:blue", "--", curve_b4),
        "Random Forest": ("tab:green", "-.", lambda: curve_ml("rf")),
        "XGBoost": ("tab:red", ":", lambda: curve_ml("xgb")),
    }
    combined = {"tempo_dias": tempos}
    col_plot, col_data = st.columns([3, 2])

    with col_plot:
        fig, ax = plt.subplots(figsize=(5.8, 3.8))
        for label, (color, linestyle, fn) in series.items():
            try:
                y = fn()
            except Exception:
                continue
            if y is None:
                continue
            combined[label] = y
            if log_x:
                ax.semilogx(tempos, y, label=label, color=color, linestyle=linestyle, linewidth=1.6)
            else:
                ax.plot(tempos, y, label=label, color=color, linestyle=linestyle, linewidth=1.6)
        ax.set_xlabel(X_LABEL)
        ax.set_ylabel(Y_LABEL)
        ax.set_title(f"Comparação de modelos — cimento {cement_display}")
        style_axes(ax)
        ax.legend(loc="best")
        fig.tight_layout()
        st.pyplot(fig, width="content")
        plt.close(fig)

    df_all = pd.DataFrame(combined).round(int(decimals))
    with col_data:
        st.download_button(
            "Baixar comparação (CSV)", df_all.to_csv(index=False).encode("utf-8"),
            file_name=f"{quantity}_comparacao.csv", mime="text/csv", key=f"download_all_{quantity}",
        )
        st.dataframe(df_all, width="stretch", height=300)
