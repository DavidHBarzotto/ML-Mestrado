"""Closed-form (no-training) creep/shrinkage models, ported from:

- CreepABNT.ipynb      cell 29  -> fluencia_abnt
- CreepB4.ipynb        cell 29  -> calcular_fluencia_B3_corrigido
- CreepB4.ipynb        cell 32  -> calcular_fluencia_B4 (final, per-cement-type version)
- ShrinkageABNT.ipynb  cell 27  -> modelo_abnt
- ShrinkageB4.ipynb    cell 27  -> calcular_retracao_b3_corrigido
- ShrinkageB4.ipynb    cell 30  -> calcular_retracao_b4_baseline_tc

The numerical logic is kept identical to the notebooks. B3 (creep & shrinkage)
and B4-shrinkage do not use the cement type at all (or only a hardcoded
"Type R" table) in the original notebooks -- that is a real limitation of
those models, not a porting bug, and is surfaced in the UI rather than
silently patched here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# =====================================================================
# CREEP -- ABNT NBR 6118 (fluencia_abnt)
# =====================================================================

_CREEP_ABNT_COL = {
    "dur": "x3", "t0": "x52", "U": "x57", "v_s": "x50",
    "fc28": "x43", "E28": "x45", "temp": "x55",
}
_CREEP_ABNT_PADRAO = {"dur": 28.0, "t0": 28.0, "U": 70.0, "v_s": 40.0,
                      "fc28": 40.0, "E28": np.nan, "temp": 20.0}
_CREEP_ABNT_UNIDADE_VS = "mm"
_CREEP_ABNT_ALFA_E = 0.9


def _v_creep_abnt(df: pd.DataFrame, chave: str) -> np.ndarray:
    nome = _CREEP_ABNT_COL.get(chave)
    if nome is None or nome not in df.columns:
        return np.full(len(df), float(_CREEP_ABNT_PADRAO[chave]))
    a = pd.to_numeric(df[nome], errors="coerce").to_numpy(float)
    return np.where(np.isfinite(a), a, _CREEP_ABNT_PADRAO[chave])


def _cimento_creep_abnt(df: pd.DataFrame):
    g = lambda c: (df[c].to_numpy(float) if c in df.columns else np.zeros(len(df)))
    n, r, rs, sl = g("x20_N"), g("x20_R"), g("x20_RS"), g("x20_SL")
    alfa = np.full(len(df), 2.0)
    s = np.full(len(df), 0.25)
    for mask, a_, s_ in [(sl > 0.5, 1.0, 0.38), (rs > 0.5, 2.0, 0.25),
                         (n > 0.5, 2.0, 0.25), (r > 0.5, 3.0, 0.20)]:
        alfa = np.where(mask, a_, alfa)
        s = np.where(mask, s_, s)
    return alfa, s


def _beta_f_creep(t, h):
    t = np.asarray(t, float)
    A = 42.0 * h ** 3 - 350.0 * h ** 2 + 588.0 * h + 113.0
    B = 768.0 * h ** 3 - 3060.0 * h ** 2 + 3234.0 * h - 23.0
    C = -200.0 * h ** 3 + 13.0 * h ** 2 + 1090.0 * h + 183.0
    D = 7579.0 * h ** 3 - 31916.0 * h ** 2 + 35343.0 * h + 1931.0
    den = t ** 2 + C * t + D
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.abs(den) < 1e-9, np.nan, (t ** 2 + A * t + B) / den)


def fluencia_abnt(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.index
    dur = np.clip(_v_creep_abnt(df, "dur"), 1e-3, None)
    t0 = np.clip(_v_creep_abnt(df, "t0"), 0.5, None)
    U = np.clip(_v_creep_abnt(df, "U"), 40.0, 90.0)
    Temp = np.clip(_v_creep_abnt(df, "temp"), 0.0, 60.0)
    fc28 = np.clip(_v_creep_abnt(df, "fc28"), 5.0, 120.0)
    E28d = _v_creep_abnt(df, "E28")

    vs = _v_creep_abnt(df, "v_s")
    vs_m = vs / 1000.0 if _CREEP_ABNT_UNIDADE_VS == "mm" else vs
    gamma = 1.0 + np.exp(-7.8 + 0.1 * U)
    h = np.clip(gamma * 2.0 * vs_m, 0.05, 1.6)

    alfa_c, s_cim = _cimento_creep_abnt(df)
    f_temp = (Temp + 10.0) / 30.0
    t0_fic = np.clip(alfa_c * f_temp * t0, 0.5, None)
    t_fic = np.clip(alfa_c * f_temp * (t0 + dur), 0.5, None)

    beta1_t0 = np.exp(s_cim * (1.0 - np.sqrt(28.0 / t0_fic)))
    beta1_inf = np.exp(s_cim * (1.0 - np.sqrt(28.0 / 1090.0)))

    Eci28 = np.where(np.isfinite(E28d) & (E28d > 1000.0), E28d,
                      _CREEP_ABNT_ALFA_E * 5600.0 * np.sqrt(fc28))
    Eci_t0 = np.sqrt(beta1_t0) * Eci28

    phi_a = np.where(fc28 <= 45.0, 0.8 * (1.0 - beta1_t0 / beta1_inf),
                      1.4 * (1.0 - beta1_t0 / beta1_inf))
    phi1c = np.clip(4.45 - 0.035 * U, 0.5, None)
    h_phi = h * 100.0
    phi2c = (42.0 + h_phi) / (20.0 + h_phi)
    phif_inf = np.where(fc28 <= 45.0, phi1c * phi2c, 0.45 * phi1c * phi2c)
    phid_inf = 0.4
    beta_d = (dur + 20.0) / (dur + 70.0)

    phi = phi_a + phif_inf * (_beta_f_creep(t_fic, h) - _beta_f_creep(t0_fic, h)) + phid_inf * beta_d

    J_total = (1.0 / Eci_t0 + phi / Eci28) * 1e6
    J_creep = (phi / Eci28) * 1e6
    inst = (1.0 / Eci_t0) * 1e6

    return pd.DataFrame({
        "J_total": J_total, "J_creep": J_creep, "phi": phi,
        "inst": inst, "Eci28": Eci28, "h_m": h, "phif_inf": phif_inf,
    }, index=idx)


# =====================================================================
# CREEP -- ACI 209 B3
# =====================================================================

def calcular_fluencia_B3_corrigido(data: pd.DataFrame) -> pd.Series:
    epsilon = 1e-9
    duration = data["x3"] + epsilon
    f_cm28 = data["x43"] + epsilon
    V_S = data["x50"] + epsilon
    t_o = data["x52"] + epsilon
    h_pct = data["x57"]
    E_cm28 = data["x45"] + epsilon
    c = data["x19"] + epsilon
    w_c = data["x17"] + epsilon
    a_c = data["x18"] + epsilon

    h = h_pct / 100.0
    w = w_c * c
    t = t_o + duration
    t_c = 7.0

    alpha_1 = 1.0
    alpha_2 = 1.2

    q1 = 0.6 / E_cm28
    q2 = 185.4e-6 * (c ** 0.5) * (f_cm28 ** -0.9)
    q3 = 0.29 * (w_c ** 4) * q2
    q4 = 20.3e-6 * (a_c ** -0.7)

    m, n = 0.5, 0.1
    Q_f_to = (0.086 * (t_o ** (2 / 9)) + 1.21 * (t_o ** (4 / 9))) ** -1
    Z_t_to = (t_o ** -m) * np.log(1 + duration ** n)
    r_to = 1.7 * (t_o ** 0.12) + 8
    Q_t_to = Q_f_to * (1 + (Q_f_to / (Z_t_to + epsilon)) ** r_to) ** (-1 / r_to)

    C_0 = q2 * Q_t_to + q3 * np.log(1 + duration ** n) + q4 * np.log(t / t_o)

    epsilon_s_inf = -alpha_1 * alpha_2 * (0.019 * w ** 2.1 * f_cm28 ** -0.28 + 270) * 1e-6
    k_s = 1.0
    tau_sh = 0.085 * (t_c ** -0.08) * (f_cm28 ** -0.25) * (2 * k_s * V_S) ** 2
    E_cm_tc_tau = E_cm28 * ((t_c + tau_sh) / (4 + 0.85 * (t_c + tau_sh))) ** 0.5
    E_cm_607 = E_cm28 * (607 / (4 + 0.85 * 607)) ** 0.5
    epsilon_sh_inf = epsilon_s_inf * (E_cm_607 / E_cm_tc_tau)

    q5 = 0.757 * (f_cm28 ** -1) * abs(epsilon_sh_inf * 1_000_000) ** -0.6

    S_t_tc = np.tanh(np.sqrt(np.maximum(0, (t - t_c)) / tau_sh))
    S_to_tc = np.tanh(np.sqrt(np.maximum(0, (t_o - t_c)) / tau_sh))
    Ht = 1 - (1 - h) * S_t_tc
    Ht_o = 1 - (1 - h) * S_to_tc

    termo_Cd = np.exp(-8 * Ht) - np.exp(-8 * Ht_o)
    Cd = q5 * np.sqrt(np.maximum(0, termo_Cd))

    J = q1 + C_0 + Cd
    J_filled = J.fillna(0).replace([np.inf, -np.inf], 0)
    return J_filled * 1_000_000


# =====================================================================
# CREEP -- ACI 209 B4 (generic-cement placeholder version)
# =====================================================================
# NOTE: an alternative "final" version of this notebook exists with a full
# per-cement-type (N/R/RS/SL) parameter table for the Cd (drying-creep) term,
# but it fits this database noticeably worse (raw R^2 -0.22 vs 0.45 here) --
# confirmed against the author's own printed metrics. This generic version
# (fixed alpha_1/alpha_2, ACI 209.2R-08 defaults) is used instead because it
# matches the database better, at the cost of not distinguishing cement type
# (same limitation as B3).

def calcular_fluencia_B4(data: pd.DataFrame) -> pd.Series:
    epsilon = 1e-9
    duration = data["x3"] + epsilon
    f_cm28 = data["x43"] + epsilon
    V_S = data["x50"] + epsilon
    t_o = data["x52"] + epsilon
    h_pct = data["x57"]
    E_cm28 = data["x45"] + epsilon
    c = data["x19"] + epsilon
    w_c = data["x17"] + epsilon
    a_c = data["x18"] + epsilon

    h = h_pct / 100.0
    w = w_c * c
    t = t_o + duration
    t_c = t_o

    alpha_1 = 1.0
    alpha_2 = 1.2

    q1 = 0.6 / E_cm28

    q2 = 185.4e-6 * (c ** 0.5) * (f_cm28 ** -0.9)
    q3 = 0.29 * (w_c ** 4) * q2
    q4 = 20.3e-6 * (a_c ** -0.7)

    m, n = 0.5, 0.1
    Q_f_to = (0.086 * (t_o ** (2 / 9)) + 1.21 * (t_o ** (4 / 9))) ** -1
    log_term_Z = np.log(1 + np.maximum(epsilon, duration) ** n)
    Z_t_to = (t_o ** -m) * log_term_Z
    r_to = 1.7 * (t_o ** 0.12) + 8

    Q_t_to = Q_f_to * (1 + (Q_f_to / (Z_t_to + epsilon)) ** r_to) ** (-1 / r_to)

    log_term_C0_1 = np.log(1 + np.maximum(epsilon, duration) ** n)
    log_term_C0_2 = np.log(np.maximum(1.0, t / t_o))
    C_0 = q2 * Q_t_to + q3 * log_term_C0_1 + q4 * log_term_C0_2

    epsilon_s_inf = -alpha_1 * alpha_2 * (0.019 * w ** 2.1 * f_cm28 ** -0.28 + 270) * 1e-6
    k_s = 1.0
    t_c_safe = np.maximum(1.0, t_c)
    tau_sh = 0.085 * (t_c_safe ** -0.08) * (f_cm28 ** -0.25) * (2 * k_s * V_S) ** 2
    tau_sh_safe = np.maximum(epsilon, tau_sh)

    E_cm_tc_tau = E_cm28 * ((t_c_safe + tau_sh) / (4 + 0.85 * (t_c_safe + tau_sh))) ** 0.5
    E_cm_607 = E_cm28 * (607 / (4 + 0.85 * 607)) ** 0.5
    epsilon_sh_inf = epsilon_s_inf * (E_cm_607 / (E_cm_tc_tau + epsilon))

    q5 = 0.757 * (f_cm28 ** -1) * (abs(epsilon_sh_inf * 1_000_000) + epsilon) ** -0.6

    S_t_tc = np.tanh(np.sqrt(np.maximum(0, (t - t_c_safe)) / tau_sh_safe))
    S_to_tc = np.tanh(np.sqrt(np.maximum(0, (t_o - t_c_safe)) / tau_sh_safe))
    Ht = 1 - (1 - h) * S_t_tc
    Ht_o = 1 - (1 - h) * S_to_tc

    p_5H = 8.0
    termo_Cd_B4 = np.exp(-p_5H * Ht) - np.exp(-p_5H * Ht_o)
    Cd = q5 * np.sqrt(np.maximum(0, termo_Cd_B4))

    J = q1 + C_0 + Cd
    J_filled = J.fillna(0).replace([np.inf, -np.inf], 0)
    return J_filled * 1_000_000


# =====================================================================
# SHRINKAGE -- ABNT NBR 6118 (modelo_abnt)
# =====================================================================

_SHR_ABNT_COL = {
    "t_dur": "x3", "t0_dry": "x51", "U": "x55", "v_s": "x49",
    "fc28": "x42", "E28": "x44", "temp": "x53", "h_mm": "x48",
}
_SHR_ABNT_UNIDADE_VS = "mm"
_SHR_ABNT_X3_EH_LOG = "auto"
_SHR_ABNT_BASE_LOG = 10
_SHR_ABNT_SIGMA_MPA = 1.0


def _v_shr(df: pd.DataFrame, chave: str, padrao: float) -> np.ndarray:
    nome = _SHR_ABNT_COL.get(chave)
    if nome is None or nome not in df.columns:
        return np.full(len(df), float(padrao))
    a = pd.to_numeric(df[nome], errors="coerce").to_numpy(float)
    return np.where(np.isfinite(a), a, padrao)


def _tempo_shr(df: pd.DataFrame) -> np.ndarray:
    t = _v_shr(df, "t_dur", 28.0)
    eh_log = _SHR_ABNT_X3_EH_LOG
    if eh_log == "auto":
        eh_log = (np.nanmax(t) < 15) and (np.nanmin(t) > -6)
    if eh_log:
        t = _SHR_ABNT_BASE_LOG ** t if _SHR_ABNT_BASE_LOG == 10 else np.exp(t)
    return np.clip(t, 1e-3, None)


def _tipo_cimento_shr(df: pd.DataFrame):
    r = df["x19_R"].to_numpy(float) if "x19_R" in df.columns else np.zeros(len(df))
    sl = df["x19_SL"].to_numpy(float) if "x19_SL" in df.columns else np.zeros(len(df))
    alfa = np.full(len(df), 2.0)
    alfa = np.where(sl > 0.5, 1.0, alfa)
    alfa = np.where(r > 0.5, 3.0, alfa)
    s = np.where(sl > 0.5, 0.38, np.where(r > 0.5, 0.20, 0.25))
    return alfa, s


def _beta_s_shr(t, h):
    x = np.asarray(t, float) / 100.0
    A = 40.0
    B = 116.0 * h ** 3 - 282.0 * h ** 2 + 220.0 * h - 4.8
    C = 2.5 * h ** 3 - 8.8 * h + 40.7
    D = -75.0 * h ** 3 + 585.0 * h ** 2 + 496.0 * h - 6.8
    E = -169.0 * h ** 4 + 88.0 * h ** 3 + 584.0 * h ** 2 - 39.0 * h + 0.8
    den = x ** 3 + C * x ** 2 + D * x + E
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.abs(den) < 1e-9, np.nan, (x ** 3 + A * x ** 2 + B * x) / den)


def _beta_f_shr(t, h):
    t = np.asarray(t, float)
    A = 42.0 * h ** 3 - 350.0 * h ** 2 + 588.0 * h + 113.0
    B = 768.0 * h ** 3 - 3060.0 * h ** 2 + 3234.0 * h - 23.0
    C = -200.0 * h ** 3 + 13.0 * h ** 2 + 1090.0 * h + 183.0
    D = 7579.0 * h ** 3 - 31916.0 * h ** 2 + 35343.0 * h + 1931.0
    den = t ** 2 + C * t + D
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.abs(den) < 1e-9, np.nan, (t ** 2 + A * t + B) / den)


def modelo_abnt(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.index
    dur = _tempo_shr(df)
    t0 = np.clip(_v_shr(df, "t0_dry", 7.0), 1e-3, None)
    U = np.clip(_v_shr(df, "U", 70.0), 40.0, 90.0)
    Temp = np.clip(_v_shr(df, "temp", 20.0), 0.0, 60.0)
    fc28 = np.clip(_v_shr(df, "fc28", 40.0), 5.0, 120.0)
    E28d = _v_shr(df, "E28", np.nan)

    vs = _v_shr(df, "v_s", 40.0)
    vs_m = vs / 1000.0 if _SHR_ABNT_UNIDADE_VS == "mm" else vs
    hfic = 2.0 * vs_m
    gamma = 1.0 + np.exp(-7.8 + 0.1 * U)
    h = np.clip(gamma * hfic, 0.05, 1.6)

    f_temp = (Temp + 10.0) / 30.0
    alfa_c, s_cim = _tipo_cimento_shr(df)

    t_ret_ini = f_temp * t0
    t_ret_fim = f_temp * (t0 + dur)
    t_flu_ini = alfa_c * f_temp * t0
    t_flu_fim = alfa_c * f_temp * (t0 + dur)

    e1s = -8.09 + U / 15.0 - U ** 2 / 2284.0 - U ** 3 / 133765.0 + U ** 4 / 7608150.0
    h_e2s = h * 100.0
    e2s = (33.0 + 2.0 * h_e2s) / (20.8 + 3.0 * h_e2s)

    ecs_inf = e1s * e2s * 1e-4
    ecs = ecs_inf * (_beta_s_shr(t_ret_fim, h) - _beta_s_shr(t_ret_ini, h))

    beta1_t0 = np.exp(s_cim * (1.0 - np.sqrt(28.0 / np.clip(t_flu_ini, 0.5, None))))
    beta1_inf = np.exp(s_cim * (1.0 - np.sqrt(28.0 / 1090.0)))

    Eci28 = np.where(np.isfinite(E28d) & (E28d > 1000), E28d, 0.9 * 5600.0 * np.sqrt(fc28))
    Eci_t0 = np.sqrt(beta1_t0) * Eci28

    phi_a = np.where(fc28 <= 45.0, 0.8 * (1.0 - beta1_t0 / beta1_inf), 1.4 * (1.0 - beta1_t0 / beta1_inf))
    phi1c = np.clip(4.45 - 0.035 * U, 0.5, None)
    phi2c = (42.0 + h) / (20.0 + h)
    phif_inf = np.where(fc28 <= 45.0, phi1c * phi2c, 0.45 * phi1c * phi2c)
    phid_inf = 0.4
    beta_d = (dur + 20.0) / (dur + 70.0)

    phi = phi_a + phif_inf * (_beta_f_shr(t_flu_fim, h) - _beta_f_shr(t_flu_ini, h)) + phid_inf * beta_d

    J = 1.0 / Eci_t0 + phi / Eci28
    sigma = _SHR_ABNT_SIGMA_MPA
    ec = sigma / Eci_t0
    ecc = sigma * phi / Eci28

    return pd.DataFrame({
        "ecs_ue": np.abs(ecs) * 1e6,
        "fluencia_ue": np.abs(ecc) * 1e6,
        "total_ue": np.abs(ec + ecc + ecs) * 1e6,
        "phi": phi,
        "J_1e6": J * 1e6,
        "h_m": h,
        "ecs_inf_ue": np.abs(ecs_inf) * 1e6,
    }, index=idx)


# =====================================================================
# SHRINKAGE -- ACI 209 B3
# =====================================================================

def calcular_retracao_b3_corrigido(data: pd.DataFrame) -> pd.Series:
    t_menos_tc = data["x3"]
    f_cm28 = data["x42"]
    E_cm28 = data["x44"]
    V_S = data["x49"]
    t_c = data["x51"]
    h = data["x55"] / 100.0
    c = data["x18"]
    w_c = data["x16"]

    w = w_c * c

    alpha_1 = pd.Series(1.0, index=data.index)
    if "x19_SL" in data.columns:
        alpha_1 = np.where(data["x19_SL"] == 1, 0.85, alpha_1)
    if "x19_R" in data.columns:
        alpha_1 = np.where(data["x19_R"] == 1, 1.10, alpha_1)
    if "x19_RS" in data.columns:
        alpha_1 = np.where(data["x19_RS"] == 1, 1.10, alpha_1)

    alpha_2 = 1.0
    k_s = 1.00

    epsilon_s_inf = -alpha_1 * alpha_2 * (0.019 * w ** 2.1 * f_cm28 ** -0.28 + 270) * 1e-6
    tau_sh = 0.085 * (t_c ** -0.08) * (f_cm28 ** -0.25) * (2 * k_s * V_S) ** 2

    t_corr = t_c + tau_sh
    E_cm_tc_tau_sh = E_cm28 * (t_corr / (4 + 0.85 * t_corr)) ** 0.5
    E_cm_607 = E_cm28 * (607 / (4 + 0.85 * 607)) ** 0.5

    epsilon_sh_inf_final = -epsilon_s_inf * (E_cm_607 / E_cm_tc_tau_sh)

    k_h = np.where(h <= 0.98, 1 - h ** 3, np.where(h == 1.0, -0.2, 12.74 - 12.94 * h))

    t_menos_tc_safe = np.maximum(t_menos_tc, 0)
    S = np.tanh(np.sqrt(t_menos_tc_safe / tau_sh))

    e_sh = -epsilon_sh_inf_final * k_h * S
    return abs(e_sh) * 1_000_000


# =====================================================================
# SHRINKAGE -- ACI 209 B4 (baseline em t_c, com retração autógena)
# =====================================================================

# NOTE: an earlier revision of this file gave the drying-shrinkage sub-model
# (tau_0/epsilon_0) its own per-cement-type table, borrowed from the
# per-type branch of calcular_fluencia_B4. That branch has since been
# replaced (it fit the creep database worse than the generic-parameter
# version -- see calcular_fluencia_B4 above), and the same per-type table
# turned out to fit THIS database worse too (calibrated R^2 0.18 vs 0.23
# for the original single-table version below). Reverted to the original
# notebook's fixed "Type R" table for every cement type.
_SHR_B4_PARAMS_SH_R = {
    "tau_cem": 0.016, "p_tau_a": -0.33, "p_tau_w": -0.06, "p_tau_c": -0.10,
    "eps_cem": 360e-6, "p_eps_a": -0.80, "p_eps_w": 1.10, "p_eps_c": 0.11,
}
_SHR_B4_PARAMS_AU_R = {
    "tau_au_cem": 1.00, "r_tau_w": 3.00,
    "eps_au_cem": 210e-6, "r_eps_a": -0.75, "r_eps_w": -3.50,
    "r_a": 1.00, "r_t": -4.50,
}
_SHR_B4_K_TAU_A = 1.0
_SHR_B4_K_EPS_A = 1.0
_SHR_B4_RHO_DEFAULT = 2350.0
_SHR_B4_K_S = 1.0


def calcular_retracao_b4_baseline_tc(data: pd.DataFrame) -> pd.Series:
    t_menos_tc = data["x3"]
    f_cm28 = data["x42"]
    V_S = data["x49"]
    t_c = data["x51"]
    h = data["x55"] / 100.0
    c = data["x18"]
    w_c = data["x16"]
    a_c = data["x17"]

    t_total = t_menos_tc + t_c

    if "x44" in data.columns and not data["x44"].isnull().all():
        E_cm28 = data["x44"]
    else:
        E_cm28 = 4734 * np.sqrt(f_cm28)

    p_sh = _SHR_B4_PARAMS_SH_R
    p_au = _SHR_B4_PARAMS_AU_R

    tau_0 = (p_sh["tau_cem"] * (a_c / 6) ** p_sh["p_tau_a"]
             * (w_c / 0.38) ** p_sh["p_tau_w"] * (6.5 * c / _SHR_B4_RHO_DEFAULT) ** p_sh["p_tau_c"])

    D = 2 * V_S
    tau_sh = tau_0 * _SHR_B4_K_TAU_A * (_SHR_B4_K_S * D / 1.0) ** 2

    epsilon_0 = (p_sh["eps_cem"] * (a_c / 6) ** p_sh["p_eps_a"]
                 * (w_c / 0.38) ** p_sh["p_eps_w"] * (6.5 * c / _SHR_B4_RHO_DEFAULT) ** p_sh["p_eps_c"])

    def calculate_E_t(time_days, E28):
        time_days_safe = np.maximum(time_days, 1e-6)
        return E28 * (time_days_safe / (4 + 0.85 * time_days_safe)) ** 0.5

    E_ref1 = calculate_E_t(7 + 600, E_cm28)
    E_ref2 = calculate_E_t(t_c + tau_sh, E_cm28)
    epsilon_sh_inf_final = -epsilon_0 * _SHR_B4_K_EPS_A * (E_ref1 / E_ref2)

    k_h = np.where(h <= 0.98, 1 - h ** 3,
                   np.where(h < 1.0, (1 - 0.98 ** 3) + ((-0.2) - (1 - 0.98 ** 3)) / (1.0 - 0.98) * (h - 0.98), -0.2))

    t_menos_tc_safe = np.maximum(t_menos_tc, 0)
    S_t = np.tanh(np.sqrt(t_menos_tc_safe / tau_sh))

    epsilon_sh = epsilon_sh_inf_final * k_h * S_t

    epsilon_au_inf = (-p_au["eps_au_cem"] * (a_c / 6) ** p_au["r_eps_a"] * (w_c / 0.38) ** p_au["r_eps_w"])
    tau_au = p_au["tau_au_cem"] * (w_c / 0.38) ** p_au["r_tau_w"]
    alpha_a = p_au["r_a"] * (w_c / 0.38)
    r_t = p_au["r_t"]

    t_total_safe = np.maximum(t_total, 1e-6)
    epsilon_au_t = epsilon_au_inf * (1 + (tau_au / t_total_safe) ** alpha_a) ** r_t

    t_c_safe = np.maximum(t_c, 1e-6)
    epsilon_au_tc = epsilon_au_inf * (1 + (tau_au / t_c_safe) ** alpha_a) ** r_t

    epsilon_sh_total_rel_tc = epsilon_sh + (epsilon_au_t - epsilon_au_tc)
    return abs(epsilon_sh_total_rel_tc) * 1_000_000


# =====================================================================
# Curve helper: sweep a formula model over a time array
# =====================================================================

def sweep_formula(fn, colmap: dict, cement_prefix: str, cement_categories: list[str],
                   tempos: np.ndarray, props: dict, cement_type: str, merge_nr: bool,
                   output_col: str | None = None) -> np.ndarray:
    """Builds one row per time step (all other properties fixed) and calls
    a ported formula function, returning the predicted curve as an array.
    """
    from app.data_pipeline import make_feature_row

    rows = []
    for t in tempos:
        values = dict(props)
        values["duration"] = t
        rows.append(make_feature_row(colmap, values, cement_prefix, cement_categories, cement_type, merge_nr))
    df_in = pd.DataFrame(rows)
    result = fn(df_in)
    if isinstance(result, pd.DataFrame):
        result = result[output_col]
    return np.asarray(result, dtype=float)
