"""Dashboard Certamen 1 - Costo de clasificación y umbral óptimo.

Reingreso hospitalario a 30 dias (Diabetes 130-US Hospitals).
Lee las probabilidades del test exportadas desde el notebook (probabilidades_test.csv):
  y_test             -> reingreso real (1 = reingresa en < 30 dias)
  proba              -> probabilidad del mejor modelo reentrenado en escala real (sin balanceo)
  proba_pesos_etapaB -> probabilidad del mismo modelo con pesos (Etapa B), se usa con umbral 0,5
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Configuración general
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Reingreso a 30 días · Umbral óptimo", page_icon="🏥", layout="wide")

AZUL = "#2a78d6"      # FP: gasto en programas innecesarios
NARANJA = "#eb6834"   # FN: ahorro perdido por reingresos no detectados
TINTA = "#1f2933"
GRIS = "#6b7280"
VERDE = "#1baf7a"


# Formato de números en español (miles con punto, decimales con coma)
def usd(x):
    return f"US$ {x:,.0f}".replace(",", ".")


def dec(x, d=3):
    return f"{x:.{d}f}".replace(".", ",")


def num(x):
    return f"{x:,}".replace(",", ".")

# Supuestos base (sección 6 del notebook)
COSTO_PROGRAMA = 413      # c: programa de transición por paciente (WSIPP, US$ 2014)
COSTO_REINGRESO = 14400   # R: costo promedio de un reingreso a 30 dias (HCUP SB #248, 2016)
EFECTIVIDAD = 0.18        # e: reducción relativa del riesgo (Leppin et al., 2014)
FP_BASE = COSTO_PROGRAMA
FN_BASE = round(EFECTIVIDAD * COSTO_REINGRESO - COSTO_PROGRAMA)   # 2.179

ESCENARIOS = {
    "Base (efectividad 18%)": FN_BASE,
    "Programa poco efectivo (FN/FP = 2)": 2 * FP_BASE,
    "Programa conservador (FN/FP = 3)": 3 * FP_BASE,
    "Programa efectivo (FN/FP = 8)": 8 * FP_BASE,
    "Programa muy efectivo (FN/FP = 12)": 12 * FP_BASE,
}

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.6rem; padding-bottom: 2rem;}
      div[data-testid="stMetricValue"] {font-size: 1.55rem;}
      .resumen {background:#eef5fc; border-left:4px solid #2a78d6; padding:.8rem 1rem;
                border-radius:6px; color:#1f2933; font-size:1.02rem; margin:.4rem 0 1rem 0;}
      .mc {display:grid; grid-template-columns: 78px 1fr 1fr; gap:6px; font-family:inherit;}
      .mc div {border-radius:8px; padding:.65rem .5rem; text-align:center;}
      .mc .cab {background:transparent; color:#6b7280; font-size:.85rem; font-weight:600;}
      .mc .fila {background:transparent; color:#6b7280; font-size:.85rem; font-weight:600;
                 display:flex; align-items:center; justify-content:flex-end; text-align:right;}
      .mc .ok {background:#f1f3f5; color:#1f2933;}
      .mc .fp {background:#dbe9fa; color:#1f2933; border:2px solid #2a78d6;}
      .mc .fn {background:#fde3d8; color:#1f2933; border:2px solid #eb6834;}
      .mc .tag {font-size:.8rem; font-weight:700; letter-spacing:.04em; color:#52514e;}
      .mc .n {font-size:1.45rem; font-weight:700;}
      .mc .usd {font-size:1.05rem; font-weight:700;}
      .mc .det {font-size:.8rem; color:#52514e;}
      .nota {color:#6b7280; font-size:.85rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------
@st.cache_data
def cargar_datos():
    ruta = Path(__file__).parent / "probabilidades_test.csv"
    datos = pd.read_csv(ruta)
    return datos["y_test"].to_numpy().astype(int), datos["proba"].to_numpy(), datos["proba_pesos_etapaB"].to_numpy()


y, p, p_pesos = cargar_datos()
P = int((y == 1).sum())   # reingresos reales en test
N = int((y == 0).sum())   # no reingresos en test
n_total = len(y)


def matriz(umbral, prob=p):
    """TN, FP, FN, TP al clasificar como reingreso si prob >= umbral."""
    pred = prob >= umbral
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = P - tp
    tn = N - fp
    return tn, fp, fn, tp


# Conteos de FP y FN para una grilla de umbrales (se calcula una sola vez)
@st.cache_data
def grilla_umbrales():
    u = np.round(np.arange(0.005, 0.996, 0.005), 3)
    neg = np.sort(p[y == 0])
    pos = np.sort(p[y == 1])
    fp_u = N - np.searchsorted(neg, u, side="left")   # negativos con p >= u
    fn_u = np.searchsorted(pos, u, side="left")        # positivos con p < u
    return u, fp_u, fn_u


@st.cache_data
def curva_roc():
    orden = np.argsort(-p, kind="mergesort")
    y_ord = y[orden]
    tpr = np.concatenate([[0], np.cumsum(y_ord == 1) / P])
    fpr = np.concatenate([[0], np.cumsum(y_ord == 0) / N])
    auc = float(np.trapezoid(tpr, fpr)) if hasattr(np, "trapezoid") else float(np.trapz(tpr, fpr))
    return fpr, tpr, auc


u_grid, fp_grid, fn_grid = grilla_umbrales()
fpr_roc, tpr_roc, auc = curva_roc()


# ---------------------------------------------------------------------------
# Controles (barra lateral)
# ---------------------------------------------------------------------------
def aplicar_escenario():
    st.session_state["costo_fp"] = FP_BASE
    st.session_state["costo_fn"] = int(ESCENARIOS[st.session_state["escenario"]])


if "costo_fp" not in st.session_state:
    st.session_state["costo_fp"] = FP_BASE
    st.session_state["costo_fn"] = FN_BASE

with st.sidebar:
    st.header("Controles")
    st.selectbox("Escenario de negocio", list(ESCENARIOS), key="escenario", on_change=aplicar_escenario,
                 help="Carga costos predefinidos. Después puede ajustarlos a mano.")

    st.subheader("Costos de cada error")
    costo_fp = st.slider("Costo de un falso positivo (US$)", 50, 2000, step=1, key="costo_fp",
                         help="Se incluye en el programa a un paciente que no iba a reingresar: se gasta el programa.")
    costo_fn = st.slider("Costo de un falso negativo (US$)", 100, 10000, step=1, key="costo_fn",
                         help="No se incluye a un paciente que sí reingresa: se pierde el ahorro neto que el programa habría generado.")

    p_star = costo_fp / (costo_fp + costo_fn)
    st.markdown(f"**Umbral óptimo** p\\* = FP / (FP + FN) = **{dec(p_star)}**")

    st.subheader("Umbral de decisión")
    usar_optimo = st.toggle("Usar el umbral óptimo p*", value=True)
    umbral_manual = st.slider("Umbral manual", 0.01, 0.99, 0.50, 0.01, disabled=usar_optimo,
                              help="Desactive 'Usar el umbral óptimo' para moverlo y compararlo con p*.")
    umbral = p_star if usar_optimo else umbral_manual

    with st.expander("¿De dónde salen los costos base?"):
        st.markdown(
            f"""
            - **Programa de transición del alta:** US\\${COSTO_PROGRAMA} por paciente (WSIPP).
            - **Costo de un reingreso:** US\\${COSTO_REINGRESO:,} (AHRQ, HCUP #248, 2016).
            - **Efectividad del programa:** reduce el riesgo de reingreso un 18% (Leppin et al., 2014).

            **FP** = costo del programa = US\\${FP_BASE}
            **FN** = reingreso evitado − programa = 0,18 × 14.400 − 413 = US\\${FN_BASE:,}

            *Supuestos declarados; costos de EE.UU.*
            """
        )


# ---------------------------------------------------------------------------
# Cálculos para el umbral elegido y las referencias
# ---------------------------------------------------------------------------
def costo(fp, fn):
    return costo_fp * fp + costo_fn * fn


tn, fp, fn, tp = matriz(umbral)
costo_elegido = costo(fp, fn)
tn5, fp5, fn5, tp5 = matriz(0.5)
costo_05 = costo(fp5, fn5)
tnw, fpw, fnw, tpw = matriz(0.5, p_pesos)
costo_pesos = costo(fpw, fnw)
costo_nada = costo(0, P)
costo_todos = costo(N, 0)

ahorro = costo_05 - costo_elegido
ahorro_pct = 100 * ahorro / costo_05 if costo_05 else 0.0
recall = tp / P
precision = tp / (tp + fp) if (tp + fp) else 0.0
pct_intervenidos = 100 * (tp + fp) / n_total

curva_costo = costo_fp * fp_grid + costo_fn * fn_grid
u_min = float(u_grid[np.argmin(curva_costo)])


# ---------------------------------------------------------------------------
# Encabezado e indicadores
# ---------------------------------------------------------------------------
st.title("🏥 Reingreso hospitalario a 30 días")
st.markdown(
    "<span class='nota'>¿A qué pacientes con diabetes incluir en el programa de transición del alta? "
    f"Random Forest en escala real · {num(n_total)} altas del conjunto de test · AUC = {dec(auc)}</span>",
    unsafe_allow_html=True,
)

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Umbral en uso", f"{dec(umbral)}",
          "óptimo p*" if usar_optimo else f"{'+' if umbral >= p_star else ''}{dec(umbral - p_star)} vs p*", delta_color="off")
k2.metric("Costo total", usd(costo_elegido))
k3.metric("Ahorro vs umbral 0,5", usd(ahorro), f"{dec(ahorro_pct, 1)}%")
k4.metric("Recall", f"{dec(100 * recall, 1)}%")
k5.metric("Precisión", f"{dec(100 * precision, 1)}%")

st.markdown(
    f"<div class='resumen'>Con un umbral de <b>{dec(umbral)}</b>, el hospital incluye en el programa a "
    f"<b>{dec(pct_intervenidos, 0)} de cada 100</b> pacientes, detecta <b>{dec(100 * recall, 0)}%</b> de los reingresos "
    f"y {'ahorra' if ahorro >= 0 else 'pierde'} <b>{usd(abs(ahorro))}</b> frente a usar el umbral 0,5 "
    f"(≈ {usd(abs(ahorro) / n_total * 1000)} por cada 1.000 altas)."
    + ("" if usar_optimo else
       f" Con el umbral óptimo p* = <b>{dec(p_star)}</b> el costo sería <b>{usd(costo(*matriz(p_star)[1:3]))}</b>, "
       f"es decir, <b>{usd(costo_elegido - costo(*matriz(p_star)[1:3]))}</b> menos que con el umbral manual.")
    + "</div>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Fila 1: curva de costo y matriz de confusión
# ---------------------------------------------------------------------------
c1, c2 = st.columns([1.3, 1])

with c1:
    st.subheader("Costo total vs. umbral")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=u_grid, y=curva_costo, mode="lines", line=dict(color=AZUL, width=2.5), name="Costo total",
        customdata=np.stack([fp_grid, fn_grid], axis=1),
        hovertemplate="Umbral %{x:.3f}<br>Costo US$ %{y:,.0f}<br>FP %{customdata[0]:,} · FN %{customdata[1]:,}<extra></extra>",
    ))
    fig.add_hline(y=costo_nada, line=dict(color=GRIS, width=1, dash="dot"),
                  annotation_text="No intervenir a nadie", annotation_position="top right",
                  annotation_font_color=GRIS)
    lineas = [(0.5, "Umbral 0,5", GRIS, "dot"), (p_star, f"Óptimo p* = {dec(p_star)}", TINTA, "dash")]
    for x, texto, color, estilo in lineas:
        fig.add_vline(x=x, line=dict(color=color, width=1.5, dash=estilo))
    fig.add_trace(go.Scatter(x=[0.5], y=[costo_05], mode="markers", name="Umbral 0,5",
                             marker=dict(size=11, color="white", line=dict(color=GRIS, width=2.5)),
                             hovertemplate="Umbral 0,5<br>Costo US$ %{y:,.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[p_star], y=[costo(*matriz(p_star)[1:3])], mode="markers", name=f"Óptimo p* = {dec(p_star)}",
                             marker=dict(size=12, color=TINTA, symbol="diamond"),
                             hovertemplate="p* = %{x:.3f}<br>Costo US$ %{y:,.0f}<extra></extra>"))
    if not usar_optimo:
        fig.add_trace(go.Scatter(x=[umbral], y=[costo_elegido], mode="markers", name=f"Umbral manual = {dec(umbral, 2)}",
                                 marker=dict(size=13, color=NARANJA, line=dict(color="white", width=2)),
                                 hovertemplate="Umbral manual %{x:.2f}<br>Costo US$ %{y:,.0f}<extra></extra>"))
    fig.update_layout(
        height=380, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="white", paper_bgcolor="white", separators=",.",
        xaxis=dict(title="Umbral de decisión", range=[0, 1], gridcolor="#eef0f2", zeroline=False),
        yaxis=dict(title="Costo total en test (US$)", gridcolor="#eef0f2", zeroline=False, tickformat=",.0f"),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0), hovermode="closest",
    )
    st.plotly_chart(fig, width="stretch")
    st.markdown(f"<span class='nota'>Mínimo de la curva en test: umbral ≈ {dec(u_min)}. "
                "El óptimo p* se calcula solo con los costos, sin mirar el test.</span>", unsafe_allow_html=True)

with c2:
    st.subheader("Matriz de confusión")
    st.markdown(
        f"""
        <div class="mc">
          <div class="cab"></div><div class="cab">Predice: no intervenir</div><div class="cab">Predice: intervenir</div>
          <div class="fila">Real:<br>no reingresa</div>
          <div class="ok"><div class="tag">VERDADERO<br>NEGATIVO</div><div class="n">{num(tn)}</div><div class="usd">US$ 0</div></div>
          <div class="fp"><div class="tag">FALSO<br>POSITIVO</div><div class="n">{num(fp)}</div>
               <div class="usd">{usd(costo_fp * fp)}</div><div class="det">{num(fp)} × US$ {num(costo_fp)}</div></div>
          <div class="fila">Real:<br>reingresa</div>
          <div class="fn"><div class="tag">FALSO<br>NEGATIVO</div><div class="n">{num(fn)}</div>
               <div class="usd">{usd(costo_fn * fn)}</div><div class="det">{num(fn)} × US$ {num(costo_fn)}</div></div>
          <div class="ok"><div class="tag">VERDADERO<br>POSITIVO</div><div class="n">{num(tp)}</div><div class="usd">US$ 0</div></div>
        </div>
        <p class="nota" style="margin-top:.7rem">Umbral = {dec(umbral)}. Costo total = {usd(costo_elegido)}.
        FP: programa gastado en quien no reingresa. FN: reingreso no detectado.</p>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Fila 2: desglose del costo y ROC con iso-costo
# ---------------------------------------------------------------------------
c3, c4 = st.columns([1.45, 1])

with c3:
    st.subheader("¿Cuánto cuesta cada estrategia?")
    etiqueta_elegido = "Umbral óptimo p*" if usar_optimo else f"Umbral manual ({dec(umbral, 2)})"
    estrategias = pd.DataFrame({
        "Estrategia": ["Intervenir a todos", "No intervenir a nadie", "Umbral 0,5",
                       "Modelo con pesos (Etapa B), 0,5", etiqueta_elegido],
        "FP": [N, 0, fp5, fpw, fp],
        "FN": [0, P, fn5, fnw, fn],
    })
    estrategias["Gasto FP"] = estrategias["FP"] * costo_fp
    estrategias["Pérdida FN"] = estrategias["FN"] * costo_fn
    estrategias["Costo total"] = estrategias["Gasto FP"] + estrategias["Pérdida FN"]
    estrategias = estrategias.sort_values("Costo total", ascending=False)

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(y=estrategias["Estrategia"], x=estrategias["Gasto FP"], orientation="h", name="Gasto en FP",
                          marker=dict(color=AZUL, line=dict(color="white", width=2)),
                          hovertemplate="%{y}<br>Gasto en FP: US$ %{x:,.0f}<extra></extra>"))
    fig2.add_trace(go.Bar(y=estrategias["Estrategia"], x=estrategias["Pérdida FN"], orientation="h", name="Pérdida por FN",
                          marker=dict(color=NARANJA, line=dict(color="white", width=2)),
                          text=[usd(v) for v in estrategias["Costo total"]], textposition="outside",
                          textfont=dict(color=TINTA), cliponaxis=False,
                          hovertemplate="%{y}<br>Pérdida por FN: US$ %{x:,.0f}<extra></extra>"))
    fig2.update_layout(
        barmode="stack", height=360, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="white", paper_bgcolor="white", separators=",.",
        xaxis=dict(title="Costo total en test (US$)", gridcolor="#eef0f2", tickformat=",.0f",
                   range=[0, estrategias["Costo total"].max() * 1.38]),
        yaxis=dict(title=""), legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, traceorder="normal"), bargap=0.35,
    )
    st.plotly_chart(fig2, width="stretch")

with c4:
    st.subheader("Curva ROC y recta de iso-costo")
    m = (costo_fp * N) / (costo_fn * P)
    costo_roc = costo_fn * P * (1 - tpr_roc) + costo_fp * N * fpr_roc
    k = int(np.argmin(costo_roc))
    xs = np.linspace(0, 1, 50)
    ys = tpr_roc[k] + m * (xs - fpr_roc[k])
    fpr_t, tpr_t = fp / N, tp / P
    paso = max(1, len(fpr_roc) // 1500)   # submuestreo para que el grafico sea liviano
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=fpr_roc[::paso], y=tpr_roc[::paso], mode="lines", name=f"ROC (AUC = {dec(auc)})",
                              line=dict(color=AZUL, width=2.5),
                              hovertemplate="FPR %{x:.3f}<br>TPR %{y:.3f}<extra></extra>"))
    fig3.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Azar", line=dict(color=GRIS, width=1, dash="dot"),
                              hoverinfo="skip"))
    fig3.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=f"Iso-costo (pendiente {dec(m, 2)})",
                              line=dict(color=TINTA, width=1.5, dash="dash"), hoverinfo="skip"))
    fig3.add_trace(go.Scatter(x=[fpr_roc[k]], y=[tpr_roc[k]], mode="markers", name="Mínimo costo en test",
                              marker=dict(size=18, color="rgba(0,0,0,0)", line=dict(color=TINTA, width=2)),
                              hovertemplate="Mínimo costo<br>FPR %{x:.3f} · TPR %{y:.3f}<extra></extra>"))
    fig3.add_trace(go.Scatter(x=[fpr_t], y=[tpr_t], mode="markers", name="Umbral en uso",
                              marker=dict(size=13, color=NARANJA if not usar_optimo else TINTA,
                                          symbol="circle" if not usar_optimo else "diamond",
                                          line=dict(color="white", width=2)),
                              hovertemplate="Umbral en uso<br>FPR %{x:.3f} · TPR %{y:.3f}<extra></extra>"))
    fig3.update_layout(
        height=360, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="white", paper_bgcolor="white", separators=",.",
        xaxis=dict(title="FPR (1 − especificidad)", range=[0, 1], gridcolor="#eef0f2", zeroline=False),
        yaxis=dict(title="TPR (recall)", range=[0, 1.02], gridcolor="#eef0f2", zeroline=False),
        legend=dict(orientation="h", yanchor="top", y=-0.22, x=0, font=dict(size=11)),
    )
    st.plotly_chart(fig3, width="stretch")
    st.markdown("<span class='nota'>Los puntos de igual costo forman rectas de pendiente "
                "m = FP·N / (FN·P). El óptimo es donde la recta toca la curva ROC.</span>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Tabla resumen
# ---------------------------------------------------------------------------
with st.expander("Ver tabla de estrategias"):
    tabla = estrategias.sort_values("Costo total").copy()
    tabla["Ahorro vs 0,5 (US$)"] = costo_05 - tabla["Costo total"]
    for col in ["Gasto FP", "Pérdida FN", "Costo total", "Ahorro vs 0,5 (US$)"]:
        tabla[col] = tabla[col].map(usd)
    st.dataframe(tabla.set_index("Estrategia"), width="stretch")
    st.markdown(
        "<span class='nota'>Fuente: notebook Certamen 1 (Diabetes 130-US Hospitals, UCI). "
        "Probabilidades del Random Forest reentrenado con la proporción real (sin SMOTENC ni pesos). "
        "El modelo con pesos se evalúa con umbral 0,5.</span>",
        unsafe_allow_html=True,
    )
