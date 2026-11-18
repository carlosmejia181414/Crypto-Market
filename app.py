"""
Analizador Cripto: tu propio panel de análisis y alertas.
Ejecutar:  streamlit run app.py
"""
from __future__ import annotations

import json
from dataclasses import asdict, fields
from datetime import datetime, timezone

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import modelos as mdl
import nucleo as n

st.set_page_config(page_title="Analizador Cripto", page_icon="📈", layout="wide")

VERDE, ROJO, GRIS, AMARILLO = "#16a34a", "#dc2626", "#6b7280", "#ca8a04"


# =============================================================================
# DATOS CON CACHÉ
# =============================================================================
@st.cache_data(ttl=60, show_spinner=False)
def cargar_velas(exchange, simbolo, tf, velas):
    return n.descargar_velas(exchange, simbolo, tf, velas)


@st.cache_data(ttl=1800, show_spinner=False, max_entries=20)
def cargar_modelos(exchange, simbolo, tf, velas, params_json, horizonte, clave_vela):
    """Se recalcula solo cuando cierra una vela nueva (clave_vela) o cambian los parámetros."""
    df, _ = cargar_velas(exchange, simbolo, tf, velas)
    p_ = n.Params(**json.loads(params_json))
    mdl.H_ML[tf] = horizonte
    return mdl.evaluar(n.analizar(df, p_, tf), p_, tf)


@st.cache_data(ttl=1, show_spinner=False)
def cargar_precio(exchange, sym):
    return n.precio_actual(exchange, sym)


@st.cache_data(ttl=900, show_spinner=False)
def cargar_noticias(base):
    return n.obtener_noticias(base)


@st.cache_data(ttl=300, show_spinner=False)
def cargar_escaner(exchange, lista, tf, velas, params_json):
    p = n.Params(**json.loads(params_json))
    filas = []
    for s in lista:
        try:
            df, sym = n.descargar_velas(exchange, s, tf, velas)
            r = n.resumen(n.analizar(df, p, tf), tf)
            filas.append({"Cripto": sym, **r})
        except Exception as e:
            filas.append({"Cripto": s, "error": str(e)[:80]})
    return filas


def badge(texto, color):
    return (f"<span style='background:{color};color:white;padding:3px 10px;border-radius:12px;"
            f"font-size:0.85rem;font-weight:600'>{texto}</span>")


def color_estado(txt):
    return VERDE if txt in ("Alcista", "Alto", "COMPRA") else ROJO if txt in ("Bajista", "VENTA") else GRIS


def fmt(x):
    return n._fmt(x)


# =============================================================================
# BARRA LATERAL
# =============================================================================
cfg = n.cargar_config()
zona = cfg.get("zona_horaria", "America/Toronto")

st.sidebar.title("📈 Analizador Cripto")
lista_ex = list(n.EXCHANGES)
exchange = st.sidebar.selectbox("Fuente de datos", lista_ex, index=lista_ex.index(cfg["exchange"]) if cfg["exchange"] in lista_ex else 0,
                                format_func=n.EXCHANGES.get,
                                help="Solo se usan datos públicos de precios. Si una fuente falla, prueba otra.")
opciones = cfg["watchlist"] + ["Otra…"]
eleccion = st.sidebar.selectbox("Criptomoneda", opciones)
simbolo = st.sidebar.text_input("Par (ej. ADA/USDT)", "ADA/USDT").upper() if eleccion == "Otra…" else eleccion
tf = st.sidebar.selectbox("Temporalidad", ["1h", "4h", "1d"], index=["1h", "4h", "1d"].index(cfg["timeframe"]))
velas = st.sidebar.slider("Velas a cargar", 300, 2000, int(cfg["velas"]), step=100)
mostrar_bb = st.sidebar.checkbox("Mostrar Bandas de Bollinger", False)
auto = st.sidebar.checkbox("Actualizar en vivo", True, help="Refresca precio y gráfico automáticamente.")
seg_precio = st.sidebar.select_slider("Precio en vivo cada (segundos)", [2, 3, 5, 10], 3)
seg_graf = st.sidebar.select_slider("Gráfico cada (segundos)", [15, 30, 60, 120], 30,
                                    help="Las señales solo cambian al cierre de la vela; el gráfico no necesita ir más rápido.")
velas_graf = st.sidebar.slider("Velas visibles en el gráfico", 150, 1500, 400, step=50,
                               help="Menos velas = gráfico más rápido. El análisis siempre usa todas las velas cargadas.")
st.sidebar.markdown("**Modelos matemáticos**")
mostrar_cono = st.sidebar.checkbox("Cono de pronóstico (GARCH + Monte Carlo)", True)
mostrar_kalman = st.sidebar.checkbox("Tendencia Kalman", True)
opciones_h = {1: "1 vela", 3: "3 velas", 6: "6 velas", 12: "12 velas"}
horizonte = st.sidebar.selectbox("Horizonte del modelo ML", list(opciones_h), format_func=opciones_h.get,
                                 index=list(opciones_h).index(int(cfg.get("horizonte_ml", 6))) if int(cfg.get("horizonte_ml", 6)) in opciones_h else 2,
                                 help="Cuántas velas hacia adelante intenta pronosticar el modelo de machine learning.")

pd_ = cfg["params"]
with st.sidebar.expander("⚙️ Parámetros de la estrategia"):
    vals = {}
    vals["ema_rapida"] = st.number_input("EMA rápida", 5, 400, int(pd_["ema_rapida"]))
    vals["ema_lenta"] = st.number_input("EMA lenta", 20, 400, int(pd_["ema_lenta"]))
    vals["usar_htf"] = st.checkbox("Confirmar con tendencia diaria", bool(pd_["usar_htf"]))
    vals["ema_htf"] = st.number_input("EMA diaria", 5, 300, int(pd_["ema_htf"]))
    vals["rsi_len"] = st.number_input("RSI longitud", 2, 50, int(pd_["rsi_len"]))
    vals["rsi_compra_max"] = st.number_input("RSI máx. para comprar", 50.0, 95.0, float(pd_["rsi_compra_max"]))
    vals["rsi_venta_min"] = st.number_input("RSI mín. para vender", 5.0, 50.0, float(pd_["rsi_venta_min"]))
    vals["macd_rapida"] = st.number_input("MACD rápida", 2, 50, int(pd_["macd_rapida"]))
    vals["macd_lenta"] = st.number_input("MACD lenta", 5, 100, int(pd_["macd_lenta"]))
    vals["macd_senal"] = st.number_input("MACD señal", 2, 50, int(pd_["macd_senal"]))
    vals["usar_vol"] = st.checkbox("Exigir volumen sobre el promedio", bool(pd_["usar_vol"]))
    vals["vol_len"] = st.number_input("Promedio de volumen", 5, 100, int(pd_["vol_len"]))
    vals["atr_len"] = st.number_input("ATR longitud", 2, 50, int(pd_["atr_len"]))
    vals["sl_mult"] = st.number_input("Stop loss (x ATR)", 0.5, 10.0, float(pd_["sl_mult"]), step=0.1)
    vals["tp_mult"] = st.number_input("Take profit (x ATR)", 0.5, 20.0, float(pd_["tp_mult"]), step=0.1)
    vals["bb_len"] = st.number_input("Bollinger longitud", 5, 100, int(pd_["bb_len"]))
    vals["bb_mult"] = st.number_input("Bollinger desviación", 0.5, 5.0, float(pd_["bb_mult"]), step=0.1)
    vals["min_puntos"] = st.slider("Puntos mínimos para señal (de 5)", 1, 5, int(pd_["min_puntos"]))
    vals["usar_preaviso"] = st.checkbox("Preavisos (señal en formación)", bool(pd_["usar_preaviso"]))
    if st.button("💾 Guardar como predeterminado", width="stretch"):
        cfg.update({"exchange": exchange, "timeframe": tf, "velas": velas, "params": vals, "horizonte_ml": horizonte})
        n.guardar_config(cfg)
        st.success("Guardado. El monitor de alertas usará estos valores.")
p = n.Params(**vals)

st.sidebar.caption("Herramienta de apoyo para tus análisis. No es asesoría financiera.")

tabs = st.tabs(["📊 Gráfico", "🧮 Modelos", "🔎 Escáner", "🧪 Backtest", "📰 Noticias", "🔔 Alertas"])


def modelos_actuales():
    df, _ = cargar_velas(exchange, simbolo, tf, velas)
    clave = int(df[df["cerrada"]]["ts"].iloc[-1])
    with st.spinner("Calculando modelos matemáticos (solo al cierre de cada vela)…"):
        return cargar_modelos(exchange, simbolo, tf, velas, json.dumps(asdict(p)), horizonte, clave)


# =============================================================================
# GRÁFICO
# =============================================================================
def figura(d: pd.DataFrame, sym: str, ev: dict | None = None) -> go.Figure:
    d = d.tail(velas_graf).copy()
    d.index = d.index.tz_convert(zona)
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.025,
                        row_heights=[0.58, 0.12, 0.15, 0.15])
    fig.add_trace(go.Candlestick(x=d.index, open=d["open"], high=d["high"], low=d["low"], close=d["close"],
                                 name=sym, increasing_line_color="#26a69a", decreasing_line_color="#ef5350"), 1, 1)
    fig.add_trace(go.Scatter(x=d.index, y=d["ema_r"], name=f"EMA {p.ema_rapida}", line=dict(color="orange", width=1.3)), 1, 1)
    fig.add_trace(go.Scatter(x=d.index, y=d["ema_l"], name=f"EMA {p.ema_lenta}", line=dict(color="#3b82f6", width=2)), 1, 1)
    if mostrar_bb:
        fig.add_trace(go.Scatter(x=d.index, y=d["bb_sup"], name="Bollinger", line=dict(color="gray", width=1, dash="dot")), 1, 1)
        fig.add_trace(go.Scatter(x=d.index, y=d["bb_inf"], name="Bollinger inf.", showlegend=False,
                                 line=dict(color="gray", width=1, dash="dot")), 1, 1)

    extra_x = 8
    if ev and mostrar_kalman and ev.get("kalman", {}).get("ok"):
        kn = ev["kalman"]["nivel"]
        kn = kn[kn.index >= d.index[0].tz_convert("UTC")].copy()
        kn.index = kn.index.tz_convert(zona)
        fig.add_trace(go.Scatter(x=kn.index, y=kn.values, name="Tendencia Kalman", line=dict(color="#e879f9", width=1.6, dash="dot")), 1, 1)
    if ev and mostrar_cono and ev.get("garch", {}).get("ok"):
        cono = ev["garch"]["cono"].copy()
        cono.index = cono.index.tz_convert(zona)
        x = list(cono.index) + list(cono.index[::-1])
        fig.add_trace(go.Scatter(x=x, y=list(cono["p95"]) + list(cono["p5"][::-1]), fill="toself", name="Rango 90%",
                                 fillcolor="rgba(56,189,248,0.10)", line=dict(width=0), hoverinfo="skip"), 1, 1)
        fig.add_trace(go.Scatter(x=x, y=list(cono["p75"]) + list(cono["p25"][::-1]), fill="toself", name="Rango 50%",
                                 fillcolor="rgba(56,189,248,0.22)", line=dict(width=0), hoverinfo="skip"), 1, 1)
        fig.add_trace(go.Scatter(x=cono.index, y=cono["p50"], name="Mediana simulada", line=dict(color="#38bdf8", width=1, dash="dash")), 1, 1)
        extra_x = len(cono) + 2
    tf_td = pd.Timedelta(milliseconds=n.TF_MS[tf])
    for tipo, simb, y, col, txt in [(1, "triangle-up", d["low"] * 0.992, VERDE, "COMPRA"),
                                    (-1, "triangle-down", d["high"] * 1.008, ROJO, "VENTA")]:
        s = d[d["senal"] == tipo]
        if len(s):
            hover = [f"<b>{txt}</b><br>Entrada: {fmt(r.close)}<br>SL: {fmt(r.sl)}<br>TP: {fmt(r.tp)}" for r in s.itertuples()]
            fig.add_trace(go.Scatter(x=s.index, y=y[s.index], mode="markers+text", name=txt, text=[txt] * len(s),
                                     textposition="bottom center" if tipo == 1 else "top center",
                                     textfont=dict(color=col, size=10), hovertext=hover, hoverinfo="text",
                                     marker=dict(symbol=simb, size=14, color=col, line=dict(color="white", width=1))), 1, 1)
        for r in s.tail(40).itertuples():
            for nivel, c in ((r.sl, ROJO), (r.tp, VERDE)):
                fig.add_shape(type="line", x0=r.Index, x1=r.Index + tf_td * 15, y0=nivel, y1=nivel,
                              line=dict(color=c, width=1, dash="dash"), row=1, col=1)
    pre = d[d["preaviso"] != 0]
    if len(pre):
        fig.add_trace(go.Scatter(x=pre.index, y=pre["close"], mode="markers", name="Preaviso",
                                 hovertext=["Preaviso " + ("compra" if v == 1 else "venta") for v in pre["preaviso"]],
                                 hoverinfo="text+x", marker=dict(color="yellow", size=7, line=dict(color="black", width=1))), 1, 1)

    colores = ["#26a69a" if c >= o else "#ef5350" for c, o in zip(d["close"], d["open"])]
    fig.add_trace(go.Bar(x=d.index, y=d["volume"], name="Volumen", marker_color=colores, showlegend=False), 2, 1)
    fig.add_trace(go.Scatter(x=d.index, y=d["vol_prom"], name="Vol. promedio", line=dict(color="white", width=1), showlegend=False), 2, 1)
    fig.add_trace(go.Scatter(x=d.index, y=d["rsi"], name="RSI", line=dict(color="#a855f7", width=1.3)), 3, 1)
    fig.add_hline(y=70, line=dict(color="gray", dash="dot", width=1), row=3, col=1)
    fig.add_hline(y=30, line=dict(color="gray", dash="dot", width=1), row=3, col=1)
    fig.add_trace(go.Bar(x=d.index, y=d["macd_hist"], name="MACD hist.", showlegend=False,
                         marker_color=["#26a69a" if v >= 0 else "#ef5350" for v in d["macd_hist"].fillna(0)]), 4, 1)
    fig.add_trace(go.Scatter(x=d.index, y=d["macd"], name="MACD", line=dict(color="#3b82f6", width=1.2)), 4, 1)
    fig.add_trace(go.Scatter(x=d.index, y=d["macd_sig"], name="Señal MACD", line=dict(color="orange", width=1.2)), 4, 1)

    vis = d.tail(150)
    lo, hi = vis["low"].min(), vis["high"].max()
    if ev and mostrar_cono and ev.get("garch", {}).get("ok"):
        lo, hi = min(lo, ev["garch"]["cono"]["p5"].min()), max(hi, ev["garch"]["cono"]["p95"].max())
    fig.update_yaxes(range=[lo * 0.97, hi * 1.03], row=1, col=1)
    fig.update_xaxes(range=[vis.index[0], d.index[-1] + tf_td * extra_x], rangeslider_visible=False)
    for fila, titulo in ((1, "Precio"), (2, "Volumen"), (3, "RSI"), (4, "MACD")):
        fig.update_yaxes(title_text=titulo, row=fila, col=1, side="right")
    fig.update_layout(template="plotly_dark", height=860, margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation="h", y=1.03, x=0), hovermode="x unified",
                      uirevision=f"{sym}{tf}", dragmode="pan")
    return fig


def panel(r: dict):
    cols = st.columns(6)
    items = [(f"Tendencia {tf}", r["tendencia"]), ("Tendencia diaria", r["tendencia_diaria"] if tf != "1d" else "N/A"),
             ("MACD", r["macd"]), ("RSI", f"{r['rsi']:.1f}"), ("Volumen", r["volumen"]),
             ("Puntos C / V", f"{r['pts_c']} / {r['pts_v']}")]
    for col, (k, v) in zip(cols, items):
        if k == "RSI":
            c = ROJO if r["rsi"] >= 70 else VERDE if r["rsi"] <= 30 else GRIS
            v = v + (" · Sobrecompra" if r["rsi"] >= 70 else " · Sobreventa" if r["rsi"] <= 30 else "")
        elif k.startswith("Puntos"):
            c = VERDE if r["pts_c"] >= p.min_puntos else ROJO if r["pts_v"] >= p.min_puntos else GRIS
        else:
            c = color_estado(v)
        col.markdown(f"<div style='font-size:0.8rem;color:#9ca3af'>{k}</div>{badge(v, c)}", unsafe_allow_html=True)


def vista_precio():
    try:
        _, sym = cargar_velas(exchange, simbolo, tf, velas)
        precio, cambio = cargar_precio(exchange, sym)
    except Exception as e:
        st.caption(f"Precio en vivo no disponible: {e}")
        return
    c1, c2 = st.columns([1.2, 3])
    c1.metric(f"{sym} · en vivo", fmt(precio), f"{cambio:.2f}% 24h" if cambio is not None else None)
    c2.markdown(f"<div style='color:#9ca3af;font-size:0.8rem;padding-top:28px'>🟢 {n.EXCHANGES[exchange]} · "
                f"{datetime.now().strftime('%H:%M:%S')} · se actualiza cada {seg_precio} s</div>", unsafe_allow_html=True)


def vista_grafico():
    try:
        df, sym = cargar_velas(exchange, simbolo, tf, velas)
    except Exception as e:
        st.error(f"No pude descargar datos de {n.EXCHANGES[exchange]}: {e}\n\nPrueba otra *Fuente de datos* en la barra lateral.")
        return
    df = df.copy()
    precio, cambio = None, None
    try:
        precio, cambio = cargar_precio(exchange, sym)
    except Exception:
        pass
    if precio and not df["cerrada"].iloc[-1]:  # vela en curso con el precio en vivo
        i = df.index[-1]
        df.loc[i, "close"] = precio
        df.loc[i, "high"] = max(df.loc[i, "high"], precio)
        df.loc[i, "low"] = min(df.loc[i, "low"], precio)
    d = n.analizar(df, p, tf)
    r = n.resumen(d, tf)

    tf_ms = n.TF_MS[tf]
    ahora_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    faltan_min = (tf_ms - ahora_ms % tf_ms) // 60000
    c2, c3, c4 = st.columns([1, 1.4, 1.3])
    estado_txt = {1: "Última: compra", -1: "Última: venta", 0: "Sin señal"}[r["estado"]]
    c2.markdown(f"<div style='font-size:0.8rem;color:#9ca3af'>Dirección</div>{badge(estado_txt, {1: VERDE, -1: ROJO, 0: GRIS}[r['estado']])}",
                unsafe_allow_html=True)
    if r["ultima_tipo"]:
        f = r["ultima_fecha"].tz_convert(zona).strftime("%d/%m %H:%M")
        c3.markdown(f"<div style='font-size:0.8rem;color:#9ca3af'>Última señal</div>"
                    f"{badge(r['ultima_tipo'] + ' @ ' + fmt(r['ultima_precio']), color_estado(r['ultima_tipo']))} "
                    f"<span style='color:#9ca3af;font-size:0.8rem'>{f}</span>", unsafe_allow_html=True)
    c4.markdown(f"<div style='font-size:0.8rem;color:#9ca3af'>Próxima vela {tf}</div><b>cierra en {faltan_min // 60} h {faltan_min % 60} min</b>",
                unsafe_allow_html=True)
    st.write("")
    panel(r)
    if r["preaviso"]:
        dir_ = "COMPRA" if r["preaviso"] == 1 else "VENTA"
        falta = r["falta_c"] if r["preaviso"] == 1 else r["falta_v"]
        st.warning(f"⚠️ Preaviso de **{dir_}** en la última vela cerrada. Falta: {', '.join(falta)}")
    ev = None
    if mostrar_cono or mostrar_kalman:
        try:
            ev = modelos_actuales()
        except Exception as e:
            st.caption(f"Modelos no disponibles: {e}")
    st.plotly_chart(figura(d, sym, ev), width="stretch", config={"scrollZoom": True, "displaylogo": False})
    st.caption(f"Datos: {n.EXCHANGES[exchange]} · actualizado {datetime.now().strftime('%H:%M:%S')} · "
               "Las señales solo se confirman al cierre de la vela. Rueda del ratón = zoom, arrastrar = mover, doble clic = ajustar.")
    st.session_state["ultimo_d"] = d
    st.session_state["ultimo_sym"] = sym


with tabs[0]:
    if auto:
        st.fragment(run_every=f"{seg_precio}s")(vista_precio)()
        st.fragment(run_every=f"{seg_graf}s")(vista_grafico)()
    else:
        vista_precio()
        vista_grafico()


# =============================================================================
# MODELOS MATEMÁTICOS
# =============================================================================
ICONO = {"favor": "✅", "contra": "❌", "neutral": "➖"}

with tabs[1]:
    try:
        ev = modelos_actuales()
    except Exception as e:
        st.error(f"No se pudieron calcular los modelos: {e}")
        ev = None
    if ev:
        d_m = st.session_state.get("ultimo_d")
        r_m = n.resumen(d_m, tf) if d_m is not None else None
        sugerida = 1
        if r_m:
            sugerida = r_m["preaviso"] or r_m["estado"] or 1
        st.subheader(f"Lectura matemática · {simbolo} · {tf}")
        direc = st.radio("Evaluar la dirección", [1, -1], index=0 if sugerida == 1 else 1, horizontal=True,
                         format_func=lambda x: "Compra" if x == 1 else "Venta")
        conf = mdl.confirmacion(ev, direc)
        col_v = {"Fuerte": VERDE, "Moderada": AMARILLO, "Débil": ROJO}[conf["veredicto"]]
        st.markdown(f"### Confirmación: {badge(conf['veredicto'], col_v)} "
                    f"<span style='font-size:1rem;color:#9ca3af'>{conf['favor']} a favor · {conf['contra']} en contra · "
                    f"{conf['total'] - conf['favor'] - conf['contra']} neutrales</span>", unsafe_allow_html=True)
        for nombre, est, txt in conf["items"]:
            st.markdown(f"{ICONO[est]} **{nombre}**: {txt}")
        st.caption("Úsalo como filtro: la mejor situación es señal técnica + confirmación Fuerte + noticias sin nada en contra.")
        st.divider()

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### 🤖 Machine learning (walk-forward)")
            ml = ev["ml"]
            if not ml.get("ok"):
                st.info(ml.get("motivo", "No disponible"))
            else:
                m = st.columns(3)
                m[0].metric(f"Prob. de subir en {ml['h']} velas", f"{ml['prob_sube'] * 100:.0f}%")
                m[1].metric("Acierto fuera de muestra", f"{ml['acierto'] * 100:.1f}%", f"{(ml['acierto'] - ml['base']) * 100:+.1f} pts vs azar")
                m[2].metric("AUC", f"{ml['auc']:.3f}", f"± {ml['auc_se']:.3f}", delta_color="off")
                if ml["ventaja"]:
                    st.success("El modelo muestra una ventaja estadísticamente significativa (95%) sobre el azar en datos que no vio al entrenar.")
                else:
                    st.warning("Sin ventaja estadística demostrada en este activo y horizonte: su probabilidad no se usa para confirmar. "
                               "Prueba otro horizonte o carga más velas.")
                st.caption(f"Gradient Boosting + Regresión logística, {ml['n']} muestras, validación en 5 bloques cronológicos "
                           "(siempre entrena con el pasado y evalúa en el futuro).")
        with c2:
            st.markdown("#### 📐 Filtro de Kalman")
            k = ev["kalman"]
            if k.get("ok"):
                m = st.columns(2)
                m[0].metric("Pendiente de la tendencia", f"{k['pendiente_pct']:+.3f}% por vela")
                m[1].metric("Confianza alcista", f"{k['prob_alcista'] * 100:.0f}%")
                st.caption("Separa la tendencia real del ruido (modelo de tendencia lineal local por máxima verosimilitud). "
                           "Es la línea punteada rosa del gráfico.")
            st.markdown("#### 🌊 Exponente de Hurst")
            hu = ev["hurst"]
            if hu.get("ok"):
                st.metric("H", f"{hu['h']:.2f}", hu["tipo"], delta_color="off")
                st.caption("H > 0.55: los movimientos tienden a continuar (favorece seguir tendencias). "
                           "H < 0.45: tienden a revertir. Cerca de 0.5: comportamiento aleatorio.")

        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### 🧭 Régimen de mercado")
            rg = ev["regimen"]
            if rg.get("ok"):
                col_r = {"Alcista": VERDE, "Bajista": ROJO}.get(rg["actual"], GRIS)
                st.markdown(f"Régimen actual: {badge(rg['actual'], col_r)} con {rg['prob'] * 100:.0f}% de certeza"
                            + (" · ⚠️ volatilidad alta" if rg["volatil"] else ""), unsafe_allow_html=True)
                st.dataframe(rg["tabla"].style.format({"Retorno medio por vela %": "{:+.3f}", "Volatilidad por vela %": "{:.2f}",
                                                       "% del tiempo": "{:.0f}%"}), hide_index=True, width="stretch")
                if d_m is not None:
                    serie = rg["serie"]
                    precios = d_m["close"].reindex(serie.index)
                    figr = go.Figure()
                    for nom, colr in (("Alcista", VERDE), ("Lateral", GRIS), ("Bajista", ROJO)):
                        msk = serie == nom
                        figr.add_trace(go.Scatter(x=serie.index[msk].tz_convert(zona), y=precios[msk], mode="markers",
                                                  marker=dict(size=3, color=colr), name=nom))
                    figr.update_layout(template="plotly_dark", height=280, margin=dict(l=10, r=10, t=10, b=10),
                                       legend=dict(orientation="h"))
                    st.plotly_chart(figr, width="stretch")
                st.caption("Mezcla gaussiana sobre retorno y volatilidad de las últimas 20 velas.")
        with c2:
            st.markdown("#### 🎲 Volatilidad GARCH + Monte Carlo")
            g = ev["garch"]
            if g.get("ok"):
                m = st.columns(3)
                m[0].metric("Volatilidad por día", f"{g['vol_dia']:.2f}%")
                lp = g["vol_largo_plazo"] * (mdl.VELAS_DIA[tf] ** 0.5)
                m[1].metric("Normal (largo plazo)", f"{lp:.2f}%" if lp == lp else "-")
                m[2].metric("Persistencia", f"{g['persistencia']:.2f}")
                lo95, hi95 = g["rango_final"]
                st.markdown(f"En **{g['h']} velas** el precio quedaría entre **{fmt(lo95)}** y **{fmt(hi95)}** "
                            "con 90% de probabilidad (cono azul del gráfico).")
                tabla = pd.DataFrame([
                    {"Si entras ahora en": "Compra", "TP": fmt(g["compra"]["tp"]), "SL": fmt(g["compra"]["sl"]),
                     "Toca TP primero": f"{g['compra']['p_tp'] * 100:.0f}%", "Toca SL primero": f"{g['compra']['p_sl'] * 100:.0f}%",
                     "Ninguno": f"{g['compra']['p_ninguno'] * 100:.0f}%"},
                    {"Si entras ahora en": "Venta", "TP": fmt(g["venta"]["tp"]), "SL": fmt(g["venta"]["sl"]),
                     "Toca TP primero": f"{g['venta']['p_tp'] * 100:.0f}%", "Toca SL primero": f"{g['venta']['p_sl'] * 100:.0f}%",
                     "Ninguno": f"{g['venta']['p_ninguno'] * 100:.0f}%"}])
                st.dataframe(tabla, hide_index=True, width="stretch")
                be = p.sl_mult / (p.sl_mult + p.tp_mult) * 100
                st.caption(f"4.000 simulaciones con la volatilidad GARCH(1,1) y los movimientos reales del pasado. "
                           f"Con tu SL/TP necesitas acertar más del {be:.0f}% de las veces para ganar. "
                           "La simulación no asume dirección; muestra si el TP es alcanzable con la volatilidad actual.")

        with st.expander("¿Qué tan confiables son estos modelos? (léelo)"):
            st.markdown("""
- **Nadie puede predecir el precio con certeza.** Los mercados líquidos como BTC se acercan mucho a un paseo aleatorio, y cualquier ventaja real es pequeña.
- Por eso estos modelos dan **probabilidades y rangos**, no un precio exacto. Se usan para **filtrar** señales: entrar solo cuando la técnica y las matemáticas coinciden.
- El modelo de machine learning se evalúa **fuera de muestra**: siempre se entrena con el pasado y se mide en datos que no vio. Si no demuestra ventaja con 95% de confianza, la app lo dice y no lo cuenta.
- **No usamos LSTM ni ARIMA para el precio:** con unos pocos miles de velas, las redes neuronales sobreajustan (aprenden el ruido). Y ARIMA sobre precios cripto no supera al azar en la práctica.
- Compruébalo en la pestaña **Backtest** con la opción *Filtrar con el modelo ML*.
""")


# =============================================================================
# ESCÁNER
# =============================================================================
with tabs[2]:
    st.subheader("Escáner de tu lista")
    st.caption("Estado de cada cripto en la última vela cerrada. Se actualiza cada 5 minutos.")
    if st.button("🔄 Actualizar ahora"):
        cargar_escaner.clear()
    with st.spinner("Analizando tu lista…"):
        filas = cargar_escaner(exchange, tuple(cfg["watchlist"]), tf, velas, json.dumps(asdict(p)))
    tabla = []
    for f in filas:
        if "error" in f:
            tabla.append({"Cripto": f["Cripto"], "Precio": "error", "Estado": f["error"]})
            continue
        estado = {1: "🟢 En compra", -1: "🔴 En venta", 0: "⚪ Sin señal"}[f["estado"]]
        pre = {1: "🟡 Compra", -1: "🟡 Venta", 0: ""}[f["preaviso"]]
        ult = f"{f['ultima_tipo']} {f['ultima_fecha'].tz_convert(zona).strftime('%d/%m %H:%M')}" if f["ultima_tipo"] else "-"
        tabla.append({"Cripto": f["Cripto"], "Precio": fmt(f["precio"]), "Estado": estado, "Preaviso": pre,
                      f"Tend. {tf}": f["tendencia"], "Tend. diaria": f["tendencia_diaria"], "MACD": f["macd"],
                      "RSI": round(f["rsi"], 1), "Puntos C/V": f"{f['pts_c']}/{f['pts_v']}", "Última señal": ult})
    st.dataframe(pd.DataFrame(tabla), width="stretch", hide_index=True)


# =============================================================================
# BACKTEST
# =============================================================================
with tabs[3]:
    st.subheader("Backtest: ¿cómo le habría ido a la estrategia?")
    d_bt = st.session_state.get("ultimo_d")
    if d_bt is None:
        st.info("Abre primero la pestaña Gráfico para cargar datos.")
    else:
        c1, c2, c3 = st.columns(3)
        solo_largos = c1.checkbox("Solo compras (spot, sin ventas en corto)", True,
                                  help="En spot, una señal de VENTA significa salir de la posición, no apostar a la baja.")
        comision = c2.number_input("Comisión por operación (%)", 0.0, 1.0, 0.1, step=0.05)
        usar_filtro = c3.checkbox("Filtrar con el modelo ML (fuera de muestra)", False,
                                  help="Solo toma las señales que el modelo apoyaba EN ESE MOMENTO, sin ver el futuro.")
        filtro = None
        if usar_filtro:
            try:
                ml_bt = modelos_actuales()["ml"]
                if ml_bt.get("ok"):
                    filtro = ml_bt["oos"]
                    d_bt = d_bt[d_bt.index >= filtro.dropna().index[0]]
                    _, st_sin, _ = n.backtest(d_bt, solo_largos, comision)
                    _, st_con, _ = n.backtest(d_bt, solo_largos, comision, filtro)
                    comp = pd.DataFrame([{"": "Sin filtro", **st_sin}, {"": "Con filtro ML", **st_con}])
                    cols = {"operaciones": "Operaciones", "acierto": "% acierto", "rendimiento": "Rendimiento %",
                            "factor_beneficio": "Factor beneficio", "max_caida": "Caída máx. %"}
                    st.markdown("**Comparación en el mismo período**")
                    st.dataframe(comp[[""] + [c for c in cols if c in comp]].rename(columns=cols).round(2),
                                 hide_index=True, width="stretch")
            except Exception as e:
                st.caption(f"Filtro no disponible: {e}")
        trades, stats, curva = n.backtest(d_bt, solo_largos, comision, filtro)
        if not stats["operaciones"]:
            st.warning("No hubo operaciones en el período cargado. Prueba con más velas o menos puntos mínimos.")
        else:
            m = st.columns(6)
            m[0].metric("Operaciones", stats["operaciones"])
            m[1].metric("% de acierto", f"{stats['acierto']:.0f}%")
            m[2].metric("Rendimiento total", f"{stats['rendimiento']:.1f}%")
            m[3].metric("Comprar y mantener", f"{stats['comprar_y_mantener']:.1f}%")
            pf = stats["factor_beneficio"]
            m[4].metric("Factor de beneficio", "∞" if pf == float("inf") else f"{pf:.2f}")
            m[5].metric("Caída máxima", f"{stats['max_caida']:.1f}%")
            fig = go.Figure(go.Scatter(x=curva.index.tz_convert(zona), y=curva.values, mode="lines+markers",
                                       line=dict(color=VERDE if stats["rendimiento"] >= 0 else ROJO)))
            fig.update_layout(template="plotly_dark", height=320, margin=dict(l=10, r=10, t=30, b=10),
                              title="Evolución de 100 invertidos", yaxis_title="Capital")
            st.plotly_chart(fig, width="stretch")
            t = trades.copy()
            for c in ("Entrada", "Salida"):
                t[c] = pd.to_datetime(t[c]).dt.tz_convert(zona).dt.strftime("%d/%m/%Y %H:%M")
            st.dataframe(t.style.format({"Precio entrada": "{:,.4f}", "Precio salida": "{:,.4f}", "Resultado %": "{:+.2f}%"}),
                         width="stretch", hide_index=True)
            st.caption("Factor de beneficio > 1 = ganó más de lo que perdió. Resultados pasados no garantizan resultados futuros. "
                       "Prueba cambiar los parámetros en la barra lateral y compara.")


# =============================================================================
# NOTICIAS
# =============================================================================
with tabs[4]:
    base = simbolo.split("/")[0]
    st.subheader(f"Noticias: {base} y mercado")
    c1, c2 = st.columns([2, 1])
    filtro = c1.radio("Mostrar", ["Todas", f"Solo {base}", "Macro / regulación"], horizontal=True)
    if c2.button("🔄 Recargar noticias"):
        cargar_noticias.clear()
    with st.spinner("Leyendo noticias…"):
        noticias = cargar_noticias(base)
    if filtro.startswith("Solo"):
        noticias = [x for x in noticias if x["tipo"] == "Moneda"]
    elif filtro.startswith("Macro"):
        noticias = [x for x in noticias if x["tipo"] == "Macro / regulación"]
    if not noticias:
        st.info("No hay noticias para este filtro (o no se pudieron leer las fuentes).")
    for x in noticias:
        c_tono = {"Positivo": VERDE, "Negativo": ROJO}.get(x["tono"], GRIS)
        hora = x["fecha"].astimezone(pd.Timestamp.now(tz=zona).tz).strftime("%d/%m %H:%M")
        st.markdown(f"{badge(x['tono'], c_tono)} {badge(x['tipo'], '#374151')} "
                    f"**[{x['titulo']}]({x['link']})**  \n<span style='color:#9ca3af;font-size:0.8rem'>{x['fuente']} · {hora}</span>",
                    unsafe_allow_html=True)
    st.caption("El tono es una estimación automática por palabras clave del titular; léela antes de decidir.")


# =============================================================================
# ALERTAS
# =============================================================================
with tabs[5]:
    st.subheader("Alertas por correo")
    st.markdown("Las alertas las envía el **monitor** (`iniciar_monitor`), que corre aparte y revisa tu lista "
                "al cierre de cada vela, aunque esta página esté cerrada. Aquí lo configuras.")

    st.markdown("#### 1. Tu correo (Gmail)")
    import os
    env = n.cargar_env()
    if os.environ.get("GMAIL_APP_PASSWORD"):
        st.success(f"Correo configurado en la nube (Secrets): {env.get('GMAIL_USER', '')}. Para cambiarlo, edita los Secrets.")
    else:
        with st.form("correo"):
            usuario = st.text_input("Tu Gmail", env.get("GMAIL_USER", ""))
            clave = st.text_input("Contraseña de aplicación de Gmail (16 letras)", "", type="password",
                                  placeholder="•••• guardada (escribe solo si quieres cambiarla)" if env.get("GMAIL_APP_PASSWORD") else "",
                                  help="No es tu contraseña normal. Créala en myaccount.google.com/apppasswords")
            destino = st.text_input("Enviar alertas a (puede ser el mismo)", env.get("EMAIL_TO", ""))
            guardar = st.form_submit_button("💾 Guardar")
        if guardar:
            n.guardar_env(usuario, clave or env.get("GMAIL_APP_PASSWORD", ""), destino)
            st.success("Correo guardado en el archivo .env de tu computadora.")
    if st.button("📧 Enviar correo de prueba"):
        ok, msg = n.enviar_correo("Prueba · Analizador Cripto",
                                  "<h3>¡Funciona! ✅</h3><p>Así te llegarán las alertas de compra, venta y preaviso.</p>")
        (st.success if ok else st.error)(msg)

    st.markdown("#### 2. Qué vigilar")
    nueva_lista = st.multiselect("Criptos en tu lista", sorted(set(cfg["watchlist"] + n.DEFAULT_CONFIG["watchlist"] +
                                 ["ADA/USDT", "AVAX/USDT", "LINK/USDT", "DOT/USDT", "LTC/USDT", "TRX/USDT"])),
                                 default=cfg["watchlist"])
    extra = st.text_input("Agregar otro par (ej. PEPE/USDT)", "")
    a = cfg["alertas"]
    c1, c2, c3 = st.columns(3)
    a_sen = c1.checkbox("Señales confirmadas", a["senales"])
    a_pre = c2.checkbox("Preavisos", a["preavisos"])
    a_not = c3.checkbox("Incluir noticias en el correo", a["incluir_noticias"])
    if st.button("💾 Guardar lista y alertas"):
        lista = nueva_lista + ([extra.upper().strip()] if extra.strip() else [])
        cfg.update({"watchlist": lista, "alertas": {"senales": a_sen, "preavisos": a_pre, "incluir_noticias": a_not},
                    "exchange": exchange, "timeframe": tf, "params": vals})
        n.guardar_config(cfg)
        cargar_escaner.clear()
        st.success("Guardado. El monitor lo tomará en su próxima revisión.")

    st.markdown("#### 3. Encender el monitor")
    st.markdown("- **Windows:** doble clic en `iniciar_monitor.bat` y deja esa ventana abierta (puedes minimizarla).\n"
                "- **Mac:** doble clic en `iniciar_monitor.command`.\n"
                f"- Revisa al cierre de cada vela de **{cfg['timeframe']}** (Toronto: 19:00, 23:00, 03:00, 07:00, 11:00 y 15:00 para 4h).")

    st.markdown("#### Últimas alertas enviadas")
    if n.ESTADO_PATH.exists():
        enviados = json.loads(n.ESTADO_PATH.read_text(encoding="utf-8"))
        if enviados:
            filas = [{"Alerta": k.split("|")[3], "Cripto": k.split("|")[0], "Enviada": v} for k, v in list(enviados.items())[-20:]][::-1]
            st.dataframe(pd.DataFrame(filas), width="stretch", hide_index=True)
        else:
            st.caption("Todavía no se ha enviado ninguna.")
    else:
        st.caption("El monitor todavía no ha corrido.")
