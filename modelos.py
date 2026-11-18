"""
Modelos matemáticos del Analizador Cripto
-----------------------------------------
Estos modelos no "adivinan" el precio. Estiman PROBABILIDADES y RANGOS, y se usan
para confirmar o filtrar las señales técnicas:

1. Machine learning (Gradient Boosting + Regresión logística) con validación
   walk-forward FUERA DE MUESTRA: probabilidad de que el precio suba en las
   próximas velas. Solo cuenta si demuestra ventaja sobre el azar.
2. Filtro de Kalman (tendencia lineal local): pendiente real de la tendencia
   sin el ruido, con su nivel de confianza.
3. Régimen de mercado (mezcla gaussiana): alcista, lateral o bajista.
4. Exponente de Hurst: si el mercado tiende a seguir tendencias (>0.5) o a
   revertir (<0.5).
5. GARCH(1,1) con t de Student + Monte Carlo (simulación histórica filtrada):
   volatilidad esperada, cono de precios y probabilidad de tocar TP o SL.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm

H_ML = {"1h": 24, "4h": 6, "1d": 5}    # horizonte del modelo ML (velas): ~1 día en 1h/4h
H_MC = {"1h": 72, "4h": 30, "1d": 20}  # horizonte de la simulación (velas): ~5 días en 4h
VELAS_DIA = {"1h": 24, "4h": 6, "1d": 1}


def _cerradas(d):
    return d[d["cerrada"]] if "cerrada" in d else d


# =============================================================================
# 1. MACHINE LEARNING CON VALIDACIÓN FUERA DE MUESTRA
# =============================================================================
def _features(d: pd.DataFrame) -> pd.DataFrame:
    c = d["close"]
    lp = np.log(c)
    lr = lp.diff()
    X = pd.DataFrame(index=d.index)
    # Solo variables estacionarias (no dependen del nivel del precio) para no sobreajustar
    for k in (1, 3, 6, 12):
        X[f"r{k}"] = lp.diff(k)
    X["dist_ema_r"] = c / d["ema_r"] - 1
    X["rsi"] = d["rsi"] / 100
    X["macd_h"] = d["macd_hist"] / c
    X["vol_rel"] = np.log((d["volume"] + 1) / (d["vol_prom"] + 1))
    X["bb_pos"] = (c - d["bb_inf"]) / (d["bb_sup"] - d["bb_inf"]).replace(0, np.nan)
    X["vol20"] = lr.rolling(20).std()
    X["vol_ratio"] = X["vol20"] / lr.rolling(100).std()
    X["htf"] = (d["htf_close"] / d["htf_ema"] - 1).fillna(0)
    return X.replace([np.inf, -np.inf], np.nan)


def _modelos(h):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return [
        HistGradientBoostingClassifier(max_depth=2, learning_rate=0.03, max_iter=120,
                                       l2_regularization=2.0, min_samples_leaf=60, random_state=0),
        make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000)),
    ]


def _auc_significativo(auc: float, n_pos: int, n_neg: int, h: int) -> tuple[bool, float]:
    """Hanley-McNeil con muestra efectiva n/h (los objetivos de h velas se solapan).
    Hay ventaja solo si el AUC supera 0.5 con 95% de confianza (una cola)."""
    n1, n2 = max(n_pos / h, 2), max(n_neg / h, 2)
    q1, q2 = auc / (2 - auc), 2 * auc ** 2 / (1 + auc)
    se = np.sqrt((auc * (1 - auc) + (n1 - 1) * (q1 - auc ** 2) + (n2 - 1) * (q2 - auc ** 2)) / (n1 * n2))
    return bool(auc - 1.645 * se > 0.5), float(se)


def modelo_ml(d: pd.DataFrame, tf: str, calentamiento: int = 200, h: int | None = None) -> dict:
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import TimeSeriesSplit

    cerr = _cerradas(d)
    h = h or H_ML.get(tf, 6)
    X = _features(cerr).iloc[calentamiento:].dropna()
    fut = (np.log(cerr["close"]).shift(-h) - np.log(cerr["close"])).reindex(X.index)
    ent = X[fut.notna()]
    y = (fut[fut.notna()] > 0).astype(int)
    if len(ent) < 300 or y.nunique() < 2:
        return {"ok": False, "motivo": "Pocos datos: carga al menos 1000 velas."}

    oos = pd.Series(np.nan, index=ent.index)
    for tr, te in TimeSeriesSplit(n_splits=5, gap=h).split(ent):
        if y.iloc[tr].nunique() < 2:
            continue
        probs = []
        for m in _modelos(h):
            m.fit(ent.iloc[tr], y.iloc[tr])
            probs.append(m.predict_proba(ent.iloc[te])[:, 1])
        oos.iloc[te] = np.mean(probs, axis=0)

    mask = oos.notna()
    yo, po = y[mask], oos[mask]
    acierto = float(((po > 0.5).astype(int) == yo).mean())
    base = float(max(yo.mean(), 1 - yo.mean()))
    auc = float(roc_auc_score(yo, po)) if yo.nunique() > 1 else 0.5

    significativo, se = _auc_significativo(auc, int(yo.sum()), int((1 - yo).sum()), h)
    probs = []
    for m in _modelos(h):
        m.fit(ent, y)
        probs.append(m.predict_proba(X.iloc[[-1]])[:, 1][0])
    return {
        "ok": True, "prob_sube": float(np.mean(probs)), "acierto": acierto, "base": base, "auc": auc,
        "ventaja": significativo and acierto > base, "auc_se": se, "h": h, "oos": oos, "n": int(len(ent)),
    }


# =============================================================================
# 2. FILTRO DE KALMAN (tendencia lineal local)
# =============================================================================
def kalman(d: pd.DataFrame, n: int = 500) -> dict:
    from statsmodels.tsa.statespace.structural import UnobservedComponents
    cerr = _cerradas(d).iloc[-n:]
    y = 100 * np.log(cerr["close"].to_numpy())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = UnobservedComponents(y, level="local linear trend").fit(disp=False, maxiter=300)
    pend = float(res.filtered_state[1, -1])
    sd = float(np.sqrt(max(res.filtered_state_cov[1, 1, -1], 1e-12)))
    nivel = pd.Series(np.exp(res.smoothed_state[0] / 100), index=cerr.index)
    return {"ok": True, "pendiente_pct": pend, "prob_alcista": float(norm.cdf(pend / sd)), "nivel": nivel}


# =============================================================================
# 3. RÉGIMEN DE MERCADO (mezcla gaussiana sobre retorno y volatilidad)
# =============================================================================
def regimen(d: pd.DataFrame, ventana: int = 20) -> dict:
    from sklearn.mixture import GaussianMixture
    cerr = _cerradas(d)
    lr = np.log(cerr["close"]).diff()
    F = pd.DataFrame({"m": lr.rolling(ventana).mean(), "v": lr.rolling(ventana).std()}).dropna()
    if len(F) < 200:
        return {"ok": False, "motivo": "Pocos datos"}
    Z = (F - F.mean()) / F.std()
    gm = GaussianMixture(3, covariance_type="full", random_state=0, n_init=3).fit(Z)
    lab = gm.predict(Z)
    prob = gm.predict_proba(Z)
    orden = F.groupby(lab)["m"].mean().sort_values().index
    nombres = {orden[0]: "Bajista", orden[1]: "Lateral", orden[2]: "Alcista"}
    serie = pd.Series([nombres[x] for x in lab], index=F.index)
    tabla = pd.DataFrame({
        "Régimen": [nombres[k] for k in orden],
        "Retorno medio por vela %": [F["m"][lab == k].mean() * 100 for k in orden],
        "Volatilidad por vela %": [F["v"][lab == k].mean() * 100 for k in orden],
        "% del tiempo": [(lab == k).mean() * 100 for k in orden],
    })
    return {"ok": True, "actual": serie.iloc[-1], "prob": float(prob[-1][lab[-1]]), "serie": serie, "tabla": tabla,
            "volatil": bool(F["v"].iloc[-1] > F["v"].median() * 1.3)}


# =============================================================================
# 4. EXPONENTE DE HURST
# =============================================================================
def hurst(d: pd.DataFrame, n: int = 600, max_lag: int = 60) -> dict:
    lp = np.log(_cerradas(d)["close"].to_numpy()[-n:])
    lags = np.arange(2, max_lag)
    tau = np.array([np.std(lp[l:] - lp[:-l]) for l in lags])
    h = float(np.polyfit(np.log(lags), np.log(tau), 1)[0])
    tipo = "Tendencial" if h >= 0.55 else "Revierte a la media" if h <= 0.45 else "Aleatorio"
    return {"ok": True, "h": h, "tipo": tipo}


# =============================================================================
# 5. GARCH(1,1) + MONTE CARLO
# =============================================================================
def garch_montecarlo(d: pd.DataFrame, p, tf: str, n_sim: int = 4000, semilla: int = 42) -> dict:
    from arch import arch_model
    cerr = _cerradas(d)
    r = 100 * np.log(cerr["close"]).diff().dropna().iloc[-1500:]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = arch_model(r, mean="Zero", vol="GARCH", p=1, q=1, dist="t").fit(disp="off", show_warning=False)
    om, al, be = (float(res.params[k]) for k in ("omega", "alpha[1]", "beta[1]"))
    z = res.std_resid.dropna().to_numpy()
    s2 = float(res.forecast(horizon=1, reindex=False).variance.values[-1, 0])

    H = H_MC.get(tf, 30)
    rng = np.random.default_rng(semilla)
    rets = np.empty((n_sim, H))
    s2v = np.full(n_sim, s2)
    for t in range(H):  # simulación histórica filtrada: residuos reales x volatilidad GARCH
        eps = rng.choice(z, n_sim) * np.sqrt(s2v)
        rets[:, t] = eps
        s2v = om + al * eps ** 2 + be * s2v
    p0 = float(d["close"].iloc[-1])
    caminos = p0 * np.exp(np.cumsum(rets / 100, axis=1))
    tf_td = pd.Timedelta(hours={"1h": 1, "4h": 4, "1d": 24}[tf])
    fechas = pd.DatetimeIndex([d.index[-1] + tf_td * (i + 1) for i in range(H)])
    pct = pd.DataFrame(np.percentile(caminos, [5, 25, 50, 75, 95], axis=0).T, index=fechas,
                       columns=["p5", "p25", "p50", "p75", "p95"])

    atr = float(cerr["atr"].iloc[-1])

    def niveles(direc):
        tp, sl = p0 + direc * atr * p.tp_mult, p0 - direc * atr * p.sl_mult
        toca_tp = caminos >= tp if direc == 1 else caminos <= tp
        toca_sl = caminos <= sl if direc == 1 else caminos >= sl
        i_tp = np.where(toca_tp.any(1), toca_tp.argmax(1), H + 1)
        i_sl = np.where(toca_sl.any(1), toca_sl.argmax(1), H + 1)
        p_tp, p_sl = float(np.mean(i_tp < i_sl)), float(np.mean(i_sl < i_tp))
        return {"tp": tp, "sl": sl, "p_tp": p_tp, "p_sl": p_sl, "p_ninguno": 1 - p_tp - p_sl}

    persist = al + be
    vol_lp = float(np.sqrt(om / (1 - persist))) if persist < 1 else float("nan")
    return {
        "ok": True, "h": H, "cono": pct, "precio": p0,
        "vol_vela": float(np.sqrt(s2)), "vol_dia": float(np.sqrt(s2 * VELAS_DIA[tf])),
        "vol_largo_plazo": vol_lp, "persistencia": persist,
        "compra": niveles(1), "venta": niveles(-1),
        "rango_final": (float(pct["p5"].iloc[-1]), float(pct["p95"].iloc[-1])),
    }


# =============================================================================
# EVALUACIÓN COMPLETA Y CONFIRMACIÓN DE SEÑALES
# =============================================================================
def evaluar(d: pd.DataFrame, p, tf: str) -> dict:
    ev = {}
    for nombre, f in (("ml", lambda: modelo_ml(d, tf, p.ema_lenta)), ("kalman", lambda: kalman(d)),
                      ("regimen", lambda: regimen(d)), ("hurst", lambda: hurst(d)),
                      ("garch", lambda: garch_montecarlo(d, p, tf))):
        try:
            ev[nombre] = f()
        except Exception as e:  # un modelo que falla no debe tumbar a los demás
            ev[nombre] = {"ok": False, "motivo": str(e)[:120]}
    return ev


def confirmacion(ev: dict, direc: int) -> dict:
    """Cada modelo vota: a favor, en contra o neutral respecto a la dirección (1 compra, -1 venta)."""
    items = []
    ml = ev.get("ml", {})
    if ml.get("ok"):
        pd_ = ml["prob_sube"] if direc == 1 else 1 - ml["prob_sube"]
        txt = f"{pd_ * 100:.0f}% de probabilidad a favor en {ml['h']} velas (acierto fuera de muestra {ml['acierto'] * 100:.0f}% vs {ml['base'] * 100:.0f}% al azar)"
        if not ml["ventaja"]:
            items.append(("Machine learning", "neutral", "Sin ventaja estadística demostrada; no se cuenta. " + txt))
        else:
            items.append(("Machine learning", "favor" if pd_ >= 0.55 else "contra" if pd_ <= 0.45 else "neutral", txt))
    k = ev.get("kalman", {})
    if k.get("ok"):
        pa = k["prob_alcista"] if direc == 1 else 1 - k["prob_alcista"]
        items.append(("Tendencia Kalman", "favor" if pa >= 0.65 else "contra" if pa <= 0.35 else "neutral",
                      f"Pendiente {k['pendiente_pct']:+.3f}% por vela; {pa * 100:.0f}% de confianza a favor"))
    r = ev.get("regimen", {})
    if r.get("ok"):
        quiere = "Alcista" if direc == 1 else "Bajista"
        contra = "Bajista" if direc == 1 else "Alcista"
        est = "favor" if r["actual"] == quiere else "contra" if r["actual"] == contra else "neutral"
        items.append(("Régimen de mercado", est, f"{r['actual']} ({r['prob'] * 100:.0f}% de certeza)" + (" · volatilidad alta" if r["volatil"] else "")))
    hu = ev.get("hurst", {})
    if hu.get("ok"):
        est = "favor" if hu["h"] >= 0.55 else "contra" if hu["h"] <= 0.45 else "neutral"
        items.append(("Exponente de Hurst", est, f"H = {hu['h']:.2f} · {hu['tipo']}"))
    g = ev.get("garch", {})
    if g.get("ok"):
        nv = g["compra" if direc == 1 else "venta"]
        # Informativo: con deriva cero no predice dirección; solo avisa si el TP es inalcanzable
        est = "contra" if nv["p_ninguno"] >= 0.6 else "neutral"
        items.append(("Alcance del TP (GARCH + Monte Carlo)", est,
                      f"En {g['h']} velas: TP {nv['p_tp'] * 100:.0f}% · SL {nv['p_sl'] * 100:.0f}% · ninguno {nv['p_ninguno'] * 100:.0f}%"))
    favor = sum(1 for _, e, _ in items if e == "favor")
    contra = sum(1 for _, e, _ in items if e == "contra")
    if favor >= 3 and contra == 0:
        veredicto = "Fuerte"
    elif favor - contra >= 1:
        veredicto = "Moderada"
    else:
        veredicto = "Débil"
    return {"items": items, "favor": favor, "contra": contra, "total": len(items), "veredicto": veredicto}


def html_confirmacion(conf: dict, direc: int) -> str:
    icono = {"favor": "✅", "contra": "❌", "neutral": "➖"}
    filas = "".join(f"<tr><td style='padding:3px 8px'>{icono[e]}</td><td style='padding:3px 8px'><b>{n}</b><br>"
                    f"<span style='color:#666;font-size:12px'>{t}</span></td></tr>" for n, e, t in conf["items"])
    color = {"Fuerte": "#16a34a", "Moderada": "#ca8a04", "Débil": "#dc2626"}[conf["veredicto"]]
    return (f"<h3 style='margin-top:18px'>Confirmación matemática ({'compra' if direc == 1 else 'venta'}): "
            f"<span style='color:{color}'>{conf['veredicto']}</span> · {conf['favor']} a favor, {conf['contra']} en contra</h3>"
            f"<table>{filas}</table>")
