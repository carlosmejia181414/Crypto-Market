"""
Núcleo del Analizador Cripto
----------------------------
Datos de mercado, indicadores, señales (misma lógica que el indicador de
TradingView "Señales 4H"), backtesting, noticias y envío de correos.

Herramienta de apoyo para tus propios análisis. No es asesoría financiera.
"""
from __future__ import annotations

import json
import os
import re
import smtplib
import ssl
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import parsedate_to_datetime
from pathlib import Path

import ccxt
import numpy as np
import pandas as pd
import requests

BASE = Path(__file__).resolve().parent
CONFIG_PATH = BASE / "config.json"
ENV_PATH = BASE / ".env"
ESTADO_PATH = BASE / "estado_alertas.json"

TF_MS = {"1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}
EXCHANGES = {"binance": "Binance", "kraken": "Kraken", "okx": "OKX"}
LIMITE_LOTE = {"binance": 1000, "kraken": 720, "okx": 300}


# =============================================================================
# CONFIGURACIÓN
# =============================================================================
@dataclass
class Params:
    ema_rapida: int = 50
    ema_lenta: int = 200
    usar_htf: bool = True
    ema_htf: int = 50
    rsi_len: int = 14
    rsi_compra_max: float = 70.0
    rsi_venta_min: float = 30.0
    macd_rapida: int = 12
    macd_lenta: int = 26
    macd_senal: int = 9
    usar_vol: bool = True
    vol_len: int = 20
    atr_len: int = 14
    sl_mult: float = 1.5
    tp_mult: float = 3.0
    bb_len: int = 20
    bb_mult: float = 2.0
    min_puntos: int = 4
    usar_preaviso: bool = True


DEFAULT_CONFIG = {
    "exchange": "binance",
    "timeframe": "4h",
    "velas": 1500,
    "horizonte_ml": 6,
    "zona_horaria": "America/Toronto",
    "watchlist": ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "BNB/USDT", "DOGE/USDT"],
    "params": asdict(Params()),
    "alertas": {"senales": True, "preavisos": True, "incluir_noticias": True},
}


def cargar_config() -> dict:
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    if CONFIG_PATH.exists():
        try:
            guardado = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            for k, v in guardado.items():
                if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                    cfg[k].update(v)
                else:
                    cfg[k] = v
        except Exception:
            pass
    if os.environ.get("EXCHANGE"):  # en la nube (GitHub/Streamlit) se fija la fuente por variable
        cfg["exchange"] = os.environ["EXCHANGE"]
    return cfg


def guardar_config(cfg: dict) -> None:
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")


def params_desde(cfg: dict) -> Params:
    validos = {f.name for f in fields(Params)}
    return Params(**{k: v for k, v in cfg.get("params", {}).items() if k in validos})


# =============================================================================
# DATOS DE MERCADO
# =============================================================================
_exchanges: dict = {}


def get_exchange(nombre: str):
    if nombre not in _exchanges:
        _exchanges[nombre] = getattr(ccxt, nombre)({"enableRateLimit": True, "timeout": 20000})
    return _exchanges[nombre]


def resolver_simbolo(ex, simbolo: str) -> str:
    """Usa el par pedido; si el exchange no lo tiene, prueba USDT, USD y USDC."""
    if not ex.markets:
        ex.load_markets()
    simbolo = simbolo.upper().strip()
    if simbolo in ex.markets:
        return simbolo
    base = simbolo.split("/")[0]
    for quote in ("USDT", "USD", "USDC"):
        s = f"{base}/{quote}"
        if s in ex.markets:
            return s
    raise ValueError(f"{simbolo} no está disponible en {ex.name}")


def descargar_velas(exchange: str, simbolo: str, timeframe: str = "4h", velas: int = 1000):
    """Devuelve (DataFrame de velas en UTC, símbolo usado). La última vela puede estar abierta."""
    ex = get_exchange(exchange)
    sym = resolver_simbolo(ex, simbolo)
    tf = TF_MS[timeframe]
    ahora = ex.milliseconds()
    since = ahora - velas * tf
    lote_max = LIMITE_LOTE.get(exchange, 500)
    filas: list = []
    for _ in range(30):
        lote = ex.fetch_ohlcv(sym, timeframe, since=since, limit=lote_max)
        if not lote:
            break
        filas.extend(lote)
        ultimo = lote[-1][0]
        if ultimo + tf > ahora or ultimo < since:
            break
        since = ultimo + tf
    if not filas:
        raise RuntimeError(f"No llegaron datos de {sym} desde {ex.name}")
    df = pd.DataFrame(filas, columns=["ts", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates("ts").sort_values("ts").tail(velas)
    df.index = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df.index.name = "fecha"
    df["cerrada"] = (df["ts"] + tf) <= ahora
    return df, sym


def precio_actual(exchange: str, sym: str):
    t = get_exchange(exchange).fetch_ticker(sym)
    return t.get("last"), t.get("percentage")


# =============================================================================
# INDICADORES Y SEÑALES
# =============================================================================
def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def _rma(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1 / n, adjust=False).mean()


def analizar(df: pd.DataFrame, p: Params, timeframe: str = "4h") -> pd.DataFrame:
    d = df.copy()
    c = d["close"]

    d["ema_r"] = _ema(c, p.ema_rapida)
    d["ema_l"] = _ema(c, p.ema_lenta)

    delta = c.diff()
    rs = _rma(delta.clip(lower=0), p.rsi_len) / _rma(-delta.clip(upper=0), p.rsi_len)
    d["rsi"] = 100 - 100 / (1 + rs)

    d["macd"] = _ema(c, p.macd_rapida) - _ema(c, p.macd_lenta)
    d["macd_sig"] = _ema(d["macd"], p.macd_senal)
    d["macd_hist"] = d["macd"] - d["macd_sig"]

    tr = pd.concat([d["high"] - d["low"], (d["high"] - c.shift()).abs(), (d["low"] - c.shift()).abs()], axis=1).max(axis=1)
    d["atr"] = _rma(tr, p.atr_len)
    d["vol_prom"] = d["volume"].rolling(p.vol_len).mean()

    media = c.rolling(p.bb_len).mean()
    desv = c.rolling(p.bb_len).std(ddof=0)
    d["bb_sup"] = media + p.bb_mult * desv
    d["bb_inf"] = media - p.bb_mult * desv

    # Tendencia diaria (última vela diaria CERRADA, sin repintado)
    usa_htf = p.usar_htf and timeframe in ("1h", "4h")
    if timeframe in ("1h", "4h"):
        diario = c.resample("1D").last().dropna()
        d["htf_close"] = diario.shift(1).reindex(d.index, method="ffill")
        d["htf_ema"] = _ema(diario, p.ema_htf).shift(1).reindex(d.index, method="ffill")
    else:
        d["htf_close"] = np.nan
        d["htf_ema"] = np.nan

    d["tend_alc"] = (c > d["ema_l"]) & (d["ema_r"] > d["ema_l"])
    d["tend_baj"] = (c < d["ema_l"]) & (d["ema_r"] < d["ema_l"])
    d["htf_alc"] = d["htf_close"] > d["htf_ema"]
    d["htf_baj"] = d["htf_close"] < d["htf_ema"]
    d["htf_ok_c"] = d["htf_alc"] if usa_htf else True
    d["htf_ok_v"] = d["htf_baj"] if usa_htf else True
    d["macd_alc"] = d["macd"] > d["macd_sig"]
    d["macd_baj"] = d["macd"] < d["macd_sig"]
    d["rsi_alc"] = (d["rsi"] > 50) & (d["rsi"] < p.rsi_compra_max)
    d["rsi_baj"] = (d["rsi"] < 50) & (d["rsi"] > p.rsi_venta_min)
    sin_vol = d["volume"].fillna(0) == 0
    d["vol_ok"] = sin_vol | (d["volume"] > d["vol_prom"]) | (not p.usar_vol)

    ent = lambda s: s.astype(bool).astype(int)
    d["pts_c"] = ent(d["tend_alc"]) + ent(d["htf_ok_c"]) + ent(d["macd_alc"]) + ent(d["rsi_alc"]) + ent(d["vol_ok"])
    d["pts_v"] = ent(d["tend_baj"]) + ent(d["htf_ok_v"]) + ent(d["macd_baj"]) + ent(d["rsi_baj"]) + ent(d["vol_ok"])
    d["cond_c"] = d["tend_alc"] & (d["pts_c"] >= p.min_puntos)
    d["cond_v"] = d["tend_baj"] & (d["pts_v"] >= p.min_puntos)

    # Máquina de estados: una señal por cambio de dirección, solo velas cerradas
    n = len(d)
    senal = np.zeros(n, dtype=int)
    pre = np.zeros(n, dtype=int)
    estados = np.zeros(n, dtype=int)
    cerrada = d["cerrada"].to_numpy() if "cerrada" in d else np.ones(n, dtype=bool)
    listo = np.arange(n) >= p.ema_lenta  # calentamiento de la EMA lenta
    cc, cv = d["cond_c"].to_numpy(), d["cond_v"].to_numpy()
    pc, pv = d["pts_c"].to_numpy(), d["pts_v"].to_numpy()
    ta, tb = d["tend_alc"].to_numpy(), d["tend_baj"].to_numpy()
    estado, pre_c_env, pre_v_env = 0, False, False
    for i in range(n):
        if cerrada[i] and listo[i]:
            if cc[i] and estado != 1:
                senal[i], estado = 1, 1
            elif cv[i] and estado != -1:
                senal[i], estado = -1, -1
            if p.usar_preaviso:
                # A la señal le falta exactamente UNA condición (la tendencia es obligatoria)
                cerca_c = not cc[i] and estado != 1 and (pc[i] >= p.min_puntos - 1 if ta[i] else pc[i] >= p.min_puntos)
                cerca_v = not cv[i] and estado != -1 and (pv[i] >= p.min_puntos - 1 if tb[i] else pv[i] >= p.min_puntos)
                if cerca_c and not pre_c_env:
                    pre[i], pre_c_env = 1, True
                if not cerca_c:
                    pre_c_env = False
                if cerca_v and not pre_v_env:
                    pre[i], pre_v_env = -1, True
                if not cerca_v:
                    pre_v_env = False
        estados[i] = estado
    d["senal"], d["preaviso"], d["estado"] = senal, pre, estados

    d["sl"] = np.where(senal == 1, c - d["atr"] * p.sl_mult, np.where(senal == -1, c + d["atr"] * p.sl_mult, np.nan))
    d["tp"] = np.where(senal == 1, c + d["atr"] * p.tp_mult, np.where(senal == -1, c - d["atr"] * p.tp_mult, np.nan))
    return d


def faltantes(fila, direccion: int, timeframe: str = "4h") -> list[str]:
    if direccion == 1:
        checks = [(f"Tendencia {timeframe}", fila["tend_alc"]), ("Tendencia diaria", fila["htf_ok_c"]),
                  ("MACD", fila["macd_alc"]), ("RSI", fila["rsi_alc"]), ("Volumen", fila["vol_ok"])]
    else:
        checks = [(f"Tendencia {timeframe}", fila["tend_baj"]), ("Tendencia diaria", fila["htf_ok_v"]),
                  ("MACD", fila["macd_baj"]), ("RSI", fila["rsi_baj"]), ("Volumen", fila["vol_ok"])]
    return [nombre for nombre, ok in checks if not bool(ok)]


def ultima_senal(d: pd.DataFrame):
    s = d[d["senal"] != 0]
    return s.iloc[-1] if len(s) else None


def resumen(d: pd.DataFrame, timeframe: str = "4h") -> dict:
    """Estado actual (última vela cerrada) para el escáner y el panel."""
    cerr = d[d["cerrada"]] if "cerrada" in d else d
    f = cerr.iloc[-1]
    us = ultima_senal(d)
    return {
        "precio": float(d["close"].iloc[-1]),
        "tendencia": "Alcista" if f["tend_alc"] else "Bajista" if f["tend_baj"] else "Neutral",
        "tendencia_diaria": "Alcista" if f["htf_alc"] else "Bajista" if f["htf_baj"] else "Neutral",
        "macd": "Alcista" if f["macd_alc"] else "Bajista",
        "rsi": float(f["rsi"]),
        "volumen": "Alto" if f["volume"] > f["vol_prom"] else "Bajo",
        "pts_c": int(f["pts_c"]),
        "pts_v": int(f["pts_v"]),
        "estado": int(f["estado"]),
        "ultima_tipo": None if us is None else ("COMPRA" if us["senal"] == 1 else "VENTA"),
        "ultima_fecha": None if us is None else us.name,
        "ultima_precio": None if us is None else float(us["close"]),
        "preaviso": int(f["preaviso"]),
        "falta_c": faltantes(f, 1, timeframe),
        "falta_v": faltantes(f, -1, timeframe),
    }


# =============================================================================
# BACKTEST
# =============================================================================
def backtest(d: pd.DataFrame, solo_largos: bool = True, comision_pct: float = 0.1, filtro: pd.Series | None = None):
    """Entra al cierre de la vela con señal; sale en SL, TP o señal contraria.
    Si en una misma vela se tocan SL y TP, se asume SL (supuesto conservador).
    filtro: probabilidad de subida (fuera de muestra); solo se entra si apoya la señal."""
    cerr = d[d["cerrada"]] if "cerrada" in d else d
    idx = cerr.index
    hi, lo, cl = cerr["high"].to_numpy(), cerr["low"].to_numpy(), cerr["close"].to_numpy()
    sen, sl_a, tp_a = cerr["senal"].to_numpy(), cerr["sl"].to_numpy(), cerr["tp"].to_numpy()
    com = comision_pct / 100
    f_arr = filtro.reindex(idx).to_numpy() if filtro is not None else None
    trades, pos = [], None

    def cerrar(i, precio, motivo):
        r = pos["dir"] * (precio / pos["entrada"] - 1) - 2 * com
        trades.append({"Entrada": pos["fecha"], "Tipo": "Largo" if pos["dir"] == 1 else "Corto",
                       "Precio entrada": pos["entrada"], "Salida": idx[i], "Precio salida": precio,
                       "Motivo": motivo, "Resultado %": r * 100})

    for i in range(len(cerr)):
        if pos is not None:
            if pos["dir"] == 1:
                if lo[i] <= pos["sl"]:
                    cerrar(i, pos["sl"], "Stop loss"); pos = None
                elif hi[i] >= pos["tp"]:
                    cerrar(i, pos["tp"], "Take profit"); pos = None
                elif sen[i] == -1:
                    cerrar(i, cl[i], "Señal contraria"); pos = None
            else:
                if hi[i] >= pos["sl"]:
                    cerrar(i, pos["sl"], "Stop loss"); pos = None
                elif lo[i] <= pos["tp"]:
                    cerrar(i, pos["tp"], "Take profit"); pos = None
                elif sen[i] == 1:
                    cerrar(i, cl[i], "Señal contraria"); pos = None
        permitido = f_arr is None or (not np.isnan(f_arr[i]) and (f_arr[i] > 0.5 if sen[i] == 1 else f_arr[i] < 0.5))
        if pos is None and sen[i] != 0 and (sen[i] == 1 or not solo_largos) and permitido:
            pos = {"dir": int(sen[i]), "entrada": cl[i], "sl": sl_a[i], "tp": tp_a[i], "fecha": idx[i]}
    if pos is not None:
        cerrar(len(cerr) - 1, cl[-1], "Abierta (precio actual)")

    tr = pd.DataFrame(trades)
    if tr.empty:
        return tr, {"operaciones": 0}, pd.Series(dtype=float)
    rend = tr["Resultado %"] / 100
    curva = (1 + rend).cumprod() * 100
    curva.index = tr["Salida"]
    ganadas, perdidas = rend[rend > 0].sum(), -rend[rend < 0].sum()
    stats = {
        "operaciones": len(tr),
        "acierto": (rend > 0).mean() * 100,
        "rendimiento": (curva.iloc[-1] / 100 - 1) * 100,
        "promedio": rend.mean() * 100,
        "factor_beneficio": (ganadas / perdidas) if perdidas > 0 else float("inf"),
        "max_caida": ((curva / curva.cummax()) - 1).min() * 100,
        "comprar_y_mantener": (cl[-1] / cl[0] - 1) * 100,
    }
    return tr, stats, curva


# =============================================================================
# NOTICIAS (RSS públicos)
# =============================================================================
FEEDS = {
    "CoinDesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "Cointelegraph": "https://cointelegraph.com/rss",
    "Decrypt": "https://decrypt.co/feed",
    "Bitcoin Magazine": "https://bitcoinmagazine.com/.rss/full/",
}
PALABRAS_MONEDA = {
    "BTC": ["bitcoin", "btc"], "ETH": ["ethereum", "ether", "eth"], "SOL": ["solana"],
    "XRP": ["xrp", "ripple"], "BNB": ["bnb", "binance"], "DOGE": ["dogecoin", "doge"],
    "ADA": ["cardano"], "AVAX": ["avalanche", "avax"], "LINK": ["chainlink"], "DOT": ["polkadot"],
    "LTC": ["litecoin"], "TRX": ["tron", "trx"], "TON": ["toncoin", "ton"], "SHIB": ["shiba"],
}
PALABRAS_MACRO = ["fed", "federal reserve", "powell", "inflation", "cpi", "interest rate", "rate cut",
                  "rate hike", "sec", "etf", "tariff", "regulation", "recession", "jobs report", "treasury"]
POSITIVAS = ["surge", "soar", "rally", "gain", "jump", "rise", "rises", "bull", "record", "approve", "approval",
             "inflow", "adopt", "rebound", "breakout", "upgrade", "buys", "accumulat"]
NEGATIVAS = ["plunge", "crash", "drop", "fall", "falls", "slump", "bear", "hack", "exploit", "ban", "lawsuit",
             "outflow", "sell-off", "selloff", "liquidat", "fear", "reject", "fraud", "dump", "tumble", "sink"]


def _contiene(texto: str, palabras: list[str]) -> bool:
    return any(re.search(rf"\b{re.escape(w)}", texto) for w in palabras)


def _tono(texto: str) -> str:
    pos = sum(1 for w in POSITIVAS if re.search(rf"\b{re.escape(w)}", texto))
    neg = sum(1 for w in NEGATIVAS if re.search(rf"\b{re.escape(w)}", texto))
    return "Positivo" if pos > neg else "Negativo" if neg > pos else "Neutral"


def obtener_noticias(base: str | None = None, max_n: int = 40) -> list[dict]:
    import feedparser
    items = []
    for fuente, url in FEEDS.items():
        try:
            r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0 AnalizadorCripto"})
            feed = feedparser.parse(r.content)
        except Exception:
            continue
        for e in feed.entries[:30]:
            titulo = re.sub(r"\s+", " ", e.get("title", "")).strip()
            resumen_txt = re.sub(r"<[^>]+>", " ", e.get("summary", ""))[:400]
            texto = f"{titulo} {resumen_txt}".lower()
            try:
                fecha = parsedate_to_datetime(e.get("published", "")).astimezone(timezone.utc)
            except Exception:
                fecha = datetime.now(timezone.utc)
            de_moneda = bool(base) and _contiene(texto, PALABRAS_MONEDA.get(base, [base.lower()]))
            macro = _contiene(texto, PALABRAS_MACRO)
            items.append({"titulo": titulo, "fuente": fuente, "fecha": fecha, "link": e.get("link", ""),
                          "tono": _tono(titulo.lower()),
                          "tipo": "Moneda" if de_moneda else "Macro / regulación" if macro else "General"})
    orden = {"Moneda": 0, "Macro / regulación": 1, "General": 2}
    items.sort(key=lambda x: (orden[x["tipo"]], -x["fecha"].timestamp()))
    return items[:max_n]


# =============================================================================
# CORREO (Gmail con contraseña de aplicación)
# =============================================================================
def cargar_env() -> dict:
    # En la nube las credenciales llegan como variables de entorno (GitHub Secrets / Streamlit Secrets)
    datos = {k: os.environ[k] for k in ("GMAIL_USER", "GMAIL_APP_PASSWORD", "EMAIL_TO") if os.environ.get(k)}
    if ENV_PATH.exists():
        for linea in ENV_PATH.read_text(encoding="utf-8").splitlines():
            if "=" in linea and not linea.strip().startswith("#"):
                k, v = linea.split("=", 1)
                datos[k.strip()] = v.strip().strip('"')
    return datos


def guardar_env(usuario: str, clave: str, destino: str) -> None:
    ENV_PATH.write_text(
        f'GMAIL_USER="{usuario.strip()}"\nGMAIL_APP_PASSWORD="{clave.replace(" ", "").strip()}"\n'
        f'EMAIL_TO="{(destino or usuario).strip()}"\n', encoding="utf-8")


def enviar_correo(asunto: str, cuerpo_html: str) -> tuple[bool, str]:
    env = cargar_env()
    usuario, clave = env.get("GMAIL_USER", ""), env.get("GMAIL_APP_PASSWORD", "")
    destino = env.get("EMAIL_TO") or usuario
    if not usuario or not clave:
        return False, "Falta configurar el correo (pestaña Alertas)."
    msg = MIMEMultipart("alternative")
    msg["Subject"], msg["From"], msg["To"] = asunto, usuario, destino
    msg.attach(MIMEText(cuerpo_html, "html", "utf-8"))
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ssl.create_default_context(), timeout=20) as s:
            s.login(usuario, clave)
            s.sendmail(usuario, [x.strip() for x in destino.split(",")], msg.as_string())
        return True, f"Correo enviado a {destino}"
    except smtplib.SMTPAuthenticationError:
        return False, "Gmail rechazó el usuario o la contraseña de aplicación."
    except Exception as e:
        return False, f"No se pudo enviar: {e}"


def _fmt(x: float) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    dec = 2 if abs(x) >= 100 else 4 if abs(x) >= 1 else 6
    return f"{x:,.{dec}f}"


def correo_alerta(sym: str, tf: str, evento: str, fila, zona: str, noticias: list[dict] | None = None, extra_html: str = ""):
    es_compra = "COMPRA" in evento
    color = "#16a34a" if es_compra else "#dc2626"
    if evento.startswith("PREAVISO"):
        color = "#ca8a04"
    hora = fila.name.tz_convert(zona).strftime("%d/%m/%Y %H:%M")
    filas = [("Precio de cierre", _fmt(fila["close"])), ("Vela", f"{tf} · cerró {hora}"),
             ("RSI", f"{fila['rsi']:.1f}"), ("Puntos compra / venta", f"{int(fila['pts_c'])} / {int(fila['pts_v'])}")]
    if not evento.startswith("PREAVISO"):
        riesgo = abs(fila["close"] - fila["sl"])
        filas += [("Stop loss", _fmt(fila["sl"])), ("Take profit", _fmt(fila["tp"])),
                  ("Riesgo / beneficio", f"1 : {abs(fila['tp'] - fila['close']) / riesgo:.1f}" if np.isfinite(riesgo) and riesgo > 0 else "-")]
    else:
        falta = faltantes(fila, 1 if es_compra else -1, tf)
        filas.append(("Falta", ", ".join(falta) or "-"))
    tabla = "".join(f"<tr><td style='padding:4px 12px;color:#555'>{k}</td><td style='padding:4px 12px'><b>{v}</b></td></tr>"
                    for k, v in filas)
    bloque_noticias = ""
    if noticias:
        lis = "".join(f"<li><a href='{n['link']}'>{n['titulo']}</a> <span style='color:#888'>({n['fuente']}, {n['tono']})</span></li>"
                      for n in noticias)
        bloque_noticias = f"<h3 style='margin-top:18px'>Noticias recientes</h3><ul>{lis}</ul>"
    que_hacer = ("Prepárate: revisa noticias y niveles. Aún NO es señal confirmada."
                 if evento.startswith("PREAVISO") else
                 "Antes de entrar: verifica que el precio no se haya alejado mucho de la entrada y que no haya noticias en contra. Usa siempre el stop loss.")
    html = f"""
    <div style="font-family:Arial,sans-serif;max-width:560px">
      <div style="background:{color};color:white;padding:14px 18px;border-radius:8px 8px 0 0">
        <h2 style="margin:0">{evento} · {sym}</h2></div>
      <div style="border:1px solid #ddd;border-top:none;padding:14px 18px;border-radius:0 0 8px 8px">
        <table>{tabla}</table>
        <p style="margin-top:14px">{que_hacer}</p>
        {extra_html}
        {bloque_noticias}
        <p style="color:#999;font-size:12px;margin-top:18px">Analizador Cripto · herramienta de apoyo, no es asesoría financiera.</p>
      </div></div>"""
    return f"{evento} · {sym} · {_fmt(fila['close'])}", html
