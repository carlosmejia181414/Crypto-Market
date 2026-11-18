"""
Monitor de alertas del Analizador Cripto.

Revisa tu lista al cierre de cada vela y te envía un correo cuando hay
señal de COMPRA, VENTA o PREAVISO. Usa la configuración guardada desde la web.

    python monitor.py            -> corre siempre (deja la ventana abierta)
    python monitor.py --una-vez  -> revisa una vez y termina (para el Programador de tareas)
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone

import modelos as mdl
import nucleo as n

EVENTOS = {1: "COMPRA", -1: "VENTA"}


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%d/%m %H:%M:%S')}] {msg}", flush=True)


def cargar_estado() -> dict:
    try:
        return json.loads(n.ESTADO_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def guardar_estado(enviados: dict) -> None:
    ultimos = dict(list(enviados.items())[-500:])
    n.ESTADO_PATH.write_text(json.dumps(ultimos, indent=1, ensure_ascii=False), encoding="utf-8")


def revisar() -> None:
    cfg = n.cargar_config()
    p, tf, zona = n.params_desde(cfg), cfg["timeframe"], cfg.get("zona_horaria", "America/Toronto")
    al = cfg["alertas"]
    enviados = cargar_estado()
    for simbolo in cfg["watchlist"]:
        try:
            df, sym = n.descargar_velas(cfg["exchange"], simbolo, tf, cfg["velas"])
            d = n.analizar(df, p, tf)
            cerr = d[d["cerrada"]]
            if cerr.empty:
                continue
            f = cerr.iloc[-1]
            ev_modelos = None
            eventos = []
            if al.get("senales", True) and f["senal"] != 0:
                eventos.append(EVENTOS[int(f["senal"])])
            if al.get("preavisos", True) and f["preaviso"] != 0:
                eventos.append("PREAVISO " + EVENTOS[int(f["preaviso"])])
            if not eventos:
                log(f"{sym}: sin novedades (puntos C/V {int(f['pts_c'])}/{int(f['pts_v'])})")
            for ev in eventos:
                clave = f"{sym}|{tf}|{int(f['ts'])}|{ev}"
                if clave in enviados:
                    continue
                noticias = []
                if al.get("incluir_noticias", True):
                    try:
                        noticias = [x for x in n.obtener_noticias(sym.split("/")[0]) if x["tipo"] != "General"][:4]
                    except Exception:
                        pass
                extra = ""
                try:
                    if ev_modelos is None:
                        mdl.H_ML[tf] = int(cfg.get("horizonte_ml", mdl.H_ML.get(tf, 6)))
                        ev_modelos = mdl.evaluar(d, p, tf)
                    direc = -1 if "VENTA" in ev else 1
                    extra = mdl.html_confirmacion(mdl.confirmacion(ev_modelos, direc), direc)
                except Exception as e:
                    log(f"{sym}: modelos no disponibles ({e})")
                asunto, cuerpo = n.correo_alerta(sym, tf, ev, f, zona, noticias, extra)
                ok, msg = n.enviar_correo(asunto, cuerpo)
                log(f"{sym}: {ev} -> {msg}")
                if ok:
                    enviados[clave] = datetime.now().strftime("%d/%m/%Y %H:%M")
                    guardar_estado(enviados)
        except Exception as e:
            log(f"{simbolo}: error -> {e}")


def segundos_hasta_proximo_cierre(tf: str, margen: int = 45) -> float:
    tf_ms = n.TF_MS[tf]
    ahora = datetime.now(timezone.utc).timestamp() * 1000
    return (tf_ms - ahora % tf_ms) / 1000 + margen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--una-vez", action="store_true", help="Revisar una vez y salir")
    args = ap.parse_args()
    log("Monitor de alertas iniciado. Revisando tu lista…")
    revisar()
    if args.una_vez:
        return
    while True:
        tf = n.cargar_config()["timeframe"]
        espera = segundos_hasta_proximo_cierre(tf)
        log(f"Próxima revisión en {int(espera // 60)} min (al cierre de la vela de {tf}). No cierres esta ventana.")
        time.sleep(espera)
        revisar()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("Monitor detenido.")
