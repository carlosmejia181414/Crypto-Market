# Analizador Cripto

Tu propio panel de análisis, parecido a TradingView pero sin límites: gráfico en vivo, señales de compra y venta, escáner, backtesting, noticias y alertas por correo.

## 1. Instalar Python (solo una vez)
1. Descarga Python desde https://www.python.org/downloads/
2. **Windows:** al instalar, marca la casilla **"Add Python to PATH"**.

## 2. Abrir la web
1. Descomprime la carpeta `analizador_cripto` donde quieras (por ejemplo, en Documentos).
2. Abre la web:
   - **Windows:** doble clic en `iniciar_web.bat`.
   - **Mac:** doble clic en `iniciar_web.command`. Si el Mac no lo deja abrir: clic derecho → Abrir.
3. La primera vez tarda unos minutos porque instala las librerías. Después se abre sola en tu navegador (http://localhost:8501).

Pestañas:
- **Gráfico:** velas en vivo con las EMA, señales (▲ compra, ▼ venta), líneas de SL y TP, preavisos (●), volumen, RSI, MACD y el panel de estado.
- **Modelos:** lectura matemática de la cripto (ver abajo).
- **Escáner:** todas tus criptos en una tabla.
- **Backtest:** cómo le habría ido a la estrategia en el pasado. Cambia los parámetros en la barra lateral y compara.
- **Noticias:** titulares de CoinDesk, Cointelegraph, Decrypt y Bitcoin Magazine, filtrados por moneda.
- **Alertas:** aquí configuras tu correo y tu lista.

### Modelos matemáticos (pestaña Modelos)
| Modelo | Qué responde |
|---|---|
| **Machine learning** (Gradient Boosting + Regresión logística, validado *walk-forward*) | ¿Qué probabilidad hay de que suba en las próximas velas? Solo cuenta si demuestra ventaja sobre el azar con 95% de confianza en datos que no vio. |
| **Filtro de Kalman** | ¿Cuál es la pendiente real de la tendencia sin el ruido, y con qué confianza? |
| **Régimen de mercado** (mezcla gaussiana) | ¿El mercado está alcista, lateral o bajista? |
| **Exponente de Hurst** | ¿Los movimientos tienden a continuar o a revertir? |
| **GARCH(1,1) + Monte Carlo** (4.000 simulaciones) | ¿Cuánto se puede mover? ¿Qué probabilidad hay de tocar primero el TP o el SL? (cono azul del gráfico) |

Cada modelo vota **a favor**, **en contra** o **neutral** de la señal. El resultado es una **confirmación Fuerte, Moderada o Débil**, que también llega en los correos de alerta.

Ningún modelo predice el precio exacto; dan probabilidades. Usa el **Backtest** con la opción *Filtrar con el modelo ML* para ver si el filtro mejora los resultados en cada cripto. Los modelos necesitan bastante historia: deja 1500 velas o más. Kraken solo entrega 720, así que para los modelos usa Binance u OKX.

## 3. Alertas por correo (Gmail)
1. Activa la **verificación en 2 pasos** en tu cuenta de Google.
2. Entra a https://myaccount.google.com/apppasswords y crea una contraseña de aplicación con el nombre "Analizador". Copia las 16 letras.
3. En la pestaña **Alertas** pon tu Gmail y esa contraseña, pulsa **Guardar** y después **Enviar correo de prueba**.
4. Enciende el monitor:
   - **Windows:** doble clic en `iniciar_monitor.bat`.
   - **Mac:** doble clic en `iniciar_monitor.command`.

   Deja esa ventana abierta, aunque sea minimizada.

El monitor revisa tu lista al cierre de cada vela de 4h (hora de Toronto: 19:00, 23:00, 03:00, 07:00, 11:00 y 15:00) y te escribe cuando hay:
- **Preaviso:** a la señal le falta una sola condición. Prepárate y revisa noticias.
- **COMPRA / VENTA:** señal confirmada, con entrada, stop loss y take profit.

La computadora debe estar encendida y sin suspenderse para que el monitor funcione.

## 4. Fuente de datos
Por defecto se usan los precios públicos de **Binance**. Si esa fuente no carga desde tu conexión, cambia a **Kraken** u **OKX** en la barra lateral y pulsa "Guardar como predeterminado". Solo se leen precios públicos: no se necesita cuenta ni claves.

## Archivos
| Archivo | Para qué sirve |
|---|---|
| `app.py` | La web |
| `monitor.py` | Envía las alertas |
| `nucleo.py` | Indicadores, señales, backtest, noticias y correo |
| `config.json` | Tu lista y tus parámetros (se edita desde la web) |
| `.env` | Tu correo y tu contraseña de aplicación. **No lo compartas.** |

> Herramienta de apoyo para tus propios análisis. Ningún indicador garantiza resultados. No es asesoría financiera.
