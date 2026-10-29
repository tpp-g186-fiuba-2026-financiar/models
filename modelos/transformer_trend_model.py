"""Modelo Transformer de tendencia, portado a Modal desde api-ml.

Copia de `api-ml/src/transformer.py` con su configuracion de produccion: un
solo modelo para todas las acciones (no uno por ticker) que clasifica si la
accion sube o baja **en dolares CCL** en 5 ruedas.

- Features: las 8 de `api-ml/src/lstm.py` (retornos de precio / volumen,
  rango, RSI, SMA, MACD, momentum y una tasa de interes) mas el cambio a 5
  ruedas del riesgo pais, sobre los precios divididos por el CCL del dia.
  Los retornos y el target van en unidades de volatilidad de cada accion.
- Datos limpios: saltos de precio que el mercado no acompano (splits y canjes
  sin ajustar) y precios <= 0, como `api-ml/src/simulacion/limpieza.py`.
- Clasificacion balanceada: la red da el logit de "sube" y se entrena con BCE
  con `pos_weight` = bajas / subas. El retorno esperado es `(2p - 1) x` el
  movimiento tipico de train.
- Walk-forward: el ultimo 15% de las fechas se parte en 3 bloques; cada uno se
  predice con una red entrenada solo con lo anterior (cortando por fecha para
  todas las acciones, con gap). De ahi salen las metricas y la cantidad de
  epocas; al final se entrena con todas las fechas.
- No predice lo que ya cotiza en dolares (GOLD, OIL y especies `D`).

El reentrenamiento diario (dias habiles) lo dispara el cron del LSTM (ver
`retrain_models`) y guarda el artefacto en un Modal Volume solo si el
walk-forward pasa el piso del gate; el endpoint solo ejecuta inferencia.

Si se cambia la logica en api-ml hay que replicarla aca a mano -- es una
copia independiente a proposito, para que esto funcione aunque el
servicio de api-ml en Render este caido.
"""

import json
import math
import os
import pickle
import time
from datetime import datetime, timezone
from pathlib import Path

import modal
import numpy as np
import pandas as pd
import requests

image = modal.Image.debian_slim().pip_install(
    "fastapi[standard]", "torch", "numpy", "pandas", "requests"
)
app = modal.App("transformer-trend-model")
artifact_volume = modal.Volume.from_name("trend-model-artifacts", create_if_missing=True)
ARTIFACT_ROOT = Path("/artifacts/transformer")

DATA_COLLECTOR_URL = "https://data-colector-k7h6.onrender.com"


def _post_with_retry(url: str, timeout: int = 90, attempts: int = 4) -> requests.Response:
    """POST al data-colector tolerando el arranque en frio de Render.

    El plan gratuito duerme el servicio y despertarlo tarda mas de 30 s: con
    el timeout viejo el cron fallaba sin reintentar. Se reintenta ante
    timeouts, errores de conexion y 5xx (502/503 mientras levanta).
    """
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            response = requests.post(url, timeout=timeout)
            if response.status_code < 500:
                return response
            last_error = requests.HTTPError(f"{response.status_code} en {url}", response=response)
        except (requests.Timeout, requests.ConnectionError) as error:
            last_error = error
        if attempt < attempts - 1:
            time.sleep(5 * 2**attempt)
    raise last_error

def fetch_available_tickers() -> list[str]:
    response = _post_with_retry(f"{DATA_COLLECTOR_URL}/available-tickers")
    response.raise_for_status()
    payload = response.json()
    tickers = payload.get("tickers") or (payload.get("message") or {}).get("tickers") or []
    if not tickers:
        raise ValueError("data-colector no devolvio tickers disponibles")
    return sorted({str(ticker).strip().upper() for ticker in tickers if str(ticker).strip()})
# Las mismas 8 features que `api-ml/src/lstm.py::FEATURE_NAMES` (que usa
# tambien el Transformer de api-ml) y que `xgboost_trend_model.py`, mas el
# riesgo pais (`api-ml/src/transformer.py`, opcion `riesgo_pais`).
FEATURE_NAMES = [
    "log_return",
    "log_volume_change",
    "range_pct",
    "rsi_norm",
    "sma20_ratio",
    "macd_norm",
    "momentum_10",
    "macro_rate_chg5",
    "riesgo_pais_chg5",
]
# Columnas que son retornos: se dividen por la volatilidad del ticker.
VOL_SCALED_FEATURES = ["log_return", "momentum_10"]
# Hiperparametros de `api-ml/src/transformer.py::TransformerTrainConfig`, que
# salen de la busqueda con Optuna de api-ml (`src/simulacion/optuna_transformer.py`,
# trial 17): learning rate bajo y red chica.
WINDOW = 20
D_MODEL = 16
NHEAD = 4
NUM_LAYERS = 2
DIM_FEEDFORWARD = 32
DROPOUT = 0.3593
EPOCHS = 40
BATCH_SIZE = 128
LEARNING_RATE = 1.783e-4
WEIGHT_DECAY = 2.083e-3
WARMUP_EPOCHS = 3
# La mejor epoca sale entre la 1 y la 4.
PATIENCE = 3
VAL_FRACTION = 0.15
# Bloques del walk-forward dentro del ultimo VAL_FRACTION de las fechas.
FOLDS = 3
SEED = 42
# Un solo horizonte: el modelo se entrena con todas las acciones juntas, y
# entrenar uno por horizonte multiplicaria el tiempo del cron. El back y
# api-ml lo piden siempre a 5 ruedas.
DEFAULT_HORIZON = 5
NEUTRAL_BAND = 0.01
# Ruedas de historia por ticker para entrenar (`api-ml` settings.history_days).
HISTORY_DAYS = 750
MIN_RUEDAS = 60
# Volatilidad del ticker (desvio de retornos log diarios) para escalar
# retornos y target.
RUEDAS_VOL = 20
# Un ticker con mas de esta fraccion de ruedas sin cambio de precio no se usa
# (`api-ml/src/data_quality.py`).
STALE_MAX_UNCHANGED = 0.30
# Tasa de interes exogena (ver `fetch_macro_series`): rendimiento a 10 anios
# del Tesoro de EEUU, proxy de apetito por riesgo global que afecta flujos a
# mercados emergentes (Merval incluido). La alternativa local mas relevante
# para acciones argentinas (BCRA: source="ar", series="TPM") dependia de una
# API que estaba caida del lado de data-colector al portar esta feature.
MACRO_RATE_SOURCE = "us"
MACRO_RATE_SERIES = "TNX"
# Series del data-colector: (ruta, serie, retraso de publicacion en dias
# calendario), como `api-ml/src/macro_data.py::SERIES`. El CCL se usa sin
# retraso para pasar precios a dolares (se conoce al cierre del dia).
CCL_SERIES = ("macro/argdatos", "CCL")
RIESGO_PAIS_SERIES = ("macro/argdatos", "RIESGO_PAIS")
RIESGO_PAIS_LAG_DAYS = 2

# Panel lider (`api-ml/src/black_litterman/black_litterman.py`): mercado para
# la limpieza de saltos y excepcion de YPFD (cotiza en pesos aunque termine en D).
PANEL_LIDER_TICKERS = [
    "ALUA", "BBAR", "BMA", "BYMA", "CEPU", "COME", "CRES", "EDN", "GGAL", "IRSA",
    "LOMA", "METR", "PAMP", "SUPV", "TECO2", "TGNO4", "TGSU2", "TRAN", "TXAR", "VALO",
    "YPFD",
]  # fmt: skip
# Commodities del data-colector: futuros de Yahoo, ya en dolares.
COMMODITIES = {"GOLD", "OIL"}
# Retorno log diario a partir del cual se sospecha un salto (x1,8 o /1,8).
UMBRAL_SALTO = math.log(1.8)


def fetch_ticker_history(ticker: str) -> pd.DataFrame:
    resp = _post_with_retry(f"{DATA_COLLECTOR_URL}/historical-data/{ticker}")
    resp.raise_for_status()
    rows = resp.json().get("data") or []
    if not rows:
        raise ValueError(f"data-colector no devolvio velas para {ticker}")
    frame = pd.DataFrame(
        {
            "open": [float(r["open_amount"]) for r in rows],
            "high": [float(r["high_amount"]) for r in rows],
            "low": [float(r["low_amount"]) for r in rows],
            "close": [float(r["close_amount"]) for r in rows],
            "volume": [float(r["volume"]) for r in rows],
        },
        index=pd.to_datetime([int(r["ts"]) for r in rows], unit="ms"),
    )
    return frame.sort_index()


def fetch_collector_series(route: str, series: str) -> pd.Series:
    """Serie diaria del data-colector (`api-ml/src/macro_data.py`).

    El data-colector responde 200 y manda el error en el body, por eso se
    valida ``payload["status"]``.
    """
    resp = _post_with_retry(f"{DATA_COLLECTOR_URL}/{route}/{series}")
    resp.raise_for_status()
    payload = resp.json()
    if payload.get("status") != 200:
        reason = (payload.get("message") or {}).get("error", "error desconocido")
        raise ValueError(f"data-colector no pudo obtener {route}/{series}: {reason}")
    rows = payload.get("data") or []
    if not rows:
        raise ValueError(f"data-colector no devolvio datos para {route}/{series}")
    values = pd.Series(
        [float(r["value"]) for r in rows],
        index=pd.to_datetime([int(r["ts"]) for r in rows], unit="ms").normalize(),
    )
    return values[~values.index.duplicated(keep="last")].sort_index()


def fetch_ccl_y_riesgo_pais() -> tuple[pd.Series, pd.Series]:
    """CCL (sin retraso) y riesgo pais diario con su retraso de publicacion."""
    ccl = fetch_collector_series(*CCL_SERIES)
    riesgo = fetch_collector_series(*RIESGO_PAIS_SERIES)
    dias = pd.date_range(riesgo.index[0], max(riesgo.index[-1], pd.Timestamp.now().normalize()))
    diario = riesgo.reindex(dias.union(riesgo.index)).ffill().reindex(dias)
    return ccl, diario.shift(RIESGO_PAIS_LAG_DAYS)


def fetch_macro_series(source: str = MACRO_RATE_SOURCE, series: str = MACRO_RATE_SERIES):
    """Serie de tasa de interes exogena (identico a `api-ml/src/data.py`).

    `data-colector` responde siempre HTTP 200 y codifica el error real en el
    body (``{"status": 500, "message": {...}}``) en vez de usar el status
    code -- pasa justo con el endpoint de BCRA (``ar``/``TPM``) -- por eso se
    valida ``payload["status"]`` ademas de ``raise_for_status()``. Nunca
    propaga la excepcion: si falla, se loguea y se sigue sin esta feature
    (queda en 0 via ``nan_to_num`` en ``build_features``) en vez de romper
    la prediccion por una serie opcional.
    """
    try:
        resp = _post_with_retry(f"{DATA_COLLECTOR_URL}/interest-rate/{source}/{series}")
        resp.raise_for_status()
        payload = resp.json()
        if payload.get("status") != 200:
            reason = (payload.get("message") or {}).get("error", "error desconocido")
            raise ValueError(f"data-colector no pudo obtener {source}/{series}: {reason}")
        rows = payload.get("data") or []
        if not rows:
            raise ValueError(f"data-colector no devolvio datos para {source}/{series}")
        values = pd.Series(
            [float(r["value"]) for r in rows],
            index=pd.to_datetime([int(r["ts"]) for r in rows], unit="ms"),
        )
        return values[~values.index.duplicated(keep="last")].sort_index()
    except Exception as exc:  # noqa: BLE001 - feature opcional, no debe romper la prediccion
        print(f"  [warn] no se pudo obtener la serie macro ({source}/{series}): {exc}")
        return None


def attach_macro_feature(df: pd.DataFrame, macro) -> pd.DataFrame:
    """Alinea la serie macro al indice de `df` (ffill, nunca un valor futuro)."""
    df = df.copy()
    if macro is None or macro.empty:
        df["macro_rate"] = float("nan")
        return df
    df["macro_rate"] = macro.reindex(df.index, method="ffill")
    return df


# --------------------------------------------------------------------------- #
# Universo y limpieza (`api-ml/src/transformer.py`, `src/simulacion/limpieza.py`)
# --------------------------------------------------------------------------- #
def cotiza_en_dolares(symbol: str) -> bool:
    """Commodities y especies en dolares de BYMA (sufijo ``D``: ALUAD, A3D,
    ...). YPFD es la excepcion: cotiza en pesos."""
    symbol = symbol.strip().upper()
    if symbol in COMMODITIES:
        return True
    return symbol.endswith("D") and symbol not in PANEL_LIDER_TICKERS


def is_stale(df: pd.DataFrame) -> bool:
    """True si el precio no cambia en mas del 30% de las ruedas."""
    if len(df) < 2:
        return True
    return bool((df["close"].diff().dropna() == 0).mean() > STALE_MAX_UNCHANGED)


def sin_precios_no_positivos(df: pd.DataFrame) -> pd.DataFrame:
    """Descarta las ruedas con algun precio <= 0 (no tienen retorno log)."""
    columnas = [c for c in ("open", "high", "low", "close") if c in df.columns]
    return df[(df[columnas] > 0).all(axis=1)]


def retorno_mercado(prices: dict[str, pd.DataFrame]) -> pd.Series:
    """Mediana diaria del retorno log del panel lider (robusta a un salto aislado)."""
    retornos = {}
    for s in PANEL_LIDER_TICKERS:
        if s in prices:
            serie = np.log(prices[s]["close"]).diff()
            serie.index = serie.index.normalize()
            retornos[s] = serie[~serie.index.duplicated(keep="last")]
    return pd.DataFrame(retornos).median(axis=1) if retornos else pd.Series(dtype=float)


def detectar_saltos(df: pd.DataFrame, mercado: pd.Series) -> list[tuple[pd.Timestamp, float]]:
    """Fechas y factores de los saltos de mas de ``UMBRAL_SALTO`` que el mercado
    no acompano (un derrumbe general, como las PASO de 2019, no se toca)."""
    retorno = np.log(df["close"]).diff()
    del_mercado = mercado.reindex(retorno.index.normalize()).fillna(0.0).to_numpy()
    relativo = retorno - del_mercado
    es_salto = (retorno.abs() > UMBRAL_SALTO) & (relativo.abs() > UMBRAL_SALTO)
    return [(fecha, float(np.exp(retorno[fecha]))) for fecha in retorno.index[es_salto]]


def ajustar_salto(df: pd.DataFrame, fecha: pd.Timestamp, factor: float) -> pd.DataFrame:
    """Ajusta las ruedas anteriores a ``fecha`` como un split de ``factor`` (copia)."""
    df = df.copy()
    antes = df.index < fecha
    columnas = [c for c in ("open", "high", "low", "close") if c in df.columns]
    df.loc[antes, columnas] = df.loc[antes, columnas] * factor
    if "volume" in df.columns:
        df["volume"] = df["volume"].astype(float)
        df.loc[antes, "volume"] = df.loc[antes, "volume"] / factor
    return df


def limpiar(df: pd.DataFrame, mercado: pd.Series) -> pd.DataFrame:
    df = sin_precios_no_positivos(df)
    for fecha, factor in detectar_saltos(df, mercado):
        df = ajustar_salto(df, fecha, factor)
    return df


def en_dolares(df: pd.DataFrame, ccl: pd.Series) -> pd.DataFrame:
    """Precios divididos por el ultimo CCL conocido de cada rueda (nunca uno
    posterior); las ruedas anteriores al primer CCL se descartan."""
    serie = ccl.dropna().sort_index()
    serie = serie[~serie.index.duplicated(keep="last")]
    dias = df.index.normalize()
    tipo = serie.reindex(serie.index.union(dias)).ffill().reindex(dias).to_numpy()
    out = df.copy()
    for col in ("open", "high", "low", "close"):
        out[col] = out[col].to_numpy(dtype=np.float64) / tipo
    return out.dropna(subset=["close"])


def preparar(
    df: pd.DataFrame, mercado: pd.Series, ccl: pd.Series, riesgo_pais: pd.Series | None
) -> pd.DataFrame:
    """OHLCV limpio, en dolares y con la columna ``riesgo_pais``."""
    df = en_dolares(limpiar(df, mercado), ccl)
    rp = None if riesgo_pais is None else riesgo_pais.reindex(df.index.normalize())
    df["riesgo_pais"] = np.nan if rp is None else rp.to_numpy()
    return df


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #
def rsi_series(close: np.ndarray, period: int = 14) -> np.ndarray:
    """RSI de Wilder, alineado a `close` (NaN en las primeras `period` posiciones)."""
    n = len(close)
    out = np.full(n, np.nan)
    if n < period + 1:
        return out

    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)

    def _rsi_from_averages(avg_gain: float, avg_loss: float) -> float:
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    avg_gain = gain[:period].mean()
    avg_loss = loss[:period].mean()
    out[period] = _rsi_from_averages(avg_gain, avg_loss)
    for i in range(period, len(delta)):
        avg_gain = (avg_gain * (period - 1) + gain[i]) / period
        avg_loss = (avg_loss * (period - 1) + loss[i]) / period
        out[i + 1] = _rsi_from_averages(avg_gain, avg_loss)
    return out


def sma(values: np.ndarray, window: int) -> np.ndarray:
    """Media movil simple, alineada (NaN en las primeras `window - 1` posiciones)."""
    n = len(values)
    out = np.full(n, np.nan)
    if n < window:
        return out
    cumsum = np.cumsum(np.insert(values, 0, 0.0))
    out[window - 1 :] = (cumsum[window:] - cumsum[:-window]) / window
    return out


def ema(values: np.ndarray, span: int) -> np.ndarray:
    """Media movil exponencial, alineada (arranca en el primer valor, sin NaN)."""
    alpha = 2.0 / (span + 1.0)
    out = np.empty(len(values), dtype=np.float64)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    return out


def macd_histogram(close: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> np.ndarray:
    """Histograma MACD (linea MACD menos su EMA de senal), alineado a `close`."""
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    return macd_line - signal_line


def build_features(df: pd.DataFrame, horizon: int = DEFAULT_HORIZON):
    """Features y target de un OHLCV ya preparado (`preparar`), como
    `api-ml/src/transformer.py::_features`: las 8 de `api-ml/src/lstm.py`
    mas el riesgo pais, con los retornos y el target divididos por la
    volatilidad del ticker. Devuelve ``(feats, target, escala)``; la fila ``j``
    corresponde a la rueda ``df.index[j + 1]``."""
    close = df["close"].to_numpy(dtype=np.float64)
    volume = df["volume"].to_numpy(dtype=np.float64)
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)

    log_close = np.log(close)
    log_return = np.diff(log_close)
    safe_vol = np.where(volume <= 0, np.nan, volume)
    log_volume_change = np.diff(np.log(safe_vol))
    range_pct = ((high - low) / np.where(close == 0, np.nan, close))[1:]

    rsi_norm = (rsi_series(close) - 50.0) / 50.0
    sma20_ratio = close / sma(close, 20) - 1.0
    macd_norm = macd_histogram(close) / np.where(close == 0, np.nan, close)
    momentum_10 = np.full(len(close), np.nan)
    momentum_10[10:] = log_close[10:] - log_close[:-10]

    if "macro_rate" in df.columns:
        macro_rate = df["macro_rate"].to_numpy(dtype=np.float64)
    else:
        macro_rate = np.full(len(close), np.nan)
    macro_chg5 = np.full(len(close), np.nan)
    macro_chg5[5:] = macro_rate[5:] - macro_rate[:-5]

    if "riesgo_pais" in df.columns:
        rp = df["riesgo_pais"].to_numpy(dtype=np.float64)
    else:
        rp = np.full(len(close), np.nan)
    log_rp = np.log(np.where(rp > 0, rp, np.nan))
    rp_chg5 = np.full(len(close), np.nan)
    rp_chg5[5:] = log_rp[5:] - log_rp[:-5]

    feats = np.column_stack(
        [
            log_return,
            log_volume_change,
            range_pct,
            rsi_norm[1:],
            sma20_ratio[1:],
            macd_norm[1:],
            momentum_10[1:],
            macro_chg5[1:],
            rp_chg5[1:],
        ]
    )

    n_feats = len(feats)
    target = np.full(n_feats, np.nan)
    for j in range(n_feats):
        d = j + 1
        if d + horizon < len(close):
            target[j] = log_close[d + horizon] - log_close[d]

    vol = pd.Series(log_close).diff().rolling(RUEDAS_VOL).std().to_numpy()[1:]
    escala = np.where(vol > 0, vol, np.nan)
    target = target / escala
    columnas = [FEATURE_NAMES.index(c) for c in VOL_SCALED_FEATURES]
    feats[:, columnas] = feats[:, columnas] / escala[:, None]
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    return feats, target, escala


def make_windows(feats: np.ndarray, target: np.ndarray, window: int):
    xs, ys = [], []
    for i in range(len(feats) - window):
        y = target[i + window - 1]
        if np.isnan(y):
            continue
        xs.append(feats[i : i + window])
        ys.append(y)
    if not xs:
        return np.empty((0, window, feats.shape[1])), np.empty((0,))
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def ventanas_con_fecha(df: pd.DataFrame, horizon: int = DEFAULT_HORIZON):
    """Ventanas, target y la fecha (solo el dia) de la ultima rueda de cada una."""
    feats, target, _ = build_features(df, horizon)
    X, y = make_windows(feats.astype(np.float32), target, WINDOW)
    validas = [i for i in range(len(feats) - WINDOW) if not np.isnan(target[i + WINDOW - 1])]
    fechas = df.index.normalize()[[i + WINDOW for i in validas]].to_numpy()
    return X, y, fechas


def rsi(close: np.ndarray, period: int = 14):
    """RSI de Wilder sobre el ultimo valor de la serie (post-procesamiento)."""
    series = rsi_series(close, period)
    if len(series) == 0 or np.isnan(series[-1]):
        return None
    return float(series[-1])


def derive_trend_output(df: pd.DataFrame, log_return: float, horizon: int, neutral_band: float = NEUTRAL_BAND) -> dict:
    last_close = float(df["close"].iloc[-1])
    expected_return = float(np.expm1(log_return))
    predicted_close = float(last_close * np.exp(log_return))

    if expected_return > neutral_band:
        signal = "alza"
    elif expected_return < -neutral_band:
        signal = "baja"
    else:
        signal = "neutral"

    rsi_value = rsi(df["close"].to_numpy(dtype=np.float64))
    if rsi_value is None:
        condition = "indeterminado"
    elif rsi_value >= 70:
        condition = "sobrecompra"
    elif rsi_value <= 30:
        condition = "sobreventa"
    else:
        condition = "neutral"

    confidence = float(min(1.0, abs(expected_return) / 0.03))

    return {
        "signal": signal,
        "horizon_days": horizon,
        "expected_return": round(expected_return, 6),
        "predicted_close": round(predicted_close, 4),
        "last_close": round(last_close, 4),
        "rsi": round(rsi_value, 2) if rsi_value is not None else None,
        "condition": condition,
        "confidence": round(confidence, 4),
        "as_of": df.index[-1].strftime("%Y-%m-%d"),
    }


# --------------------------------------------------------------------------- #
# Red y entrenamiento
# --------------------------------------------------------------------------- #
def sinusoidal_encoding(max_len: int, d_model: int):
    """Encoding posicional fijo de "Attention is all you need" (como
    `api-ml/src/transformer.py::_SinusoidalPositionalEncoding`):
    ``PE(p, 2i) = sin(p / 10000^(2i/d))``, ``PE(p, 2i+1) = cos(p / 10000^(2i/d))``.
    La posicion 0 es el token [CLS]; las siguientes, los dias de la ventana."""
    import torch

    posicion = torch.arange(max_len, dtype=torch.float32)[:, None]
    frecuencia = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
    )
    pe = torch.zeros(1, max_len, d_model)
    pe[0, :, 0::2] = torch.sin(posicion * frecuencia)
    pe[0, :, 1::2] = torch.cos(posicion * frecuencia)[:, : d_model // 2]
    return pe


def build_transformer_network():
    import torch
    from torch import nn

    class _TransformerRegressor(nn.Module):
        """Encoder de Transformer con token [CLS] y encoding posicional
        sinusoidal (misma arquitectura que `api-ml/src/transformer.py`); la
        salida es el logit de "sube"."""

        def __init__(self):
            super().__init__()
            self.input_proj = nn.Linear(len(FEATURE_NAMES), D_MODEL)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, D_MODEL))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
            # +1 por el token [CLS] que se antepone a la secuencia. Fijo, no se
            # entrena ni se guarda en el artefacto (ver `sinusoidal_encoding`).
            self.register_buffer("pe", sinusoidal_encoding(WINDOW + 1, D_MODEL), persistent=False)
            layer = nn.TransformerEncoderLayer(
                d_model=D_MODEL, nhead=NHEAD, dim_feedforward=DIM_FEEDFORWARD,
                dropout=DROPOUT, batch_first=True, activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=NUM_LAYERS)
            self.norm = nn.LayerNorm(D_MODEL)
            self.head = nn.Linear(D_MODEL, 1)

        def forward(self, x):
            h = self.input_proj(x) * math.sqrt(D_MODEL)
            cls = self.cls_token.expand(x.size(0), -1, -1)
            h = torch.cat([cls, h], dim=1)
            h = h + self.pe[:, : h.size(1), :]
            h = self.encoder(h)
            return self.head(self.norm(h[:, 0, :])).squeeze(-1)

    return _TransformerRegressor()


def train_transformer(
    X: np.ndarray,
    y: np.ndarray,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    epochs: int | None = None,
) -> tuple[dict, int]:
    """Entrena una red con ventanas ya armadas (``y`` en unidades de vol).

    Con validacion, early stopping por la BCE de validacion; sin validacion,
    ``epochs`` (default ``EPOCHS``). Devuelve el estado (red y escalas) y la
    mejor epoca (1-based).
    """
    import torch
    from torch import nn

    if len(X) < 30:
        raise ValueError("no hay suficientes ventanas para entrenar")
    flat = X.reshape(-1, X.shape[-1])
    feature_mean = flat.mean(axis=0)
    feature_std = flat.std(axis=0) + 1e-8
    # Movimiento tipico (|retorno| medio en unidades de vol): pasa la
    # probabilidad a un retorno esperado (ver `predict_transformer_state`).
    movimiento = float(np.mean(np.abs(y)) + 1e-8)

    def to_tensor(values):
        return torch.tensor(values, dtype=torch.float32)

    X_t = to_tensor((X - feature_mean) / feature_std)
    y_t = to_tensor((y > 0).astype(np.float32))
    has_val = X_val is not None and len(X_val) > 0
    if has_val:
        X_val_t = to_tensor((X_val - feature_mean) / feature_std)
        y_val_t = to_tensor((y_val > 0).astype(np.float32))

    torch.manual_seed(SEED)
    net = build_transformer_network()
    optimizer = torch.optim.AdamW(net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Warmup lineal + decaimiento coseno: lo estandar para estabilizar el
    # entrenamiento de un Transformer, incluso uno chico como este.
    def lr_lambda(epoch: int) -> float:
        if epoch < WARMUP_EPOCHS:
            return (epoch + 1) / WARMUP_EPOCHS
        progress = (epoch - WARMUP_EPOCHS) / max(1, EPOCHS - WARMUP_EPOCHS)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    # BCE balanceada: sin el peso, en un periodo con mas subas aprende "siempre alza".
    subas = float(y_t.sum().item())
    peso = torch.tensor((len(y_t) - subas) / max(subas, 1.0))
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=peso)

    n = len(X_t)
    best_val = float("inf")
    best_state = None
    best_epoch = epochs or EPOCHS
    epochs_without_improvement = 0
    for epoch in range(epochs or EPOCHS):
        net.train()
        perm = torch.randperm(n)
        for start in range(0, n, BATCH_SIZE):
            idx = perm[start : start + BATCH_SIZE]
            optimizer.zero_grad()
            loss = loss_fn(net(X_t[idx]), y_t[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()
        scheduler.step()

        if has_val:
            net.eval()
            with torch.no_grad():
                val_loss = float(loss_fn(net(X_val_t), y_val_t).item())
            if val_loss < best_val - 1e-5:
                best_val = val_loss
                best_state = {k: v.clone() for k, v in net.state_dict().items()}
                best_epoch = epoch + 1
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= PATIENCE:
                    break

    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    state = {
        "net_state": net.state_dict(),
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "movimiento": movimiento,
    }
    return state, best_epoch


def predict_transformer_state(state: dict, windows: np.ndarray) -> np.ndarray:
    """Retorno esperado (en unidades de vol) de un batch de ventanas:
    ``(2p - 1) * movimiento``, con ``p`` la probabilidad balanceada de suba.
    Con ``p`` cerca de 0,5 queda dentro de la banda neutral."""
    import torch

    scaled = (windows - state["feature_mean"]) / state["feature_std"]
    net = build_transformer_network()
    net.load_state_dict(state["net_state"])
    net.eval()
    with torch.no_grad():
        logits = net(torch.tensor(scaled, dtype=torch.float32)).numpy()
    return np.tanh(logits / 2) * state["movimiento"]  # tanh(z / 2) = 2 * sigmoid(z) - 1


# --------------------------------------------------------------------------- #
# Walk-forward y metricas
# --------------------------------------------------------------------------- #
def bloques_walk_forward(fechas: np.ndarray, gap: int = DEFAULT_HORIZON):
    """Bloques ``(fin_train, inicio, fin)`` comunes a todos los tickers.

    El ultimo ``VAL_FRACTION`` de las fechas se parte en ``FOLDS`` bloques
    consecutivos ``[inicio, fin)``; cada uno se entrena con las fechas
    anteriores a ``fin_train``, con ``gap`` fechas descartadas en el medio.
    Cortar por fecha y no por ticker evita que el movimiento del mercado de
    las fechas de validacion de un ticker se filtre por otro.
    """
    fechas = np.unique(fechas)
    n = len(fechas)
    n_val = int(round(n * VAL_FRACTION))
    if n_val == 0:
        return []
    folds = min(FOLDS, n_val)
    inicios = [n - n_val + k * n_val // folds for k in range(folds)]
    if inicios[0] - gap <= 0:
        raise ValueError(f"muy pocas fechas ({n}) para validar {n_val} con gap {gap}")
    finales = [fechas[i] for i in inicios[1:]] + [fechas[-1] + np.timedelta64(1, "D")]
    return [(fechas[i - gap], fechas[i], f) for i, f in zip(inicios, finales, strict=True)]


def acierto_balanceado(pred: np.ndarray, real: np.ndarray) -> float:
    """Promedio del acierto en las subas y en las bajas: decir siempre lo mismo da 0,5."""
    sube = real > 0
    recalls = [np.mean(pred[m] > 0) if c else np.mean(pred[m] <= 0)
               for m, c in ((sube, True), (~sube, False)) if m.any()]  # fmt: skip
    return float(np.mean(recalls)) if recalls else float("nan")


def ic_por_fecha(pred: np.ndarray, real: np.ndarray, fechas: np.ndarray) -> float:
    """Spearman entre prediccion y retorno real de las acciones de cada dia, promediado."""
    df = pd.DataFrame({"pred": pred, "real": real, "fecha": fechas})
    ics = []
    for _, g in df.groupby("fecha"):
        if len(g) >= 3 and g["pred"].nunique() > 1 and g["real"].nunique() > 1:
            ics.append(g["pred"].rank().corr(g["real"].rank()))
    return float(np.mean(ics)) if ics else float("nan")


def entrenar_panel(ventanas: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]) -> dict:
    """Walk-forward + reentreno final con todas las fechas.

    ``ventanas``: ``{ticker: (X, y, fechas)}``. Devuelve ``{"state", "metrics"}``;
    las metricas salen de las predicciones fuera de muestra de cada bloque.
    """
    if not ventanas:
        raise ValueError("no hay suficientes datos para entrenar")

    def juntar(desde, hasta):
        partes = []
        for X, y, f in ventanas.values():
            m = np.ones(len(f), dtype=bool)
            if desde is not None:
                m &= f >= desde
            if hasta is not None:
                m &= f < hasta
            partes.append((X[m], y[m], f[m]))
        return tuple(np.concatenate(cols) for cols in zip(*partes, strict=True))

    todas = np.concatenate([f for _, _, f in ventanas.values()])
    preds, reales, fechas_oos, epocas = [], [], [], []
    for fin_train, inicio, fin in bloques_walk_forward(todas):
        X_tr, y_tr, _ = juntar(None, fin_train)
        X_va, y_va, f_va = juntar(inicio, fin)
        state, mejor = train_transformer(X_tr, y_tr, X_va, y_va)
        epocas.append(mejor)
        preds.append(predict_transformer_state(state, X_va))
        reales.append(y_va)
        fechas_oos.append(f_va)

    n_epocas = max(1, int(np.median(epocas))) if epocas else EPOCHS
    X, y, _ = juntar(None, None)
    state, _ = train_transformer(X, y, epochs=n_epocas)
    metrics = {"folds": len(epocas), "epocas": n_epocas, "observations": 0}
    if preds:
        pred, real = np.concatenate(preds), np.concatenate(reales)
        metrics.update(
            directional_accuracy=float(np.mean(np.sign(pred) == np.sign(real))),
            acierto_balanceado=acierto_balanceado(pred, real),
            ic=ic_por_fecha(pred, real, np.concatenate(fechas_oos)),
            observations=int(len(real)),
        )
    return {"state": state, "metrics": metrics}


def pasa_el_gate(metrics: dict) -> bool:
    """Piso del gate (`api-ml/src/retrain_guard.py`, metrica "walk_forward"):
    acierto balanceado fuera de muestra mayor a 0,5 e IC positivo. No se
    compara contra el vigente: su walk-forward es de otra ventana y la
    diferencia entre dos reentrenos seguidos es ruido."""
    acierto, ic = metrics.get("acierto_balanceado"), metrics.get("ic")
    return acierto is not None and acierto > 0.5 and ic is not None and ic > 0


# --------------------------------------------------------------------------- #
# Reentrenamiento y endpoint
# --------------------------------------------------------------------------- #
def _artifact_path() -> Path:
    return ARTIFACT_ROOT / f"h{DEFAULT_HORIZON}" / "production.pkl"


def historias_para_entrenar(tickers: list[str]) -> dict[str, pd.DataFrame]:
    """Ultimas ``HISTORY_DAYS`` ruedas de cada ticker que el modelo acepta,
    sin las series estancadas o cortas. Un ticker que falla se saltea."""
    historias = {}
    for ticker in tickers:
        if cotiza_en_dolares(ticker):
            continue
        try:
            df = fetch_ticker_history(ticker).iloc[-HISTORY_DAYS:]
        except Exception as exc:  # noqa: BLE001 - se reporta y se sigue con el resto
            print(f"  [skip] {ticker}: {exc}")
            continue
        if len(df) < MIN_RUEDAS or is_stale(df):
            print(f"  [skip] {ticker}: serie corta o estancada")
            continue
        historias[ticker] = df
    return historias


def retrain() -> dict:
    """Entrena con todas las acciones y guarda el artefacto si pasa el gate."""
    historias = historias_para_entrenar(fetch_available_tickers())
    macro = fetch_macro_series()
    ccl, riesgo_pais = fetch_ccl_y_riesgo_pais()
    positivos = {s: sin_precios_no_positivos(df) for s, df in historias.items()}
    mercado = retorno_mercado(positivos)
    ventanas = {}
    for ticker, df in historias.items():
        preparado = preparar(attach_macro_feature(df, macro), mercado, ccl, riesgo_pais)
        X, y, fechas = ventanas_con_fecha(preparado)
        if len(X) >= 30:
            ventanas[ticker] = (X, y, fechas)
    resultado = entrenar_panel(ventanas)
    metrics = resultado["metrics"]
    promoted = pasa_el_gate(metrics)
    if promoted:
        artifact = {
            "state": resultado["state"],
            "metrics": metrics,
            "mercado": mercado,
            "tickers": sorted(ventanas),
            "horizon": DEFAULT_HORIZON,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "training_data_as_of": str(max(df.index[-1] for df in historias.values()).date()),
        }
        path = _artifact_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        with temporary.open("wb") as fh:
            pickle.dump(artifact, fh)
        os.replace(temporary, path)
        artifact_volume.commit()
    return {"promoted": promoted, "tickers": len(ventanas), "metrics": metrics}


# Sin `schedule=` propio a proposito: el plan gratis de Modal permite 5
# funciones programadas por workspace y ya estan ocupadas (LSTM, XGBoost,
# SVM, ARIMA y GARCH). El job diario del LSTM (`lstm_trend_model.py`,
# 20:00 ART) dispara esta funcion con `.spawn()`, asi se reentrena igual
# cada dia habil sin gastar un cron. A mano: `modal run
# modelos/transformer_trend_model.py::retrain_models`.
@app.function(
    image=image, volumes={"/artifacts": artifact_volume},
    timeout=7200, cpu=2.0, memory=4096,
)
def retrain_models() -> dict:
    """Un solo modelo para todas las acciones de `fetch_available_tickers()`."""
    try:
        result = retrain()
    except Exception as exc:  # noqa: BLE001 - el cron no tiene que morir sin dejar rastro
        result = {"promoted": False, "error": str(exc)}
    print(json.dumps(result))
    return result


# Recursos del contenedor: la red es chica (d_model 16) y predice una sola
# ventana, asi que no se beneficia de mas nucleos. No se fija
# `min_containers` (default 0, escala a cero sin trafico): se prioriza no
# pagar por contenedores idle antes que eliminar el cold start.
@app.function(
    image=image, timeout=60, cpu=1.0, memory=512,
    volumes={"/artifacts": artifact_volume},
)
@modal.fastapi_endpoint()
def main(ticker: str, horizon: int = DEFAULT_HORIZON):
    if horizon != DEFAULT_HORIZON:
        return {"error": f"horizon solo puede ser {DEFAULT_HORIZON} dias (se pidio {horizon})"}

    ticker = ticker.strip().upper()
    if cotiza_en_dolares(ticker):
        return {"error": f"el transformer no predice '{ticker}' (ya cotiza en dolares)"}
    artifact_volume.reload()
    path = _artifact_path()
    if not path.exists():
        return {"error": "todavia no hay un modelo entrenado"}
    with path.open("rb") as fh:
        artifact = pickle.load(fh)

    df = fetch_ticker_history(ticker).iloc[-HISTORY_DAYS:]
    if len(df) < WINDOW + RUEDAS_VOL + 30:
        return {"error": f"se necesitan mas ruedas historicas para '{ticker}' (hay {len(df)})"}

    try:
        ccl, riesgo_pais = fetch_ccl_y_riesgo_pais()
    except Exception as exc:  # noqa: BLE001 - sin CCL no hay precio en dolares
        return {"error": f"no se pudo obtener el CCL o el riesgo pais: {exc}"}
    df_macro = attach_macro_feature(df, fetch_macro_series())
    preparado = preparar(df_macro, artifact["mercado"], ccl, riesgo_pais)
    feats, _, escala = build_features(preparado, horizon)
    if len(feats) < WINDOW or not np.isfinite(escala[-1]):
        return {"error": f"se necesitan mas ruedas historicas para '{ticker}' (hay {len(df)})"}
    expected = float(predict_transformer_state(artifact["state"], feats[-WINDOW:][None, :, :])[0])

    # Retorno en dolares: el precio esperado se informa en pesos, a CCL constante.
    result = derive_trend_output(df, expected * float(escala[-1]), horizon)
    result["ticker"] = ticker
    result["model"] = "transformer"
    result["source"] = "modal (artefacto preentrenado)"
    result["model_version"] = artifact["trained_at"]
    result["backtest"] = artifact["metrics"]
    return result
