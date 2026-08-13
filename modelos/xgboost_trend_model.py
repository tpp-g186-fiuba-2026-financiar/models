"""Modelo XGBoost de tendencia, portado a Modal desde api-ml.

Mismo criterio que `lstm_trend_model.py` en este mismo directorio: misma
logica de features/target base que `api-ml/src/xgb_trend.py` (retornos log
de precio/volumen + rango %, indicadores tecnicos -- RSI/SMA/MACD/momentum
-- y una tasa de interes exogena). Un cron diario en dias habiles hace backtest, promocion
y persistencia en un Modal Volume; el endpoint solo ejecuta inferencia.

A diferencia de `lstm_trend_model.py`, este archivo se queda con las 8
features de siempre (no suma `dist_max60`/`rsi_vol_interaction`): se
probaron ahi tambien (issue #160) y dieron peor, no mejor -- ver la nota
junto a `FEATURE_NAMES` mas abajo.

Si se cambia la logica en api-ml hay que replicarla aca a mano -- es una
copia independiente a proposito, para que esto funcione aunque el
servicio de api-ml en Render este caido.
"""

import json
import os
import pickle
from datetime import datetime, timezone
from pathlib import Path

import modal
import numpy as np
import pandas as pd
import requests

image = (
    modal.Image.debian_slim()
    # xgboost necesita OpenMP en runtime (mismo motivo por el que el
    # Dockerfile de api-ml instala libgomp1). Las versiones nuevas de
    # xgboost (3.x) ademas requieren scikit-learn instalado para poder usar
    # la API sklearn-compatible (XGBRegressor), aunque no se importe directo.
    .apt_install("libgomp1")
    .pip_install("fastapi[standard]", "xgboost", "scikit-learn", "numpy", "pandas", "requests")
)
app = modal.App("xgboost-trend-model")
artifact_volume = modal.Volume.from_name("trend-model-artifacts", create_if_missing=True)
ARTIFACT_ROOT = Path("/artifacts/xgboost")
BACKTEST_DAYS = 60

DATA_COLLECTOR_URL = "https://data-colector.onrender.com"


def fetch_available_tickers() -> list[str]:
    response = requests.post(f"{DATA_COLLECTOR_URL}/available-tickers", timeout=30)
    response.raise_for_status()
    payload = response.json()
    tickers = payload.get("tickers") or (payload.get("message") or {}).get("tickers") or []
    if not tickers:
        raise ValueError("data-colector no devolvio tickers disponibles")
    return sorted({str(ticker).strip().upper() for ticker in tickers if str(ticker).strip()})
FEATURE_NAMES = [
    "log_return",
    "log_volume_change",
    "range_pct",
    "rsi_norm",
    "sma20_ratio",
    "macd_norm",
    "momentum_10",
    "macro_rate_chg5",
]
# WINDOW sale de una busqueda con backtest walk-forward (issue #160,
# `scripts/tune_hyperparams.py`, resultados en `scripts/TUNING_RESULTS.md`)
# sobre GGAL/YPFD/ALUA -- antes era un valor a ojo (30, que resulto ser el
# mejor igual). La arquitectura actual (300 arboles, profundidad 4) tambien
# ya andaba bien, no hizo falta tocarla. Se probaron las 2 features nuevas
# que sí ayudaron al LSTM (`dist_max60`/`rsi_vol_interaction`, ver
# `scripts/tune_features_and_horizon.py`) pero ACA dieron peor (52.5% contra
# 54.7% de acierto sin ellas) -- por eso este archivo se queda con las 8
# features de siempre, a proposito. Ojo: la busqueda uso solo 3 tickers y
# ~570-590 predicciones -- prometedor pero no concluyente.
WINDOW = 30
# El horizonte de prediccion es un parametro de `main()`, no una constante
# fija -- ver la nota mas larga en lstm_trend_model.py sobre por que (10
# dias predecia mejor pero es poco util para el cliente final; se probo 1-5
# y el acierto crece con el horizonte tambien ahi, asi que el default es 5).
DEFAULT_HORIZON = 5
MIN_HORIZON = 1
MAX_HORIZON = 5
NEUTRAL_BAND = 0.01
# Tasa de interes exogena (ver `fetch_macro_series`): rendimiento a 10 anios
# del Tesoro de EEUU, proxy de apetito por riesgo global que afecta flujos a
# mercados emergentes (Merval incluido). La alternativa local mas relevante
# para acciones argentinas (BCRA: source="ar", series="TPM") dependia de una
# API que estaba caida del lado de data-colector al portar esta feature.
MACRO_RATE_SOURCE = "us"
MACRO_RATE_SERIES = "TNX"


def fetch_ticker_history(ticker: str) -> pd.DataFrame:
    resp = requests.post(f"{DATA_COLLECTOR_URL}/historical-data/{ticker}", timeout=30)
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


# Umbral para detectar un salto de precio que casi seguro es un error de la
# fuente, no un movimiento real de mercado (EDA issue #135: ECOG cayo ~90%
# en un dia -- log_return -2.31 -- con mucha menos historia que el resto,
# huele a split mal ajustado o dato roto de Yahoo). El peor dia REAL que
# vimos en el EDA fue un dia de elecciones, log_return ~0.34 -- este umbral
# queda comodo por encima de eso y muy por debajo del caso roto, para no
# tocar movimientos grandes pero legitimos.
BAD_DATA_LOG_RETURN_THRESHOLD = 1.0


def clean_close_series(
    close: np.ndarray, threshold: float = BAD_DATA_LOG_RETURN_THRESHOLD
) -> tuple[np.ndarray, int]:
    """Repara saltos de precio que son casi seguro un error de dato, en vez
    de descartar el ticker entero: capa (winsoriza) cualquier retorno diario
    mas grande que `threshold` en valor absoluto y reconstruye la serie de
    precios a partir de esos retornos ya capados. Asi todos los indicadores
    que se calculan mas abajo (SMA, RSI, MACD, etc.) quedan sobre una serie
    sana, en vez de arrastrar el salto por 20-60 ruedas. Devuelve la serie
    reparada y cuantos dias se tuvieron que capar (0 en el caso normal, sin
    nada que limpiar)."""
    log_close = np.log(close)
    log_return = np.diff(log_close)
    clipped = np.clip(log_return, -threshold, threshold)
    fixed = int(np.sum(clipped != log_return))
    log_close_clean = np.concatenate([[log_close[0]], log_close[0] + np.cumsum(clipped)])
    return np.exp(log_close_clean), fixed


def clean_dataframe(df: pd.DataFrame, threshold: float = BAD_DATA_LOG_RETURN_THRESHOLD) -> tuple[pd.DataFrame, int]:
    """Aplica `clean_close_series` sobre la columna `close` de todo el
    historico, una sola vez -- asi todo lo que consuma `df` despues (las
    features, el entrenamiento, el ultimo precio que se le muestra al
    cliente, el RSI de post-procesamiento) ve la misma serie ya reparada."""
    cleaned, fixed = clean_close_series(df["close"].to_numpy(dtype=np.float64), threshold)
    df = df.copy()
    df["close"] = cleaned
    return df, fixed


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
        resp = requests.post(f"{DATA_COLLECTOR_URL}/interest-rate/{source}/{series}", timeout=30)
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
    """Identico a `api-ml/src/lstm.py::build_features` (LSTM/XGBoost/Transformer
    en api-ml comparten el mismo feature engineering)."""
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
        ]
    )

    n_feats = len(feats)
    target = np.full(n_feats, np.nan)
    for j in range(n_feats):
        d = j + 1
        if d + horizon < len(close):
            target[j] = log_close[d + horizon] - log_close[d]

    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    return feats, target


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


def flatten_windows(feats: np.ndarray, target: np.ndarray, window: int):
    X, y = make_windows(feats, target, window)
    if len(X) == 0:
        return X, y
    return X.reshape(len(X), -1), y


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


@app.function()
def train_xgboost(df: pd.DataFrame, horizon: int = DEFAULT_HORIZON) -> dict:
    import xgboost as xgb

    feats, target = build_features(df, horizon)
    X, y = flatten_windows(feats, target, WINDOW)
    if len(X) < 30:
        raise ValueError(
            f"no hay suficientes ruedas para entrenar (se necesitan al menos ~{WINDOW + horizon + 30})"
        )

    booster = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective="reg:squarederror",
    )
    booster.fit(X, y, verbose=False)

    return {"booster": booster}


def backtest_xgboost(df: pd.DataFrame, horizon: int) -> dict:
    """Holdout temporal: el candidato nunca ve las ultimas ruedas evaluadas."""
    split = len(df) - BACKTEST_DAYS - horizon
    if split < WINDOW + horizon + 30:
        raise ValueError("no hay suficiente historia para el backtest temporal")
    state = train_xgboost.local(df.iloc[:split], horizon)
    feats, target = build_features(df, horizon)
    start = max(WINDOW, split - 1)
    predicted, actual = [], []
    for feature_day in range(start, len(feats)):
        if np.isnan(target[feature_day]):
            continue
        window = feats[feature_day - WINDOW + 1 : feature_day + 1].reshape(1, -1)
        predicted.append(float(state["booster"].predict(window)[0]))
        actual.append(float(target[feature_day]))
    if not actual:
        raise ValueError("el backtest no produjo observaciones")
    predicted_arr, actual_arr = np.asarray(predicted), np.asarray(actual)
    return {
        "directional_accuracy": float(np.mean(np.sign(predicted_arr) == np.sign(actual_arr))),
        "mae": float(np.mean(np.abs(predicted_arr - actual_arr))),
        "observations": len(actual),
    }


def _artifact_path(ticker: str, horizon: int) -> Path:
    return ARTIFACT_ROOT / ticker / f"h{horizon}" / "production.pkl"


def retrain_one(ticker: str, horizon: int) -> dict:
    ticker = ticker.strip().upper()
    df, repaired = clean_dataframe(fetch_ticker_history(ticker))
    df = attach_macro_feature(df, fetch_macro_series())
    metrics = backtest_xgboost(df, horizon)
    path = _artifact_path(ticker, horizon)
    incumbent_score = -1.0
    if path.exists():
        with path.open("rb") as fh:
            incumbent_score = float(pickle.load(fh)["metrics"]["directional_accuracy"])
    promoted = metrics["directional_accuracy"] > incumbent_score
    if promoted:
        artifact = {
            "state": train_xgboost.local(df, horizon),
            "metrics": metrics,
            "ticker": ticker,
            "horizon": horizon,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "training_data_as_of": df.index[-1].strftime("%Y-%m-%d"),
            "price_data_repaired_days": repaired,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        with temporary.open("wb") as fh:
            pickle.dump(artifact, fh)
        os.replace(temporary, path)
        artifact_volume.commit()
    return {"ticker": ticker, "horizon": horizon, "promoted": promoted, "metrics": metrics}


@app.function(
    image=image,
    volumes={"/artifacts": artifact_volume},
    schedule=modal.Cron("0 21 * * 1-5", timezone="America/Argentina/Buenos_Aires"),
    timeout=86400,
    cpu=1.0,
    memory=1024,
)
def retrain_models() -> list[dict]:
    """Reentrena cada dia habil. Tambien se puede ejecutar a mano desde Modal."""
    results = []
    for ticker in fetch_available_tickers():
        for horizon in range(MIN_HORIZON, MAX_HORIZON + 1):
            try:
                results.append(retrain_one(ticker, horizon))
            except Exception as exc:  # un ticker no cancela el resto del lote
                results.append({"ticker": ticker, "horizon": horizon, "error": str(exc)})
    print(json.dumps(results))
    return results


@app.function(
    image=image, volumes={"/artifacts": artifact_volume}, timeout=3600,
    cpu=1.0, memory=1024,
)
@modal.fastapi_endpoint()
def prepare(ticker: str) -> dict:
    """Bootstrap bajo demanda: crea solamente los horizontes que faltan."""
    ticker = ticker.strip().upper()
    artifact_volume.reload()
    missing = [
        horizon for horizon in range(MIN_HORIZON, MAX_HORIZON + 1)
        if not _artifact_path(ticker, horizon).exists()
    ]
    if not missing:
        return {"ticker": ticker, "status": "ready", "trained_horizons": []}
    results = []
    for horizon in missing:
        try:
            results.append(retrain_one(ticker, horizon))
        except Exception as exc:
            results.append({"ticker": ticker, "horizon": horizon, "error": str(exc)})
    return {"ticker": ticker, "status": "completed", "results": results}


# Recursos del contenedor: 300 arboles chicos (max_depth=4) sobre ~700 filas
# de un solo ticker. No se benchmarkeo este archivo puntual, pero se probo el
# mismo tipo de comparacion en lstm_trend_model.py (cpu=1 vs cpu=2) y mas
# CPU dio *peor* tiempo de respuesta para un modelo de este tamaño -- mismo
# criterio aca, no hay motivo para esperar algo distinto con 300 arboles chicos.
# No se fija `min_containers` (default 0, escala a cero sin trafico): se
# prioriza no pagar por contenedores idle antes que eliminar el cold start.
@app.function(
    image=image,
    timeout=60,
    cpu=1.0,
    memory=512,
    volumes={"/artifacts": artifact_volume},
)
@modal.fastapi_endpoint()
def main(ticker: str, horizon: int = DEFAULT_HORIZON):
    if not (MIN_HORIZON <= horizon <= MAX_HORIZON):
        return {"error": f"horizon debe estar entre {MIN_HORIZON} y {MAX_HORIZON} dias (se pidio {horizon})"}

    ticker = ticker.strip().upper()
    artifact_volume.reload()
    path = _artifact_path(ticker, horizon)
    if not path.exists():
        return {"error": f"todavia no hay un modelo entrenado para '{ticker}' con horizonte {horizon}"}
    with path.open("rb") as fh:
        artifact = pickle.load(fh)

    df = fetch_ticker_history(ticker)
    if len(df) < WINDOW + horizon + 30:
        return {"error": f"se necesitan mas ruedas historicas para '{ticker}' (hay {len(df)})"}
    df, days_repaired = clean_dataframe(df)

    macro = fetch_macro_series()
    df = attach_macro_feature(df, macro)

    feats, _ = build_features(df, horizon)
    window = feats[-WINDOW:].reshape(1, -1)
    log_return = float(artifact["state"]["booster"].predict(window)[0])

    result = derive_trend_output(df, log_return, horizon)
    result["ticker"] = ticker
    result["model"] = "xgboost"
    result["source"] = "modal (artefacto preentrenado)"
    result["model_version"] = artifact["trained_at"]
    result["backtest"] = artifact["metrics"]
    if days_repaired:
        result["price_data_repaired_days"] = days_repaired
    return result
