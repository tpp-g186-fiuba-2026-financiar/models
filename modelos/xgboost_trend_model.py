"""Modelo XGBoost de tendencia, portado a Modal desde api-ml.

Mismo criterio que `lstm_trend_model.py` en este mismo directorio: misma
logica de features/target que `api-ml/src/xgb_trend.py`, entrenado
on-demand con los datos del ticker pedido -- igual que el resto de
`modelos/*.py` en este repo (arima/garch/svm), sin pooling entre tickers
ni artefactos persistidos.

Si se cambia la logica en api-ml hay que replicarla aca a mano -- es una
copia independiente a proposito, para que esto funcione aunque el
servicio de api-ml en Render este caido.
"""

import modal
import numpy as np
import pandas as pd
import requests

image = (
    modal.Image.debian_slim()
    # xgboost necesita OpenMP en runtime (mismo motivo por el que el
    # Dockerfile de api-ml instala libgomp1).
    .apt_install("libgomp1")
    .pip_install("fastapi[standard]", "xgboost", "numpy", "pandas", "requests")
)
app = modal.App("xgboost-trend-model")

DATA_COLLECTOR_URL = "https://data-colector.onrender.com"
FEATURE_NAMES = ["log_return", "log_volume_change", "range_pct"]
WINDOW = 30
HORIZON = 5
NEUTRAL_BAND = 0.01


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


def build_features(df: pd.DataFrame, horizon: int = HORIZON):
    close = df["close"].to_numpy(dtype=np.float64)
    volume = df["volume"].to_numpy(dtype=np.float64)
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)

    log_close = np.log(close)
    log_return = np.diff(log_close)
    safe_vol = np.where(volume <= 0, np.nan, volume)
    log_volume_change = np.diff(np.log(safe_vol))
    range_pct = ((high - low) / np.where(close == 0, np.nan, close))[1:]

    feats = np.column_stack([log_return, log_volume_change, range_pct])

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
    if len(close) < period + 1:
        return None
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = gain[:period].mean()
    avg_loss = loss[:period].mean()
    for i in range(period, len(delta)):
        avg_gain = (avg_gain * (period - 1) + gain[i]) / period
        avg_loss = (avg_loss * (period - 1) + loss[i]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100.0 - (100.0 / (1.0 + rs)))


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
def train_xgboost(df: pd.DataFrame) -> dict:
    import xgboost as xgb

    feats, target = build_features(df)
    X, y = flatten_windows(feats, target, WINDOW)
    if len(X) < 30:
        raise ValueError(
            f"no hay suficientes ruedas para entrenar (se necesitan al menos ~{WINDOW + HORIZON + 30})"
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


@app.function(image=image, timeout=300)
@modal.fastapi_endpoint()
def main(ticker: str):
    df = fetch_ticker_history(ticker)
    if len(df) < WINDOW + HORIZON + 30:
        return {"error": f"se necesitan mas ruedas historicas para '{ticker}' (hay {len(df)})"}

    state = train_xgboost.local(df)

    feats, _ = build_features(df)
    window = feats[-WINDOW:].reshape(1, -1)
    log_return = float(state["booster"].predict(window)[0])

    result = derive_trend_output(df, log_return, HORIZON)
    result["ticker"] = ticker.strip().upper()
    result["model"] = "xgboost"
    result["source"] = "modal (entrenado en el contenedor, solo con el ticker pedido)"
    return result
