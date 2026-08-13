"""Modelo LSTM de tendencia, portado a Modal desde api-ml.

Parte de las mismas features y el mismo target que `api-ml/src/lstm.py`
(retornos log de precio/volumen + rango %, indicadores tecnicos --
RSI/SMA/MACD/momentum -- y una tasa de interes exogena; target = retorno
log acumulado a `horizon` ruedas, elegible por quien pide la prediccion
entre `MIN_HORIZON` y `MAX_HORIZON`), mas 2 features propias de este repo
que salieron del EDA (ver `FEATURE_NAMES`). Entrenado on-demand con los
datos de cada ticker configurado. Un cron diario en dias habiles hace backtest, promocion
y persistencia en un Modal Volume; el endpoint solo ejecuta inferencia.

(api-ml en cambio entrena un unico modelo global pooleando los tickers
disponibles -- acá se prioriza consistencia con el resto de este repo por
sobre replicar ese diseño puntual.)

Si se cambia la logica de features/target en api-ml hay que replicar el
cambio aca a mano -- son dos copias independientes a proposito, para que
esto funcione aunque el servicio de api-ml en Render este caido.
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

image = modal.Image.debian_slim().pip_install(
    "fastapi[standard]", "torch", "numpy", "pandas", "requests"
)
app = modal.App("lstm-trend-model")
artifact_volume = modal.Volume.from_name("trend-model-artifacts", create_if_missing=True)
ARTIFACT_ROOT = Path("/artifacts/lstm")
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
# Las 2 ultimas salen del EDA (issue #135, api-ml/notebooks/eda.ipynb,
# secciones 13.b/13.c) y se validaron con backtest walk-forward (issue #160,
# `scripts/tune_features_and_horizon.py`): sumarlas subio el acierto
# direccional de 50.4% a 58.3% en LSTM. En XGBoost dieron peor (por eso ese
# archivo se quedo con las 8 de siempre, ver xgboost_trend_model.py) -- no
# son "mejores porque si", ayudan a este modelo en particular.
FEATURE_NAMES = [
    "log_return",
    "log_volume_change",
    "range_pct",
    "rsi_norm",
    "sma20_ratio",
    "macd_norm",
    "momentum_10",
    "macro_rate_chg5",
    "dist_max60",
    "rsi_vol_interaction",
]
# WINDOW/hidden_size/num_layers salen de una busqueda con backtest
# walk-forward (issue #160, `scripts/tune_hyperparams.py`, resultados en
# `scripts/TUNING_RESULTS.md`) sobre GGAL/YPFD/ALUA -- antes eran valores a
# ojo (30, 1 capa). 2 capas le gano claro a 1 (63.3% vs 56.8% de acierto
# direccional). Ojo: la busqueda uso solo 3 tickers y ~570 predicciones --
# prometedor pero no concluyente, revisar si con mas tickers/mas historia
# se sostiene.
WINDOW = 45
# El horizonte de prediccion es un parametro de `main()`, no una constante
# fija -- un horizonte de 10 dias predecia mejor en el backtest, pero para
# el cliente final es poco util ("¿en 10 dias, subio o bajo?" dice poco el
# dia a dia). Se probo 1-5 dias (rango util) y el acierto crece con el
# horizonte dentro de ese rango tambien (51% a 1 dia, 58% a 5) -- por eso el
# default es 5, pero queda a eleccion de quien pida la prediccion. A cambio,
# a 1 dia el error de magnitud es mucho mas chico (0.023 vs 0.070 a 5 dias)
# aunque acierte la direccion casi como moneda al aire.
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
    que se calculan mas abajo (SMA, RSI, MACD, distancia al maximo, etc.)
    quedan sobre una serie sana, en vez de arrastrar el salto por 20-60
    ruedas. Devuelve la serie reparada y cuantos dias se tuvieron que capar
    (0 en el caso normal, sin nada que limpiar)."""
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
    cliente, el RSI de post-procesamiento) ve la misma serie ya reparada, en
    vez de que cada lugar tenga que acordarse de limpiarla por su cuenta."""
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
    """Retornos log de precio/volumen + rango %, indicadores tecnicos
    (RSI/SMA/MACD/momentum), tasa macro exogena, y 2 features del EDA
    (issue #135): distancia al maximo de 60 ruedas y la interaccion
    RSI x volumen. Target = retorno log acumulado a `horizon` ruedas.
    """
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

    # dist_max60: que tan lejos esta el precio de hoy del maximo de los
    # ultimos 60 dias (0 = esta en el maximo, negativo = por debajo). En el
    # EDA fue la feature con mas señal de las 10 -- capta rebotes despues de
    # una caida fuerte, algo que RSI/SMA20 no ven porque miran ventanas mas
    # cortas (14 y 20 dias).
    roll_max60 = df["close"].rolling(60, min_periods=30).max().to_numpy(dtype=np.float64)
    dist_max60 = close / np.where(roll_max60 == 0, np.nan, roll_max60) - 1.0

    # rsi_vol_interaction: RSI normalizado x volumen "raro" (z-score contra
    # el promedio propio de 60 dias). En el EDA, "volumen alto + RSI en zona
    # media" predecia mas que cualquiera de las dos por separado.
    vol_series = df["volume"].astype(float)
    roll_mean_vol = vol_series.rolling(60, min_periods=20).mean()
    roll_std_vol = vol_series.rolling(60, min_periods=20).std()
    vol_z = ((vol_series - roll_mean_vol) / roll_std_vol.replace(0, np.nan)).to_numpy(dtype=np.float64)
    rsi_vol_interaction = rsi_norm * np.nan_to_num(vol_z, nan=0.0, posinf=0.0, neginf=0.0)

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
            dist_max60[1:],
            rsi_vol_interaction[1:],
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


def build_lstm_network():
    import torch
    from torch import nn

    class _LSTMRegressor(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=len(FEATURE_NAMES), hidden_size=32, num_layers=2,
                batch_first=True, dropout=0.1,
            )
            self.head = nn.Linear(32, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(out[:, -1, :]).squeeze(-1)

    return _LSTMRegressor()


@app.function()
def train_lstm(df: pd.DataFrame, horizon: int = DEFAULT_HORIZON) -> dict:
    import torch
    from torch import nn

    feats, target = build_features(df, horizon)
    X, y = make_windows(feats, target, WINDOW)
    if len(X) < 30:
        raise ValueError(
            f"no hay suficientes ruedas para entrenar (se necesitan al menos ~{WINDOW + horizon + 30})"
        )

    flat = X.reshape(-1, X.shape[-1])
    feature_mean = flat.mean(axis=0)
    feature_std = flat.std(axis=0) + 1e-8
    target_mean = float(y.mean())
    target_std = float(y.std() + 1e-8)

    X_scaled = (X - feature_mean) / feature_std
    X_t = torch.tensor(X_scaled, dtype=torch.float32)
    y_t = torch.tensor((y - target_mean) / target_std, dtype=torch.float32)

    torch.manual_seed(42)
    net = build_lstm_network()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    n = len(X_t)
    batch_size = 64
    epochs = 40
    for epoch in range(epochs):
        net.train()
        perm = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            optimizer.zero_grad()
            pred = net(X_t[idx])
            loss = loss_fn(pred, y_t[idx])
            loss.backward()
            optimizer.step()

    net.eval()
    return {
        "net_state": net.state_dict(),
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "target_mean": target_mean,
        "target_std": target_std,
    }


def predict_lstm_state(state: dict, window: np.ndarray) -> float:
    import torch

    scaled = (window - state["feature_mean"]) / state["feature_std"]
    net = build_lstm_network()
    net.load_state_dict(state["net_state"])
    net.eval()
    with torch.no_grad():
        pred_scaled = float(net(torch.tensor(scaled, dtype=torch.float32)).item())
    return pred_scaled * state["target_std"] + state["target_mean"]


def backtest_lstm(df: pd.DataFrame, horizon: int) -> dict:
    split = len(df) - BACKTEST_DAYS - horizon
    if split < WINDOW + horizon + 30:
        raise ValueError("no hay suficiente historia para el backtest temporal")
    state = train_lstm.local(df.iloc[:split], horizon)
    feats, target = build_features(df, horizon)
    predicted, actual = [], []
    for feature_day in range(max(WINDOW, split - 1), len(feats)):
        if np.isnan(target[feature_day]):
            continue
        window = feats[feature_day - WINDOW + 1 : feature_day + 1][None, :, :]
        predicted.append(predict_lstm_state(state, window))
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
    metrics = backtest_lstm(df, horizon)
    path = _artifact_path(ticker, horizon)
    incumbent_score = -1.0
    if path.exists():
        with path.open("rb") as fh:
            incumbent_score = float(pickle.load(fh)["metrics"]["directional_accuracy"])
    promoted = metrics["directional_accuracy"] > incumbent_score
    if promoted:
        artifact = {
            "state": train_lstm.local(df, horizon), "metrics": metrics,
            "ticker": ticker, "horizon": horizon,
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
    image=image, volumes={"/artifacts": artifact_volume},
    schedule=modal.Cron("0 20 * * 1-5", timezone="America/Argentina/Buenos_Aires"),
    timeout=86400, cpu=1.0, memory=1024,
)
def retrain_models() -> list[dict]:
    results = []
    for ticker in fetch_available_tickers():
        for horizon in range(MIN_HORIZON, MAX_HORIZON + 1):
            try:
                results.append(retrain_one(ticker, horizon))
            except Exception as exc:
                results.append({"ticker": ticker, "horizon": horizon, "error": str(exc)})
    print(json.dumps(results))
    return results


# Recursos del contenedor: benchmarkeado a mano (curl con tiempos, request
# en caliente) contra cpu=2.0/memory=1024 -- contra la intuicion, quedo mas
# LENTO (~11s en caliente) que con cpu=1.0/memory=512 (~7.6s). La red es tan
# chica (32 neuronas) que no se beneficia de mas nucleos, y de paso sale mas
# barato. No se fija `min_containers` (default 0, escala a cero sin
# trafico): se prioriza no pagar por contenedores idle antes que eliminar
# el cold start (~33s la primera vez, aceptable para este caso de uso).
@app.function(
    image=image, timeout=60, cpu=1.0, memory=512,
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
    window = feats[-WINDOW:][None, :, :]
    log_return = predict_lstm_state(artifact["state"], window)

    result = derive_trend_output(df, log_return, horizon)
    result["ticker"] = ticker
    result["model"] = "lstm"
    result["source"] = "modal (artefacto preentrenado)"
    result["model_version"] = artifact["trained_at"]
    result["backtest"] = artifact["metrics"]
    if days_repaired:
        result["price_data_repaired_days"] = days_repaired
    return result
