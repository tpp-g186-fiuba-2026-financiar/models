import json
import os
import pickle
from datetime import datetime, timezone
from pathlib import Path

import modal
import numpy as np
import pandas as pd
import requests
from statsmodels.tsa.arima.model import ARIMA

image = modal.Image.debian_slim().pip_install(
    "fastapi[standard]", "pandas", "numpy", "requests", "statsmodels"
)
app = modal.App("arima-model")
artifact_volume = modal.Volume.from_name("arima-model-artifacts", create_if_missing=True)
ARTIFACT_ROOT = Path("/artifacts")
DATA_COLLECTOR_URL = "https://data-colector.onrender.com"
MEDIA_MOVIL = 20
BACKTEST_HORIZON = 5
NEUTRAL_BAND = 0.01


def fetch_available_tickers() -> list[str]:
    response = requests.post(f"{DATA_COLLECTOR_URL}/available-tickers", timeout=30)
    response.raise_for_status()
    payload = response.json()
    tickers = payload.get("tickers") or (payload.get("message") or {}).get("tickers") or []
    if not tickers:
        raise ValueError("data-colector no devolvio tickers disponibles")
    return sorted({str(t).strip().upper() for t in tickers if str(t).strip()})


def get_ticker_rows(ticker: str) -> list[dict]:
    response = requests.post(f"{DATA_COLLECTOR_URL}/historical-data/{ticker}", timeout=30)
    response.raise_for_status()
    rows = response.json().get("data") or []
    if not rows:
        raise ValueError(f"no hay historia para {ticker}")
    return rows


def get_ticker_data(ticker: str) -> list[float]:
    return [float(row["close_amount"]) for row in get_ticker_rows(ticker)]


def rsi_last(close: list[float], period: int = 14) -> float | None:
    """RSI de Wilder sobre el ultimo valor de la serie (mismo calculo que lstm_trend_model.py::rsi)."""
    if len(close) < period + 1:
        return None
    deltas = np.diff(np.asarray(close, dtype=np.float64))
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = gains[:period].mean()
    avg_loss = losses[:period].mean()
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def rsi_condition(rsi_value: float | None) -> str:
    if rsi_value is None:
        return "indeterminado"
    if rsi_value >= 70:
        return "sobrecompra"
    if rsi_value <= 30:
        return "sobreventa"
    return "neutral"


@app.function()
def train_model(values: list[float], media_movil: int = MEDIA_MOVIL):
    return ARIMA(pd.Series(values), order=(1, 1, media_movil)).fit()


# Backtest walk-forward: hasta BACKTEST_FOLDS tramos consecutivos de
# BACKTEST_FOLD_DAYS ruedas. En cada tramo se ajusta ARIMA SOLO con lo anterior
# y se predice a BACKTEST_HORIZON ruedas desde origenes separados esa misma
# cantidad de ruedas (casos independientes), comparando la direccion predicha
# contra la que realmente ocurrio. Antes se hacia UNA sola prediccion de 30 dias
# y se contaba cada dia del forecast como un caso: el modelo "acertaba" ~90%
# solo por quedar del mismo lado que el punto de partida durante una racha,
# aunque el precio predicho estuviera muy lejos del real. Eso no es lo que
# hace el modelo en produccion (predicciones a 5 dias) y no es comparable con
# los demas modelos ni con paper trading.
BACKTEST_FOLDS = 3
BACKTEST_FOLD_DAYS = 40


def wilson_interval(hits: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Intervalo de confianza (95%) de una proporcion: cuanta confianza dar a una accuracy."""
    if n == 0:
        return 0.0, 1.0
    p = hits / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return float(max(0.0, center - half)), float(min(1.0, center + half))


def backtest_plan(n_rows: int, horizon: int, min_train: int) -> list[tuple[int, list[int]]]:
    """[(split, [origenes])]: se ajusta con `[:split]` y se predice desde cada origen.

    Un origen es el indice del ultimo cierre conocido. Si la historia no
    alcanza para todos los tramos se usan menos; si no alcanza ni para uno, falla.
    """
    available = (n_rows - horizon - min_train) // BACKTEST_FOLD_DAYS
    folds = min(BACKTEST_FOLDS, available)
    if folds < 1:
        raise ValueError("historia insuficiente para backtest")
    first_split = n_rows - horizon - folds * BACKTEST_FOLD_DAYS
    return [
        (split, list(range(split - 1, split - 1 + BACKTEST_FOLD_DAYS, horizon)))
        for split in (first_split + fold * BACKTEST_FOLD_DAYS for fold in range(folds))
    ]


def backtest_arima(values: list[float], horizon: int = BACKTEST_HORIZON) -> dict:
    plan = backtest_plan(len(values), horizon, MEDIA_MOVIL + 30)
    predicted, actual, origin_close, evaluation_series = [], [], [], []
    for split, origins in plan:
        state = train_model.local(values[:split], MEDIA_MOVIL)
        known = split - 1  # ultimo indice que el modelo ya "vio"
        for origin in origins:
            if origin + horizon >= len(values):
                continue
            if origin > known:
                # Actualiza el estado con las ruedas nuevas sin reajustar parametros
                # (no mira nada posterior al origen).
                new = pd.Series(values[known + 1 : origin + 1], index=pd.RangeIndex(known + 1, origin + 1))
                state = state.append(new, refit=False)
                known = origin
            forecast = float(np.asarray(state.forecast(steps=horizon), dtype=float)[-1])
            predicted.append(forecast)
            actual.append(values[origin + horizon])
            origin_close.append(values[origin])
            evaluation_series.append(
                {"date": str(len(evaluation_series) + 1), "predicted": forecast, "actual": values[origin + horizon]}
            )
    if not actual:
        raise ValueError("el backtest no produjo observaciones")
    predicted_arr, actual_arr, base = map(np.asarray, (predicted, actual, origin_close))
    hits = int(np.sum(np.sign(predicted_arr - base) == np.sign(actual_arr - base)))
    low, high = wilson_interval(hits, len(actual_arr))
    expected = predicted_arr / base - 1.0
    position = np.where(expected > NEUTRAL_BAND, 1.0, np.where(expected < -NEUTRAL_BAND, -1.0, 0.0))
    active = position != 0
    return {
        "directional_accuracy": hits / len(actual_arr),
        "accuracy_low": low,
        "accuracy_high": high,
        "signal_hit_rate": (
            float(np.mean(np.sign(actual_arr[active] - base[active]) == position[active]))
            if active.any()
            else None
        ),
        "neutral_rate": float(np.mean(~active)),
        # En precio (no en retorno): la web lo muestra como % del ultimo cierre.
        "mae": float(np.mean(np.abs(predicted_arr - actual_arr))),
        "observations": len(actual_arr),
        "folds": len(plan),
        "series": evaluation_series[-30:],
    }


def retrain_one(ticker: str) -> dict:
    values = get_ticker_data(ticker)
    metrics = backtest_arima(values)
    path = ARTIFACT_ROOT / ticker / "production.pkl"
    incumbent_mae = float("inf")
    incumbent_has_series = False
    if path.exists():
        with path.open("rb") as fh:
            incumbent = pickle.load(fh)
            incumbent_mae = float(incumbent["metrics"]["mae"])
            incumbent_has_series = "folds" in incumbent["metrics"]  # metrica vieja: no comparable
    promoted = metrics["mae"] < incumbent_mae or not incumbent_has_series
    if promoted:
        artifact = {
            "state": train_model.local(values, MEDIA_MOVIL), "metrics": metrics,
            "ticker": ticker, "media_movil": MEDIA_MOVIL,
            "trained_at": datetime.now(timezone.utc).isoformat(),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        with temporary.open("wb") as fh:
            pickle.dump(artifact, fh)
        os.replace(temporary, path)
        artifact_volume.commit()
    return {"ticker": ticker, "promoted": promoted, "metrics": metrics}


@app.function(
    image=image, volumes={"/artifacts": artifact_volume}, timeout=86400,
    schedule=modal.Cron("0 22 * * 1-5", timezone="America/Argentina/Buenos_Aires"),
    cpu=1.0, memory=512,
)
def retrain_models() -> list[dict]:
    results = []
    for ticker in fetch_available_tickers():
        try:
            results.append(retrain_one(ticker))
        except Exception as exc:
            results.append({"ticker": ticker, "error": str(exc)})
    print(json.dumps(results))
    return results


# Recursos elegidos para mantener bajo el costo de un modelo chico. Sin
# min_containers: los artefactos evitan entrenar en requests aunque haya cold start.
@app.function(image=image, cpu=1.0, memory=512, volumes={"/artifacts": artifact_volume})
@modal.fastapi_endpoint()
def main(ticker: str, predictions: int, media_movil: int = MEDIA_MOVIL):
    ticker = ticker.strip().upper()
    if media_movil != MEDIA_MOVIL:
        return {"error": f"solo hay artefactos preentrenados para media_movil={MEDIA_MOVIL}"}
    artifact_volume.reload()
    path = ARTIFACT_ROOT / ticker / "production.pkl"
    if not path.exists():
        return {"error": f"todavia no hay un modelo entrenado para '{ticker}'"}
    with path.open("rb") as fh:
        artifact = pickle.load(fh)
    rows = get_ticker_rows(ticker)
    values = [float(row["close_amount"]) for row in rows]
    last_ts = rows[-1].get("ts")
    as_of = (
        datetime.fromtimestamp(last_ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        if last_ts is not None
        else None
    )
    rsi_value = rsi_last(values)
    return {
        "prediction": artifact["state"].forecast(steps=predictions).tolist(),
        "valor_actual": values[-1], "cant_predicciones": predictions,
        "model_version": artifact["trained_at"], "backtest": artifact["metrics"],
        "rsi": round(rsi_value, 2) if rsi_value is not None else None,
        "condition": rsi_condition(rsi_value),
        "as_of": as_of,
    }
