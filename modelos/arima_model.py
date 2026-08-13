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
BACKTEST_DAYS = 30


def fetch_available_tickers() -> list[str]:
    response = requests.post(f"{DATA_COLLECTOR_URL}/available-tickers", timeout=30)
    response.raise_for_status()
    payload = response.json()
    tickers = payload.get("tickers") or (payload.get("message") or {}).get("tickers") or []
    if not tickers:
        raise ValueError("data-colector no devolvio tickers disponibles")
    return sorted({str(t).strip().upper() for t in tickers if str(t).strip()})


def get_ticker_data(ticker: str) -> list[float]:
    response = requests.post(f"{DATA_COLLECTOR_URL}/historical-data/{ticker}", timeout=30)
    response.raise_for_status()
    rows = response.json().get("data") or []
    if not rows:
        raise ValueError(f"no hay historia para {ticker}")
    return [float(row["close_amount"]) for row in rows]


@app.function()
def train_model(values: list[float], media_movil: int = MEDIA_MOVIL):
    return ARIMA(pd.Series(values), order=(1, 1, media_movil)).fit()


def retrain_one(ticker: str) -> dict:
    values = get_ticker_data(ticker)
    if len(values) <= BACKTEST_DAYS + MEDIA_MOVIL + 5:
        raise ValueError("historia insuficiente para backtest")
    split = len(values) - BACKTEST_DAYS
    candidate_test = train_model.local(values[:split], MEDIA_MOVIL)
    forecast = np.asarray(candidate_test.forecast(steps=BACKTEST_DAYS), dtype=float)
    actual = np.asarray(values[split:])
    metrics = {
        "mae": float(np.mean(np.abs(forecast - actual))),
        "observations": BACKTEST_DAYS,
        "series": [
            {"date": str(index + 1), "predicted": float(predicted), "actual": float(real)}
            for index, (predicted, real) in enumerate(zip(forecast, actual))
        ],
    }
    path = ARTIFACT_ROOT / ticker / "production.pkl"
    incumbent_mae = float("inf")
    incumbent_has_series = False
    if path.exists():
        with path.open("rb") as fh:
            incumbent = pickle.load(fh)
            incumbent_mae = float(incumbent["metrics"]["mae"])
            incumbent_has_series = bool(incumbent["metrics"].get("series"))
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
    values = get_ticker_data(ticker)
    return {
        "prediction": artifact["state"].forecast(steps=predictions).tolist(),
        "valor_actual": values[-1], "cant_predicciones": predictions,
        "model_version": artifact["trained_at"], "backtest": artifact["metrics"],
    }
