import json
import os
import pickle
from datetime import datetime, timezone
from pathlib import Path

import modal
import numpy as np
import requests
from arch import arch_model

image = modal.Image.debian_slim().pip_install(
    "fastapi[standard]", "arch", "pandas", "numpy", "requests"
)
app = modal.App("garch-model")
artifact_volume = modal.Volume.from_name("garch-model-artifacts", create_if_missing=True)
ARTIFACT_ROOT = Path("/artifacts")
DATA_COLLECTOR_URL = "https://data-colector.onrender.com"
FORECAST_HORIZON = 5
BACKTEST_DAYS = 30


def fetch_available_tickers() -> list[str]:
    response = requests.post(f"{DATA_COLLECTOR_URL}/available-tickers", timeout=30)
    response.raise_for_status()
    payload = response.json()
    tickers = payload.get("tickers") or (payload.get("message") or {}).get("tickers") or []
    if not tickers:
        raise ValueError("data-colector no devolvio tickers disponibles")
    return sorted({str(t).strip().upper() for t in tickers if str(t).strip()})


def get_ticker_returns(ticker: str) -> np.ndarray:
    response = requests.post(f"{DATA_COLLECTOR_URL}/historical-data/{ticker}", timeout=30)
    response.raise_for_status()
    rows = response.json().get("data") or []
    if not rows:
        raise ValueError(f"no hay historia para {ticker}")
    prices = np.asarray([float(row["close_amount"]) for row in rows], dtype=float)
    return np.diff(np.log(prices)) * 100.0


@app.function()
def train_model(returns: np.ndarray):
    return arch_model(returns, vol="Garch", p=1, q=1).fit(update_freq=0, disp="off")


def retrain_one(ticker: str) -> dict:
    returns = get_ticker_returns(ticker)
    if len(returns) <= BACKTEST_DAYS + 50:
        raise ValueError("historia insuficiente para backtest")
    split = len(returns) - BACKTEST_DAYS
    candidate_test = train_model.local(returns[:split])
    # Holdout temporal: mide varianza pronosticada contra retorno observado^2.
    predicted = candidate_test.forecast(horizon=BACKTEST_DAYS).variance.iloc[-1].to_numpy(dtype=float)
    actual = returns[split:] ** 2
    metrics = {"variance_mae": float(np.mean(np.abs(np.asarray(predicted) - actual))), "observations": len(actual)}
    path = ARTIFACT_ROOT / ticker / "production.pkl"
    incumbent = float("inf")
    if path.exists():
        with path.open("rb") as fh:
            incumbent = float(pickle.load(fh)["metrics"]["variance_mae"])
    promoted = metrics["variance_mae"] < incumbent
    if promoted:
        artifact = {"state": train_model.local(returns), "metrics": metrics, "ticker": ticker,
                    "trained_at": datetime.now(timezone.utc).isoformat()}
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        with temporary.open("wb") as fh:
            pickle.dump(artifact, fh)
        os.replace(temporary, path)
        artifact_volume.commit()
    return {"ticker": ticker, "promoted": promoted, "metrics": metrics}


@app.function(
    image=image, volumes={"/artifacts": artifact_volume}, timeout=86400,
    schedule=modal.Cron("0 23 * * 1-5", timezone="America/Argentina/Buenos_Aires"),
    cpu=1.0, memory=512,
)
def retrain_models() -> list[dict]:
    # Sin cron propio para GARCH-ANN (limite de 5 del plan gratis): lo dispara este job.
    try:
        modal.Function.from_name("garch-ann-model", "retrain_models").spawn()
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] no se pudo disparar el reentrenamiento de GARCH-ANN: {exc}")
    results = []
    for ticker in fetch_available_tickers():
        try:
            results.append(retrain_one(ticker))
        except Exception as exc:
            results.append({"ticker": ticker, "error": str(exc)})
    print(json.dumps(results))
    return results


@app.function(image=image, cpu=1.0, memory=512, volumes={"/artifacts": artifact_volume})
@modal.fastapi_endpoint()
def main(ticker: str):
    ticker = ticker.strip().upper()
    artifact_volume.reload()
    path = ARTIFACT_ROOT / ticker / "production.pkl"
    if not path.exists():
        return {"error": f"todavia no hay un modelo entrenado para '{ticker}'"}
    with path.open("rb") as fh:
        artifact = pickle.load(fh)
    forecast = artifact["state"].forecast(horizon=FORECAST_HORIZON).variance.iloc[-1:]
    return {"prediction": forecast.to_dict(), "model_version": artifact["trained_at"],
            "backtest": artifact["metrics"]}
