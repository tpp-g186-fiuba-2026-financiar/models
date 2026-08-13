import json
import os
import pickle
from datetime import datetime, timezone
from pathlib import Path

import modal
import numpy as np
import pandas as pd
import requests
from sklearn import svm

image = modal.Image.debian_slim().pip_install(
    "fastapi[standard]", "pandas", "numpy", "scikit-learn", "requests"
)
app = modal.App("svm-model")
artifact_volume = modal.Volume.from_name("svm-model-artifacts", create_if_missing=True)
ARTIFACT_ROOT = Path("/artifacts")
DATA_COLLECTOR_URL = "https://data-colector.onrender.com"
BACKTEST_DAYS = 60


def fetch_available_tickers() -> list[str]:
    response = requests.post(f"{DATA_COLLECTOR_URL}/available-tickers", timeout=30)
    response.raise_for_status()
    payload = response.json()
    tickers = payload.get("tickers") or (payload.get("message") or {}).get("tickers") or []
    if not tickers:
        raise ValueError("data-colector no devolvio tickers disponibles")
    return sorted({str(t).strip().upper() for t in tickers if str(t).strip()})


def get_ticker_data_and_transform(ticker: str) -> pd.DataFrame:
    response = requests.post(f"{DATA_COLLECTOR_URL}/historical-data/{ticker}", timeout=30)
    response.raise_for_status()
    rows = response.json().get("data") or []
    if not rows:
        raise ValueError(f"no hay historia para {ticker}")
    df = pd.DataFrame(rows)
    for column in ("close_amount", "open_amount", "high_amount", "low_amount"):
        df[column] = df[column].astype(float)
    df["Open-Close"] = df["close_amount"] - df["open_amount"]
    df["High-Low"] = df["high_amount"] - df["low_amount"]
    return df


def training_arrays(df: pd.DataFrame):
    features = df[["Open-Close", "High-Low"]].to_numpy(dtype=float)
    target = (df["close_amount"].shift(-1) > df["close_amount"]).astype(int).to_numpy()
    return features[:-1], target[:-1]


@app.function()
def train_classifier(x: np.ndarray, y: np.ndarray):
    classifier = svm.SVC()
    classifier.fit(x, y)
    return classifier


def retrain_one(ticker: str) -> dict:
    df = get_ticker_data_and_transform(ticker)
    x, y = training_arrays(df)
    if len(x) <= BACKTEST_DAYS + 30:
        raise ValueError("historia insuficiente para backtest")
    split = len(x) - BACKTEST_DAYS
    candidate_test = train_classifier.local(x[:split], y[:split])
    accuracy = float(np.mean(candidate_test.predict(x[split:]) == y[split:]))
    metrics = {"directional_accuracy": accuracy, "observations": BACKTEST_DAYS}
    path = ARTIFACT_ROOT / ticker / "production.pkl"
    incumbent = -1.0
    if path.exists():
        with path.open("rb") as fh:
            incumbent = float(pickle.load(fh)["metrics"]["directional_accuracy"])
    promoted = accuracy > incumbent
    if promoted:
        artifact = {"state": train_classifier.local(x, y), "metrics": metrics, "ticker": ticker,
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
    schedule=modal.Cron("30 21 * * 1-5", timezone="America/Argentina/Buenos_Aires"),
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
    df = get_ticker_data_and_transform(ticker)
    latest = df[["Open-Close", "High-Low"]].iloc[[-1]].to_numpy(dtype=float)
    # Conserva el contrato historico del endpoint.
    prediction = "Buy" if int(artifact["state"].predict(latest)[0]) == 0 else "Sell"
    return {"prediction": prediction, "model_version": artifact["trained_at"],
            "backtest": artifact["metrics"]}
