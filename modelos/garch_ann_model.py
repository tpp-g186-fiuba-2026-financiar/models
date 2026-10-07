import json
import os
import pickle
from datetime import datetime, timezone
from pathlib import Path

import modal
import numpy as np
import requests

image = modal.Image.debian_slim().pip_install(
    "fastapi[standard]", "arch", "scikit-learn", "pandas", "numpy", "requests"
)
app = modal.App("garch-ann-model")
artifact_volume = modal.Volume.from_name("garch-ann-model-artifacts", create_if_missing=True)
ARTIFACT_ROOT = Path("/artifacts")
DATA_COLLECTOR_URL = "https://data-colector.onrender.com"
BACKTEST_DAYS = 60
MIN_TRAIN_DAYS = 250


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


def fit_garch_params(returns: np.ndarray) -> np.ndarray:
    from arch import arch_model

    result = arch_model(returns, vol="Garch", p=1, q=1, rescale=False).fit(update_freq=0, disp="off")
    return np.asarray(result.params, dtype=float)


def conditional_volatility(returns: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Volatilidad condicional de toda la serie con parametros GARCH ya estimados."""
    from arch import arch_model

    model = arch_model(returns, vol="Garch", p=1, q=1, rescale=False)
    return np.asarray(model.fix(params).conditional_volatility, dtype=float)


def build_features(returns: np.ndarray, vol: np.ndarray) -> np.ndarray:
    """Cada fila usa el retorno y la volatilidad del dia t para predecir |retorno| de t+1."""
    return np.column_stack([returns, vol])


def train_ann(features: np.ndarray, target: np.ndarray) -> dict:
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(features)
    ann = MLPRegressor(
        hidden_layer_sizes=(64, 32), activation="relu", solver="adam",
        alpha=0.01, max_iter=2000, random_state=42,
    )
    ann.fit(scaler.transform(features), target)
    return {"ann": ann, "scaler": scaler}


def train_model(returns: np.ndarray) -> dict:
    params = fit_garch_params(returns)
    vol = conditional_volatility(returns, params)
    features = build_features(returns, vol)[:-1]
    target = np.abs(returns[1:])
    return {"garch_params": params, **train_ann(features, target)}


def predict_next_abs_return(state: dict, returns: np.ndarray) -> float:
    vol = conditional_volatility(returns, state["garch_params"])
    last = build_features(returns, vol)[-1:]
    # Se usa el scaler del entrenamiento, no uno nuevo ajustado a los datos de prediccion.
    return float(state["ann"].predict(state["scaler"].transform(last))[0])


def retrain_one(ticker: str) -> dict:
    returns = get_ticker_returns(ticker)
    if len(returns) <= BACKTEST_DAYS + MIN_TRAIN_DAYS:
        raise ValueError("historia insuficiente para backtest")
    split = len(returns) - BACKTEST_DAYS
    candidate = train_model(returns[:split])
    # Holdout temporal: prediccion a un dia de |retorno| con datos hasta ese dia.
    predicted = np.asarray([
        predict_next_abs_return(candidate, returns[: i + 1]) for i in range(split - 1, len(returns) - 1)
    ])
    actual = np.abs(returns[split:])
    metrics = {
        "abs_return_mae": float(np.mean(np.abs(predicted - actual))),
        "baseline_mae": float(np.mean(np.abs(np.mean(np.abs(returns[:split])) - actual))),
        "observations": len(actual),
    }
    path = ARTIFACT_ROOT / ticker / "production.pkl"
    incumbent = float("inf")
    if path.exists():
        with path.open("rb") as fh:
            incumbent = float(pickle.load(fh)["metrics"]["abs_return_mae"])
    promoted = metrics["abs_return_mae"] < incumbent
    if promoted:
        artifact = {"state": train_model(returns), "metrics": metrics, "ticker": ticker,
                    "trained_at": datetime.now(timezone.utc).isoformat()}
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        with temporary.open("wb") as fh:
            pickle.dump(artifact, fh)
        os.replace(temporary, path)
        artifact_volume.commit()
    return {"ticker": ticker, "promoted": promoted, "metrics": metrics}


# Sin schedule propio: el plan gratis de Modal limita a 5 funciones programadas
# y ya estan ocupadas. El cron de GARCH (23:00) dispara esta funcion con `.spawn()`.
@app.function(image=image, volumes={"/artifacts": artifact_volume}, timeout=86400, cpu=1.0, memory=1024)
def retrain_models() -> list[dict]:
    results = []
    for ticker in fetch_available_tickers():
        try:
            results.append(retrain_one(ticker))
        except Exception as exc:
            results.append({"ticker": ticker, "error": str(exc)})
    print(json.dumps(results))
    return results


@app.function(image=image, cpu=1.0, memory=1024, volumes={"/artifacts": artifact_volume})
@modal.fastapi_endpoint()
def main(ticker: str):
    ticker = ticker.strip().upper()
    artifact_volume.reload()
    path = ARTIFACT_ROOT / ticker / "production.pkl"
    if not path.exists():
        return {"error": f"todavia no hay un modelo entrenado para '{ticker}'"}
    with path.open("rb") as fh:
        artifact = pickle.load(fh)
    returns = get_ticker_returns(ticker)
    return {
        "prediction": {"next_day_abs_return_pct": predict_next_abs_return(artifact["state"], returns)},
        "model_version": artifact["trained_at"],
        "backtest": artifact["metrics"],
    }
