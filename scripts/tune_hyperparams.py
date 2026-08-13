"""Busqueda de ventana/horizonte/arquitectura para LSTM y XGBoost (issue #160).

Usa lo que salio del EDA (issue #135) para no ir a ciegas:
- El horizonte de 1 dia NO mostro mas señal que 5 en el EDA (ver seccion 13.a
  de api-ml/notebooks/eda.ipynb) -- por eso ac'a no se prueba 1, se prueba
  alrededor de los valores ya usados en produccion (3/5/7/10).
- El resto (ventana de dias, tamaño de la red, cantidad de arboles) no tenia
  ningun analisis previo -- se barre un grid chico y se mide con un backtest
  walk-forward real (entrena, predice hacia adelante sin mirar el futuro,
  reentrena mas adelante), no con una sola metrica de entrenamiento.

Autocontenido a proposito, igual que el resto de este repo: duplica el
feature engineering de `modelos/lstm_trend_model.py` /
`modelos/xgboost_trend_model.py` en vez de importarlos (esos archivos tienen
WINDOW/HORIZON como constantes de modulo pensadas para Modal, no parametros
de funcion -- mas simple copiar la logica pura que parchear eso). No le pega
a api-ml en ningun momento, solo a data-colector, igual que el resto de
`modelos/*.py`.

Correr con: `.venv/bin/python scripts/tune_hyperparams.py`
"""

from __future__ import annotations

import itertools
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import requests

DATA_COLLECTOR_URL = "https://data-colector.onrender.com"
MACRO_RATE_SOURCE = "us"
MACRO_RATE_SERIES = "TNX"
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

# Tickers de validacion: bancos/energia/industria, los tres con historial
# completo (735 ruedas, ver EDA seccion 1) y sin el problema de dato de ECOG.
VALIDATION_TICKERS = ["GGAL", "YPFD", "ALUA"]

TEST_DAYS = 200  # cuantas ruedas del final se usan para evaluar (walk-forward)
RETRAIN_STEP = 40  # cada cuantos dias se reentrena durante la evaluacion


# --------------------------------------------------------------------------- #
# Datos y features (copiado de modelos/lstm_trend_model.py, sin los
# decoradores de Modal)
# --------------------------------------------------------------------------- #
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


def fetch_macro_series(source: str = MACRO_RATE_SOURCE, series: str = MACRO_RATE_SERIES):
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
    except Exception as exc:  # noqa: BLE001 - feature opcional
        print(f"  [warn] no se pudo obtener la serie macro ({source}/{series}): {exc}")
        return None


def attach_macro_feature(df: pd.DataFrame, macro) -> pd.DataFrame:
    df = df.copy()
    if macro is None or macro.empty:
        df["macro_rate"] = float("nan")
        return df
    df["macro_rate"] = macro.reindex(df.index, method="ffill")
    return df


def rsi_series(close: np.ndarray, period: int = 14) -> np.ndarray:
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
    n = len(values)
    out = np.full(n, np.nan)
    if n < window:
        return out
    cumsum = np.cumsum(np.insert(values, 0, 0.0))
    out[window - 1 :] = (cumsum[window:] - cumsum[:-window]) / window
    return out


def ema(values: np.ndarray, span: int) -> np.ndarray:
    alpha = 2.0 / (span + 1.0)
    out = np.empty(len(values), dtype=np.float64)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    return out


def macd_histogram(close: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> np.ndarray:
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    return macd_line - signal_line


def build_features(df: pd.DataFrame, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """Identico a modelos/lstm_trend_model.py::build_features. Causal: cada
    fila k solo usa datos hasta esa fecha (mas el target, que mira `horizon`
    dias para adelante -- por eso el chequeo de "no-leakage" para train vive
    en `safe_training_cutoff`, no aca)."""
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


def make_windows(feats: np.ndarray, target: np.ndarray, window: int, upto: int) -> tuple[np.ndarray, np.ndarray]:
    """Ventanas deslizantes usando `feats`/`target` solo hasta el indice
    `upto` (exclusive) -- este es el unico lugar donde se corta para
    entrenar, para no filtrar futuro."""
    xs, ys = [], []
    limit = min(upto, len(feats))
    for i in range(limit - window):
        y = target[i + window - 1]
        if np.isnan(y):
            continue
        xs.append(feats[i : i + window])
        ys.append(y)
    if not xs:
        return np.empty((0, window, feats.shape[1])), np.empty((0,))
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


# --------------------------------------------------------------------------- #
# Entrenamiento (parametrizado, a diferencia de los .py de produccion)
# --------------------------------------------------------------------------- #
def train_lstm_local(X: np.ndarray, y: np.ndarray, hidden_size: int, num_layers: int, seed: int = 42):
    import torch
    from torch import nn

    flat = X.reshape(-1, X.shape[-1])
    feature_mean = flat.mean(axis=0)
    feature_std = flat.std(axis=0) + 1e-8
    target_mean = float(y.mean())
    target_std = float(y.std() + 1e-8)

    X_scaled = (X - feature_mean) / feature_std
    X_t = torch.tensor(X_scaled, dtype=torch.float32)
    y_t = torch.tensor((y - target_mean) / target_std, dtype=torch.float32)

    class _LSTMRegressor(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, dropout=0.1):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.head = nn.Linear(hidden_size, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(out[:, -1, :]).squeeze(-1)

    torch.manual_seed(seed)
    net = _LSTMRegressor(X.shape[-1], hidden_size, num_layers)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    n = len(X_t)
    batch_size = 64
    epochs = 40
    for _ in range(epochs):
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
    return {"net": net, "feature_mean": feature_mean, "feature_std": feature_std,
            "target_mean": target_mean, "target_std": target_std}


def predict_lstm_local(state, window_feats: np.ndarray) -> float:
    import torch

    scaled = (window_feats - state["feature_mean"]) / state["feature_std"]
    x = torch.tensor(scaled[None, :, :], dtype=torch.float32)
    with torch.no_grad():
        pred_scaled = float(state["net"](x).item())
    return pred_scaled * state["target_std"] + state["target_mean"]


def train_xgb_local(X: np.ndarray, y: np.ndarray, n_estimators: int, max_depth: int, seed: int = 42):
    import xgboost as xgb

    booster = xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        objective="reg:squarederror",
    )
    booster.fit(X.reshape(len(X), -1), y, verbose=False)
    return booster


# --------------------------------------------------------------------------- #
# Backtest walk-forward
# --------------------------------------------------------------------------- #
@dataclass
class WFResult:
    correct: int = 0
    total: int = 0
    abs_err_sum: float = 0.0

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else float("nan")

    @property
    def mae(self) -> float:
        return self.abs_err_sum / self.total if self.total else float("nan")


def walk_forward(
    feats: np.ndarray,
    target: np.ndarray,
    window: int,
    test_days: int,
    retrain_step: int,
    train_fn,
    predict_fn,
) -> WFResult:
    """Entrena en `retrain_step`, evalua en los dias siguientes sin volver a
    entrenar, hasta el proximo retrain -- estandar en backtests, evita pagar
    el costo de reentrenar en cada dia evaluado."""
    n = len(feats)
    test_start = max(window + 30, n - test_days)
    if test_start >= n - 1:
        return WFResult()

    result = WFResult()
    r = test_start
    while r < n:
        X, y = make_windows(feats, target, window, upto=r)
        if len(X) < 30:
            r += retrain_step
            continue
        state = train_fn(X, y)
        for p in range(r, min(r + retrain_step, n)):
            if p < window - 1 or np.isnan(target[p]):
                continue
            window_feats = feats[p - window + 1 : p + 1]
            pred = predict_fn(state, window_feats)
            actual = target[p]
            result.correct += int(np.sign(pred) == np.sign(actual))
            result.total += 1
            result.abs_err_sum += abs(pred - actual)
        r += retrain_step
    return result


# --------------------------------------------------------------------------- #
# Orquestacion
# --------------------------------------------------------------------------- #
def load_ticker_data() -> dict[str, pd.DataFrame]:
    macro = fetch_macro_series()
    data = {}
    for t in VALIDATION_TICKERS:
        df = fetch_ticker_history(t)
        df = attach_macro_feature(df, macro)
        data[t] = df
        print(f"  {t}: {len(df)} ruedas")
    return data


def run_combo(dfs: dict[str, pd.DataFrame], horizon: int, window: int, train_fn, predict_fn) -> WFResult:
    agg = WFResult()
    for t, df in dfs.items():
        feats, target = build_features(df, horizon)
        r = walk_forward(feats, target, window, TEST_DAYS, RETRAIN_STEP, train_fn, predict_fn)
        agg.correct += r.correct
        agg.total += r.total
        agg.abs_err_sum += r.abs_err_sum
    return agg


def main():
    print("bajando historicos de", VALIDATION_TICKERS)
    dfs = load_ticker_data()

    results = {"lstm": [], "xgboost": []}

    # --- LSTM: ventana x horizonte (arquitectura default: hidden=32, layers=1) ---
    print("\n=== LSTM: ventana x horizonte ===")
    windows = [20, 30, 45]
    horizons = [3, 5, 7, 10]
    for window, horizon in itertools.product(windows, horizons):
        t0 = time.time()

        def train_fn(X, y, hidden=32, layers=1):
            return train_lstm_local(X, y, hidden, layers)

        agg = run_combo(dfs, horizon, window, train_fn, predict_lstm_local)
        dt = time.time() - t0
        row = {"window": window, "horizon": horizon, "hidden": 32, "layers": 1,
               "accuracy": agg.accuracy, "mae": agg.mae, "n": agg.total, "seconds": round(dt, 1)}
        results["lstm"].append(row)
        print(f"  window={window:2d} horizon={horizon:2d} -> acc={agg.accuracy:.3f} "
              f"mae={agg.mae:.4f} n={agg.total} ({dt:.0f}s)")

    best_wh = max(results["lstm"], key=lambda r: r["accuracy"])
    print(f"\nmejor ventana/horizonte (LSTM): window={best_wh['window']} horizon={best_wh['horizon']} "
          f"acc={best_wh['accuracy']:.3f}")

    # --- LSTM: arquitectura, con la mejor ventana/horizonte de arriba ---
    print("\n=== LSTM: arquitectura (hidden_size x num_layers) ===")
    hidden_sizes = [16, 32, 64]
    layers_opts = [1, 2]
    for hidden, layers in itertools.product(hidden_sizes, layers_opts):
        t0 = time.time()

        def train_fn(X, y, hidden=hidden, layers=layers):
            return train_lstm_local(X, y, hidden, layers)

        agg = run_combo(dfs, best_wh["horizon"], best_wh["window"], train_fn, predict_lstm_local)
        dt = time.time() - t0
        row = {"window": best_wh["window"], "horizon": best_wh["horizon"], "hidden": hidden, "layers": layers,
               "accuracy": agg.accuracy, "mae": agg.mae, "n": agg.total, "seconds": round(dt, 1)}
        results["lstm"].append(row)
        print(f"  hidden={hidden:2d} layers={layers} -> acc={agg.accuracy:.3f} "
              f"mae={agg.mae:.4f} n={agg.total} ({dt:.0f}s)")

    # --- XGBoost: ventana x horizonte (arquitectura default) ---
    print("\n=== XGBoost: ventana x horizonte ===")
    for window, horizon in itertools.product(windows, horizons):
        t0 = time.time()

        def train_fn(X, y, n_est=300, depth=4):
            return train_xgb_local(X, y, n_est, depth)

        def predict_fn(booster, window_feats):
            return float(booster.predict(window_feats.reshape(1, -1))[0])

        agg = run_combo(dfs, horizon, window, train_fn, predict_fn)
        dt = time.time() - t0
        row = {"window": window, "horizon": horizon, "n_estimators": 300, "max_depth": 4,
               "accuracy": agg.accuracy, "mae": agg.mae, "n": agg.total, "seconds": round(dt, 1)}
        results["xgboost"].append(row)
        print(f"  window={window:2d} horizon={horizon:2d} -> acc={agg.accuracy:.3f} "
              f"mae={agg.mae:.4f} n={agg.total} ({dt:.0f}s)")

    best_wh_xgb = max(results["xgboost"], key=lambda r: r["accuracy"])
    print(f"\nmejor ventana/horizonte (XGBoost): window={best_wh_xgb['window']} horizon={best_wh_xgb['horizon']} "
          f"acc={best_wh_xgb['accuracy']:.3f}")

    # --- XGBoost: arquitectura ---
    print("\n=== XGBoost: arquitectura (n_estimators x max_depth) ===")
    n_estimators_opts = [200, 300, 450]
    max_depth_opts = [3, 4, 5]
    for n_est, depth in itertools.product(n_estimators_opts, max_depth_opts):
        t0 = time.time()

        def train_fn(X, y, n_est=n_est, depth=depth):
            return train_xgb_local(X, y, n_est, depth)

        def predict_fn(booster, window_feats):
            return float(booster.predict(window_feats.reshape(1, -1))[0])

        agg = run_combo(dfs, best_wh_xgb["horizon"], best_wh_xgb["window"], train_fn, predict_fn)
        dt = time.time() - t0
        row = {"window": best_wh_xgb["window"], "horizon": best_wh_xgb["horizon"],
               "n_estimators": n_est, "max_depth": depth,
               "accuracy": agg.accuracy, "mae": agg.mae, "n": agg.total, "seconds": round(dt, 1)}
        results["xgboost"].append(row)
        print(f"  n_estimators={n_est:3d} max_depth={depth} -> acc={agg.accuracy:.3f} "
              f"mae={agg.mae:.4f} n={agg.total} ({dt:.0f}s)")

    out_path = Path(__file__).parent / "tune_results.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nresultados guardados en {out_path}")


if __name__ == "__main__":
    main()
