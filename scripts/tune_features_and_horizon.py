"""Ronda 2 de la busqueda de #160: features nuevas + horizonte acotado a 1-5 dias.

Motivo del cambio de horizonte: la ronda 1 (`tune_hyperparams.py`) encontro
que horizonte=10 predecia mejor que 5 en el walk-forward -- pero un horizonte
de 10 dias es poco util para un cliente mirando la app dia a dia. En vez de
imponer 10, esta ronda prueba 1..5 (el rango que sirve de verdad) y el
horizonte queda como parametro que el que pide la prediccion puede elegir
(`main(ticker, horizon=...)` en los .py de produccion), no un valor fijo.

Features nuevas, las dos que salieron del EDA (issue #135,
`api-ml/notebooks/eda.ipynb`, secciones 13.b y 13.c) como candidatas reales
(no solo "podria servir"):
- `dist_max60`: que tan lejos esta el precio de hoy del maximo de los
  ultimos 60 dias. En el EDA le gano a las 8 features actuales por lejos
  (diferencia mejor-peor grupo 0.0236 contra 0.0200 de la mejor de las 8).
- `rsi_vol_interaction`: RSI normalizado x volumen "raro" (z-score contra el
  propio promedio de 60 dias). En el EDA la combinacion RSI-medio +
  volumen-alto predecia mas que cualquiera de las dos por separado.

Reusa toda la infraestructura de `tune_hyperparams.py` (fetch de datos,
entrenamiento, backtest walk-forward) -- no la duplica.

Correr con: `.venv/bin/python scripts/tune_features_and_horizon.py`
"""

from __future__ import annotations

import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import tune_hyperparams as base

FEATURE_NAMES_OLD = base.FEATURE_NAMES  # las 8 actuales
FEATURE_NAMES_NEW = FEATURE_NAMES_OLD + ["dist_max60", "rsi_vol_interaction"]

TEST_DAYS = base.TEST_DAYS
RETRAIN_STEP = 50  # un poco mas largo que la ronda 1 para no volar el tiempo de corrida


def build_features_v2(df: pd.DataFrame, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """Igual a `tune_hyperparams.build_features`, pero le suma 2 columnas al
    final. Devuelve las 10; quien la llama recorta a `[:, :8]` si quiere
    comparar contra el feature set viejo (misma corrida, sin duplicar
    logica)."""
    feats, target = base.build_features(df, horizon)

    close = df["close"]
    roll_max60 = close.rolling(60, min_periods=30).max().to_numpy(dtype=np.float64)
    close_arr = close.to_numpy(dtype=np.float64)
    dist_max60 = (close_arr / np.where(roll_max60 == 0, np.nan, roll_max60) - 1.0)[1:]

    volume = df["volume"].astype(float)
    roll_mean_vol = volume.rolling(60, min_periods=20).mean()
    roll_std_vol = volume.rolling(60, min_periods=20).std()
    vol_z = ((volume - roll_mean_vol) / roll_std_vol.replace(0, np.nan)).to_numpy(dtype=np.float64)[1:]

    # rsi_norm ya esta en la columna 3 (indice 3) de `feats`, pero esa ya
    # viene con nan_to_num aplicado (0 en vez de NaN) -- para el producto no
    # afecta: si rsi_norm quedo en 0 por falta de historia, el producto
    # tambien da 0, que es el mismo "neutral" que se usa en el resto.
    rsi_norm_col = feats[:, 3]
    rsi_vol_interaction = rsi_norm_col * np.nan_to_num(vol_z, nan=0.0, posinf=0.0, neginf=0.0)

    extra = np.column_stack([dist_max60, rsi_vol_interaction])
    extra = np.nan_to_num(extra, nan=0.0, posinf=0.0, neginf=0.0)
    return np.concatenate([feats, extra], axis=1), target


def run_combo(dfs, horizon, window, n_features, train_fn, predict_fn) -> base.WFResult:
    agg = base.WFResult()
    for _t, df in dfs.items():
        feats, target = build_features_v2(df, horizon)
        feats = feats[:, :n_features]
        r = base.walk_forward(feats, target, window, TEST_DAYS, RETRAIN_STEP, train_fn, predict_fn)
        agg.correct += r.correct
        agg.total += r.total
        agg.abs_err_sum += r.abs_err_sum
    return agg


def main():
    print("bajando historicos de", base.VALIDATION_TICKERS)
    dfs = base.load_ticker_data()

    results = {"lstm_features": [], "lstm_horizon": [], "xgb_features": [], "xgb_horizon": []}

    # Ganadores de la ronda 1: LSTM window=45 hidden=32 layers=2, XGBoost window=30 n_estimators=300 max_depth=4.
    LSTM_WINDOW, LSTM_HIDDEN, LSTM_LAYERS = 45, 32, 2
    XGB_WINDOW, XGB_N_EST, XGB_DEPTH = 30, 300, 4
    REF_HORIZON = 5  # punto medio del rango 1-5, para decidir si las features suman antes de barrer horizonte

    def lstm_train_fn(X, y):
        return base.train_lstm_local(X, y, LSTM_HIDDEN, LSTM_LAYERS)

    def xgb_train_fn(X, y):
        return base.train_xgb_local(X, y, XGB_N_EST, XGB_DEPTH)

    def xgb_predict_fn(booster, w):
        return float(booster.predict(w.reshape(1, -1))[0])

    # --- Paso 1: ¿las features nuevas suman? (8 vs 10, en horizonte=5) ---
    print(f"\n=== LSTM: 8 features (actual) vs 10 (con dist_max60 + rsi_vol_interaction), horizon={REF_HORIZON} ===")
    for n_feat, label in [(8, "8 (actual)"), (10, "10 (nuevas)")]:
        t0 = time.time()
        agg = run_combo(dfs, REF_HORIZON, LSTM_WINDOW, n_feat, lstm_train_fn, base.predict_lstm_local)
        dt = time.time() - t0
        row = {"features": label, "n": agg.total, "accuracy": agg.accuracy, "mae": agg.mae, "seconds": round(dt, 1)}
        results["lstm_features"].append(row)
        print(f"  {label:14s} -> acc={agg.accuracy:.3f} mae={agg.mae:.4f} n={agg.total} ({dt:.0f}s)")

    print(f"\n=== XGBoost: 8 features vs 10, horizon={REF_HORIZON} ===")
    for n_feat, label in [(8, "8 (actual)"), (10, "10 (nuevas)")]:
        t0 = time.time()
        agg = run_combo(dfs, REF_HORIZON, XGB_WINDOW, n_feat, xgb_train_fn, xgb_predict_fn)
        dt = time.time() - t0
        row = {"features": label, "n": agg.total, "accuracy": agg.accuracy, "mae": agg.mae, "seconds": round(dt, 1)}
        results["xgb_features"].append(row)
        print(f"  {label:14s} -> acc={agg.accuracy:.3f} mae={agg.mae:.4f} n={agg.total} ({dt:.0f}s)")

    lstm_best_n = max(results["lstm_features"], key=lambda r: r["accuracy"])
    xgb_best_n = max(results["xgb_features"], key=lambda r: r["accuracy"])
    lstm_n_features = 10 if "10" in lstm_best_n["features"] else 8
    xgb_n_features = 10 if "10" in xgb_best_n["features"] else 8
    print(f"\nLSTM se queda con {lstm_n_features} features, XGBoost con {xgb_n_features} features.")

    # --- Paso 2: horizonte 1..5, con el feature set ganador de cada modelo ---
    print(f"\n=== LSTM: horizonte 1-5 (con {lstm_n_features} features) ===")
    for horizon in range(1, 6):
        t0 = time.time()
        agg = run_combo(dfs, horizon, LSTM_WINDOW, lstm_n_features, lstm_train_fn, base.predict_lstm_local)
        dt = time.time() - t0
        row = {"horizon": horizon, "n": agg.total, "accuracy": agg.accuracy, "mae": agg.mae, "seconds": round(dt, 1)}
        results["lstm_horizon"].append(row)
        print(f"  horizon={horizon} -> acc={agg.accuracy:.3f} mae={agg.mae:.4f} n={agg.total} ({dt:.0f}s)")

    print(f"\n=== XGBoost: horizonte 1-5 (con {xgb_n_features} features) ===")
    for horizon in range(1, 6):
        t0 = time.time()
        agg = run_combo(dfs, horizon, XGB_WINDOW, xgb_n_features, xgb_train_fn, xgb_predict_fn)
        dt = time.time() - t0
        row = {"horizon": horizon, "n": agg.total, "accuracy": agg.accuracy, "mae": agg.mae, "seconds": round(dt, 1)}
        results["xgb_horizon"].append(row)
        print(f"  horizon={horizon} -> acc={agg.accuracy:.3f} mae={agg.mae:.4f} n={agg.total} ({dt:.0f}s)")

    lstm_best_h = max(results["lstm_horizon"], key=lambda r: r["accuracy"])
    xgb_best_h = max(results["xgb_horizon"], key=lambda r: r["accuracy"])
    print(f"\nmejor horizonte LSTM (1-5): {lstm_best_h['horizon']} (acc={lstm_best_h['accuracy']:.3f})")
    print(f"mejor horizonte XGBoost (1-5): {xgb_best_h['horizon']} (acc={xgb_best_h['accuracy']:.3f})")

    out_path = Path(__file__).parent / "tune_results_v2.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nresultados guardados en {out_path}")


if __name__ == "__main__":
    main()
