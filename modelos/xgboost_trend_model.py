"""Modelo XGBoost de tendencia (v2), portado a Modal desde api-ml.

Replica `api-ml/src/xgb_trend.py` (v2): un clasificador P(sube a 20 ruedas) con
arboles chicos y regularizados, variables tecnicas + contexto de mercado y el
**rango de cada variable frente al resto del universo ese mismo dia**. Solo
opina (alza/baja) en los extremos, con umbrales calibrados fuera de muestra;
en el resto devuelve `neutral`. Su accuracy global NO supera a "siempre sube":
lo que mide es cuanto mejor que la tasa base acierta cuando opina.

A diferencia de la v1 (un modelo por ticker y horizonte 1-5), la v2 es UN modelo
pooled sobre todo el universo y un horizonte fijo de 20 ruedas, porque la senal
es cross-seccional: para rankear una accion hace falta ver a las demas ese dia.
Por eso el cron diario hace dos cosas:

- refresca una **foto de los pares** (ultimas filas de variables de cada accion
  + contexto de mercado) en el Volume, para que el endpoint no tenga que bajar
  20 historicos por request;
- reentrena el modelo pooled solo si el artefacto no existe o tiene mas de
  `RETRAIN_EVERY_DAYS` dias, y lo promueve solo si el AUC fuera de muestra
  supera `MIN_AUC`.

El endpoint solo ejecuta inferencia.

Si se cambia la logica en api-ml hay que replicarla aca a mano -- es una
copia independiente a proposito, para que esto funcione aunque el servicio de
api-ml en Render este caido.
"""

import json
import os
import pickle
import time
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
    # la API sklearn-compatible (XGBClassifier), y sklearn se usa para el AUC.
    .apt_install("libgomp1")
    .pip_install("fastapi[standard]", "xgboost", "scikit-learn", "numpy", "pandas", "requests")
)
app = modal.App("xgboost-trend-model")
artifact_volume = modal.Volume.from_name("trend-model-artifacts", create_if_missing=True)
ARTIFACT_ROOT = Path("/artifacts/xgboost")
MODEL_PATH = ARTIFACT_ROOT / "v2" / "production.pkl"
PEERS_PATH = ARTIFACT_ROOT / "v2" / "peers.pkl"

DATA_COLLECTOR_URL = "https://data-colector.onrender.com"

HORIZON = 20
DEFAULT_HORIZON = HORIZON
MIN_ROWS = 260
RETRAIN_EVERY_DAYS = 28
MIN_AUC = 0.52  # AUC fuera de muestra minimo para promover un candidato
SIGNAL_QUANTILE = 0.10  # alza/baja solo en el 10% mas extremo de cada lado
OOS_FRACTION = 0.25
OOS_FOLDS = 4
N_SEEDS = 3
HALF_LIFE_YEARS = 3.0
MIN_PEERS = 5
PEER_MAX_AGE_DAYS = 10
PEER_TAIL_ROWS = 15
MARKET_TAIL_ROWS = 150
STALE_MAX_UNCHANGED = 0.30
# Canasta de lideres que hace de proxy del Merval (fija: el contexto es
# identico en entrenamiento y en prediccion).
LEADERS = ("GGAL", "YPFD", "BMA", "PAMP", "TXAR", "ALUA", "TECO2", "CEPU")
XGB_PARAMS = {
    "max_depth": 2,
    "n_estimators": 250,
    "learning_rate": 0.03,
    "subsample": 0.7,
    "colsample_bytree": 0.6,
    "min_child_weight": 200,
    "reg_lambda": 10.0,
}

TECH_FEATURES = [
    "t_ret1", "t_ret3", "t_ret5", "t_ret10", "t_ret20", "t_ret60", "t_ret120", "t_ret250",
    "t_down_streak", "t_up_streak", "t_dd252", "t_dd60", "t_dist_min60", "t_new_low252",
    "t_sma20", "t_sma50", "t_sma200", "t_death_cross", "t_gap", "t_gap_down",
    "t_vol5", "t_vol20", "t_vol60", "t_vol_ratio5_20", "t_vol_ratio20_60", "t_downdev",
    "t_skew20", "t_kurt20", "t_gk20", "t_gk_ratio", "t_volz", "t_volz_x_ret", "t_vol_chg5",
    "t_obv_slope", "t_amihud", "t_rsi14", "t_rsi5", "t_macd", "t_stoch", "t_boll",
]  # fmt: skip
CONTEXT_FEATURES = [
    "m_ret5", "m_ret20", "m_ret60", "m_vol20", "m_dd252", "m_breadth50",
    "m_rel5", "m_rel20", "m_rel60", "m_beta60",
]  # fmt: skip
FEATURES = TECH_FEATURES + CONTEXT_FEATURES
RANK_FEATURES = [f"r_{name}" for name in FEATURES]
ALL_FEATURES = FEATURES + RANK_FEATURES


def _post_with_retry(url: str, timeout: int = 90, attempts: int = 4) -> requests.Response:
    """POST al data-colector tolerando el arranque en frio de Render.

    El plan gratuito duerme el servicio y despertarlo tarda mas de 30 s: con
    el timeout viejo el cron fallaba sin reintentar. Se reintenta ante
    timeouts, errores de conexion y 5xx (502/503 mientras levanta).
    """
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            response = requests.post(url, timeout=timeout)
            if response.status_code < 500:
                return response
            last_error = requests.HTTPError(f"{response.status_code} en {url}", response=response)
        except (requests.Timeout, requests.ConnectionError) as error:
            last_error = error
        if attempt < attempts - 1:
            time.sleep(5 * 2**attempt)
    raise last_error


def fetch_available_tickers() -> list[str]:
    response = _post_with_retry(f"{DATA_COLLECTOR_URL}/available-tickers")
    response.raise_for_status()
    payload = response.json()
    tickers = payload.get("tickers") or (payload.get("message") or {}).get("tickers") or []
    if not tickers:
        raise ValueError("data-colector no devolvio tickers disponibles")
    return sorted({str(ticker).strip().upper() for ticker in tickers if str(ticker).strip()})


def fetch_ticker_history(ticker: str) -> pd.DataFrame:
    resp = _post_with_retry(f"{DATA_COLLECTOR_URL}/historical-data/{ticker}")
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
    """Repara saltos de precio que son casi seguro un error de dato: capa
    (winsoriza) cualquier retorno diario mas grande que `threshold` en valor
    absoluto y reconstruye la serie de precios. Devuelve la serie reparada y
    cuantos dias se tuvieron que capar."""
    log_close = np.log(close)
    log_return = np.diff(log_close)
    clipped = np.clip(log_return, -threshold, threshold)
    fixed = int(np.sum(clipped != log_return))
    log_close_clean = np.concatenate([[log_close[0]], log_close[0] + np.cumsum(clipped)])
    return np.exp(log_close_clean), fixed


def clean_dataframe(
    df: pd.DataFrame, threshold: float = BAD_DATA_LOG_RETURN_THRESHOLD
) -> tuple[pd.DataFrame, int]:
    cleaned, fixed = clean_close_series(df["close"].to_numpy(dtype=np.float64), threshold)
    df = df.copy()
    df["close"] = cleaned
    return df, fixed


def is_stale(df: pd.DataFrame) -> bool:
    """True si el precio no cambia en mas del 30% de las ruedas (ej. A3)."""
    if len(df) < 2:
        return True
    return bool((df["close"].diff().dropna() == 0).mean() > STALE_MAX_UNCHANGED)


# --------------------------------------------------------------------------- #
# Indicadores (identicos a api-ml/src/indicators.py)
# --------------------------------------------------------------------------- #
def _rsi_from_averages(avg_gain: float, avg_loss: float) -> float:
    if avg_loss == 0:
        return 50.0 if avg_gain == 0 else 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def rsi_series(close: np.ndarray, period: int = 14) -> np.ndarray:
    """RSI de Wilder, alineado a `close` (NaN en las primeras `period` posiciones)."""
    n = len(close)
    out = np.full(n, np.nan)
    if n < period + 1:
        return out
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = gain[:period].mean()
    avg_loss = loss[:period].mean()
    out[period] = _rsi_from_averages(avg_gain, avg_loss)
    for i in range(period, len(delta)):
        avg_gain = (avg_gain * (period - 1) + gain[i]) / period
        avg_loss = (avg_loss * (period - 1) + loss[i]) / period
        out[i + 1] = _rsi_from_averages(avg_gain, avg_loss)
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


def rsi(close: np.ndarray, period: int = 14):
    """RSI de Wilder sobre el ultimo valor de la serie (post-procesamiento)."""
    series = rsi_series(close, period)
    if len(series) == 0 or np.isnan(series[-1]):
        return None
    return float(series[-1])


# --------------------------------------------------------------------------- #
# Variables (identicas a api-ml/src/macro_trend.py::technical_features y
# api-ml/src/context_trend.py)
# --------------------------------------------------------------------------- #
def technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Indicadores tecnicos de la accion."""
    c, o, h, lo = df["close"], df["open"], df["high"], df["low"]
    volume = df["volume"].replace(0, np.nan)
    log_c = np.log(c)
    r = log_c.diff()
    f = pd.DataFrame(index=df.index)
    for k in (1, 3, 5, 10, 20, 60, 120, 250):
        f[f"t_ret{k}"] = log_c.diff(k)
    down, up = (r < 0).astype(int), (r > 0).astype(int)
    group = (down.diff() != 0).cumsum()
    f["t_down_streak"] = down.groupby(group).cumsum() * down
    group = (up.diff() != 0).cumsum()
    f["t_up_streak"] = up.groupby(group).cumsum() * up
    f["t_dd252"] = c / c.rolling(252, min_periods=60).max() - 1
    f["t_dd60"] = c / c.rolling(60).max() - 1
    f["t_dist_min60"] = c / c.rolling(60).min() - 1
    f["t_new_low252"] = (c <= c.rolling(252, min_periods=60).min() * 1.0001).astype(float)
    s20, s50 = c.rolling(20).mean(), c.rolling(50).mean()
    s200 = c.rolling(200, min_periods=120).mean()
    f["t_sma20"], f["t_sma50"], f["t_sma200"] = c / s20 - 1, c / s50 - 1, c / s200 - 1
    f["t_death_cross"] = (s50 < s200).astype(float)
    f["t_gap"] = o / c.shift() - 1
    f["t_gap_down"] = (f["t_gap"] < -0.02).astype(float)
    for k in (5, 20, 60):
        f[f"t_vol{k}"] = r.rolling(k).std()
    f["t_vol_ratio5_20"] = f["t_vol5"] / f["t_vol20"]
    f["t_vol_ratio20_60"] = f["t_vol20"] / f["t_vol60"]
    f["t_downdev"] = r.where(r < 0).rolling(20, min_periods=3).std() / f["t_vol20"]
    f["t_skew20"], f["t_kurt20"] = r.rolling(20).skew(), r.rolling(20).kurt()
    hl = np.log(h / lo.replace(0, np.nan)) ** 2
    co = np.log(c / o.replace(0, np.nan)) ** 2
    gk = (0.5 * hl - (2 * np.log(2) - 1) * co).clip(lower=0)
    f["t_gk20"] = np.sqrt(gk.rolling(20).mean())
    f["t_gk_ratio"] = np.sqrt(gk.rolling(5).mean()) / f["t_gk20"].replace(0, np.nan)
    log_volume = np.log1p(df["volume"])
    f["t_volz"] = (log_volume - log_volume.rolling(20).mean()) / log_volume.rolling(20).std()
    f["t_volz_x_ret"] = f["t_volz"].fillna(0) * np.sign(r)
    f["t_vol_chg5"] = log_volume.diff(5)
    obv = (np.sign(r).fillna(0) * df["volume"]).cumsum()
    f["t_obv_slope"] = (obv - obv.shift(20)) / (df["volume"].rolling(20).sum() + 1)
    amihud = (r.abs() / (c * volume)).rolling(20).mean()
    f["t_amihud"] = np.log1p(amihud / (amihud.rolling(250, min_periods=60).median() + 1e-12))
    close = np.asarray(c, dtype=float)
    f["t_rsi14"], f["t_rsi5"] = rsi_series(close), rsi_series(close, 5)
    f["t_macd"] = macd_histogram(close) / close
    lo14, hi14 = lo.rolling(14).min(), h.rolling(14).max()
    f["t_stoch"] = (c - lo14) / (hi14 - lo14).replace(0, np.nan)
    f["t_boll"] = (c - s20) / c.rolling(20).std()
    return f.replace([np.inf, -np.inf], np.nan)


def market_frame(histories: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Serie del mercado (canasta de lideres) a partir de sus historicos."""
    closes = pd.DataFrame({s: d["close"] for s, d in histories.items() if s in LEADERS})
    if closes.shape[1] < 3:
        raise ValueError("faltan historicos de los lideres para armar el contexto")
    returns = np.log(closes).diff()
    mkt = returns.mean(axis=1).fillna(0.0)
    index = mkt.cumsum()
    frame = pd.DataFrame(index=closes.index)
    frame["m_lr"] = mkt
    for k in (5, 20, 60):
        frame[f"m_ret{k}"] = index.diff(k)
    frame["m_vol20"] = mkt.rolling(20).std()
    frame["m_dd252"] = index - index.rolling(252, min_periods=100).max()
    frame["m_breadth50"] = (closes > closes.rolling(50).mean()).where(closes.notna()).mean(axis=1)
    return frame


def context_features(df: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    """Variables de contexto de una accion, alineadas a su calendario."""
    m = market.reindex(df.index, method="ffill")
    log_c = np.log(df["close"])
    r = log_c.diff()
    f = pd.DataFrame(index=df.index)
    for k in (5, 20, 60):
        f[f"m_ret{k}"] = m[f"m_ret{k}"]
    f["m_vol20"], f["m_dd252"], f["m_breadth50"] = m["m_vol20"], m["m_dd252"], m["m_breadth50"]
    for k in (5, 20, 60):
        f[f"m_rel{k}"] = log_c.diff(k) - m[f"m_ret{k}"]
    f["m_beta60"] = r.rolling(60).cov(m["m_lr"]) / m["m_lr"].rolling(60).var()
    return f.replace([np.inf, -np.inf], np.nan)


def build_features(df: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    """Variables de una accion, en el orden de `FEATURES`."""
    return technical_features(df).join(context_features(df, market))[FEATURES]


def rank_against_peers(
    own: pd.DataFrame, peer_rows: list[pd.Series], min_peers: int = MIN_PEERS
) -> pd.DataFrame:
    """Percentil de cada variable de `own` (1 fila) frente a los pares ese dia."""
    if len(peer_rows) < min_peers:
        raise ValueError(f"faltan acciones para rankear (hay {len(peer_rows)})")
    both = pd.concat([pd.DataFrame(peer_rows)[FEATURES], own[FEATURES]], ignore_index=True)
    ranked = both.rank(pct=True).iloc[[-1]]
    ranked.columns = RANK_FEATURES
    ranked.index = own.index
    return ranked


def peers_as_of(peers: dict[str, pd.DataFrame], asof: pd.Timestamp) -> list[pd.Series]:
    """Ultima fila conocida de cada par hasta `asof` (ignora pares sin datos recientes)."""
    rows = []
    for frame in peers.values():
        part = frame.loc[:asof]
        if len(part) and (asof - part.index[-1]).days <= PEER_MAX_AGE_DAYS:
            rows.append(part.iloc[-1])
    return rows


def build_peer_snapshot(histories: dict[str, pd.DataFrame]) -> dict:
    """Foto de los pares para inferencia: contexto de mercado y ultimas filas de variables."""
    market = market_frame(histories)
    peers = {
        symbol: build_features(df, market).tail(PEER_TAIL_ROWS)
        for symbol, df in histories.items()
        if len(df) >= MIN_ROWS and not is_stale(df)
    }
    return {
        "market": market.tail(MARKET_TAIL_ROWS),
        "peers": peers,
        "as_of": market.index[-1].strftime("%Y-%m-%d"),
        "built_at": datetime.now(timezone.utc).isoformat(),
    }


# --------------------------------------------------------------------------- #
# Entrenamiento
# --------------------------------------------------------------------------- #
def wilson_interval(hits: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Intervalo de confianza (95%) de una proporcion: cuanta confianza dar a una accuracy."""
    if n == 0:
        return 0.0, 1.0
    p = hits / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return float(max(0.0, center - half)), float(min(1.0, center + half))


def build_panel(histories: dict[str, pd.DataFrame], market: pd.DataFrame, horizon: int) -> pd.DataFrame:
    frames = []
    for symbol, df in histories.items():
        f = build_features(df, market)
        f["fwd"] = df["close"].shift(-horizon) / df["close"] - 1.0
        f["ticker"] = symbol
        frames.append(f)
    panel = pd.concat(frames)
    panel = panel[panel["fwd"].notna() & (panel["fwd"] != 0)].sort_index(kind="stable")
    # El indice (fecha) se repite entre acciones: asignar por posicion, no con join.
    panel[RANK_FEATURES] = panel.groupby(level=0)[FEATURES].rank(pct=True).to_numpy()
    panel["up"] = (panel["fwd"] > 0).astype(int)
    return panel


def _decay_weights(index: pd.Index, cutoff: pd.Timestamp) -> np.ndarray:
    age_years = (cutoff - pd.DatetimeIndex(index)).days.to_numpy() / 365.0
    return 0.5 ** (age_years / HALF_LIFE_YEARS)


def _fit_ensemble(x: np.ndarray, y: np.ndarray, weights: np.ndarray, seed: int = 0) -> list:
    import xgboost as xgb

    models = []
    for k in range(N_SEEDS):
        model = xgb.XGBClassifier(
            **XGB_PARAMS, random_state=seed + k, n_jobs=4, tree_method="hist", eval_metric="logloss"
        )
        models.append(model.fit(x, y, sample_weight=weights))
    return models


def _score(models: list, x: np.ndarray) -> np.ndarray:
    """P(sube): promedio del ensamble."""
    return np.mean([m.predict_proba(x)[:, 1] for m in models], axis=0)


def oos_scores(panel: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Predicciones fuera de muestra del ultimo tramo, con reentrenos sucesivos y purga."""
    days = np.sort(panel.index.unique())
    first = int(len(days) * (1 - OOS_FRACTION))
    edges = np.linspace(first, len(days), OOS_FOLDS + 1).astype(int)
    index, x, y = panel.index.values, panel[ALL_FEATURES].to_numpy(), panel["up"].to_numpy()
    parts = []
    for k in range(OOS_FOLDS):
        start = days[edges[k]]
        end = days[edges[k + 1]] if edges[k + 1] < len(days) else None
        purge = days[max(edges[k] - horizon, 0)]
        tr = np.where(index <= purge)[0]
        te = np.where((index >= start) & ((index < end) if end is not None else True))[0]
        if len(tr) < 3000 or len(te) == 0:
            continue
        weights = _decay_weights(panel.index[tr], pd.Timestamp(purge))
        models = _fit_ensemble(x[tr], y[tr], weights)
        parts.append(
            pd.DataFrame(
                {"p": _score(models, x[te]), "up": y[te], "fwd": panel["fwd"].to_numpy()[te]},
                index=panel.index[te],
            )
        )
    if not parts:
        raise ValueError("no hay historia suficiente para medir fuera de muestra")
    return pd.concat(parts)


def train_xgboost(histories: dict[str, pd.DataFrame], horizon: int = HORIZON) -> dict:
    """Entrena el modelo pooled, calibra umbrales fuera de muestra y mide el candidato."""
    from sklearn.metrics import roc_auc_score

    histories = {s: d for s, d in histories.items() if len(d) >= MIN_ROWS and not is_stale(d)}
    if not histories:
        raise ValueError("no hay suficientes datos para entrenar")
    market = market_frame(histories)
    panel = build_panel(histories, market, horizon)
    if len(panel) < 3000:
        raise ValueError(f"muy pocas filas para entrenar ({len(panel)})")

    oos = oos_scores(panel, horizon)
    p, up = oos["p"].to_numpy(), oos["up"].to_numpy()
    alza_thr = float(np.quantile(p, 1 - SIGNAL_QUANTILE))
    baja_thr = float(np.quantile(p, SIGNAL_QUANTILE))
    fwd_log = np.log1p(oos["fwd"].to_numpy())
    is_alza, is_baja = p >= alza_thr, p <= baja_thr
    ups, downs = fwd_log[is_alza & (fwd_log > 0)], fwd_log[is_baja & (fwd_log < 0)]
    alza_move = float(np.median(ups)) if len(ups) else 0.0
    baja_move = float(np.median(downs)) if len(downs) else 0.0
    alza_hits = float(up[is_alza].mean()) if is_alza.any() else float("nan")
    baja_hits = float(1 - up[is_baja].mean()) if is_baja.any() else float("nan")
    base_up = float(up.mean())
    # Un lado solo se emite si fuera de muestra le gano a su tasa base.
    alza_off, baja_off = not alza_hits > base_up, not baja_hits > 1 - base_up
    if alza_off:
        alza_thr, is_alza = float("inf"), np.zeros_like(is_alza)
    if baja_off:
        baja_thr, is_baja = float("-inf"), np.zeros_like(is_baja)

    position = np.where(is_alza, 1.0, np.where(is_baja, -1.0, 0.0))
    predicted_move = np.where(is_alza, alza_move, np.where(is_baja, baja_move, 0.0))
    active = position != 0
    hits = int(np.sum(np.sign(fwd_log[active]) == position[active]))
    n = len(fwd_log)
    low, high = wilson_interval(hits, n)
    metrics = {
        # neutral cuenta como error (igual que el gate de api-ml): es la accuracy
        # global del modelo, comparable con "siempre sube" (`base_up_rate`).
        "directional_accuracy": hits / n,
        "accuracy_low": low,
        "accuracy_high": high,
        "signal_hit_rate": float(hits / active.sum()) if active.any() else None,
        "neutral_rate": float(np.mean(~active)),
        "mae": float(np.mean(np.abs(predicted_move - fwd_log))),
        "observations": n,
        "folds": OOS_FOLDS,
        "series": [],
        "oos_auc": float(roc_auc_score(up, p)),
        "base_up_rate": base_up,
        "alza_hit_rate": None if np.isnan(alza_hits) else alza_hits,
        "baja_hit_rate": None if np.isnan(baja_hits) else baja_hits,
        "alza_disabled": bool(alza_off),
        "baja_disabled": bool(baja_off),
    }
    weights = _decay_weights(panel.index, pd.Timestamp(panel.index.max()))
    models = _fit_ensemble(panel[ALL_FEATURES].to_numpy(), panel["up"].to_numpy(), weights)
    return {
        "models": models,
        "alza_threshold": alza_thr,
        "baja_threshold": baja_thr,
        "alza_move": alza_move,
        "baja_move": baja_move,
        "metrics": metrics,
        "tickers": sorted(histories),
        "horizon": horizon,
    }


def predict_log_return(state: dict, df: pd.DataFrame, snapshot: dict) -> tuple[str, float, float]:
    """(senal, retorno log proyectado, P(sube)) de `df` frente a los pares de `snapshot`."""
    asof = df.index[-1]
    # Contexto "as of": solo filas hasta la ultima fecha de `df`.
    market = snapshot["market"].loc[:asof]
    if market.empty:
        raise ValueError("la foto de pares es posterior a los datos de la accion")
    own = build_features(df, market).iloc[[-1]]
    ranked = rank_against_peers(own, peers_as_of(snapshot["peers"], asof))
    row = own.join(ranked)[ALL_FEATURES].to_numpy()
    p_up = float(_score(state["models"], row)[0])
    if p_up >= state["alza_threshold"]:
        return "alza", state["alza_move"], p_up
    if p_up <= state["baja_threshold"]:
        return "baja", state["baja_move"], p_up
    return "neutral", 0.0, p_up


def derive_trend_output(df: pd.DataFrame, log_return: float, horizon: int, neutral_band: float = 0.0) -> dict:
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


# --------------------------------------------------------------------------- #
# Persistencia y cron
# --------------------------------------------------------------------------- #
def _dump_atomic(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as fh:
        pickle.dump(obj, fh)
    os.replace(temporary, path)


def _load(path: Path):
    with path.open("rb") as fh:
        return pickle.load(fh)


def fetch_universe_histories() -> dict[str, pd.DataFrame]:
    histories = {}
    for ticker in fetch_available_tickers():
        try:
            df, _ = clean_dataframe(fetch_ticker_history(ticker))
            histories[ticker] = df
        except Exception as exc:  # un ticker no cancela el resto del lote
            print(f"  [skip] {ticker}: {exc}")
    return histories


def needs_retraining(artifact: dict | None, now: datetime, every_days: int = RETRAIN_EVERY_DAYS) -> bool:
    if artifact is None or "trained_at" not in artifact:
        return True
    trained_at = datetime.fromisoformat(artifact["trained_at"])
    return (now - trained_at).days >= every_days


def run_retraining(force: bool = False) -> dict:
    """Refresca la foto de pares y, si hace falta, reentrena el modelo pooled."""
    histories = fetch_universe_histories()
    if not histories:
        raise ValueError("no se pudo descargar data de ningun ticker")
    snapshot = build_peer_snapshot(histories)
    _dump_atomic(snapshot, PEERS_PATH)

    incumbent = _load(MODEL_PATH) if MODEL_PATH.exists() else None
    result = {"peers": len(snapshot["peers"]), "peers_as_of": snapshot["as_of"], "promoted": False}
    if force or needs_retraining(incumbent, datetime.now(timezone.utc)):
        state = train_xgboost(histories)
        result["metrics"] = state["metrics"]
        if state["metrics"]["oos_auc"] >= MIN_AUC:
            _dump_atomic(
                {
                    "state": state,
                    "metrics": state["metrics"],
                    "horizon": state["horizon"],
                    "trained_at": datetime.now(timezone.utc).isoformat(),
                    "training_data_as_of": snapshot["as_of"],
                },
                MODEL_PATH,
            )
            result["promoted"] = True
    artifact_volume.commit()
    return result


@app.function(
    image=image,
    volumes={"/artifacts": artifact_volume},
    schedule=modal.Cron("0 21 * * 1-5", timezone="America/Argentina/Buenos_Aires"),
    timeout=86400,
    cpu=1.0,
    memory=2048,
)
def retrain_models() -> dict:
    """Cada dia habil refresca los pares; reentrena el modelo cada ~4 semanas.

    Tambien se puede ejecutar a mano desde Modal (`modal run ...::retrain_models`).
    """
    # `fetch_available_tickers()` define el universo (ver `fetch_universe_histories`).
    result = run_retraining()
    print(json.dumps(result, default=str))
    return result


@app.function(
    image=image, volumes={"/artifacts": artifact_volume}, timeout=3600,
    cpu=1.0, memory=2048,
)
@modal.fastapi_endpoint()
def prepare(ticker: str) -> dict:
    """Bootstrap bajo demanda: entrena el modelo pooled solo si todavia no existe."""
    ticker = ticker.strip().upper()
    artifact_volume.reload()
    if MODEL_PATH.exists() and PEERS_PATH.exists():
        return {"ticker": ticker, "status": "ready", "trained": False}
    try:
        result = run_retraining(force=not MODEL_PATH.exists())
    except Exception as exc:
        return {"ticker": ticker, "status": "error", "error": str(exc)}
    return {"ticker": ticker, "status": "completed", "trained": result["promoted"], "result": result}


# Recursos del contenedor: 3 ensambles de 250 arboles de profundidad 2 sobre
# una fila de variables por request. Inferencia liviana: se lee el artefacto y
# la foto de pares del Volume y solo se descarga el historico del ticker pedido.
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
    if horizon != HORIZON:
        return {"error": f"el modelo XGBoost v2 solo predice a {HORIZON} dias (se pidio {horizon})"}

    ticker = ticker.strip().upper()
    artifact_volume.reload()
    if not MODEL_PATH.exists() or not PEERS_PATH.exists():
        return {"error": "todavia no hay un modelo entrenado para el universo (corre el cron o 'prepare')"}
    artifact = _load(MODEL_PATH)
    snapshot = _load(PEERS_PATH)

    df = fetch_ticker_history(ticker)
    if len(df) < MIN_ROWS:
        return {"error": f"se necesitan al menos {MIN_ROWS} ruedas historicas para '{ticker}' (hay {len(df)})"}
    df, days_repaired = clean_dataframe(df)
    if is_stale(df):
        return {"error": f"'{ticker}' casi no tiene movimiento de precio (>30% de ruedas sin cambio)"}

    try:
        signal, log_return, p_up = predict_log_return(artifact["state"], df, snapshot)
    except ValueError as exc:
        return {"error": str(exc)}

    result = derive_trend_output(df, log_return, horizon)
    result["signal"] = signal  # neutral => precio proyectado = ultimo cierre
    result["probability_up"] = round(p_up, 4)
    result["probability_down"] = round(1 - p_up, 4)
    result["ticker"] = ticker
    result["model"] = "xgboost"
    result["source"] = "modal (artefacto preentrenado)"
    result["model_version"] = artifact["trained_at"]
    result["backtest"] = artifact["metrics"]
    result["peers_as_of"] = snapshot["as_of"]
    if days_repaired:
        result["price_data_repaired_days"] = days_repaired
    return result
