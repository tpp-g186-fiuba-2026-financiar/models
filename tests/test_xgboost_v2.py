"""Tests del XGBoost v2 (modelo pooled h=20 con rangos cross-seccionales).

Sin red ni Modal: historicos sinteticos y rutas de artefactos en un directorio
temporal. Lo que se verifica es lo que lo hace comparable con `api-ml`: mismas
variables, rangos calculados solo con pares "as of", lados sin ventaja apagados
y que el endpoint nunca entrene.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import xgboost_trend_model as xgb

TICKERS = list(xgb.LEADERS) + ["BBAR", "CRES"]
ROWS = 700


def _history(rows: int = ROWS, seed: int = 0, flat: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = np.full(rows, 100.0) if flat else 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, rows)))
    return pd.DataFrame(
        {
            "open": close * (1 + rng.normal(0, 0.002, rows)),
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": rng.integers(1_000, 50_000, rows).astype(float),
        },
        index=pd.bdate_range("2021-01-04", periods=rows),
    )


@pytest.fixture(scope="module")
def histories() -> dict[str, pd.DataFrame]:
    return {t: _history(seed=i) for i, t in enumerate(TICKERS)}


@pytest.fixture(scope="module")
def fast_state(histories):
    with patch.object(xgb, "N_SEEDS", 1), patch.object(xgb, "OOS_FOLDS", 2):
        return xgb.train_xgboost(histories)


@pytest.fixture(scope="module")
def snapshot(histories):
    return xgb.build_peer_snapshot(histories)


def test_feature_layout_matches_api_ml():
    assert xgb.ALL_FEATURES == xgb.FEATURES + xgb.RANK_FEATURES
    assert len(xgb.FEATURES) == 50 and len(set(xgb.ALL_FEATURES)) == 100


def test_panel_has_one_row_per_stock_and_day_with_ranks_inside_each_day(histories):
    market = xgb.market_frame(histories)
    panel = xgb.build_panel(histories, market, xgb.HORIZON)
    assert len(panel) == len(histories) * (ROWS - xgb.HORIZON)
    day = panel.loc[panel.index.unique()[300]]
    assert day["r_t_ret20"].between(0, 1).all()
    assert day.sort_values("t_ret20")["r_t_ret20"].is_monotonic_increasing


def test_flat_rsi_is_neutral_not_overbought():
    assert xgb.rsi(np.full(50, 10.0)) == 50.0


def test_rank_needs_enough_peers(histories, snapshot):
    own = xgb.build_features(histories["CRES"], snapshot["market"]).iloc[[-1]]
    rows = xgb.peers_as_of(snapshot["peers"], own.index[-1])
    with pytest.raises(ValueError):
        xgb.rank_against_peers(own, rows[: xgb.MIN_PEERS - 1])


def test_peers_are_taken_as_of_the_date_and_stale_peers_are_ignored(snapshot):
    last = pd.Timestamp(snapshot["as_of"])
    assert len(xgb.peers_as_of(snapshot["peers"], last)) == len(snapshot["peers"])
    assert xgb.peers_as_of(snapshot["peers"], last + timedelta(days=60)) == []
    early = snapshot["peers"]["GGAL"].index[2]
    rows = xgb.peers_as_of(snapshot["peers"], early)
    assert all(row.name <= early for row in rows)


def test_snapshot_drops_stale_series(histories):
    snap = xgb.build_peer_snapshot(histories | {"A3": _history(flat=True, seed=99)})
    assert "A3" not in snap["peers"]
    assert all(len(f) == xgb.PEER_TAIL_ROWS for f in snap["peers"].values())


def test_training_records_out_of_sample_metrics(fast_state):
    m = fast_state["metrics"]
    assert 0.0 <= m["oos_auc"] <= 1.0
    assert m["observations"] > 0 and m["series"] == []
    assert fast_state["baja_threshold"] <= fast_state["alza_threshold"]
    assert {"alza_disabled", "baja_disabled", "signal_hit_rate", "neutral_rate"} <= set(m)
    assert fast_state["horizon"] == xgb.HORIZON


def test_prediction_follows_calibrated_thresholds(fast_state, histories, snapshot):
    signal, log_return, p_up = xgb.predict_log_return(fast_state, histories["BBAR"], snapshot)
    if p_up >= fast_state["alza_threshold"]:
        assert (signal, log_return) == ("alza", fast_state["alza_move"])
    elif p_up <= fast_state["baja_threshold"]:
        assert (signal, log_return) == ("baja", fast_state["baja_move"])
    else:
        assert (signal, log_return) == ("neutral", 0.0)


def test_prediction_ignores_snapshot_data_after_the_cutoff(fast_state, histories, snapshot):
    cut = histories["BBAR"].iloc[:500]
    truncated = {
        "market": snapshot["market"],
        "peers": {s: f for s, f in snapshot["peers"].items()},
    }
    # Con el historico cortado antes de la foto no hay pares recientes: no se inventa un rango.
    with pytest.raises(ValueError):
        xgb.predict_log_return(fast_state, cut, truncated)


def test_a_side_without_edge_out_of_sample_is_never_emitted(histories):
    original = xgb.oos_scores

    def rigged(panel, horizon):
        oos = original(panel, horizon)
        noise = np.random.default_rng(0).normal(0, 0.01, len(oos))
        oos["p"] = np.where(oos["up"] == 1, 0.1, 0.9) + noise  # extremos al reves
        return oos

    with (
        patch.object(xgb, "N_SEEDS", 1),
        patch.object(xgb, "OOS_FOLDS", 2),
        patch.object(xgb, "oos_scores", rigged),
    ):
        state = xgb.train_xgboost(histories)
    assert state["metrics"]["alza_disabled"] is True
    assert state["alza_threshold"] == float("inf")


def test_training_rejects_too_little_data():
    few = {t: _history(rows=xgb.MIN_ROWS + 5, seed=i) for i, t in enumerate(xgb.LEADERS)}
    with pytest.raises(ValueError):
        xgb.train_xgboost(few)


def test_needs_retraining_by_age():
    now = datetime.now(timezone.utc)
    fresh = {"trained_at": (now - timedelta(days=3)).isoformat()}
    old = {"trained_at": (now - timedelta(days=40)).isoformat()}
    assert xgb.needs_retraining(None, now)
    assert not xgb.needs_retraining(fresh, now)
    assert xgb.needs_retraining(old, now)


@pytest.fixture
def volume(tmp_path, monkeypatch, histories):
    monkeypatch.setattr(xgb, "MODEL_PATH", tmp_path / "v2" / "production.pkl")
    monkeypatch.setattr(xgb, "PEERS_PATH", tmp_path / "v2" / "peers.pkl")
    monkeypatch.setattr(xgb, "fetch_universe_histories", lambda: histories)
    monkeypatch.setattr(xgb, "N_SEEDS", 1)
    monkeypatch.setattr(xgb, "OOS_FOLDS", 2)
    monkeypatch.setattr(xgb.artifact_volume, "commit", lambda: None)
    monkeypatch.setattr(xgb.artifact_volume, "reload", lambda: None)
    return tmp_path


def test_cron_refreshes_peers_daily_but_retrains_only_when_due(volume, monkeypatch):
    monkeypatch.setattr(xgb, "MIN_AUC", 0.0)
    first = xgb.run_retraining()
    assert first["promoted"] and xgb.MODEL_PATH.exists() and xgb.PEERS_PATH.exists()
    trained_at = xgb._load(xgb.MODEL_PATH)["trained_at"]
    second = xgb.run_retraining()
    assert second["promoted"] is False and "metrics" not in second
    assert xgb._load(xgb.MODEL_PATH)["trained_at"] == trained_at
    assert xgb.run_retraining(force=True)["promoted"]


def test_candidate_below_min_auc_is_not_promoted(volume, monkeypatch):
    monkeypatch.setattr(xgb, "MIN_AUC", 2.0)
    result = xgb.run_retraining()
    assert result["promoted"] is False
    assert not xgb.MODEL_PATH.exists()
    assert xgb.PEERS_PATH.exists()  # los pares se refrescan igual


def test_endpoint_predicts_with_the_stored_model_and_never_trains(volume, monkeypatch, histories):
    monkeypatch.setattr(xgb, "MIN_AUC", 0.0)
    xgb.run_retraining()
    monkeypatch.setattr(xgb, "fetch_ticker_history", lambda t: histories["BBAR"])
    result = xgb.main.local(ticker="bbar")
    assert result["model"] == "xgboost" and result["ticker"] == "BBAR"
    assert result["signal"] in {"alza", "baja", "neutral"}
    assert result["horizon_days"] == 20
    assert result["probability_up"] == pytest.approx(1 - result["probability_down"], abs=1e-3)
    assert {"as_of", "last_close", "predicted_close", "model_version", "backtest"} <= set(result)
    if result["signal"] == "neutral":
        assert result["predicted_close"] == pytest.approx(result["last_close"])


def test_endpoint_rejects_other_horizons_and_missing_artifacts(volume, histories, monkeypatch):
    assert "error" in xgb.main.local(ticker="GGAL", horizon=5)
    assert "error" in xgb.main.local(ticker="GGAL")  # todavia no hay modelo
    monkeypatch.setattr(xgb, "MIN_AUC", 0.0)
    xgb.run_retraining()
    monkeypatch.setattr(xgb, "fetch_ticker_history", lambda t: _history(rows=100))
    assert "error" in xgb.main.local(ticker="GGAL")
    monkeypatch.setattr(xgb, "fetch_ticker_history", lambda t: _history(flat=True))
    assert "error" in xgb.main.local(ticker="GGAL")


def test_prepare_trains_only_when_missing(volume, monkeypatch):
    monkeypatch.setattr(xgb, "MIN_AUC", 0.0)
    first = xgb.prepare.local(ticker="GGAL")
    assert first["status"] == "completed" and first["trained"] is True
    assert xgb.prepare.local(ticker="GGAL")["status"] == "ready"
