"""Tests funcionales de `transformer_trend_model.py` con datos sinteticos.

No tocan la red ni Modal: `fetch_*` se mockea y el Volume se reemplaza por
un directorio temporal. El entrenamiento es real (red chica, pocas epocas
gracias al early stopping), asi se verifica de punta a punta que el modelo
entrena, promueve, persiste y predice.
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import transformer_trend_model as tm


def _history(n: int = 420, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, n)))
    index = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {
            "open": close * (1 + rng.normal(0, 0.002, n)),
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": rng.integers(1_000, 5_000, n).astype(float),
        },
        index=index,
    )


@pytest.fixture
def artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(tm, "ARTIFACT_ROOT", tmp_path)
    monkeypatch.setattr(tm.artifact_volume, "commit", lambda: None)
    monkeypatch.setattr(tm.artifact_volume, "reload", lambda: None)
    monkeypatch.setattr(tm, "fetch_ticker_history", lambda ticker: _history())
    monkeypatch.setattr(tm, "fetch_macro_series", lambda *a, **k: None)
    return tmp_path


def test_build_features_has_the_shared_feature_set():
    feats, target = tm.build_features(_history(), horizon=5)
    assert feats.shape[1] == len(tm.FEATURE_NAMES) == 8
    assert len(feats) == len(target)
    assert np.isfinite(feats).all()
    # los ultimos `horizon` targets no existen todavia
    assert np.isnan(target[-5:]).all()


def test_network_forward_shape():
    import torch

    net = tm.build_transformer_network()
    out = net(torch.zeros(3, tm.WINDOW, len(tm.FEATURE_NAMES)))
    assert tuple(out.shape) == (3,)


def test_train_transformer_requires_enough_history():
    with pytest.raises(ValueError):
        tm.train_transformer.local(_history(60), 5)


def test_train_and_predict_returns_a_finite_log_return():
    df = _history()
    state = tm.train_transformer.local(df, 5)
    feats, _ = tm.build_features(df, 5)
    log_return = tm.predict_transformer_state(state, feats[-tm.WINDOW :][None, :, :])
    assert np.isfinite(log_return)
    assert abs(log_return) < 1.0


def test_backtest_reports_direction_accuracy_and_series():
    metrics = tm.backtest_transformer(_history(), 5)
    assert 0.0 <= metrics["directional_accuracy"] <= 1.0
    assert metrics["mae"] >= 0
    assert metrics["observations"] > 0
    assert 0 < len(metrics["series"]) <= 30
    assert {"date", "predicted", "actual"} <= set(metrics["series"][0])


def test_backtest_requires_enough_history():
    with pytest.raises(ValueError):
        tm.backtest_transformer(_history(120), 5)


def test_retrain_one_promotes_then_only_replaces_with_a_better_score(artifacts):
    first = tm.retrain_one("ggal", 5)
    assert first["promoted"] is True
    assert first["ticker"] == "GGAL"
    assert (artifacts / "GGAL" / "h5" / "production.pkl").exists()

    # Con la misma historia y semilla el candidato empata: no supera al
    # vigente, asi que no se reemplaza.
    second = tm.retrain_one("GGAL", 5)
    assert second["promoted"] is False


def test_endpoint_serves_a_prediction_from_the_saved_artifact(artifacts):
    tm.retrain_one("GGAL", 5)

    result = tm.main.local(ticker="ggal", horizon=5)

    assert "error" not in result
    assert result["ticker"] == "GGAL"
    assert result["model"] == "transformer"
    assert result["signal"] in {"alza", "baja", "neutral"}
    assert result["horizon_days"] == 5
    assert result["backtest"]["directional_accuracy"] is not None
    assert result["source"].startswith("modal")


def test_endpoint_rejects_invalid_horizon_and_short_history(artifacts):
    assert "horizon" in tm.main.local(ticker="GGAL", horizon=99)["error"]
    tm.retrain_one("GGAL", 5)
    with patch.object(tm, "fetch_ticker_history", lambda ticker: _history(40)):
        assert "mas ruedas" in tm.main.local(ticker="GGAL", horizon=5)["error"]


def test_prepare_only_trains_missing_horizons(artifacts):
    with patch.object(tm, "MAX_HORIZON", 2):
        first = tm.prepare.local("GGAL")
        assert first["status"] == "completed"
        assert {r["horizon"] for r in first["results"]} == {1, 2}
        assert tm.prepare.local("GGAL")["status"] == "ready"


def test_retrain_models_keeps_going_when_a_ticker_fails(artifacts):
    def flaky_history(ticker):
        if ticker == "BAD":
            raise ValueError("sin datos")
        return _history()

    with patch.object(tm, "MAX_HORIZON", 1), patch.object(
        tm, "fetch_available_tickers", lambda: ["BAD", "GGAL"]
    ), patch.object(tm, "fetch_ticker_history", flaky_history):
        results = tm.retrain_models.local()

    by_ticker = {r["ticker"]: r for r in results}
    assert "error" in by_ticker["BAD"]
    assert by_ticker["GGAL"]["promoted"] is True


def test_lstm_cron_spawns_the_transformer_retraining_and_survives_failures(monkeypatch):
    import lstm_trend_model as lm

    monkeypatch.setattr(lm, "fetch_available_tickers", lambda: [])
    spawned = []

    class FakeFunction:
        def spawn(self):
            spawned.append(True)

    calls = []

    def fake_from_name(app_name, function_name):
        calls.append((app_name, function_name))
        return FakeFunction()

    monkeypatch.setattr(lm.modal.Function, "from_name", staticmethod(fake_from_name))
    assert lm.retrain_models.local() == []
    assert calls == [("transformer-trend-model", "retrain_models")]
    assert spawned == [True]

    def broken(*args, **kwargs):
        raise RuntimeError("no desplegado")

    monkeypatch.setattr(lm.modal.Function, "from_name", staticmethod(broken))
    assert lm.retrain_models.local() == []
