from __future__ import annotations

import pickle
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

import arima_model
import example_model
import garch_model
import lstm_trend_model as lstm
import svm_model
import xgboost_trend_model as xgb


def history(rows: int = 150) -> pd.DataFrame:
    close = np.linspace(100.0, 140.0, rows) + np.sin(np.arange(rows))
    return pd.DataFrame(
        {
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.linspace(1_000.0, 2_000.0, rows),
            "macro_rate": np.linspace(4.0, 4.5, rows),
        },
        index=pd.date_range("2026-01-01", periods=rows),
    )


def collector_rows(rows: int = 150) -> list[dict]:
    frame = history(rows)
    return [
        {
            "open_amount": str(row.open),
            "high_amount": str(row.high),
            "low_amount": str(row.low),
            "close_amount": str(row.close),
            "volume": str(row.volume),
            "ts": int(index.timestamp() * 1000),
        }
        for index, row in frame.iterrows()
    ]


def response(payload):
    value = Mock()
    value.json.return_value = payload
    return value


class PickleBooster:
    def __init__(self, **kwargs):
        self.rows = 0

    def fit(self, x, y, **kwargs):
        self.rows = len(x)

    def predict(self, values):
        return np.full(len(values), 0.02)


class PickleArimaFit:
    def forecast(self, steps):
        return pd.Series(np.linspace(120, 125, steps))


class FakeVariance:
    def __init__(self, values):
        self.iloc = self
        self.values = np.asarray(values)

    def __getitem__(self, item):
        return self

    def to_numpy(self, dtype=None):
        return self.values.astype(dtype)

    def to_dict(self):
        return {"h.1": {0: float(self.values.flat[0])}}


class PickleGarchFit:
    def forecast(self, horizon):
        return SimpleNamespace(variance=FakeVariance(np.ones(horizon)))


class PickleClassifier:
    def predict(self, values):
        return np.zeros(len(values), dtype=int)


@pytest.mark.parametrize("module", [lstm, xgb])
def test_data_collection_and_macro_fallbacks(monkeypatch, module):
    post = Mock(
        side_effect=[
            response({"tickers": [" ggal ", "YPFD", "GGAL"]}),
            response({"data": collector_rows(3)}),
            response(
                {
                    "status": 200,
                    "data": [
                        {"value": "4.1", "ts": 1_767_225_600_000},
                        {"value": "4.2", "ts": 1_767_312_000_000},
                    ],
                }
            ),
            response({"status": 500, "message": {"error": "offline"}}),
        ]
    )
    monkeypatch.setattr(module.requests, "post", post)
    assert module.fetch_available_tickers() == ["GGAL", "YPFD"]
    frame = module.fetch_ticker_history("GGAL")
    assert list(frame.columns) == ["open", "high", "low", "close", "volume"]
    assert len(module.fetch_macro_series()) == 2
    assert module.fetch_macro_series() is None

    monkeypatch.setattr(module.requests, "post", Mock(return_value=response({"tickers": []})))
    with pytest.raises(ValueError):
        module.fetch_available_tickers()
    monkeypatch.setattr(module.requests, "post", Mock(return_value=response({"data": []})))
    with pytest.raises(ValueError):
        module.fetch_ticker_history("NONE")


@pytest.mark.parametrize("module", [lstm, xgb])
def test_feature_engineering_helpers(module, monkeypatch):
    frame = history()
    broken = np.array([100.0, 101.0, 1_000.0, 102.0])
    cleaned, fixed = module.clean_close_series(broken, threshold=0.5)
    assert fixed == 2
    assert np.isfinite(cleaned).all()
    cleaned_frame, fixed = module.clean_dataframe(
        pd.DataFrame({"close": broken}), threshold=0.5
    )
    assert fixed == 2
    assert len(cleaned_frame) == 4

    missing_macro = module.attach_macro_feature(frame.drop(columns="macro_rate"), None)
    assert missing_macro["macro_rate"].isna().all()
    macro = pd.Series([3.0, 4.0], index=[frame.index[0], frame.index[30]])
    attached = module.attach_macro_feature(frame.drop(columns="macro_rate"), macro)
    assert attached["macro_rate"].iloc[40] == 4.0

    assert np.isnan(module.rsi_series(np.arange(5.0), period=14)).all()
    assert module.rsi_series(np.arange(20.0), period=14)[-1] == 100.0
    mixed_rsi = module.rsi_series(np.array([1.0, 2.0, 1.0, 3.0, 2.0]), period=2)
    assert np.isfinite(mixed_rsi[-1])
    assert np.isnan(module.sma(np.arange(3.0), 5)).all()
    assert module.sma(np.arange(5.0), 3)[-1] == 3.0
    assert len(module.ema(np.arange(5.0), 3)) == 5
    assert len(module.macd_histogram(np.arange(1.0, 10.0))) == 9

    feats, target = module.build_features(frame, horizon=3)
    assert feats.shape == (len(frame) - 1, len(module.FEATURE_NAMES))
    assert np.isfinite(feats).all()
    windows, values = module.make_windows(feats, target, window=10)
    assert windows.shape[0] == values.shape[0] > 0
    empty_x, empty_y = module.make_windows(feats[:2], np.array([np.nan, np.nan]), 1)
    assert len(empty_x) == len(empty_y) == 0
    assert module.rsi(np.arange(5.0), period=14) is None

    monkeypatch.setattr(module, "rsi", lambda close: None)
    assert module.derive_trend_output(frame, 0.1, 5)["condition"] == "indeterminado"
    monkeypatch.setattr(module, "rsi", lambda close: 75.0)
    assert module.derive_trend_output(frame, -0.1, 5)["signal"] == "baja"
    monkeypatch.setattr(module, "rsi", lambda close: 25.0)
    assert module.derive_trend_output(frame, 0.0, 5)["condition"] == "sobreventa"
    monkeypatch.setattr(module, "rsi", lambda close: 50.0)
    assert module.derive_trend_output(frame, 0.0, 5)["condition"] == "neutral"


def test_lstm_network_training_prediction_and_backtest(monkeypatch):
    import torch

    network = lstm.build_lstm_network()
    output = network(torch.zeros(2, lstm.WINDOW, len(lstm.FEATURE_NAMES)))
    assert output.shape == (2,)

    feats = np.ones((80, len(lstm.FEATURE_NAMES)), dtype=np.float32)
    feats[:, 0] = np.linspace(0, 1, 80)
    target = np.linspace(-0.1, 0.1, 80)
    monkeypatch.setattr(lstm, "build_features", lambda df, horizon: (feats, target))
    state = lstm.train_lstm.local(history())
    prediction = lstm.predict_lstm_state(
        state, feats[-lstm.WINDOW :][None, :, :]
    )
    assert np.isfinite(prediction)

    monkeypatch.setattr(
        lstm,
        "train_lstm",
        SimpleNamespace(local=lambda df, horizon: {"state": True}),
    )
    monkeypatch.setattr(lstm, "predict_lstm_state", lambda state, window: 0.02)
    backtest_feats = np.ones((159, len(lstm.FEATURE_NAMES)), dtype=np.float32)
    backtest_target = np.linspace(-0.1, 0.1, 159)
    backtest_target[-2:] = np.nan
    monkeypatch.setattr(
        lstm, "build_features", lambda df, horizon: (backtest_feats, backtest_target)
    )
    metrics = lstm.backtest_lstm(history(160), 2)
    assert metrics["observations"] > 0
    with pytest.raises(ValueError):
        lstm.backtest_lstm(history(30), 2)


def test_xgboost_training_and_backtest(monkeypatch):
    import xgboost

    monkeypatch.setattr(xgboost, "XGBRegressor", PickleBooster)
    feats = np.ones((80, len(xgb.FEATURE_NAMES)), dtype=np.float32)
    target = np.linspace(-0.1, 0.1, 80)
    monkeypatch.setattr(xgb, "build_features", lambda df, horizon: (feats, target))
    state = xgb.train_xgboost.local(history())
    assert state["booster"].rows > 0
    assert xgb.flatten_windows(feats[:2], np.array([np.nan, np.nan]), 1)[0].size == 0

    monkeypatch.setattr(
        xgb,
        "train_xgboost",
        SimpleNamespace(local=lambda df, horizon: {"booster": PickleBooster()}),
    )
    backtest_feats = np.ones((159, len(xgb.FEATURE_NAMES)), dtype=np.float32)
    backtest_target = np.linspace(-0.1, 0.1, 159)
    backtest_target[-2:] = np.nan
    monkeypatch.setattr(
        xgb, "build_features", lambda df, horizon: (backtest_feats, backtest_target)
    )
    metrics = xgb.backtest_xgboost(history(160), 2)
    assert metrics["observations"] > 0
    with pytest.raises(ValueError):
        xgb.backtest_xgboost(history(30), 2)


@pytest.mark.parametrize("module", [lstm, xgb])
def test_trend_retraining_batch_prepare_and_main(monkeypatch, tmp_path, module):
    monkeypatch.setattr(module, "ARTIFACT_ROOT", tmp_path)
    monkeypatch.setattr(module, "fetch_ticker_history", lambda ticker: history())
    monkeypatch.setattr(module, "fetch_macro_series", lambda: None)
    monkeypatch.setattr(module.artifact_volume, "commit", Mock())
    monkeypatch.setattr(module.artifact_volume, "reload", Mock())
    metric = {
        "directional_accuracy": 0.75,
        "mae": 0.1,
        "observations": 2,
        "series": [{"date": "2026-01-01", "predicted": 1, "actual": 1}],
    }
    if module is lstm:
        monkeypatch.setattr(module, "backtest_lstm", lambda df, horizon: metric)
        monkeypatch.setattr(
            module,
            "train_lstm",
            SimpleNamespace(local=lambda df, horizon: {"weights": True}),
        )
    else:
        monkeypatch.setattr(module, "backtest_xgboost", lambda df, horizon: metric)
        monkeypatch.setattr(
            module,
            "train_xgboost",
            SimpleNamespace(local=lambda df, horizon: {"booster": PickleBooster()}),
        )

    result = module.retrain_one(" ggal ", 2)
    assert result["promoted"] is True
    assert module._artifact_path("GGAL", 2).exists()
    assert module.retrain_one("GGAL", 2)["promoted"] is False

    monkeypatch.setattr(module, "fetch_available_tickers", lambda: ["GGAL"])
    original = module.retrain_one
    monkeypatch.setattr(
        module,
        "retrain_one",
        lambda ticker, horizon: (
            (_ for _ in ()).throw(ValueError("bad"))
            if horizon == module.MAX_HORIZON
            else {"ticker": ticker, "horizon": horizon}
        ),
    )
    batch = module.retrain_models.local()
    assert any("error" in item for item in batch)

    monkeypatch.setattr(module, "retrain_one", original)
    ready = module.prepare.local("GGAL")
    assert ready["status"] == "completed"

    with module._artifact_path("GGAL", 2).open("rb") as handle:
        artifact = pickle.load(handle)
    # Complete all horizons so the prepare fast path is covered.
    for horizon in range(module.MIN_HORIZON, module.MAX_HORIZON + 1):
        path = module._artifact_path("GGAL", horizon)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(artifact, handle)
    assert module.prepare.local(" ggal ")["status"] == "ready"
    assert "error" in module.main.local("GGAL", horizon=0)

    if module is lstm:
        monkeypatch.setattr(module, "predict_lstm_state", lambda state, window: 0.02)
    output = module.main.local(" ggal ", horizon=2)
    assert output["ticker"] == "GGAL"
    assert output["model"] in {"lstm", "xgboost"}


@pytest.mark.parametrize("module", [arima_model, garch_model, svm_model])
def test_classic_fetch_available_tickers(monkeypatch, module):
    monkeypatch.setattr(
        module.requests,
        "post",
        Mock(return_value=response({"message": {"tickers": [" ggal ", "YPFD"]}})),
    )
    assert module.fetch_available_tickers() == ["GGAL", "YPFD"]
    monkeypatch.setattr(module.requests, "post", Mock(return_value=response({})))
    with pytest.raises(ValueError):
        module.fetch_available_tickers()


def test_arima_helpers_retraining_batch_and_endpoint(monkeypatch, tmp_path):
    rows = collector_rows(100)
    monkeypatch.setattr(arima_model, "ARTIFACT_ROOT", tmp_path)
    monkeypatch.setattr(
        arima_model.requests, "post", Mock(return_value=response({"data": rows}))
    )
    assert len(arima_model.get_ticker_rows("GGAL")) == 100
    assert len(arima_model.get_ticker_data("GGAL")) == 100
    assert arima_model.rsi_last([1, 2, 3]) is None
    assert arima_model.rsi_last(list(range(20))) == 100.0
    assert arima_model.rsi_condition(None) == "indeterminado"
    assert arima_model.rsi_condition(75) == "sobrecompra"
    assert arima_model.rsi_condition(25) == "sobreventa"
    assert arima_model.rsi_condition(50) == "neutral"

    monkeypatch.setattr(
        arima_model,
        "train_model",
        SimpleNamespace(local=lambda values, order: PickleArimaFit()),
    )
    monkeypatch.setattr(arima_model.artifact_volume, "commit", Mock())
    assert arima_model.retrain_one("GGAL")["promoted"] is True
    assert arima_model.retrain_one("GGAL")["promoted"] is False
    monkeypatch.setattr(arima_model.artifact_volume, "reload", Mock())
    assert arima_model.main.local("GGAL", 3)["cant_predicciones"] == 3
    assert "error" in arima_model.main.local("GGAL", 3, media_movil=1)

    monkeypatch.setattr(arima_model, "fetch_available_tickers", lambda: ["OK", "BAD"])
    monkeypatch.setattr(
        arima_model,
        "retrain_one",
        lambda ticker: {"ticker": ticker}
        if ticker == "OK"
        else (_ for _ in ()).throw(ValueError("bad")),
    )
    assert any("error" in item for item in arima_model.retrain_models.local())


def test_garch_helpers_retraining_batch_and_endpoint(monkeypatch, tmp_path):
    rows = collector_rows(120)
    monkeypatch.setattr(garch_model, "ARTIFACT_ROOT", tmp_path)
    monkeypatch.setattr(
        garch_model.requests, "post", Mock(return_value=response({"data": rows}))
    )
    assert len(garch_model.get_ticker_returns("GGAL")) == 119

    monkeypatch.setattr(
        garch_model,
        "train_model",
        SimpleNamespace(local=lambda values: PickleGarchFit()),
    )
    monkeypatch.setattr(garch_model.artifact_volume, "commit", Mock())
    assert garch_model.retrain_one("GGAL")["promoted"] is True
    monkeypatch.setattr(garch_model.artifact_volume, "reload", Mock())
    assert "prediction" in garch_model.main.local("GGAL")

    monkeypatch.setattr(garch_model, "fetch_available_tickers", lambda: ["OK", "BAD"])
    monkeypatch.setattr(
        garch_model,
        "retrain_one",
        lambda ticker: {"ticker": ticker}
        if ticker == "OK"
        else (_ for _ in ()).throw(ValueError("bad")),
    )
    assert any("error" in item for item in garch_model.retrain_models.local())


def test_svm_helpers_retraining_batch_and_endpoint(monkeypatch, tmp_path):
    rows = collector_rows(140)
    monkeypatch.setattr(svm_model, "ARTIFACT_ROOT", tmp_path)
    monkeypatch.setattr(svm_model.requests, "post", Mock(return_value=response({"data": rows})))
    frame = svm_model.get_ticker_data_and_transform("GGAL")
    x_values, y_values = svm_model.training_arrays(frame)
    assert len(x_values) == len(y_values) == 139

    monkeypatch.setattr(
        svm_model,
        "train_classifier",
        SimpleNamespace(local=lambda x, y: PickleClassifier()),
    )
    monkeypatch.setattr(svm_model.artifact_volume, "commit", Mock())
    assert svm_model.retrain_one("GGAL")["promoted"] is True
    monkeypatch.setattr(svm_model.artifact_volume, "reload", Mock())
    assert svm_model.main.local("GGAL")["prediction"] == "Buy"

    monkeypatch.setattr(svm_model, "fetch_available_tickers", lambda: ["OK", "BAD"])
    monkeypatch.setattr(
        svm_model,
        "retrain_one",
        lambda ticker: {"ticker": ticker}
        if ticker == "OK"
        else (_ for _ in ()).throw(ValueError("bad")),
    )
    assert any("error" in item for item in svm_model.retrain_models.local())


def test_example_model_trains_once():
    example_model.REG = None
    model = example_model.train_model.local(
        np.array([[1.0, 2.0], [2.0, 3.0]]), np.array([3.0, 5.0])
    )
    assert example_model.train_model.local(np.ones((2, 2)), np.ones(2)) is None
    assert model.predict(np.array([[3.0, 4.0]])).shape == (1,)
