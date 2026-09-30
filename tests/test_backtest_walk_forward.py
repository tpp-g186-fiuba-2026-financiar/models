"""Tests del backtest walk-forward compartido por ARIMA, XGBoost, LSTM y Transformer.

Sin red ni Modal. Lo que se verifica es lo que hace que el backtest sirva para
comparar modelos: que nunca entrene con datos que despues evalua, que cada
caso sea independiente y que la confianza dependa de la cantidad de casos.
"""

import numpy as np
import pandas as pd
import pytest

import arima_model
import lstm_trend_model as lstm
import transformer_trend_model as transformer
import xgboost_trend_model as xgb

PLAN_MODULES = [arima_model, xgb, lstm, transformer]


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


@pytest.mark.parametrize("module", PLAN_MODULES)
def test_plan_uses_independent_cases_inside_each_fold(module):
    plan = module.backtest_plan(420, 5, 65)
    assert len(plan) == module.BACKTEST_FOLDS == 3
    for split, origins in plan:
        # casos separados exactamente `horizon` ruedas: sin solapamiento
        assert {b - a for a, b in zip(origins, origins[1:])} == {5}
        assert origins[0] == split - 1
        assert origins[-1] < split - 1 + module.BACKTEST_FOLD_DAYS


@pytest.mark.parametrize("module", PLAN_MODULES)
def test_plan_folds_are_consecutive_and_end_where_the_target_exists(module):
    n, horizon = 420, 5
    plan = module.backtest_plan(n, horizon, 65)
    splits = [split for split, _ in plan]
    assert np.diff(splits).tolist() == [module.BACKTEST_FOLD_DAYS] * 2
    last_origin = plan[-1][1][-1]
    assert last_origin + 1 + horizon <= n - 1  # el cierre real a `horizon` ruedas existe


@pytest.mark.parametrize("module", PLAN_MODULES)
def test_plan_uses_fewer_folds_with_short_history_and_fails_without_history(module):
    assert len(module.backtest_plan(160, 5, 65)) < 3
    with pytest.raises(ValueError):
        module.backtest_plan(80, 5, 65)


def test_wilson_interval_narrows_with_more_cases():
    small = xgb.wilson_interval(6, 10)
    large = xgb.wilson_interval(600, 1000)
    assert (small[1] - small[0]) > 5 * (large[1] - large[0])
    assert xgb.wilson_interval(0, 0) == (0.0, 1.0)


def test_summary_distinguishes_direction_from_real_signal():
    # Predice siempre +0.2% (por debajo de la banda neutral de 1%) y sube siempre:
    # acierta la direccion el 100% pero nunca emite una senal alza/baja.
    metrics = xgb.summarize_backtest([0.002] * 20, [0.03] * 20, [], folds=1, mae=0.0)
    assert metrics["directional_accuracy"] == 1.0
    assert metrics["neutral_rate"] == 1.0
    assert metrics["signal_hit_rate"] is None
    assert metrics["accuracy_low"] < 1.0  # 20 casos no alcanzan para certeza total


def _spy_on_training(monkeypatch, module, trainer_name, kind):
    """Reemplaza el entrenador por un espia que anota cuantas filas recibe."""
    seen = []
    real = getattr(module, trainer_name).local

    class Spy:
        @staticmethod
        def local(data, *args):
            seen.append(len(data))
            return real(data, *args)

    monkeypatch.setattr(module, trainer_name, Spy())
    return seen


def test_arima_backtest_never_trains_on_evaluated_data(monkeypatch):
    values = _history(300)["close"].tolist()
    seen = _spy_on_training(monkeypatch, arima_model, "train_model", "arima")
    metrics = arima_model.backtest_arima(values)
    plan = arima_model.backtest_plan(len(values), 5, arima_model.MEDIA_MOVIL + 30)
    assert seen == [split for split, _ in plan]
    assert metrics["folds"] == 3
    assert metrics["observations"] == sum(len(o) for _, o in plan)


def test_arima_backtest_on_pure_noise_is_not_inflated():
    # Regresion del error viejo: una sola prediccion a 30 dias contada dia por dia
    # daba ~90% de "acierto" en una racha aunque el precio predicho estuviera lejos
    # del real. Sobre un paseo aleatorio no hay nada que predecir: el promedio de
    # varias series tiene que quedar cerca del azar, no cerca del 90%.
    accuracies = []
    for seed in range(6):
        rng = np.random.default_rng(seed)
        values = (100 * np.exp(np.cumsum(rng.normal(0, 0.01, 300)))).tolist()
        accuracies.append(arima_model.backtest_arima(values)["directional_accuracy"])
    assert 0.35 < float(np.mean(accuracies)) < 0.65


def test_xgboost_backtest_never_trains_on_evaluated_data(monkeypatch):
    # El booster real necesita libomp (no siempre instalado localmente); el
    # walk-forward no depende de el, asi que se entrena un doble.
    class Booster:
        @staticmethod
        def predict(window):
            return [0.02]

    seen = []
    monkeypatch.setattr(
        xgb,
        "train_xgboost",
        type("S", (), {"local": staticmethod(lambda d, h: seen.append(len(d)) or {"booster": Booster})}),
    )
    df = _history(420)
    metrics = xgb.backtest_xgboost(df, 5)
    plan = xgb.backtest_plan(len(df), 5, xgb.WINDOW + 5 + 30)
    assert seen == [split for split, _ in plan]
    assert metrics["folds"] == 3
    assert metrics["observations"] == 24
    assert {"accuracy_low", "accuracy_high", "signal_hit_rate", "neutral_rate"} <= set(metrics)
    assert {"date", "predicted", "actual"} <= set(metrics["series"][0])


@pytest.mark.parametrize(
    "module, name, trainer, predictor",
    [
        (lstm, "backtest_lstm", "train_lstm", "predict_lstm_state"),
        (transformer, "backtest_transformer", "train_transformer", "predict_transformer_state"),
    ],
)
def test_neural_backtests_wire_the_same_walk_forward(monkeypatch, module, name, trainer, predictor):
    df = _history(420)
    seen = _spy_on_training(monkeypatch, module, trainer, "nn")
    monkeypatch.setattr(module, trainer, type("S", (), {"local": staticmethod(lambda d, h: seen.append(len(d)) or {})}))
    monkeypatch.setattr(module, predictor, lambda state, window: 0.02)
    metrics = getattr(module, name)(df, 5)
    plan = module.backtest_plan(len(df), 5, module.WINDOW + 5 + 30)
    assert seen[-len(plan):] == [split for split, _ in plan]
    assert metrics["folds"] == 3
    assert metrics["observations"] == 24
    assert metrics["neutral_rate"] == 0.0  # +2% supera la banda neutral
