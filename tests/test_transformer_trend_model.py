"""Tests funcionales de `transformer_trend_model.py` con datos sinteticos.

No tocan la red ni Modal: `fetch_*` se mockea y el Volume se reemplaza por
un directorio temporal. El entrenamiento es real (red chica, pocas epocas
gracias al early stopping), asi se verifica de punta a punta que el modelo
entrena con todas las acciones, pasa el gate, persiste y predice.
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import transformer_trend_model as tm


def _history(n: int = 420, seed: int = 0, hora: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, n)))
    index = pd.date_range("2024-01-01", periods=n, freq="B") + pd.Timedelta(hours=hora)
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


def _ccl(n: int = 600) -> pd.Series:
    """CCL que sube ~0,3% por rueda (devaluacion)."""
    index = pd.date_range("2023-12-01", periods=n, freq="B")
    return pd.Series(1000 * np.exp(0.003 * np.arange(n)), index=index)


def _riesgo_pais() -> pd.Series:
    dias = pd.date_range("2023-12-01", "2026-12-31", freq="D")
    return pd.Series(np.linspace(2000, 800, len(dias)), index=dias)


HISTORIAS = {t: _history(seed=i, hora=14) for i, t in enumerate(["GGAL", "BMA", "YPFD", "PAMP"])}


@pytest.fixture
def artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(tm, "ARTIFACT_ROOT", tmp_path)
    monkeypatch.setattr(tm.artifact_volume, "commit", lambda: None)
    monkeypatch.setattr(tm.artifact_volume, "reload", lambda: None)
    monkeypatch.setattr(tm, "fetch_available_tickers", lambda: [*HISTORIAS, "OIL", "ALUAD"])
    monkeypatch.setattr(tm, "fetch_ticker_history", lambda ticker: HISTORIAS[ticker])
    monkeypatch.setattr(tm, "fetch_macro_series", lambda *a, **k: None)
    monkeypatch.setattr(tm, "fetch_ccl_y_riesgo_pais", lambda: (_ccl(), _riesgo_pais()))
    # El gate pide senal fuera de muestra; con ruido sintetico no la hay.
    monkeypatch.setattr(tm, "pasa_el_gate", lambda metrics: True)
    return tmp_path


def _preparado(df=None):
    df = HISTORIAS["GGAL"] if df is None else df
    return tm.preparar(df, pd.Series(dtype=float), _ccl(), _riesgo_pais())


def test_build_features_has_the_shared_feature_set_plus_riesgo_pais():
    feats, target, escala = tm.build_features(_preparado(), horizon=5)
    assert feats.shape[1] == len(tm.FEATURE_NAMES) == 9
    assert len(feats) == len(target) == len(escala)
    assert np.isfinite(feats).all()
    # los ultimos `horizon` targets no existen todavia
    assert np.isnan(target[-5:]).all()
    # riesgo pais en baja: su cambio a 5 ruedas es negativo
    assert (feats[30:, -1] < 0).all()


def test_preparar_pasa_a_dolares_y_ventanas_por_dia():
    preparado = _preparado()
    esperado = HISTORIAS["GGAL"]["close"].to_numpy() / _ccl().reindex(
        HISTORIAS["GGAL"].index.normalize()
    ).to_numpy()
    assert preparado["close"].to_numpy() == pytest.approx(esperado)
    _, _, fechas = tm.ventanas_con_fecha(preparado)
    assert (pd.DatetimeIndex(fechas) == pd.DatetimeIndex(fechas).normalize()).all()


def test_limpieza_ajusta_un_split_que_el_mercado_no_acompano():
    df = HISTORIAS["GGAL"].copy()
    fecha = df.index[200]
    df.loc[df.index >= fecha, ["open", "high", "low", "close"]] /= 10
    df.iloc[10, df.columns.get_loc("close")] = 0.0
    limpio = tm.limpiar(df, pd.Series(dtype=float))
    assert len(limpio) == len(df) - 1
    assert abs(np.log(limpio["close"]).diff().loc[fecha]) < 0.1
    # Si el mercado entero cayo igual, no es un split.
    mercado = pd.Series({fecha.normalize(): np.log(0.1)})
    assert len(tm.detectar_saltos(tm.sin_precios_no_positivos(df), mercado)) == 0


def test_cotiza_en_dolares():
    assert tm.cotiza_en_dolares("GOLD") and tm.cotiza_en_dolares("aluad")
    assert not tm.cotiza_en_dolares("YPFD") and not tm.cotiza_en_dolares("GGAL")


def test_network_forward_shape():
    import torch

    net = tm.build_transformer_network()
    out = net(torch.zeros(3, tm.WINDOW, len(tm.FEATURE_NAMES)))
    assert tuple(out.shape) == (3,)


def test_train_transformer_requires_enough_windows():
    with pytest.raises(ValueError):
        tm.train_transformer(np.zeros((10, tm.WINDOW, 9)), np.ones(10))


def test_prediccion_acotada_por_el_movimiento_tipico():
    X, y, _ = tm.ventanas_con_fecha(_preparado())
    state, mejor = tm.train_transformer(X[:-60], y[:-60], X[-60:], y[-60:])
    assert 1 <= mejor <= tm.EPOCHS
    pred = tm.predict_transformer_state(state, X[-10:])
    assert np.all(np.abs(pred) <= state["movimiento"])


def test_bloques_walk_forward_comunes_y_con_gap():
    fechas = pd.date_range("2024-01-01", periods=100, freq="B").to_numpy()
    with patch.object(tm, "VAL_FRACTION", 0.3):
        bloques = tm.bloques_walk_forward(np.concatenate([fechas, fechas[60:]]), gap=5)
    assert [(b[0], b[1]) for b in bloques] == [
        (fechas[65], fechas[70]),
        (fechas[75], fechas[80]),
        (fechas[85], fechas[90]),
    ]
    assert bloques[-1][2] > fechas[-1]
    with patch.object(tm, "VAL_FRACTION", 0.0):
        assert tm.bloques_walk_forward(fechas) == []
    with patch.object(tm, "VAL_FRACTION", 0.5), pytest.raises(ValueError):
        tm.bloques_walk_forward(fechas[:20], gap=10)


def test_metricas_y_gate():
    real = np.array([1.0, 1.0, 1.0, -1.0])
    assert tm.acierto_balanceado(np.ones(4), real) == 0.5  # siempre "sube"
    assert tm.acierto_balanceado(real, real) == 1.0
    fechas = np.repeat(np.array(["2024-01-01", "2024-01-02"], dtype="datetime64[D]"), 3)
    assert tm.ic_por_fecha(np.arange(6.0), np.arange(6.0), fechas) == pytest.approx(1.0)
    assert tm.pasa_el_gate({"acierto_balanceado": 0.55, "ic": 0.02})
    assert not tm.pasa_el_gate({"acierto_balanceado": 0.5, "ic": 0.02})
    assert not tm.pasa_el_gate({"acierto_balanceado": 0.6, "ic": -0.01})
    assert not tm.pasa_el_gate({"folds": 0})


def test_retrain_entrena_un_solo_modelo_sin_los_dolarizados(artifacts):
    result = tm.retrain_models.local()
    assert result["promoted"] is True
    assert result["tickers"] == 4
    assert result["metrics"]["folds"] == 3
    assert 0.0 <= result["metrics"]["acierto_balanceado"] <= 1.0
    assert (artifacts / "h5" / "production.pkl").exists()


def test_retrain_no_guarda_si_no_pasa_el_gate(artifacts):
    with patch.object(tm, "pasa_el_gate", lambda metrics: False):
        result = tm.retrain_models.local()
    assert result["promoted"] is False
    assert not (artifacts / "h5" / "production.pkl").exists()


def test_retrain_sigue_si_un_ticker_falla_y_reporta_errores(artifacts):
    def flaky(ticker):
        if ticker == "BMA":
            raise ValueError("sin datos")
        return HISTORIAS[ticker]

    with patch.object(tm, "fetch_ticker_history", flaky):
        assert tm.retrain_models.local()["tickers"] == 3
    with patch.object(tm, "fetch_available_tickers", lambda: []):
        assert "error" in tm.retrain_models.local()


def test_endpoint_serves_a_prediction_in_pesos(artifacts):
    tm.retrain_models.local()

    result = tm.main.local(ticker="ggal", horizon=5)

    assert "error" not in result
    assert result["ticker"] == "GGAL"
    assert result["model"] == "transformer"
    assert result["signal"] in {"alza", "baja", "neutral"}
    assert result["horizon_days"] == 5
    last = float(HISTORIAS["GGAL"]["close"].iloc[-1])
    assert result["last_close"] == pytest.approx(last, rel=1e-4)
    assert 0.5 * last < result["predicted_close"] < 1.5 * last
    assert result["backtest"]["acierto_balanceado"] is not None
    assert result["source"].startswith("modal")


def test_endpoint_rechaza_horizonte_dolarizados_y_faltantes(artifacts):
    assert "horizon" in tm.main.local(ticker="GGAL", horizon=3)["error"]
    assert "dolares" in tm.main.local(ticker="OIL")["error"]
    assert "entrenado" in tm.main.local(ticker="GGAL")["error"]
    tm.retrain_models.local()
    with patch.object(tm, "fetch_ticker_history", lambda ticker: HISTORIAS["GGAL"].iloc[:40]):
        assert "mas ruedas" in tm.main.local(ticker="GGAL")["error"]

    def sin_ccl():
        raise ValueError("collector caido")

    with patch.object(tm, "fetch_ccl_y_riesgo_pais", sin_ccl):
        assert "CCL" in tm.main.local(ticker="GGAL")["error"]


def test_fetch_ccl_y_riesgo_pais_aplica_el_retraso(monkeypatch):
    index = pd.date_range("2024-01-01", periods=10, freq="D")

    def fetch(route, series):
        return pd.Series(np.arange(1.0, 11.0), index=index)

    monkeypatch.setattr(tm, "fetch_collector_series", fetch)
    ccl, riesgo = tm.fetch_ccl_y_riesgo_pais()
    assert ccl.iloc[-1] == 10.0
    assert riesgo.loc[index[9]] == 8.0  # dos dias de retraso de publicacion


def test_fetch_collector_series_valida_el_status_del_body(monkeypatch):
    class _Resp:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    ok = {"status": 200, "data": [{"value": "1.5", "ts": 1704067200000}]}
    monkeypatch.setattr(tm, "_post_with_retry", lambda url: _Resp(ok))
    assert tm.fetch_collector_series("macro/argdatos", "CCL").iloc[0] == 1.5
    for payload in ({"status": 500, "message": {"error": "x"}}, {"status": 200, "data": []}):
        monkeypatch.setattr(tm, "_post_with_retry", lambda url, p=payload: _Resp(p))
        with pytest.raises(ValueError):
            tm.fetch_collector_series("macro/argdatos", "CCL")


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
