"""Smoke tests de los endpoints productivos.

Llaman a la funcion real del endpoint (`main.local(...)`, que corre el
mismo codigo que Modal ejecuta en produccion, sin pasar por la red de
Modal) mockeando solo `artifact_volume.reload`, que en local tira error
porque no hay volumen montado. No hay artefacto entrenado en el entorno
de CI, asi que el resultado esperado es el path de "todavia no hay un
modelo entrenado" -- lo que importa aca es que el endpoint importe sin
romperse y responda un dict en vez de tirar una excepcion.
"""

from unittest.mock import patch

import arima_model
import garch_model
import lstm_trend_model
import svm_model
import transformer_trend_model
import xgboost_trend_model


def _assert_responds_gracefully(result: dict) -> None:
    assert isinstance(result, dict)
    assert "error" in result
    assert isinstance(result["error"], str) and result["error"]


def test_lstm_endpoint_responds():
    with patch.object(lstm_trend_model.artifact_volume, "reload", return_value=None):
        result = lstm_trend_model.main.local(ticker="GGAL")
    _assert_responds_gracefully(result)


def test_xgboost_endpoint_responds():
    with patch.object(xgboost_trend_model.artifact_volume, "reload", return_value=None):
        result = xgboost_trend_model.main.local(ticker="GGAL")
    _assert_responds_gracefully(result)


def test_svm_endpoint_responds():
    with patch.object(svm_model.artifact_volume, "reload", return_value=None):
        result = svm_model.main.local(ticker="GGAL")
    _assert_responds_gracefully(result)


def test_arima_endpoint_responds():
    with patch.object(arima_model.artifact_volume, "reload", return_value=None):
        result = arima_model.main.local(ticker="GGAL", predictions=5)
    _assert_responds_gracefully(result)


def test_garch_endpoint_responds():
    with patch.object(garch_model.artifact_volume, "reload", return_value=None):
        result = garch_model.main.local(ticker="GGAL")
    _assert_responds_gracefully(result)


def test_transformer_endpoint_responds():
    with patch.object(transformer_trend_model.artifact_volume, "reload", return_value=None):
        result = transformer_trend_model.main.local(ticker="GGAL")
    _assert_responds_gracefully(result)
