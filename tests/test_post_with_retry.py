import importlib

import pytest
import requests

MODULES = [
    "arima_model",
    "garch_ann_model",
    "garch_model",
    "lstm_trend_model",
    "svm_model",
    "transformer_trend_model",
    "xgboost_trend_model",
]


def _load(name):
    return importlib.import_module(name)


class _Resp:
    def __init__(self, status):
        self.status_code = status


@pytest.mark.parametrize("name", MODULES)
def test_retries_after_cold_start_timeout(name, monkeypatch):
    module = _load(name)
    monkeypatch.setattr(module.time, "sleep", lambda _s: None)
    calls = []

    def fake_post(url, timeout):
        calls.append(timeout)
        if len(calls) < 3:
            raise requests.Timeout("cold start")
        return _Resp(200)

    monkeypatch.setattr(module.requests, "post", fake_post)
    assert module._post_with_retry("http://x").status_code == 200
    assert len(calls) == 3 and calls[0] >= 60


def test_retries_5xx_and_gives_up(monkeypatch):
    module = _load("garch_model")
    monkeypatch.setattr(module.time, "sleep", lambda _s: None)
    monkeypatch.setattr(module.requests, "post", lambda url, timeout: _Resp(503))
    with pytest.raises(requests.HTTPError):
        module._post_with_retry("http://x", attempts=2)


def test_returns_4xx_without_retry(monkeypatch):
    module = _load("garch_model")
    calls = []
    monkeypatch.setattr(
        module.requests, "post", lambda url, timeout: calls.append(1) or _Resp(404)
    )
    assert module._post_with_retry("http://x").status_code == 404
    assert len(calls) == 1
