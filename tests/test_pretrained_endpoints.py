import ast
from pathlib import Path


ROOT = Path(__file__).parents[1] / "modelos"


def _function_calls(path: Path, function_name: str) -> set[str]:
    tree = ast.parse(path.read_text())
    function = next(
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name
    )
    calls = set()
    for node in ast.walk(function):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                calls.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                calls.add(node.func.attr)
    return calls


def test_lstm_endpoint_never_trains():
    calls = _function_calls(ROOT / "lstm_trend_model.py", "main")
    assert "train_lstm" not in calls
    assert "retrain_one" not in calls


def test_xgboost_endpoint_never_trains():
    calls = _function_calls(ROOT / "xgboost_trend_model.py", "main")
    assert "train_xgboost" not in calls
    assert "retrain_one" not in calls


def test_classic_model_endpoints_never_train():
    for name in ("svm_model.py", "arima_model.py", "garch_model.py"):
        calls = _function_calls(ROOT / name, "main")
        assert "train_model" not in calls
        assert "train_classifier" not in calls
        assert "retrain_one" not in calls


def test_both_models_have_a_scheduled_retraining_function():
    schedules = {
        "lstm_trend_model.py": 'schedule=modal.Cron("0 20 * * 1-5"',
        "xgboost_trend_model.py": 'schedule=modal.Cron("0 21 * * 1-5"',
    }
    for name, schedule in schedules.items():
        source = (ROOT / name).read_text()
        assert schedule in source
        assert "def retrain_models()" in source
        assert "fetch_available_tickers()" in source
        assert "TRAINING_TICKERS" not in source


def test_classic_models_train_all_tickers_on_a_schedule():
    schedules = {
        "svm_model.py": 'schedule=modal.Cron("30 21 * * 1-5"',
        "arima_model.py": 'schedule=modal.Cron("0 22 * * 1-5"',
        "garch_model.py": 'schedule=modal.Cron("0 23 * * 1-5"',
    }
    for name, schedule in schedules.items():
        source = (ROOT / name).read_text()
        assert schedule in source
        assert "for ticker in fetch_available_tickers():" in source
        assert "def retrain_models()" in source
