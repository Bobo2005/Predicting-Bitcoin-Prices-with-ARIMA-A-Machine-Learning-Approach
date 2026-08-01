import sys

from main import parse_args, parse_candidate_orders


def test_parse_candidate_orders_from_list():
    orders = parse_candidate_orders([[1, 1, 1], [2, 1, 2]])
    assert orders == [(1, 1, 1), (2, 1, 2)]


def test_parse_args_loads_config_file(tmp_path, monkeypatch):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
seed: 42
forecast_steps: 10
preprocess_lags: 5
outlier_multiplier: 2.0
seasonal_period: 14
rf_lags: 3
ma_window: 5
"""
    )
    monkeypatch.setattr(sys, "argv", ["main.py", "--config", str(config_file)])
    args = parse_args()
    assert args.forecast_steps == 10
    assert args.preprocess_lags == 5
    assert args.outlier_multiplier == 2.0
    assert args.seasonal_period == 14
    assert args.rf_lags == 3
    assert args.ma_window == 5


def test_parse_args_overrides_config_with_cli(tmp_path, monkeypatch):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
seed: 42
forecast_steps: 10
output_dir: outputs
"""
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--config",
            str(config_file),
            "--forecast-steps",
            "20",
            "--output-dir",
            "test_outputs",
        ],
    )
    args = parse_args()
    assert args.forecast_steps == 20
    assert args.output_dir == "test_outputs"
