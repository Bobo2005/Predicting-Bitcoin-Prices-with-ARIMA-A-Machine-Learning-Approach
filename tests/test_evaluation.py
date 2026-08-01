from forecasting import build_sample_series, evaluate_forecast


def test_evaluate_forecast_basic():
    s = build_sample_series()
    # create a trivial forecast equal to actual shifted by 0
    forecast = s.copy()
    metrics = evaluate_forecast(s, forecast)
    assert metrics["mae"] == 0
    assert metrics["rmse"] == 0
    assert metrics["mape"] == 0


def test_evaluate_forecast_alignment():
    s = build_sample_series()
    # forecast with different index (subset)
    forecast = s.iloc[10:20]
    metrics = evaluate_forecast(s, forecast)
    assert "mae" in metrics
    assert metrics["mae"] >= 0
