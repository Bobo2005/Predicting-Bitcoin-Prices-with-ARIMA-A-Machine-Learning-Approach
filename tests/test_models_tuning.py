from forecasting import build_sample_series
from models_tuning import grid_search_rf, rolling_cv_scores_arima


def test_grid_search_rf_basic():
    s = build_sample_series()
    grid = {"n_estimators": [10, 20], "max_depth": [3, None]}
    res = grid_search_rf(s, grid, lags=3)
    assert "best_params" in res
    assert "best_score" in res


def test_rolling_cv_arima():
    s = build_sample_series()
    score = rolling_cv_scores_arima(
        s, order=(1, 1, 1), train_window=100, horizon=5, max_splits=2
    )
    assert isinstance(score, float)
