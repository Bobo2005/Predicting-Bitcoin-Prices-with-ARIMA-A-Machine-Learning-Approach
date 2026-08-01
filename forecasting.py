from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

try:
    import yfinance as yf
except ImportError:  # pragma: no cover - exercised when dependency is absent
    yf = None

from utils import retry

DEFAULT_TICKER = "BTC-USD"
DEFAULT_START_DATE = "2017-07-31"
DEFAULT_END_DATE = "2026-07-31"


def build_sample_series() -> pd.Series:
    dates = pd.date_range(start=DEFAULT_START_DATE, periods=400, freq="D")
    trend = np.linspace(10000, 18000, len(dates))
    seasonality = 250 * np.sin(np.linspace(0, 3 * np.pi, len(dates)))
    noise = np.random.RandomState(42).normal(0, 120, size=len(dates))
    values = trend + seasonality + noise
    return pd.Series(values, index=dates, name="Close")


def build_sample_df() -> pd.DataFrame:
    # Create a simple synthetic OHLCV-like DataFrame for offline testing
    close = build_sample_series()
    open_ = close.shift(1).fillna(close.iloc[0])
    high = pd.concat([open_, close], axis=1).max(axis=1) + 50
    low = pd.concat([open_, close], axis=1).min(axis=1) - 50
    volume = (np.random.RandomState(42).randint(1000, 10000, size=len(close))).astype(
        float
    )
    df = pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume}
    )
    return df


def load_price_series(
    ticker: str = DEFAULT_TICKER,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    use_sample: bool = False,
    return_df: bool = False,
) -> pd.Series | pd.DataFrame:
    """Load price series or full OHLCV DataFrame.

    If return_df is False (default), returns a pandas Series containing closing prices.
    If return_df is True, returns a DataFrame with OHLCV columns when available.
    """
    if use_sample:
        return build_sample_df() if return_df else build_sample_series()

    if yf is None:
        warnings.warn(
            "yfinance is not installed; using a built-in sample series instead."
        )
        return build_sample_df() if return_df else build_sample_series()

    try:
        # Retry the yfinance download in case of transient network issues
        @retry(times=3, delay=1.0, backoff=2.0)
        def _download():
            return yf.download(
                ticker, start=start_date, end=end_date, interval="1d", progress=False
            )

        data = _download()
        if data.empty or "Close" not in data.columns:
            raise ValueError(f"No closing-price data returned for {ticker}")
        data = data.sort_index()
        if return_df:
            return data[
                [
                    c
                    for c in ["Open", "High", "Low", "Close", "Volume"]
                    if c in data.columns
                ]
            ]
        prices = data["Close"].astype(float).dropna().sort_index()
        if prices.empty:
            raise ValueError(f"No valid price values available for {ticker}")
        return prices
    except Exception as exc:  # pragma: no cover - depends on network availability
        warnings.warn(
            (
                f"Unable to download {ticker}: {exc}. "
                "Using a built-in sample series instead."
            )
        )
        return build_sample_df() if return_df else build_sample_series()


def split_train_test(
    series: pd.Series, train_split: float = 0.8
) -> tuple[pd.Series, pd.Series]:
    if not 0 < train_split < 1:
        raise ValueError("train_split must be between 0 and 1")
    train_size = int(len(series) * train_split)
    return series.iloc[:train_size], series.iloc[train_size:]


def compute_adf(series: pd.Series) -> tuple[float, float]:
    result = adfuller(series.dropna())
    return float(result[0]), float(result[1])


def evaluate_forecast(actual: pd.Series, forecast: pd.Series) -> dict[str, float]:
    common_index = actual.index.intersection(forecast.index)
    actual_aligned = actual.loc[common_index].astype(float)
    forecast_aligned = forecast.loc[common_index].astype(float)
    if actual_aligned.empty:
        raise ValueError("No overlapping values for evaluation")

    mae = mean_absolute_error(actual_aligned, forecast_aligned)
    rmse = np.sqrt(mean_squared_error(actual_aligned, forecast_aligned))
    denominator = actual_aligned.where(actual_aligned != 0, np.finfo(float).eps)
    mape = np.mean(np.abs((actual_aligned - forecast_aligned) / denominator)) * 100
    return {"mae": float(mae), "rmse": float(rmse), "mape": float(mape)}


def select_arima_order(
    train_series: pd.Series,
    candidate_orders: Sequence[tuple[int, int, int]] | None = None,
) -> tuple[tuple[int, int, int], object]:
    candidate_orders = list(
        candidate_orders or [(1, 1, 1), (2, 1, 2), (1, 1, 2), (2, 1, 1), (0, 1, 1)]
    )
    best_order: tuple[int, int, int] | None = None
    best_fit = None
    best_aic = None

    for order in candidate_orders:
        try:
            model = ARIMA(
                train_series,
                order=order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            fit = model.fit()
        except Exception:
            continue
        if best_aic is None or fit.aic < best_aic:
            best_aic = fit.aic
            best_order = order
            best_fit = fit

    if best_order is None or best_fit is None:
        raise RuntimeError(
            "Unable to fit any ARIMA model with the supplied candidate orders"
        )

    return best_order, best_fit


def build_forecast_series(
    train_series: pd.Series,
    test_series: pd.Series,
    full_series: pd.Series,
    forecast_steps: int,
    candidate_orders: Sequence[tuple[int, int, int]] | None = None,
) -> dict[str, object]:
    order, fit = select_arima_order(train_series, candidate_orders)
    forecast_values = fit.forecast(steps=len(test_series))
    forecast = pd.Series(
        forecast_values, index=test_series.index, name="ARIMA_Forecast"
    )

    future_dates = pd.date_range(
        start=full_series.index[-1] + pd.Timedelta(days=1),
        periods=forecast_steps,
        freq="D",
    )
    future_values = fit.forecast(steps=forecast_steps)
    future_forecast = pd.Series(
        future_values, index=future_dates, name="ARIMA_Future_Forecast"
    )

    return {
        "order": order,
        "model_fit": fit,
        "forecast": forecast,
        "future_forecast": future_forecast,
    }


def build_naive_forecast(test_series: pd.Series) -> tuple[pd.Series, pd.Series]:
    naive_forecast = test_series.shift(1).dropna()
    actual_aligned = test_series.iloc[1:]
    return naive_forecast, actual_aligned
