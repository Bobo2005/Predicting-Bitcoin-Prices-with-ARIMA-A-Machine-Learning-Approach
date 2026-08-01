from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

try:
    from prophet import Prophet
except Exception:  # Prophet may not be installed in all environments
    Prophet = None


def _safe_mape(actual: pd.Series, pred: pd.Series) -> float:
    denom = actual.where(actual != 0, np.finfo(float).eps)
    return float(np.mean(np.abs((actual - pred) / denom)) * 100)


def sarimax_forecast(
    train: pd.Series,
    test_index: pd.DatetimeIndex,
    order=(1, 1, 1),
    seasonal_order=(0, 0, 0, 0),
) -> pd.Series:
    model = SARIMAX(
        train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fit = model.fit(disp=False)
    preds = fit.get_forecast(steps=len(test_index)).predicted_mean
    preds.index = test_index
    return pd.Series(preds, index=test_index, name="SARIMAX_Forecast")


def prophet_forecast(train: pd.Series, periods: int, freq: str = "D") -> pd.Series:
    if Prophet is None:
        raise RuntimeError("prophet package is not available")
    df = train.reset_index()
    df.columns = ["ds", "y"]
    m = Prophet()
    m.fit(df)
    future = m.make_future_dataframe(periods=periods, freq=freq)
    forecast = m.predict(future)
    # Return only the forecasted values beyond the training end
    fc = forecast.set_index("ds")["yhat"].loc[~forecast["ds"].isin(df["ds"])]
    fc.index = pd.DatetimeIndex(fc.index)
    return pd.Series(fc.values, index=fc.index, name="Prophet_Forecast")


def naive_forecast(train: pd.Series, test_index: pd.DatetimeIndex) -> pd.Series:
    last_value = float(train.iloc[-1])
    return pd.Series(
        [last_value] * len(test_index), index=test_index, name="Naive_Forecast"
    )


def moving_average_forecast(
    train: pd.Series, test_index: pd.DatetimeIndex, window: int = 7
) -> pd.Series:
    window = min(window, len(train))
    avg_value = float(train.iloc[-window:].mean())
    return pd.Series(
        [avg_value] * len(test_index), index=test_index, name="MovingAverage_Forecast"
    )


def exp_smoothing_forecast(
    train: pd.Series,
    test_index: pd.DatetimeIndex,
) -> pd.Series:
    model = SimpleExpSmoothing(train)
    fit = model.fit(optimized=True)
    preds = fit.forecast(len(test_index))
    preds.index = test_index
    return pd.Series(preds.values, index=test_index, name="ExpSmoothing_Forecast")


def rf_baseline_forecast(
    train: pd.Series, test_index: pd.DatetimeIndex, lags: int = 7
) -> pd.Series:
    # Build lag features on train
    df = pd.DataFrame({"y": train})
    for lag in range(1, lags + 1):
        df[f"lag_{lag}"] = df["y"].shift(lag)
    df = df.dropna()
    if df.empty:
        raise ValueError("Not enough data to build lag features for RF baseline")

    X = df.drop(columns=["y"]).values
    y = df["y"].values
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Iteratively predict for the test_index horizon
    history = train.copy()
    preds = []
    for _ in range(len(test_index)):
        last_vals = history.iloc[-lags:].values
        if len(last_vals) < lags:
            # pad with zeros if insufficient history (shouldn't normally happen)
            padded = np.concatenate([np.zeros(lags - len(last_vals)), last_vals])
            last_vals = padded
        x = last_vals[::-1]  # make same order as lag_1, lag_2, ...
        x = x.reshape(1, -1)
        yhat = model.predict(x)[0]
        preds.append(yhat)
        history = pd.concat(
            [
                history,
                pd.Series([yhat], index=[history.index[-1] + pd.Timedelta(days=1)]),
            ]
        )

    preds_series = pd.Series(preds, index=test_index, name="RF_Forecast")
    return preds_series


def compare_models(
    full_series: pd.Series,
    train_split: float = 0.8,
    forecast_steps: int = 30,
    output_dir: str | None = None,
    seasonal_period: int = 7,
    rf_lags: int = 7,
    ma_window: int = 7,
    persist_models: bool = True,
) -> pd.DataFrame:
    """
    Compare SARIMAX, Prophet (if available), and RandomForest baseline on a train/test split.
    Returns a DataFrame with metrics for each model and also writes a metrics CSV if output_dir is provided.
    """
    n = len(full_series)
    train_size = int(n * train_split)
    train = full_series.iloc[:train_size]
    test = full_series.iloc[train_size : train_size + forecast_steps]
    if test.empty:
        raise ValueError("Test set is empty for the requested forecast_steps")

    results = []
    # ARIMA via SARIMAX (non-seasonal)
    try:
        fit_obj = None
        sarima_pred = sarimax_forecast(train, test.index, order=(1, 1, 1))
        mae = mean_absolute_error(test, sarima_pred)
        rmse = np.sqrt(mean_squared_error(test, sarima_pred))
        mape = _safe_mape(test, sarima_pred)
        results.append({"model": "SARIMAX", "mae": mae, "rmse": rmse, "mape": mape})
        if persist_models and output_dir:
            # attempt to save the fitted SARIMAX model using joblib via utils
            try:
                from utils import save_model

                # fit to get the object
                model_obj = SARIMAX(
                    train,
                    order=(1, 1, 1),
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit(disp=False)
                save_model(model_obj, f"{output_dir}/sarimax_model.joblib")
            except Exception:
                pass
    except Exception as exc:
        warnings.warn(f"SARIMAX failed: {exc}")

    # Seasonal SARIMAX with weekly seasonality
    try:
        sarimax_pred = sarimax_forecast(
            train,
            test.index,
            order=(1, 1, 1),
            seasonal_order=(1, 0, 1, seasonal_period),
        )
        mae = mean_absolute_error(test, sarimax_pred)
        rmse = np.sqrt(mean_squared_error(test, sarimax_pred))
        mape = _safe_mape(test, sarimax_pred)
        results.append(
            {
                "model": f"SARIMAX_seasonal_{seasonal_period}",
                "mae": mae,
                "rmse": rmse,
                "mape": mape,
            }
        )
        if persist_models and output_dir:
            try:
                from utils import save_model

                model_obj = SARIMAX(
                    train,
                    order=(1, 1, 1),
                    seasonal_order=(1, 0, 1, seasonal_period),
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit(disp=False)
                save_model(model_obj, f"{output_dir}/sarimax_seasonal_model.joblib")
            except Exception:
                pass
    except Exception as exc:
        warnings.warn(f"Seasonal SARIMAX failed: {exc}")

    # Prophet
    if Prophet is not None:
        try:
            prophet_fc = prophet_forecast(train, periods=len(test))
            # Align index: Prophet returns future dates; ensure same length and index
            prophet_fc = prophet_fc.reindex(test.index, method=None)
            mae = mean_absolute_error(test, prophet_fc)
            rmse = np.sqrt(mean_squared_error(test, prophet_fc))
            mape = _safe_mape(test, prophet_fc)
            results.append({"model": "Prophet", "mae": mae, "rmse": rmse, "mape": mape})
        except Exception as exc:
            warnings.warn(f"Prophet failed: {exc}")
    else:
        warnings.warn("Prophet not installed; skipping Prophet comparison")

    # Naive baseline
    try:
        naive_pred = naive_forecast(train, test.index)
        mae = mean_absolute_error(test, naive_pred)
        rmse = np.sqrt(mean_squared_error(test, naive_pred))
        mape = _safe_mape(test, naive_pred)
        results.append({"model": "Naive", "mae": mae, "rmse": rmse, "mape": mape})
    except Exception as exc:
        warnings.warn(f"Naive baseline failed: {exc}")

    # Moving average baseline
    try:
        ma_pred = moving_average_forecast(train, test.index, window=ma_window)
        mae = mean_absolute_error(test, ma_pred)
        rmse = np.sqrt(mean_squared_error(test, ma_pred))
        mape = _safe_mape(test, ma_pred)
        results.append(
            {
                "model": f"MovingAverage_window{ma_window}",
                "mae": mae,
                "rmse": rmse,
                "mape": mape,
            }
        )
    except Exception as exc:
        warnings.warn(f"Moving average baseline failed: {exc}")

    # Exponential smoothing baseline
    try:
        exp_pred = exp_smoothing_forecast(train, test.index)
        mae = mean_absolute_error(test, exp_pred)
        rmse = np.sqrt(mean_squared_error(test, exp_pred))
        mape = _safe_mape(test, exp_pred)
        results.append(
            {"model": "ExpSmoothing", "mae": mae, "rmse": rmse, "mape": mape}
        )
    except Exception as exc:
        warnings.warn(f"Exponential smoothing failed: {exc}")

    # RandomForest baseline
    try:
        rf_pred = rf_baseline_forecast(train, test.index, lags=rf_lags)
        common_idx = test.index.intersection(rf_pred.index)
        if common_idx.empty:
            raise ValueError(
                "RandomForest produced no predictions aligned with the test index"
            )
        mae = mean_absolute_error(test.loc[common_idx], rf_pred.loc[common_idx])
        rmse = np.sqrt(
            mean_squared_error(test.loc[common_idx], rf_pred.loc[common_idx])
        )
        mape = _safe_mape(test.loc[common_idx], rf_pred.loc[common_idx])
        results.append(
            {
                "model": f"RandomForest_lags{rf_lags}",
                "mae": mae,
                "rmse": rmse,
                "mape": mape,
            }
        )
        if persist_models and output_dir:
            try:
                from utils import save_model

                # fit the RF again and save it
                df = pd.DataFrame({"y": train})
                for lag in range(1, rf_lags + 1):
                    df[f"lag_{lag}"] = df["y"].shift(lag)
                df = df.dropna()
                X = df.drop(columns=["y"]).values
                y = df["y"].values
                rf = RandomForestRegressor(n_estimators=100, random_state=42)
                rf.fit(X, y)
                save_model(rf, f"{output_dir}/rf_model.joblib")
            except Exception:
                pass
    except Exception as exc:
        warnings.warn(f"RandomForest baseline failed: {exc}")

    df = pd.DataFrame(results)
    if output_dir is not None:
        try:
            df.to_csv(f"{output_dir}/model_comparison_metrics.csv", index=False)
        except Exception:
            pass
    return df
