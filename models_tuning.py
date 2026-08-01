from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.statespace.sarimax import SARIMAX


def rolling_cv_scores_arima(
    series: pd.Series,
    order: tuple[int, int, int],
    train_window: int,
    horizon: int,
    max_splits: int = 5,
) -> float:
    """Compute average MAE for ARIMA order using expanding/rolling evaluation.

    This is a simple routine to estimate parameter quality without a full
    production CV.
    """
    n = len(series)
    if n < train_window + horizon:
        raise ValueError("Series too short for train window and horizon")
    maes = []
    start = n - (train_window + horizon)
    splits = 0
    for i in range(start, n - horizon):
        if splits >= max_splits:
            break
        train = series.iloc[i : i + train_window]
        test_index = series.index[i + train_window : i + train_window + horizon]
        try:
            model = SARIMAX(
                train,
                order=order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            fit = model.fit(disp=False)
            preds = fit.get_forecast(steps=horizon).predicted_mean
            preds.index = test_index
            actual = series.loc[test_index]
            maes.append(mean_absolute_error(actual, preds))
            splits += 1
        except Exception:
            continue
    return float(np.mean(maes)) if maes else float("nan")


def grid_search_rf(
    train_series: pd.Series,
    param_grid: dict,
    lags: int = 7,
) -> dict:
    """Search RandomForest hyperparameters using a simple validation split.

    Returns a dict with best_params and best_score. Supports a cross-validation
    mode if "cv_splits" is included in param_grid.
    """
    n = len(train_series)
    split = int(n * 0.8)
    train = train_series.iloc[:split]
    val = train_series.iloc[split:]

    # build lag features
    def make_Xy(s: pd.Series):
        df = pd.DataFrame({"y": s})
        for lag in range(1, lags + 1):
            df[f"lag_{lag}"] = df["y"].shift(lag)
        df = df.dropna()
        return df.drop(columns=["y"]), df["y"]

    X_train, y_train = make_Xy(train)
    X_val, y_val = make_Xy(pd.concat([train.iloc[-lags:], val]))

    best_score = None
    best_params = None
    from itertools import product

    # Allow cv splits via a key 'cv_splits' in param_grid. This does not tune
    # the split count itself.
    cv_splits = None
    if "cv_splits" in param_grid:
        cv_splits = param_grid.pop("cv_splits")

    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]
    for combo in product(*values):
        params = dict(zip(keys, combo))
        rf = RandomForestRegressor(random_state=42, **params)
        rf.fit(X_train.values, y_train.values)
        preds = rf.predict(X_val.values)
        mae = mean_absolute_error(y_val.values, preds)
        if best_score is None or mae < best_score:
            best_score = mae
            best_params = params

    # If cv_splits is provided, run a simple expanding-window CV and override
    # best_score with the average.
    if cv_splits and isinstance(cv_splits, int) and cv_splits > 1:
        # split the combined train+val into cv_splits folds and average
        n = len(train_series)
        fold_size = max(1, n // (cv_splits + 1))
        cv_scores = []
        for start in range(0, cv_splits):
            split_point = (start + 1) * fold_size
            tr = train_series.iloc[:split_point]
            va = train_series.iloc[split_point : split_point + fold_size]
            if len(va) < 1:
                continue

            # build features
            def make_Xy(s: pd.Series):
                df = pd.DataFrame({"y": s})
                for lag in range(1, lags + 1):
                    df[f"lag_{lag}"] = df["y"].shift(lag)
                df = df.dropna()
                return df.drop(columns=["y"]), df["y"]

            X_tr, y_tr = make_Xy(tr)
            X_va, y_va = make_Xy(pd.concat([tr.iloc[-lags:], va]))
            model = RandomForestRegressor(random_state=42, **best_params)
            model.fit(X_tr.values, y_tr.values)
            p = model.predict(X_va.values)
            cv_scores.append(mean_absolute_error(y_va.values, p))
        if cv_scores:
            best_score = float(np.mean(cv_scores))

    return {"best_params": best_params, "best_score": best_score}
