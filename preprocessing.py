from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def detect_outliers_iqr(series: pd.Series, multiplier: float = 1.5) -> pd.Series:
    """Detect outliers using the IQR rule and clip them to the fence values.

    Returns a copy with outliers clipped to the nearest fence (no removal).
    """
    if series.empty:
        return series
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    return series.clip(lower=lower, upper=upper)


def create_lag_features(series: pd.Series, lags: int = 7) -> pd.DataFrame:
    """Create lag features for a univariate series. Returns DataFrame with lag_1..lag_n and y."""
    df = pd.DataFrame({"y": series})
    for lag in range(1, lags + 1):
        df[f"lag_{lag}"] = df["y"].shift(lag)
    return df.dropna()


def create_ohlcv_features(df: pd.DataFrame) -> pd.DataFrame:
    """Given a DataFrame containing Open/High/Low/Close/Volume (or a Close series),
    return a features DataFrame including returns, log-returns, rolling stats, and volume features.
    """
    if "Close" not in df.columns:
        raise ValueError("DataFrame must contain a 'Close' column")

    out = pd.DataFrame(index=df.index)
    out["close"] = df["Close"]
    out["return_1d"] = df["Close"].pct_change()
    out["log_return_1d"] = np.log(df["Close"]).diff()
    out["volume"] = df["Volume"] if "Volume" in df.columns else 0.0

    # Rolling features
    out["roll_mean_7"] = df["Close"].rolling(7).mean()
    out["roll_std_7"] = df["Close"].rolling(7).std()
    out["roll_mean_30"] = df["Close"].rolling(30).mean()
    out["roll_std_30"] = df["Close"].rolling(30).std()

    # High-low range feature if available
    if "High" in df.columns and "Low" in df.columns:
        out["hl_range"] = (df["High"] - df["Low"]) / df["Close"].replace(0, np.nan)

    return out.dropna()


def scale_features(
    df: pd.DataFrame, method: str = "standard"
) -> tuple[pd.DataFrame, object]:
    """Scale features using StandardScaler or MinMaxScaler. Returns (scaled_df, scaler)."""
    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError("Unknown scaler method")

    scaled = scaler.fit_transform(df.values)
    scaled_df = pd.DataFrame(scaled, index=df.index, columns=df.columns)
    return scaled_df, scaler


def save_features(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path)
