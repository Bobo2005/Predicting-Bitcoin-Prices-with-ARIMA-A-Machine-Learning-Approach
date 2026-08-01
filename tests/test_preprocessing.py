from forecasting import build_sample_df
from preprocessing import create_ohlcv_features, detect_outliers_iqr


def test_ohlcv_features_on_sample():
    df = build_sample_df()
    feats = create_ohlcv_features(df)
    assert "close" in feats.columns
    assert "roll_mean_7" in feats.columns


def test_outlier_clipping():
    df = build_sample_df()
    feats = create_ohlcv_features(df)
    # artificially inject an outlier
    feats.loc[feats.index[0], "close"] = feats["close"].max() * 1000
    clipped = detect_outliers_iqr(feats["close"])
    assert clipped.max() < feats["close"].max() * 1000
