from forecasting import build_sample_series, load_price_series


def test_load_sample_series():
    s = build_sample_series()
    assert len(s) > 0


def test_load_price_series_df():
    df = load_price_series(use_sample=True, return_df=True)
    assert "Close" in df.columns
