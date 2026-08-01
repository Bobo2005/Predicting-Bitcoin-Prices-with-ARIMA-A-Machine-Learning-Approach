from forecasting import build_sample_series
from models import compare_models


def test_compare_models_runs_on_sample():
    s = build_sample_series()
    df = compare_models(
        s, train_split=0.8, forecast_steps=14, output_dir=None, persist_models=False
    )
    # Should return a DataFrame
    assert hasattr(df, "shape")
    assert "Naive" in df["model"].values
    assert any(name.startswith("MovingAverage") for name in df["model"].values)
    assert "ExpSmoothing" in df["model"].values
