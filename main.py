from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path
from typing import Any

import matplotlib
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from forecasting import (
    DEFAULT_END_DATE,
    DEFAULT_START_DATE,
    DEFAULT_TICKER,
    build_forecast_series,
    build_naive_forecast,
    compute_adf,
    evaluate_forecast,
    load_price_series,
    split_train_test,
)
from models import compare_models
from preprocessing import (
    create_lag_features,
    create_ohlcv_features,
    detect_outliers_iqr,
    save_features,
    scale_features,
)
from utils import save_model, set_global_seed


def flatten_config(
    config: dict[str, Any], parent_key: str = "", sep: str = "_"
) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in config.items():
        name = f"{parent_key}{sep}{key}" if parent_key else key
        if isinstance(value, dict):
            flattened.update(flatten_config(value, name, sep=sep))
        else:
            flattened[name] = value
    return flattened


def load_config(path: str) -> dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh) or {}
            if not isinstance(loaded, dict):
                raise ValueError("Config file must contain a YAML mapping")
            return flatten_config(loaded)
    except FileNotFoundError:
        return {}
    except Exception as exc:
        raise RuntimeError(f"Unable to read configuration {path}: {exc}") from exc


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument(
        "--config", default="config.yaml", help="Path to config YAML file"
    )
    known_args, _ = config_parser.parse_known_args()
    config_values = load_config(known_args.config)

    parser = argparse.ArgumentParser(description="Forecast Bitcoin prices with ARIMA")
    parser.add_argument(
        "--config", default=known_args.config, help="Path to config YAML file"
    )
    parser.add_argument(
        "--ticker", default=DEFAULT_TICKER, help="Ticker symbol to fetch"
    )
    parser.add_argument(
        "--start-date",
        default=DEFAULT_START_DATE,
        help="Start date in YYYY-MM-DD format",
    )
    parser.add_argument(
        "--end-date", default=DEFAULT_END_DATE, help="End date in YYYY-MM-DD format"
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.8,
        help="Fraction of the data used for training",
    )
    parser.add_argument(
        "--forecast-steps",
        type=int,
        default=30,
        help="Number of future days to forecast",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory where plots and metrics will be stored",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global random seed for reproducible behavior",
    )
    parser.add_argument(
        "--use-sample",
        action="store_true",
        help="Use built-in sample data instead of downloading from Yahoo Finance",
    )
    parser.add_argument(
        "--preprocess",
        action="store_true",
        help="Run preprocessing/feature-engineering and save features to outputs/features.csv",
    )
    parser.add_argument(
        "--preprocess-lags",
        type=int,
        default=7,
        help="Number of lag features to create during preprocessing",
    )
    parser.add_argument(
        "--outlier-multiplier",
        type=float,
        default=1.5,
        help="IQR multiplier used for outlier clipping during preprocessing",
    )
    parser.add_argument(
        "--candidate-orders",
        default="1,1,1;2,1,2;1,1,2;2,1,1;0,1,1",
        help="Semi-colon separated ARIMA orders to evaluate (for example: 1,1,1;2,1,2)",
    )
    parser.add_argument(
        "--compare-models",
        action="store_true",
        help="Run model comparison (SARIMAX, Prophet if available, and RandomForest baseline)",
    )
    parser.add_argument(
        "--seasonal-period",
        type=int,
        default=7,
        help="Weekly seasonality period used for seasonal SARIMAX",
    )
    parser.add_argument(
        "--rf-lags",
        type=int,
        default=7,
        help="Number of lag features for the RandomForest baseline",
    )
    parser.add_argument(
        "--ma-window",
        type=int,
        default=7,
        help="Window size for the moving average baseline",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run a short tuning routine for RF and SARIMAX (time-consuming)",
    )
    parser.set_defaults(**config_values)
    return parser.parse_args()


def parse_candidate_orders(
    spec: str | Sequence[Sequence[int]],
) -> list[tuple[int, int, int]]:
    if isinstance(spec, list):
        orders = [tuple(int(value) for value in item) for item in spec]
    else:
        orders = []
        for chunk in spec.split(";"):
            cleaned = chunk.strip()
            if not cleaned:
                continue
            params = tuple(int(value) for value in cleaned.split(",") if value.strip())
            if len(params) != 3:
                raise ValueError(
                    f"Invalid order '{chunk}'. Expected three integers separated by commas"
                )
            orders.append(params)
    if not orders:
        raise ValueError("No candidate orders were provided")
    return orders


def save_forecast_plot(
    output_dir: Path,
    prices: pd.Series,
    train_series: pd.Series,
    test_series: pd.Series,
    forecast: pd.Series,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        train_series.index, train_series, label="Training Data", color="gray", alpha=0.7
    )
    ax.plot(test_series.index, test_series, label="Actual Prices", color="blue")
    ax.plot(forecast.index, forecast, label="ARIMA Forecast", color="red")
    ax.set_title("Bitcoin Price Forecast with ARIMA")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_dir / "arima_forecast.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_future_forecast_plot(
    output_dir: Path, prices: pd.Series, future_forecast: pd.Series
) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(prices.index, prices, label="Historical Prices", color="black")
    ax.plot(
        future_forecast.index, future_forecast, label="Future Forecast", color="green"
    )
    ax.set_title("Bitcoin Future Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_dir / "future_forecast.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_metrics(
    output_dir: Path,
    arima_metrics: dict[str, float],
    naive_metrics: dict[str, float],
    order: tuple[int, int, int],
    adf_statistic: float,
    adf_pvalue: float,
) -> None:
    metrics = pd.DataFrame(
        [
            {
                "model": "ARIMA",
                "order": f"({order[0]}, {order[1]}, {order[2]})",
                **arima_metrics,
            },
            {"model": "Naive", "order": "N/A", **naive_metrics},
        ]
    )
    metrics.to_csv(output_dir / "metrics.csv", index=False)

    summary_lines = [
        f"ARIMA order: {order}",
        f"ADF Statistic: {adf_statistic:.4f}",
        f"ADF p-value: {adf_pvalue:.4f}",
        "",
        "Performance metrics:",
        f"- ARIMA MAE: {arima_metrics['mae']:.2f}",
        f"- ARIMA RMSE: {arima_metrics['rmse']:.2f}",
        f"- ARIMA MAPE: {arima_metrics['mape']:.2f}%",
        f"- Naive MAE: {naive_metrics['mae']:.2f}",
        f"- Naive RMSE: {naive_metrics['rmse']:.2f}",
        f"- Naive MAPE: {naive_metrics['mape']:.2f}%",
    ]
    (output_dir / "summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")

    # append run metadata to runs log for experiment tracking
    try:
        from utils import log_run

        run_meta = {
            "config": {
                "arima_order": f"{order}",
                "train_samples": (
                    len(arima_metrics) if isinstance(arima_metrics, dict) else None
                ),
            },
            "metrics": {"arima": arima_metrics, "naive": naive_metrics},
        }
        log_run(str(output_dir / "runs.log"), run_meta)
    except Exception:
        pass


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    warnings.filterwarnings("ignore")

    # Optionally return a full OHLCV DataFrame when preprocessing is requested
    if args.preprocess:
        df = load_price_series(
            ticker=args.ticker,
            start_date=args.start_date,
            end_date=args.end_date,
            use_sample=args.use_sample,
            return_df=True,
        )
        # Create OHLCV features (will raise if no Close column)
        features = create_ohlcv_features(df)
        # Detect and clip outliers on close
        features["close"] = detect_outliers_iqr(
            features["close"], multiplier=args.outlier_multiplier
        )
        # Add lag features for supervised models
        lagged = create_lag_features(features["close"], lags=args.preprocess_lags)
        # Join rolling features with lagged (aligning indexes)
        combined = lagged.join(features.drop(columns=["close"]).reindex(lagged.index))
        scaled, _scaler = scale_features(combined.fillna(0.0), method="standard")
        output_features_path = output_dir / "features.csv"
        save_features(scaled, str(output_features_path))
        print(f"Features saved to {output_features_path}")
        return

    prices = load_price_series(
        ticker=args.ticker,
        start_date=args.start_date,
        end_date=args.end_date,
        use_sample=args.use_sample,
    )

    if len(prices) < 20:
        raise ValueError("Not enough data to build a meaningful forecast")

    adf_statistic, adf_pvalue = compute_adf(prices)
    train_series, test_series = split_train_test(prices, train_split=args.train_split)
    candidate_orders = parse_candidate_orders(args.candidate_orders)

    # If requested, run a model comparison across several model types
    if args.compare_models:
        cmp_df = compare_models(
            prices,
            train_split=args.train_split,
            forecast_steps=args.forecast_steps,
            output_dir=str(output_dir),
            seasonal_period=args.seasonal_period,
            rf_lags=args.rf_lags,
            ma_window=args.ma_window,
        )
        print("Model comparison results:")
        print(cmp_df.to_string(index=False))
        # Also save the comparison table in the outputs folder (models.compare_models already writes a CSV when possible)
        try:
            cmp_df.to_csv(output_dir / "model_comparison_summary.csv", index=False)
        except Exception:
            pass
        return

    forecast_result = build_forecast_series(
        train_series=train_series,
        test_series=test_series,
        full_series=prices,
        forecast_steps=args.forecast_steps,
        candidate_orders=candidate_orders,
    )

    arima_metrics = evaluate_forecast(test_series, forecast_result["forecast"])
    naive_forecast, naive_actual = build_naive_forecast(test_series)
    naive_metrics = evaluate_forecast(naive_actual, naive_forecast)

    save_forecast_plot(
        output_dir, prices, train_series, test_series, forecast_result["forecast"]
    )
    save_future_forecast_plot(output_dir, prices, forecast_result["future_forecast"])
    save_metrics(
        output_dir,
        arima_metrics,
        naive_metrics,
        forecast_result["order"],
        adf_statistic,
        adf_pvalue,
    )

    print(f"Using ticker: {args.ticker}")
    print(f"Training samples: {len(train_series)}")
    print(f"Test samples: {len(test_series)}")
    print(f"Selected ARIMA order: {forecast_result['order']}")
    print(f"ADF Statistic: {adf_statistic:.4f}")
    print(f"ADF p-value: {adf_pvalue:.4f}")
    print("Performance metrics:")
    print(f"- ARIMA MAE: {arima_metrics['mae']:.2f}")
    print(f"- ARIMA RMSE: {arima_metrics['rmse']:.2f}")
    print(f"- ARIMA MAPE: {arima_metrics['mape']:.2f}%")
    print(f"- Naive MAE: {naive_metrics['mae']:.2f}")
    print(f"- Naive RMSE: {naive_metrics['rmse']:.2f}")
    print(f"- Naive MAPE: {naive_metrics['mape']:.2f}%")
    print(f"Results written to {output_dir}")


if __name__ == "__main__":
    main()
