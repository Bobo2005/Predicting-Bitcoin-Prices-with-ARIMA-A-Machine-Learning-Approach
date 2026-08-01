# Bitcoin Price Forecasting with ARIMA

This project forecasts Bitcoin prices using ARIMA and baseline comparison models. It downloads historical BTC-USD price data from Yahoo Finance when available, and falls back to a synthetic built-in sample dataset if network access is unavailable.

## Features

- Download historical BTC-USD price data or use built-in sample data
- Preprocess OHLCV data, create lag features, and scale inputs
- Select ARIMA order from configurable candidate parameters
- Compare multiple models: SARIMAX, seasonal SARIMAX, naive baseline, moving average, exponential smoothing, Prophet (optional), and RandomForest
- Save forecast plots, CSV metrics, model artifacts, and run metadata
- Support configuration via `config.yaml` and command-line overrides
- Minimal Streamlit dashboard for interactive exploration

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Running the project

Run the main forecasting pipeline:

```bash
python main.py
```

Override defaults with CLI arguments or `config.yaml` values:

```bash
python main.py --ticker BTC-USD --start-date 2017-04-01 --end-date 2025-04-05 --forecast-steps 30 --output-dir outputs
```

Load a config file and override a value on the command line:

```bash
python main.py --config config.yaml --forecast-steps 14
```

Run preprocessing and save the feature matrix:

```bash
python main.py --preprocess --use-sample --output-dir outputs
```

Run model comparison across all available baselines and optional Prophet:

```bash
python main.py --compare-models --use-sample --forecast-steps 14 --output-dir outputs
```

If you want to avoid network access, use the built-in sample data:

```bash
python main.py --use-sample
```

## Config file

The default configuration is stored in `config.yaml`. Use it to set default values for ticker, dates, model parameters, and preprocessing options. Command-line flags override the config file.

## Output files

The script writes the following files into the selected output directory:

- `arima_forecast.png`
- `future_forecast.png`
- `metrics.csv`
- `summary.txt`
- `model_comparison_metrics.csv`
- `model_comparison_summary.csv`
- `features.csv`
- `runs.log`
- Saved model artifacts (`rf_model.joblib`, `sarimax_model.joblib`, `sarimax_seasonal_model.joblib`)

## Streamlit dashboard

Run the interactive app with:

```bash
streamlit run streamlit_app.py
```

The dashboard loads feature data and model comparison results from the output folder.

### Dashboard features

- Overview of generated feature data and model-comparison metrics
- Download buttons for CSV output files and saved model artifacts
- Background forecast runner for `main.py` with CLI options exposed in the UI
- SHAP explainability for the saved RandomForest baseline model

### Notes

- SHAP is optional but recommended for explainability. It is included in `requirements.txt`.
- If a saved RandomForest model or feature dataset is not available, the dashboard will prompt you to generate them first.
