# Bitcoin Price Forecasting with ARIMA

This project forecasts Bitcoin prices using an ARIMA time-series model. It downloads historical BTC-USD price data from Yahoo Finance when available, and falls back to a built-in sample series if network access is unavailable.

## Features

- Downloads or uses fallback price data for BTC-USD
- Checks stationarity with the Augmented Dickey-Fuller test
- Evaluates a small set of ARIMA candidate orders and selects the best one using AIC
- Compares ARIMA forecasts to a naive baseline
- Saves forecast plots and evaluation metrics to an output directory

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Running the project

Run the project with:

```bash
python main.py
```

You can also override defaults:

```bash
python main.py --ticker BTC-USD --start-date 2017-04-01 --end-date 2025-04-05 --forecast-steps 30 --output-dir outputs
```

To run a model comparison (SARIMAX, optional Prophet, and a RandomForest baseline):

```bash
python main.py --compare-models --use-sample --forecast-steps 14 --output-dir outputs
```

If you want to avoid network access, use the built-in sample data:

```bash
python main.py --use-sample
```

## Output files

The script writes the following files into the selected output directory:

- `arima_forecast.png`
- `future_forecast.png`
- `metrics.csv`
- `summary.txt`
