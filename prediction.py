 #Install necessary libraries (run in your terminal or Jupyter notebook if not already installed)

# !pip install pandas yfinance statsmodels matplotlib sklearn
# Import libraries





import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
warnings.filterwarnings("ignore")  # Suppress warnings for cleaner output

# Fetch Bitcoin price data (e.g., last 8 years)
symbol = "BTC-USD"
data = yf.download(symbol, start="2017-04-01", end="2025-04-05", interval="1d") #Daily data
prices = data['Close'] # Use closing prices

# Display the first few rows
print("First few rows of prices:")
print(prices.head())

# Plot the price series
plt.figure(figsize=(12, 6))
plt.plot(prices, label="BTC Closing Price")
plt.title("Bitcoin Price (BTC-USD)")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.show()

# Check basic statistics
print("\nBasic statistics of prices:")
print(prices.describe())

# ADF test for stationarity
result = adfuller(prices.dropna())
print(f"\nADF Statistic: {result[0]}")
print(f"p-value: {result[1]}")

# First differencing
prices_diff = prices.diff().dropna()

# Plot differenced series
plt.figure(figsize=(12, 6))
plt.plot(prices_diff, label="Differenced BTC Price")
plt.title("Differenced Bitcoin Price")
plt.xlabel("Date")
plt.ylabel("Price Difference (USD)")
plt.legend()
plt.show()

# Re-run ADF test on differenced data
result_diff = adfuller(prices_diff)
print(f"\nADF Statistic (Differenced): {result_diff[0]}")
print(f"p-value (Differenced): {result_diff[1]}")

# Plot ACF and PACF
plt.figure(figsize=(12, 4))
plt.subplot(121)
plot_acf(prices_diff, ax=plt.gca(), lags=20)
plt.title("ACF Plot")
plt.subplot(122)
plot_pacf(prices_diff, ax=plt.gca(), lags=20)
plt.title("PACF Plot")
plt.show()

# Stationarity test result plots like ADF test visual
rolling_mean_raw = prices.rolling(window=50).mean()
rolling_mean_diff = prices_diff.rolling(window=50).mean()
plt.figure(figsize=(12, 8))
plt.subplot(211)
plt.plot(prices, label="Raw Prices")
plt.plot(rolling_mean_raw, label="50-Day Rolling Mean", color="red")
plt.title("Raw Bitcoin Prices with Rolling Mean")
plt.legend()
plt.subplot(212)
plt.plot(prices_diff, label="Differenced Prices")
plt.plot(rolling_mean_diff, label="50-Day Rolling Mean", color="red")
plt.title("Differenced Bitcoin Prices with Rolling Mean")
plt.legend()
plt.tight_layout()
plt.show()

# Split data into training and testing (e.g., 80% train, 20% test)
train_size = int(len(prices) * 0.8)
train, test = prices[:train_size], prices[train_size:]

# Fit ARIMA model
model = ARIMA(train, order=(1, 1, 1))
model_fit = model.fit()

# Summary of the model
print("\nARIMA Model Summary:")
print(model_fit.summary())

# Forecast on test set
forecast = model_fit.forecast(steps=len(test))

# Plot actual vs forecasted prices (ARIMA)
plt.figure(figsize=(12, 6))
plt.plot(train.index, train, label="Training Data")
plt.plot(test.index, test, label="Actual Prices", color="blue")
plt.plot(test.index, forecast, label="ARIMA Forecasted Prices", color="red")
plt.title("Bitcoin Price Forecast with ARIMA")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.show()

# Forecast future prices (e.g., 30 days beyond the last date)
future_forecast = model_fit.forecast(steps=30)
future_dates = pd.date_range(start=prices.index[-1], periods=30, freq="D")
plt.figure(figsize=(12, 6))
plt.plot(prices.index, prices, label="Historical Prices")
plt.plot(future_dates, future_forecast, label="Future Forecast", color="green")
plt.title("Bitcoin Price Future Forecast")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.show()

# Import metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Calculate ARIMA errors
mae = mean_absolute_error(test, forecast)
rmse = np.sqrt(mean_squared_error(test, forecast))
mape = np.mean(np.abs((test - forecast) / test.where(test != 0, np.finfo(float).eps))) * 100
print(f"\nARIMA Model Performance:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

# Naive Forecast Implementation
# Shift test data by 1 period to use the previous day's price as the forecast
naive_forecast = test.shift(1)

# Drop the first NaN value in naive_forecast (since shift introduces NaN at the start)
naive_forecast = naive_forecast.dropna()
test_aligned = test[1:] # Align test data by removing the first value
# Plot actual vs naive forecasted prices
plt.figure(figsize=(12, 6))
plt.plot(test.index, test, label="Actual Prices", color="blue")
plt.plot(test_aligned.index, naive_forecast, label="Naive Forecasted Prices",
color="orange")
plt.title("Bitcoin Price Forecast with Naive Method")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.show()

# Calculate Naive Forecast errors
naive_mae = mean_absolute_error(test_aligned, naive_forecast)
naive_rmse = np.sqrt(mean_squared_error(test_aligned, naive_forecast))
naive_mape = np.mean(np.abs((test_aligned - naive_forecast) /
test_aligned.where(test_aligned != 0, np.finfo(float).eps))) * 100
print(f"\nNaive Forecast Performance:")
print(f"Mean Absolute Error (MAE): {naive_mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {naive_rmse:.2f}")
print(f"Mean Absolute Percentage Error (MAPE): {naive_mape:.2f}%")

# Comparison of ARIMA vs Naive
print("\nPerformance Comparison:")
print(f"ARIMA MAE: {mae:.2f} vs Naive MAE: {naive_mae:.2f}")
print(f"ARIMA RMSE: {rmse:.2f} vs Naive RMSE: {naive_rmse:.2f}")
print(f"ARIMA MAPE: {mape:.2f}% vs Naive MAPE: {naive_mape:.2f}%")
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Calculate errors
mae = mean_absolute_error(test, forecast)
rmse = np.sqrt(mean_squared_error(test, forecast))
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

# Import necessary libraries
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore") # Suppress warnings for cleaner output

# Fetch Bitcoin price data (e.g., last 8 years)
symbol = "BTC-USD"
data = yf.download(symbol, start="2017-04-01", end="2025-04-05", interval="1d") # Daily data
prices = data['Close'] # Use closing prices

# Split data into training and testing (e.g., 80% train, 20% test)
train_size = int(len(prices) * 0.8)
train, test = prices[:train_size], prices[train_size:]

# Fit ARIMA model
model = ARIMA(train, order=(1, 1, 1))
model_fit = model.fit()

# Forecast on test set
forecast = model_fit.forecast(steps=len(test))

# Generate Figure 4.6.1: Forecast Plot
plt.figure(figsize=(12, 6))
plt.plot(train.index, train, label="Training Data", color="gray", alpha=0.5)
plt.plot(test.index, test, label="Actual Prices", color="blue")
plt.plot(test.index, forecast, label="ARIMA Forecasted Prices", color="red")
plt.title("Bitcoin Price Forecast with ARIMA", fontsize=14)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Price (USD)", fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, linestyle="--", alpha=0.7)
plt.tight_layout()

# Save the plot (optional, uncomment to save)140
# plt.savefig("Figure_4.6.1_Bitcoin_ARIMA_Forecast.png", dpi=300)
# Display the plot
plt.show()