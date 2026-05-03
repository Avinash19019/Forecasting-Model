# Seattle Weather Forecasting

Predict future weather trends from historical Seattle daily weather data using Python.

This project cleans and preprocesses the dataset, builds time-series regression features, evaluates model accuracy on a chronological test split, and visualizes future forecasts.

## Dataset

The project uses `data/seattle-weather.csv` with these columns:

- `date`
- `precipitation`
- `temp_max`
- `temp_min`
- `wind`
- `weather`



## Features

- Cleans missing values and sorts observations by date
- Creates lag, rolling average, trend, and seasonal calendar features
- Uses chronological train/test splitting for realistic forecasting
- Trains a ridge-regularized linear regression model with NumPy
- Evaluates predictions with MAE, RMSE, and R2
- Forecasts future weather values for the next N days
- Saves prediction tables and a forecast visualization



## Outputs

After running the script, files are saved in `outputs/`:

- `metrics.txt`: model accuracy scores
- `forecast.csv`: future forecast values
- `test_predictions.csv`: actual vs predicted values on the test set
- `forecast_plot.png`: chart showing historical data, test predictions, and future forecast
- `forecast_plot.svg`: fallback chart created when `matplotlib` is not installed


