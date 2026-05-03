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

## Project Structure

```text
Seattle-Weather-Forecasting/
├── data/
│   └── seattle-weather.csv
├── outputs/
│   ├── forecast.csv
│   ├── metrics.txt
│   └── forecast_plot.png
├── src/
│   └── train_model.py
├── .gitignore
├── requirements.txt
└── README.md
```

## Features

- Cleans missing values and sorts observations by date
- Creates lag, rolling average, trend, and seasonal calendar features
- Uses chronological train/test splitting for realistic forecasting
- Trains a ridge-regularized linear regression model with NumPy
- Evaluates predictions with MAE, RMSE, and R2
- Forecasts future weather values for the next N days
- Saves prediction tables and a forecast visualization

## Setup

```bash
pip install -r requirements.txt
```

## Run

Forecast maximum temperature for the next 30 days:

```bash
python src/train_model.py
```

Forecast another numeric column:

```bash
python src/train_model.py --target precipitation --forecast-days 45
```

Available numeric targets are:

- `precipitation`
- `temp_max`
- `temp_min`
- `wind`

## Outputs

After running the script, files are saved in `outputs/`:

- `metrics.txt`: model accuracy scores
- `forecast.csv`: future forecast values
- `test_predictions.csv`: actual vs predicted values on the test set
- `forecast_plot.png`: chart showing historical data, test predictions, and future forecast
- `forecast_plot.svg`: fallback chart created when `matplotlib` is not installed

## GitHub Upload

```bash
git init
git add .
git commit -m "Initial commit: Seattle weather forecasting project"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/Seattle-Weather-Forecasting.git
git push -u origin main
```

Replace `YOUR_USERNAME` with your GitHub username.
