from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "seattle-weather.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"
NUMERIC_TARGETS = ("precipitation", "temp_max", "temp_min", "wind")


@dataclass
class ModelResult:
    weights: np.ndarray
    feature_means: pd.Series
    feature_stds: pd.Series
    feature_columns: list[str]


def load_and_clean_data(data_path: Path) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    df.columns = [column.strip().lower() for column in df.columns]

    required_columns = {"date", "precipitation", "temp_max", "temp_min", "wind", "weather"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").drop_duplicates(subset=["date"])
    df = df.set_index("date").asfreq("D")

    for column in NUMERIC_TARGETS:
        df[column] = pd.to_numeric(df[column], errors="coerce")
        df[column] = df[column].interpolate(method="time").ffill().bfill()

    df["weather"] = df["weather"].ffill().bfill().fillna("unknown")
    return df.reset_index()


def add_time_series_features(df: pd.DataFrame, target: str, require_target: bool = True) -> pd.DataFrame:
    featured = df.copy()
    dates = featured["date"]

    featured["day_of_year"] = dates.dt.dayofyear
    featured["month"] = dates.dt.month
    featured["year"] = dates.dt.year
    featured["time_index"] = np.arange(len(featured), dtype=float)
    featured["day_sin"] = np.sin(2 * np.pi * featured["day_of_year"] / 365.25)
    featured["day_cos"] = np.cos(2 * np.pi * featured["day_of_year"] / 365.25)

    for lag in (1, 2, 3, 7, 14, 30):
        featured[f"{target}_lag_{lag}"] = featured[target].shift(lag)

    for window in (7, 14, 30):
        shifted = featured[target].shift(1)
        featured[f"{target}_rolling_mean_{window}"] = shifted.rolling(window).mean()
        featured[f"{target}_rolling_std_{window}"] = shifted.rolling(window).std()

    required_columns = build_feature_columns(target)
    if require_target:
        required_columns = [target, *required_columns]

    return featured.dropna(subset=required_columns).reset_index(drop=True)


def chronological_split(df: pd.DataFrame, test_size: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")

    split_index = int(len(df) * (1 - test_size))
    if split_index < 30 or len(df) - split_index < 10:
        raise ValueError("Dataset is too small for the requested train/test split")

    return df.iloc[:split_index].copy(), df.iloc[split_index:].copy()


def fit_ridge_regression(
    train_df: pd.DataFrame,
    target: str,
    feature_columns: list[str],
    alpha: float,
) -> ModelResult:
    x_train = train_df[feature_columns].astype(float)
    y_train = train_df[target].astype(float).to_numpy()

    feature_means = x_train.mean()
    feature_stds = x_train.std().replace(0, 1)
    x_scaled = ((x_train - feature_means) / feature_stds).to_numpy()
    x_design = np.column_stack([np.ones(len(x_scaled)), x_scaled])

    penalty = np.eye(x_design.shape[1]) * alpha
    penalty[0, 0] = 0
    weights = np.linalg.solve(x_design.T @ x_design + penalty, x_design.T @ y_train)

    return ModelResult(
        weights=weights,
        feature_means=feature_means,
        feature_stds=feature_stds,
        feature_columns=feature_columns,
    )


def predict(model: ModelResult, df: pd.DataFrame) -> np.ndarray:
    x = df[model.feature_columns].astype(float)
    x_scaled = ((x - model.feature_means) / model.feature_stds).to_numpy()
    x_design = np.column_stack([np.ones(len(x_scaled)), x_scaled])
    return x_design @ model.weights


def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    residuals = actual - predicted
    mae = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(np.mean(residuals**2)))
    denominator = np.sum((actual - np.mean(actual)) ** 2)
    r2 = float(1 - np.sum(residuals**2) / denominator) if denominator != 0 else float("nan")
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def build_feature_columns(target: str) -> list[str]:
    return [
        "month",
        "year",
        "time_index",
        "day_sin",
        "day_cos",
        f"{target}_lag_1",
        f"{target}_lag_2",
        f"{target}_lag_3",
        f"{target}_lag_7",
        f"{target}_lag_14",
        f"{target}_lag_30",
        f"{target}_rolling_mean_7",
        f"{target}_rolling_std_7",
        f"{target}_rolling_mean_14",
        f"{target}_rolling_std_14",
        f"{target}_rolling_mean_30",
        f"{target}_rolling_std_30",
    ]


def make_future_forecast(
    clean_df: pd.DataFrame,
    model: ModelResult,
    target: str,
    forecast_days: int,
) -> pd.DataFrame:
    history = clean_df[["date", target]].copy()
    forecasts: list[dict[str, float | pd.Timestamp]] = []

    for _ in range(forecast_days):
        next_date = history["date"].max() + pd.Timedelta(days=1)
        placeholder = pd.DataFrame({"date": [next_date], target: [np.nan]})
        candidate_history = pd.concat([history, placeholder], ignore_index=True)

        featured = add_time_series_features(candidate_history, target, require_target=False)
        next_row = featured[featured["date"] == next_date]
        if next_row.empty:
            raise RuntimeError("Unable to create features for future forecast row")

        predicted_value = float(predict(model, next_row)[0])
        if target == "precipitation":
            predicted_value = max(0.0, predicted_value)

        history.loc[len(history)] = [next_date, predicted_value]
        forecasts.append({"date": next_date, f"predicted_{target}": predicted_value})

    return pd.DataFrame(forecasts)


def save_metrics(metrics: dict[str, float], output_path: Path, target: str) -> None:
    lines = [f"Target: {target}", ""]
    lines.extend(f"{metric}: {value:.4f}" for metric, value in metrics.items())
    output_path.write_text("\n".join(lines), encoding="utf-8")


def save_plot(
    clean_df: pd.DataFrame,
    test_predictions: pd.DataFrame,
    future_forecast: pd.DataFrame,
    target: str,
    output_path: Path,
) -> Path:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        svg_path = output_path.with_suffix(".svg")
        save_svg_plot(clean_df, test_predictions, future_forecast, target, svg_path)
        return svg_path

    plt.figure(figsize=(13, 7))
    plt.plot(clean_df["date"], clean_df[target], label="Historical", color="#2868a8", linewidth=1.6)
    plt.plot(
        test_predictions["date"],
        test_predictions[f"predicted_{target}"],
        label="Test prediction",
        color="#d95f02",
        linewidth=1.8,
    )
    plt.plot(
        future_forecast["date"],
        future_forecast[f"predicted_{target}"],
        label="Future forecast",
        color="#1b9e77",
        linewidth=2.2,
    )
    plt.title(f"Seattle Weather Forecast: {target}")
    plt.xlabel("Date")
    plt.ylabel(target)
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return output_path


def save_svg_plot(
    clean_df: pd.DataFrame,
    test_predictions: pd.DataFrame,
    future_forecast: pd.DataFrame,
    target: str,
    output_path: Path,
) -> None:
    width, height = 1100, 620
    margin = 70

    series = [
        ("Historical", clean_df["date"], clean_df[target], "#2868a8"),
        ("Test prediction", test_predictions["date"], test_predictions[f"predicted_{target}"], "#d95f02"),
        ("Future forecast", future_forecast["date"], future_forecast[f"predicted_{target}"], "#1b9e77"),
    ]
    all_dates = pd.concat([pd.Series(values[1]) for values in series])
    all_values = pd.concat([pd.Series(values[2]) for values in series]).astype(float)

    min_date = all_dates.min()
    max_date = all_dates.max()
    min_value = float(all_values.min())
    max_value = float(all_values.max())
    value_padding = (max_value - min_value) * 0.08 or 1.0
    min_value -= value_padding
    max_value += value_padding

    date_span = max((max_date - min_date).days, 1)
    value_span = max_value - min_value

    def x_scale(date_value: pd.Timestamp) -> float:
        return margin + ((date_value - min_date).days / date_span) * (width - 2 * margin)

    def y_scale(value: float) -> float:
        return height - margin - ((value - min_value) / value_span) * (height - 2 * margin)

    def polyline(dates: pd.Series, values: pd.Series) -> str:
        points = [
            f"{x_scale(pd.Timestamp(date_value)):.2f},{y_scale(float(value)):.2f}"
            for date_value, value in zip(dates, values)
        ]
        return " ".join(points)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2}" y="34" text-anchor="middle" font-family="Arial" font-size="22" fill="#222">Seattle Weather Forecast: {target}</text>',
        f'<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="#333" stroke-width="1"/>',
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" stroke="#333" stroke-width="1"/>',
    ]

    for tick in range(6):
        value = min_value + tick * value_span / 5
        y = y_scale(value)
        lines.append(f'<line x1="{margin}" y1="{y:.2f}" x2="{width - margin}" y2="{y:.2f}" stroke="#e5e5e5" stroke-width="1"/>')
        lines.append(f'<text x="{margin - 10}" y="{y + 4:.2f}" text-anchor="end" font-family="Arial" font-size="12" fill="#555">{value:.1f}</text>')

    for label, dates, values, color in series:
        lines.append(
            f'<polyline points="{polyline(dates, values)}" fill="none" stroke="{color}" stroke-width="2.5"/>'
        )

    legend_x = width - margin - 180
    for index, (label, _, _, color) in enumerate(series):
        y = margin + index * 26
        lines.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 28}" y2="{y}" stroke="{color}" stroke-width="3"/>')
        lines.append(f'<text x="{legend_x + 38}" y="{y + 4}" font-family="Arial" font-size="13" fill="#333">{label}</text>')

    lines.append(f'<text x="{width / 2}" y="{height - 18}" text-anchor="middle" font-family="Arial" font-size="13" fill="#555">Date range: {min_date.date()} to {max_date.date()}</text>')
    lines.append(f'<text x="22" y="{height / 2}" transform="rotate(-90 22 {height / 2})" text-anchor="middle" font-family="Arial" font-size="13" fill="#555">{target}</text>')
    lines.append("</svg>")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Seattle weather forecasting model.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target", choices=NUMERIC_TARGETS, default="temp_max")
    parser.add_argument("--forecast-days", type=int, default=30)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--alpha", type=float, default=1.0, help="Ridge regularization strength.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.forecast_days < 1:
        raise ValueError("forecast-days must be at least 1")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    clean_df = load_and_clean_data(args.data_path)
    model_df = add_time_series_features(clean_df, args.target)
    feature_columns = build_feature_columns(args.target)
    train_df, test_df = chronological_split(model_df, args.test_size)

    model = fit_ridge_regression(train_df, args.target, feature_columns, args.alpha)
    test_df[f"predicted_{args.target}"] = predict(model, test_df)
    if args.target == "precipitation":
        test_df[f"predicted_{args.target}"] = test_df[f"predicted_{args.target}"].clip(lower=0)

    metrics = calculate_metrics(
        test_df[args.target].to_numpy(),
        test_df[f"predicted_{args.target}"].to_numpy(),
    )
    future_forecast = make_future_forecast(clean_df, model, args.target, args.forecast_days)

    test_predictions = test_df[["date", args.target, f"predicted_{args.target}"]]
    test_predictions.to_csv(args.output_dir / "test_predictions.csv", index=False)
    future_forecast.to_csv(args.output_dir / "forecast.csv", index=False)
    save_metrics(metrics, args.output_dir / "metrics.txt", args.target)
    plot_path = save_plot(
        clean_df,
        test_predictions,
        future_forecast,
        args.target,
        args.output_dir / "forecast_plot.png",
    )

    print(f"Model trained for target: {args.target}")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    print(f"Forecast saved to: {args.output_dir / 'forecast.csv'}")
    print(f"Plot saved to: {plot_path}")


if __name__ == "__main__":
    main()
