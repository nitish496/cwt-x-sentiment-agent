"""Leakage-safe expanding walk-forward forecasts and simple strategy metrics."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import HORIZONS


def _feature_columns(frame: pd.DataFrame, supplied: Sequence[str] | None) -> list[str]:
    if supplied is None:
        excluded_prefixes = ("target_",)
        excluded = {"timestamp_utc", "open", "symbol", "venue", "quote_currency", "source"}
        columns = [
            col
            for col in frame.select_dtypes(include=[np.number]).columns
            if col not in excluded and not col.startswith(excluded_prefixes)
        ]
    else:
        columns = list(supplied)
    if not columns:
        raise ValueError("at least one numeric feature column is required")
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"feature columns are missing: {sorted(missing)}")
    if any(col == "timestamp_utc" or col.startswith("target_") for col in columns):
        raise ValueError("timestamps and target columns cannot be model features")
    non_numeric = [col for col in columns if not pd.api.types.is_numeric_dtype(frame[col])]
    if non_numeric:
        raise ValueError(f"features must be numeric: {non_numeric}")
    return columns


def expanding_walk_forward_predictions(
    supervised: pd.DataFrame,
    *,
    n_splits: int = 4,
    test_size: int = 21,
    min_train_size: int = 60,
    purge_bars: int = 5,
    horizons: Sequence[int] = HORIZONS,
    feature_columns: Sequence[str] | None = None,
    ridge_alpha: float = 1.0,
) -> pd.DataFrame:
    """Generate expanding-window OOS predictions for horizons 1..5.

    Each fold fits its imputer, scaler, and Ridge model only on that fold's
    training rows. Training origins are purged before each validation start,
    and labels whose target date reaches the validation period are excluded.
    Synthetic fixtures are suitable for exercising this function, but their
    output must never be presented as empirical performance.
    """
    required = {"timestamp_utc", "close"}
    missing = required - set(supervised.columns)
    if missing:
        raise ValueError(f"supervised frame is missing required columns: {sorted(missing)}")
    if n_splits < 1 or test_size < 1 or min_train_size < 2 or purge_bars < 0:
        raise ValueError("split sizes must be positive and purge_bars cannot be negative")
    if ridge_alpha <= 0:
        raise ValueError("ridge_alpha must be positive")

    frame = supervised.sort_values("timestamp_utc").reset_index(drop=True).copy()
    if frame["timestamp_utc"].duplicated().any():
        raise ValueError("supervised frame must contain one origin per UTC day")
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True, errors="raise")
    columns = _feature_columns(frame, feature_columns)
    total_rows = len(frame)
    first_test_start = total_rows - n_splits * test_size
    if first_test_start < min_train_size + purge_bars:
        raise ValueError(
            "not enough rows for requested expanding folds, minimum training window, and purge"
        )

    output: list[dict[str, object]] = []
    for fold_number in range(n_splits):
        test_start = first_test_start + fold_number * test_size
        test_end = min(test_start + test_size, total_rows)
        test_start_time = frame.loc[test_start, "timestamp_utc"]
        for horizon in horizons:
            target_col = f"target_log_return_h{horizon}"
            target_date_col = f"target_date_h{horizon}"
            target_close_col = f"target_close_h{horizon}"
            needed = {target_col, target_date_col, target_close_col}
            if not needed.issubset(frame.columns):
                raise ValueError(f"supervised frame is missing labels for horizon {horizon}")

            purge = max(purge_bars, horizon)
            train_end = max(0, test_start - purge)
            train = frame.iloc[:train_end].copy()
            target_dates = pd.to_datetime(train[target_date_col], utc=True, errors="coerce")
            train_mask = train[target_col].notna() & target_dates.lt(test_start_time)
            train = train.loc[train_mask]
            test = frame.iloc[test_start:test_end].copy()
            test = test.loc[test[target_col].notna() & test[target_close_col].notna()]
            if len(train) < min_train_size:
                raise ValueError(
                    f"fold {fold_number + 1}, horizon {horizon}: only {len(train)} training rows"
                )
            if test.empty:
                continue

            x_train = train[columns].replace([np.inf, -np.inf], np.nan)
            x_test = test[columns].replace([np.inf, -np.inf], np.nan)
            model = make_pipeline(
                SimpleImputer(strategy="median", keep_empty_features=True),
                StandardScaler(),
                Ridge(alpha=ridge_alpha),
            )
            model.fit(x_train, train[target_col].astype(float))
            predicted_return = model.predict(x_test)
            if not np.isfinite(predicted_return).all():
                raise RuntimeError("model produced a non-finite prediction")

            for position, (_, row) in enumerate(test.iterrows()):
                origin_close = float(row["close"])
                actual_return = float(row[target_col])
                predicted = float(predicted_return[position])
                actual_close = float(row[target_close_col])
                output.append(
                    {
                        "origin_timestamp_utc": row["timestamp_utc"],
                        "target_timestamp_utc": pd.to_datetime(
                            row[target_date_col], utc=True, errors="raise"
                        ),
                        "fold": fold_number + 1,
                        "horizon_days": int(horizon),
                        "origin_close": origin_close,
                        "actual_log_return": actual_return,
                        "predicted_log_return": predicted,
                        "actual_close": actual_close,
                        "predicted_close": origin_close * float(np.exp(predicted)),
                        "next_open": (
                            float(row[f"target_open_h{horizon}"])
                            if f"target_open_h{horizon}" in row.index
                            and pd.notna(row[f"target_open_h{horizon}"])
                            else np.nan
                        ),
                        "random_walk_log_return": 0.0,
                        "random_walk_close": origin_close,
                        "train_rows": len(train),
                        "train_end_utc": train["timestamp_utc"].max(),
                        "test_start_utc": test_start_time,
                        "model_name": "ridge",
                    }
                )

    return pd.DataFrame(output).sort_values(
        ["origin_timestamp_utc", "horizon_days"]
    ).reset_index(drop=True)


def forecast_next_horizons(
    supervised: pd.DataFrame,
    *,
    horizons: Sequence[int] = HORIZONS,
    feature_columns: Sequence[str] | None = None,
    min_train_size: int = 60,
    ridge_alpha: float = 1.0,
) -> pd.DataFrame:
    """Fit on all fully observed labels and forecast after the latest close.

    The latest row is the known n+0 anchor. For each horizon, training rows
    must have target dates no later than that anchor, preventing use of labels
    that would not have been known at forecast time.
    """
    required = {"timestamp_utc", "close"}
    missing = required - set(supervised.columns)
    if missing:
        raise ValueError(f"supervised frame is missing required columns: {sorted(missing)}")
    if min_train_size < 2 or ridge_alpha <= 0:
        raise ValueError("min_train_size must be at least 2 and ridge_alpha must be positive")
    frame = supervised.sort_values("timestamp_utc").reset_index(drop=True).copy()
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True, errors="raise")
    if frame.empty or frame["timestamp_utc"].duplicated().any():
        raise ValueError("supervised frame must contain unique, non-empty daily rows")
    origin = frame.iloc[-1]
    columns = _feature_columns(frame, feature_columns)
    predictions: list[dict[str, object]] = []
    for horizon in horizons:
        if horizon < 1:
            raise ValueError("forecast horizons must be positive integers")
        target_col = f"target_log_return_h{horizon}"
        target_date_col = f"target_date_h{horizon}"
        if target_col not in frame or target_date_col not in frame:
            raise ValueError(f"supervised frame is missing labels for horizon {horizon}")
        target_dates = pd.to_datetime(frame[target_date_col], utc=True, errors="coerce")
        train = frame.loc[
            frame[target_col].notna() & target_dates.le(origin["timestamp_utc"])
        ].copy()
        if len(train) < min_train_size:
            raise ValueError(
                f"horizon {horizon}: only {len(train)} fully observed training rows; "
                f"need {min_train_size}"
            )
        x_train = train[columns].replace([np.inf, -np.inf], np.nan)
        x_origin = frame.iloc[[-1]][columns].replace([np.inf, -np.inf], np.nan)
        model = make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler(),
            Ridge(alpha=ridge_alpha),
        )
        model.fit(x_train, train[target_col].astype(float))
        predicted_return = float(model.predict(x_origin)[0])
        if not np.isfinite(predicted_return):
            raise RuntimeError("model produced a non-finite forecast")
        anchor_close = float(origin["close"])
        predictions.append({
            "origin_timestamp_utc": origin["timestamp_utc"],
            "target_timestamp_utc": origin["timestamp_utc"] + pd.Timedelta(days=int(horizon)),
            "horizon_days": int(horizon),
            "origin_close": anchor_close,
            "predicted_log_return": predicted_return,
            "predicted_close": anchor_close * float(np.exp(predicted_return)),
            "train_rows": len(train),
            "model_name": "ridge",
        })
    return pd.DataFrame(predictions)


def forecast_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Return price/return error metrics by horizon for Ridge and random walk."""
    required = {
        "horizon_days",
        "actual_log_return",
        "predicted_log_return",
        "origin_close",
        "actual_close",
        "predicted_close",
        "random_walk_close",
    }
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"predictions are missing required columns: {sorted(missing)}")
    if predictions.empty:
        return pd.DataFrame(
            columns=[
                "model_name", "horizon_days", "n_predictions", "mae_price",
                "rmse_price", "mae_log_return", "rmse_log_return",
                "directional_accuracy",
            ]
        )

    rows: list[dict[str, object]] = []
    for horizon, group in predictions.groupby("horizon_days", sort=True):
        for model_name, predicted_close_col, predicted_return_col in (
            ("ridge", "predicted_close", "predicted_log_return"),
            ("random_walk", "random_walk_close", None),
        ):
            actual_price = group["actual_close"].astype(float).to_numpy()
            predicted_price = group[predicted_close_col].astype(float).to_numpy()
            price_error = predicted_price - actual_price
            actual_return = group["actual_log_return"].astype(float).to_numpy()
            if predicted_return_col:
                predicted_return = group[predicted_return_col].astype(float).to_numpy()
                direction = float(
                    (np.sign(actual_return) == np.sign(predicted_return)).mean()
                )
                return_error = predicted_return - actual_return
            else:
                predicted_return = np.zeros_like(actual_return)
                direction = float("nan")  # the zero-return baseline makes no directional call
                return_error = predicted_return - actual_return
            rows.append(
                {
                    "model_name": model_name,
                    "horizon_days": int(horizon),
                    "n_predictions": len(group),
                    "mae_price": float(np.mean(np.abs(price_error))),
                    "rmse_price": float(np.sqrt(np.mean(price_error**2))),
                    "mae_log_return": float(np.mean(np.abs(return_error))),
                    "rmse_log_return": float(np.sqrt(np.mean(return_error**2))),
                    "directional_accuracy": direction,
                }
            )
    return pd.DataFrame(rows)


def one_day_long_flat_risk_metrics(
    predictions: pd.DataFrame,
    *,
    round_trip_cost_bps: float = 10.0,
    annualization: int = 365,
    return_threshold: float = 0.0,
) -> dict[str, float | int | None]:
    """Compute OOS h=1 long/flat risk metrics with explicit round-trip costs.

    A 1.0 position is held when the model's predicted close is above the next
    bar's observed open by more than the threshold; otherwise the portfolio is
    in cash. This uses the open as the first executable price after a completed
    day's signal. If open prices are absent, it falls back to a close-to-close
    proxy and marks that assumption in the result. Positions are capped at one,
    with no leverage or shorting. Round-trip cost is charged as half on each
    unit of entry/exit turnover. Undefined ratios return None, not infinity.
    Max drawdown is a negative fraction.
    """
    required = {
        "horizon_days",
        "origin_timestamp_utc",
        "actual_log_return",
        "predicted_log_return",
    }
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"predictions are missing required columns: {sorted(missing)}")
    if round_trip_cost_bps < 0 or annualization < 1:
        raise ValueError("cost cannot be negative and annualization must be positive")
    if not np.isfinite(return_threshold):
        raise ValueError("return_threshold must be finite")

    daily = predictions.loc[predictions["horizon_days"] == 1].copy()
    if daily.empty:
        return {
            "n_days": 0,
            "round_trip_cost_bps": float(round_trip_cost_bps),
            "annualization": int(annualization),
            "execution_assumption": "unavailable",
            "sharpe": None,
            "sortino": None,
            "max_drawdown": None,
            "total_return": None,
            "total_cost_fraction": 0.0,
            "active_days": 0,
            "round_trips": 0.0,
        }

    daily["origin_timestamp_utc"] = pd.to_datetime(
        daily["origin_timestamp_utc"], utc=True, errors="raise"
    )
    daily = daily.sort_values("origin_timestamp_utc").reset_index(drop=True)
    actual_log = pd.to_numeric(daily["actual_log_return"], errors="coerce").to_numpy()
    predicted_log = pd.to_numeric(daily["predicted_log_return"], errors="coerce").to_numpy()
    if not np.isfinite(actual_log).all() or not np.isfinite(predicted_log).all():
        raise ValueError("risk metrics require finite actual and predicted returns")

    if (
        "origin_close" in daily
        and "actual_close" in daily
        and "next_open" in daily
        and daily["next_open"].notna().all()
    ):
        origin_close = pd.to_numeric(daily["origin_close"], errors="coerce").to_numpy()
        next_open = pd.to_numeric(daily["next_open"], errors="coerce").to_numpy()
        actual_close = pd.to_numeric(daily["actual_close"], errors="coerce").to_numpy()
        if (
            not np.isfinite(origin_close).all()
            or not np.isfinite(next_open).all()
            or not np.isfinite(actual_close).all()
            or (origin_close <= 0).any()
            or (next_open <= 0).any()
            or (actual_close <= 0).any()
        ):
            raise ValueError("origin, execution-open, and target-close prices must be positive")
        predicted_close = origin_close * np.exp(predicted_log)
        expected_execution_return = predicted_close / next_open - 1.0
        asset_returns = actual_close / next_open - 1.0
        execution_assumption = "next_open_to_target_close"
    else:
        expected_execution_return = np.expm1(predicted_log)
        asset_returns = np.expm1(actual_log)
        execution_assumption = "close_to_close_proxy_open_unavailable"
    positions = (expected_execution_return > return_threshold).astype(float)
    prior = np.r_[0.0, positions[:-1]]
    turnover = np.abs(positions - prior)
    turnover[-1] += positions[-1]  # close any remaining position at the sample end
    per_side_cost = round_trip_cost_bps / 20_000.0
    costs = turnover * per_side_cost
    gross = positions * asset_returns
    net = np.maximum(gross - costs, -1.0)  # unlevered long-only equity cannot lose more than 100%
    if not np.isfinite(net).all():
        raise ValueError("net portfolio returns are not finite")

    mean_return = float(net.mean())
    volatility = float(net.std(ddof=1)) if len(net) > 1 else 0.0
    sharpe = (
        float(mean_return / volatility * np.sqrt(annualization))
        if volatility > 0
        else None
    )
    downside_deviation = float(np.sqrt(np.mean(np.minimum(net, 0.0) ** 2)))
    sortino = (
        float(mean_return / downside_deviation * np.sqrt(annualization))
        if downside_deviation > 0
        else None
    )
    equity = np.cumprod(1.0 + net)
    peaks = np.maximum.accumulate(np.r_[1.0, equity])[1:]
    drawdowns = equity / peaks - 1.0
    max_drawdown = float(drawdowns.min())
    return {
        "n_days": len(net),
        "round_trip_cost_bps": float(round_trip_cost_bps),
        "annualization": int(annualization),
        "execution_assumption": execution_assumption,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_drawdown,
        "total_return": float(equity[-1] - 1.0),
        "total_cost_fraction": float(costs.sum()),
        "active_days": int(positions.sum()),
        "round_trips": float(turnover.sum() / 2.0),
    }
