"""
Walk-forward backtest: compares the trained LSTM against two baselines
(naive "no change" and "historical drift") on held-out real data.

For each ticker, this:
  1. Reserves the last `--test-days` days as a held-out window.
  2. At each point in that window, generates a 7-day-ahead forecast using
     (a) the naive baseline, (b) the drift baseline, and (c) the trained LSTM.
  3. Compares each against what actually happened.
  4. Reports MAE / RMSE per forecast-day-ahead, plus directional accuracy
     (did the forecast correctly call up vs down over the horizon).

With --window, the held-out period is split into consecutive, non-overlapping
sub-windows (e.g. four 30-day chunks inside a 120-day test period) so you can
see whether the LSTM's edge (or deficit) vs baselines is a persistent pattern
or just a one-off lucky/unlucky stretch.

This answers the question every reviewer of a forecasting project will ask:
"Is this actually better than doing nothing, and is that consistent over time?"

Usage:
    python scripts/backtest.py --ticker NVDA
    python scripts/backtest.py --ticker NVDA --test-days 60
    python scripts/backtest.py --all                              # every ticker in config.TICKERS
    python scripts/backtest.py --ticker AMZN --test-days 120 --window 30   # rolling view
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app import config
from app.data.market_data import get_prepared_data
from app.models.lstm_model import LSTMForecaster
from app.models.baselines import compute_drift_log_returns, drift_price_forecast, naive_price_forecast


def _load_model(ticker: str):
    ckpt_path = config.MODELS_DIR / f"{ticker}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No trained model for {ticker}. Run `python -m app.models.train` first.")
    ckpt = torch.load(ckpt_path, weights_only=False)
    model = LSTMForecaster(
        n_features=len(ckpt["features"]),
        hidden_size=config.HIDDEN_SIZE,
        num_layers=config.NUM_LAYERS,
        horizon=ckpt["horizon"],
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def _lstm_forecast_at(model, ckpt, df, i: int) -> np.ndarray:
    """Point forecast (no MC sampling here — backtest wants a single deterministic
    estimate to compare fairly against the deterministic baselines) of prices for
    days i..i+horizon-1, using data up to (not including) day i.

    The model predicts a RESIDUAL over the drift baseline (see train.py /
    baselines.py), so we add the same drift component back here using the
    identical helper — this is what keeps the backtest honest: the LSTM is
    being scored on drift + residual, exactly what it would output at
    real inference time via predict.py.
    """
    features = ckpt["features"]
    seq_len = ckpt["seq_len"]
    horizon = ckpt["horizon"]
    mean = np.array(ckpt["mean"], dtype=np.float32)
    std = np.array(ckpt["std"], dtype=np.float32)

    window = df[features].values[i - seq_len:i].astype(np.float32)
    window_norm = (window - mean) / std
    x = torch.tensor(window_norm).unsqueeze(0)

    with torch.no_grad():
        residual_log_returns = model(x).squeeze(0).numpy()

    close_values = df["close"].values.astype(np.float32)
    target_type = ckpt.get("target_type", "raw_log_return")
    if target_type == "residual_over_drift":
        drift_lookback = ckpt.get("drift_lookback", config.DRIFT_LOOKBACK)
        drift_component = compute_drift_log_returns(close_values, i, horizon, drift_lookback)
    else:
        drift_component = np.zeros(horizon, dtype=np.float32)

    last_close = close_values[i - 1]
    return last_close * np.exp(residual_log_returns + drift_component)


def _naive_forecast_at(df, i: int, horizon: int) -> np.ndarray:
    return naive_price_forecast(df["close"].values.astype(np.float32), i, horizon)


def _drift_forecast_at(df, i: int, horizon: int, lookback: int = None) -> np.ndarray:
    lookback = lookback or config.DRIFT_LOOKBACK
    return drift_price_forecast(df["close"].values.astype(np.float32), i, horizon, lookback)


def _evaluate_points(model, ckpt, df, start: int, end: int, horizon: int) -> list[dict]:
    """Runs all three forecasters at every evaluation point in [start, end) and
    returns per-point raw results (not yet aggregated), so callers can slice
    this into whatever window granularity they want (whole-period or rolling)."""
    points = []
    for i in range(start, end):
        actual = df["close"].values[i:i + horizon]
        last_known = df["close"].values[i - 1]
        date = str(df["date"].values[i])[:10]

        preds = {
            "lstm": _lstm_forecast_at(model, ckpt, df, i),
            "naive": _naive_forecast_at(df, i, horizon),
            "drift": _drift_forecast_at(df, i, horizon),
        }

        point = {"date": date, "errors": {}, "direction_hit": {}}
        for name, pred in preds.items():
            point["errors"][name] = np.abs(pred - actual)  # per-horizon-day absolute error
            actual_direction = actual[-1] > last_known
            pred_direction = pred[-1] > last_known
            point["direction_hit"][name] = bool(actual_direction == pred_direction)
        points.append(point)
    return points


def _aggregate(points: list[dict]) -> dict:
    """Aggregates a list of per-point evaluations into MAE/RMSE/directional accuracy."""
    n = len(points)
    agg = {"n_evaluations": n}
    for name in ("lstm", "naive", "drift"):
        err_matrix = np.array([p["errors"][name] for p in points])  # (n, horizon)
        hits = sum(p["direction_hit"][name] for p in points)
        agg[name] = {
            "mae_per_horizon_day": [round(float(x), 3) for x in err_matrix.mean(axis=0)],
            "rmse_per_horizon_day": [round(float(x), 3) for x in np.sqrt((err_matrix ** 2).mean(axis=0))],
            "mae_overall": round(float(err_matrix.mean()), 3),
            "directional_accuracy_pct": round(100 * hits / n, 1) if n else 0.0,
        }
    return agg


def backtest_ticker(ticker: str, test_days: int = 40, window: int | None = None) -> dict:
    model, ckpt = _load_model(ticker)
    seq_len = ckpt["seq_len"]
    horizon = ckpt["horizon"]

    df = get_prepared_data(ticker, days=max(700, seq_len + test_days + horizon + 50))
    n = len(df)

    # Walk-forward window: need seq_len days of history before, and horizon days
    # of actual future after, each evaluation point `i`.
    start = max(seq_len, n - test_days - horizon)
    end = n - horizon

    points = _evaluate_points(model, ckpt, df, start, end, horizon)
    overall = _aggregate(points)
    overall["ticker"] = ticker
    overall["horizon_days"] = horizon

    result = {"ticker": ticker, "horizon_days": horizon, "overall": overall, "sub_windows": []}

    if window and window < len(points):
        for chunk_start in range(0, len(points), window):
            chunk = points[chunk_start:chunk_start + window]
            if len(chunk) < 3:  # skip tiny trailing remainder, not statistically meaningful
                continue
            agg = _aggregate(chunk)
            agg["date_range"] = f"{chunk[0]['date']} to {chunk[-1]['date']}"
            result["sub_windows"].append(agg)

    return result


def _print_metrics_table(agg: dict, indent: str = ""):
    print(f"{indent}{'Model':<10}{'MAE (avg $)':<14}{'Directional acc.':<18}")
    for name, label in [("naive", "Naive"), ("drift", "Drift"), ("lstm", "LSTM")]:
        r = agg[name]
        print(f"{indent}{label:<10}{r['mae_overall']:<14}{r['directional_accuracy_pct']}%")


def _verdict_line(agg: dict) -> str:
    lstm_mae = agg["lstm"]["mae_overall"]
    naive_mae = agg["naive"]["mae_overall"]
    drift_mae = agg["drift"]["mae_overall"]
    best_baseline = min(naive_mae, drift_mae)
    if lstm_mae < best_baseline:
        improvement = round(100 * (best_baseline - lstm_mae) / best_baseline, 1)
        return f"LSTM beats the best baseline by {improvement}% (lower MAE)."
    worse_by = round(100 * (lstm_mae - best_baseline) / best_baseline, 1)
    return f"LSTM is {worse_by}% WORSE than the best baseline on MAE."


def print_report(result: dict):
    ticker = result["ticker"]
    overall = result["overall"]
    print(f"\n{'='*60}")
    print(f"  Backtest: {ticker}  ({overall['n_evaluations']} evaluation points, "
          f"{result['horizon_days']}-day horizon)")
    print(f"{'='*60}")
    _print_metrics_table(overall)
    print(f"\n  -> {_verdict_line(overall)}")
    if not result["sub_windows"]:
        if overall["lstm"]["mae_overall"] >= min(overall["naive"]["mae_overall"], overall["drift"]["mae_overall"]):
            print("     This is common on short/noisy horizons — consider more training data,")
            print("     more epochs, feature selection, or reporting directional accuracy")
            print("     as the primary metric instead of raw price error.")

    if result["sub_windows"]:
        print(f"\n  --- Rolling view ({len(result['sub_windows'])} sub-windows) ---")
        wins, losses = 0, 0
        for idx, sw in enumerate(result["sub_windows"], 1):
            verdict = _verdict_line(sw)
            beats = "beats" in verdict
            wins += beats
            losses += not beats
            print(f"\n  Window {idx}  ({sw['date_range']}, n={sw['n_evaluations']})")
            _print_metrics_table(sw, indent="    ")
            print(f"    -> {verdict}")

        print(f"\n  Consistency: LSTM beat the best baseline in {wins}/{wins + losses} sub-windows.")
        if wins == 0:
            print("  -> Consistently underperforming across time, not just a one-off bad stretch.")
        elif losses == 0:
            print("  -> Consistently outperforming across time — a real, stable edge.")
        else:
            print("  -> Mixed across time — the model's edge (or deficit) is regime-dependent,")
            print("     not a stable property. Worth investigating what differs between the")
            print("     winning and losing windows (volatility regime, trend vs chop, etc).")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--test-days", type=int, default=40)
    parser.add_argument("--window", type=int, default=None,
                         help="Split the held-out period into consecutive sub-windows of this "
                              "many evaluation points each, to check consistency over time.")
    args = parser.parse_args()

    tickers = config.TICKERS if args.all else [args.ticker.upper()] if args.ticker else ["NVDA"]

    all_results = []
    for ticker in tickers:
        try:
            result = backtest_ticker(ticker, test_days=args.test_days, window=args.window)
            print_report(result)
            all_results.append(result)
        except FileNotFoundError as e:
            print(f"[{ticker}] skipped: {e}")

    if len(all_results) > 1:
        print(f"\n{'='*60}\n  Summary across {len(all_results)} tickers\n{'='*60}")
        for r in all_results:
            overall = r["overall"]
            lstm_mae = overall["lstm"]["mae_overall"]
            naive_mae = overall["naive"]["mae_overall"]
            drift_mae = overall["drift"]["mae_overall"]
            beats = "beats" if lstm_mae < min(naive_mae, drift_mae) else "loses to"
            print(f"  {r['ticker']:<8} LSTM {beats} best baseline "
                  f"(LSTM={lstm_mae}, naive={naive_mae}, drift={drift_mae})")


if __name__ == "__main__":
    main()