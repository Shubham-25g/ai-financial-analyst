"""7-day inference from a trained checkpoint, with a simple Monte-Carlo-dropout
confidence band (cheap uncertainty estimate without training an ensemble)."""
from __future__ import annotations
import numpy as np
import torch

from app import config
from app.data.market_data import get_prepared_data
from app.models.lstm_model import LSTMForecaster
from app.models.baselines import compute_drift_log_returns


def _load_checkpoint(ticker: str):
    ckpt_path = config.MODELS_DIR / f"{ticker}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"No trained model for {ticker}. Run `python -m app.models.train` first."
        )
    return torch.load(ckpt_path, weights_only=False)


def forecast(ticker: str, mc_samples: int = 30) -> dict:
    ckpt = _load_checkpoint(ticker)
    features = ckpt["features"]
    seq_len = ckpt["seq_len"]
    horizon = ckpt["horizon"]
    mean = np.array(ckpt["mean"], dtype=np.float32)
    std = np.array(ckpt["std"], dtype=np.float32)

    model = LSTMForecaster(
        n_features=len(features),
        hidden_size=config.HIDDEN_SIZE,
        num_layers=config.NUM_LAYERS,
        horizon=horizon,
    )
    model.load_state_dict(ckpt["state_dict"])

    df = get_prepared_data(ticker, days=max(seq_len + config.DRIFT_LOOKBACK + 30, 200))
    last_window = df[features].values[-seq_len:].astype(np.float32)
    last_window_norm = (last_window - mean) / std
    x = torch.tensor(last_window_norm).unsqueeze(0)  # (1, seq_len, n_features)

    last_close = float(df["close"].values[-1])
    last_date = df["date"].values[-1]

    # The model was trained to predict a RESIDUAL over the drift baseline
    # (see train.py / baselines.py). Add the drift component back to get the
    # actual predicted price path. Old checkpoints without this metadata
    # (target_type == "raw_log_return") skip this step for compatibility.
    target_type = ckpt.get("target_type", "raw_log_return")
    if target_type == "residual_over_drift":
        drift_lookback = ckpt.get("drift_lookback", config.DRIFT_LOOKBACK)
        close_values = df["close"].values.astype(np.float32)
        drift_component = compute_drift_log_returns(close_values, len(df), horizon, drift_lookback)
    else:
        drift_component = np.zeros(horizon, dtype=np.float32)

    # Monte-Carlo dropout: keep dropout active at inference to sample an
    # approximate predictive distribution instead of a single point estimate.
    model.train()  # enables dropout layers
    samples = []
    with torch.no_grad():
        for _ in range(mc_samples):
            residual_log_returns = model(x).squeeze(0).numpy()  # horizon steps
            total_log_returns = residual_log_returns + drift_component
            prices = last_close * np.exp(total_log_returns)
            samples.append(prices)
    samples = np.array(samples)  # (mc_samples, horizon)

    p50 = np.median(samples, axis=0)
    p10 = np.percentile(samples, 10, axis=0)
    p90 = np.percentile(samples, 90, axis=0)

    future_dates = np.busday_offset(
        np.datetime64(last_date, "D"), np.arange(1, horizon + 1), roll="forward"
    )

    return {
        "ticker": ticker,
        "last_close": round(last_close, 2),
        "last_date": str(np.datetime_as_string(np.datetime64(last_date, "D"))),
        "horizon_days": horizon,
        "forecast": [
            {
                "date": str(d),
                "median": round(float(m), 2),
                "low_p10": round(float(lo), 2),
                "high_p90": round(float(hi), 2),
            }
            for d, m, lo, hi in zip(future_dates, p50, p10, p90)
        ],
        "expected_return_pct": round(float(p50[-1] / last_close - 1) * 100, 2),
    }


if __name__ == "__main__":
    import json
    print(json.dumps(forecast("NVDA"), indent=2))