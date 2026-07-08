"""
Train one LSTM per ticker on real (or synthetic) OHLCV+TA data.

Target: RESIDUAL over the drift baseline, cumulative over the next
FORECAST_HORIZON days (see baselines.py for why — raw returns are dominated
by whatever the drift baseline already explains for free).

Run: python -m app.models.train
"""
from __future__ import annotations
import json
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from app import config
from app.data.market_data import get_prepared_data
from app.models.lstm_model import LSTMForecaster
from app.models.baselines import compute_drift_log_returns


def build_sequences(df, seq_len: int, horizon: int, features: list[str],
                     drift_lookback: int = config.DRIFT_LOOKBACK):
    """
    Sliding-window sequences -> (X, y) where y is the RESIDUAL over the drift
    baseline: y = actual_cumulative_log_return - drift_cumulative_log_return.

    Why: raw price/return targets are dominated by whatever the recent trend
    already tells you (the drift baseline captures that for free). Having the
    LSTM predict the residual forces it to focus its capacity on the part
    drift *doesn't* explain, and makes it directly comparable to the drift
    baseline at inference time (see baselines.py + predict.py).
    """
    values = df[features].values.astype(np.float32)
    close = df["close"].values.astype(np.float32)
    log_close = np.log(close)

    X, y = [], []
    n = len(df)
    for i in range(seq_len, n - horizon):
        X.append(values[i - seq_len:i])
        # cumulative log-return path over the next `horizon` days, relative to day i-1
        future_log_returns = log_close[i:i + horizon] - log_close[i - 1]
        drift_log_returns = compute_drift_log_returns(close, i, horizon, drift_lookback)
        y.append(future_log_returns - drift_log_returns)
    return np.array(X), np.array(y)


def normalize(X: np.ndarray):
    """Per-feature z-score normalization, fit on training data, returned for reuse at inference."""
    mean = X.reshape(-1, X.shape[-1]).mean(axis=0)
    std = X.reshape(-1, X.shape[-1]).std(axis=0) + 1e-8
    X_norm = (X - mean) / std
    return X_norm, mean, std


def train_one_ticker(ticker: str, verbose: bool = True):
    df = get_prepared_data(ticker, days=2500 + config.DRIFT_LOOKBACK)
    X, y = build_sequences(df, config.SEQ_LEN, config.FORECAST_HORIZON, config.FEATURES)
    X_norm, mean, std = normalize(X)

    split = int(len(X_norm) * 0.85)
    X_train, X_val = X_norm[:split], X_norm[split:]
    y_train, y_val = y[:split], y[split:]

    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_ds = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE)

    torch.manual_seed(42)
    np.random.seed(42)

    model = LSTMForecaster(
        n_features=len(config.FEATURES),
        hidden_size=config.HIDDEN_SIZE,
        num_layers=config.NUM_LAYERS,
        horizon=config.FORECAST_HORIZON,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    loss_fn = torch.nn.MSELoss()

    best_val = float("inf")
    best_state = None

    for epoch in range(config.EPOCHS):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                pred = model(xb)
                val_loss += loss_fn(pred, yb).item() * len(xb)
        val_loss /= max(len(val_ds), 1)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if verbose and (epoch % 10 == 0 or epoch == config.EPOCHS - 1):
            print(f"[{ticker}] epoch {epoch:3d}  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}")

    model.load_state_dict(best_state)

    # Save model + normalization stats + target-type metadata for inference
    ckpt_path = config.MODELS_DIR / f"{ticker}.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "features": config.FEATURES,
        "seq_len": config.SEQ_LEN,
        "horizon": config.FORECAST_HORIZON,
        "drift_lookback": config.DRIFT_LOOKBACK,
        "target_type": "residual_over_drift",
        "best_val_loss": best_val,
    }, ckpt_path)
    print(f"[{ticker}] saved checkpoint -> {ckpt_path}  (best_val_loss={best_val:.6f})")
    return best_val


def train_all():
    results = {}
    for ticker in config.TICKERS:
        results[ticker] = train_one_ticker(ticker)
    summary_path = config.MODELS_DIR / "training_summary.json"
    summary_path.write_text(json.dumps(results, indent=2))
    print("\nTraining complete. Summary:")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    train_all()