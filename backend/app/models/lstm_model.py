"""Multivariate LSTM forecaster: sequence of [close, volume, sma_10, sma_30, rsi_14, macd]
-> 7-day ahead close price forecast."""
from __future__ import annotations
import torch
import torch.nn as nn


class LSTMForecaster(nn.Module):
    def __init__(self, n_features: int, hidden_size: int = 64, num_layers: int = 2,
                 horizon: int = 7, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features)
        out, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]              # (batch, hidden_size) — final layer's hidden state
        last_hidden = self.norm(last_hidden)
        return self.head(last_hidden)      # (batch, horizon) — predicted returns for next `horizon` days
