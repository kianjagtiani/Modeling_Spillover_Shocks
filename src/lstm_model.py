"""
HAR-LSTM Hybrid model.

Architecture:
  Stage 1: HAR-RV-SJ (OLS) — captures linear long-memory structure
  Stage 2: LSTM on HAR residuals — learns nonlinear temporal dependencies
            that a single-step model (XGBoost/LightGBM) misses

The LSTM sees a rolling window of the last `seq_len` days of features,
giving it true sequence awareness rather than just a snapshot of lagged scalars.

Design choices:
  - seq_len = 22  (matches HAR monthly window)
  - 2-layer LSTM with dropout
  - LayerNorm for training stability on financial data
  - StandardScaler per feature (LSTM is scale-sensitive)
  - Early stopping on validation MSE
  - Re-uses the HAR instance from Stage 1 so residuals are consistent
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# ── Dataset ───────────────────────────────────────────────────────────────────

class ResidualSequenceDataset(Dataset):
    """
    Each sample = (X_seq, y_scalar) where:
      X_seq : (seq_len, n_features)  — rolling window of scaled features
      y_scalar : float               — HAR residual to predict
    """
    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.y) - self.seq_len

    def __getitem__(self, idx):
        x_seq = self.X[idx : idx + self.seq_len]          # (seq_len, features)
        y_val = self.y[idx + self.seq_len]                 # scalar residual
        return x_seq, y_val


# ── LSTM Architecture ─────────────────────────────────────────────────────────

class LSTMResidualNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):                     # x: (batch, seq_len, features)
        out, _ = self.lstm(x)                 # (batch, seq_len, hidden)
        last = self.norm(out[:, -1, :])       # take last timestep
        return self.head(last).squeeze(-1)    # (batch,)


# ── HAR-LSTM Hybrid ───────────────────────────────────────────────────────────

class HARLSTMHybrid:
    """
    Two-stage hybrid:
      Stage 1 → HAR-RV-SJ (OLS)
      Stage 2 → LSTM trained on HAR residuals using a seq_len-day rolling window
    """
    name = "HAR-LSTM Hybrid"

    def __init__(
        self,
        seq_len: int = 22,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        lr: float = 1e-3,
        max_epochs: int = 80,
        batch_size: int = 32,
        patience: int = 10,
    ):
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.patience = patience

        self.har_ = None
        self.net_ = None
        self.scaler_ = StandardScaler()
        self.feature_names_ = None

    def _make_sequences(self, X_scaled: np.ndarray, y: np.ndarray):
        return ResidualSequenceDataset(X_scaled, y, self.seq_len)

    def fit(self, X: pd.DataFrame, y: pd.Series,
            X_val: pd.DataFrame | None = None,
            y_val: pd.Series | None = None) -> "HARLSTMHybrid":
        from models import HARModel

        self.feature_names_ = X.columns.tolist()
        n_features = len(self.feature_names_)

        # Stage 1: HAR
        self.har_ = HARModel()
        self.har_.fit(X, y)
        har_pred = self.har_.predict(X)
        residuals = y.values - har_pred

        # Scale features
        X_filled = X.fillna(0).values
        X_scaled = self.scaler_.fit_transform(X_filled)

        # Validation residuals
        if X_val is not None and y_val is not None:
            har_val_pred = self.har_.predict(X_val)
            val_res = y_val.values - har_val_pred
            X_val_scaled = self.scaler_.transform(X_val.fillna(0).values)
        else:
            # Use last 15% of train as validation
            cut = int(len(X_scaled) * 0.85)
            X_val_scaled = X_scaled[cut:]
            val_res = residuals[cut:]
            X_scaled = X_scaled[:cut]
            residuals = residuals[:cut]

        # Build datasets
        train_ds = self._make_sequences(X_scaled, residuals)
        val_ds   = self._make_sequences(X_val_scaled, val_res)

        if len(train_ds) < self.batch_size:
            # Not enough data to train LSTM meaningfully — fall back to zero residual
            self.net_ = None
            return self

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=False)
        val_loader   = DataLoader(val_ds,   batch_size=self.batch_size, shuffle=False)

        # Build network
        self.net_ = LSTMResidualNet(
            input_size=n_features,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(DEVICE)

        optimizer = torch.optim.Adam(self.net_.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5
        )
        criterion = nn.MSELoss()

        best_val = float("inf")
        wait = 0
        best_state = None

        for epoch in range(self.max_epochs):
            # Train
            self.net_.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                optimizer.zero_grad()
                pred = self.net_(xb)
                loss = criterion(pred, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net_.parameters(), 1.0)
                optimizer.step()

            # Validate
            self.net_.eval()
            val_losses = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    val_losses.append(criterion(self.net_(xb), yb).item())
            val_loss = np.mean(val_losses)
            scheduler.step(val_loss)

            if val_loss < best_val - 1e-6:
                best_val = val_loss
                best_state = {k: v.clone() for k, v in self.net_.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= self.patience:
                    break

        if best_state is not None:
            self.net_.load_state_dict(best_state)

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        har_pred = self.har_.predict(X)

        if self.net_ is None or len(X) < self.seq_len:
            return har_pred

        X_scaled = self.scaler_.transform(X.fillna(0).values)
        n = len(X_scaled)

        # Build sequences for every prediction point
        sequences = np.stack([
            X_scaled[i : i + self.seq_len]
            for i in range(n - self.seq_len)
        ])  # (n - seq_len, seq_len, features)

        self.net_.eval()
        with torch.no_grad():
            xb = torch.tensor(sequences, dtype=torch.float32).to(DEVICE)
            lstm_res = self.net_(xb).cpu().numpy()

        # Pad the first seq_len predictions with 0 (no LSTM correction available)
        lstm_correction = np.concatenate([np.zeros(self.seq_len), lstm_res])
        return har_pred + lstm_correction
