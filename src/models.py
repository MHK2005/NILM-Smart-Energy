"""
The six ANN architectures (PyTorch).

Every model:
  * takes x of shape (batch, window) = scaled whole-house power only;
  * returns (logits, power):
        logits : (batch, n_appliances)  -> ON/OFF via sigmoid
        power  : (batch, n_appliances)  -> appliance power (scaled units),
                 kept non-negative with softplus.
Each model has a "feature extractor" (different per architecture) and the
same two output heads, built by the shared NILMModel base class.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class NILMModel(nn.Module):
    """Base class: subclasses build self.features and set self.feat_dim."""

    def __init__(self, n_app):
        super().__init__()
        self.n_app = n_app

    def build_heads(self, feat_dim):
        self.cls_head = nn.Linear(feat_dim, self.n_app)   # classification head
        self.reg_head = nn.Linear(feat_dim, self.n_app)   # power regression head

    def extract(self, x):
        raise NotImplementedError

    def forward(self, x):
        h = self.extract(x)
        return self.cls_head(h), F.softplus(self.reg_head(h))


# ---------------------------------------------------------------- Model 1
class MLP(NILMModel):
    """Baseline: 180 inputs -> 256 -> 128 -> 64 (ReLU) -> dropout -> heads."""

    def __init__(self, n_app, window):
        super().__init__(n_app)
        self.net = nn.Sequential(
            nn.Linear(window, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Dropout(0.2))
        self.build_heads(64)

    def extract(self, x):
        return self.net(x)


# ---------------------------------------------------------------- Model 2
class CNN1D(NILMModel):
    """Conv(16) -> Conv(32) -> MaxPool -> Conv(32) -> MaxPool -> Flatten -> Dense(128)."""

    def __init__(self, n_app, window):
        super().__init__(n_app)
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, 5, padding=2), nn.ReLU(),
            nn.Conv1d(16, 32, 5, padding=2), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool1d(2))
        flat = 32 * (window // 2 // 2)
        self.dense = nn.Sequential(nn.Flatten(), nn.Linear(flat, 128), nn.ReLU(),
                                   nn.Dropout(0.2))
        self.build_heads(128)

    def extract(self, x):
        return self.dense(self.conv(x.unsqueeze(1)))


# ---------------------------------------------------------------- Models 3 and 4
class RecurrentNet(NILMModel):
    """LSTM or GRU with 64 units -> dropout -> Dense(64). Uses the last time step."""

    def __init__(self, n_app, window, cell="lstm"):
        super().__init__(n_app)
        rnn = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnn = rnn(input_size=1, hidden_size=64, batch_first=True)
        self.drop = nn.Dropout(0.2)
        self.dense = nn.Sequential(nn.Linear(64, 64), nn.ReLU())
        self.build_heads(64)

    def extract(self, x):
        out, _ = self.rnn(x.unsqueeze(-1))       # (B, T, 64)
        return self.dense(self.drop(out[:, -1]))


# ---------------------------------------------------------------- Model 5
class CNNLSTM(NILMModel):
    """Conv(16,k5) -> MaxPool -> Conv(32,k3) -> LSTM(32) -> dropout -> Dense(64)."""

    def __init__(self, n_app, window):
        super().__init__(n_app)
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, 5, padding=2), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 3, padding=1), nn.ReLU())
        self.lstm = nn.LSTM(32, 32, batch_first=True)
        self.drop = nn.Dropout(0.2)
        self.dense = nn.Sequential(nn.Linear(32, 64), nn.ReLU())
        self.build_heads(64)

    def extract(self, x):
        c = self.conv(x.unsqueeze(1)).transpose(1, 2)   # (B, T/2, 32)
        out, _ = self.lstm(c)
        return self.dense(self.drop(out[:, -1]))


# ---------------------------------------------------------------- Model 6
class CausalConv(nn.Module):
    """Conv1d that only looks at the current and past time steps."""

    def __init__(self, cin, cout, k, dilation):
        super().__init__()
        self.pad = (k - 1) * dilation
        self.conv = nn.Conv1d(cin, cout, k, dilation=dilation)

    def forward(self, x):
        return self.conv(F.pad(x, (self.pad, 0)))


class TCN(NILMModel):
    """Two causal conv layers (16 and 32 filters, dilation 1 and 2), ReLU, dropout,
    then global average pooling joined with the final time step -> Dense(64)."""

    def __init__(self, n_app, window, k=5):
        super().__init__(n_app)
        self.block = nn.Sequential(
            CausalConv(1, 16, k, 1), nn.ReLU(), nn.Dropout(0.2),
            CausalConv(16, 32, k, 2), nn.ReLU(), nn.Dropout(0.2))
        self.dense = nn.Sequential(nn.Linear(64, 64), nn.ReLU())
        self.build_heads(64)

    def extract(self, x):
        h = self.block(x.unsqueeze(1))                       # (B, 32, T)
        feat = torch.cat([h.mean(dim=2), h[:, :, -1]], dim=1)  # (B, 64)
        return self.dense(feat)


# ---------------------------------------------------------------- interface
def build_model(name, n_app, window):
    """Common interface used by train.py, evaluate.py and app.py."""
    if name == "mlp":
        return MLP(n_app, window)
    if name == "cnn":
        return CNN1D(n_app, window)
    if name == "lstm":
        return RecurrentNet(n_app, window, "lstm")
    if name == "gru":
        return RecurrentNet(n_app, window, "gru")
    if name == "cnn_lstm":
        return CNNLSTM(n_app, window)
    if name == "tcn":
        return TCN(n_app, window)
    raise ValueError(f"unknown model '{name}'")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
