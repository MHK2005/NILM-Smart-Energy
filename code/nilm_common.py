"""Shared helpers: window gathering and the two ANN models (MLP and 1D-CNN)."""
import json
import os
import numpy as np
import torch
import torch.nn as nn

WINDOW = int(os.environ.get("NILM_WINDOW", 60))
N_APP = 4


def load_prep(prep="prep_austin.npz", meta="prep_meta.json"):
    z = dict(np.load(prep))
    m = json.load(open(meta))
    return z, m


def gather(z, homes, splits, stride=1):
    """Windows of the previous WINDOW minutes of total power; targets are appliance power at the last minute."""
    xs, ys, hs, ts = [], [], [], []
    offs = np.arange(-WINDOW + 1, 1)
    for h in homes:
        idx = np.sort(np.concatenate([z[f"idx_{h}_{s}"] for s in splits]))[::stride]
        xs.append(z[f"X_{h}"][idx[:, None] + offs])
        ys.append(z[f"Y_{h}"][idx])
        hs.append(np.full(len(idx), h, dtype=np.int32))
        ts.append(idx)
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(hs), np.concatenate(ts)


class MLP(nn.Module):
    """Basic MLP: 60 whole-house readings -> ON/OFF logits and scaled power for each appliance."""

    def __init__(self, hidden=(256, 128, 64), drop=0.2):
        super().__init__()
        layers, d = [], WINDOW
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(drop)]
            d = h
        self.body = nn.Sequential(*layers)
        self.cls = nn.Linear(d, N_APP)
        self.reg = nn.Linear(d, N_APP)

    def forward(self, x):                       # x: (B, WINDOW)
        h = self.body(x)
        return self.cls(h), self.reg(h)


class CNN1D(nn.Module):
    """Advanced model: 1D-CNN over the 60-minute window."""

    def __init__(self, ch=(16, 32, 32), drop=0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, ch[0], 5, padding=2), nn.ReLU(),
            nn.Conv1d(ch[0], ch[1], 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(ch[1], ch[2], 3, padding=1), nn.ReLU(), nn.MaxPool1d(2))
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(ch[2] * (WINDOW // 4), 128), nn.ReLU(), nn.Dropout(drop))
        self.cls = nn.Linear(128, N_APP)
        self.reg = nn.Linear(128, N_APP)

    def forward(self, x):                       # x: (B, WINDOW)
        h = self.fc(self.conv(x.unsqueeze(1)))
        return self.cls(h), self.reg(h)


MODELS = {"mlp": MLP, "cnn1d": CNN1D}
