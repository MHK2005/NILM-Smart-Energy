"""
Loads the processed homes and builds model batches.

A sample is one 180-minute window of whole-house power (the ONLY input)
and, at the last minute of the window, the targets for every appliance:
ON/OFF label, power in kW, and a mask that says whether that target is
available (an appliance that is not metered or has an invalid reading is
masked out, never treated as zero).

This file does not need PyTorch, so the website and the sanity check can
reuse it. Batches are plain NumPy arrays; train.py turns them into tensors.
"""
import numpy as np

from src.utils import PROCESSED_DIR, load_json


class NILMData:
    def __init__(self, processed_dir=PROCESSED_DIR):
        self.prep = load_json(processed_dir / "preprocessing.json")
        self.window = self.prep["window"]
        self.appliances = self.prep["appliances"]
        self.homes = self.prep["homes_order"]
        self.mean = self.prep["input_scaler"]["mean"]
        self.std = self.prep["input_scaler"]["std"]
        self.power_scale = np.array([self.prep["power_scale_kW"][a] for a in self.appliances],
                                    dtype=np.float32)
        self.h = []
        for home in self.homes:
            z = np.load(processed_dir / f"home_{home}.npz")
            self.h.append({k: z[k] for k in z.files})
        w = np.load(processed_dir / "windows.npz")
        self.windows = {k: w[k] for k in w.files}

    # ------------------------------------------------------------------
    def windows_for(self, split, max_n=None, seed=42):
        """Window index pairs (home_index, end_minute) for a split.
        With max_n, a fixed random subsample is returned (same seed -> same windows)."""
        idx = self.windows[split]
        if max_n is not None and len(idx) > max_n:
            rng = np.random.default_rng(seed)
            idx = idx[np.sort(rng.choice(len(idx), max_n, replace=False))]
        return idx

    def raw_inputs(self, idx):
        """Aggregate kW windows (unscaled), shape (B, window)."""
        X = np.empty((len(idx), self.window), dtype=np.float32)
        offs = np.arange(-self.window + 1, 1)
        for hi in np.unique(idx[:, 0]):
            sel = np.where(idx[:, 0] == hi)[0]
            X[sel] = self.h[hi]["aggregate"][idx[sel, 1][:, None] + offs]
        return X

    def scale_inputs(self, X_kw):
        return ((X_kw - self.mean) / self.std).astype(np.float32)

    def batch(self, idx):
        """Return X (scaled), y_on, y_power_scaled, mask, y_power_kW."""
        X = self.scale_inputs(self.raw_inputs(idx))
        A = len(self.appliances)
        y_on = np.zeros((len(idx), A), np.float32)
        y_kw = np.zeros((len(idx), A), np.float32)
        mask = np.zeros((len(idx), A), np.float32)
        for hi in np.unique(idx[:, 0]):
            sel = np.where(idx[:, 0] == hi)[0]
            e = idx[sel, 1]
            H = self.h[hi]
            mask[sel] = H["valid"][e]
            y_on[sel] = H["on"][e]
            y_kw[sel] = np.nan_to_num(H["power"][e])
        return X, y_on, y_kw / self.power_scale, mask, y_kw

    def iterate(self, idx, batch_size, shuffle=False, seed=42):
        order = np.arange(len(idx))
        if shuffle:
            np.random.default_rng(seed).shuffle(order)
        for s in range(0, len(order), batch_size):
            yield self.batch(idx[order[s:s + batch_size]])

    def timestamps(self, hi, ends):
        return self.h[hi]["ts"][ends]
