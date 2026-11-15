"""
Shared settings and small helper functions for the NILM Smart Energy project.

Every script imports its settings from here, so a change made here
(for example the window length) applies to the whole project.
"""
import json
import os
import random
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
CHECKPOINT_DIR = MODELS_DIR / "checkpoints"
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
LOG_DIR = RESULTS_DIR / "training_logs"
PRED_DIR = RESULTS_DIR / "predictions"

# ---------------------------------------------------------------------------
# Dataset selection (chosen after inspecting the Austin file, see README)
# ---------------------------------------------------------------------------
REGION = "Austin, Texas (Pecan Street 1-minute, 2018)"

# Development homes: used for chronological train / validation / test.
# These are the only 5 Austin homes where air1, drye1 and car1 all have
# real ON periods.
DEV_HOMES = [661, 4373, 4767, 6139, 8156]

# Unseen homes: never used for training, validation, thresholds or scaling.
# They are evaluated as whole homes for the extra unseen-home test.
UNSEEN_HOMES = [1642, 7719]

APPLIANCES = ["air1", "drye1", "car1", "refrigerator1"]
APPLIANCE_NAMES = {
    "air1": "Air Conditioner",
    "drye1": "Electric Dryer",
    "car1": "EV Charging",
    "refrigerator1": "Refrigerator",
}
# The 3 appliances used for the compulsory CCP benchmark.
BENCHMARK_APPLIANCES = ["air1", "drye1", "car1"]

# Columns that make up whole-house consumption: grid is net power from the
# utility (negative when solar exports), so use = grid + solar + solar2.
GRID_COL = "grid"
SOLAR_COLS = ["solar", "solar2"]

# ---------------------------------------------------------------------------
# Preprocessing settings
# ---------------------------------------------------------------------------
WINDOW = 180                 # input window length in minutes (change here only)
MAX_INTERP_GAP = 5           # interpolate missing runs of at most 5 minutes
NEG_NOISE_KW = 0.05          # appliance readings in [-0.05, 0) kW are sensor noise -> 0
OVER_AGG_MARGIN_KW = 0.10    # appliance > aggregate + 0.1 kW is impossible -> invalid
THRESH_MIN_KW = 0.02         # threshold = max(0.02 kW, 10% of training p99)
THRESH_FRACTION = 0.10

# Split: "monthly" (default) or "global".
#  global : the whole year per home, first 70% train, next 15% val, last 15% test.
#  monthly: inside every calendar month, first 70% train, next 15% val,
#           last 15% test. Still in time order, but every season appears in
#           every split. This is needed because a global split puts the test
#           set in Nov-Dec, when the air conditioner is ON only 0.6% of the time.
SPLIT_MODE = "monthly"
SPLIT_RATIOS = (0.70, 0.15, 0.15)
LOCAL_TZ = "America/Chicago"

# ---------------------------------------------------------------------------
# Training settings (identical for all six models)
# ---------------------------------------------------------------------------
SEED = 42
BATCH_SIZE = 512
LEARNING_RATE = 1e-3
MAX_EPOCHS = 30
PATIENCE = 5                  # early stopping patience (epochs)
MAX_TRAIN_WINDOWS = 200_000   # random subsample of training windows (CPU budget)
MAX_VAL_WINDOWS = 60_000      # random subsample of validation windows for early stopping
MODEL_NAMES = ["mlp", "cnn", "lstm", "gru", "cnn_lstm", "tcn"]
MODEL_LABELS = {
    "mlp": "MLP", "cnn": "1D-CNN", "lstm": "LSTM",
    "gru": "GRU", "cnn_lstm": "CNN-LSTM", "tcn": "TCN",
}

# CCP benchmark targets (classification only)
MIN_TARGET = {"accuracy": 0.85, "balanced_accuracy": 0.75, "macro_f1": 0.70}
HIGH_TARGET = {"accuracy": 0.90, "balanced_accuracy": 0.85, "macro_f1": 0.80}


def set_seed(seed=SEED):
    """Fix random seeds for Python, NumPy and (if installed) PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(False)
    except ImportError:
        pass


def ensure_dirs():
    for d in [PROCESSED_DIR, MODELS_DIR, CHECKPOINT_DIR, RESULTS_DIR,
              FIG_DIR, LOG_DIR, PRED_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def find_raw_csv():
    """Find the 1-minute CSV inside data/raw (any sub-folder)."""
    files = []
    for dirpath, _, names in os.walk(RAW_DIR, followlinks=True):
        files += [Path(dirpath) / n for n in names
                  if n.lower().endswith(".csv") and "metadata" not in n.lower()]
    files = sorted(files)
    if not files:
        raise FileNotFoundError(
            f"No dataset CSV found under {RAW_DIR}. Extract "
            "1minute_data_austin.tar.gz into data/raw/ first (see README).")
    return files
