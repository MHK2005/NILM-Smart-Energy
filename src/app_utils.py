"""
Helper functions for the Streamlit website (app.py).

Kept separate from app.py so they can be tested without Streamlit.
Only whole-house aggregate power is ever passed to a model.
"""
import numpy as np
import pandas as pd

from src.dataset import NILMData
from src.utils import (MODELS_DIR, RESULTS_DIR, LOG_DIR, MODEL_NAMES, MODEL_LABELS,
                       APPLIANCE_NAMES, LOCAL_TZ, load_json)

SPLIT_NAMES = {0: "training", 1: "validation", 2: "test", 3: "unseen home"}


def model_status():
    """Which of the six models have a saved checkpoint."""
    rows = []
    for n in MODEL_NAMES:
        p = MODELS_DIR / f"{n}.pth"
        info = {"name": n, "label": MODEL_LABELS[n], "exists": p.exists(),
                "finished": False, "best_epoch": None}
        if p.exists():
            try:
                import torch
                ck = torch.load(p, map_location="cpu", weights_only=False)
                info["finished"] = bool(ck.get("finished"))
                info["best_epoch"] = ck.get("best_epoch")
            except ImportError:
                info["finished"] = None   # file exists but PyTorch is not installed
        rows.append(info)
    return pd.DataFrame(rows)


def read_csv_if(path):
    return pd.read_csv(path) if path.exists() else None


def load_results():
    return {
        "comparison": read_csv_if(RESULTS_DIR / "model_comparison.csv"),
        "results": read_csv_if(RESULTS_DIR / "results.csv"),
        "best": read_csv_if(RESULTS_DIR / "best_models.csv"),
        "unseen": read_csv_if(RESULTS_DIR / "unseen_home_results.csv"),
        "class_counts": read_csv_if(RESULTS_DIR / "class_counts.csv"),
        "bench": read_csv_if(RESULTS_DIR / "benchmark_summary.csv"),
        "prob_thr": load_json(RESULTS_DIR / "prob_thresholds.json")
        if (RESULTS_DIR / "prob_thresholds.json").exists() else {},
        "logs": {n: pd.read_csv(LOG_DIR / f"{n}_log.csv") for n in MODEL_NAMES
                 if (LOG_DIR / f"{n}_log.csv").exists()},
    }


def home_index(data: NILMData, home):
    return data.homes.index(int(home))


def local_times(data, hi, a=None, b=None):
    ts = data.h[hi]["ts"][a:b]
    return pd.to_datetime(ts, unit="s", utc=True).tz_convert(LOCAL_TZ)


def minute_at(data, hi, when_local):
    """Index of the minute closest to a local timestamp."""
    t = pd.Timestamp(when_local)
    if t.tzinfo is None:
        t = t.tz_localize(LOCAL_TZ, ambiguous=True, nonexistent="shift_forward")
    sec = int(t.tz_convert("UTC").timestamp())
    i = int(np.searchsorted(data.h[hi]["ts"], sec))
    return min(max(i, 0), len(data.h[hi]["ts"]) - 1)


def window_ok(data, hi, end):
    """True when the full input window before `end` has valid aggregate readings."""
    if end < data.window - 1:
        return False
    w = data.h[hi]["aggregate"][end - data.window + 1:end + 1]
    return bool(np.isfinite(w).all())


def default_sample(data, hi):
    """A test-split minute where the most appliances are ON (ground truth),
    used as the dashboard's starting example."""
    H = data.h[hi]
    idx = data.windows["unseen" if data.homes[hi] in data.prep["unseen_homes"] else "test"]
    ends = idx[idx[:, 0] == hi, 1]
    if len(ends) == 0:
        return data.window
    n_on = (H["on"][ends] & H["valid"][ends]).sum(axis=1)
    return int(ends[np.argmax(n_on)])


def load_torch_model(name, data):
    """Load a saved model. Returns (model, message)."""
    path = MODELS_DIR / f"{name}.pth"
    if not path.exists():
        return None, f"{MODEL_LABELS[name]} is not trained yet: models/{name}.pth is missing."
    try:
        import torch
        from src.models import build_model
    except ImportError:
        return None, "PyTorch is not installed, so the saved models cannot be run."
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(name, len(data.appliances), data.window)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    msg = "" if ck.get("finished") else "Training of this model did not finish; " \
                                        "showing its best checkpoint so far."
    return model, msg


def predict_ends(model, data, hi, ends, batch=2048):
    """Run a model on the windows ending at `ends` of home `hi`.
    Input: aggregate power only. Returns prob (n, A) and power in W (n, A)."""
    import torch
    idx = np.stack([np.full(len(ends), hi), ends], axis=1)
    probs, powers = [], []
    with torch.no_grad():
        for s in range(0, len(idx), batch):
            X = torch.from_numpy(data.scale_inputs(data.raw_inputs(idx[s:s + batch])))
            logits, p = model(X)
            probs.append(torch.sigmoid(logits).numpy())
            powers.append(p.numpy() * data.power_scale * 1000.0)
    return np.concatenate(probs), np.concatenate(powers)


def prob_cutoffs(results, name, appliances):
    thr = results["prob_thr"].get(name, {})
    return {a: float(thr.get(a, 0.5)) for a in appliances}


def range_frame(data, hi, a, b):
    """Ground-truth table for minutes a..b-1 of a home (kW -> W)."""
    H = data.h[hi]
    df = pd.DataFrame({"time": local_times(data, hi, a, b),
                       "aggregate_W": H["aggregate"][a:b] * 1000})
    for i, ap in enumerate(data.appliances):
        df[f"{ap}_W"] = np.where(H["valid"][a:b, i], H["power"][a:b, i] * 1000, np.nan)
        df[f"{ap}_on"] = np.where(H["valid"][a:b, i], H["on"][a:b, i].astype(float), np.nan)
    df["split"] = [SPLIT_NAMES[s] for s in H["split"][a:b]]
    return df


def valid_ends_in(data, hi, a, b):
    ends = np.arange(max(a, data.window - 1), b)
    return np.array([e for e in ends if window_ok(data, hi, e)], dtype=np.int64)


def appliance_label(a):
    return APPLIANCE_NAMES.get(a, a)
