"""
Train the MLP and the 1D-CNN (multi-output: ON/OFF + power for air1, drye1, car1, refrigerator1; loss masked where a home has no such circuit).

Input  : prep_austin.npz, prep_meta.json  (from data_prep.py)
Output : model_<name>.pt, preds_<name>.npz, train_log.txt
Run    : python train_models.py            (SEED = 42, CPU)
"""
import time
import numpy as np
import torch
import torch.nn as nn
from nilm_common import MODELS, gather, load_prep

SEED = 42
BATCH = 1024
LR = 1e-3
EPOCHS = {"mlp": 15, "cnn1d": 8}
PATIENCE = 3
TRAIN_STRIDE = 10         # every 10th train window (consecutive windows overlap ~98%)
VAL_STRIDE = 4

torch.set_num_threads(1)
torch.manual_seed(SEED)
np.random.seed(SEED)
log_file = open("train_log.txt", "w")


def log(msg):
    print(msg, flush=True)
    log_file.write(msg + "\n")
    log_file.flush()


z, meta = load_prep()
APPS, THR = meta["appliances"], np.array([meta["thresholds_kw"][a] for a in meta["appliances"]], dtype=np.float32)
SC = meta["scaling"]
P_SCALE = np.array([SC[f"{a}_scale"] for a in APPS], dtype=np.float32)
seen, unseen = meta["seen_homes"], meta["unseen_homes"]
log(f"seed={SEED} seen={seen} unseen={unseen} thresholds_kW={dict(zip(APPS, THR.round(3)))}")


def norm_x(x):
    return ((x - SC["total_mean"]) / SC["total_std"]).astype(np.float32)


def prep(homes, splits, stride):
    x, y, h, t = gather(z, homes, splits, stride)
    mask = (~np.isnan(y)).astype(np.float32)          # 0 where the home has no such circuit
    y0 = np.nan_to_num(y)
    return norm_x(x), (y0 > THR).astype(np.float32), (y0 / P_SCALE).astype(np.float32), y, h, t, mask


sets = {
    "train": prep(seen, ["train"], TRAIN_STRIDE),
    "val": prep(seen, ["val"], VAL_STRIDE),
}
for k, v in sets.items():
    log(f"{k:7s} windows={len(v[0]):8d}  ON counts per appliance={(v[1] * v[6]).sum(0).astype(int).tolist()}  valid targets={v[6].sum(0).astype(int).tolist()}")

on_tr, m_tr = sets["train"][1], sets["train"][6]
rate = (on_tr * m_tr).sum(0) / m_tr.sum(0)
pos_w = torch.tensor(np.clip(np.sqrt((1 - rate) / rate), 1, 10), dtype=torch.float32)
log(f"BCE pos_weight={pos_w.numpy().round(2).tolist()}")
bce = nn.BCEWithLogitsLoss(pos_weight=pos_w, reduction="none")
huber = nn.SmoothL1Loss(reduction="none")


def run_epoch(model, data, opt=None):
    x, on, pw, mk = (torch.from_numpy(data[i]) for i in (0, 1, 2, 6))
    order = torch.randperm(len(x)) if opt else torch.arange(len(x))
    model.train(bool(opt))
    tot, n = 0.0, 0
    with torch.set_grad_enabled(bool(opt)):
        for i in range(0, len(x), BATCH):
            b = order[i:i + BATCH]
            lg, rg = model(x[b])
            loss = ((bce(lg, on[b]) + huber(rg, pw[b])) * mk[b]).sum() / mk[b].sum()
            if opt:
                opt.zero_grad()
                loss.backward()
                opt.step()
            tot += loss.item() * len(b)
            n += len(b)
    return tot / n


@torch.no_grad()
def predict(model, x):
    model.eval()
    xs = torch.from_numpy(x)
    lg, rg = [], []
    for i in range(0, len(xs), 8192):
        a, b = model(xs[i:i + 8192])
        lg.append(torch.sigmoid(a))
        rg.append(b)
    return torch.cat(lg).numpy(), np.clip(torch.cat(rg).numpy() * P_SCALE, 0, None)   # prob, power in kW


for name, cls in MODELS.items():
    torch.manual_seed(SEED)
    model = cls()
    log(f"\n=== {name}: {sum(p.numel() for p in model.parameters())} parameters")
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    best, best_state, bad = 1e9, None, 0
    for ep in range(1, EPOCHS[name] + 1):
        t0 = time.time()
        tr, va = run_epoch(model, sets["train"], opt), run_epoch(model, sets["val"])
        log(f"{name} epoch {ep:2d}  train_loss={tr:.4f}  val_loss={va:.4f}  ({time.time() - t0:.0f}s)")
        if va < best - 1e-4:
            best, bad = va, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                log(f"{name}: early stop at epoch {ep}")
                break
    model.load_state_dict(best_state)
    torch.save(best_state, f"model_{name}.pt")
    out = {}
    p, w = predict(model, sets["val"][0])
    out.update({"val_prob": p, "val_pow": w, "val_true_pow": sets["val"][3], "val_home": sets["val"][4], "val_idx": sets["val"][5]})
    for k, homes, splits in (("test", seen, ["test"]), ("unseen", unseen, ["train", "val", "test"])):
        parts = []                                    # windows are built one home at a time to save memory
        for h in homes:
            x, y, hh, ii = gather(z, [h], splits, 1)
            parts.append((*predict(model, norm_x(x)), y, hh, ii))
        p, w, y, hh, ii = (np.concatenate(a) for a in zip(*parts))
        out.update({f"{k}_prob": p, f"{k}_pow": w, f"{k}_true_pow": y, f"{k}_home": hh, f"{k}_idx": ii})
    np.savez_compressed(f"preds_{name}.npz", **out)
    log(f"{name}: best val_loss={best:.4f}, predictions saved")
