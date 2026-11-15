"""
Phase 4: train the six ANN models with identical data and settings.

Run all six:              python -m src.train
Run some models:          python -m src.train --models mlp cnn
Quick smoke test:         python -m src.train --models mlp --epochs 1 --max-train-windows 5000
Continue after a stop:    python -m src.train --resume
Retrain a finished model: python -m src.train --models gru --force

For every model this saves
  models/<name>.pth                        best weights (lowest validation loss)
  models/checkpoints/<name>_last.pth       last epoch, used by --resume
  results/training_logs/<name>_log.csv     loss per epoch
  results/training_logs/<name>_config.json training settings
Finished models are skipped unless --force is given.
"""
import argparse
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.dataset import NILMData
from src.models import build_model, count_parameters
from src.utils import (BATCH_SIZE, LEARNING_RATE, MAX_EPOCHS, PATIENCE, MAX_TRAIN_WINDOWS,
                       MAX_VAL_WINDOWS, MODEL_NAMES, SEED, MODELS_DIR, CHECKPOINT_DIR,
                       LOG_DIR, ensure_dirs, save_json, set_seed)


def masked_losses(logits, power, y_on, y_pow, mask, pos_weight):
    """Classification + regression loss, counting only valid targets."""
    bce = nn.functional.binary_cross_entropy_with_logits(
        logits, y_on, pos_weight=pos_weight, reduction="none")
    reg = nn.functional.smooth_l1_loss(power, y_pow, reduction="none")
    n = mask.sum().clamp(min=1.0)
    cls_loss = (bce * mask).sum() / n
    reg_loss = (reg * mask).sum() / n
    return cls_loss, reg_loss


def to_t(*arrays):
    return [torch.from_numpy(a) for a in arrays]


def run_epoch(model, data, idx, pos_weight, optimizer=None, seed=0):
    train = optimizer is not None
    model.train(train)
    tot = np.zeros(3)
    n = 0
    with torch.set_grad_enabled(train):
        for X, y_on, y_pow, mask, _ in data.iterate(idx, BATCH_SIZE, shuffle=train, seed=seed):
            X, y_on, y_pow, mask = to_t(X, y_on, y_pow, mask)
            logits, power = model(X)
            cls_loss, reg_loss = masked_losses(logits, power, y_on, y_pow, mask, pos_weight)
            loss = cls_loss + reg_loss
            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            b = len(X)
            tot += b * np.array([loss.item(), cls_loss.item(), reg_loss.item()])
            n += b
    return tot / max(n, 1)


def train_one(name, data, train_idx, val_idx, pos_weight, args):
    final_path = MODELS_DIR / f"{name}.pth"
    last_path = CHECKPOINT_DIR / f"{name}_last.pth"
    if final_path.exists() and not args.force:
        ck = torch.load(final_path, map_location="cpu", weights_only=False)
        if ck.get("finished"):
            print(f"[{name}] already trained (best epoch {ck['best_epoch']}). "
                  "Skipping. Use --force to retrain.")
            return

    set_seed(SEED)
    model = build_model(name, len(data.appliances), data.window)
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    history, start, best, best_epoch, bad = [], 1, float("inf"), 0, 0
    if args.resume and last_path.exists() and not args.force:
        ck = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        history, start = ck["history"], ck["epoch"] + 1
        best, best_epoch, bad = ck["best_val"], ck["best_epoch"], ck["bad_epochs"]
        print(f"[{name}] resuming from epoch {ck['epoch']}")

    config = {
        "model": name, "parameters": count_parameters(model), "seed": SEED,
        "optimizer": "Adam", "learning_rate": LEARNING_RATE, "batch_size": BATCH_SIZE,
        "max_epochs": args.epochs, "patience": PATIENCE,
        "train_windows": int(len(train_idx)), "val_windows": int(len(val_idx)),
        "window": data.window, "appliances": data.appliances,
        "pos_weight": pos_weight.tolist(),
        "loss": "BCEWithLogits(pos_weight, masked) + SmoothL1(scaled power, masked)",
    }
    save_json(config, LOG_DIR / f"{name}_config.json")
    print(f"[{name}] {config['parameters']:,} parameters, "
          f"{len(train_idx):,} train / {len(val_idx):,} val windows")

    for epoch in range(start, args.epochs + 1):
        t0 = time.time()
        tr = run_epoch(model, data, train_idx, pos_weight, opt, seed=SEED + epoch)
        va = run_epoch(model, data, val_idx, pos_weight)
        dt = time.time() - t0
        history.append({"epoch": epoch, "train_loss": tr[0], "train_cls": tr[1],
                        "train_reg": tr[2], "val_loss": va[0], "val_cls": va[1],
                        "val_reg": va[2], "seconds": round(dt, 1)})
        improved = va[0] < best - 1e-5
        if improved:
            best, best_epoch, bad = va[0], epoch, 0
            torch.save({"state_dict": model.state_dict(), "model_name": name,
                        "config": config, "best_epoch": epoch, "val_loss": best,
                        "finished": False}, final_path)
        else:
            bad += 1
        print(f"[{name}] epoch {epoch:2d}  train {tr[0]:.4f}  val {va[0]:.4f}  "
              f"{'*best*' if improved else f'no gain {bad}/{PATIENCE}'}  {dt:.0f}s")
        pd.DataFrame(history).to_csv(LOG_DIR / f"{name}_log.csv", index=False)
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                    "epoch": epoch, "history": history, "best_val": best,
                    "best_epoch": best_epoch, "bad_epochs": bad}, last_path)
        if bad >= PATIENCE:
            print(f"[{name}] early stopping at epoch {epoch}")
            break

    ck = torch.load(final_path, map_location="cpu", weights_only=False)
    ck["finished"] = True
    ck["epochs_run"] = history[-1]["epoch"]
    torch.save(ck, final_path)
    print(f"[{name}] done. best epoch {best_epoch}, val loss {best:.4f} -> {final_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=MODEL_NAMES, choices=MODEL_NAMES)
    p.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    p.add_argument("--max-train-windows", type=int, default=MAX_TRAIN_WINDOWS)
    p.add_argument("--max-val-windows", type=int, default=MAX_VAL_WINDOWS)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--threads", type=int, default=0, help="CPU threads (0 = PyTorch default)")
    args = p.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    ensure_dirs()
    set_seed(SEED)
    data = NILMData()
    # Same subsample for all models (fixed seed): identical training data.
    train_idx = data.windows_for("train", args.max_train_windows, seed=SEED)
    val_idx = data.windows_for("val", args.max_val_windows, seed=SEED + 1)

    # Positive class weights from the TRAINING windows only.
    _, y_on, _, mask, _ = data.batch(train_idx)
    n_on = (y_on * mask).sum(0)
    n_off = ((1 - y_on) * mask).sum(0)
    pos_weight = torch.tensor(np.clip(n_off / np.maximum(n_on, 1), 0.1, 100.0),
                              dtype=torch.float32)
    print("pos_weight:", dict(zip(data.appliances, np.round(pos_weight.numpy(), 2))))

    for name in args.models:
        train_one(name, data, train_idx, val_idx, pos_weight, args)


if __name__ == "__main__":
    main()
